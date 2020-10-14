###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################
import sys, os
sys.path.append('/home/fujenchu/projects/affordanceContext/DANet/')

import os
import copy
import numpy as np
from tqdm import tqdm

import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses,BatchNorm2d
from encoding.nn import SegmentationMultiLosses, SegMultiKpLosses
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_segmentation_dataset
from encoding.models import get_segmentation_model


from option import Options
from utils.weblogger import Dashboard

torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

class Trainer():
    def __init__(self, args):
        self.args = args
        args.log_name = str(args.checkname)
        self.logger = utils.create_logger(args.log_root, args.log_name)
        # data transforms
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.358388, .348858, .294015], [.153398, .137741, .230031])])
        # dataset
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                       'crop_size': args.crop_size, 'logger': self.logger,
                       'scale': args.scale}
        trainset = get_segmentation_dataset(args.dataset, split='train', mode='train',
                                            **data_kwargs)
        testset = get_segmentation_dataset(args.dataset, split='val', mode='val',
                                           **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} \
            if args.cuda else {}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size,
                                           drop_last=True, shuffle=True, **kwargs)
        self.valloader = data.DataLoader(testset, batch_size=args.batch_size,
                                         drop_last=False, shuffle=False, **kwargs)

        self.nclass = trainset.num_class
        # model
        model = get_segmentation_model(args.model, dataset=args.dataset,
                                       backbone=args.backbone,
                                       aux=args.aux, se_loss=args.se_loss,
                                       norm_layer=torch.nn.BatchNorm2d,
                                       base_size=args.base_size, crop_size=args.crop_size,
                                       multi_grid=args.multi_grid,
                                       multi_dilation=args.multi_dilation)
        #print(model)
        # optimizer using different LR
        params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr}, ]
        if hasattr(model, 'head'):
            params_list.append({'params': model.head.parameters(), 'lr': args.lr * 10})
        if hasattr(model, 'auxlayer'):
            params_list.append({'params': model.auxlayer.parameters(), 'lr': args.lr * 10})
        if hasattr(model, 'auxlayer'):
            params_list.append({'params': model.auxlayer.parameters(), 'lr': args.lr * 10})
        params_list.append({'params': model.deconv_layers.parameters(), 'lr': args.lr})
        for head in model.heads:
            if head == '1' or head == '2' or head == '3' or head == '4' or head == '5':
                params_list.append({'params': model.__getattr__(head).parameters(), 'lr': args.lr / 100})
            elif head == '1_reg' or head == '2_reg' or head == '3_reg' or head == '4_reg' or head == '5_reg':
                params_list.append({'params': model.__getattr__(head).parameters(), 'lr': args.lr / 10})
            else:
                params_list.append({'params': model.__getattr__(head).parameters(), 'lr': args.lr})

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        self.criterion = SegMultiKpLosses(args.pull_weight, args.push_weight, args.regr_weight, args.seg_weight)


        self.model, self.optimizer = model, optimizer
        # using cuda
        if args.cuda:
            self.model = DataParallelModel(self.model).cuda()
            self.criterion = DataParallelCriterion(self.criterion).cuda()

        # finetune from a trained model
        if args.ft:
            args.start_epoch = 0
            checkpoint = torch.load(args.ft_resume)
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.ft_resume, checkpoint['epoch']))
        # resuming checkpoint
        if args.resume:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            self.logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))


        ## logport (sudo python -m visdom.server --port 5000)
        if args.logport:
            args.logport = Dashboard(args.logport, 'UMD2UMD')
        # lr scheduler
        self.scheduler = utils.LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.trainloader), logger=self.logger,
                                            lr_step=args.lr_step)
        self.best_pred = 0.0

    def training(self, epoch):
        train_loss = 0.0
        train_loss_seg = 0.0
        train_loss_focal = 0.0
        train_loss_reg = 0.0
        train_loss_push = 0.0
        train_loss_pull = 0.0
        self.model.train()
        self.criterion.train()
        tbar = tqdm(self.trainloader)

        for i, (image, target) in enumerate(tbar):
            image = image.cuda()
            for key, value in target.items():
                if not target[key].is_cuda:
                    target[key] = value.cuda()

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            if torch_ver == "0.3":
                image = Variable(image)
                target = Variable(target)
            outputs = self.model(image)

            if args.logport and i % 500 == 0:
                args.logport.image(image, 'umd train img')
                args.logport.image(outputs[0], 'umd train pred', denorm=False)

            loss, loss_stat = self.criterion(outputs, target)
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_loss_seg += loss_stat['seg_loss'].item()
            train_loss_focal += loss_stat['focal_loss'].item()
            train_loss_reg += loss_stat['reg_loss'].item()
            train_loss_push += loss_stat['push_loss'].item()
            train_loss_pull += loss_stat['pull_loss'].item()
            tbar.set_description(
                'Train loss: %.3f, loss seg: %.3f, loss focal: %.3f, loss reg: %.3f, loss push: %.3f, loss pull: %.3f' % \
                (train_loss / (i + 1), train_loss_seg / (i + 1), train_loss_focal / (i + 1), train_loss_reg / (i + 1),
                 train_loss_push / (i + 1), train_loss_pull / (i + 1)))
        self.logger.info('Train loss: %.3f, loss seg: %.3f, loss focal: %.3f, loss reg: %.3f, loss push: %.3f, loss pull: %.3f' % \
                         (train_loss / (i + 1), train_loss_seg / (i + 1), train_loss_focal / (i + 1),
                          train_loss_reg / (i + 1), train_loss_push / (i + 1), train_loss_pull / (i + 1)))

        if self.args.no_val:
            # save checkpoint every 10 epoch
            filename = "checkpoint_%s.pth.tar" % (epoch + 1)
            is_best = False
            if epoch > 99:
                if not epoch % 5:
                    utils.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.module.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'best_pred': self.best_pred,
                    }, self.args, is_best, filename)

    def validation(self, epoch):
        # Fast test during the training
        def eval_batch(model, image, target, ite, criterion):
            outputs = model(image)

            loss, loss_stat = criterion(outputs, target)

            pred = outputs['sasc']

            if args.logport and ite % 300 == 0:
                # print('logging umd val...')
                args.logport.image(image, 'umd val img')
                args.logport.image(pred, 'umd val pred', denorm=False)

            # target = target.cuda()
            correct, labeled = utils.batch_pix_accuracy(pred.data, target['seg'])
            inter, union = utils.batch_intersection_union(pred.data, target['seg'], self.nclass)
            return correct, labeled, inter, union, loss_stat

        is_best = False
        self.model.eval()
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        val_loss_focal, val_loss_reg, val_loss_push, val_loss_pull = 0., 0., 0., 0.
        tbar = tqdm(self.valloader, desc='\r')

        criterion = SegMultiKpLosses(args.pull_weight, args.push_weight, args.regr_weight, args.seg_weight)
        criterion.eval()

        for i, (image, target) in enumerate(tbar):
            if torch_ver == "0.3":
                image = Variable(image, volatile=True)
                correct, labeled, inter, union, loss_stat = eval_batch(self.model, image, target, i, criterion)
            else:
                with torch.no_grad():
                    image = image.cuda()
                    for key, value in target.items():
                        if not target[key].is_cuda:
                            target[key] = value.cuda()
                    correct, labeled, inter, union, loss_stat = eval_batch(self.model, image, target, i, criterion)

            val_loss_focal += loss_stat['focal_loss'].item()
            val_loss_reg += loss_stat['reg_loss'].item()
            val_loss_pull += loss_stat['pull_loss'].item()
            val_loss_push += loss_stat['push_loss'].item()

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            tbar.set_description(
                'pixAcc: %.3f, mIoU: %.3f, loss focal: %.3f, loss reg: %.3f, loss push: %.3f, loss pull: %.3f' % \
                (pixAcc, mIoU, val_loss_focal / (i + 1), val_loss_reg / (i + 1), val_loss_push / (i + 1),
                 val_loss_pull / (i + 1)))
        self.logger.info(
            'pixAcc: %.3f, mIoU: %.3f, loss focal: %.3f, loss reg: %.3f, loss push: %.3f, loss pull: %.3f' % \
            (pixAcc, mIoU, val_loss_focal / (i + 1), val_loss_reg / (i + 1), val_loss_push / (i + 1),
             val_loss_pull / (i + 1)))

        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        utils.save_checkpoint_all({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
            'focal_loss': val_loss_focal / (i + 1),
            'reg_loss': val_loss_reg / (i + 1),
        }, self.args, is_best)

if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    trainer.logger.info(['Starting Epoch:', str(args.start_epoch)])
    trainer.logger.info(['Total Epoches:', str(args.epochs)])

    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        if not args.no_val:
            trainer.validation(epoch)
