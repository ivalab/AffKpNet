# An Affordance Keypoint Detection Network for Robot Manipulation
Ruinian Xu, Fu-Jen CHu, Chao Tang, Weiyu Liu and Patricio A. Vela
## Introduction

General purpose robotic manipulation requires robots to understand how objects may serve different purposes. Affordancces describe potential physical interactions between objects parts, and associate objects parts to actions for planning manipulation sequences. General manipulation ransoning and execution requires identifying: ***what*** the affordances of seen object parts are, ***where*** on the object that affordances act or should be acted on by the manipulator, and ***how*** the manipulator should execute the affordance's action. Recent studies of affordance cast affordance detection of object parts in the scene as an image segmentation problem. Object part pixels sharing the same functionality are predicted to have the same category and grouped into one instance. The segmentation-only methods obtain ***what*** information from the affordance mask and ***where*** information by post-processing the mask to compute its center location, as shown in the Figure 1. However ***how*** information is missing in these methods. Taking the knife as example, from the segmentation mask, it is hard to predict which side is blade. In order to provide the complete information for robotic manipulation, we propose to use a set of keypoints to represent each affordance. As shown in the Figure 2, for those affordances require certain way to operate like the cut and pound, keypoints 1 and 2 determine the operational direction. We also propose an Affordance Keypoint Detection Network (AffKp) to provide affordance segmentation and keypoint prediction for object parts. The network architecture is shown in the Figure 3.

![image](img/fig_seg_only.png)
Fig. 1 - Pipeline of segmentation-only methods.
![image](img/fig_kp_rep.png)
Fig. 2 - Affordance keypoint representation for six affordances.
![image](img/fig_network.png)
Fig. 3 - Affordance keypoint detection network architecture.

## UMD+GT segmentation result


![image](img/fig_seg_result.png)
Fig. 4 - Result of affordance segmentation over UMD+GT dataset.

## UMD+GT keypoint result


![image](img/fig_kp_result.png)
Fig. 5 - Result of affordance keypoint detection over UMD+GT dataset.

## Usage

1. Install pytorch 

  - The code is tested on python3.6 and official [Pytorch@commitfd25a2a](https://github.com/pytorch/pytorch/tree/fd25a2a86c6afa93c7062781d013ad5f41e0504b#from-source), please install PyTorch from source.
  - The code is modified from [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding). 
  
2. Clone the repository:

   ```shell
   git clone https://github.com/junfu1115/DANet.git 
   cd DANet 
   python setup.py install
   ```
   
3. Dataset

  - Download the [Cityscapes](https://www.cityscapes-dataset.com/) dataset and convert the dataset to [19 categories](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py). 
  - Please put dataset in folder `./datasets`

4 . Evaluation

  - Download trained model [DANet101](https://drive.google.com/open?id=1XmpFEF-tbPH0Rmv4eKRxYJngr3pTbj6p) and put it in folder `./danet/cityscapes/model`
  - Evaluation code is in folder `./danet/cityscapes`
  - `cd danet`

  - For single scale testing, please run:
  
   ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset cityscapes --model danet --resume-dir cityscapes/model --base-size 2048 --crop-size 768 --workers 1 --backbone resnet101 --multi-grid --multi-dilation 4 8 16 --eval
   ```
   
  - For multi-scale testing, please run:
  
   ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset cityscapes --model danet --resume-dir cityscapes/model --base-size 2048 --crop-size 1024 --workers 1 --backbone resnet101 --multi-grid --multi-dilation 4 8 16 --eval --multi-scales
   ```  
   
  - If you want to visualize the result of DAN-101, you can run:
 
   ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset cityscapes --model danet --resume-dir cityscapes/model --base-size 2048 --crop-size 768 --workers 1 --backbone resnet101 --multi-grid --multi-dilation 4 8 16
   ```
   
5. Evaluation Result:

   The expected scores will show as follows:
   
   (single scale testing denotes as 'ss' and multiple scale testing denotes as 'ms')
   
   DANet101 on cityscapes val set (mIoU/pAcc): **79.93/95.97** (ss) and **81.49/96.41** (ms)


6. Training:

  - Training code is in folder `./danet/cityscapes`
  - `cd danet`
  
   You can reproduce our result by run:

  ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset cityscapes --model  danet --backbone resnet101 --checkname danet101  --base-size 1024 --crop-size 768 --epochs 240 --batch-size 8 --lr 0.003 --workers 2 --multi-grid --multi-dilation 4 8 16
   ```
 
   Note that: We adopt multiple losses in end of the network for better training. 

## Keypoint grouping algorithm

## Affordance keypoint used in physical experiment

## Install

### environment
Use `anaconda` for `python3`, `pytorch 1.0` and `cuda8.0`.
Currently the `maskrcnn_benchmark` environment is shared with this project. `source activate maskrcnn_benchmark
`

### library issue
1. No need to install the lib.   
2. To use the library, clone `https://github.com/huanghoujing/PyTorch-Encoding` for libraries. Otherwise, you will have problem when compiling the libs. 
3. Replace `encoding/lib` with the one in `huanghoujing`
4. In `encoding/functions/syncbn.py`, replace with 'xsum, xsqusum = lib.cpu.sumsquare_forward(input)##gpu->cpu
' for cpu implementation

### single GPU
In `danet/train.py`, replace `norm_layer=BatchNorm2d` with `norm_layer=torch.nn.BatchNorm2d`

### train
In `DANet/danet`, run
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cityscapes --model  danet --backbone resnet101 --checkname danet101  --base-size 1024 --crop-size 768 --epochs 240 --batch-size 1 --lr 0.003 --workers 2 --multi-grid --multi-dilation 4 8 16
```

### test
In `DANet/danet`, run
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset cityscapes --model danet --resume-dir cityscapes/model --base-size 2048 --crop-size 768 --workers 1 --backbone resnet101 --multi-grid --multi-dilation 4 8 16 --eval

```


## Citation
If DANet is useful for your research, please consider citing:
```
@article{fu2018dual,
  title={Dual Attention Network for Scene Segmentation},
  author={Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang,and Hanqing Lu},
  booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```
## Acknowledgement
Thanks [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding), especially the Synchronized BN!
