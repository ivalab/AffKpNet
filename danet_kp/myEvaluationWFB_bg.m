% my script for affordance evaluation using wFb metric
% 
% Original evaluation code from UMD is redundant. Their proposed method requires to train
% seperate models for each of the affordances. Hence, during evaluation the
% outer loop is all affordance models, and the same images and GTs are loaded
% multiple times (8x) Furthermore, the code evaluate FP on all images every
% time (3x). The code also uses cropped image (?).
% 
% This code is 24x faster
%

clear; close all;
%% Prepare predicted affordance map
% run demo_img_saveToDisk.py with according model in
% /AFFORDANCENETDAROOT/tools

%% I/O
MODEL = 'danet_affnet_instance_noatt_7att_bg165'; % Please check the FILTER below!!!
MODEL = 'adaptNet2';
% PRED_AFF_ROOT = ['/media/fujenchu/home3/data/affordanceSeg/SRFAff_v1/data/Affordance_Part_Data/' MODEL];
PRED_AFF_ROOT = '/home/fujenchu/projects/affordanceContext/DANet/datasets/UMD_affordance/pred_seg_kp_umdself';
TEST_DATA_ROOT = '/home/fujenchu/projects/affordanceContext/DANet/datasets/UMD_affordance/test_compound_only';
SAVE_FILE = [MODEL '_WFb_scores.txt'];


WFbS=[];

objectFolderList = dir(TEST_DATA_ROOT);
for idx_objFolder = 3:length(objectFolderList)
    objectFolderName = objectFolderList(idx_objFolder).name;
    subFolderList = dir(fullfile(TEST_DATA_ROOT,objectFolderName));
    fprintf(['processing ' objectFolderName '..\n']);
    
    WFbS_per_obj=[];
    parfor idx_subFolder = 3:length(subFolderList)
        subFolderName = subFolderList(idx_subFolder).name;
        GTList = dir(fullfile(TEST_DATA_ROOT, objectFolderName, subFolderName,'*_label.mat'));
        fprintf(['processing ' objectFolderName ' ' subFolderName '..\n']);
        
        for idx_gt = 1:length(GTList)
            GTfileName = GTList(idx_gt).name;
            GTfile = load(fullfile(TEST_DATA_ROOT, objectFolderName, subFolderName, GTfileName));
            GT = GTfile.gt_label;
            
            % assume pred_model9 has the same data structure
            predFileName = strrep(GTfileName, '_label', '_rgbd_pred');%_rgbd_crop_pred %_labelid_pred %_rgb_pred
            predFile = load(fullfile(PRED_AFF_ROOT, objectFolderName, subFolderName, predFileName));
            pred = predFile.pred_label;
            
            
%             [argvalue, argmax] = max(pred,[],3);
%             argmax= argmax-1;
            
            % test across all affordance
            WFbS_per_img = nan(6,1);
            for affID = 1:6
                GT_affMap = GT==affID;

                 %pred_affMap = pred(:,:,affID); 
%                  pred_affMap = argmax==affID;
                pred_affMap = pred==affID;
                if sum(GT_affMap(:)>0),WFbS_per_img(affID)=WFb(double(pred_affMap),GT_affMap); end

            end
            WFbS = [WFbS WFbS_per_img];
            WFbS_per_obj = [WFbS_per_obj WFbS_per_img];
        
        end 
    end
    WFbS_per_obj_mean = nanmean(WFbS_per_obj,2)
    dlmwrite(['per_' objectFolderName SAVE_FILE], WFbS_per_obj_mean);
end
WFbS_mean = nanmean(WFbS,2)
dlmwrite(SAVE_FILE, WFbS_mean);
