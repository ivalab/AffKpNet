## Introduction
This tutorial mainly introduces how we annotate the ground truth for UMD+GT dataset. It can be mainly divided into two parts: segmenatation and keypoint. 

## Segmentation
The annotation for segmentation involves annotating objects come from GT dataset. The entire process involves genreating coarse result and then fine-tune to 
get the final ground truth.

### UND
Since the original UMD dataset already provide the ground truth of segmentation mask, we don't need to annotate for thoes objects. 

###
For the added objects, we employ the open-source tool [PixelAnnotationTool](https://github.com/abreheret/PixelAnnotationTool). It provides two types of pixel annotation. One is to simply 
annotate pixel one by one. The size of the circle can be changed. This way is pretty laborous. The other way is to generate Watershed mask. This way can automatically
genreate mask for large areas but there are some imperfect pixels. Therefore our strategy is to use Watershed mask to generate the coarse mask and then use the first way 
to fine-tune the mask.

## Keypoint 
The annotation for keypoint involves annotating objects come from UMD and GT dataset. The entire process still involves genreating coarse result and then fine-tune to 
get the final ground truth.

### UMD 
The problem of UMD dataset is that there is no any markers in the scene can be easily detected to provide the reference frame. 
In order to make the annotation process easy, we develop a [Matlab tool](https://github.com/fujenchu/UMD_affordanceKP_toolbox) by ourselves. The detailed usage 
can be found in its repository.

### GT
We put four aruco tags on the rotating table during collecting the images. The aruco tag can be detected as the reference frame. The key idea is that the relative
position between the object and aruco tags won't change. After determining keypoints for one image, we can use detected aruco tags to automatically generate keypoints
for new images. The [script]() is provided. Since the frame detected by the aruco tag isn't perfect, these keypoints still need further modification.

### Fine-tune
This step is laborous since we need to check the keypoint ground truth for each image. In order to make the life easier, based on the assumption that the general geometry
of the keypoint won't change a lot with the error of the aruco tag, we develop two ways. One is to manually annotate the single affordance keypoint if it is imperfect. 
The other way is to shift the entire five keypoints up, down, left or right. The [script](https://github.com/ivalab/AffKpNet/blob/master/utils/annotation_kp_finetune.py) is provided here.
