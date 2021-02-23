## Introduction
This markdown file includes the investigation result of performing the proposed network for the occlusion scenarios. Since the current keypoint grouping algorithm only supports outputing one keypoint group for each category, we only include objects whose affordance categories are different. 

## Visualization

### Scene 1

![image](../img/occlusion/scene_1/image.png)

![image](../img/occlusion/scene_1/mask.png)

<img src="https://github.com/ivalab/AffKpNet/blob/master/img/occlusion/scene_1/0_kp.png" width="640">

<img src="https://github.com/ivalab/AffKpNet/blob/master/img/occlusion/scene_1/2_kp.png" width="640">

<img src="https://github.com/ivalab/AffKpNet/blob/master/img/occlusion/scene_1/3_kp.png" width="640">

<img src="https://github.com/ivalab/AffKpNet/blob/master/img/occlusion/scene_1/4_kp.png" width="640">

### Scene 2

![image](../img/occlusion/scene_2/image.png)

![image](../img/occlusion/scene_2/mask.png)

<img src="https://github.com/ivalab/AffKpNet/blob/master/img/occlusion/scene_2/0_kp.png" width="640">

<img src="https://github.com/ivalab/AffKpNet/blob/master/img/occlusion/scene_2/2_kp.png" width="640">

<img src="https://github.com/ivalab/AffKpNet/blob/master/img/occlusion/scene_2/3_kp.png" width="640">

<img src="https://github.com/ivalab/AffKpNet/blob/master/img/occlusion/scene_2/5_kp.png" width="640">

### Scene 3

![image](../img/occlusion/scene_3/image.png)

![image](../img/occlusion/scene_3/mask.png)

<img src="https://github.com/ivalab/AffKpNet/blob/master/img/occlusion/scene_3/0_kp.png" width="640">

<img src="https://github.com/ivalab/AffKpNet/blob/master/img/occlusion/scene_3/1_kp.png" width="640">

<img src="https://github.com/ivalab/AffKpNet/blob/master/img/occlusion/scene_3/3_kp.png" width="640">


### Scene 4

![image](../img/occlusion/scene_4/image.png)

![image](../img/occlusion/scene_4/mask.png)

<img src="https://github.com/ivalab/AffKpNet/blob/master/img/occlusion/scene_4/0_kp.png" width="640">


## Observation
As shown in the visualization,  the proposed method is still capable to provide reasonable affordance keypoint prediction. The unoccluded part is still predicted with the correct keypoint but the keypoints of the occluded part are off. Interesting, the wrap-grasp affordance can't be predicted for all scenarios. There is also the case that the network predicted the unexpected affordance. 

## Discussion
The results show that the proposed network is still able to provide reasonable prediction for the occlusion scenario while there is no such kind of image in the training datset. 
Theoretically, occlusion won't cause too much affect for the performance as long as the object part still remains its geometry. The heatmap of each keypoint is still capable to 
provide the prediction based on the visual features. The problem mainly comes from the grouping stage. Since there is no occlusion scenario in the training dataset, the embedding heatmap will associate some pixels to the object part that not belong to itself. To address this problem, we can work on the expanding the training dataset or incorporating the post-process algorithm. 
