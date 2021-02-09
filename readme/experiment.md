## Introduction
The physical experiments involve the perception and manipulation modules. The perception module mainly works for predicting the affordance keypoints and converting the computed keypoints to the operational location and direction. The 2D information will be converted into 3D space with the depth information. The manipulation module is mainly responsible for planning the trajectory and performing the execution to achieve the task. 

## Perception
We uploaded all experiment scripts [here](https://github.com/ivalab/AffKpNet/tree/master/exp). There should be 6 scripts in total. 3 (grasp, contain, w-grasp) for the single object manipulation experiment. 1 for the object arrangment task. 2 (cut and pound) for the cutting the string and pounding the nail experiments.

In the following, we provide the detail of how keypoints are used to infer the execution-related information to perform the corresponding experiment.

For the grasp affordance, we employ 2.5D top-down grasp, which means we only need grasping location and orientation to perform valid grasp. The keypoint 5 indicates the location while keypoints 3 and 4 represent the orientation. 

For the wrap-grasp affordance, we employ 2.5D grasping strategy. Since the mugs/cups are always placed on the table with top-down pose, the orientation information is known. During the experiment, keypoint 5 indicates the grasping location. For the case where mugs/cups are placed with random pose, keypoints 1 and 2 can be used to infer the orientation. 

For the contain affordance, the only information required is the location to drop the grasped object. The 2D keypoint 5 isn't enough to obtain the correct 3D dropping location. THe height information should be acquired from keypoints 1 and 2.

For the cut affordance, keypoint 2 indicate the contacting, while location and keypoints 1 and 2 indicate the operational direction.

For the pound affordance, keypoint 2 indicate the contacting, while location and keypoints 1 and 2 indicate the operational direction.

## Manipulation
We used Robot Operating System(ROS) for robotic mainpulation. All source code is uploaded to this [Github repository](https://github.com/ivaROS/ivaHandyExperiment).
If you'd like to reproduce the experiment, please remember that for experiments of object arrangement, cutting the string and pounding the nail, the locations of placing the objects, the string and the nail are predefied. You should modify these locations by your own experimental setup.
