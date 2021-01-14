## Introduction
The physical experiments involve the perception and manipulation modules. The perception module mainly works for predicting the affordance keypoints and converting the computed keypoints to the operational location and direction. The 2D information will be converted into 3D space with the depth information. The manipulation module is mainly responsible for planning the trajectory and performing the execution to achieve the task. 

## Perception
We uploaded all experiment scripts [here](https://github.com/ivalab/AffKpNet/tree/master/exp). There should be 6 scripts in total. 3 (grasp, contain, w-grasp) for the single object manipulation experiment. 1 for the object arrangment task. 2 (cut and pound) for the cutting the string and pounding the nail experiments.

## Manipulation
We used Robot Operating System(ROS) for robotic mainpulation. All source code is uploaded to this [Github repository](https://github.com/ivaROS/ivaHandyExperiment).
If you'd like to reproduce the experiment, please remember that for experiments of object arrangement, cutting the string and pounding the nail, the locations of placing the objects, the string and the nail are predefied. You should modify these locations by your own experimental setup.
