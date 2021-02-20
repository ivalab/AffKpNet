# Overview
The objective of this experiment is to compare the performance gap between the segmentation and keypoint model and the keypoint-only model in the real-world physical 
experiment. The segmentation and keypoint model is the proposed model with both segmentation and keypoint branches. The segmentation branch is served as supplementary task to form 
the  multi-task learning framework. The keypoint-only model represents the model with only keypoint detection branch. We've shown in the paper that the segmentation and keypoint 
model outperforms the keypoint-only model in vision benchmark. Here, we design this experiment to explore the performance of the keypoint-only model in the physical experiment.

# Experimental Setup
The test considers manipulating objects from unseen instances and categories. The affordance testes are: grasp, contain, and wrap-grasp. For each affordance, there are 2 obejcts
with 10 trials per object. Each object will be placed on the table with a random location and orientation in z-axis.

## Grasp
We employ top-down (2.5D) grasp strategy when performing grasp action. During each trial, the manipulator will first approach above the object. Then it will come down and close 
the gripper to grab the object. In the end, it will lift the object into the air.

## Wrap-grasp
We employ the similar 2.5D grasp strategy. The manipulator will only approach the object in one direction, which means the manipulator only needs to know how far it should travel.
During each trial, the manipulator will first move to place in front of the object. Then it will move forward and then close the gripper. In the end, the object will be lifted 
into the air.

# Contain
During each trial, the manipulator will transport the pre-grasped object into the place indicated by the contain affordance. The manipulator will move above the object and release
the gripper.

# Object set
TBD

# Evaluation metric

## Grasp
Each trial is considered as successful if the object is lifted in the air for 3 seconds.

## Wrap-grasp
Each trial is considered as successful if the object is lifted in the air for 3 seconds.

# Contain
Each trial is considered as successful if the pre-grasped object falls into the tested object.

# Result

## Unseen object instance
|   |  | Keypoint-only |  |  |
| :----------: | :----------: | :----------: | :----------: | :----------: |
| Object  | Perception  | Plan  | Action  | Affordance  |
| knife  | 7  | 7  | 7  |  grasp |
| trowel  | 9  | 9  | 9  |  grasp |
| bowl  |  9 |  9 | 9  | contain  |
| mug  |  10 | 10  | 10  | contain  |
| cup  | 9  | 9  |  9 | wrap-grasp  |
| mug  | 10  | 10  | 8  |  wrap-grasp |
| Success rate  | 90.0%  |  90.0% |  86.7% |   |

## Unseen object category
|   |  | Keypoint-only |  |  |
| :----------: | :----------: | :----------: | :----------: | :----------: |
| Object  | Perception  | Plan  | Action  | Affordance  |
| letter opener  | 10  |  10 |  10 | grasp  |
| wrench  |  9 |  9 |  9 |  grasp |
| jar  | 9  | 9  | 9  | contain  |
| pot  | 10  | 10  |  6 |  contain |
| pepsi can  | 4  | 4  | 4  |  wrap-grasp |
| medicine bottle  | 5  | 5  | 5  | wrap-grasp  |
| Success rate  |  78.3%  | 78.3%  | 71.7%  |   |

# Discussion
As shown in the Table of unseen object instance, the keypoint-only model achieves 90.0% success rate for Perception and Plan stage, and 86.7% for Action stage. Compared to the result of AffKp, even though the performance drop isn't too large, we observed that the keypoint-only model provided relatively poor affordance keypoint detection during the experiment. First of all, there are more cases of no detection appeared. For example, the three failure cases of knife and the only failure case of bowl were caused by no detection result of the certain affordance. Secondly, the model shows the relatively poor capability of differentiating symmetric keypoints. Taking the failure case of trowel for example, the keypoints 3 and 4 were predicted into the same side such that grasp orientation was predicted wrong. The same issue happened for the failure of the cup in the wrap-grasp task. The keypoints 1 and 2 of cup were predicted into the same side such that the estimated radius was nearly zero, which leading to that the gripper can't move deep enough to perform the robust grasp closure.

For the result of unseen object category, the keypoint-only model achieves 78.3% success rate for Perception and Plan stage, and 71.7% for Action stage.
