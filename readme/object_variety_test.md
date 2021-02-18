# Overview

The objective of this experiment is to test the generalizability of AffKp on object's variety of instances. We mainly select the mug, spoon and knife as the test object set.
Each set has 8 instances, which vary a lot in geometry, color and texture. Since the mug and cup share the similar geometry, we expand the mug set with additional cup set, which 
owns 5 different instances. 

# Object set

## Mug&Cup

![image](../img/object_variety/mug_set.png)

![image](../img/object_variety/cup_set.png)

## Spoon

![image](../img/object_variety/spoon_set.png)

## Knife

![image](../img/object_variety/knife_set.png)

# Detection result

## Mug&Cup

### Mug No.1

![image](../img/object_variety/mug_1.png)

### Mug No.2

![image](../img/object_variety/mug_2.png)

### Mug No.3

![image](../img/object_variety/mug_4.png)

### Mug No.4

![image](../img/object_variety/mug_7.png)

### Mug No.5

![image](../img/object_variety/mug_8.png)

### Mug No.6

![image](../img/object_variety/mug_5.png)

### Mug No.7

![image](../img/object_variety/mug_6.png)

### Mug No.8

![image](../img/object_variety/mug_3.png)

### Cup No.1

![image](../img/object_variety/cup_4.png)

### Cup No.2

![image](../img/object_variety/cup_2.png)

### Cup No.3

![image](../img/object_variety/cup_3.png)

### Cup No.4

![image](../img/object_variety/cup_1.png)

### Cup No.5

![image](../img/object_variety/cup_5.png)

## Spoon

### No.1

![image](../img/object_variety/spoon_6.png)

### No.2

![image](../img/object_variety/spoon_7.png)

### No.3

![image](../img/object_variety/spoon_3.png)

### No.4

![image](../img/object_variety/spoon_8.png)

### No.5

![image](../img/object_variety/spoon_4.png)

### No.6

![image](../img/object_variety/spoon_5.png)

### No.7

![image](../img/object_variety/spoon_1.png)

### No.8

![image](../img/object_variety/spoon_2.png)

## Knife

### No.1

![image](../img/object_variety/knife_7_v1.png)

![image](../img/object_variety/knife_7_v2.png)

### No.2

![image](../img/object_variety/knife_5.png)

### No.3

![image](../img/object_variety/knife_8.png)

### No.4

![image](../img/object_variety/knife_1_v1.png)

![image](../img/object_variety/knife_1_v2.png)

### No.5

![image](../img/object_variety/knife_2.png)

![image](../img/object_variety/knife_2_v2.png)

### No.6

![image](../img/object_variety/knife_3.png)

### No.7

![image](../img/object_variety/knife_6.png)

### No.8

![image](../img/object_variety/knife_4.png)

# Disucssion

## Mug & Cup
For the mug and cup set, as shown in the above figures, AffKp provides the robust affordance keypoint detection over the test object set. The key reason can be the similar geometry between different mug and cup instances. Since AffKp takes the RG-D image as input, which captures the geometry information, it can provide accurate and robust affordance keypoint detection for different mug and cup instances.

## Spoon
For the spoon set, as shown in the above figures, AffKp provides satisfactory detection results for the both grasp and scoop part. The network is capable of detecting correct
keypoints over different spoon instnaces.

The only problem I observe is that AffKp can't differentiate keypoints 3 and 4 for the grasp affordance, which is also a common problem for the other objects in the training 
set. The keypoints 3 and 4 of the grasp affordance share the similar regional features, which causes the difficutly for the network. However, the flipped keypoints 3 and 4 won't 
affect the experiment performance, which the grasp orientation for top-down grasp remains the same.

## Knife
For the knife set, we include some Chinese-style knives, which owns the long and wide blades, to enrich the diversity of the object set. By comparing the knife in the [dataset](https://github.com/ivalab/AffKpNet/blob/master/readme/dataset_visualization.md), 
we can find that all knives in the dataset have long but narrow blade.  
Another significant similarity of knives in the dataset is different colors between the handle and blade part. 
Therefore, we include some knives whose blade and handle own the same color, which raises the difficulty of differentiating different parts.
Lastly, considering all knives in the dataset are made of metal, we add one plastic knife with entirely white body, which is even hard for human to differentiate the cut and grasp parts, into the test set.

As shown in the above figures, for those two Chinese-style knives (No. 4 and 5), we attach two figures with detection results for each instance. AffKp is capable of providing 
detection results for both cut and grasp parts in some scenarios but it is possible that the cut part is missing. I think the main reason is that there is no such kind of 
instance in the training dataset, and the long and wide blade shares the similar geometry with other objects like plate, which is misleading for the network.

The same problem occurred for the No.1 knife with all black body. For the No. 3 knife, AffKp can't detect the cut part. AffKp treats the entire knife as the grasp part and 
predict the grasp affordance keypoints for it. 
I think the main reason is that the blade part is too similar to the handle part. 
As the human. we can only differentiate them by jagged part. 
