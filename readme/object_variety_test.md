# Overview

The objective of this experiment is to test the generalizability of AffKp on object's variety of instances. We mainly select the mug, spoon and knife as the test object set.
Each set has 8 instances, which vary in geometry, color and texture. Since the mug and cup share the similar geometry and affordance category (_wrap-grasp_), we aggregate to the mug set a cup set consisting of 5 different instances.

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

### Multiple cups

![image](../img/object_variety/multi_cups.png)

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
As seen in the above figures, AffKp provides the robust affordance keypoint detection over the mug and cup test objects set. The key reason is the similar geometry between different mug and cup instances. Since AffKp takes the RG-D image as input, which captures the geometry information, it can provide accurate and robust affordance keypoint detection for different mug and cup instances. Furthermore, keypoint networks are input agnostic. They can be provided inputs or arbitrary size or resolution. When this nuisance factor is included in the training set, the network learns to be robust to sizing. Other networks with constrained input dimensions do not have this affordance to exploit.

The visually separable, multi-object implementation depicted as the last test case for the mug+cup object set depicts this latter affordance being used. The first pass through the network provides a segmentation of the objects and their affordances, which are then grouped for each object to create an object-specific bounding box. Each object image sub-region is cropped and run through the network a second time to arrive at the final affordance segmentation and keypoint outputs.

## Spoon
For the spoon set, as shown in the above figures, AffKp provides satisfactory detection results for the both grasp and scoop part. The network is capable of detecting correct keypoints over different spoon instances. The length of the spoon or width of a spoon's _scoop_ affordance region does not influence the keypoint outcomes. 

The spoon does have flip symmetry, which leads to keypoint 3-4 swaps for the _grasp_ affordance associated to the handle. Keypoints 3 and 4 are not always labeled with a consistent planar handed-ness. The result is that the recoverd 3D frame will have the local z-axis point up or down depending on whether keypoints 3 and 4 are swapped. The keypoints 3 and 4 of the grasp affordance share the similar regional features, which may be difficult to overcome without additional contextual information. That said, the flipped keypoints do not affect the experiment performance since the top-down grasp estimator re-processes the reference frame to arrive at a grasp compatible orientation.

## Knife
For the knife set, we include additional knives commonly found in kitchens (or in cafeterias for the plastic one) but not in the training data set. These knives are better called cleavers or butcher knives; they have long, wide blades. These knives differ from those in the annotated [dataset](https://github.com/ivalab/AffKpNet/blob/master/readme/dataset_visualization.md), which tend to have a blade width to handle width ratio close to unity. There are some with a higher ratio, but not as high as in this additional testing knife set. Furthermore, the plastic knife is fairly different.
 
Another significant similarity for the knives in the annotated dataset is the difference in colors between the handle and blade parts. 
We include some knives whose blade and handle have the same color, which may increase the difficulty of differentiating the two parts.
Lastly, considering all knives in the dataset are made of metal, we add one plastic knife with entirely white body, which is difficult to  differentiate between the cut and grasp parts.

As shown in the above figures, for those two Chinese-style knives (No. 4 and 5), we attach two figures with detection results for each instance. AffKp is capable of providing detection results for both cut and grasp parts in some scenarios but it is possible that the cut part is missing. As the width of the knife grows closer to its length and away from its handle's width, the knife blade starts to resemble a plate or other larger flat surface. This intermediate geometry (not typical blade, not plate) may mislead the network.

The same problem occurrs for the No.1 knife with an all black body. For the No. 3 knife, AffKp can't detect the cut part. AffKp treats the entire knife as the grasp part and predict the grasp affordance keypoints for it. The main reason may be that the blade part is too similar to the handle part. The primary means to differentiate is through the serrated edge, which may not be an internal feature of the network representation for knives.
