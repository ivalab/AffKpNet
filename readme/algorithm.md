## Introduction
This tutorial mainly discusses the keypoint grouping algorithm. The pipeline can be mainly divided into four steps. We include rich figures for better illustration. The example
is shown in the following:

![image](../img/fig_kp_git_img.png)

Fig. 1 - Example image.

The pseudo-code is attach for better understanding. 

![image](../img/pseudo_code.png)

Algorithm. 1 - Pseudo code for the keypoint grouping algorithm.

## Step 1
As shown in the Fig. 2-6, top-N keypoints will be selected from each heatmap. Since there are five heatmaps, there will five sets of selected keypoints. After collecting keypoints 
from heatmap, associated offset and embedding values will be extracted from offset and embedding heatmap based on the coordinates of the keypoint. All selected keypoints will form
N1\*N2\*N3\*N4\*N5 groups. In order to rank each group, the score will be computed as the equation 1.

![image](../img/fig_kp_git_step1_1.png)

Fig. 2 - Select top keypoints from the heatmap of keypoint 1.

![image](../img/fig_kp_git_step1_2.png)

Fig. 3 - Select top keypoints from the heatmap of keypoint 2.

![image](../img/fig_kp_git_step1_3.png)

Fig. 4 - Select top keypoints from the heatmap of keypoint 3.

![image](../img/fig_kp_git_step1_4.png)

Fig. 5 - Select top keypoints from the heatmap of keypoint 4.

![image](../img/fig_kp_git_step1_5.png)

Fig. 6 - Select top keypoints from the heatmap of keypoint 5.
