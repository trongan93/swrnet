# SWRNET
This repository contains the full source code of the paper:

Trong-An Bui and Pei-Jun Lee, 
"**SWRNet: A Deep Learning Approach for Small Surface Water Area Recognition Onboard Satellite**," 
in _IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing,_ vol. 16, pp. 10369-10380, 2023, doi: 10.1109/JSTARS.2023.3328118.

**Abstract:** 
This article proposes a deep learning approach for small surface water recognition using multispectral satellite imaging, which reduces the computational complexity by 18.66 times and increases the accuracy of surface water recognition by up to 14.1%. The proposed model uses near infrared combined with RGB spectral imagery to increase the accuracy of surface water recognition. In addition, since surface water only accounts for a small percentage of the remote sensing dataset, thus creating an imbalance problem, a proposed loss function is introduced to combine region-based and distribution-based loss. This article introduces an adaptive factor that automatically adjusts the weighting between distribution- and region-based loss functions. The proposed adaptive factor is determined based on the loss value of the previous training step. The mean intersection over union of surface water between predicted and ground truth regions is recorded as 0.80
URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10298631&isnumber=9973430

The main training file of the SWRNet Model is **flooding_transfer_learning-rgb_ir.ipynb**.

Please reference this paper in your manuscript when you use this source code.
