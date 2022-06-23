# AdaptNet-MICCAI2021
This repository provides the official PyTorch implementation of AdaptNet (Shape/Scale Adaptive U-Net).

AdaptNet is initially proposed for semantic segmentation in cataract surgery videos, but can be adopted for any medical or general purpose image segmentation problem.

This neural network architecture is especially designed to deal with severe deformations and scale variations by fusing sequential and parallel feature maps adaptively.

**The overall architecture of AdaptNet:**

<img src="./Network-Architecture-Images/AdaptNet.png" alt="The detailed architecture of the CPF and SFF modules of AdaptNet." width="700">

**The detailed architecture of the CPF and SFF modules of AdaptNet:**

<img src="./Network-Architecture-Images/CPF-SSF.png" alt="The detailed architecture of the CPF and SFF modules of AdaptNet." width="700">

## Citation
If you use AdaptNet for your research, please cite our paper:

```
@InProceedings{10.1007/978-3-030-87237-3_8,
author="Ghamsarian, Negin
and Taschwer, Mario
and Putzgruber-Adamitsch, Doris
and Sarny, Stephanie
and El-Shabrawi, Yosuf
and Schoeffmann, Klaus",
editor="de Bruijne, Marleen
and Cattin, Philippe C.
and Cotin, St{\'e}phane
and Padoy, Nicolas
and Speidel, Stefanie
and Zheng, Yefeng
and Essert, Caroline",
title="LensID: A CNN-RNN-Based Framework Towards Lens Irregularity Detection in Cataract Surgery Videos",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2021",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="76--86",
abstract="A critical complication after cataract surgery is the dislocation of the lens implant leading to vision deterioration and eye trauma. In order to reduce the risk of this complication, it is vital to discover the risk factors during the surgery. However, studying the relationship between lens dislocation and its suspicious risk factors using numerous videos is a time-extensive procedure. Hence, the surgeons demand an automatic approach to enable a larger-scale and, accordingly, more reliable study. In this paper, we propose a novel framework as the major step towards lens irregularity detection. In particular, we propose (I) an end-to-end recurrent neural network to recognize the lens-implantation phase and (II) a novel semantic segmentation network to segment the lens and pupil after the implantation phase. The phase recognition results reveal the effectiveness of the proposed surgical phase recognition approach. Moreover, the segmentation results confirm the proposed segmentation network's effectiveness compared to state-of-the-art rival approaches.",
isbn="978-3-030-87237-3"
}


```

## Acknowledgments

This work was funded by the FWF Austrian Science Fund under grant P 31486-N31.
