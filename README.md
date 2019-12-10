## Delving Deep into Spatial Pooling for Squeeze-and-Excitation Networks
Xin Jin, Yanping Xie, Xiu-Shen Wei*, Borui Zhao, Xiaoyang Tan

This repository is the official PyTorch implementation of paper "Delving Deep into Spatial Pooling for Squeeze-and-Excitation Networks". The paper is under revision, and will be released after acceptance.

## Introduction
Squeeze-and-Excitation (SE) blocks have demonstrated significant accuracy gains for state-of-the-art deep architectures by re-weighting channel-wise feature responses. The SE block is an architecture unit that integrates two operations: a squeeze operation that employs global average pooling to aggregate spatial convolutional features into a channel feature, and an excitation operation that learns instance-specific channel weights from the squeezed feature to re-weight each channel. In this paper, we revisit the squeeze operation in SE blocks, and shed lights on why and how to embed rich (both global and local) information into the excitation module at minimal extra costs. In particular, we introduce a simple but effective two-stage spatial pooling process: rich descriptor extraction and information fusion. The rich descriptor extraction step aims to obtain a set of diverse (i.e., global and especially local) deep descriptors that contain more informative cues than global average-pooling. While, absorbing more information delivered by these descriptors via a fusion step can aid the excitation operation to return more accurate re-weight scores in a data-driven manner. We validate the effectiveness of our method by extensive experiments on ImageNet for image classification and on MS-COCO for object detection and instance segmentation. For these experiments, our method achieves consistent improvements over the SENets on all tasks, in some cases, by a large margin.

In particular, we introduce two implementation strategies for rich descriptor extraction: (a) spatial pyramid (SP) pooling that generates a fixed number of deep descriptors for all stages of a CNN, and (b) resolution-guided (RG) pooling that generates stage-aware number of deep descriptors by using a fixed pooling window across all stages.The proposed two improvements of SENets are referred as **SP-SENet** and **RG-SENet**, respectively.
    
## Requirements

    numpy
    
    torch-1.1.0
    
    torchvision-0.3.0
    

## Usage
    
    1.download your dataset by yourself, such as ImageNet-1k
    
    2.create a list for your dataset,such as 
        imagename label
        xxx.jpg 1
        xxx.jpg 3
        xxx.jpg 999
    
    3.python3 imagenet_train.py --test_data_path your_path --train_data_path  your_path -a RGSE50 --epochs 100 --schedule 30 -b 256 --lr 0.1

## Options
- `lr`: learning rate
- `lrp`: factor for learning rate of pretrained layers. The learning rate of the pretrained layers is `lr * lrp`
- `batch-size`: number of images per batch
- `image-size`: size of the image
- `epochs`: number of training epochs
- `evaluate`: evaluate model on validation set
- `resume`: path to checkpoint

## Key Results
**Image classification results on ImageNet-1k, where single-crop validation errors are reported.**

Model | Top-1 err. | Top5 err. | GFLOPs
:- | :-: | :-: | :-: 
ResNet-50 | 24.48 | 7.49| 3.86 |
ResNet-50 + SE | 23.21| 6.60| 3.87|
ResNet-50 + CBAM | 22.65| 6.32| 3.90|
ResNet-50 + SPSE (Ours) | 22.40| 6.18| 3.88|
ResNet-50 + RGSE (Ours) | **22.27**| **6.15**| 3.87|
ResNet-101 | 23.28 | 6.70 | 7.58 |
ResNet-101 + SE | 22.35 | 6.13 | 7.60 |
ResNet-101 + CBAM| 21.49 | 5.68 | 7.64 |
ResNet-101 + SPSE (Ours) | 21.27 | 5.66 | 7.62 |
ResNet-101 + RGSE (Ours) | **21.24** | **5.62** | 7.60 |
ResNet-152 | 22.44 | 6.37| 11.30 |
ResNet-152 + SE| 21.59 | 5.74| 11.32 |
ResNet-152 + CBAM | 21.37 | 5.70| 11.39 |
ResNet-152 + SPSE (Ours)| 21.16 | 5.60| 11.36 |
ResNet-152 + RGSE (Ours)| **21.09** | **5.58**| 11.32 |
ResNeXt-50 | 22.72 | 6.44 | 3.89 |
ResNeXt-50 + SE | 21.89 | 6.02 | 3.90 |
ResNeXt-50 + CBAM | 21.91 | 5.89 | 3.93 |
ResNeXt-50 + SPSE (Ours)| 21.42 | 5.76 | 3.91 |
ResNeXt-50 + RGSE (Ours)| **21.36** | **5.72** | 3.90 |
ResNeXt-101 | 21.53 | 5.77| 7.63 |
ResNeXt-101 + SE | 21.18 | 5.67| 7.65 |
ResNeXt-101 + CBAM| 21.10 | 5.58| 7.69 |
ResNeXt-101 + SPSE (Ours) | 20.71 | 5.45| 7.67 |
ResNeXt-101 + RGSE (Ours)| **20.67** | **5.43**| 7.65 |


## Contacts
If you have any questions about our work, please do not hesitate to contact us by emails.

Xiu-Shen Wei: weixs.gm@gmail.com

Xin Jin: x.jin@nuaa.edu.cn

Yanping Xie: nuaaxyp@nuaa.edu.cn

