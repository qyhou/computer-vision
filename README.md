# computer-vision
models specific to Computer Vision

## models

### backbones

#### CNN
- [ResNet](models/backbones/resnet.py)
  - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), (Microsoft), CVPR-2016
  - [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187), (Amazon), CVPR-2019
- [VGG](models/backbones/vgg.py)
  - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556), (Oxford), ICLR-2015

### segmentation
- [UNet 3+](models/segmentation/unet3p.py)
  - [UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation](https://arxiv.org/abs/2004.08790), (ZJU, SRRSH, Ritsumeikan, Zhejiang Lab), ICASSP-2020
- [UNet++](models/segmentation/unetpp.py)
  - [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165), (ASU), DLMIA-2018
  - [UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation](https://arxiv.org/abs/1912.05074), (ASU), Journal
- [U-Net](models/segmentation/unet.py)
  - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), (Uni Freiburg), MICCAI-2015
