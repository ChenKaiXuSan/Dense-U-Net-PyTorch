# U-Net-PyTorch
This repository contains simple PyTorch implementations of U-Net structure.

## Overview
This repository contains an Pytorch implementation of U-Net structure.
With full coments and my code style.

There have two toy structure of U-Net and FCN implementaion.
- [U-Net](toy/unet_pytorch.ipynb)
- [FCN](toy/FCN_pytorch.ipynb)

## About U-Net
If you're new to U-Net structure, here's an abstract straight from the paper[1]:

There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The  architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very
few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU.

## Dataset 
<!-- - MNIST
`python3 main.py --dataset mnist --channels 1`
- FashionMNIST
`python3 main.py --dataset fashion --channels 1`
- Cifar10
`python3 main.py --dataset cifar10 --channels 3` -->

## Implement
``` python

```
## Usage
<!-- - MNSIT  
`python3 main.py --dataset mnist --channels 1 --version [version] --batch_size [] --adv_loss [] >logs/[log_path]`
- FashionMNIST  
`python3 main.py --dataset fashion --channels 1 --version [version] --batch_size [] --adv_loss [] >logs/[log_path]`
- Cifar10  
`python3 main.py --dataset cifar10 --channels 3 -version [version] --batch_size [] --adv_loss [] >logs/[log_path]` -->

<!-- ## FID
FID is a measure of similarity between two datasets of images. It was shown to correlate well with human judgement of visual quality and is most often used to evaluate the quality of samples of Generative Adversarial Networks. FID is calculated by computing the Fréchet distance between two Gaussians fitted to feature representations of the Inception network.

For the FID, I use the pytorch implement of this repository. [FID score for PyTorch](https://github.com/mseitzer/pytorch-fid)

For the 10k epochs training on different dataset, compare with about 10000 samples, I get the FID: 

| dataset | wgan-div |
| ---- | ---- |
| MNIST | 84.4945873660393(5900epoch) |
| FASHION-MNIST | null | 
| CIFAR10 | 54.480231280904434(1000epoch) |
 
> :warning: I dont konw if the FID is right or not, because I cant get the lowwer score like the paper or the other people get it.  -->

## Directory Structure
``` bash
.
|-- README.md
|-- dataset
|   |-- __init__.py
|   |-- brain_dataset.ipynb
|   |-- dataset.png
|   `-- dataset.py
|-- list.txt
|-- main.py
|-- models
|   |-- U_Net.py
|   `-- __init__.py
|-- requirements.txt
|-- toy
|   |-- FCN_pytorch.ipynb
|   |-- helper.py
|   |-- simulation.py
|   `-- unet_pytorch.ipynb
|-- trainer.py
`-- utils
    |-- __init__.py
    `-- utils.py
```

## Reference
1. [U-Net](https://arxiv.org/abs/1505.04597)
2. [FCN](https://arxiv.org/abs/1411.4038)
