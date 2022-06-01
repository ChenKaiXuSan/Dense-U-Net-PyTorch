# Dense-U-Net-PyTorch

This repository contains simple PyTorch implementations of U-Net with Dense Encoder structure.

## Overview

This repository contains an Pytorch implementation of U-Net structure.
And combine the DenseNet into the Encoder part.
With full coments and my code style.

There have two toy structure of U-Net and FCN implementaion.

- [U-Net](toy/unet_pytorch.ipynb)
- [FCN](toy/FCN_pytorch.ipynb)

## About U-Net

<!-- If you're new to U-Net structure, here's an abstract straight from the paper[1]:

There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The  architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very
few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU. -->

## Dataset

Dataset used for development and evaluation was made publicly available on Kaggle: [kaggle.com/mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation).
It contains MR images from [TCIA LGG collection](https://wiki.cancerimagingarchive.net/display/Public/TCGA-LGG) with segmentation masks approved by a board-certified radiologist at Duke University.

![brain](imgs/dataset.png)

## Model

A segmentation model implemented in this repository is U-Net with dense encoder with added batch normalization.

![strucutre](imgs/structure.png)

## Dense Structure

The Dense Block output feature is:

`output feature = pre feature + num layer * growth rate`

The num layer means how many layer in one dense block.
The growth rate means how many filters to add each layer ('k' in paper).

For the lgg brain dataset, we set the num layer and growth rate like:

|               | dense block 1 | dense block 2 | dense block 3 | dense block bottleneck | mean IoU |
| ------------- | ------------- | ------------- | ------------- | ---------------------- | -------- |
| growth rate_1 | 4             | 8             | 16            | 32                     |
| num layer_1   | 4             | 8             | 16            | 16                     | 89% |
| growth rate_2 | 1             | 4             | 16            | 32                     |
| num layer_2   | 16            | 16            | 16            | 16                     | 92% |
| growth rate_3 | 2             | 8             | 32            | 64                     |
| num layer_3   | 8             | 8             | 8             | 8                      | 90%      |

| unet | mean iou |
| ---- | -------- |
| bn16 | 87%  |
| bn32 | 91%  |
| bn64 | 97%  |


## Implement

``` python
usage: main.py [-h] [--model {unet,dense_unet}] [--img_size IMG_SIZE] [--channels CHANNELS] [--version VERSION] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--lr LR] [--beta1 BETA1] [--beta2 BETA2] [--pretrained_model PRETRAINED_MODEL] [--train TRAIN]
               [--parallel PARALLEL] [--dataset {lgg}] [--use_tensorboard USE_TENSORBOARD] [--dataroot DATAROOT] [--log_path LOG_PATH] [--model_save_path MODEL_SAVE_PATH] [--sample_path SAMPLE_PATH] [--log_step LOG_STEP] [--sample_step SAMPLE_STEP] [--model_save_step MODEL_SAVE_STEP]

optional arguments:
  -h, --help            show this help message and exit
  --model {unet,dense_unet}
  --img_size IMG_SIZE
  --channels CHANNELS   number of image channels
  --version VERSION     the version of the path, for implement
  --epochs EPOCHS       numer of epochs of training
  --batch_size BATCH_SIZE
                        batch size for the dataloader
  --num_workers NUM_WORKERS
  --lr LR               use TTUR lr rate for Adam
  --beta1 BETA1
  --beta2 BETA2
  --pretrained_model PRETRAINED_MODEL
  --train TRAIN
  --parallel PARALLEL
  --dataset {lgg}
  --use_tensorboard USE_TENSORBOARD
                        use tensorboard to record the loss
  --dataroot DATAROOT   dataset path
  --log_path LOG_PATH   the output log path
  --model_save_path MODEL_SAVE_PATH
                        model save path
  --sample_path SAMPLE_PATH
                        the generated sample saved path
  --log_step LOG_STEP   every default{10} epoch save to the log
  --sample_step SAMPLE_STEP
                        every default{100} epoch save the generated images and real images
  --model_save_step MODEL_SAVE_STEP
```

## Usage

- dense unet model  
`python3 main.py --version [version] --batch_size [] --model dense_unet >logs/[log_path]`
- unet model
`python3 main.py --version [version] --batch_size [] --model unet >logs/[log_path]`

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
3. [U-Net for brain segmentation](https://github.com/mateuszbuda/brain-segmentation-pytorch)
