# %%
import os

from trainer import Trainer_unet
from trainer_dense import Trainer_dense_unet
from utils.utils import *
from dataset.dataset import get_Dataloader

# set the gpu number
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
# %%
def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='dense_unet', choices=['unet', 'dense_unet'])
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--version', type=str, default='test', help='the version of the path, for implement')

    # Training setting
    parser.add_argument('--epochs', type=int, default=500, help='numer of epochs of training')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for the dataloader')
    parser.add_argument('--num_workers', type=int, default=2)
    # TTUR 
    parser.add_argument('--lr', type=float, default=1e-3, help='use TTUR lr rate for Adam')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str, default='lgg', choices=['lgg'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True, help='use tensorboard to record the loss')

    # Path
    parser.add_argument('--dataroot', type=str, default="/workspace/data/lgg-mri-segmentation/kaggle_3m/", help='dataset path')
    parser.add_argument('--log_path', type=str, default='./logs', help='the output log path')
    parser.add_argument('--model_save_path', type=str, default='./checkpoint', help='model save path')
    parser.add_argument('--sample_path', type=str, default='./samples', help='the generated sample saved path')

    # Step size
    parser.add_argument('--log_step', type=int, default=100, help='every default{10} epoch save to the log')
    parser.add_argument('--sample_step', type=int, default=1000, help='every default{100} epoch save the generated images and real images')
    parser.add_argument('--model_save_step', type=int, default=1000)


    return parser

# %%
def main(config):
    # data loader 
    train_dataloader, val_dataloader, test_dataloader = get_Dataloader(config)

    # delete the exists path
    del_folder(config.sample_path, config.version)
    del_folder(config.log_path, config.version)
    del_folder(config.model_save_path, config.version)

    # create directories if not exist
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.model_save_path, config.version)
    
    if config.train:
        if config.model == 'unet':
            trainer = Trainer_unet(train_dataloader, val_dataloader, test_dataloader, config)
        if config.model == 'dense_unet':
            trainer = Trainer_dense_unet(train_dataloader, val_dataloader, test_dataloader, config)
        trainer.train()
    
# %% 
if __name__ == '__main__':
    config = get_parameters().parse_args()
    print(config)
    main(config)