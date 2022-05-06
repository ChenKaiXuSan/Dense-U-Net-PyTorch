# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


from torch.utils.data import Dataset, DataLoader

import glob
import numpy as np
import pandas as pd
import random

import os

import cv2

import albumentations as A
from albumentations.pytorch import ToTensor

from sklearn.model_selection import train_test_split

# %%
# File path line length images for later sorting
# len(/kaggle/input/lgg-mri-segmentation/kaggle_3m/TCGA_DU_6404_19850629/TCGA_DU_6404_19850629_ <-!!!43.tif)
BASE_LEN = 91
# len(/kaggle/input/lgg-mri-segmentation/kaggle_3m/TCGA_DU_6404_19850629/TCGA_DU_6404_19850629_43 !!!->.tif)
END_IMG_LEN = 4
# (/kaggle/input/lgg-mri-segmentation/kaggle_3m/TCGA_DU_6404_19850629/TCGA_DU_6404_19850629_43 !!!->_mask.tif)
END_MASK_LEN = 9

# %%
# Raw data
def get_raw_data(opt):
    data_map = []
    for sub_dir_path in glob.glob(opt.dataroot+"*"):
        if os.path.isdir(sub_dir_path):
            dirname = sub_dir_path.split("/")[-1]
            for filename in os.listdir(sub_dir_path):
                image_path = sub_dir_path + "/" + filename
                data_map.extend([dirname, image_path])
        else:
            print("This is not a dir:", sub_dir_path)

    df = pd.DataFrame({"dirname": data_map[::2],
                       "path": data_map[1::2]})

    # masks / not masks
    df_imgs = df[~df['path'].str.contains("mask")]
    df_masks = df[df['path'].str.contains("mask")]

    # data sorting
    imgs = sorted(df_imgs["path"].values,
                  key=lambda x: int(x[BASE_LEN: -END_IMG_LEN]))
    masks = sorted(df_masks["path"].values,
                   key=lambda x: int(x[BASE_LEN:-END_MASK_LEN]))

    # sorting check
    idx = random.randint(0, len(imgs)-1)
    print("path to the image:", imgs[idx])
    print("path to the mask:", masks[idx])

    # final dataframe
    df = pd.DataFrame({
        "patient": df_imgs.dirname.values,
        "image_path": imgs,
        "mask_path": masks
    })

    # adding A/B column for diagnosis
    def positiv_negativ_diagnosis(mask_path):
        value = np.max(cv2.imread(mask_path))
        if value > 0:
            return 1
        else:
            return 0

    df["diagnosis"] = df["mask_path"].apply(
        lambda m: positiv_negativ_diagnosis(m))

    return df

# %%
# DataGenerator
class BrainMriDataset(Dataset):
    def __init__(self, df, transforms):

        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx, 1])
        mask = cv2.imread(self.df.iloc[idx, 2], 0)

        augmented = self.transforms(image=image,
                                    mask=mask
                                    )

        image = augmented['image']
        mask = augmented['mask']

        return image, mask


# %%
# def transforms, use albumentations
def get_Aug_transform(opt):
    strong_transforms = A.Compose([
        A.RandomResizedCrop(width=opt.img_size, height=opt.img_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04,
                        rotate_limit=0, p=0.25),

        # Pixels
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.25),
        A.IAAEmboss(p=0.25),
        A.Blur(p=0.01, blur_limit=3),

        # Affine
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * \
                            0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
        ], p=0.8),

        A.Normalize(p=1.0),
        ToTensor(),
    ])

    transforms = A.Compose([
        A.Resize(width=opt.img_size, height=opt.img_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04,
                        rotate_limit=0, p=0.25),

        A.Normalize(p=1.0),
        ToTensor(),
    ])

    return strong_transforms, transforms


# %%
def get_Dataloader(opt):

    if opt.dataset == 'lgg':
        # get raw data
        df = get_raw_data(opt)

        # split df into train_df and val_df
        train_df, val_df = train_test_split(
            df, stratify=df.diagnosis, test_size=0.1)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        # split train_df into train_df and test_df
        train_df, test_df = train_test_split(
            train_df, stratify=train_df.diagnosis, test_size=0.15)
        train_df = train_df.reset_index(drop=True)

        print(
            f"Train: {train_df.shape} \nVal: {val_df.shape} \nTest: {test_df.shape}")

        # create dataloader
        _, transforms = get_Aug_transform(opt)

        # train dataset
        train_dataset = BrainMriDataset(df=train_df, transforms=transforms)
        train_dataloader = DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=True)

        # val dataset
        val_dataset = BrainMriDataset(df=val_df, transforms=transforms)
        val_dataloader = DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=True)

        # test dataset
        test_dataset = BrainMriDataset(df=test_df, transforms=transforms)
        test_dataloader = DataLoader(
            test_dataset, batch_size=opt.batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


# %%

if __name__ == "__main__":
    class opt:
        dataroot = "/workspace/data/lgg-mri-segmentation/kaggle_3m/"
        dataset = 'lgg'
        img_size = 128
        batch_size = 5

    train, val, test = get_Dataloader(opt)
    for i, (imgs, masks) in enumerate(test):

        print(i, imgs.shape, masks.shape)
        print(masks.shape)

        img = make_grid(imgs, normalize=True).numpy()
        img = np.transpose(img, (1, 2, 0))

        mask = masks
        mask = make_grid(mask, normalize=True).numpy()
        mask = np.transpose(mask, (1, 2, 0))

        plt.imshow(img)
        plt.show()
        plt.imshow(mask)
        plt.show()
        plt.close()
        break

# %%
