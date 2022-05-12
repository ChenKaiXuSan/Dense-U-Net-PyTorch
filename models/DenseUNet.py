# %%
from collections import OrderedDict

import torch
import torch.nn as nn
from models.DenseBlock import DenseNet


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=16):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1") # 16
        self.dense1 = DenseNet(growth_rate=1, num_layers=16, num_init_features=features, name='1', efficient=True) # f * 2 = 32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = UNet._block(features * 2, features * 4, name="enc2") # f * 4 = 64
        self.dense2 = DenseNet(growth_rate=4, num_layers=16, num_init_features=features * 4, name='2', efficient=True) # f * 8 = 128 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = UNet._block(features * 8, features * 16, name="enc3") # f * 16 = 256
        self.dense3 = DenseNet(growth_rate=16, num_layers=16, num_init_features=features * 16, name='3', efficient=True) # f * 32 = 512
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # bottleneck 
        self.bottleneck = UNet._block(features * 32, features * 32, name="bottleneck") # f * 32 = 512
        self.bottleneck_dense = DenseNet(growth_rate=32, num_layers=16, num_init_features=features * 32, name = 'bottleneck', efficient=True) # f * 64 = 1024

        self.upconv3 = nn.ConvTranspose2d(
            features * 64, features * 32, kernel_size=2, stride=2
        ) # f * 32 = 512

        self.decoder_skip3 = UNet._block((features * 32) * 2, features * 32, name="dec_skip3") # f * 32 = 512

        self.upconv2 = nn.ConvTranspose2d(
            features * 32, features * 16, kernel_size=2, stride=2
        ) # f * 16 = 256
        self.decoder3 = UNet._block(features * 16, features * 8, name="dec3") # f * 8 = 128

        self.decoder_skip2 = UNet._block((features * 8) * 2, features * 8, name="dec_skip2") # f * 16 = 128

        self.upconv1 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        ) # f * 4 = 64

        self.decoder2 = UNet._block(features * 4, features * 2, name="dec1") # f * 2 = 32 

        self.decoder_skip1 = UNet._block((features * 2) * 2, features * 2, name="dec_skip1") # f * 2 = 32

        # self.upconv1 = nn.ConvTranspose2d(
        #     features * 8, features * 4 , kernel_size=2, stride=2
        # ) # f * 4 = 64

        # self.decoder1 = UNet._block((features * 4) * 2, features * 4, name="dec1") # f * = 64

        self.conv = nn.Conv2d(
            in_channels=features * 2, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        dense1 = self.dense1(enc1)

        enc2 = self.encoder2(self.pool1(dense1))
        dense2 = self.dense2(enc2)

        enc3 = self.encoder3(self.pool2(dense2))
        dense3 = self.dense3(enc3)
        
        bottleneck = self.bottleneck(self.pool3(dense3)) # 512
        bottleneck_dense = self.bottleneck_dense(bottleneck) # 1024

        dec3 = self.upconv3(bottleneck_dense) # 512

        dec3_c = torch.cat((dec3, dense3), dim=1) # 1024
        dec2 = self.decoder_skip3(dec3_c) # 512

        dec2 = self.upconv2(dec2) # 256
        dec2 = self.decoder3(dec2) # 128

        dec2_c = torch.cat((dec2, dense2), dim=1) # 256
        dec1 = self.decoder_skip2(dec2_c) # 128

        dec1 = self.upconv1(dec1) # 64
        dec1 = self.decoder2(dec1) # 32

        dec1_c = torch.cat((dec1, dense1), dim=1) # 64

        dec_fn = self.decoder_skip1(dec1_c) # 32

        return torch.sigmoid(self.conv(dec_fn)) # out channels

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

# %%
from torchinfo import summary

model = UNet()
print(model)

# summary(model, input_size=(3, 3, 256))
# %%
input = torch.randn((3, 3, 256, 256))

pred = model(input)
# %%
