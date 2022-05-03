'''
U-Net structure.

For the model structure detail, look model_structure.log
or run this script.
'''
#%%
import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.conv_down1 = double_conv(3, 64)
        self.conv_down2 = double_conv(64, 128)
        self.conv_down3 = double_conv(128, 256)
        self.conv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.conv_up3 = double_conv(256 + 512, 256)
        self.conv_up2 = double_conv(128 + 256, 128)
        self.conv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.conv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.conv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.conv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.conv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.conv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.conv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.conv_up1(x)
        
        out = self.conv_last(x)
        out = torch.sigmoid(out)
        
        return out

# %%
from torchinfo import summary

if __name__ == "__main__":
    
    unet = UNet(1)

    batch_size = 16
    result = summary(unet, input_size=(batch_size, 3, 256, 256))
    print(result)

# %%
# import pprint

# # get model summary as string 
# model_stats = summary(unet, (1, 3, 224, 224), verbose=0)
# summary_str = str(model_stats)
# with open('model_structure.log', 'w') as tf:
#     pprint.pprint(summary_str, stream=tf)
# %%
