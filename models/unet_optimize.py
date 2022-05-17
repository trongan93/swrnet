import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np

from models.architecture import double_conv, SEblock, Res2Net, up_conv, Attblock


class UNet(nn.Module):
    def __init__(self, n_channels, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(n_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out

from .full_u_net import *
class FullUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(FullUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = double_conv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
class SimpleUNet(nn.Module):
    def __init__(self, n_channels, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(n_channels, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)
        self.dconv_down4 = double_conv(128, 256)

        self.maxpool = nn.MaxPool2d(2)

        self.dconv_up3 = double_conv(128 + 256, 128)
        self.dconv_up2 = double_conv(64 + 128, 64)
        self.dconv_up1 = double_conv(64 + 32, 32)

        self.conv_last = nn.Conv2d(32, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out

    
class UNet_dropout(nn.Module):
    def __init__(self, n_channels, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(n_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.dropout = nn.Dropout2d()

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dropout(self.dconv_down1(x))
        x = self.maxpool(conv1)

        conv2 = self.dropout(self.dconv_down2(x))
        x = self.maxpool(conv2)

        conv3 = self.dropout(self.dconv_down3(x))
        x = self.maxpool(conv3)

        x = self.dropout(self.dconv_down4(x))

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv3], dim=1)

        x = self.dropout(self.dconv_up3(x))
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv2], dim=1)

        x = self.dropout(self.dconv_up2(x))
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv1], dim=1)

        x = self.dropout(self.dconv_up1(x))

        out = self.conv_last(x)

        return out
    
class Res2_UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(Res2_UNET, self).__init__()
         
        self.inc = DoubleConv(in_channels, 64)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = Res2Net(64, 128)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = Res2Net(128, 256)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.down3 = Res2Net(256, 512)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.ups=nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


    def forward(self, x):
        skip_connections = []

        x = self.inc(x)
        skip_connections.append(x)
        x = self.pool(x) 
        x = self.down1(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.down2(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.down3(x)
        skip_connections.append(x)
        x = self.pool(x)
        
        x=self.bottleneck(x)
        skip_connections=skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x=self.ups[idx](x)
            skip_connection=skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x=TF.resize(x, size=skip_connection.shape[2:])
                # x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])

            concat_skip=torch.cat((skip_connection, x), dim=1)
            x=self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    
class Res2_AttUNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1):
        super(Res2_AttUNET, self).__init__()
        
        self.inc = DoubleConv(in_channels, 64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = Res2Net(64, 128)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = Res2Net(128, 256)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.down3 = Res2Net(256, 512)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
      
        self.de3 = up_conv(1024, 512)
        self.att3= Attblock(F_g=512, F_l=512, F_int=256)   
        self.up3 = DoubleConv(1024, 512)
        
        self.de2 = up_conv(512, 256)
        self.att2= Attblock(F_g=256, F_l=256, F_int=128)   
        self.up2 = DoubleConv(512, 256)
        
        self.de1 = up_conv(256, 128)
        self.att1= Attblock(F_g=128, F_l=128, F_int=64)   
        self.up1 = DoubleConv(256, 128)
        
        self.de0 = up_conv(128, 64)
        self.att0= Attblock(F_g=64, F_l=64, F_int=32)   
        self.up0 = DoubleConv(128, 64)

        
        self.bottleneck = DoubleConv(512, 1024)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)


    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.pool(x1) 
        x2 = self.down1(x2)
        x3 = self.pool(x2)
        x3 = self.down2(x3)
        x4 = self.pool(x3)
        x4 = self.down3(x4)
        x5 = self.pool(x4)
        
        x5=self.bottleneck(x5)
       
        d4 = self.de3(x5)
        if d4.shape != x4.shape:
                 d4=TF.resize(d4, size=x4.shape[2:])
        t4 = self.att3(d4, x4)
        d4 = torch.cat((t4, d4), dim=1)
        d4 = self.up3(d4)
      
        d3 = self.de2(d4)
        if d3.shape != x3.shape:
                 d3=TF.resize(d3, size=x3.shape[2:])
        t3 = self.att2(g=d3, x=x3)
        d3 = torch.cat((t3, d3), dim=1)
        d3 = self.up2(d3)
        
        d2 = self.de1(d3)
        if d2.shape != x2.shape:
                 d2=TF.resize(d4, size=x2.shape[2:])
        t2 = self.att1(g=d2, x=x2)
        d2 = torch.cat((t2, d2), dim=1)
        d2 = self.up1(d2)
        
        d1 = self.de0(d2)
        if d1.shape != x1.shape:
                 d1=TF.resize(d1, size=x4.shape[2:])
        t1 = self.att0(g=d1, x=x1)
        d1 = torch.cat((t1, d1), dim=1)
        d1 = self.up0(d1)
        
        fx = self.final_conv(d1)
    
        return fx    

    
class Res2_SimpleUNet(nn.Module):
    def __init__(self, n_channels, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(n_channels, 32)
        self.dconv_down2 = Res2Net(32, 64)
        self.dconv_down3 = Res2Net(64, 128)
        self.dconv_down4 = Res2Net(128, 256)

        self.maxpool = nn.MaxPool2d(2)

        self.dconv_up3 = double_conv(128 + 256, 128)
        self.dconv_up2 = double_conv(64 + 128, 64)
        self.dconv_up1 = double_conv(64 + 32, 32)

        self.conv_last = nn.Conv2d(32, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out
    
class AttUNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(AttUNET, self).__init__()
        
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
       
        self.de3 = up_conv(1024, 512)
        self.att3= Attblock(F_g=512, F_l=512, F_int=256)   
        self.up3 = DoubleConv(1024, 512)
        
        self.de2 = up_conv(512, 256)
        self.att2= Attblock(F_g=256, F_l=256, F_int=128)   
        self.up2 = DoubleConv(512, 256)
        
        self.de1 = up_conv(256, 128)
        self.att1= Attblock(F_g=128, F_l=128, F_int=64)   
        self.up1 = DoubleConv(256, 128)
        
        self.de0 = up_conv(128, 64)
        self.att0= Attblock(F_g=64, F_l=64, F_int=32)   
        self.up0 = DoubleConv(128, 64)

        
        self.bottleneck = DoubleConv(512, 1024)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)


    def forward(self, x):

        x1 = self.down1(x)
        x2 = self.pool(x1) 
        x2 = self.down2(x2)
        x3 = self.pool(x2)
        x3 = self.down3(x3)
        x4 = self.pool(x3)
        x4 = self.down4(x4)
        x5 = self.pool(x4)
        
        x5=self.bottleneck(x5)
        
        d4 = self.de3(x5)
        if d4.shape != x4.shape:
                 d4=TF.resize(d4, size=x4.shape[2:])
        t4 = self.att3(d4, x4)
        d4 = torch.cat((t4, d4), dim=1)
        d4 = self.up3(d4)
      
        d3 = self.de2(d4)
        if d3.shape != x3.shape:
                 d3=TF.resize(d3, size=x3.shape[2:])
        t3 = self.att2(g=d3, x=x3)
        d3 = torch.cat((t3, d3), dim=1)
        d3 = self.up2(d3)
        
        d2 = self.de1(d3)
        if d2.shape != x2.shape:
                 d2=TF.resize(d4, size=x2.shape[2:])
        t2 = self.att1(g=d2, x=x2)
        d2 = torch.cat((t2, d2), dim=1)
        d2 = self.up1(d2)
        
        d1 = self.de0(d2)
        if d1.shape != x1.shape:
                 d1=TF.resize(d1, size=x4.shape[2:])
        t1 = self.att0(g=d1, x=x1)
        d1 = torch.cat((t1, d1), dim=1)
        d1 = self.up0(d1)
        
        fx = self.final_conv(d1)
    
        return fx