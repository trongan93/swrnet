import torch
import torch.nn as nn
import torch.nn.functional as F

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class SimpleCNN(nn.Module):
    """
    5-layer fully conv CNN
    """
    def __init__(self, n_channels, n_class):
        super().__init__()

        self.conv = nn.Sequential(
            double_conv(n_channels, 64),
            double_conv(64, 128),
            nn.Conv2d(128, n_class, 1)
        )
        
    def forward(self, x):
        res = self.conv(x)
        return res
    
class SEblock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x
    
class Res2Net(nn.Module):
    expansion = 4  #輸出channel=輸入channel*expansion

    def __init__(self, inplanes, planes, downsample=None, stride=1, scales=4, groups=1, se=False,  norm_layer=None):
        super(Res2Net, self).__init__()
        if planes % scales != 0:   # 输出通道数为4的倍数
            raise ValueError('Planes must be divisible by scales')
        
        bottleneck_planes = groups * planes    
        self.inconv = nn.Sequential(
            nn.Conv2d(inplanes, bottleneck_planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(bottleneck_planes))
          
        self.x3conv = nn.Sequential(
            nn.Conv2d(bottleneck_planes // scales, bottleneck_planes // scales,  kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(bottleneck_planes // scales))
       
        self.outconv = nn.Sequential(
            nn.Conv2d(bottleneck_planes, bottleneck_planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(bottleneck_planes),
            nn.ReLU(inplace=True))
        
        self.se = SEblock(bottleneck_planes)
 

    def forward(self, x):
        # identity = x
        identity = self.inconv(x)
        
        xs = torch.chunk(identity, 4, 1)
        # 用来将tensor分成很多个块，简而言之就是切分吧，可以在不同维度上切分。
        ys = []
        for s in range(4):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.x3conv(xs[s]))
            else:
                ys.append(self.x3conv(xs[s] + ys[s-1]))
        out = torch.cat(ys, 1)
        out = self.outconv(out)
        out = self.se(out)
        out += identity
        
        return out