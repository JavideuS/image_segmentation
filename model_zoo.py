import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self , in_chan , out_chan):
        super(DoubleConv, self).__init__()
        self.DoubleConv = nn.Sequential(
            nn.Conv2d(in_chan , out_chan , kernel_size = 3 , stride = 2, padding = 1),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(kernel_size = 2),
            nn.Conv2d(out_chan , out_chan , kernel_size = 3 , stride = 2, padding = 1),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(kernel_size=2)
        )

    def forward(self , x):
        return self.Enc(x)


class Dec(nn.Module):
    def __init__(self , in_chan , out_chan):
        super(Dec, self).__init__()
        self.Dec = nn.Sequential(
            nn.ConvTranspose2d(in_chan, out_chan, 3, stride=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(out_chan, out_chan ,3, stride=3, padding=2),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self , x):
        return self.Dec(x)


class Upscaling(nn.Module):
    def __init__(self):
        super(Upscaling, self).__init__()
        self.s = nn.Softmax(dim=1)

    def forward(self , x , o_shape1 , o_shape2):
        return self.s(F.interpolate(x, size=(o_shape1 , o_shape2), mode='bilinear', align_corners=False))


