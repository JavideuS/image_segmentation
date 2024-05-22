import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

class DoubleConv(nn.Module):
    def __init__(self , in_chan , out_chan):
        super(DoubleConv, self).__init__()
        self.DoubleConv = nn.Sequential(
            nn.Conv2d(in_chan , out_chan , kernel_size = 3 , stride = 1, padding = 0),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(out_chan , out_chan , kernel_size = 3 , stride = 1, padding = 0),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self , x):
        return self.DoubleConv(x)


class Dec(nn.Module):
    def __init__(self , in_chan , out_chan , kc , pc):
        super(Dec, self).__init__()
        self.Dec = nn.Sequential(
            nn.ConvTranspose2d(in_chan, out_chan, 2, stride=3, padding=(2,4)),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(out_chan, out_chan ,2, stride=3, padding=(2,4)),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_chan , out_chan , kernel_size = kc , stride = 1, padding = pc),
            nn.BatchNorm2d(out_chan),
            nn.Softmax(dim=1)
        )

    def forward(self , x):
        return self.Dec(x)


class Upscaling(nn.Module):
    def __init__(self):
        super(Upscaling, self).__init__()
        self.s = nn.Softmax(dim=1)

    def forward(self , x , o_shape1 , o_shape2):
        return self.s(F.interpolate(x, size=(o_shape1 , o_shape2), mode='bilinear', align_corners=False))


class ZeroNeurons(nn.Module):
    def __init__(self):
        super(ZeroNeurons, self).__init__()
        self.conv1 = DoubleConv(in_chan = 4 , out_chan = 64)
        self.conv2 = DoubleConv(in_chan = 64 , out_chan = 128)
        self.conv3 = DoubleConv(in_chan = 128 , out_chan = 256)
        self.deconv1 = Dec(in_chan = 256, out_chan = 128 , kc=(1,9) , pc=3)
        self.deconv2 = Dec(in_chan = 128 , out_chan= 60 , kc=(2,13) , pc=(6,3))

    def forward(self , y):
        x1 = self.conv1(y)
        #print('x1' , x1.shape)
        x2 = self.conv2(x1)
        #print('x2' , x2.shape)
        x3 = self.conv3(x2)
        #print('x3' , x3.shape)
        x5 = self.deconv1(x3)
        print('x5' , x5.shape)
        x6 = self.deconv2(x5)
        #print('x6' , x6.shape)
        print()

        return x6
