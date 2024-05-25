import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self , in_chan , out_chan):
        super(DoubleConv, self).__init__()
        self.DoubleConv = nn.Sequential(
            nn.Conv2d(in_chan , out_chan , kernel_size = (3,5) , stride = 1, padding = 0),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(out_chan , out_chan , kernel_size = (3,5) , stride = 1, padding = 0),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self , x):
        return self.DoubleConv(x)


class Dec(nn.Module):
    def __init__(self , in_chan , out_chan , kc1 , pc1 , kc2 , pc2):
        super(Dec, self).__init__()
        self.Dec = nn.Sequential(
            nn.ConvTranspose2d(in_chan, out_chan, (1,1), stride=2, padding=(4,7)),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(out_chan, out_chan , (2,1), stride=3, padding=(2,7)),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(out_chan, out_chan , 2, stride=3, padding=(2,6)),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_chan , out_chan , kernel_size = kc1 , stride = 1, padding = pc1),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_chan , out_chan , kernel_size = kc2 , stride = 1, padding = pc2),
            nn.BatchNorm2d(out_chan),
            nn.Softmax(dim=1)
        )

    def forward(self , x):
        return self.Dec(x)


class ZeroNeurons(nn.Module):
    def __init__(self):
        super(ZeroNeurons, self).__init__()
        self.conv1 = DoubleConv(in_chan = 4 , out_chan = 64)
        self.conv2 = DoubleConv(in_chan = 64 , out_chan = 128)
        self.deconv = Dec(in_chan = 128 , out_chan= 60 , kc1 = (5,5) , pc1 = (0,0) , kc2 = (4,5) , pc2 = (0,0))

    def forward(self , y):
        x1 = self.conv1(y)
        #print('x1' , x1.shape)
        x2 = self.conv2(x1)
        #print('x2' , x2.shape)
        x3 = self.deconv(x2)
        #print('x6' , x3.shape)
        #print()

        return x3