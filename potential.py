class DynTranspose(nn.Module):
    def __init__(self , in_chan , out_chan , in_size , out_size , kernel = 3 , padding = 1):
        super(DynTranspose, self).__init__()
        self.stride = int(round((2*padding + out_size  - kernel) / (in_size - 1) , 0))
        self.conv = nn.ConvTranspose2d(in_chan , out_chan , kernel , stride = self.stride, padding = padding)

    def forward(self , x):
        print(self.stride)
        s = nn.Softmax(dim=1)
        return s(self.conv(x))