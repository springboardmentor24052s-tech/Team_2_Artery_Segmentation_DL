import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=(32,64,128,256)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f)); ch = f
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        ch = features[-1]*2
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(ch, f, 2, 2))
            self.ups.append(DoubleConv(f*2, f))
            ch = f
        self.final = nn.Conv2d(features[0], out_channels, 1)
    def forward(self, x):
        skips=[]
        for down in self.downs:
            x=down(x); skips.append(x); x=self.pool(x)
        x=self.bottleneck(x); skips=skips[::-1]
        for i in range(0,len(self.ups),2):
            x=self.ups[i](x); skip=skips[i//2]
            if x.shape[2:] != skip.shape[2:]:
                x=nn.functional.interpolate(x,size=skip.shape[2:],mode="bilinear",align_corners=False)
            x=torch.cat((skip,x),dim=1); x=self.ups[i+1](x)
        return self.final(x)
