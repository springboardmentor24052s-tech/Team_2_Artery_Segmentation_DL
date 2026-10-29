import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1 = DoubleConv(in_channels, 32)
        self.e2 = DoubleConv(32, 64)
        self.e3 = DoubleConv(64, 128)
        self.b = DoubleConv(128, 256)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d3 = DoubleConv(256, 128)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d2 = DoubleConv(128, 64)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d1 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, out_channels, 1)
    def forward(self, x):
        a = self.e1(x); b = self.e2(self.pool(a)); c = self.e3(self.pool(b))
        z = self.b(self.pool(c))
        z = self.d3(torch.cat([self.u3(z), c], 1))
        z = self.d2(torch.cat([self.u2(z), b], 1))
        z = self.d1(torch.cat([self.u1(z), a], 1))
        return self.out(z)
