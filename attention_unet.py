
import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class AttentionGate(nn.Module):

    def __init__(self, gate_channels, skip_channels, intermediate_channels):
        super().__init__()

        self.gate_conv = nn.Sequential(
            nn.Conv2d(
                gate_channels,
                intermediate_channels,
                kernel_size=1
            ),
            nn.BatchNorm2d(intermediate_channels)
        )

        self.skip_conv = nn.Sequential(
            nn.Conv2d(
                skip_channels,
                intermediate_channels,
                kernel_size=1
            ),
            nn.BatchNorm2d(intermediate_channels)
        )

        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, gate, skip):

        gate_features = self.gate_conv(gate)
        skip_features = self.skip_conv(skip)

        attention_map = self.attention(
            gate_features + skip_features
        )

        return skip * attention_map


class AttentionUNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(1, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(64, 128)

        # Decoder 3
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.att3 = AttentionGate(64, 64, 32)
        self.dec3 = DoubleConv(128, 64)

        # Decoder 2
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.att2 = AttentionGate(32, 32, 16)
        self.dec2 = DoubleConv(64, 32)

        # Decoder 1
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)

        self.att1 = AttentionGate(16, 16, 8)
        self.dec1 = DoubleConv(32, 16)

        # Output layer
        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder 3 with attention
        d3 = self.up3(b)
        a3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, a3], dim=1))

        # Decoder 2 with attention
        d2 = self.up2(d3)
        a2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, a2], dim=1))

        # Decoder 1 with attention
        d1 = self.up1(d2)
        a1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, a1], dim=1))

        # Return segmentation logits
        return self.output(d1)
