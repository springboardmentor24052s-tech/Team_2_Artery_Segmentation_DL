import torch
from attention_unet import AttentionUNet

model = AttentionUNet()

sample = torch.randn(1, 1, 512, 512)

with torch.no_grad():
    output = model(sample)

print("Input shape:", sample.shape)
print("Output shape:", output.shape)
print("Architecture test completed!")