
import torch
import cv2
from pathlib import Path

from attention_unet import AttentionUNet

DEVICE = torch.device("cpu")

MODEL_PATH = Path(
    "backend/results/attention_unet_stenosis_v3.pth"
)

IMAGE_PATH = Path(
    "backend/app/data/arcade/arcade/arcade/"
    "stenosis/val/images/134.png"
)

model = AttentionUNet().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

image = cv2.imread(
    str(IMAGE_PATH),
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise ValueError("Image could not be loaded.")

normalized = image.astype("float32") / 255.0

tensor = torch.from_numpy(normalized)
tensor = tensor.unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    logits = model(tensor)
    probabilities = torch.sigmoid(logits)

print("Input image shape:", image.shape)
print("Input tensor shape:", tuple(tensor.shape))
print("Raw model output shape:", tuple(logits.shape))
print("Probability output shape:", tuple(probabilities.shape))

print(
    "Probability range:",
    probabilities.min().item(),
    probabilities.max().item()
)

print(
    "Predicted pixels at threshold 0.20:",
    (probabilities >= 0.20).sum().item()
)

print(
    "Predicted pixels at threshold 0.50:",
    (probabilities >= 0.50).sum().item()
)
