
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from train_stenosis import UNet

ROOT = Path("backend/app/data/arcade/arcade/arcade")
MASK_ROOT = Path("backend/results/arcade_masks/stenosis/val")
MODEL_PATH = Path("backend/results/unet_stenosis.pth")
OUTPUT_DIR = Path("backend/results/stenosis_threshold_tests")

IMAGE_PATH = ROOT / "stenosis/val/images/100.png"
MASK_PATH = MASK_ROOT / "100_mask.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cpu")

model = UNet().to(device)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model.eval()

# Load the validation image as grayscale.
image = np.array(
    Image.open(IMAGE_PATH).convert("L"),
    dtype=np.float32
) / 255.0

image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

# Load the corresponding reference mask.
reference = np.array(
    Image.open(MASK_PATH).convert("L")
) > 0

with torch.no_grad():
    logits = model(image_tensor.to(device))
    probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

print("Image: 100.png")
print("Reference foreground pixels:", int(reference.sum()))
print()

best_dice = -1.0
best_threshold = None

for threshold in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15]:
    prediction = probabilities >= threshold

    intersection = np.logical_and(prediction, reference).sum()
    denominator = prediction.sum() + reference.sum()

    dice = (
        2.0 * intersection / denominator
        if denominator > 0
        else 1.0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Predicted pixels: {int(prediction.sum())} | "
        f"Dice: {dice:.4f}"
    )

    if dice > best_dice:
        best_dice = dice
        best_threshold = threshold

# Save a diagnostic mask at the best tested threshold.
best_prediction = (
    probabilities >= best_threshold
).astype(np.uint8) * 255

output_path = OUTPUT_DIR / "100_threshold_test.png"
Image.fromarray(best_prediction).save(output_path)

print()
print("Best tested threshold:", best_threshold)
print("Best Dice for this image:", round(best_dice, 4))
print("Diagnostic mask saved to:", output_path.resolve())
