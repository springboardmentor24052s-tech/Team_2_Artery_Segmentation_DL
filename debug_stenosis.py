
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from train_stenosis import UNet


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path("backend/app/data/arcade/arcade/arcade")

MODEL_PATH = Path("backend/results/unet_stenosis.pth")

IMAGE_PATH = ROOT / "stenosis/val/images/100.png"


# --------------------------------------------------
# 2. Check required files
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")


# --------------------------------------------------
# 3. Load the trained model
# --------------------------------------------------

device = torch.device("cpu")

model = UNet().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully.")


# --------------------------------------------------
# 4. Load and preprocess the image
# --------------------------------------------------

# The stenosis U-Net expects a single grayscale channel.
image = Image.open(IMAGE_PATH).convert("L")

image = np.array(image, dtype=np.float32) / 255.0

# Shape before adding dimensions: (512, 512)
# Required model shape: (batch, channel, height, width)
image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

image_tensor = image_tensor.to(device)

print("Input tensor shape:", tuple(image_tensor.shape))


# --------------------------------------------------
# 5. Generate raw predictions
# --------------------------------------------------

with torch.no_grad():
    output = model(image_tensor)

    # Convert logits into probabilities between 0 and 1.
    probabilities = torch.sigmoid(output).squeeze().cpu().numpy()


# --------------------------------------------------
# 6. Display prediction statistics
# --------------------------------------------------

print("\nStenosis prediction statistics")
print("--------------------------------")
print("Minimum probability:", float(probabilities.min()))
print("Maximum probability:", float(probabilities.max()))
print("Mean probability:", float(probabilities.mean()))

print("\nPixels above each threshold")
print("--------------------------------")

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
    count = int((probabilities > threshold).sum())
    print(f"Threshold {threshold:.1f}: {count} pixels")


# --------------------------------------------------
# 7. Save a probability map for visual inspection
# --------------------------------------------------

OUTPUT_PATH = Path("backend/results/stenosis_debug_probability.png")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

probability_image = np.clip(probabilities * 255, 0, 255).astype(np.uint8)

Image.fromarray(probability_image).save(OUTPUT_PATH)

print("\nProbability map saved to:", OUTPUT_PATH.resolve())
print("Debugging completed.")
