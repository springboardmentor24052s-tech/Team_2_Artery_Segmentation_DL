
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Paths
ROOT = Path("backend/app/data/arcade/arcade/arcade")
MASK_ROOT = Path("backend/results/arcade_masks")

IMAGE_DIR = ROOT / "syntax/val/images"
MASK_DIR = MASK_ROOT / "syntax/val"
MODEL_PATH = Path("backend/results/unet_syntax_improved.pth")
OUTPUT_DIR = Path("backend/results/unet_predictions")

DEVICE = torch.device("cpu")
BATCH_SIZE = 2


# Dataset
class ArcadeDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.image_paths = sorted(self.image_dir.glob("*.png"))
        self.image_paths = [
            p for p in self.image_paths
            if (self.mask_dir / f"{p.stem}_mask.png").exists()
        ]

        if not self.image_paths:
            raise RuntimeError("No validation image-mask pairs found.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        mask_path = self.mask_dir / f"{path.stem}_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise RuntimeError(f"Could not load {path.name}")

        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        return (
            torch.from_numpy(image).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
            path.name,
        )


# Same U-Net architecture used for training
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(1, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(64, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = DoubleConv(32, 16)
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.output(d1)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    dataset = ArcadeDataset(IMAGE_DIR, MASK_DIR)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = UNet().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_images = 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for images, masks, filenames in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            probabilities = torch.sigmoid(model(images))
            predictions = probabilities >= 0.5
            targets = masks >= 0.5

            for i, filename in enumerate(filenames):
                pred = predictions[i, 0]
                target = targets[i, 0]

                tp = (pred & target).sum().item()
                fp = (pred & ~target).sum().item()
                fn = (~pred & target).sum().item()

                dice = (2 * tp) / max(2 * tp + fp + fn, 1)
                iou = tp / max(tp + fp + fn, 1)
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)

                total_dice += dice
                total_iou += iou
                total_precision += precision
                total_recall += recall
                total_images += 1

                # Save binary prediction mask.
                pred_array = pred.cpu().numpy().astype(np.uint8) * 255
                cv2.imwrite(
                    str(OUTPUT_DIR / f"{Path(filename).stem}_pred.png"),
                    pred_array,
                )

    print("\nValidation results")
    print("Images evaluated:", total_images)
    print(f"Mean Dice:      {total_dice / total_images:.4f}")
    print(f"Mean IoU:       {total_iou / total_images:.4f}")
    print(f"Mean Precision: {total_precision / total_images:.4f}")
    print(f"Mean Recall:    {total_recall / total_images:.4f}")
    print("Predicted masks saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
