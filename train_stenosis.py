
from pathlib import Path
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ROOT = Path("backend/app/data/arcade/arcade/arcade")
MASK_ROOT = Path("backend/results/arcade_masks")

TRAIN_IMAGE_DIR = ROOT / "stenosis/train/images"
TRAIN_MASK_DIR = MASK_ROOT / "stenosis/train"
TRAIN_ANNOTATION = ROOT / "stenosis/train/annotations/train.json"

VAL_IMAGE_DIR = ROOT / "stenosis/val/images"
VAL_MASK_DIR = MASK_ROOT / "stenosis/val"
VAL_ANNOTATION = ROOT / "stenosis/val/annotations/val.json"

# Save v4 separately so the v3 checkpoint is not overwritten.
OUTPUT_PATH = Path("backend/results/unet_stenosis_v4.pth")

DEVICE = torch.device("cpu")

BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 0.0003

# V4 change: reduced from 30 to 10
POSITIVE_WEIGHT = 10.0

# Keep the same Tversky settings as v3
ALPHA = 0.7
BETA = 0.3
THRESHOLD = 0.5

torch.manual_seed(42)
np.random.seed(42)


# --------------------------------------------------
# 2. DATASET
# --------------------------------------------------

class StenosisDataset(Dataset):

    def __init__(self, image_dir, mask_dir, annotation_file):

        with annotation_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        stenosis_ids = {
            category["id"]
            for category in data["categories"]
            if category["name"].lower() == "stenosis"
        }

        if not stenosis_ids:
            raise RuntimeError("Stenosis category not found.")

        annotated_ids = {
            annotation["image_id"]
            for annotation in data["annotations"]
            if annotation["category_id"] in stenosis_ids
        }

        self.image_paths = []

        for info in data["images"]:

            if info["id"] not in annotated_ids:
                continue

            image_path = image_dir / info["file_name"]

            mask_path = (
                mask_dir /
                f"{Path(info['file_name']).stem}_mask.png"
            )

            if image_path.exists() and mask_path.exists():
                self.image_paths.append((image_path, mask_path))

        self.image_paths.sort(key=lambda pair: pair[0].name)

        if not self.image_paths:
            raise RuntimeError(
                f"No annotated image-mask pairs found in {image_dir}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path, mask_path = self.image_paths[index]

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None or mask is None:
            raise RuntimeError(f"Could not read {image_path.name}")

        if image.shape != mask.shape:
            raise RuntimeError(
                f"Image-mask size mismatch: {image_path.name}"
            )

        # Original preprocessing: grayscale + normalization
        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        return (
            torch.from_numpy(image).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0)
        )


# --------------------------------------------------
# 3. U-NET ARCHITECTURE
# --------------------------------------------------

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

        d3 = self.dec3(
            torch.cat([self.up3(b), e3], dim=1)
        )

        d2 = self.dec2(
            torch.cat([self.up2(d3), e2], dim=1)
        )

        d1 = self.dec1(
            torch.cat([self.up1(d2), e1], dim=1)
        )

        return self.output(d1)


# --------------------------------------------------
# 4. BCE + TVERSKY LOSS
# --------------------------------------------------

def combined_loss(logits, targets):

    pos_weight = torch.tensor(
        [POSITIVE_WEIGHT],
        device=logits.device
    )

    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight
    )

    probabilities = torch.sigmoid(logits)

    tp = (probabilities * targets).sum(dim=(1, 2, 3))

    fp = (
        probabilities * (1 - targets)
    ).sum(dim=(1, 2, 3))

    fn = (
        (1 - probabilities) * targets
    ).sum(dim=(1, 2, 3))

    tversky = (
        (tp + 1.0) /
        (tp + ALPHA * fp + BETA * fn + 1.0)
    )

    tversky_loss = 1.0 - tversky.mean()

    return 0.5 * bce + 0.5 * tversky_loss


# --------------------------------------------------
# 5. VALIDATION
# --------------------------------------------------

def validate(model, loader):

    model.eval()

    dice_scores = []
    precision_scores = []
    recall_scores = []

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            probabilities = torch.sigmoid(model(images))
            predictions = probabilities >= THRESHOLD
            targets = masks >= 0.5

            for i in range(len(images)):

                pred = predictions[i, 0]
                target = targets[i, 0]

                tp = (pred & target).sum().item()
                fp = (pred & ~target).sum().item()
                fn = (~pred & target).sum().item()

                dice = (
                    2 * tp / max(2 * tp + fp + fn, 1)
                )

                precision = (
                    tp / max(tp + fp, 1)
                )

                recall = (
                    tp / max(tp + fn, 1)
                )

                dice_scores.append(dice)
                precision_scores.append(precision)
                recall_scores.append(recall)

    return (
        np.mean(dice_scores),
        np.mean(precision_scores),
        np.mean(recall_scores)
    )


# --------------------------------------------------
# 6. TRAINING
# --------------------------------------------------

def main():

    train_dataset = StenosisDataset(
        TRAIN_IMAGE_DIR,
        TRAIN_MASK_DIR,
        TRAIN_ANNOTATION
    )

    val_dataset = StenosisDataset(
        VAL_IMAGE_DIR,
        VAL_MASK_DIR,
        VAL_ANNOTATION
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("=" * 55)
    print("STENOSIS SEGMENTATION - U-NET V4")
    print("=" * 55)

    print("Device:", DEVICE)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Positive weight:", POSITIVE_WEIGHT)
    print("Tversky alpha:", ALPHA)
    print("Tversky beta:", BETA)
    print("Threshold:", THRESHOLD)

    model = UNet().to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_dice = -1.0

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for batch_index, (images, masks) in enumerate(train_loader):

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)

            loss = combined_loss(logits, masks)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

            if (batch_index + 1) % 100 == 0:

                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {batch_index + 1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        average_loss = total_loss / len(train_loader)

        val_dice, val_precision, val_recall = validate(
            model,
            val_loader
        )

        print("\nEpoch", epoch + 1, "completed")
        print("Training loss:", round(average_loss, 4))
        print("Validation Dice:", round(val_dice, 4))
        print("Validation Precision:", round(val_precision, 4))
        print("Validation Recall:", round(val_recall, 4))

        if val_dice > best_dice:

            best_dice = val_dice

            OUTPUT_PATH.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                OUTPUT_PATH
            )

            print("Best model checkpoint saved.")

        print("-" * 55)

    print("\nV4 training completed.")
    print("Best validation Dice:", round(best_dice, 4))
    print("Best model:", OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
