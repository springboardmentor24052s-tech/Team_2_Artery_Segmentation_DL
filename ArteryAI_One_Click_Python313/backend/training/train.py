
"""Train U-Net from paired angiography images and binary vessel masks.

From backend/: python -m training.train --epochs 60 --batch-size 4
Matching filenames are required in dataset/images and dataset/masks.
White (>=128) mask pixels are vessel; black pixels are background.
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from models.unet import UNet
from services.preprocessing import preprocess_array, IMAGE_SIZE

BACKEND = Path(__file__).resolve().parents[1]
PROJECT = BACKEND.parent
IMAGE_DIR = PROJECT / "dataset" / "images"
MASK_DIR = PROJECT / "dataset" / "masks"
OUTPUT = BACKEND / "model_weights" / "unet.pth"
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

class PairedVesselDataset(Dataset):
    def __init__(self, pairs=None, augment=False):
        if pairs is None:
            pairs = []
            for image_path in sorted(IMAGE_DIR.iterdir() if IMAGE_DIR.exists() else []):
                if image_path.suffix.lower() in VALID_EXT:
                    mask_path = MASK_DIR / image_path.name
                    if mask_path.exists():
                        pairs.append((image_path, mask_path))
        self.pairs = pairs
        self.augment = augment
        if not self.pairs:
            raise RuntimeError(
                f"No paired images/masks found. Put images in {IMAGE_DIR} "
                f"and same-named binary masks in {MASK_DIR}."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Could not read image/mask pair: {image_path.name}")
        image = preprocess_array(image, IMAGE_SIZE).astype(np.float32) / 255.0
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = (mask >= 128).astype(np.float32)
        if self.augment:
            if random.random() < 0.5:
                image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
            if random.random() < 0.5:
                image, mask = np.flipud(image).copy(), np.flipud(mask).copy()
        return torch.from_numpy(image[None]), torch.from_numpy(mask[None])

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        dice = (2 * (probs * target).sum(dims) + 1) / (
            probs.sum(dims) + target.sum(dims) + 1
        )
        return bce + (1 - dice).mean()

def dice_score(logits, target, threshold=0.5):
    pred = (torch.sigmoid(logits) >= threshold).float()
    dims = (1, 2, 3)
    score = (2 * (pred * target).sum(dims) + 1) / (
        pred.sum(dims) + target.sum(dims) + 1
    )
    return score.mean().item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.validation_split < 0.5:
        parser.error("Use positive epochs/batch-size and validation split between 0 and 0.5.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base = PairedVesselDataset()
    count = len(base)
    if count < 10:
        print(f"WARNING: only {count} paired images; validation metrics may be unstable.")
    if count < 2:
        raise RuntimeError("At least two paired images are required for train/validation split.")
    val_count = max(1, round(count * args.validation_split))
    train_count = count - val_count
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(count, generator=generator).tolist()
    train_pairs = [base.pairs[i] for i in indices[:train_count]]
    val_pairs = [base.pairs[i] for i in indices[train_count:]]
    train_loader = DataLoader(
        PairedVesselDataset(train_pairs, augment=True),
        batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        PairedVesselDataset(val_pairs, augment=False),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    criterion = BCEDiceLoss()
    best_dice = -1.0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.inference_mode():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                val_loss += criterion(logits, masks).item()
                val_dice += dice_score(logits, masks)
        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        val_dice /= max(1, len(val_loader))
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | val_dice={val_dice:.4f}"
        )
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": "UNet",
                "in_channels": 1,
                "out_channels": 1,
                "image_size": IMAGE_SIZE,
                "threshold": 0.5,
                "best_validation_dice": best_dice,
                "epoch": epoch,
                "training_pairs": count,
                "seed": args.seed,
            }, OUTPUT)
            print(f"  Saved best checkpoint: {OUTPUT}")

    report = {
        "checkpoint": str(OUTPUT),
        "paired_images": count,
        "train_images": train_count,
        "validation_images": val_count,
        "best_validation_dice": best_dice,
        "device": str(device),
        "epochs": args.epochs,
    }
    (BACKEND / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nTraining finished. Best checkpoint:", OUTPUT)
    print("Validation Dice is a research metric, not evidence of clinical performance.")

if __name__ == "__main__":
    main()
