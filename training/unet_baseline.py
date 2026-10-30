"""
ArteryAI baseline: binary U-Net for coronary anatomy segmentation.

Expected layout:
ArteryAI/
  dataset/syntax/{train,val,test}/images/
  dataset/syntax/{train,val,test}/annotations/{train,val,test}.json
  training/unet_baseline.py

This creates a binary mask by combining all polygon categories except "stenosis".
That target definition is an initial research assumption and should be checked
against the ARCADE annotation protocol before interpreting results clinically.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def polygon_mask(image_info: dict, annotations: list, categories: dict) -> Image.Image:
    """Rasterize supported COCO polygons; exclude the stenosis category."""
    width, height = int(image_info["width"]), int(image_info["height"])
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for ann in annotations:
        category_id = ann.get("category_id")
        category_name = str(categories.get(category_id, "")).strip().lower()
        if category_name == "stenosis":
            continue

        segmentation = ann.get("segmentation", [])
        # RLE is deliberately skipped; this implementation supports polygons.
        if not isinstance(segmentation, list):
            continue

        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
                continue
            points = [
                (float(polygon[i]), float(polygon[i + 1]))
                for i in range(0, len(polygon), 2)
            ]
            draw.polygon(points, fill=1)

    return mask


class ArcadeSyntaxDataset(Dataset):
    def __init__(self, dataset_root: Path, split: str, image_size: int):
        self.image_size = image_size
        split_dir = dataset_root / "syntax" / split
        annotation_path = split_dir / "annotations" / f"{split}.json"
        image_dir = split_dir / "images"

        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing annotation JSON: {annotation_path}")
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image folder: {image_dir}")

        with annotation_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.categories = {int(c["id"]): c["name"] for c in data.get("categories", [])}
        annotations_by_image: Dict[int, list] = {}
        for ann in data.get("annotations", []):
            annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)

        self.records: List[Tuple[dict, Path, list]] = []
        for info in data.get("images", []):
            path = image_dir / info["file_name"]
            if path.is_file():
                self.records.append(
                    (info, path, annotations_by_image.get(int(info["id"]), []))
                )

        if not self.records:
            raise RuntimeError(f"No usable image records found in {split_dir}")

        missing = len(data.get("images", [])) - len(self.records)
        print(
            f"{split}: {len(self.records)} images, "
            f"{sum(len(row[2]) for row in self.records)} annotations"
        )
        if missing:
            print(f"Warning: skipped {missing} missing image files.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        info, path, annotations = self.records[index]

        with Image.open(path) as source:
            image = source.convert("RGB")
        mask = polygon_mask(info, annotations, self.categories)

        # Keep image and mask spatially aligned. Nearest-neighbour for labels.
        image = image.resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        mask = mask.resize(
            (self.image_size, self.image_size), Image.Resampling.NEAREST
        )

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return image_tensor, mask_tensor


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 8):
        super().__init__()
        c = base_channels
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c * 8, c * 16)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = DoubleConv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)
        self.output = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.output(d1)  # logits; sigmoid is applied for metrics/predictions


def dice_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    predictions = torch.sigmoid(logits) >= threshold
    targets = targets >= 0.5

    dims = (1, 2, 3)
    intersection = (predictions & targets).sum(dim=dims).float()
    pred_sum = predictions.sum(dim=dims).float()
    target_sum = targets.sum(dim=dims).float()
    union = (predictions | targets).sum(dim=dims).float()

    eps = 1e-7
    dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probabilities * targets).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
        dice_loss = 1.0 - ((2.0 * intersection + 1e-7) / (denominator + 1e-7))
        return bce_loss + dice_loss.mean()


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_images = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, masks)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            dice, iou = dice_iou_from_logits(logits.detach(), masks)
            total_loss += loss.item() * batch_size
            total_dice += dice * batch_size
            total_iou += iou * batch_size
            total_images += batch_size

    if total_images == 0:
        raise RuntimeError("DataLoader produced no batches.")

    return {
        "loss": total_loss / total_images,
        "dice": total_dice / total_images,
        "iou": total_iou / total_images,
    }


def save_prediction_examples(model, dataset, device, output_dir: Path, count: int = 8):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    for index in range(min(count, len(dataset))):
        image, mask = dataset[index]
        with torch.no_grad():
            probability = torch.sigmoid(model(image.unsqueeze(0).to(device)))[0, 0]
        prediction = (probability.cpu().numpy() >= 0.5).astype(np.uint8) * 255
        target = (mask[0].numpy() >= 0.5).astype(np.uint8) * 255
        image_array = (image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

        Image.fromarray(image_array).save(output_dir / f"image_{index + 1:02d}.png")
        Image.fromarray(target).save(output_dir / f"ground_truth_{index + 1:02d}.png")
        Image.fromarray(prediction).save(output_dir / f"prediction_{index + 1:02d}.png")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate ArteryAI baseline U-Net.")
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parent.parent / "dataset")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs" / "unet_baseline")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        parser.error("epochs/batch-size must be positive and image-size must be >= 32.")
    if args.image_size % 16 != 0:
        parser.error("--image-size must be divisible by 16 (for example, 128 or 256).")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("CPU training selected. Start with 128/256 image size and batch size 1.")

    train_dataset = ArcadeSyntaxDataset(args.dataset_root, "train", args.image_size)
    val_dataset = ArcadeSyntaxDataset(args.dataset_root, "val", args.image_size)
    test_dataset = ArcadeSyntaxDataset(args.dataset_root, "test", args.image_size)

    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    model = UNet(base_channels=args.base_channels).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_dice = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_metrics = run_epoch(model, val_loader, criterion, None, device, False)

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f}, Dice {train_metrics['dice']:.4f}, "
            f"IoU {train_metrics['iou']:.4f} | "
            f"val loss {val_metrics['loss']:.4f}, Dice {val_metrics['dice']:.4f}, "
            f"IoU {val_metrics['iou']:.4f}"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": args.image_size,
                    "base_channels": args.base_channels,
                    "epoch": epoch,
                    "validation_metrics": val_metrics,
                },
                args.output_dir / "best_model.pth",
            )

    # Reload the best validation checkpoint before the final held-out test.
    checkpoint = torch.load(args.output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, criterion, None, device, False)

    with (args.output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    if history:
        with (args.output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    save_prediction_examples(
        model, test_dataset, device, args.output_dir / "prediction_examples"
    )

    print("\nBest checkpoint epoch:", checkpoint["epoch"])
    print(
        "TEST | "
        f"loss {test_metrics['loss']:.4f}, "
        f"Dice {test_metrics['dice']:.4f}, "
        f"IoU {test_metrics['iou']:.4f}"
    )
    print(f"Outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
