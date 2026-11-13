
from pathlib import Path
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path("backend/app/data/arcade/arcade/arcade")
MASK_ROOT = Path("backend/results/arcade_masks")

ANNOTATION_FILE = ROOT / "stenosis/val/annotations/val.json"
IMAGE_DIR = ROOT / "stenosis/val/images"
MASK_DIR = MASK_ROOT / "stenosis/val"

MODEL_PATH = Path("backend/results/unet_stenosis_v4.pth")
OUTPUT_DIR = Path("backend/results/stenosis_predictions")

DEVICE = torch.device("cpu")
BATCH_SIZE = 2

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


class StenosisDataset(Dataset):
    def __init__(self):
        with ANNOTATION_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)

        categories = {
            c["id"]: c["name"] for c in data["categories"]
        }

        stenosis_ids = {
            cid for cid, name in categories.items()
            if name.lower() == "stenosis"
        }

        annotated_ids = {
            a["image_id"] for a in data["annotations"]
            if a["category_id"] in stenosis_ids
        }

        self.paths = []

        for info in data["images"]:
            if info["id"] not in annotated_ids:
                continue

            image_path = IMAGE_DIR / info["file_name"]
            mask_path = (
                MASK_DIR /
                f"{Path(info['file_name']).stem}_mask.png"
            )

            if image_path.exists() and mask_path.exists():
                self.paths.append((image_path, mask_path))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_path, mask_path = self.paths[index]

        image = cv2.imread(
            str(image_path), cv2.IMREAD_GRAYSCALE
        )
        mask = cv2.imread(
            str(mask_path), cv2.IMREAD_GRAYSCALE
        )

        if image is None or mask is None:
            raise RuntimeError(
                f"Could not read {image_path.name}"
            )

        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        return (
            torch.from_numpy(image).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
            image_path.name,
        )


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


def calculate_metrics(pred, target):
    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()

    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return [dice, iou, precision, recall]


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    dataset = StenosisDataset()

    if len(dataset) == 0:
        raise RuntimeError(
            "No annotated stenosis validation pairs found."
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = UNet().to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    )

    model.eval()

    # Store metric totals for every threshold.
    metric_sums = {
        threshold: np.zeros(4, dtype=np.float64)
        for threshold in THRESHOLDS
    }

    total = 0

    print("Evaluating stenosis model...")
    print("Validation images:", len(dataset))

    # Evaluate all thresholds.
    with torch.no_grad():
        for images, masks, filenames in loader:
            probabilities = torch.sigmoid(
                model(images.to(DEVICE))
            )

            targets = masks.to(DEVICE) >= 0.5

            for threshold in THRESHOLDS:
                predictions = probabilities >= threshold

                for i in range(len(filenames)):
                    metrics = calculate_metrics(
                        predictions[i, 0],
                        targets[i, 0],
                    )

                    metric_sums[threshold] += metrics

            total += len(filenames)

    # Display results.
    print("\nTHRESHOLD COMPARISON")
    print("-" * 78)
    print(
        f"{'Threshold':<12}"
        f"{'Dice':<12}"
        f"{'IoU':<12}"
        f"{'Precision':<15}"
        f"{'Recall':<12}"
    )
    print("-" * 78)

    best_threshold = None
    best_dice = -1

    for threshold in THRESHOLDS:
        means = metric_sums[threshold] / total

        print(
            f"{threshold:<12.2f}"
            f"{means[0]:<12.4f}"
            f"{means[1]:<12.4f}"
            f"{means[2]:<15.4f}"
            f"{means[3]:<12.4f}"
        )

        if means[0] > best_dice:
            best_dice = means[0]
            best_threshold = threshold

    print("-" * 78)

    print("\nBest threshold:", best_threshold)
    print("Best validation Dice:", round(best_dice, 4))

    # Save predictions using the best validation threshold.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for images, masks, filenames in loader:
            probabilities = torch.sigmoid(
                model(images.to(DEVICE))
            )

            predictions = probabilities >= best_threshold

            for i, filename in enumerate(filenames):
                output = (
                    predictions[i, 0]
                    .cpu()
                    .numpy()
                    .astype(np.uint8) * 255
                )

                output_path = (
                    OUTPUT_DIR /
                    f"{Path(filename).stem}_pred.png"
                )

                cv2.imwrite(str(output_path), output)

    print("\nPredictions saved to:", OUTPUT_DIR.resolve())
    print("Threshold evaluation completed.")


if __name__ == "__main__":
    main()
