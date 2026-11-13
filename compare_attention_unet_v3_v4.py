
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from attention_unet import AttentionUNet


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DEVICE = torch.device("cpu")
BATCH_SIZE = 2
THRESHOLD = 0.20
IMAGE_SIZE = 512

IMAGE_DIR = Path(
    "backend/app/data/arcade/arcade/arcade/stenosis/test/images"
)

MASK_DIR = Path(
    "backend/results/arcade_masks/stenosis/test"
)

CHECKPOINTS = {
    "V3": Path("backend/results/attention_unet_stenosis_v3.pth"),
    "V4": Path("backend/results/attention_unet_stenosis_v4.pth"),
}

OUTPUT_DIR = Path("backend/results/attention_unet_v3_v4_test_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# DATASET
# --------------------------------------------------

class StenosisTestDataset(Dataset):

    def __init__(self):
        self.samples = []

        image_paths = sorted(
            IMAGE_DIR.glob("*.png"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
        )

        for image_path in image_paths:
            mask_path = MASK_DIR / f"{image_path.stem}_mask.png"

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Missing mask for {image_path.name}: {mask_path}"
                )

            self.samples.append((image_path, mask_path))

        if len(self.samples) != 300:
            raise ValueError(
                f"Expected 300 test images, found {len(self.samples)}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise ValueError(f"Could not read {image_path.name}")

        image = cv2.resize(
            image, (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        mask = cv2.resize(
            mask, (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return image_tensor, mask_tensor, image_path.name


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(predictions, targets):

    predictions = predictions.astype(bool)
    targets = targets.astype(bool)

    tp = np.logical_and(predictions, targets).sum()
    fp = np.logical_and(predictions, ~targets).sum()
    fn = np.logical_and(~predictions, targets).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

def evaluate_model(model_name, checkpoint_path, loader):

    print(f"\nEvaluating Attention U-Net {model_name}...")

    model = AttentionUNet().to(DEVICE)

    state = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state, strict=True)
    model.eval()

    per_image_metrics = []

    with torch.no_grad():

        for batch_index, (images, masks, filenames) in enumerate(loader):

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities >= THRESHOLD).float()

            for i in range(images.size(0)):

                metrics = calculate_metrics(
                    predictions[i, 0].cpu().numpy(),
                    masks[i, 0].cpu().numpy()
                )

                metrics["filename"] = filenames[i]
                per_image_metrics.append(metrics)

            if (batch_index + 1) % 25 == 0:
                print(
                    f"{model_name}: "
                    f"{min((batch_index + 1) * BATCH_SIZE, len(loader.dataset))}"
                    f"/{len(loader.dataset)} images processed"
                )

    metric_names = ["dice", "iou", "precision", "recall"]

    averages = {
        metric: float(np.mean([
            item[metric] for item in per_image_metrics
        ]))
        for metric in metric_names
    }

    print(f"\n{model_name} Results:")
    for metric, value in averages.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    return {
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "threshold": THRESHOLD,
        "images_evaluated": len(per_image_metrics),
        "aggregation": "Mean of per-image metrics",
        "metrics": averages,
        "per_image_metrics": per_image_metrics,
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("Attention U-Net V3 vs V4 Test Evaluation")
    print("Device:", DEVICE)
    print("Threshold:", THRESHOLD)

    dataset = StenosisTestDataset()

    print("Test images and masks:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    results = {}

    for model_name, checkpoint_path in CHECKPOINTS.items():

        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        results[model_name] = evaluate_model(
            model_name,
            checkpoint_path,
            loader
        )

    comparison = {
        "dataset": "ARCADE stenosis test set",
        "test_images": len(dataset),
        "threshold": THRESHOLD,
        "device": str(DEVICE),
        "results": results,
    }

    report_path = OUTPUT_DIR / "v3_v4_test_comparison.json"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=4)

    print("\n" + "=" * 50)
    print("FINAL MODEL COMPARISON")
    print("=" * 50)

    for model_name in ["V3", "V4"]:
        print(f"\n{model_name}")
        for metric, value in results[model_name]["metrics"].items():
            print(f"{metric.capitalize()}: {value:.4f}")

    print("\nComparison report saved to:")
    print(report_path)


if __name__ == "__main__":
    main()
