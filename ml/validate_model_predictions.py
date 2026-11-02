
from pathlib import Path
import sys
import csv
import random

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unet_model import UNet
from attention_unet_model import AttentionUNet


DATA_ROOT = ROOT / "data" / "processed" / "stenosis" / "val"
IMAGE_DIR = DATA_ROOT / "images"
MASK_DIR = DATA_ROOT / "masks"

MODEL_DIR = ROOT / "outputs" / "models"
OUTPUT_DIR = ROOT / "outputs" / "validation_review"
VISUAL_DIR = OUTPUT_DIR / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 256
THRESHOLD = 0.3
NUM_IMAGES = 200
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# 2. LOAD CHECKPOINTS
# --------------------------------------------------

def load_model(model_class, checkpoint_path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = model_class().to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model


# --------------------------------------------------
# 3. METRICS
# --------------------------------------------------

def calculate_metrics(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(prediction, target).sum()

    pred_count = prediction.sum()
    target_count = target.sum()

    denominator = pred_count + target_count

    dice = (
        1.0 if denominator == 0
        else (2.0 * intersection / denominator)
    )

    union = np.logical_or(prediction, target).sum()

    iou = (
        1.0 if union == 0
        else intersection / union
    )

    precision = (
        1.0 if pred_count == 0 and target_count == 0
        else 0.0 if pred_count == 0
        else intersection / pred_count
    )

    recall = (
        1.0 if target_count == 0 and pred_count == 0
        else 0.0 if target_count == 0
        else intersection / target_count
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "gt_pixels": int(target_count),
        "pred_pixels": int(pred_count),
    }


# --------------------------------------------------
# 4. GROUND TRUTH / PREDICTION OVERLAY
# --------------------------------------------------

def make_match_overlay(image, ground_truth, prediction):
    """
    Green  = Ground Truth only (missed region)
    Red    = Prediction only (extra region)
    Yellow = Ground Truth and prediction overlap
    """

    base = np.stack(
        [image, image, image],
        axis=-1
    ).astype(np.float32)

    result = base.copy()

    gt = ground_truth.astype(bool)
    pred = prediction.astype(bool)

    gt_only = gt & ~pred
    pred_only = pred & ~gt
    overlap = gt & pred

    result[gt_only] = (
        0.45 * base[gt_only]
        + 0.55 * np.array([0, 255, 0])
    )

    result[pred_only] = (
        0.45 * base[pred_only]
        + 0.55 * np.array([255, 0, 0])
    )

    result[overlap] = (
        0.35 * base[overlap]
        + 0.65 * np.array([255, 255, 0])
    )

    return np.clip(result, 0, 255).astype(np.uint8)


# --------------------------------------------------
# 5. MAIN VALIDATION REVIEW
# --------------------------------------------------

def main():
    print("=" * 65)
    print("VALIDATION IMAGE REVIEW")
    print("=" * 65)
    print("Device:", DEVICE)
    print("Threshold:", THRESHOLD)

    if not IMAGE_DIR.exists() or not MASK_DIR.exists():
        raise FileNotFoundError(
            f"Validation folders not found:\n"
            f"{IMAGE_DIR}\n{MASK_DIR}"
        )

    images = sorted(IMAGE_DIR.glob("*.png"))

    pairs = [
        (image_path, MASK_DIR / image_path.name)
        for image_path in images
        if (MASK_DIR / image_path.name).exists()
    ]

    if not pairs:
        raise RuntimeError("No validation image-mask pairs found.")

    rng = random.Random(SEED)
    chosen = rng.sample(pairs, min(NUM_IMAGES, len(pairs)))

    print("Available validation pairs:", len(pairs))
    print("Selected images:", len(chosen))

    unet = load_model(
        UNet,
        MODEL_DIR / "unet_best.pth"
    )

    attention = load_model(
        AttentionUNet,
        MODEL_DIR / "attention_unet_best.pth"
    )

    rows = []

    for index, (image_path, mask_path) in enumerate(chosen, start=1):
        original = cv2.imread(
            str(image_path), cv2.IMREAD_GRAYSCALE
        )

        gt_original = cv2.imread(
            str(mask_path), cv2.IMREAD_GRAYSCALE
        )

        if original is None or gt_original is None:
            print("Skipping unreadable pair:", image_path.name)
            continue

        # Preprocess as during model training
        resized = cv2.resize(
            original,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        gt_small = cv2.resize(
            gt_original,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        gt_small = (gt_small > 127).astype(np.uint8)

        tensor = torch.from_numpy(
            resized.astype(np.float32) / 255.0
        ).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            unet_prob = torch.sigmoid(unet(tensor))[0, 0]
            attention_prob = torch.sigmoid(attention(tensor))[0, 0]

        unet_small = (
            unet_prob.cpu().numpy() >= THRESHOLD
        ).astype(np.uint8)

        attention_small = (
            attention_prob.cpu().numpy() >= THRESHOLD
        ).astype(np.uint8)

        # Calculate all metrics at the same 256x256 resolution
        unet_metrics = calculate_metrics(
            unet_small, gt_small
        )

        attention_metrics = calculate_metrics(
            attention_small, gt_small
        )

        rows.append({
            "image": image_path.name,
            "threshold": THRESHOLD,
            "unet_dice": unet_metrics["dice"],
            "unet_iou": unet_metrics["iou"],
            "unet_precision": unet_metrics["precision"],
            "unet_recall": unet_metrics["recall"],
            "unet_pred_pixels": unet_metrics["pred_pixels"],
            "gt_pixels": unet_metrics["gt_pixels"],
            "attention_dice": attention_metrics["dice"],
            "attention_iou": attention_metrics["iou"],
            "attention_precision": attention_metrics["precision"],
            "attention_recall": attention_metrics["recall"],
            "attention_pred_pixels": attention_metrics["pred_pixels"],
        })

        # Return masks to original dimensions for visual inspection
        h, w = original.shape

        gt_display = cv2.resize(
            gt_small, (w, h), interpolation=cv2.INTER_NEAREST
        )

        unet_display = cv2.resize(
            unet_small, (w, h), interpolation=cv2.INTER_NEAREST
        )

        attention_display = cv2.resize(
            attention_small, (w, h), interpolation=cv2.INTER_NEAREST
        )

        unet_match = make_match_overlay(
            original, gt_display, unet_display
        )

        attention_match = make_match_overlay(
            original, gt_display, attention_display
        )

        # Save one comparison image per selected validation image
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        axes[0, 0].imshow(original, cmap="gray")
        axes[0, 0].set_title(f"Original: {image_path.name}")

        axes[0, 1].imshow(gt_display, cmap="gray")
        axes[0, 1].set_title("Ground Truth Mask")

        axes[0, 2].imshow(unet_display, cmap="gray")
        axes[0, 2].set_title(
            f"U-Net | Dice {unet_metrics['dice']:.3f} "
            f"| IoU {unet_metrics['iou']:.3f}"
        )

        axes[1, 0].imshow(attention_display, cmap="gray")
        axes[1, 0].set_title(
            f"Attention U-Net | Dice {attention_metrics['dice']:.3f} "
            f"| IoU {attention_metrics['iou']:.3f}"
        )

        axes[1, 1].imshow(unet_match)
        axes[1, 1].set_title("U-Net match: Green=missed, Red=extra, Yellow=overlap")

        axes[1, 2].imshow(attention_match)
        axes[1, 2].set_title(
            "Attention U-Net match: Green=missed, Red=extra, Yellow=overlap"
        )

        for ax in axes.flat:
            ax.axis("off")

        fig.suptitle(
            f"Validation Review | {image_path.name} | Threshold={THRESHOLD}"
        )

        fig.tight_layout()

        fig.savefig(
            VISUAL_DIR / f"{image_path.stem}_comparison.png",
            dpi=130,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            f"{index}/{len(chosen)}: {image_path.name} | "
            f"U-Net Dice={unet_metrics['dice']:.3f}, "
            f"Attention Dice={attention_metrics['dice']:.3f}"
        )

    # Save per-image validation metrics
    csv_path = OUTPUT_DIR / "validation_metrics.csv"

    if rows:
        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(rows)

    print("\n" + "=" * 65)
    print("VALIDATION REVIEW COMPLETE")
    print("=" * 65)
    print("Images analysed:", len(rows))
    print("Metrics CSV:", csv_path)
    print("Visual comparisons:", VISUAL_DIR)

    if rows:
        print(
            "Mean U-Net Dice:",
            round(np.mean([r["unet_dice"] for r in rows]), 4)
        )
        print(
            "Mean Attention U-Net Dice:",
            round(np.mean([r["attention_dice"] for r in rows]), 4)
        )


if __name__ == "__main__":
    main()
