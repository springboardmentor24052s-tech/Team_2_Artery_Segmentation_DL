
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from attention_unet import AttentionUNet


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DEVICE = torch.device("cpu")
THRESHOLD = 0.20
IMAGE_SIZE = 512

IMAGE_DIR = Path(
    "backend/app/data/arcade/arcade/arcade/stenosis/test/images"
)

MASK_DIR = Path(
    "backend/results/arcade_masks/stenosis/test"
)

REPORT_PATH = Path(
    "backend/results/attention_unet_v3_v4_test_comparison/"
    "v3_v4_test_comparison.json"
)

CHECKPOINTS = {
    "V3": Path("backend/results/attention_unet_stenosis_v3.pth"),
    "V4": Path("backend/results/attention_unet_stenosis_v4.pth"),
}

OUTPUT_DIR = Path(
    "backend/results/attention_unet_v3_v4_visualizations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

def load_model(checkpoint_path):

    model = AttentionUNet().to(DEVICE)

    state = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state, strict=True)
    model.eval()

    return model


# --------------------------------------------------
# LOAD IMAGE AND MASK
# --------------------------------------------------

def load_image_and_mask(filename):

    image_path = IMAGE_DIR / filename
    mask_path = MASK_DIR / f"{Path(filename).stem}_mask.png"

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or mask is None:
        raise ValueError(f"Unable to read {filename}")

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    mask = cv2.resize(
        mask,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    image_tensor = (
        torch.from_numpy(
            image.astype(np.float32) / 255.0
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    return image, mask > 127, image_tensor


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict(model, image_tensor):

    with torch.no_grad():

        output = model(image_tensor)

        probability = torch.sigmoid(output)

        prediction = probability >= THRESHOLD

    return prediction[0, 0].cpu().numpy()


# --------------------------------------------------
# CALCULATE INDIVIDUAL DICE
# --------------------------------------------------

def calculate_dice(prediction, ground_truth):

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    total = prediction.sum() + ground_truth.sum()

    if total == 0:
        return 1.0

    return float(
        (2.0 * intersection) / total
    )


# --------------------------------------------------
# CREATE OVERLAY
# --------------------------------------------------

def create_overlay(image, ground_truth, prediction):

    base = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR
    )

    gt_only = ground_truth & ~prediction
    pred_only = prediction & ~ground_truth
    overlap = ground_truth & prediction

    overlay = base.copy()

    # Green: missed ground-truth pixels
    overlay[gt_only] = (0, 255, 0)

    # Red: false-positive prediction pixels
    overlay[pred_only] = (0, 0, 255)

    # Yellow: correct overlap
    overlay[overlap] = (0, 255, 255)

    return cv2.addWeighted(
        base,
        0.65,
        overlay,
        0.35,
        0
    )


# --------------------------------------------------
# ADD PANEL LABEL
# --------------------------------------------------

def add_label(image, label):

    result = image.copy()

    cv2.rectangle(
        result,
        (0, 0),
        (result.shape[1], 35),
        (25, 25, 25),
        -1
    )

    cv2.putText(
        result,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    return result


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    if not REPORT_PATH.exists():
        raise FileNotFoundError(REPORT_PATH)

    with REPORT_PATH.open(
        "r",
        encoding="utf-8"
    ) as f:
        report = json.load(f)

    # Map each filename to its own model metrics.
    v3_metrics = {
        item["filename"]: item
        for item in report["results"]["V3"]["per_image_metrics"]
    }

    v4_metrics = {
        item["filename"]: item
        for item in report["results"]["V4"]["per_image_metrics"]
    }

    if set(v3_metrics) != set(v4_metrics):
        raise ValueError(
            "V3 and V4 evaluation filenames do not match."
        )

    # Select worst, median and best examples based on V4 Dice.
    ranked = sorted(
        v4_metrics.values(),
        key=lambda x: x["dice"]
    )

    selected = [
        ("Worst V4 Dice", ranked[0]),
        ("Median V4 Dice", ranked[len(ranked) // 2]),
        ("Best V4 Dice", ranked[-1]),
    ]

    print("\nLoading V3 and V4 models...")

    models = {
        name: load_model(path)
        for name, path in CHECKPOINTS.items()
    }

    # Generate comparisons.
    for description, v4_item in selected:

        filename = v4_item["filename"]

        v3_item = v3_metrics[filename]

        image, ground_truth, image_tensor = (
            load_image_and_mask(filename)
        )

        pred_v3 = predict(
            models["V3"],
            image_tensor
        )

        pred_v4 = predict(
            models["V4"],
            image_tensor
        )

        # Calculate actual Dice separately.
        v3_dice = calculate_dice(
            pred_v3,
            ground_truth
        )

        v4_dice = calculate_dice(
            pred_v4,
            ground_truth
        )

        # Ground truth panel.
        ground_truth_panel = np.repeat(
            (
                ground_truth.astype(np.uint8) * 255
            )[..., None],
            3,
            axis=2
        )

        panels = [

            add_label(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_GRAY2BGR
                ),
                f"Original: {filename}"
            ),

            add_label(
                ground_truth_panel,
                "Ground Truth"
            ),

            add_label(
                create_overlay(
                    image,
                    ground_truth,
                    pred_v3
                ),
                f"V3 Dice: {v3_dice:.4f}"
            ),

            add_label(
                create_overlay(
                    image,
                    ground_truth,
                    pred_v4
                ),
                f"V4 Dice: {v4_dice:.4f}"
            ),
        ]

        comparison = np.hstack(panels)

        safe_name = Path(filename).stem

        output_path = OUTPUT_DIR / (
            f"{safe_name}_"
            f"{description.lower().replace(' ', '_')}.png"
        )

        cv2.imwrite(
            str(output_path),
            comparison
        )

        print("\n" + "-" * 45)
        print("Image:", filename)
        print(f"V3 Dice: {v3_dice:.4f}")
        print(f"V4 Dice: {v4_dice:.4f}")
        print("Saved:", output_path)

    print("\n" + "=" * 45)
    print("CORRECTED VISUAL COMPARISON COMPLETED")
    print("=" * 45)

    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
