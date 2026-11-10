
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

# Allow importing model files from the ml directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from unet_model import UNet
from attention_unet_model import AttentionUNet


DATA_ROOT = ROOT / "data" / "processed" / "stenosis"
MODEL_DIR = ROOT / "outputs" / "models"

OUTPUT_DIR = ROOT / "outputs" / "selected_image_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNET_PATH = MODEL_DIR / "unet_best.pth"
ATTENTION_PATH = MODEL_DIR / "attention_unet_best.pth"

IMAGE_SIZE = 256

# Use the same threshold for both models
THRESHOLD = 0.3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------

def load_model(model_class, checkpoint_path):
    """Load a trained model from its saved checkpoint."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"\nCheckpoint not found:\n{checkpoint_path}\n"
            "Check that the trained model exists in outputs/models."
        )

    model = model_class().to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    # Support plain state_dict and checkpoint dictionaries
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print("Loaded checkpoint:", checkpoint_path.name)

    return model


# --------------------------------------------------
# 3. CALCULATE DICE AND IOU
# --------------------------------------------------

def calculate_metrics(prediction, target):
    """Calculate per-image binary Dice and IoU."""

    prediction = prediction.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(
        prediction, target
    ).sum()

    predicted_pixels = prediction.sum()
    target_pixels = target.sum()

    denominator = predicted_pixels + target_pixels

    if denominator == 0:
        dice = 1.0
    else:
        dice = (
            2.0 * intersection / denominator
        )

    union = np.logical_or(
        prediction, target
    ).sum()

    if union == 0:
        iou = 1.0
    else:
        iou = intersection / union

    return float(dice), float(iou)


# --------------------------------------------------
# 4. CREATE PREDICTION OVERLAY
# --------------------------------------------------

def make_prediction_overlay(image, prediction, color):
    """
    Highlight predicted pixels over the original image.

    image:
        Grayscale image in the range 0-255.

    prediction:
        Binary mask with values 0 or 1.

    color:
        RGB color used for predicted regions.
    """

    base = np.stack(
        [image, image, image],
        axis=-1
    ).astype(np.float32)

    result = base.copy()

    selected = prediction.astype(bool)
    color_array = np.asarray(color, dtype=np.float32)

    result[selected] = (
        0.45 * base[selected]
        + 0.55 * color_array
    )

    return np.clip(result, 0, 255).astype(np.uint8)


# --------------------------------------------------
# 5. CREATE GROUND TRUTH VS PREDICTION COMPARISON
# --------------------------------------------------

def make_match_overlay(image, ground_truth, prediction):
    """
    Colour legend:
        Green  = Ground Truth only (missed by model)
        Red    = Prediction only (extra predicted pixels)
        Yellow = Ground Truth and Prediction overlap
        Dark background = Neither mask contains the pixel
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

    # Blend colours with the underlying original image
    # so the angiography structures remain visible.
    green = np.array([0, 255, 0], dtype=np.float32)
    red = np.array([255, 0, 0], dtype=np.float32)
    yellow = np.array([255, 255, 0], dtype=np.float32)

    result[gt_only] = (
        0.45 * base[gt_only] + 0.55 * green
    )

    result[pred_only] = (
        0.45 * base[pred_only] + 0.55 * red
    )

    result[overlap] = (
        0.35 * base[overlap] + 0.65 * yellow
    )

    return np.clip(result, 0, 255).astype(np.uint8)


# --------------------------------------------------
# 6. SELECT AN IMAGE FROM THE DATASET
# --------------------------------------------------

def select_image():
    print("\nAvailable dataset splits:")
    print("1. train")
    print("2. val")
    print("3. test")

    split = input(
        "\nSelect split (train/val/test): "
    ).strip().lower()

    if split not in ("train", "val", "test"):
        raise ValueError(
            "Invalid split. Please enter train, val or test."
        )

    image_dir = DATA_ROOT / split / "images"
    mask_dir = DATA_ROOT / split / "masks"

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}"
        )

    if not mask_dir.exists():
        raise FileNotFoundError(
            f"Mask directory not found: {mask_dir}"
        )

    image_files = sorted([
        path for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ])

    if not image_files:
        raise RuntimeError(
            f"No supported image files found in {image_dir}"
        )

    print(f"\nAvailable images in {split}: {len(image_files)}")
    print("First 20 image filenames:")

    for path in image_files[:20]:
        print("  ", path.name)

    filename = input(
        "\nEnter image filename (example: 1.png): "
    ).strip()

    # Prevent a typed path from escaping the image directory.
    filename = Path(filename).name

    if not Path(filename).suffix:
        filename += ".png"

    image_path = image_dir / filename
    mask_path = mask_dir / f"{Path(filename).stem}.png"

    if not image_path.exists():
        raise FileNotFoundError(
            f"Selected image does not exist: {image_path}"
        )

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Matching Ground Truth mask does not exist: {mask_path}"
        )

    return split, image_path, mask_path


# --------------------------------------------------
# 7. MAIN PREDICTION AND VISUAL COMPARISON
# --------------------------------------------------

def main():
    print("=" * 65)
    print("SINGLE IMAGE SEGMENTATION COMPARISON")
    print("=" * 65)

    print("Project:", ROOT)
    print("Device:", DEVICE)
    print("Threshold:", THRESHOLD)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Load trained checkpoints
    unet = load_model(
        UNet,
        UNET_PATH
    )

    attention_unet = load_model(
        AttentionUNet,
        ATTENTION_PATH
    )

    # Select image
    split, image_path, mask_path = select_image()

    print("\nSelected split:", split)
    print("Selected image:", image_path.name)

    # Read the original grayscale image and its Ground Truth mask
    original = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    ground_truth = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if original is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    if ground_truth is None:
        raise RuntimeError(
            f"Could not read Ground Truth mask: {mask_path}"
        )

    original_height, original_width = original.shape

    # Ensure the Ground Truth mask matches original image dimensions
    if ground_truth.shape != original.shape:
        ground_truth = cv2.resize(
            ground_truth,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST
        )

    ground_truth_binary = (
        ground_truth > 127
    ).astype(np.uint8)

    # Apply the same preprocessing used during training
    resized = cv2.resize(
        original,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    normalized = resized.astype(np.float32) / 255.0

    image_tensor = torch.from_numpy(
        normalized
    ).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Run both models
    with torch.no_grad():
        unet_probabilities = torch.sigmoid(
            unet(image_tensor)
        )[0, 0]

        attention_probabilities = torch.sigmoid(
            attention_unet(image_tensor)
        )[0, 0]

    # Apply a common threshold
    unet_small = (
        unet_probabilities.cpu().numpy() >= THRESHOLD
    ).astype(np.uint8)

    attention_small = (
        attention_probabilities.cpu().numpy() >= THRESHOLD
    ).astype(np.uint8)

    # Restore prediction masks to original image dimensions
    unet_prediction = cv2.resize(
        unet_small,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    )

    attention_prediction = cv2.resize(
        attention_small,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    )

    # Compute per-image scores against the matching Ground Truth
    unet_dice, unet_iou = calculate_metrics(
        unet_prediction,
        ground_truth_binary
    )

    attention_dice, attention_iou = calculate_metrics(
        attention_prediction,
        ground_truth_binary
    )

    # Create overlays
    unet_overlay = make_prediction_overlay(
        original,
        unet_prediction,
        (255, 0, 0)
    )

    attention_overlay = make_prediction_overlay(
        original,
        attention_prediction,
        (0, 100, 255)
    )

    # Create direct Ground Truth vs Prediction comparisons
    unet_match = make_match_overlay(
        original,
        ground_truth_binary,
        unet_prediction
    )

    attention_match = make_match_overlay(
        original,
        ground_truth_binary,
        attention_prediction
    )

    # --------------------------------------------------
    # 8. PRINT RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("SELECTED IMAGE RESULTS")
    print("=" * 65)

    print("Image:", image_path.name)
    print("Threshold:", THRESHOLD)

    print(
        f"U-Net       | Dice: {unet_dice:.4f} "
        f"({unet_dice * 100:.2f}%) "
        f"| IoU: {unet_iou:.4f} ({unet_iou * 100:.2f}%)"
    )

    print(
        f"Attention U-Net | Dice: {attention_dice:.4f} "
        f"({attention_dice * 100:.2f}%) "
        f"| IoU: {attention_iou:.4f} "
        f"({attention_iou * 100:.2f}%)"
    )

    # --------------------------------------------------
    # 9. DISPLAY ALL COMPARISONS
    # --------------------------------------------------

    fig, axes = plt.subplots(
        2, 4,
        figsize=(19, 10)
    )

    # Row 1: original, Ground Truth, predictions
    axes[0, 0].imshow(
        original,
        cmap="gray"
    )
    axes[0, 0].set_title(
        f"Original Image: {image_path.name}"
    )

    axes[0, 1].imshow(
        ground_truth_binary,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[0, 1].set_title(
        "Ground Truth Mask"
    )

    axes[0, 2].imshow(
        unet_prediction,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[0, 2].set_title(
        f"U-Net Prediction\n"
        f"Dice={unet_dice:.3f} | IoU={unet_iou:.3f}"
    )

    axes[0, 3].imshow(
        attention_prediction,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[0, 3].set_title(
        f"Attention U-Net Prediction\n"
        f"Dice={attention_dice:.3f} | IoU={attention_iou:.3f}"
    )

    # Row 2: overlays and matching comparisons
    axes[1, 0].imshow(unet_overlay)
    axes[1, 0].set_title(
        "U-Net Overlay (red prediction)"
    )

    axes[1, 1].imshow(attention_overlay)
    axes[1, 1].set_title(
        "Attention U-Net Overlay (blue prediction)"
    )

    axes[1, 2].imshow(unet_match)
    axes[1, 2].set_title(
        "U-Net vs Ground Truth\n"
        "Green=missed | Red=extra | Yellow=overlap"
    )

    axes[1, 3].imshow(attention_match)
    axes[1, 3].set_title(
        "Attention U-Net vs Ground Truth\n"
        "Green=missed | Red=extra | Yellow=overlap"
    )

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        f"Stenosis Segmentation Comparison: {image_path.name} "
        f"(threshold={THRESHOLD})",
        fontsize=14
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = (
        OUTPUT_DIR / f"{split}_{image_path.stem}_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print("\nComparison image saved to:")
    print(output_path)

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
