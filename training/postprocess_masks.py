
from pathlib import Path
import cv2
import numpy as np

# Project output directory
OUTPUT_DIR = Path("training/outputs")

# Existing model prediction folders
MODEL_FOLDERS = {
    "baseline_unet": OUTPUT_DIR / "unet_baseline_5epochs" / "prediction_examples",
    "attention_unet": OUTPUT_DIR / "attention_unet_5epochs" / "prediction_examples",
}

# Save processed masks separately
POSTPROCESS_DIR = OUTPUT_DIR / "postprocessed_masks"

# Initial experimental settings
KERNEL_SIZE = 3
MIN_COMPONENT_SIZE = 20


def remove_small_components(mask, min_size):
    """Remove very small isolated foreground regions."""

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    cleaned = np.zeros_like(mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_size:
            cleaned[labels == label] = 255

    return cleaned


def postprocess_mask(mask_path, output_path):
    """Apply morphological closing and remove small components."""

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Could not read: {mask_path}")
        return False

    # Convert prediction into a binary mask
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Morphological closing helps fill small gaps
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)

    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    # Remove tiny isolated regions
    cleaned = remove_small_components(
        closed,
        MIN_COMPONENT_SIZE
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), cleaned)

    return True


def main():
    total_processed = 0

    for model_name, prediction_folder in MODEL_FOLDERS.items():

        print(f"\nProcessing: {model_name}")

        if not prediction_folder.exists():
            print(f"Prediction folder not found: {prediction_folder}")
            continue

        prediction_files = sorted(
            prediction_folder.glob("prediction_*.png")
        )

        if not prediction_files:
            print("No prediction PNG files found.")
            continue

        model_output = POSTPROCESS_DIR / model_name

        for prediction_path in prediction_files:

            output_path = model_output / prediction_path.name

            success = postprocess_mask(
                prediction_path,
                output_path
            )

            if success:
                total_processed += 1
                print(f"Processed: {prediction_path.name}")

    print("\n--------------------------------")
    print(f"Total masks processed: {total_processed}")
    print(f"Saved results in: {POSTPROCESS_DIR}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
