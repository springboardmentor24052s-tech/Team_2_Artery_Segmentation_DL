
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "outputs"
    / "artery_masks_stenosis_test"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "outputs"
    / "stenosis_postprocessed_masks"
)

KERNEL_SIZE = 3
MIN_COMPONENT_SIZE = 20


def remove_small_components(mask):
    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= MIN_COMPONENT_SIZE:
            cleaned[labels == label] = 255

    return cleaned


def process_mask(mask_path):
    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        print(f"Could not read: {mask_path}")
        return False

    _, binary = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones(
        (KERNEL_SIZE, KERNEL_SIZE),
        np.uint8
    )

    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    cleaned = remove_small_components(closed)

    output_path = (
        OUTPUT_DIR
        / f"prediction_{mask_path.stem.replace('_mask', '')}.png"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(str(output_path), cleaned)

    original_pixels = cv2.countNonZero(binary)
    processed_pixels = cv2.countNonZero(cleaned)

    print(
        f"{mask_path.name}: "
        f"Original pixels = {original_pixels}, "
        f"Processed pixels = {processed_pixels}"
    )

    return True


def main():
    if not INPUT_DIR.exists():
        print(f"Input folder not found: {INPUT_DIR}")
        return

    mask_files = sorted(
        INPUT_DIR.glob("*_mask.png")
    )

    if not mask_files:
        print("No segmentation masks found.")
        return

    total = 0

    for mask_path in mask_files:
        if process_mask(mask_path):
            total += 1

    print("\n------------------------------")
    print(f"Masks processed: {total}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("------------------------------")


if __name__ == "__main__":
    main()
