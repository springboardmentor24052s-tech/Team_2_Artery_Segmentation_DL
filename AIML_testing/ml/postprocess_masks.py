
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

IMAGE_ROOT = ROOT / "Dataset" / "arcade" / "arcade" / "syntax"
MASK_ROOT = ROOT / "data" / "processed" / "syntax"

OUTPUT_ROOT = ROOT / "data" / "processed" / "syntax_postprocessed"
PREVIEW_ROOT = ROOT / "outputs" / "postprocess_previews"

SPLITS = ["train", "val", "test"]

# Keep this low to avoid deleting thin vessel branches.
MIN_COMPONENT_AREA = 10

# Small closing operation to connect tiny gaps and fill small holes.
KERNEL_SIZE = 3


# --------------------------------------------------
# 2. POST-PROCESSING FUNCTION
# --------------------------------------------------

def postprocess_mask(mask):
    """
    Clean a binary vessel mask.

    Input:
        Grayscale mask with vessel pixels represented by 255.

    Output:
        Binary mask with background=0 and vessel=255.
    """

    # Ensure the mask is binary.
    binary = (mask > 127).astype(np.uint8) * 255

    # Close tiny gaps without using erosion/opening,
    # which can remove thin vessel branches.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (KERNEL_SIZE, KERNEL_SIZE)
    )

    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    # Remove only very small connected components.
    number, labels, stats, _ = cv2.connectedComponentsWithStats(
        (cleaned > 0).astype(np.uint8),
        connectivity=8
    )

    result = np.zeros_like(cleaned)
    removed_components = 0

    for component_id in range(1, number):
        area = stats[component_id, cv2.CC_STAT_AREA]

        if area >= MIN_COMPONENT_AREA:
            result[labels == component_id] = 255
        else:
            removed_components += 1

    return result, removed_components


# --------------------------------------------------
# 3. PROCESS ONE DATASET SPLIT
# --------------------------------------------------

def process_split(split):
    input_dir = MASK_ROOT / split / "masks"
    image_dir = IMAGE_ROOT / split / "images"

    output_dir = OUTPUT_ROOT / split / "masks"
    preview_dir = PREVIEW_ROOT / split

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[SKIP] Mask directory missing: {input_dir}")
        return

    mask_paths = sorted(input_dir.glob("*.png"))

    if not mask_paths:
        print(f"[SKIP] No masks found in: {input_dir}")
        return

    processed = 0
    nonempty_before = 0
    nonempty_after = 0
    removed_total = 0
    preview_count = 0

    print(f"\n--- Processing {split.upper()} ---")
    print("Input masks:", len(mask_paths))

    for mask_path in mask_paths:
        original_mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if original_mask is None:
            print("[WARN] Could not read:", mask_path.name)
            continue

        if np.any(original_mask > 127):
            nonempty_before += 1

        cleaned_mask, removed = postprocess_mask(original_mask)

        output_path = output_dir / mask_path.name

        if not cv2.imwrite(str(output_path), cleaned_mask):
            raise RuntimeError(
                f"Could not save processed mask: {output_path}"
            )

        if np.any(cleaned_mask > 127):
            nonempty_after += 1

        removed_total += removed
        processed += 1

        # Save up to five overlays for visual inspection.
        if preview_count < 5 and np.any(cleaned_mask > 0):
            image_path = image_dir / mask_path.name

            if image_path.exists():
                image = cv2.imread(
                    str(image_path),
                    cv2.IMREAD_GRAYSCALE
                )

                if image is not None:
                    image = cv2.resize(
                        image,
                        (cleaned_mask.shape[1], cleaned_mask.shape[0])
                    )

                    original_color = cv2.cvtColor(
                        image,
                        cv2.COLOR_GRAY2BGR
                    )

                    overlay = original_color.copy()
                    overlay[cleaned_mask > 0] = (0, 255, 0)

                    blended = cv2.addWeighted(
                        original_color,
                        0.7,
                        overlay,
                        0.3,
                        0
                    )

                    preview_path = (
                        preview_dir /
                        f"{mask_path.stem}_postprocessed.png"
                    )

                    cv2.imwrite(str(preview_path), blended)
                    preview_count += 1

        if processed % 100 == 0:
            print(f"Processed {processed}/{len(mask_paths)}")

    print("\nSummary:", split)
    print("Processed masks:", processed)
    print("Non-empty before:", nonempty_before)
    print("Non-empty after:", nonempty_after)
    print("Small components removed:", removed_total)
    print("Cleaned masks:", output_dir)
    print("Preview overlays:", preview_dir)


# --------------------------------------------------
# 4. MAIN
# --------------------------------------------------

def main():
    print("Project root:", ROOT)
    print("Input masks:", MASK_ROOT)
    print("Output masks:", OUTPUT_ROOT)

    for split in SPLITS:
        process_split(split)

    print("\nPost-processing completed.")
    print("Visually inspect the previews before continuing.")


if __name__ == "__main__":
    main()
