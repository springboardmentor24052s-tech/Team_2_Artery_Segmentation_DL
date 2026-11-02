
from pathlib import Path
import csv

import cv2
import numpy as np
from skimage.morphology import skeletonize


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

IMAGE_ROOT = ROOT / "Dataset" / "arcade" / "arcade" / "syntax"

MASK_ROOT = (
    ROOT / "data" / "processed" / "syntax_postprocessed"
)

OUTPUT_ROOT = ROOT / "data" / "processed" / "syntax_centerlines"

PREVIEW_ROOT = ROOT / "outputs" / "centerline_previews"

REPORT_ROOT = ROOT / "outputs" / "centerline_reports"

SPLITS = ["train", "val", "test"]


# --------------------------------------------------
# 2. GENERATE CENTERLINE
# --------------------------------------------------

def extract_centerline(mask):
    """
    Convert a binary vessel mask into a one-pixel-wide skeleton.

    Input:
        Grayscale binary mask (background=0, vessel=255).

    Returns:
        Skeleton mask (background=0, centerline=255).
    """

    binary = mask > 127

    # Zhang-Suen-style thinning through scikit-image.
    skeleton = skeletonize(binary)

    return skeleton.astype(np.uint8) * 255


# --------------------------------------------------
# 3. COUNT ENDPOINTS AND JUNCTION PIXELS
# --------------------------------------------------

def analyze_skeleton(skeleton):
    """
    Calculate simple 8-neighbour topology indicators.

    These are approximate pixel-level indicators, not a
    complete anatomical branch analysis.
    """

    binary = (skeleton > 0).astype(np.uint8)

    # Count all skeleton pixels in each 3x3 neighbourhood.
    kernel = np.ones((3, 3), dtype=np.uint8)

    neighbour_counts = cv2.filter2D(
        binary,
        cv2.CV_16U,
        kernel,
        borderType=cv2.BORDER_CONSTANT
    )

    neighbour_counts = (
        neighbour_counts.astype(np.int16)
        - binary.astype(np.int16)
    )

    skeleton_pixels = int(binary.sum())

    # Approximate endpoint pixels: exactly one neighbour.
    endpoints = int(
        np.logical_and(
            binary == 1,
            neighbour_counts == 1
        ).sum()
    )

    # Junction pixels may form clusters around a branch point.
    junction_pixels = int(
        np.logical_and(
            binary == 1,
            neighbour_counts >= 3
        ).sum()
    )

    return skeleton_pixels, endpoints, junction_pixels


# --------------------------------------------------
# 4. CREATE RED OVERLAY
# --------------------------------------------------

def create_overlay(image, skeleton):
    """Draw the skeleton in red over the original image."""

    original = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR
    )

    overlay = original.copy()

    # Red in OpenCV's BGR colour order.
    overlay[skeleton > 0] = (0, 0, 255)

    blended = cv2.addWeighted(
        original,
        0.75,
        overlay,
        0.75,
        0
    )

    return blended


# --------------------------------------------------
# 5. PROCESS ONE DATASET SPLIT
# --------------------------------------------------

def process_split(split):
    input_mask_dir = MASK_ROOT / split / "masks"
    image_dir = IMAGE_ROOT / split / "images"

    output_dir = OUTPUT_ROOT / split / "masks"
    preview_dir = PREVIEW_ROOT / split

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    if not input_mask_dir.exists():
        print(f"[SKIP] Mask folder not found: {input_mask_dir}")
        return

    mask_files = sorted(input_mask_dir.glob("*.png"))

    if not mask_files:
        print(f"[SKIP] No masks found for {split}")
        return

    rows = []
    processed = 0
    empty_masks = 0

    print(f"\n--- CENTERLINE EXTRACTION: {split.upper()} ---")
    print("Input masks:", len(mask_files))

    for index, mask_path in enumerate(mask_files, start=1):
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            print("[WARN] Cannot read:", mask_path.name)
            continue

        # Extract the centreline from the cleaned binary mask.
        centerline = extract_centerline(mask)

        skeleton_pixels, endpoints, junction_pixels = (
            analyze_skeleton(centerline)
        )

        if skeleton_pixels == 0:
            empty_masks += 1

        # Save the skeleton as a binary PNG.
        output_path = output_dir / mask_path.name

        if not cv2.imwrite(str(output_path), centerline):
            raise RuntimeError(
                f"Could not save centerline: {output_path}"
            )

        # Find original angiography image for previews.
        image_path = image_dir / mask_path.name

        if index <= 10 and image_path.exists():
            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_GRAYSCALE
            )

            if image is not None:
                if image.shape != centerline.shape:
                    image = cv2.resize(
                        image,
                        (centerline.shape[1], centerline.shape[0]),
                        interpolation=cv2.INTER_AREA
                    )

                overlay = create_overlay(image, centerline)

                preview_path = (
                    preview_dir /
                    f"{mask_path.stem}_centerline.png"
                )

                cv2.imwrite(str(preview_path), overlay)

        rows.append({
            "split": split,
            "image": mask_path.name,
            "skeleton_pixels": skeleton_pixels,
            "endpoint_pixels": endpoints,
            "junction_pixels": junction_pixels
        })

        processed += 1

        if processed % 100 == 0:
            print(f"Processed {processed}/{len(mask_files)}")

    # Save per-image topology indicators.
    report_path = REPORT_ROOT / f"{split}_centerline_report.csv"

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "image",
                "skeleton_pixels",
                "endpoint_pixels",
                "junction_pixels"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nSplit summary:", split)
    print("Images processed:", processed)
    print("Empty centerlines:", empty_masks)
    print("Centerline masks:", output_dir)
    print("Preview overlays:", preview_dir)
    print("Report:", report_path)


# --------------------------------------------------
# 6. MAIN
# --------------------------------------------------

def main():
    print("Project root:", ROOT)
    print("Input masks:", MASK_ROOT)

    if not MASK_ROOT.exists():
        raise FileNotFoundError(
            f"Post-processed mask root not found: {MASK_ROOT}"
        )

    for split in SPLITS:
        process_split(split)

    print("\nCenterline extraction completed.")
    print("Inspect the red overlays before diameter estimation.")


if __name__ == "__main__":
    main()
