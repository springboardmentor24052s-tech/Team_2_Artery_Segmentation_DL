
from pathlib import Path
import csv

import cv2
import numpy as np


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

IMAGE_ROOT = ROOT / "Dataset" / "arcade" / "arcade" / "syntax"

MASK_ROOT = (
    ROOT / "data" / "processed" / "syntax_postprocessed"
)

CENTERLINE_ROOT = (
    ROOT / "data" / "processed" / "syntax_centerlines"
)

REPORT_ROOT = ROOT / "outputs" / "diameter_reports"
PREVIEW_ROOT = ROOT / "outputs" / "diameter_previews"

SPLITS = ["train", "val", "test"]


# --------------------------------------------------
# 2. ESTIMATE LOCAL DIAMETER
# --------------------------------------------------

def estimate_diameter(mask, centerline):
    """
    Estimate local vessel diameter at each skeleton pixel.

    Distance transform gives the distance from an interior
    mask pixel to the nearest background pixel.

    Approximate local diameter = 2 * distance-to-boundary.

    Returns:
        diameter_map: diameter values at centerline pixels
        diameter_values: array of estimated diameters in pixels
    """

    binary_mask = (mask > 127).astype(np.uint8)
    skeleton = (centerline > 127).astype(np.uint8)

    if binary_mask.shape != skeleton.shape:
        raise ValueError(
            "Mask and centerline dimensions do not match."
        )

    # Keep only skeleton pixels inside the cleaned vessel mask.
    skeleton = (
        (skeleton > 0) & (binary_mask > 0)
    ).astype(np.uint8)

    if not np.any(binary_mask) or not np.any(skeleton):
        return np.zeros_like(binary_mask, dtype=np.float32), np.array([])

    # Distance to nearest background pixel.
    distance_map = cv2.distanceTransform(
        binary_mask,
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE
    )

    # Radius * 2 approximates the local vessel diameter.
    diameter_map = np.zeros_like(
        distance_map,
        dtype=np.float32
    )

    diameter_map[skeleton > 0] = (
        2.0 * distance_map[skeleton > 0]
    )

    diameter_values = diameter_map[skeleton > 0]

    return diameter_map, diameter_values


# --------------------------------------------------
# 3. CREATE DIAMETER VISUALIZATION
# --------------------------------------------------

def save_diameter_overlay(image, diameter_map, output_path):
    """
    Color-code centerline pixels according to their estimated
    local diameter. This is a visualization, not a clinical map.
    """

    original = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR
    )

    valid = diameter_map > 0

    if not np.any(valid):
        cv2.imwrite(str(output_path), original)
        return

    # Normalize using the 95th percentile to improve visibility.
    upper = float(np.percentile(diameter_map[valid], 95))
    upper = max(upper, 1e-6)

    normalized = np.clip(
        diameter_map / upper * 255.0,
        0,
        255
    ).astype(np.uint8)

    color_map = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET
    )

    overlay = original.copy()
    overlay[valid] = color_map[valid]

    result = cv2.addWeighted(
        original,
        0.45,
        overlay,
        0.55,
        0
    )

    cv2.imwrite(str(output_path), result)


# --------------------------------------------------
# 4. PROCESS ONE DATASET SPLIT
# --------------------------------------------------

def process_split(split):
    image_dir = IMAGE_ROOT / split / "images"
    mask_dir = MASK_ROOT / split / "masks"
    centerline_dir = CENTERLINE_ROOT / split / "masks"

    output_dir = PREVIEW_ROOT / split
    output_dir.mkdir(parents=True, exist_ok=True)

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    for folder in [image_dir, mask_dir, centerline_dir]:
        if not folder.exists():
            print(f"[SKIP] Folder not found: {folder}")
            return

    mask_paths = sorted(mask_dir.glob("*.png"))

    if not mask_paths:
        print(f"[SKIP] No masks found for {split}")
        return

    rows = []
    processed = 0
    empty_count = 0
    preview_count = 0

    print(f"\n--- DIAMETER ESTIMATION: {split.upper()} ---")
    print("Input masks:", len(mask_paths))

    for index, mask_path in enumerate(mask_paths, start=1):
        image_path = image_dir / mask_path.name
        centerline_path = centerline_dir / mask_path.name

        if not image_path.exists() or not centerline_path.exists():
            print("[WARN] Missing image or centerline:", mask_path.name)
            continue

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        centerline = cv2.imread(
            str(centerline_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None or mask is None or centerline is None:
            print("[WARN] Could not read files:", mask_path.name)
            continue

        if image.shape != mask.shape or mask.shape != centerline.shape:
            print(
                "[WARN] Shape mismatch:",
                mask_path.name,
                image.shape,
                mask.shape,
                centerline.shape
            )
            continue

        diameter_map, diameters = estimate_diameter(
            mask,
            centerline
        )

        if diameters.size == 0:
            empty_count += 1

            rows.append({
                "image": mask_path.name,
                "centerline_pixels": 0,
                "mean_diameter_px": "",
                "median_diameter_px": "",
                "p10_diameter_px": "",
                "p90_diameter_px": "",
                "min_diameter_px": "",
                "max_diameter_px": ""
            })

            continue

        rows.append({
            "image": mask_path.name,
            "centerline_pixels": int(diameters.size),
            "mean_diameter_px": round(float(np.mean(diameters)), 4),
            "median_diameter_px": round(float(np.median(diameters)), 4),
            "p10_diameter_px": round(float(np.percentile(diameters, 10)), 4),
            "p90_diameter_px": round(float(np.percentile(diameters, 90)), 4),
            "min_diameter_px": round(float(np.min(diameters)), 4),
            "max_diameter_px": round(float(np.max(diameters)), 4)
        })

        # Save previews for the first five valid images.
        if preview_count < 5:
            preview_path = (
                output_dir /
                f"{mask_path.stem}_diameter.png"
            )

            save_diameter_overlay(
                image,
                diameter_map,
                preview_path
            )

            preview_count += 1

        processed += 1

        if index % 100 == 0:
            print(f"Processed {index}/{len(mask_paths)}")

    report_path = (
        REPORT_ROOT / f"{split}_diameter_report.csv"
    )

    fields = [
        "image",
        "centerline_pixels",
        "mean_diameter_px",
        "median_diameter_px",
        "p10_diameter_px",
        "p90_diameter_px",
        "min_diameter_px",
        "max_diameter_px"
    ]

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSplit summary:", split)
    print("Images with measurable centerlines:", processed)
    print("Images without measurable centerlines:", empty_count)
    print("Report saved:", report_path)
    print("Preview images:", output_dir)

    # Summarize the per-image median diameters.
    image_medians = [
        float(row["median_diameter_px"])
        for row in rows
        if row["median_diameter_px"] != ""
    ]

    if image_medians:
        print(
            "Mean of image median diameters (px):",
            round(float(np.mean(image_medians)), 4)
        )


# --------------------------------------------------
# 5. MAIN
# --------------------------------------------------

def main():
    print("Project root:", ROOT)
    print("Post-processed masks:", MASK_ROOT)
    print("Centerlines:", CENTERLINE_ROOT)

    for split in SPLITS:
        process_split(split)

    print("\nDiameter estimation completed.")
    print("All reported diameters are in pixels.")


if __name__ == "__main__":
    main()
