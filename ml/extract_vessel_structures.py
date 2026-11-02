
from pathlib import Path
import csv

import cv2
import numpy as np


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

IMAGE_ROOT = ROOT / "Dataset" / "arcade" / "arcade" / "syntax"

MASK_ROOT = (
    ROOT / "data" / "processed" / "syntax_postprocessed"
)

OUTPUT_ROOT = ROOT / "data" / "processed" / "syntax_structures"

REPORT_ROOT = ROOT / "outputs" / "vessel_structure_reports"

SPLITS = ["train", "val", "test"]

MIN_COMPONENT_AREA = 10


# --------------------------------------------------
# 2. COMPONENT COLOUR VISUALIZATION
# --------------------------------------------------

def create_component_visualization(image, labels):
    """
    Display each connected component in a different colour
    over the original grayscale angiography image.
    """

    original = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    colors = np.zeros_like(original)

    component_ids = np.unique(labels)
    rng = np.random.default_rng(42)

    for component_id in component_ids:
        if component_id == 0:
            continue

        color = rng.integers(
            low=60,
            high=256,
            size=3,
            dtype=np.uint8
        )

        colors[labels == component_id] = color

    # Blend original image and component colours
    overlay = cv2.addWeighted(
        original, 0.75,
        colors, 0.75,
        0
    )

    return overlay


# --------------------------------------------------
# 3. EXTRACT CONNECTED VESSEL REGIONS
# --------------------------------------------------

def process_split(split):
    input_mask_dir = MASK_ROOT / split / "masks"
    image_dir = IMAGE_ROOT / split / "images"

    label_dir = OUTPUT_ROOT / split / "labels"
    preview_dir = OUTPUT_ROOT / split / "previews"

    label_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    if not input_mask_dir.exists():
        print(f"[SKIP] Mask folder not found: {input_mask_dir}")
        return

    mask_files = sorted(input_mask_dir.glob("*.png"))

    if not mask_files:
        print(f"[SKIP] No masks found for {split}")
        return

    report_rows = []
    total_images = 0
    total_components = 0
    empty_images = 0

    print(f"\n{'=' * 55}")
    print(f"VESSEL STRUCTURE EXTRACTION: {split.upper()}")
    print(f"{'=' * 55}")
    print("Input masks:", len(mask_files))

    for index, mask_path in enumerate(mask_files, start=1):
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            print("[WARN] Cannot read mask:", mask_path.name)
            continue

        binary = (mask > 127).astype(np.uint8)

        # Find individual connected regions
        count, raw_labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8
            )
        )

        # Store compact component IDs in a 16-bit label map.
        # Zero remains the background.
        labels = np.zeros(
            binary.shape,
            dtype=np.uint16
        )

        next_label = 1
        image_component_count = 0

        for component_id in range(1, count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])

            if area < MIN_COMPONENT_AREA:
                continue

            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            w = int(stats[component_id, cv2.CC_STAT_WIDTH])
            h = int(stats[component_id, cv2.CC_STAT_HEIGHT])

            centroid_x, centroid_y = centroids[component_id]

            labels[raw_labels == component_id] = next_label

            report_rows.append({
                "split": split,
                "image": mask_path.name,
                "component_id": next_label,
                "area_pixels": area,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_width": w,
                "bbox_height": h,
                "centroid_x": round(float(centroid_x), 3),
                "centroid_y": round(float(centroid_y), 3)
            })

            next_label += 1
            image_component_count += 1

        total_images += 1
        total_components += image_component_count

        if image_component_count == 0:
            empty_images += 1

        # Save the labelled image.
        label_path = label_dir / mask_path.name

        if not cv2.imwrite(str(label_path), labels):
            raise RuntimeError(f"Cannot save: {label_path}")

        # Save coloured component previews for the first 5 images.
        if index <= 5:
            original_path = image_dir / mask_path.name

            image = cv2.imread(
                str(original_path),
                cv2.IMREAD_GRAYSCALE
            )

            if image is not None:
                image = cv2.resize(
                    image,
                    (labels.shape[1], labels.shape[0])
                )

                overlay = create_component_visualization(
                    image, labels
                )

                preview_path = (
                    preview_dir /
                    f"{mask_path.stem}_components.png"
                )

                cv2.imwrite(str(preview_path), overlay)

        if index % 100 == 0:
            print(f"Processed {index}/{len(mask_files)}")

    # Save component measurements for this split.
    report_path = REPORT_ROOT / f"{split}_components.csv"

    fields = [
        "split",
        "image",
        "component_id",
        "area_pixels",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "centroid_x",
        "centroid_y"
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
        writer.writerows(report_rows)

    print("\nSplit summary:", split)
    print("Images processed:", total_images)
    print("Connected components:", total_components)
    print("Images without retained components:", empty_images)
    print("Label maps saved to:", label_dir)
    print("Preview images saved to:", preview_dir)
    print("Component report saved to:", report_path)


# --------------------------------------------------
# 4. MAIN
# --------------------------------------------------

def main():
    print("Project root:", ROOT)
    print("Post-processed masks:", MASK_ROOT)

    for split in SPLITS:
        process_split(split)

    print("\nVessel structure extraction completed.")
    print("Review component previews before centerline extraction.")


if __name__ == "__main__":
    main()
