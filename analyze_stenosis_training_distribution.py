
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent

DATASET = (
    ROOT / "backend/app/data/arcade/arcade/arcade"
)

MASK_ROOT = ROOT / "backend/results/arcade_masks"

SPLITS = ["train", "val", "test"]

for split in SPLITS:

    print(f"\n========== {split.upper()} ==========")

    annotation_file = (
        DATASET / f"stenosis/{split}/annotations/{split}.json"
    )

    mask_dir = MASK_ROOT / f"stenosis/{split}"

    with open(annotation_file, "r") as file:
        data = json.load(file)

    annotations_by_image = {}

    for ann in data["annotations"]:

        if ann["category_id"] == 26:

            annotations_by_image.setdefault(
                ann["image_id"], []
            ).append(ann)

    foreground_percentages = []
    empty_masks = 0
    total_foreground = 0

    for image_info in data["images"]:

        filename = Path(
            image_info["file_name"]
        ).stem

        mask_path = mask_dir / f"{filename}_mask.png"

        if not mask_path.is_file():
            continue

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            continue

        foreground = np.count_nonzero(mask)

        total_pixels = mask.size

        percentage = (
            foreground / total_pixels
        ) * 100

        foreground_percentages.append(percentage)

        total_foreground += foreground

        if foreground == 0:
            empty_masks += 1

    values = np.array(
        foreground_percentages,
        dtype=np.float64
    )

    if len(values) == 0:
        print("No masks found.")
        continue

    print("Images analyzed:", len(values))
    print("Empty masks:", empty_masks)

    print(
        "Mean foreground percentage:",
        round(float(values.mean()), 4)
    )

    print(
        "Median foreground percentage:",
        round(float(np.median(values)), 4)
    )

    print(
        "Minimum foreground percentage:",
        round(float(values.min()), 4)
    )

    print(
        "Maximum foreground percentage:",
        round(float(values.max()), 4)
    )

    print(
        "25th percentile:",
        round(float(np.percentile(values, 25)), 4)
    )

    print(
        "75th percentile:",
        round(float(np.percentile(values, 75)), 4)
    )

    print(
        "Total foreground pixels:",
        int(total_foreground)
    )

print("\nDistribution analysis completed.")
