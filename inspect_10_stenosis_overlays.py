
import json
import random
from pathlib import Path

import cv2
import numpy as np


DATASET = Path("backend/app/data/arcade/arcade/arcade")
OUTPUT = Path("stenosis_overlap_samples")

IMAGE_DIR = DATASET / "stenosis" / "val" / "images"
SYNTAX_JSON = DATASET / "syntax" / "val" / "annotations" / "val.json"
STENOSIS_JSON = DATASET / "stenosis" / "val" / "annotations" / "val.json"

SAMPLE_COUNT = 10
RANDOM_SEED = 42


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_mask(annotations, image_id, exclude_stenosis=False):
    mask = np.zeros((512, 512), dtype=np.uint8)

    for annotation in annotations:
        if annotation["image_id"] != image_id:
            continue

        if exclude_stenosis and annotation["category_id"] == 26:
            continue

        segmentation = annotation.get("segmentation", [])

        for polygon in segmentation:
            points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
            points = np.round(points).astype(np.int32)
            cv2.fillPoly(mask, [points], 255)

    return mask


def main():
    OUTPUT.mkdir(exist_ok=True)

    syntax_data = load_json(SYNTAX_JSON)
    stenosis_data = load_json(STENOSIS_JSON)

    images = {
        item["id"]: item
        for item in stenosis_data["images"]
    }

    annotated_ids = sorted({
        annotation["image_id"]
        for annotation in stenosis_data["annotations"]
        if annotation["category_id"] == 26
    })

    random.seed(RANDOM_SEED)
    selected_ids = random.sample(
        annotated_ids,
        min(SAMPLE_COUNT, len(annotated_ids))
    )

    print(f"Generating overlays for {len(selected_ids)} images...")

    for image_id in selected_ids:
        image_info = images[image_id]
        filename = image_info["file_name"]
        image_path = IMAGE_DIR / filename

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Skipping unreadable image: {filename}")
            continue

        vessel_mask = create_mask(
            syntax_data["annotations"],
            image_id,
            exclude_stenosis=True
        )

        stenosis_mask = create_mask(
            stenosis_data["annotations"],
            image_id
        )

        intersection = cv2.bitwise_and(
            vessel_mask,
            stenosis_mask
        )

        stenosis_pixels = np.count_nonzero(stenosis_mask)
        overlap_pixels = np.count_nonzero(intersection)

        overlap_percentage = (
            overlap_pixels / stenosis_pixels * 100
            if stenosis_pixels else 0
        )

        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        vessel_only = (vessel_mask > 0) & (stenosis_mask == 0)
        stenosis_only = (stenosis_mask > 0) & (vessel_mask == 0)
        overlap = (vessel_mask > 0) & (stenosis_mask > 0)

        overlay[vessel_only] = (0, 255, 0)
        overlay[stenosis_only] = (0, 0, 255)
        overlay[overlap] = (0, 255, 255)

        output_path = OUTPUT / f"overlay_{image_id}.png"
        cv2.imwrite(str(output_path), overlay)

        print(
            f"{filename}: "
            f"vessel={np.count_nonzero(vessel_mask)}, "
            f"stenosis={stenosis_pixels}, "
            f"overlap={overlap_pixels}, "
            f"overlap_percentage={overlap_percentage:.2f}%"
        )

    print(f"\nCompleted. Images saved in: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()

