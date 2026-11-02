
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "arcade" / "arcade" / "stenosis"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "stenosis"

SPLITS = ["train", "val", "test"]
STENOSIS_CATEGORY_ID = 26


def process_split(split):
    print(f"\nProcessing {split.upper()} dataset...")

    split_dir = DATASET_ROOT / split
    image_dir = split_dir / "images"
    annotation_file = split_dir / "annotations" / f"{split}.json"

    if not image_dir.exists():
        print(f"Image folder not found: {image_dir}")
        return

    if not annotation_file.exists():
        print(f"Annotation file not found: {annotation_file}")
        return

    with open(annotation_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    output_images = OUTPUT_ROOT / split / "images"
    output_masks = OUTPUT_ROOT / split / "masks"
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)

    image_info = {item["id"]: item for item in coco["images"]}
    annotations_by_image = defaultdict(list)

    for ann in coco["annotations"]:
        if ann["category_id"] == STENOSIS_CATEGORY_ID:
            annotations_by_image[ann["image_id"]].append(ann)

    processed = 0
    failed = 0
    masks_with_stenosis = 0

    for image_id, info in image_info.items():
        filename = info["file_name"]
        image_path = image_dir / filename

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Unreadable image: {image_path}")
            failed += 1
            continue

        height, width = image.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        for ann in annotations_by_image[image_id]:
            segmentation = ann.get("segmentation", [])

            # COCO polygon format
            if isinstance(segmentation, list):
                for polygon in segmentation:
                    if len(polygon) >= 6 and len(polygon) % 2 == 0:
                        points = np.array(
                            polygon, dtype=np.float32
                        ).reshape(-1, 2)

                        points[:, 0] = np.clip(
                            points[:, 0], 0, width - 1
                        )
                        points[:, 1] = np.clip(
                            points[:, 1], 0, height - 1
                        )

                        points = np.round(points).astype(np.int32)
                        cv2.fillPoly(mask, [points], 255)

        # Save original grayscale image and binary mask
        cv2.imwrite(str(output_images / filename), image)
        cv2.imwrite(str(output_masks / filename), mask)

        if np.any(mask > 0):
            masks_with_stenosis += 1

        processed += 1

    print(f"Images processed: {processed}")
    print(f"Failed images: {failed}")
    print(f"Masks containing stenosis: {masks_with_stenosis}")
    print(f"Images saved to: {output_images}")
    print(f"Masks saved to: {output_masks}")


if __name__ == "__main__":
    for split_name in SPLITS:
        process_split(split_name)

    print("\nPreprocessing finished.")
