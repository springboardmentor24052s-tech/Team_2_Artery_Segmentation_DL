
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("backend/app/data/arcade/arcade/arcade")
OUTPUT = Path("backend/results/arcade_masks")

TASKS = ["syntax", "stenosis"]
SPLITS = ["train", "val", "test"]


def create_mask(image_info, annotations):
    width = image_info["width"]
    height = image_info["height"]

    mask = np.zeros((height, width), dtype=np.uint8)

    for annotation in annotations:
        segmentation = annotation.get("segmentation", [])

        # Handle COCO polygon annotations.
        if not isinstance(segmentation, list):
            continue

        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6:
                continue

            points = np.asarray(
                polygon, dtype=np.float32
            ).reshape(-1, 2)

            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)

            points = np.rint(points).astype(np.int32)
            cv2.fillPoly(mask, [points], 255)

    return mask


def process_split(task, split):
    annotation_file = (
        ROOT / task / split / "annotations" / f"{split}.json"
    )
    image_dir = ROOT / task / split / "images"
    output_dir = OUTPUT / task / split

    if not annotation_file.exists():
        print(f"SKIP: Missing annotation file: {annotation_file}")
        return

    if not image_dir.exists():
        print(f"SKIP: Missing image directory: {image_dir}")
        return

    with annotation_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    annotations_by_image = {}
    for annotation in data["annotations"]:
        annotations_by_image.setdefault(
            annotation["image_id"], []
        ).append(annotation)

    output_dir.mkdir(parents=True, exist_ok=True)
    no_annotation_images = []
    processed = 0

    for image_info in data["images"]:
        image_id = image_info["id"]
        image_path = image_dir / image_info["file_name"]

        if not image_path.exists():
            print(f"WARNING: Missing image: {image_path}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"WARNING: Cannot read image: {image_path}")
            continue

        annotations = annotations_by_image.get(image_id, [])

        if not annotations:
            no_annotation_images.append(image_info["file_name"])

        mask = create_mask(image_info, annotations)

        # Keep masks separate by task and split.
        mask_path = output_dir / (
            Path(image_info["file_name"]).stem + "_mask.png"
        )
        cv2.imwrite(str(mask_path), mask)
        processed += 1

    report_path = output_dir / "report.txt"
    report_path.write_text(
        f"Task: {task}\n"
        f"Split: {split}\n"
        f"Images listed: {len(data['images'])}\n"
        f"Images processed: {processed}\n"
        f"Images with no annotations: {len(no_annotation_images)}\n"
        f"Files with no annotations:\n"
        + "\n".join(no_annotation_images),
        encoding="utf-8",
    )

    print(
        f"{task}/{split}: processed {processed} images; "
        f"{len(no_annotation_images)} images have no annotations."
    )
    print(f"Report: {report_path}")


def main():
    for task in TASKS:
        for split in SPLITS:
            process_split(task, split)

    print("\nMask generation finished.")
    print("Output directory:", OUTPUT.resolve())


if __name__ == "__main__":
    main()
