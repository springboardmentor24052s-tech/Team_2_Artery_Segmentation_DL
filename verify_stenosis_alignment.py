
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent

DATASET = (
    ROOT / "backend/app/data/arcade/arcade/arcade"
)

IMAGE_DIR = DATASET / "stenosis/val/images"

ANNOTATION_FILE = (
    DATASET / "stenosis/val/annotations/val.json"
)

PREDICTION_DIR = (
    ROOT / "backend/results/stenosis_error_visualization"
)

OUTPUT_DIR = (
    ROOT / "backend/results/stenosis_alignment_check"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = ["134.png", "156.png", "129.png", "163.png"]

with open(ANNOTATION_FILE, "r") as f:
    data = json.load(f)

annotations = {}

for ann in data["annotations"]:
    if ann["category_id"] == 26:
        annotations.setdefault(ann["image_id"], []).append(ann)


def create_gt(image_info):

    mask = np.zeros(
        (image_info["height"], image_info["width"]),
        dtype=np.uint8
    )

    for ann in annotations.get(image_info["id"], []):

        for polygon in ann.get("segmentation", []):

            if len(polygon) < 6:
                continue

            points = np.array(
                polygon,
                dtype=np.float32
            ).reshape(-1, 2)

            points = np.round(points).astype(np.int32)

            cv2.fillPoly(mask, [points], 255)

    return mask


for filename in FILES:

    image_info = next(
        (
            item for item in data["images"]
            if Path(item["file_name"]).name == filename
        ),
        None
    )

    if image_info is None:
        print("Missing annotation:", filename)
        continue

    original_path = IMAGE_DIR / filename

    gt_path = OUTPUT_DIR / f"{filename}_ground_truth.png"

    pred_path = PREDICTION_DIR / f"{filename}_prediction.png"

    original = cv2.imread(
        str(original_path),
        cv2.IMREAD_GRAYSCALE
    )

    if original is None:
        print("Missing original:", filename)
        continue

    gt = create_gt(image_info)

    if not cv2.imwrite(str(gt_path), gt):
        raise RuntimeError(f"Could not save GT: {filename}")

    if not pred_path.is_file():
        print("Missing prediction:", pred_path)
        continue

    pred = cv2.imread(
        str(pred_path),
        cv2.IMREAD_GRAYSCALE
    )

    if pred is None or pred.shape != gt.shape:
        print("Invalid prediction:", filename)
        continue

    # Create separate overlays.
    original_bgr = cv2.cvtColor(
        original,
        cv2.COLOR_GRAY2BGR
    )

    gt_overlay = original_bgr.copy()
    pred_overlay = original_bgr.copy()

    gt_overlay[gt > 0] = (
        0.5 * gt_overlay[gt > 0] +
        0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    pred_overlay[pred > 0] = (
        0.5 * pred_overlay[pred > 0] +
        0.5 * np.array([0, 0, 255])
    ).astype(np.uint8)

    cv2.imwrite(
        str(OUTPUT_DIR / f"{filename}_gt_overlay.png"),
        gt_overlay
    )

    cv2.imwrite(
        str(OUTPUT_DIR / f"{filename}_pred_overlay.png"),
        pred_overlay
    )

    print(
        filename,
        "| Original:", original.shape,
        "| GT:", gt.shape,
        "| Prediction:", pred.shape
    )

print("\nAlignment visualization completed.")
print("Output:", OUTPUT_DIR)
