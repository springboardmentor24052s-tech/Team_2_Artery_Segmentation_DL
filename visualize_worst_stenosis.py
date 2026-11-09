
import sys
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"

sys.path.insert(0, str(BACKEND))

from app.services.segmentation import predict_stenosis

DATASET_ROOT = (
    ROOT / "backend/app/data/arcade/arcade/arcade"
)

IMAGE_DIR = DATASET_ROOT / "stenosis/val/images"

ANNOTATION_PATH = (
    DATASET_ROOT / "stenosis/val/annotations/val.json"
)

OUTPUT_DIR = (
    ROOT / "backend/results/stenosis_error_visualization"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORST_IMAGES = [
    "134.png",
    "156.png",
    "129.png",
    "163.png"
]


# Load annotations
with open(ANNOTATION_PATH, "r") as file:
    data = json.load(file)

annotations_by_image = {}

for ann in data["annotations"]:
    if ann["category_id"] == 26:
        annotations_by_image.setdefault(
            ann["image_id"], []
        ).append(ann)


def create_ground_truth(image_info):

    gt = np.zeros(
        (image_info["height"], image_info["width"]),
        dtype=np.uint8
    )

    for ann in annotations_by_image.get(image_info["id"], []):

        segmentation = ann.get("segmentation", [])

        if isinstance(segmentation, list):

            for polygon in segmentation:

                if len(polygon) < 6:
                    continue

                points = np.array(
                    polygon,
                    dtype=np.float32
                ).reshape(-1, 2)

                points = np.round(points).astype(np.int32)

                cv2.fillPoly(gt, [points], 255)

    return gt


# Process each worst-performing image
for filename in WORST_IMAGES:

    print("\nProcessing:", filename)

    image_info = next(
        (
            image for image in data["images"]
            if Path(image["file_name"]).name == filename
        ),
        None
    )

    if image_info is None:
        print("Image not found in annotations:", filename)
        continue

    image_path = IMAGE_DIR / filename

    if not image_path.is_file():
        print("Image file missing:", filename)
        continue

    original = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if original is None:
        print("Unable to load:", filename)
        continue

    gt = create_ground_truth(image_info)

    prediction = predict_stenosis(image_path)

    probability = np.asarray(
        prediction["probability_map"],
        dtype=np.float32
    )

    pred = (
        probability >= 0.20
    ).astype(np.uint8) * 255

    if pred.shape != gt.shape:
        print("Dimension mismatch:", filename)
        continue

    # Create RGB visualization
    original_rgb = cv2.cvtColor(
        original,
        cv2.COLOR_GRAY2BGR
    )

    gt_rgb = cv2.cvtColor(
        gt,
        cv2.COLOR_GRAY2BGR
    )

    pred_rgb = cv2.cvtColor(
        pred,
        cv2.COLOR_GRAY2BGR
    )

    # Green = overlap
    # Blue = ground truth only
    # Red = prediction only

    comparison = np.zeros(
        (*gt.shape, 3),
        dtype=np.uint8
    )

    gt_bool = gt > 0
    pred_bool = pred > 0

    comparison[
        np.logical_and(gt_bool, ~pred_bool)
    ] = [255, 0, 0]

    comparison[
        np.logical_and(~gt_bool, pred_bool)
    ] = [0, 0, 255]

    comparison[
        np.logical_and(gt_bool, pred_bool)
    ] = [0, 255, 0]

    # Save individual images
    cv2.imwrite(
        str(OUTPUT_DIR / f"{filename}_original.png"),
        original_rgb
    )

    cv2.imwrite(
        str(OUTPUT_DIR / f"{filename}_ground_truth.png"),
        gt_rgb
    )

    cv2.imwrite(
        str(OUTPUT_DIR / f"{filename}_prediction.png"),
        pred_rgb
    )

    cv2.imwrite(
        str(OUTPUT_DIR / f"{filename}_comparison.png"),
        comparison
    )

    print("Ground-truth pixels:", np.count_nonzero(gt))
    print("Predicted pixels:", np.count_nonzero(pred))
    print(
        "Maximum probability:",
        round(float(probability.max()), 4)
    )

print("\nVisualization completed.")
print("Output directory:", OUTPUT_DIR)
