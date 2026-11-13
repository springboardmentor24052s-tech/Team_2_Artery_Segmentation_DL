
import json
from pathlib import Path

import cv2
import numpy as np

IMAGE_ID = "74996d2933414f26a69c7dfa254ad28f"

MASK_PATH = Path(
    "backend/results/api_predictions/"
    f"{IMAGE_ID}_stenosis_mask.png"
)

ANNOTATION_PATH = Path(
    "backend/app/data/arcade/arcade/arcade/"
    "stenosis/test/annotations/test.json"
)

OUTPUT_PATH = Path(
    "backend/results/"
    f"{IMAGE_ID}_candidate_evaluation.json"
)

if not MASK_PATH.exists():
    raise FileNotFoundError(MASK_PATH)

if not ANNOTATION_PATH.exists():
    raise FileNotFoundError(ANNOTATION_PATH)

mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise ValueError("Could not read prediction mask.")

mask = (mask > 0).astype(np.uint8)

with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

image_info = next(
    img for img in data["images"]
    if Path(img["file_name"]).name == "1.png"
)

gt_annotations = [
    ann for ann in data["annotations"]
    if ann["image_id"] == image_info["id"]
    and ann["category_id"] == 26
]

if len(gt_annotations) != 1:
    raise ValueError(
        f"Expected one GT stenosis annotation, "
        f"found {len(gt_annotations)}"
    )

gt_bbox = gt_annotations[0]["bbox"]

def bbox_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = aw * ah
    area_b = bw * bh

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0

count, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )
)

candidates = []

for i in range(1, count):
    x, y, w, h, area = stats[i]

    if area < 10:
        continue

    bbox = [int(x), int(y), int(w), int(h)]
    iou = bbox_iou(bbox, gt_bbox)

    candidates.append({
        "candidate_id": len(candidates) + 1,
        "bbox": bbox,
        "area": int(area),
        "bbox_iou": round(iou, 6)
    })

candidates.sort(
    key=lambda item: item["bbox_iou"],
    reverse=True
)

report = {
    "image_id": IMAGE_ID,
    "ground_truth_bbox": gt_bbox,
    "ground_truth_count": len(gt_annotations),
    "predicted_candidate_count": len(candidates),
    "best_candidate": candidates[0] if candidates else None,
    "candidates": candidates
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

print("\n========== CANDIDATE LOCALIZATION EVALUATION ==========")
print("Ground truth bbox:", gt_bbox)
print("Predicted candidates:", len(candidates))

for candidate in candidates[:10]:
    print(
        f"Candidate {candidate['candidate_id']:2d} | "
        f"BBox: {candidate['bbox']} | "
        f"Area: {candidate['area']:5d} | "
        f"IoU: {candidate['bbox_iou']:.4f}"
    )

print("\nSaved report:", OUTPUT_PATH)
