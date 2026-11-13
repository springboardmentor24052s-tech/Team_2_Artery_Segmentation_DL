
import os
import json
import cv2
import numpy as np

# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(PROJECT_DIR, "backend")

ANNOTATION_FILE = os.path.join(
    BASE_DIR,
    "app",
    "data",
    "arcade",
    "arcade",
    "arcade",
    "stenosis",
    "test",
    "annotations",
    "test.json"
)

ORIGINAL_IMAGE = os.path.join(
    BASE_DIR,
    "app",
    "data",
    "arcade",
    "arcade",
    "arcade",
    "stenosis",
    "test",
    "images",
    "13.png"
)

PREDICTION_MASK = os.path.join(
    BASE_DIR,
    "results",
    "api_predictions",
    "5f9b3c542b06444493e5c7ad2c3c3e89_stenosis_mask.png"
)

CLEANED_MASK = os.path.join(
    BASE_DIR,
    "results",
    "api_predictions",
    "5f9b3c542b06444493e5c7ad2c3c3e89_stenosis_cleaned_mask.png"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "stenosis_single_evaluation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_ID = 35
CATEGORY_ID = 26

# ==================================================
# 2. VERIFY INPUT FILES
# ==================================================

required_files = [
    ANNOTATION_FILE,
    ORIGINAL_IMAGE,
    PREDICTION_MASK,
    CLEANED_MASK
]

for path in required_files:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

print("\nAll required files found.")

# ==================================================
# 3. LOAD COCO ANNOTATIONS
# ==================================================

with open(ANNOTATION_FILE, "r") as file:
    coco_data = json.load(file)

annotations = [
    annotation
    for annotation in coco_data["annotations"]
    if annotation["image_id"] == IMAGE_ID
    and annotation["category_id"] == CATEGORY_ID
]

if not annotations:
    raise ValueError(
        f"No stenosis annotations found for image ID {IMAGE_ID}."
    )

print("Image ID:", IMAGE_ID)
print("Stenosis annotations:", len(annotations))

# ==================================================
# 4. LOAD ORIGINAL IMAGE AND PREDICTION MASKS
# ==================================================

original = cv2.imread(
    ORIGINAL_IMAGE,
    cv2.IMREAD_COLOR
)

prediction = cv2.imread(
    PREDICTION_MASK,
    cv2.IMREAD_GRAYSCALE
)

cleaned_prediction = cv2.imread(
    CLEANED_MASK,
    cv2.IMREAD_GRAYSCALE
)

if original is None:
    raise ValueError("Could not load original image.")

if prediction is None:
    raise ValueError("Could not load raw prediction mask.")

if cleaned_prediction is None:
    raise ValueError("Could not load cleaned prediction mask.")

height, width = original.shape[:2]

if prediction.shape != (height, width):
    raise ValueError("Raw prediction dimensions do not match original.")

if cleaned_prediction.shape != (height, width):
    raise ValueError("Cleaned prediction dimensions do not match original.")

print("Image dimensions:", width, "x", height)

# ==================================================
# 5. GENERATE COCO GROUND-TRUTH MASK
# ==================================================

ground_truth = np.zeros(
    (height, width),
    dtype=np.uint8
)

for annotation in annotations:

    segmentation = annotation.get("segmentation", [])

    if not isinstance(segmentation, list):
        raise ValueError(
            "Expected COCO polygon segmentation format."
        )

    for polygon in segmentation:

        points = np.asarray(
            polygon,
            dtype=np.float32
        ).reshape(-1, 2)

        if len(points) < 3:
            continue

        points = np.round(points).astype(np.int32)

        cv2.fillPoly(
            ground_truth,
            [points],
            255
        )

if np.count_nonzero(ground_truth) == 0:
    raise ValueError("Generated ground-truth mask is empty.")

# Convert masks to binary
gt_binary = ground_truth > 0
raw_binary = prediction > 0
clean_binary = cleaned_prediction > 0

# ==================================================
# 6. EVALUATION FUNCTION
# ==================================================

def calculate_metrics(gt, pred):

    TP = int(np.logical_and(gt, pred).sum())
    FP = int(np.logical_and(~gt, pred).sum())
    FN = int(np.logical_and(gt, ~pred).sum())
    TN = int(np.logical_and(~gt, ~pred).sum())

    epsilon = 1e-8

    dice = (
        2 * TP
    ) / (
        2 * TP + FP + FN + epsilon
    )

    iou = TP / (
        TP + FP + FN + epsilon
    )

    precision = TP / (
        TP + FP + epsilon
    )

    recall = TP / (
        TP + FN + epsilon
    )

    specificity = TN / (
        TN + FP + epsilon
    )

    pixel_accuracy = (
        TP + TN
    ) / (
        TP + TN + FP + FN + epsilon
    )

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "pixel_accuracy": float(pixel_accuracy)
    }

# ==================================================
# 7. EVALUATE RAW AND CLEANED MASKS
# ==================================================

raw_metrics = calculate_metrics(
    gt_binary,
    raw_binary
)

clean_metrics = calculate_metrics(
    gt_binary,
    clean_binary
)

print("\n==========================================")
print("       STENOSIS SEGMENTATION EVALUATION")
print("==========================================")

print("\nImage: 13.png")
print("Image ID:", IMAGE_ID)
print("Dimensions:", width, "x", height)

print("\nGROUND TRUTH")
print("Foreground pixels:", int(gt_binary.sum()))

print("\nRAW PREDICTION")
print("Foreground pixels:", int(raw_binary.sum()))

print("\nCLEANED PREDICTION")
print("Foreground pixels:", int(clean_binary.sum()))

def display_metrics(title, metrics):

    print(f"\n========== {title} ==========")

    print("True Positives :", metrics["TP"])
    print("False Positives:", metrics["FP"])
    print("False Negatives:", metrics["FN"])
    print("True Negatives :", metrics["TN"])

    print(f"Dice Score     : {metrics['dice']:.4f}")
    print(f"IoU            : {metrics['iou']:.4f}")
    print(f"Precision      : {metrics['precision']:.4f}")
    print(f"Recall         : {metrics['recall']:.4f}")
    print(f"Specificity    : {metrics['specificity']:.4f}")
    print(f"Pixel Accuracy : {metrics['pixel_accuracy']:.4f}")

display_metrics("RAW MASK EVALUATION", raw_metrics)
display_metrics("CLEANED MASK EVALUATION", clean_metrics)

# ==================================================
# 8. SAVE GROUND TRUTH AND PREDICTION MASKS
# ==================================================

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "13_ground_truth.png"),
    ground_truth
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "13_raw_prediction.png"),
    prediction
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "13_cleaned_prediction.png"),
    cleaned_prediction
)

# ==================================================
# 9. CREATE COLOR-CODED COMPARISON
# ==================================================

def create_overlay(image, gt, pred):

    overlay = image.copy()

    gt_only = np.logical_and(gt, ~pred)
    pred_only = np.logical_and(pred, ~gt)
    overlap = np.logical_and(gt, pred)

    # Green = Ground truth only
    overlay[gt_only] = [0, 255, 0]

    # Red = Prediction only
    overlay[pred_only] = [0, 0, 255]

    # Yellow = Overlap
    overlay[overlap] = [0, 255, 255]

    return overlay

raw_overlay = create_overlay(
    original,
    gt_binary,
    raw_binary
)

clean_overlay = create_overlay(
    original,
    gt_binary,
    clean_binary
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "13_raw_overlay.png"),
    raw_overlay
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "13_cleaned_overlay.png"),
    clean_overlay
)

# ==================================================
# 10. CREATE FOUR-PANEL COMPARISON
# ==================================================

def add_label(image, label):

    result = image.copy()

    cv2.rectangle(
        result,
        (0, 0),
        (result.shape[1], 38),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        result,
        label,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    return result

original_panel = add_label(
    original,
    "Original Image"
)

gt_panel = add_label(
    cv2.cvtColor(ground_truth, cv2.COLOR_GRAY2BGR),
    "Ground Truth"
)

raw_panel = add_label(
    cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR),
    "Raw Prediction"
)

overlay_panel = add_label(
    raw_overlay,
    "Raw Prediction Comparison"
)

top = np.hstack((
    original_panel,
    gt_panel
))

bottom = np.hstack((
    raw_panel,
    overlay_panel
))

comparison = np.vstack((
    top,
    bottom
))

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "13_four_panel_comparison.png"
    ),
    comparison
)

# ==================================================
# 11. SAVE METRICS TO JSON
# ==================================================

results = {
    "dataset": "ARCADE",
    "split": "stenosis_test",
    "image": "13.png",
    "image_id": IMAGE_ID,
    "category_id": CATEGORY_ID,
    "dimensions": [width, height],
    "ground_truth_pixels": int(gt_binary.sum()),
    "raw_prediction_pixels": int(raw_binary.sum()),
    "cleaned_prediction_pixels": int(clean_binary.sum()),
    "raw_metrics": raw_metrics,
    "cleaned_metrics": clean_metrics,
    "metric_note": (
        "Single-image pixel-level evaluation. "
        "Not representative of full test-set performance."
    )
}

json_path = os.path.join(
    OUTPUT_DIR,
    "13_evaluation_metrics.json"
)

with open(json_path, "w") as file:
    json.dump(results, file, indent=4)

# ==================================================
# 12. COMPLETION
# ==================================================

print("\n==========================================")
print("       EVALUATION COMPLETED")
print("==========================================")

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-", filename)
