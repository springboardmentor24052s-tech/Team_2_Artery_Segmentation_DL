
from pathlib import Path
import cv2
import json
from ultralytics import YOLO

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

MODEL_PATH = (
    BACKEND_ROOT / "results" / "stenosis_yolo_baseline" / "best.pt"
)

DATASET_ROOT = BACKEND_ROOT / "results" / "stenosis_yolo_dataset"

IMAGE_DIR = DATASET_ROOT / "images" / "test"
LABEL_DIR = DATASET_ROOT / "labels" / "test"

OUTPUT_DIR = (
    BACKEND_ROOT / "results" / "nms_analysis"
    / "final_test_evaluation" / "error_visualizations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

IMAGE_SIZE = 512
CONFIDENCE = 0.10
NMS_IOU = 0.30
MATCH_IOU = 0.50

SELECTED_IMAGES = [
    "168.png",
    "107.png",
    "79.png",
    "11.png",
    "136.png"
]

# --------------------------------------------------
# 3. IOU CALCULATION
# --------------------------------------------------

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = (
        max(0, x2 - x1) *
        max(0, y2 - y1)
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


# --------------------------------------------------
# 4. LOAD GROUND-TRUTH BOXES
# --------------------------------------------------

def load_ground_truth(label_path):

    boxes = []

    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing label file: {label_path}"
        )

    with open(label_path, "r") as file:

        for line in file:

            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id, cx, cy, width, height = map(
                float, values
            )

            if int(class_id) != 0:
                continue

            x1 = (cx - width / 2) * IMAGE_SIZE
            y1 = (cy - height / 2) * IMAGE_SIZE
            x2 = (cx + width / 2) * IMAGE_SIZE
            y2 = (cy + height / 2) * IMAGE_SIZE

            boxes.append([x1, y1, x2, y2])

    return boxes


# --------------------------------------------------
# 5. MATCH GROUND TRUTH AND PREDICTIONS
# --------------------------------------------------

def match_predictions(gt_boxes, predictions):

    candidates = []

    for gt_index, gt in enumerate(gt_boxes):

        for pred_index, prediction in enumerate(predictions):

            iou = calculate_iou(
                gt,
                prediction["box"]
            )

            if iou >= MATCH_IOU:

                candidates.append(
                    (iou, gt_index, pred_index)
                )

    candidates.sort(reverse=True)

    matched_gt = set()
    matched_predictions = set()

    for iou, gt_index, pred_index in candidates:

        if (
            gt_index not in matched_gt
            and pred_index not in matched_predictions
        ):

            matched_gt.add(gt_index)
            matched_predictions.add(pred_index)

    missed_gt = [
        i for i in range(len(gt_boxes))
        if i not in matched_gt
    ]

    false_predictions = [
        i for i in range(len(predictions))
        if i not in matched_predictions
    ]

    matched_pairs = [
        (gt_index, pred_index, iou)
        for iou, gt_index, pred_index in candidates
        if (
            gt_index in matched_gt
            and pred_index in matched_predictions
        )
    ]

    # Keep only the actual one-to-one pairs
    matched_pairs = []
    for iou, gt_index, pred_index in candidates:
        if (
            gt_index in matched_gt
            and pred_index in matched_predictions
            and not any(
                pair[0] == gt_index or pair[1] == pred_index
                for pair in matched_pairs
            )
        ):
            matched_pairs.append(
                (gt_index, pred_index, iou)
            )

    return matched_pairs, missed_gt, false_predictions


# --------------------------------------------------
# 6. DRAW VISUAL ERROR ANALYSIS
# --------------------------------------------------

def draw_error_boxes(
    image,
    gt_boxes,
    predictions,
    matched_pairs,
    missed_gt,
    false_predictions
):

    output = image.copy()

    # Matched ground-truth boxes: GREEN
    for gt_index, pred_index, iou in matched_pairs:

        x1, y1, x2, y2 = map(
            int, gt_boxes[gt_index]
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            f"Matched GT IoU:{iou:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0),
            1
        )

    # Missed ground-truth boxes: YELLOW
    for gt_index in missed_gt:

        x1, y1, x2, y2 = map(
            int, gt_boxes[gt_index]
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2
        )

        cv2.putText(
            output,
            "MISSED GT",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1
        )

    # Matched predictions: BLUE
    for gt_index, pred_index, iou in matched_pairs:

        prediction = predictions[pred_index]

        x1, y1, x2, y2 = map(
            int, prediction["box"]
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            output,
            f"MATCH PRED {prediction['confidence']:.2f}",
            (x1, min(output.shape[0] - 5, y2 + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1
        )

    # False-positive predictions: RED
    for pred_index in false_predictions:

        prediction = predictions[pred_index]

        x1, y1, x2, y2 = map(
            int, prediction["box"]
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            output,
            f"FALSE POS {prediction['confidence']:.2f}",
            (x1, min(output.shape[0] - 5, y2 + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1
        )

    return output


# --------------------------------------------------
# 7. MAIN
# --------------------------------------------------

def main():

    print("Loading V1 YOLO model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    model = YOLO(str(MODEL_PATH))

    report = []

    for image_name in SELECTED_IMAGES:

        image_path = IMAGE_DIR / image_name
        label_path = LABEL_DIR / f"{Path(image_name).stem}.txt"

        if not image_path.exists():
            print(f"Image not found: {image_name}")
            continue

        print(f"\nProcessing {image_name}")

        image = cv2.imread(str(image_path))

        if image is None:
            print("Unable to read image.")
            continue

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        gt_boxes = load_ground_truth(label_path)

        result = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            iou=NMS_IOU,
            device="cpu",
            max_det=300,
            verbose=False
        )[0]

        predictions = []

        for box in result.boxes:

            predictions.append({
                "box": box.xyxy[0].cpu().tolist(),
                "confidence": float(box.conf[0].cpu())
            })

        matched_pairs, missed_gt, false_predictions = (
            match_predictions(gt_boxes, predictions)
        )

        tp = len(matched_pairs)
        fp = len(false_predictions)
        fn = len(missed_gt)

        final_image = draw_error_boxes(
            image,
            gt_boxes,
            predictions,
            matched_pairs,
            missed_gt,
            false_predictions
        )

        output_path = (
            OUTPUT_DIR
            / f"{Path(image_name).stem}_error_analysis.png"
        )

        cv2.imwrite(
            str(output_path),
            final_image
        )

        report.append({
            "image": image_name,
            "ground_truth": len(gt_boxes),
            "predictions": len(predictions),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "output": str(output_path)
        })

        print(
            f"TP={tp}, FP={fp}, FN={fn}"
        )

        print(f"Saved: {output_path}")

    report_path = OUTPUT_DIR / "selected_image_error_report.json"

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    print("\nVisual error analysis completed.")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
