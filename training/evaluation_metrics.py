
from pathlib import Path
import csv
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

OUTPUT_DIR = Path("training/outputs")

MODEL_FOLDERS = {
    "baseline_unet":
        OUTPUT_DIR / "unet_baseline_5epochs" / "prediction_examples",

    "attention_unet":
        OUTPUT_DIR / "attention_unet_5epochs" / "prediction_examples",
}

REPORT_DIR = OUTPUT_DIR / "additional_evaluation"


def calculate_metrics(prediction_path, ground_truth_path):

    prediction = cv2.imread(
        str(prediction_path), cv2.IMREAD_GRAYSCALE
    )

    ground_truth = cv2.imread(
        str(ground_truth_path), cv2.IMREAD_GRAYSCALE
    )

    if prediction is None or ground_truth is None:
        print("Could not read prediction or ground truth.")
        return None

    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Image size mismatch: {prediction.shape} "
            f"and {ground_truth.shape}"
        )

    pred = prediction > 127
    truth = ground_truth > 127

    tp = int(np.logical_and(pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())

    epsilon = 1e-8

    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)

    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)
    iou = tp / (tp + fp + fn + epsilon)

    # Extract boundaries
    pred_uint8 = pred.astype(np.uint8)
    truth_uint8 = truth.astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)

    pred_boundary = (
        pred_uint8 -
        cv2.erode(pred_uint8, kernel)
    ).astype(bool)

    truth_boundary = (
        truth_uint8 -
        cv2.erode(truth_uint8, kernel)
    ).astype(bool)

    if pred_boundary.any() and truth_boundary.any():

        # Distance maps to the nearest boundary pixel
        distance_to_truth = distance_transform_edt(
            ~truth_boundary
        )

        distance_to_pred = distance_transform_edt(
            ~pred_boundary
        )

        pred_to_truth = distance_to_truth[pred_boundary]
        truth_to_pred = distance_to_pred[truth_boundary]

        hausdorff = max(
            float(pred_to_truth.max()),
            float(truth_to_pred.max())
        )

        average_surface_distance = (
            float(pred_to_truth.mean()) +
            float(truth_to_pred.mean())
        ) / 2

    else:
        # Boundary distance is undefined if one or both
        # masks contain no boundary pixels.
        hausdorff = float("nan")
        average_surface_distance = float("nan")

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "dice": dice,
        "iou": iou,
        "pixel_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "hausdorff_distance_px": hausdorff,
        "average_surface_distance_px": average_surface_distance
    }


def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_name, folder in MODEL_FOLDERS.items():

        print(f"\nEvaluating: {model_name}")

        if not folder.exists():
            print(f"Folder not found: {folder}")
            continue

        prediction_files = sorted(
            folder.glob("prediction_*.png")
        )

        for prediction_path in prediction_files:

            image_number = prediction_path.stem.replace(
                "prediction_", ""
            )

            ground_truth_path = (
                folder / f"ground_truth_{image_number}.png"
            )

            if not ground_truth_path.exists():
                print(
                    f"Ground truth missing for "
                    f"{prediction_path.name}"
                )
                continue

            metrics = calculate_metrics(
                prediction_path,
                ground_truth_path
            )

            if metrics is None:
                continue

            result = {
                "model": model_name,
                "image": prediction_path.name,
                **metrics
            }

            all_results.append(result)

            print(
                f"{prediction_path.name}: "
                f"Dice={metrics['dice']:.4f}, "
                f"IoU={metrics['iou']:.4f}, "
                f"Accuracy={metrics['pixel_accuracy']:.4f}"
            )

    if not all_results:
        print("\nNo matching prediction and ground-truth pairs found.")
        return

    csv_path = REPORT_DIR / "evaluation_metrics.csv"

    fieldnames = list(all_results[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print("\n--------------------------------")
    print(f"Images evaluated: {len(all_results)}")
    print(f"Evaluation report: {csv_path}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
