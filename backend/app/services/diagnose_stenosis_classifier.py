
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

IMAGE_SIZE = 128
BATCH_SIZE = 16

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "stenosis_classification_dataset"
)

RESULTS_DIR = PROJECT_ROOT / "backend" / "results"

MODEL_PATH = RESULTS_DIR / "stenosis_classifier_cnn.pth"

REPORT_PATH = RESULTS_DIR / "classifier_diagnostic_report.json"
CSV_PATH = RESULTS_DIR / "classifier_prediction_probabilities.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# 2. Dataset
# --------------------------------------------------

class StenosisDataset(Dataset):

    def __init__(self, split):

        self.samples = []

        csv_path = DATASET_ROOT / f"{split}_labels.csv"

        with open(csv_path, "r", newline="") as file:

            reader = csv.DictReader(file)

            for row in reader:

                label = int(row["label"])

                folder = "positive" if label == 1 else "negative"

                image_path = (
                    DATASET_ROOT
                    / split
                    / folder
                    / row["filename"]
                )

                if image_path.exists():
                    self.samples.append(
                        (image_path, label, row["filename"])
                    )

        if not self.samples:
            raise ValueError(f"No samples found for {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label, filename = self.samples[index]

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise ValueError(f"Cannot read {image_path}")

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        image = image.astype(np.float32) / 255.0

        image = torch.tensor(image).unsqueeze(0)

        return image, label, filename


# --------------------------------------------------
# 3. Same CNN architecture as training
# --------------------------------------------------

class StenosisCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x.squeeze(1)


# --------------------------------------------------
# 4. Get prediction probabilities
# --------------------------------------------------

def get_predictions(model, loader, split):

    model.eval()

    results = []

    with torch.no_grad():

        for images, labels, filenames in loader:

            images = images.to(DEVICE)

            logits = model(images)

            probabilities = torch.sigmoid(logits)

            for i in range(len(labels)):

                results.append({
                    "split": split,
                    "filename": filenames[i],
                    "actual_label": int(labels[i]),
                    "probability": float(probabilities[i].cpu())
                })

    return results


# --------------------------------------------------
# 5. Metrics for a selected threshold
# --------------------------------------------------

def calculate_metrics(results, threshold):

    y_true = np.array([
        item["actual_label"] for item in results
    ])

    probabilities = np.array([
        item["probability"] for item in results
    ])

    y_pred = (probabilities >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-8)
    )

    specificity = tn / max(tn + fp, 1)

    accuracy = (tp + tn) / max(len(y_true), 1)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }


# --------------------------------------------------
# 6. Probability distribution
# --------------------------------------------------

def probability_summary(results):

    summary = {}

    for label, class_name in [
        (1, "positive"),
        (0, "negative")
    ]:

        values = np.array([
            item["probability"]
            for item in results
            if item["actual_label"] == label
        ])

        summary[class_name] = {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "25_percentile": float(np.percentile(values, 25)),
            "75_percentile": float(np.percentile(values, 75))
        }

    return summary


# --------------------------------------------------
# 7. Main diagnostic
# --------------------------------------------------

def main():

    print("Device:", DEVICE)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    model = StenosisCNN().to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    )

    print("Existing classifier checkpoint loaded.")

    val_dataset = StenosisDataset("val")
    test_dataset = StenosisDataset("test")

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    val_results = get_predictions(
        model, val_loader, "val"
    )

    test_results = get_predictions(
        model, test_loader, "test"
    )

    print("\n--- VALIDATION PROBABILITIES ---")

    val_summary = probability_summary(val_results)

    print(json.dumps(val_summary, indent=4))

    print("\n--- TEST PROBABILITIES ---")

    test_summary = probability_summary(test_results)

    print(json.dumps(test_summary, indent=4))

    # Evaluate existing threshold
    print("\n--- RESULTS AT THRESHOLD 0.50 ---")

    print(
        "Validation:",
        calculate_metrics(val_results, 0.50)
    )

    print(
        "Test:",
        calculate_metrics(test_results, 0.50)
    )

    # Find best threshold using validation only
    threshold_results = []

    for threshold in np.arange(0.05, 1.00, 0.05):

        metrics = calculate_metrics(
            val_results,
            float(threshold)
        )

        threshold_results.append(metrics)

    best_result = max(
        threshold_results,
        key=lambda item: (
            item["f1_score"],
            item["specificity"]
        )
    )

    best_threshold = best_result["threshold"]

    print("\n--- BEST VALIDATION THRESHOLD ---")
    print(json.dumps(best_result, indent=4))

    # Apply validation-selected threshold to test
    test_at_best_threshold = calculate_metrics(
        test_results,
        best_threshold
    )

    print("\n--- TEST RESULTS AT VALIDATION-SELECTED THRESHOLD ---")
    print(json.dumps(test_at_best_threshold, indent=4))

    # Save all probabilities
    all_results = val_results + test_results

    with open(CSV_PATH, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "filename",
                "actual_label",
                "probability"
            ]
        )

        writer.writeheader()
        writer.writerows(all_results)

    report = {
        "validation_probability_summary": val_summary,
        "test_probability_summary": test_summary,
        "best_validation_threshold": best_threshold,
        "best_validation_metrics": best_result,
        "test_metrics_at_0.50": calculate_metrics(
            test_results, 0.50
        ),
        "test_metrics_at_validation_threshold":
            test_at_best_threshold,
        "threshold_analysis": threshold_results
    }

    with open(REPORT_PATH, "w") as file:

        json.dump(report, file, indent=4)

    print("\nDiagnostic completed.")

    print("Report:", REPORT_PATH)
    print("Probabilities CSV:", CSV_PATH)


if __name__ == "__main__":
    main()
