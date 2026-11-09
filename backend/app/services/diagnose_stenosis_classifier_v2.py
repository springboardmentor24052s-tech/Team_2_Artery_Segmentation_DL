
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

IMAGE_SIZE = 128
THRESHOLD = 0.5

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "stenosis_classification_dataset_v2"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "classifier_v2_error_inspection"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "stenosis_classifier_cnn_v2.pth"
)

CSV_PATH = DATASET_ROOT / "test_labels.csv"

OUTPUT_CSV = RESULTS_DIR / "test_predictions_v2.csv"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("V2 CLASSIFIER TEST PREDICTION DIAGNOSTICS")
print("=" * 60)

print("Device:", DEVICE)


# --------------------------------------------------
# 2. CNN architecture
# Must match the trained model exactly
# --------------------------------------------------

class StenosisCNN(nn.Module):

    def __init__(self):

        super(StenosisCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
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
# 3. Load trained V2 model
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )

model = StenosisCNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

print("V2 model loaded successfully.")


# --------------------------------------------------
# 4. Read test manifest
# --------------------------------------------------

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Test manifest not found: {CSV_PATH}"
    )

samples = []

with open(CSV_PATH, "r", newline="") as file:

    reader = csv.DictReader(file)

    for row in reader:

        filename = row["filename"]
        actual_label = int(row["label"])

        class_folder = (
            "positive" if actual_label == 1 else "negative"
        )

        image_path = (
            DATASET_ROOT
            / "test"
            / class_folder
            / filename
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing image: {image_path}"
            )

        samples.append({
            "filename": filename,
            "actual_label": actual_label,
            "image_path": image_path,
            "negative_type": row.get("negative_type", ""),
            "vessel_response_score": row.get(
                "vessel_response_score", ""
            )
        })

print("Test samples loaded:", len(samples))


# --------------------------------------------------
# 5. Generate predictions
# --------------------------------------------------

results = []

with torch.no_grad():

    for index, sample in enumerate(samples):

        image = cv2.imread(
            str(sample["image_path"]),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise ValueError(
                f"Unable to read: {sample['image_path']}"
            )

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        image = image.astype(np.float32) / 255.0

        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        image_tensor = image_tensor.to(DEVICE)

        logits = model(image_tensor)

        probability = torch.sigmoid(logits).item()

        predicted_label = (
            1 if probability >= THRESHOLD else 0
        )

        actual_label = sample["actual_label"]

        if actual_label == 0 and predicted_label == 1:
            result_type = "FALSE_POSITIVE"

        elif actual_label == 1 and predicted_label == 0:
            result_type = "FALSE_NEGATIVE"

        elif actual_label == 1 and predicted_label == 1:
            result_type = "TRUE_POSITIVE"

        else:
            result_type = "TRUE_NEGATIVE"

        results.append({
            "filename": sample["filename"],
            "actual_label": actual_label,
            "predicted_probability": probability,
            "predicted_label": predicted_label,
            "result": result_type,
            "negative_type": sample["negative_type"],
            "vessel_response_score": sample[
                "vessel_response_score"
            ]
        })

        if (index + 1) % 100 == 0:
            print(
                f"Processed {index + 1}/{len(samples)}"
            )


# --------------------------------------------------
# 6. Calculate confusion matrix
# --------------------------------------------------

tp = sum(
    r["result"] == "TRUE_POSITIVE" for r in results
)

tn = sum(
    r["result"] == "TRUE_NEGATIVE" for r in results
)

fp = sum(
    r["result"] == "FALSE_POSITIVE" for r in results
)

fn = sum(
    r["result"] == "FALSE_NEGATIVE" for r in results
)


# --------------------------------------------------
# 7. Save prediction CSV
# --------------------------------------------------

fieldnames = [
    "filename",
    "actual_label",
    "predicted_probability",
    "predicted_label",
    "result",
    "negative_type",
    "vessel_response_score"
]

with open(OUTPUT_CSV, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


# --------------------------------------------------
# 8. Print summary
# --------------------------------------------------

print("\n========== V2 TEST DIAGNOSTICS ==========")

print("Total samples:", len(results))
print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)

print("\nPrediction distribution:")

print(
    "Predicted Positive:",
    sum(r["predicted_label"] == 1 for r in results)
)

print(
    "Predicted Negative:",
    sum(r["predicted_label"] == 0 for r in results)
)

print("\nFalse positive count:", fp)
print("False negative count:", fn)

print("\nPrediction CSV saved at:")
print(OUTPUT_CSV)

print("\nV2 diagnostic prediction generation completed.")
