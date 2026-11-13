
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
THRESHOLD = 0.5

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "stenosis_classification_dataset_v2"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "stenosis_classifier_cnn_v2.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "classifier_v2_error_inspection"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = OUTPUT_DIR / "validation_predictions_v2.csv"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# 2. Dataset
# --------------------------------------------------

class ValidationDataset(Dataset):

    def __init__(self):
        self.samples = []

        csv_path = DATASET_ROOT / "val_labels.csv"

        with open(csv_path, "r", newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                filename = row["filename"]
                label = int(row["label"])
                source_image = row["source_image"]

                class_folder = (
                    "positive" if label == 1 else "negative"
                )

                image_path = (
                    DATASET_ROOT
                    / "val"
                    / class_folder
                    / filename
                )

                if not image_path.exists():
                    raise FileNotFoundError(image_path)

                self.samples.append({
                    "image_path": image_path,
                    "filename": filename,
                    "source_image": source_image,
                    "label": label
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        image = cv2.imread(
            str(sample["image_path"]),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise ValueError(
                f"Cannot read {sample['image_path']}"
            )

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        image = image.astype(np.float32) / 255.0

        image = torch.from_numpy(image).unsqueeze(0)

        return image, sample["label"], index


# --------------------------------------------------
# 3. Same CNN architecture used during training
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
# 4. Load model and generate predictions
# --------------------------------------------------

print("=" * 60)
print("V2 VALIDATION PREDICTION GENERATION")
print("=" * 60)

dataset = ValidationDataset()

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
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

probabilities = np.zeros(len(dataset), dtype=np.float32)

with torch.no_grad():

    for images, labels, indices in loader:

        images = images.to(DEVICE)

        outputs = model(images)

        batch_probabilities = torch.sigmoid(outputs)

        for i, index in enumerate(indices.tolist()):
            probabilities[index] = (
                batch_probabilities[i].item()
            )


# --------------------------------------------------
# 5. Save individual predictions
# --------------------------------------------------

with open(PREDICTIONS_PATH, "w", newline="") as file:

    fieldnames = [
        "filename",
        "source_image",
        "label",
        "probability",
        "predicted_label",
        "correct"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for index, sample in enumerate(dataset.samples):

        actual = sample["label"]
        probability = float(probabilities[index])
        predicted = int(probability >= THRESHOLD)

        writer.writerow({
            "filename": sample["filename"],
            "source_image": sample["source_image"],
            "label": actual,
            "probability": probability,
            "predicted_label": predicted,
            "correct": int(actual == predicted)
        })


# --------------------------------------------------
# 6. Calculate metrics before and after exclusion
# --------------------------------------------------

def calculate_metrics(rows):

    y_true = np.array(
        [int(row["label"]) for row in rows]
    )

    y_pred = np.array(
        [int(row["predicted_label"]) for row in rows]
    )

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(len(rows), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-8)
    )

    return {
        "samples": len(rows),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "f1_score": f1,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }


with open(PREDICTIONS_PATH, "r", newline="") as file:
    rows = list(csv.DictReader(file))


# The exact duplicate is:
# Training image: 218.png
# Validation image: 58.png

all_metrics = calculate_metrics(rows)

clean_rows = [
    row for row in rows
    if row["source_image"] != "58.png"
]

clean_metrics = calculate_metrics(clean_rows)

print("\n========== ORIGINAL VALIDATION ==========")
print(json.dumps(all_metrics, indent=4))

print("\n========== VALIDATION WITHOUT DUPLICATE SOURCE ==========")
print(json.dumps(clean_metrics, indent=4))

print("\nExcluded patches:", len(rows) - len(clean_rows))
print("Predictions saved at:", PREDICTIONS_PATH)

print("\nEvaluation completed.")
