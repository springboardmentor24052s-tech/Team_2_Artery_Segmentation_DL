
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001
PATIENCE = 5
CLASSIFICATION_THRESHOLD = 0.5

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Use Dataset V2
DATASET_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "stenosis_classification_dataset_v2"
)

RESULTS_DIR = PROJECT_ROOT / "backend" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Separate V2 files so V1 is preserved
MODEL_PATH = RESULTS_DIR / "stenosis_classifier_cnn_v2.pth"
REPORT_PATH = RESULTS_DIR / "stenosis_classifier_report_v2.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 60)
print("STENOSIS CLASSIFIER - DATASET V2")
print("=" * 60)
print("Device:", DEVICE)
print("Dataset:", DATASET_ROOT)


# --------------------------------------------------
# 2. Dataset loader
# --------------------------------------------------

class StenosisDataset(Dataset):

    def __init__(self, split):

        self.split = split
        self.samples = []

        csv_path = DATASET_ROOT / f"{split}_labels.csv"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset CSV not found: {csv_path}"
            )

        with open(csv_path, "r", newline="") as file:

            reader = csv.DictReader(file)

            required_columns = {"filename", "label"}

            if not required_columns.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"CSV must contain columns: {required_columns}"
                )

            for row in reader:

                filename = row["filename"]
                label = int(row["label"])

                if label not in (0, 1):
                    raise ValueError(
                        f"Invalid label {label} in {csv_path}"
                    )

                class_folder = (
                    "positive" if label == 1 else "negative"
                )

                image_path = (
                    DATASET_ROOT
                    / split
                    / class_folder
                    / filename
                )

                if image_path.exists():
                    self.samples.append((image_path, label))
                else:
                    raise FileNotFoundError(
                        f"Missing dataset image: {image_path}"
                    )

        if not self.samples:
            raise ValueError(
                f"No samples found in {split} dataset"
            )

        positive = sum(
            label == 1 for _, label in self.samples
        )

        negative = sum(
            label == 0 for _, label in self.samples
        )

        print(
            f"{split.upper()}: {len(self.samples)} samples "
            f"(Positive={positive}, Negative={negative})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        image = image.astype(np.float32) / 255.0

        image = torch.from_numpy(image).unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.float32
        )

        return image, label


# --------------------------------------------------
# 3. CNN architecture
# --------------------------------------------------

class StenosisCNN(nn.Module):

    def __init__(self):

        super(StenosisCNN, self).__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4
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
# 4. Evaluation function
# --------------------------------------------------

def evaluate_model(model, loader, criterion):

    model.eval()

    total_loss = 0.0

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            probabilities = torch.sigmoid(outputs)

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )

    y_true = np.array(all_labels, dtype=np.int32)
    y_prob = np.array(all_probabilities, dtype=np.float32)

    y_pred = (
        y_prob >= CLASSIFICATION_THRESHOLD
    ).astype(np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    precision = tp / max(tp + fp, 1)

    recall = tp / max(tp + fn, 1)

    specificity = tn / max(tn + fp, 1)

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-8)
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2

    metrics = {
        "loss": float(total_loss / len(loader.dataset)),
        "threshold": CLASSIFICATION_THRESHOLD,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "f1_score": float(f1),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn
    }

    return metrics


# --------------------------------------------------
# 5. Load training and validation datasets
# --------------------------------------------------

train_dataset = StenosisDataset("train")
val_dataset = StenosisDataset("val")

# Do not load or use test data during training/model selection.

generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# 6. Initialize model and optimizer
# --------------------------------------------------

model = StenosisCNN().to(DEVICE)

positive_count = sum(
    label == 1 for _, label in train_dataset.samples
)

negative_count = sum(
    label == 0 for _, label in train_dataset.samples
)

pos_weight = torch.tensor(
    [negative_count / max(positive_count, 1)],
    dtype=torch.float32
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# --------------------------------------------------
# 7. Train CNN
# --------------------------------------------------

best_val_f1 = -1.0
epochs_without_improvement = 0

history = []

print("\nStarting Dataset V2 classifier training...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_dataset)

    val_metrics = evaluate_model(
        model,
        val_loader,
        criterion
    )

    scheduler.step(val_metrics["f1_score"])

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val Accuracy: {val_metrics['accuracy']:.4f} | "
        f"Val Precision: {val_metrics['precision']:.4f} | "
        f"Val Recall: {val_metrics['recall']:.4f} | "
        f"Val Specificity: {val_metrics['specificity']:.4f} | "
        f"Val F1: {val_metrics['f1_score']:.4f}"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": float(train_loss),
        **val_metrics
    })

    # Save the best checkpoint using validation F1.
    if val_metrics["f1_score"] > best_val_f1:

        best_val_f1 = val_metrics["f1_score"]

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        epochs_without_improvement = 0

        print("Best V2 model checkpoint saved.")

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


# --------------------------------------------------
# 8. Final test evaluation
# --------------------------------------------------

# Test data is loaded only after model selection is complete.
print("\nTraining completed.")
print("Loading best V2 checkpoint for final test evaluation...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

test_dataset = StenosisDataset("test")

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_metrics = evaluate_model(
    model,
    test_loader,
    criterion
)

print("\n========== FINAL V2 TEST RESULTS ==========")

for metric, value in test_metrics.items():

    if isinstance(value, float):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")


# --------------------------------------------------
# 9. Save evaluation report
# --------------------------------------------------

report = {
    "model": "Stenosis CNN Classifier V2",
    "dataset": "stenosis_classification_dataset_v2",
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs_requested": EPOCHS,
    "epochs_completed": len(history),
    "learning_rate": LEARNING_RATE,
    "classification_threshold": CLASSIFICATION_THRESHOLD,
    "best_validation_f1": best_val_f1,
    "training_samples": len(train_dataset),
    "validation_samples": len(val_dataset),
    "test_samples": len(test_dataset),
    "validation_history": history,
    "test_metrics": test_metrics
}

with open(REPORT_PATH, "w") as file:

    json.dump(report, file, indent=4)

print("\nV2 model saved at:", MODEL_PATH)
print("V2 report saved at:", REPORT_PATH)

print("\nDataset V2 classifier training completed!")
