
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from attention_unet import AttentionUNet


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ROOT = Path("backend/app/data/arcade/arcade/arcade")
MASK_ROOT = Path("backend/results/arcade_masks")

TRAIN_JSON = ROOT / "stenosis/train/annotations/train.json"
VAL_JSON = ROOT / "stenosis/val/annotations/val.json"

TRAIN_IMAGES = ROOT / "stenosis/train/images"
VAL_IMAGES = ROOT / "stenosis/val/images"

TRAIN_MASKS = MASK_ROOT / "stenosis/train"
VAL_MASKS = MASK_ROOT / "stenosis/val"

MODEL_PATH = Path(
    "backend/results/attention_unet_stenosis_v2.pth"
)

RESULTS_PATH = Path(
    "backend/results/attention_unet_v2_training.json"
)

DEVICE = torch.device("cpu")

EPOCHS = 20
BATCH_SIZE = 2
LEARNING_RATE = 0.0003
POSITIVE_WEIGHT = 10.0

ALPHA = 0.7
BETA = 0.3
THRESHOLD = 0.5


# --------------------------------------------------
# 2. DATASET
# --------------------------------------------------

class StenosisDataset(Dataset):

    def __init__(self, annotation_file, image_dir, mask_dir):

        with annotation_file.open(
            "r", encoding="utf-8"
        ) as file:
            data = json.load(file)

        stenosis_ids = {
            category["id"]
            for category in data["categories"]
            if category["name"].lower() == "stenosis"
        }

        annotated_ids = {
            annotation["image_id"]
            for annotation in data["annotations"]
            if annotation["category_id"] in stenosis_ids
        }

        self.samples = []

        for info in data["images"]:

            if info["id"] not in annotated_ids:
                continue

            image_path = image_dir / info["file_name"]

            mask_path = (
                mask_dir /
                f"{Path(info['file_name']).stem}_mask.png"
            )

            if image_path.exists() and mask_path.exists():
                self.samples.append(
                    (image_path, mask_path)
                )

        if not self.samples:
            raise RuntimeError(
                f"No image-mask pairs found in {annotation_file}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, mask_path = self.samples[index]

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None or mask is None:
            raise RuntimeError(
                f"Unable to read {image_path.name}"
            )

        if image.shape != mask.shape:
            raise RuntimeError(
                f"Image-mask dimensions mismatch: {image_path.name}"
            )

        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        return (
            torch.from_numpy(image).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0)
        )


# --------------------------------------------------
# 3. TVERSKY LOSS
# --------------------------------------------------

def tversky_loss(predictions, targets):

    predictions = torch.sigmoid(predictions)

    true_positive = (predictions * targets).sum(
        dim=(1, 2, 3)
    )

    false_positive = (
        predictions * (1 - targets)
    ).sum(dim=(1, 2, 3))

    false_negative = (
        (1 - predictions) * targets
    ).sum(dim=(1, 2, 3))

    tversky = (
        true_positive + 1e-7
    ) / (
        true_positive
        + ALPHA * false_positive
        + BETA * false_negative
        + 1e-7
    )

    return (1 - tversky).mean()


# --------------------------------------------------
# 4. COMBINED LOSS
# --------------------------------------------------

class CombinedLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [POSITIVE_WEIGHT],
                dtype=torch.float32
            )
        )

    def forward(self, predictions, targets):

        bce_loss = self.bce(
            predictions, targets
        )

        tv_loss = tversky_loss(
            predictions, targets
        )

        return 0.5 * bce_loss + 0.5 * tv_loss


# --------------------------------------------------
# 5. VALIDATION METRICS
# --------------------------------------------------

def calculate_metrics(model, loader):

    model.eval()

    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(DEVICE)

            probabilities = torch.sigmoid(
                model(images)
            )

            predictions = probabilities >= THRESHOLD

            targets = masks.to(DEVICE) >= 0.5

            for i in range(len(images)):

                pred = predictions[i, 0]
                target = targets[i, 0]

                tp = (pred & target).sum().item()
                fp = (pred & ~target).sum().item()
                fn = (~pred & target).sum().item()

                dice = (
                    2 * tp / max(2 * tp + fp + fn, 1)
                )

                iou = (
                    tp / max(tp + fp + fn, 1)
                )

                precision = (
                    tp / max(tp + fp, 1)
                )

                recall = (
                    tp / max(tp + fn, 1)
                )

                dice_scores.append(dice)
                iou_scores.append(iou)
                precision_scores.append(precision)
                recall_scores.append(recall)

    return {
        "dice": float(np.mean(dice_scores)),
        "iou": float(np.mean(iou_scores)),
        "precision": float(np.mean(precision_scores)),
        "recall": float(np.mean(recall_scores))
    }


# --------------------------------------------------
# 6. TRAINING
# --------------------------------------------------

def main():

    torch.manual_seed(42)
    np.random.seed(42)

    train_dataset = StenosisDataset(
        TRAIN_JSON,
        TRAIN_IMAGES,
        TRAIN_MASKS
    )

    val_dataset = StenosisDataset(
        VAL_JSON,
        VAL_IMAGES,
        VAL_MASKS
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = AttentionUNet().to(DEVICE)

    criterion = CombinedLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=0.00001
    )

    best_dice = -1.0
    history = []

    print("\nATTENTION U-NET V2 TRAINING")
    print("---------------------------")
    print("Device:", DEVICE)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Initial learning rate:", LEARNING_RATE)
    print("Positive weight:", POSITIVE_WEIGHT)
    print("Tversky alpha:", ALPHA)
    print("Tversky beta:", BETA)
    print("Threshold:", THRESHOLD)

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for images, masks in train_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, masks)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(train_loader)
        )

        metrics = calculate_metrics(
            model, val_loader
        )

        scheduler.step(metrics["dice"])

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {average_loss:.4f} | "
            f"Dice: {metrics['dice']:.4f} | "
            f"IoU: {metrics['iou']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        history.append({
            "epoch": epoch + 1,
            "training_loss": average_loss,
            "learning_rate": current_lr,
            **metrics
        })

        if metrics["dice"] > best_dice:

            best_dice = metrics["dice"]

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print("  Best model checkpoint saved!")

    RESULTS_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    with RESULTS_PATH.open(
        "w", encoding="utf-8"
    ) as file:

        json.dump(
            {
                "model": "Attention U-Net V2",
                "epochs": EPOCHS,
                "best_validation_dice": best_dice,
                "history": history
            },
            file,
            indent=4
        )

    print("\nTRAINING COMPLETED")
    print("------------------")
    print("Best validation Dice:", round(best_dice, 4))
    print("Checkpoint:", MODEL_PATH.resolve())
    print("Training report:", RESULTS_PATH.resolve())


if __name__ == "__main__":
    main()
