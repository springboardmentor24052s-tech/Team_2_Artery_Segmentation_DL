
from pathlib import Path
import shutil

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from attention_unet_model import AttentionUNet



# --------------------------------------------------
# 1. CONFIGURATION AND PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = ROOT / "data" / "processed" / "stenosis"
MODEL_DIR = ROOT / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DRIVE_MODEL_DIR = (
    Path("/content/drive/MyDrive/Artery_Project_Checkpoints")
)

IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4

PATIENCE = 15
MIN_DELTA = 0.001

LR_PATIENCE = 5
LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# 2. DATASET
# --------------------------------------------------

class StenosisDataset(Dataset):

    def __init__(self, split):
        self.image_dir = DATA_ROOT / split / "images"
        self.mask_dir = DATA_ROOT / split / "masks"

        image_paths = sorted(self.image_dir.glob("*.png"))

        valid_pairs = []

        for image_path in image_paths:
            mask_path = self.mask_dir / image_path.name

            if mask_path.exists():
                valid_pairs.append((image_path, mask_path))

        self.image_paths = [p[0] for p in valid_pairs]
        self.mask_paths = [p[1] for p in valid_pairs]

        if not self.image_paths:
            raise RuntimeError(
                f"No image-mask pairs found for {split}: "
                f"{self.image_dir}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(
            str(self.image_paths[index]),
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.imread(
            str(self.mask_paths[index]),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None or mask is None:
            raise RuntimeError(
                f"Could not read image-mask pair: "
                f"{self.image_paths[index].name}"
            )

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        mask = cv2.resize(
            mask,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


# --------------------------------------------------
# 3. DICE LOSS
# --------------------------------------------------

def dice_loss(logits, masks, smooth=1.0):
    probabilities = torch.sigmoid(logits)

    intersection = (
        probabilities * masks
    ).sum(dim=(1, 2, 3))

    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + masks.sum(dim=(1, 2, 3))
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    return 1.0 - dice.mean()


# --------------------------------------------------
# 4. TRAIN / VALIDATION
# --------------------------------------------------

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)

            loss = (
                criterion(logits, masks)
                + dice_loss(logits, masks)
            )

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


# --------------------------------------------------
# 5. DRIVE CHECKPOINT SAVING
# --------------------------------------------------

def save_checkpoint(filename):
    source = MODEL_DIR / filename

    if not source.exists():
        return

    if not DRIVE_MODEL_DIR.exists():
        print(
            "WARNING: Google Drive checkpoint folder "
            "is unavailable. Check Drive mounting."
        )
        return

    destination = DRIVE_MODEL_DIR / filename

    try:
        shutil.copy2(source, destination)
        print("Drive checkpoint saved:", filename)
    except OSError as error:
        print("WARNING: Could not copy checkpoint to Drive:", error)


# --------------------------------------------------
# 6. MAIN TRAINING
# --------------------------------------------------

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(42)

    train_ds = StenosisDataset("train")
    val_ds = StenosisDataset("val")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    model = AttentionUNet().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LEARNING_RATE
    )

    best_val_loss = float("inf")
    early_stop_best = float("inf")
    epochs_without_improvement = 0

    print("=" * 60)
    print("ATTENTION U-NET TRAINING")
    print("=" * 60)
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Training images:", len(train_ds))
    print("Validation images:", len(val_ds))
    print("Batch size:", BATCH_SIZE)
    print("Maximum epochs:", EPOCHS)
    print("Early stopping patience:", PATIENCE)
    print("Minimum improvement:", MIN_DELTA)
    print("Initial learning rate:", LEARNING_RATE)
    print("Drive checkpoints:", DRIVE_MODEL_DIR)
    print("=" * 60)

    for epoch in range(EPOCHS):

        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss = run_epoch(
            model,
            val_loader,
            criterion
        )

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if current_lr < old_lr:
            print("Learning rate reduced.")

        # Save latest local checkpoint every epoch
        torch.save(
            model.state_dict(),
            MODEL_DIR / "attention_unet_latest.pth"
        )

        # Save best model whenever validation loss decreases
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                MODEL_DIR / "attention_unet_best.pth"
            )

            print(
                f"Best model saved. "
                f"Validation Loss: {best_val_loss:.4f}"
            )

            save_checkpoint("attention_unet_best.pth")

        # Count epochs without significant improvement
        if val_loss < early_stop_best - MIN_DELTA:
            early_stop_best = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            "No significant validation-loss improvement: "
            f"{epochs_without_improvement}/{PATIENCE}"
        )

        # Save latest checkpoint to Drive every epoch
        save_checkpoint("attention_unet_latest.pth")

        if epochs_without_improvement >= PATIENCE:
            print(
                f"\nEarly stopping triggered at epoch {epoch + 1}."
            )
            print(
                f"Validation loss did not improve significantly "
                f"for {PATIENCE} consecutive epochs."
            )
            break

    print("\n" + "=" * 60)
    print("ATTENTION U-NET TRAINING FINISHED")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Best checkpoint:", MODEL_DIR / "attention_unet_best.pth")
    print("Latest checkpoint:", MODEL_DIR / "attention_unet_latest.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()
