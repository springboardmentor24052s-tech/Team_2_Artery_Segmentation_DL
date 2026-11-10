
from pathlib import Path
import shutil

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from unet_model import UNet


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = ROOT / "data" / "processed" / "stenosis"
MODEL_DIR = ROOT / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Google Drive checkpoint folder (used when Drive is mounted)
DRIVE_ROOT = Path("/content/drive/MyDrive")

DRIVE_MODEL_DIR = (
    DRIVE_ROOT / "Artery_Project_Checkpoints"
    if DRIVE_ROOT.exists()
    else None
)

if DRIVE_MODEL_DIR is not None:
    DRIVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)


IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 100

LEARNING_RATE = 1e-4

# Stop after 15 consecutive epochs without significant
# validation-loss improvement.
PATIENCE = 15
MIN_DELTA = 0.001

# Reduce learning rate when validation loss plateaus.
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

        self.image_paths = sorted(
            self.image_dir.glob("*.png")
        )

        valid_pairs = []

        for image_path in self.image_paths:
            mask_path = self.mask_dir / image_path.name

            if mask_path.exists():
                valid_pairs.append((image_path, mask_path))

        self.image_paths = [pair[0] for pair in valid_pairs]
        self.mask_paths = [pair[1] for pair in valid_pairs]

        if not self.image_paths:
            raise RuntimeError(
                f"No image-mask pairs found for {split}\n"
                f"Image folder: {self.image_dir}\n"
                f"Mask folder: {self.mask_dir}"
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
                f"Could not read pair: {self.image_paths[index].name}"
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
# 4. TRAINING / VALIDATION EPOCH
# --------------------------------------------------

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

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
# 5. CHECKPOINT SAVING
# --------------------------------------------------

def copy_to_drive(filename):
    """Copy a saved checkpoint to Drive when it is mounted."""
    if DRIVE_MODEL_DIR is None:
        return

    source = MODEL_DIR / filename

    if source.exists():
        shutil.copy2(
            source,
            DRIVE_MODEL_DIR / filename
        )


# --------------------------------------------------
# 6. MAIN TRAINING LOOP
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

    model = UNet().to(DEVICE)

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
    print("U-NET TRAINING")
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

    if DRIVE_MODEL_DIR is not None:
        print("Drive checkpoints:", DRIVE_MODEL_DIR)
    else:
        print("Google Drive not mounted; checkpoints are local only.")

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

        # Update learning rate based on validation loss
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

        # Save latest model weights locally every epoch
        torch.save(
            model.state_dict(),
            MODEL_DIR / "unet_latest.pth"
        )

        # Track the lowest validation loss and save best weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                MODEL_DIR / "unet_best.pth"
            )

            print(
                f"Best model saved. "
                f"Validation Loss: {best_val_loss:.4f}"
            )

            # Persist every new best checkpoint to Drive
            copy_to_drive("unet_best.pth")

        # Early stopping: require a significant improvement
        if val_loss < early_stop_best - MIN_DELTA:
            early_stop_best = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"No significant val-loss improvement: "
            f"{epochs_without_improvement}/{PATIENCE}"
        )

        # Persist the latest weights to Drive every 5 epochs
        # and at the end of training.
        if (
            (epoch + 1) % 5 == 0
            or epoch + 1 == EPOCHS
        ):
            copy_to_drive("unet_latest.pth")

        if epochs_without_improvement >= PATIENCE:
            print(
                f"\nEarly stopping triggered at epoch {epoch + 1}."
            )
            print(
                f"No significant validation-loss improvement "
                f"for {PATIENCE} consecutive epochs."
            )
            break

    # Save the final weights to Drive too
    copy_to_drive("unet_latest.pth")

    print("\n" + "=" * 60)
    print("U-NET TRAINING FINISHED")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Best model:", MODEL_DIR / "unet_best.pth")
    print("Latest model:", MODEL_DIR / "unet_latest.pth")

    if DRIVE_MODEL_DIR is not None:
        print("Drive copies:", DRIVE_MODEL_DIR)

    print("=" * 60)


if __name__ == "__main__":
    main()
