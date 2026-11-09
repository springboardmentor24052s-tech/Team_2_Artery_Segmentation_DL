
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 1. CONFIGURATION

ROOT = Path("backend/app/data/arcade/arcade/arcade")
MASK_ROOT = Path("backend/results/arcade_masks")

IMAGE_DIR = ROOT / "syntax/train/images"
MASK_DIR = MASK_ROOT / "syntax/train"

DEVICE = torch.device("cpu")
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 0.0003

OUTPUT_PATH = Path(
    "backend/results/unet_syntax_improved.pth"
)

torch.manual_seed(42)


# 2. DATASET

class ArcadeDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.image_paths = sorted(
            self.image_dir.glob("*.png")
        )

        self.image_paths = [
            path for path in self.image_paths
            if (
                self.mask_dir /
                f"{path.stem}_mask.png"
            ).exists()
        ]

        if not self.image_paths:
            raise RuntimeError(
                f"No image-mask pairs found in {self.image_dir}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = (
            self.mask_dir /
            f"{image_path.stem}_mask.png"
        )

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
                f"Could not load {image_path.name}"
            )

        if image.shape != mask.shape:
            raise RuntimeError(
                f"Image-mask dimensions differ: {image_path.name}"
            )

        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


# 3. U-NET BUILDING BLOCK

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


# 4. U-NET MODEL

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(1, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up3 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(
            64, 32, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(64, 32)

        self.up1 = nn.ConvTranspose2d(
            32, 16, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(32, 16)

        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.output(d1)


# 5. COMBINED BCE + DICE LOSS

def combined_loss(logits, targets):
    probabilities = torch.sigmoid(logits)

    # BCE penalizes incorrect pixel classifications.
    # A positive weight gives artery pixels more importance.
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=torch.tensor(
            [5.0],
            device=logits.device
        )
    )

    intersection = (
        probabilities * targets
    ).sum(dim=(1, 2, 3))

    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice_score = (
        2.0 * intersection + 1.0
    ) / (
        denominator + 1.0
    )

    dice = 1.0 - dice_score.mean()

    return 0.5 * bce + 0.5 * dice


# 6. TRAINING

def main():
    print("Device:", DEVICE)
    print("Loading ARCADE syntax training dataset...")

    dataset = ArcadeDataset(IMAGE_DIR, MASK_DIR)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    print("Training image-mask pairs:", len(dataset))
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Loss: BCE + Dice")

    model = UNet().to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_index, (images, masks) in enumerate(loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)
            loss = combined_loss(logits, masks)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_index + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {batch_index + 1}/{len(loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        average_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} completed | "
            f"Average loss: {average_loss:.4f}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(model.state_dict(), OUTPUT_PATH)

    print("\nTraining completed.")
    print("Improved model saved to:", OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
