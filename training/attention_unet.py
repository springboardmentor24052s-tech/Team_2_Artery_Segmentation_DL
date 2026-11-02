
"""
ArteryAI: Attention U-Net for coronary anatomy segmentation.

Reuses the baseline dataset, loss, training and evaluation functions
to ensure a fair comparison between U-Net and Attention U-Net.
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from unet_baseline import (
    ArcadeSyntaxDataset,
    DiceBCELoss,
    DoubleConv,
    run_epoch,
    save_prediction_examples,
    seed_everything,
)


class AttentionGate(nn.Module):
    """
    Learns which spatial features from the encoder
    should be passed to the decoder.
    """

    def __init__(self, gating_channels, skip_channels, intermediate_channels):
        super().__init__()

        self.gating_transform = nn.Sequential(
            nn.Conv2d(
                gating_channels,
                intermediate_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.BatchNorm2d(intermediate_channels),
        )

        self.skip_transform = nn.Sequential(
            nn.Conv2d(
                skip_channels,
                intermediate_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.BatchNorm2d(intermediate_channels),
        )

        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                intermediate_channels,
                1,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, gating_signal, skip_features):
        gating_features = self.gating_transform(gating_signal)
        skip_features_transformed = self.skip_transform(skip_features)

        combined = gating_features + skip_features_transformed
        attention_weights = self.attention(combined)

        return skip_features * attention_weights


class AttentionUNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        base_channels=8,
    ):
        super().__init__()

        c = base_channels

        # Encoder
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(c * 8, c * 16)

        # Decoder upsampling
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)

        # Attention gates
        self.attention4 = AttentionGate(c * 8, c * 8, c * 4)
        self.attention3 = AttentionGate(c * 4, c * 4, c * 2)
        self.attention2 = AttentionGate(c * 2, c * 2, c)
        self.attention1 = AttentionGate(c, c, max(c // 2, 1))

        # Decoder convolution blocks
        self.dec4 = DoubleConv(c * 16, c * 8)
        self.dec3 = DoubleConv(c * 8, c * 4)
        self.dec2 = DoubleConv(c * 4, c * 2)
        self.dec1 = DoubleConv(c * 2, c)

        self.output = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder + attention
        d4 = self.up4(b)
        a4 = self.attention4(d4, e4)
        d4 = self.dec4(torch.cat([d4, a4], dim=1))

        d3 = self.up3(d4)
        a3 = self.attention3(d3, e3)
        d3 = self.dec3(torch.cat([d3, a3], dim=1))

        d2 = self.up2(d3)
        a2 = self.attention2(d2, e2)
        d2 = self.dec2(torch.cat([d2, a2], dim=1))

        d1 = self.up1(d2)
        a1 = self.attention1(d1, e1)
        d1 = self.dec1(torch.cat([d1, a1], dim=1))

        return self.output(d1)


def main():

    parser = argparse.ArgumentParser(
        description="Train and evaluate ArteryAI Attention U-Net."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "dataset",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "attention_unet_5epochs",
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        parser.error("epochs/batch-size must be positive and image-size >= 32.")

    if args.image_size % 16 != 0:
        parser.error("image-size must be divisible by 16.")

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cpu":
        print("CPU training selected. Training may take some time.")

    # Load the same dataset splits as the baseline.
    train_dataset = ArcadeSyntaxDataset(
        args.dataset_root, "train", args.image_size
    )
    val_dataset = ArcadeSyntaxDataset(
        args.dataset_root, "val", args.image_size
    )
    test_dataset = ArcadeSyntaxDataset(
        args.dataset_root, "test", args.image_size
    )

    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }

    train_loader = DataLoader(
        train_dataset, shuffle=True, **loader_args
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, **loader_args
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, **loader_args
    )

    model = AttentionUNet(
        base_channels=args.base_channels
    ).to(device)

    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_val_dice = -1.0
    history = []

    # Training and validation
    for epoch in range(1, args.epochs + 1):

        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device, True
        )

        val_metrics = run_epoch(
            model, val_loader, criterion, None, device, False
        )

        row = {"epoch": epoch}
        row.update({
            f"train_{key}": value
            for key, value in train_metrics.items()
        })
        row.update({
            f"val_{key}": value
            for key, value in val_metrics.items()
        })

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train loss: {train_metrics['loss']:.4f}, "
            f"Dice: {train_metrics['dice']:.4f}, "
            f"IoU: {train_metrics['iou']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f}, "
            f"Dice: {val_metrics['dice']:.4f}, "
            f"IoU: {val_metrics['iou']:.4f}"
        )

        # Save the best validation checkpoint.
        if val_metrics["dice"] > best_val_dice:

            best_val_dice = val_metrics["dice"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": args.image_size,
                    "base_channels": args.base_channels,
                    "epoch": epoch,
                    "validation_metrics": val_metrics,
                },
                args.output_dir / "best_model.pth",
            )

    # Evaluate the best model on the held-out test set.
    checkpoint = torch.load(
        args.output_dir / "best_model.pth",
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_epoch(
        model, test_loader, criterion, None, device, False
    )

    # Save training history.
    with (
        args.output_dir / "training_history.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    with (
        args.output_dir / "test_metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(test_metrics, file, indent=2)

    with (
        args.output_dir / "training_history.csv"
    ).open("w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)

    # Save prediction examples.
    save_prediction_examples(
        model,
        test_dataset,
        device,
        args.output_dir / "prediction_examples",
    )

    print("\nBest checkpoint epoch:", checkpoint["epoch"])

    print(
        "ATTENTION U-NET TEST | "
        f"Loss: {test_metrics['loss']:.4f}, "
        f"Dice: {test_metrics['dice']:.4f}, "
        f"IoU: {test_metrics['iou']:.4f}"
    )

    print(f"Outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
