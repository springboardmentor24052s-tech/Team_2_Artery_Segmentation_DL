
from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

from stenosis_dataset import StenosisDataset, detection_collate_fn


def build_model():
    """Create a Faster R-CNN model for one object class: stenosis."""
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=2,  # 0 = background, 1 = stenosis
        min_size=512,
        max_size=512,
    )
    return model


def move_targets_to_device(targets, device):
    """Move every tensor in each target dictionary to CPU/GPU."""
    moved_targets = []

    for target in targets:
        moved_target = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in target.items()
        }
        moved_targets.append(moved_target)

    return moved_targets


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    processed_images = 0

    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)

        optimizer.zero_grad()

        loss_dict = model(images, targets)
        total_batch_loss = sum(loss for loss in loss_dict.values())

        if not torch.isfinite(total_batch_loss):
            raise RuntimeError(
                f"Non-finite loss detected: {total_batch_loss.item()}"
            )

        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item()
        processed_images += len(images)

    average_loss = total_loss / max(len(loader), 1)

    return average_loss, processed_images


@torch.no_grad()
def validate_model(model, loader, device):
    """Run inference on validation images and count predicted boxes."""
    model.eval()

    processed_images = 0
    predicted_boxes = 0

    for images, _targets in loader:
        images = [image.to(device) for image in images]
        predictions = model(images)

        for prediction in predictions:
            # Count predictions with confidence score >= 0.05.
            confident_boxes = prediction["scores"] >= 0.05
            predicted_boxes += int(confident_boxes.sum().item())

        processed_images += len(images)

    return processed_images, predicted_boxes


def main():
    parser = argparse.ArgumentParser(
        description="Train a Faster R-CNN stenosis detector."
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-images", type=int, default=8)
    parser.add_argument("--max-val-images", type=int, default=4)
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the stenosis datasets.
    train_dataset = StenosisDataset("train")
    val_dataset = StenosisDataset("val")

    # Limit the first run so it is manageable on CPU.
    if args.max_train_images > 0:
        train_dataset = Subset(
            train_dataset,
            range(min(args.max_train_images, len(train_dataset))),
        )

    if args.max_val_images > 0:
        val_dataset = Subset(
            val_dataset,
            range(min(args.max_val_images, len(val_dataset))),
        )

    print(f"Training images in this run: {len(train_dataset)}")
    print(f"Validation images in this run: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=detection_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate_fn,
    )

    # Build model and optimizer.
    model = build_model().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005,
    )

    # Keep smoke-test outputs separate from other model outputs.
    output_dir = Path("training/outputs/stenosis_fasterrcnn_smoke_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        average_loss, trained_images = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        print(f"Training images processed: {trained_images}")
        print(f"Average training loss: {average_loss:.4f}")

        val_images, predicted_boxes = validate_model(
            model,
            val_loader,
            device,
        )

        print(f"Validation images processed: {val_images}")
        print(f"Predicted boxes (score >= 0.05): {predicted_boxes}")

        checkpoint_path = output_dir / "stenosis_detector_smoke_test.pth"

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "average_training_loss": average_loss,
                "num_classes": 2,
                "model_name": "Faster R-CNN MobileNetV3 Large FPN",
            },
            checkpoint_path,
        )

        print(f"Checkpoint saved: {checkpoint_path}")

    print("\nSmoke test completed.")


if __name__ == "__main__":
    main()
