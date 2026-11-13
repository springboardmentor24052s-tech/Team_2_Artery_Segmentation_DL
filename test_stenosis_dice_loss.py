
import torch
from torch.utils.data import DataLoader, Subset

from train_stenosis import DEVICE, StenosisDataset, UNet


NUM_IMAGES = 4
EPOCHS = 40
LEARNING_RATE = 0.0001


def dice_loss(logits, targets):
    probabilities = torch.sigmoid(logits)

    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice = (2 * intersection + 1) / (denominator + 1)
    return 1 - dice.mean()


def calculate_dice(logits, targets):
    predictions = torch.sigmoid(logits) >= 0.5
    targets = targets >= 0.5

    intersection = (predictions & targets).sum().item()
    denominator = predictions.sum().item() + targets.sum().item()

    if denominator == 0:
        return 1.0

    return 2 * intersection / denominator


def main():
    torch.manual_seed(42)

    dataset = StenosisDataset()
    subset = Subset(dataset, list(range(NUM_IMAGES)))

    loader = DataLoader(
        subset,
        batch_size=2,
        shuffle=False
    )

    model = UNet().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("Training images:", NUM_IMAGES)
    print("Loss: Dice loss only")

    # Training
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)
            loss = dice_loss(logits, masks)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch == 0 or (epoch + 1) % 5 == 0:
            model.eval()

            with torch.no_grad():
                dice_total = 0.0
                max_probability = 0.0
                predicted_pixels = 0

                for images, masks in loader:
                    logits = model(images.to(DEVICE))
                    probabilities = torch.sigmoid(logits)

                    dice_total += calculate_dice(
                        logits, masks.to(DEVICE)
                    )

                    max_probability = max(
                        max_probability,
                        float(probabilities.max().item())
                    )

                    predicted_pixels += int(
                        (probabilities >= 0.5).sum().item()
                    )

            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS} | "
                f"Loss: {total_loss / len(loader):.4f} | "
                f"Training Dice: {dice_total / len(loader):.4f} | "
                f"Predicted pixels: {predicted_pixels} | "
                f"Max probability: {max_probability:.4f}"
            )

    # Gradient diagnostic
    model.train()

    images, masks = next(iter(loader))
    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    optimizer.zero_grad()

    logits = model(images)
    loss = dice_loss(logits, masks)
    loss.backward()

    grad = model.output.weight.grad

    print("\nGradient diagnostic")
    print("-------------------")
    print("Input min/max:", images.min().item(), images.max().item())
    print("Mask foreground pixels:", int(masks.sum().item()))
    print("Logits min/max:", logits.min().item(), logits.max().item())
    print("Loss:", loss.item())

    print(
        "Output-layer gradient mean:",
        None if grad is None else grad.abs().mean().item()
    )


if __name__ == "__main__":
    main()
