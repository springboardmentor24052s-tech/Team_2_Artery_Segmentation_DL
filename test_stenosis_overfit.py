
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from train_stenosis import (
    DEVICE,
    StenosisDataset,
    UNet,
    combined_loss,
)

NUM_IMAGES = 4
EPOCHS = 30
LEARNING_RATE = 0.0003


def main():
    torch.manual_seed(42)

    dataset = StenosisDataset()
    subset = Subset(dataset, list(range(NUM_IMAGES)))
    loader = DataLoader(subset, batch_size=2, shuffle=False)

    model = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Device:", DEVICE)
    print("Training images:", NUM_IMAGES)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = combined_loss(logits, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch == 0 or (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                images, masks = next(iter(loader))
                logits = model(images.to(DEVICE))
                probabilities = torch.sigmoid(logits)

                predicted_pixels = int((probabilities >= 0.5).sum().item())
                target_pixels = int((masks >= 0.5).sum().item())
                max_probability = float(probabilities.max().item())
                mean_probability = float(probabilities.mean().item())

            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS} | "
                f"Loss: {total_loss / len(loader):.4f} | "
                f"Target pixels: {target_pixels} | "
                f"Predicted pixels: {predicted_pixels} | "
                f"Max probability: {max_probability:.4f} | "
                f"Mean probability: {mean_probability:.4f}"
            )


if __name__ == "__main__":
    main()
