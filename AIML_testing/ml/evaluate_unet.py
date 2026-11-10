
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from unet_model import UNet


# --------------------------------------------------
# 1. CONFIGURATION AND PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = ROOT / "data" / "processed" / "stenosis"
TEST_IMAGE_DIR = DATA_ROOT / "test" / "images"
TEST_MASK_DIR = DATA_ROOT / "test" / "masks"

MODEL_PATH = ROOT / "outputs" / "models" / "unet_best.pth"

OUTPUT_DIR = ROOT / "outputs" / "evaluation_unet"
VISUAL_DIR = OUTPUT_DIR / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 256

# Evaluate the model with threshold 0.3
THRESHOLD = 0.3

BATCH_SIZE = 1

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# 2. TEST DATASET
# --------------------------------------------------

class StenosisTestDataset(Dataset):

    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        image_paths = sorted(image_dir.glob("*.png"))

        valid_pairs = []

        for image_path in image_paths:
            mask_path = mask_dir / image_path.name

            if mask_path.exists():
                valid_pairs.append((image_path, mask_path))

        self.image_paths = [pair[0] for pair in valid_pairs]
        self.mask_paths = [pair[1] for pair in valid_pairs]

        if not self.image_paths:
            raise RuntimeError(
                "No matching test image-mask pairs found.\n"
                f"Image directory: {image_dir}\n"
                f"Mask directory: {mask_dir}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

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
                f"Could not read image or mask: {image_path.name}"
            )

        # Resize image and mask to model input size
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

        # Normalize image and convert mask to binary
        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return image_tensor, mask_tensor, image_path.name


# --------------------------------------------------
# 3. LOAD TRAINED U-NET
# --------------------------------------------------

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            "Check whether unet_best.pth exists."
        )

    model = UNet().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Support plain state_dict and checkpoint dictionaries
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model


# --------------------------------------------------
# 4. DICE AND IOU METRICS
# --------------------------------------------------

def calculate_metrics(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(
        prediction, target
    ).sum()

    pred_pixels = prediction.sum()
    target_pixels = target.sum()

    denominator = pred_pixels + target_pixels

    if denominator == 0:
        dice = 1.0
    else:
        dice = (2.0 * intersection) / denominator

    union = np.logical_or(
        prediction, target
    ).sum()

    if union == 0:
        iou = 1.0
    else:
        iou = intersection / union

    return float(dice), float(iou)


# --------------------------------------------------
# 5. EVALUATION
# --------------------------------------------------

def main():
    print("=" * 60)
    print("U-NET TEST EVALUATION")
    print("=" * 60)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Checkpoint:", MODEL_PATH)
    print("Threshold:", THRESHOLD)

    dataset = StenosisTestDataset(
        TEST_IMAGE_DIR,
        TEST_MASK_DIR
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Test images:", len(dataset))

    model = load_model()

    dice_scores = []
    iou_scores = []
    results = []

    with torch.no_grad():
        for index, (images, masks, filenames) in enumerate(loader):

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            probabilities = torch.sigmoid(logits)

            # Apply threshold 0.3
            predictions = (
                probabilities >= THRESHOLD
            ).float()

            prediction_np = (
                predictions[0, 0].cpu().numpy()
            ).astype(np.uint8)

            target_np = (
                masks[0, 0].cpu().numpy()
            ).astype(np.uint8)

            dice, iou = calculate_metrics(
                prediction_np,
                target_np
            )

            dice_scores.append(dice)
            iou_scores.append(iou)

            filename = filenames[0]

            results.append({
                "image": filename,
                "dice": dice,
                "iou": iou
            })

            # Save prediction visualizations for first 5 images
            if index < 5:
                image_np = (
                    images[0, 0].cpu().numpy()
                )

                fig, axes = plt.subplots(
                    1, 4, figsize=(16, 4)
                )

                axes[0].imshow(
                    image_np, cmap="gray"
                )
                axes[0].set_title("Original Image")

                axes[1].imshow(
                    target_np, cmap="gray"
                )
                axes[1].set_title("Ground Truth Mask")

                axes[2].imshow(
                    prediction_np, cmap="gray"
                )
                axes[2].set_title(
                    f"U-Net Prediction (T={THRESHOLD})"
                )

                # Overlay predicted region in red
                overlay = np.stack(
                    [image_np] * 3, axis=-1
                )

                red = np.zeros_like(overlay)
                red[:, :, 0] = 1.0

                overlay = np.where(
                    prediction_np[..., None] == 1,
                    0.6 * overlay + 0.4 * red,
                    overlay
                )

                axes[3].imshow(
                    np.clip(overlay, 0, 1)
                )
                axes[3].set_title("Prediction Overlay")

                for ax in axes:
                    ax.axis("off")

                fig.suptitle(
                    f"{filename} | Dice: {dice:.4f} | IoU: {iou:.4f}"
                )

                plt.tight_layout()

                save_path = (
                    VISUAL_DIR / f"prediction_{index + 1}.png"
                )

                fig.savefig(
                    save_path,
                    dpi=150,
                    bbox_inches="tight"
                )

                plt.close(fig)

            if (index + 1) % 50 == 0:
                print(
                    f"Evaluated {index + 1}/{len(dataset)} images"
                )

    # --------------------------------------------------
    # 6. FINAL RESULTS
    # --------------------------------------------------

    mean_dice = float(np.mean(dice_scores))
    mean_iou = float(np.mean(iou_scores))

    print("\n" + "=" * 60)
    print("U-NET TEST RESULTS")
    print("=" * 60)

    print("Evaluated Images :", len(dice_scores))
    print("Threshold        :", THRESHOLD)

    print(
        f"Mean Dice Score  : {mean_dice:.4f} "
        f"({mean_dice * 100:.2f}%)"
    )

    print(
        f"Mean IoU Score   : {mean_iou:.4f} "
        f"({mean_iou * 100:.2f}%)"
    )

    # Save per-image metrics to CSV
    import csv

    csv_path = OUTPUT_DIR / "test_metrics.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=["image", "dice", "iou"]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nVisualizations saved in:", VISUAL_DIR)
    print("Per-image metrics saved in:", csv_path)
    print("=" * 60)


if __name__ == "__main__":
    main()
