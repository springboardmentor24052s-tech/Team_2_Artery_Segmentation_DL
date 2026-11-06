
from pathlib import Path
import argparse

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from attention_unet import AttentionUNet
from stenosis_dataset import StenosisDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "training"
    / "outputs"
    / "attention_unet_5epochs"
    / "best_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "outputs"
    / "artery_masks_stenosis_test"
)


def load_segmentation_model(device):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model = AttentionUNet(
        base_channels=checkpoint["base_channels"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    image_size = checkpoint["image_size"]

    print(f"Loaded checkpoint epoch: {checkpoint['epoch']}")
    print(f"Model input size: {image_size} x {image_size}")

    return model, image_size


def create_overlay(original_image, mask):
    original = original_image.convert("RGB")

    red_layer = Image.new(
        "RGB",
        original.size,
        (255, 0, 0),
    )

    highlighted = Image.composite(
        red_layer,
        original,
        mask,
    )

    return Image.blend(original, highlighted, 0.4)


@torch.no_grad()
def run_inference(model, dataset, device, image_size, max_images):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = min(max_images, len(dataset))

    print(f"Images selected: {total}")

    for index in range(total):
        image_tensor, target = dataset[index]

        image_id = int(target["image_id"].item())
        image_info = dataset.images[index]
        image_name = image_info["file_name"]

        original_image = TF.to_pil_image(image_tensor)

        # Match the image resizing used during Attention U-Net training.
        resized_image = original_image.resize(
            (image_size, image_size),
            Image.Resampling.BILINEAR,
        )

        input_tensor = TF.to_tensor(resized_image).unsqueeze(0)
        input_tensor = input_tensor.to(device)

        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)[0, 0]

        mask_array = (
            probabilities.cpu().numpy() >= 0.5
        ).astype(np.uint8) * 255

        small_mask = Image.fromarray(mask_array)

        # Restore the mask to the original image dimensions.
        full_mask = small_mask.resize(
            original_image.size,
            Image.Resampling.NEAREST,
        )

        overlay = create_overlay(original_image, full_mask)

        image_stem = Path(image_name).stem

        original_image.save(
            OUTPUT_DIR / f"{image_stem}_original.png"
        )

        full_mask.save(
            OUTPUT_DIR / f"{image_stem}_mask.png"
        )

        overlay.save(
            OUTPUT_DIR / f"{image_stem}_overlay.png"
        )

        foreground_pixels = int(
            np.count_nonzero(mask_array)
        )

        print(
            f"{index + 1}/{total}: {image_name} | "
            f"Mask foreground pixels at model resolution: "
            f"{foreground_pixels}"
        )

    print("\nSegmentation inference completed.")
    print(f"Results saved in: {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Attention U-Net artery segmentation."
    )

    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    if args.max_images < 1:
        parser.error("--max-images must be at least 1.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    dataset = StenosisDataset(args.split)

    model, image_size = load_segmentation_model(device)

    run_inference(
        model,
        dataset,
        device,
        image_size,
        args.max_images,
    )


if __name__ == "__main__":
    main()
