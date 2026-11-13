
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from attention_unet import AttentionUNet


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ROOT = Path("backend/app/data/arcade/arcade/arcade")
MASK_ROOT = Path("backend/results/arcade_masks")

ANNOTATION_FILE = ROOT / "stenosis/val/annotations/val.json"
IMAGE_DIR = ROOT / "stenosis/val/images"
MASK_DIR = MASK_ROOT / "stenosis/val"

# Attention U-Net V3 checkpoint
MODEL_PATH = Path(
    "backend/results/attention_unet_stenosis_v3.pth"
)

# Separate V3 prediction folder
OUTPUT_DIR = Path(
    "backend/results/attention_unet_v3_predictions"
)

DEVICE = torch.device("cpu")
BATCH_SIZE = 2

THRESHOLDS = [
    0.05, 0.10, 0.15, 0.20,
    0.25, 0.30, 0.40, 0.50
]


# --------------------------------------------------
# 2. VALIDATION DATASET
# --------------------------------------------------

class StenosisValidationDataset(Dataset):

    def __init__(self):

        with ANNOTATION_FILE.open(
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

            image_path = IMAGE_DIR / info["file_name"]

            mask_path = (
                MASK_DIR /
                f"{Path(info['file_name']).stem}_mask.png"
            )

            if image_path.exists() and mask_path.exists():
                self.samples.append(
                    (image_path, mask_path, info["file_name"])
                )

        if not self.samples:
            raise RuntimeError(
                "No validation image-mask pairs found."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, mask_path, filename = self.samples[index]

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
                f"Could not read validation sample: {filename}"
            )

        if image.shape != mask.shape:
            raise RuntimeError(
                f"Image-mask dimensions mismatch: {filename}"
            )

        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        return (
            torch.from_numpy(image).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
            filename
        )


# --------------------------------------------------
# 3. MODEL LOADING
# --------------------------------------------------

def load_model():

    model = AttentionUNet().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model


# --------------------------------------------------
# 4. EVALUATION
# --------------------------------------------------

def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    dataset = StenosisValidationDataset()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = load_model()

    print("\nEvaluating Attention U-Net V3...")
    print("Validation images:", len(dataset))

    all_probabilities = []
    all_targets = []
    all_filenames = []

    with torch.no_grad():

        for images, masks, filenames in loader:

            images = images.to(DEVICE)

            probabilities = torch.sigmoid(model(images))

            for i in range(len(images)):

                all_probabilities.append(
                    probabilities[i, 0].cpu().numpy()
                )

                all_targets.append(
                    masks[i, 0].numpy() > 0.5
                )

                all_filenames.append(filenames[i])

    results = []

    print("\nThreshold  Dice     IoU      Precision  Recall")

    for threshold in THRESHOLDS:

        dice_scores = []
        iou_scores = []
        precision_scores = []
        recall_scores = []

        for probabilities, target in zip(
            all_probabilities, all_targets
        ):

            prediction = probabilities >= threshold

            tp = np.logical_and(
                prediction, target
            ).sum()

            fp = np.logical_and(
                prediction, ~target
            ).sum()

            fn = np.logical_and(
                ~prediction, target
            ).sum()

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

        result = {
            "threshold": threshold,
            "dice": float(np.mean(dice_scores)),
            "iou": float(np.mean(iou_scores)),
            "precision": float(np.mean(precision_scores)),
            "recall": float(np.mean(recall_scores))
        }

        results.append(result)

        print(
            f"{threshold:<10.2f}"
            f"{result['dice']:.4f}   "
            f"{result['iou']:.4f}   "
            f"{result['precision']:.4f}      "
            f"{result['recall']:.4f}"
        )

    # Select best threshold using validation Dice
    best = max(results, key=lambda x: x["dice"])

    print("\nBEST VALIDATION RESULT")
    print("---------------------")
    print("Best threshold:", best["threshold"])
    print("Best Dice:", round(best["dice"], 4))
    print("IoU:", round(best["iou"], 4))
    print("Precision:", round(best["precision"], 4))
    print("Recall:", round(best["recall"], 4))

    # Save predictions using selected threshold
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for probabilities, filename in zip(
        all_probabilities, all_filenames
    ):

        prediction = (
            probabilities >= best["threshold"]
        ).astype(np.uint8) * 255

        output_path = (
            OUTPUT_DIR /
            f"{Path(filename).stem}_pred.png"
        )

        cv2.imwrite(str(output_path), prediction)

    # Save evaluation report
    report_path = OUTPUT_DIR / "evaluation_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "model": "Attention U-Net V3",
                "validation_images": len(dataset),
                "threshold_results": results,
                "best_result": best
            },
            file,
            indent=4
        )

    print("\nPredictions saved to:", OUTPUT_DIR.resolve())
    print("Evaluation report saved to:", report_path.resolve())


if __name__ == "__main__":
    main()
