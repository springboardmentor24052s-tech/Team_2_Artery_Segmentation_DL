
from pathlib import Path
import argparse
import json

import torch
from PIL import ImageDraw
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as TF

from stenosis_dataset import StenosisDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "training"
    / "outputs"
    / "stenosis_fasterrcnn_smoke_test"
    / "stenosis_detector_smoke_test.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "outputs"
    / "stenosis_predictions_smoke_test"
)


def build_model():
    """Build the same Faster R-CNN architecture used during training."""
    return fasterrcnn_mobilenet_v3_large_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=2,
        min_size=512,
        max_size=512,
    )


def load_detector(device):
    """Load the saved model checkpoint."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = build_model()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(
        f"Loaded checkpoint from epoch: "
        f"{checkpoint['epoch']}"
    )

    return model


def draw_predictions(image_tensor, prediction, score_threshold, max_detections):
    """Draw selected predicted stenosis boxes on an image."""
    output_image = TF.to_pil_image(image_tensor.cpu())
    draw = ImageDraw.Draw(output_image)

    boxes = prediction["boxes"].cpu()
    scores = prediction["scores"].cpu()
    labels = prediction["labels"].cpu()

    selected = []

    for box, score, label in zip(boxes, scores, labels):
        if int(label.item()) != 1:
            continue

        if score.item() < score_threshold:
            continue

        selected.append((box, score))

    # Keep the highest-confidence predictions first.
    selected.sort(key=lambda item: item[1].item(), reverse=True)
    selected = selected[:max_detections]

    detections = []

    for box, score in selected:
        x1, y1, x2, y2 = [
            int(round(value))
            for value in box.tolist()
        ]

        confidence = float(score.item())

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline="red",
            width=3,
        )

        draw.text(
            (x1, max(0, y1 - 15)),
            f"Stenosis: {confidence:.2f}",
            fill="red",
        )

        detections.append({
            "label": "stenosis",
            "confidence": confidence,
            "box": [x1, y1, x2, y2],
        })

    return output_image, detections


@torch.no_grad()
def run_inference(model, dataset, device, score_threshold, max_detections, max_images):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    total_images = len(dataset)

    if max_images > 0:
        total_images = min(max_images, total_images)

    print(f"Images selected for inference: {total_images}")

    for index in range(total_images):
        image_tensor, target = dataset[index]

        image_info = dataset.images[index]
        image_name = image_info["file_name"]
        image_id = image_info["id"]

        predictions = model([image_tensor.to(device)])
        prediction = predictions[0]

        output_image, detections = draw_predictions(
            image_tensor,
            prediction,
            score_threshold,
            max_detections,
        )

        output_name = f"prediction_{image_id}.png"
        output_path = OUTPUT_DIR / output_name

        output_image.save(output_path)

        results.append({
            "image_id": int(image_id),
            "image_file": image_name,
            "ground_truth_box_count": int(
                target["boxes"].shape[0]
            ),
            "predicted_stenosis_count": len(detections),
            "detections": detections,
            "output_image": output_name,
        })

        print(
            f"{index + 1}/{total_images}: "
            f"{image_name} | "
            f"Ground-truth boxes: {target['boxes'].shape[0]} | "
            f"Predicted boxes: {len(detections)}"
        )

    results_path = OUTPUT_DIR / "predictions.json"

    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nInference completed.")
    print(f"Prediction images saved in: {OUTPUT_DIR}")
    print(f"Detection results saved in: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run stenosis detection inference."
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

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--max-detections",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    if args.max_images < 0:
        raise ValueError("--max-images cannot be negative.")

    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be between 0 and 1.")

    if args.max_detections < 1:
        raise ValueError("--max-detections must be at least 1.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    dataset = StenosisDataset(args.split)

    model = load_detector(device)

    run_inference(
        model=model,
        dataset=dataset,
        device=device,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
