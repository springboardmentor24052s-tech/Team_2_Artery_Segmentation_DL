
import os
import sys
import json
import csv
import inspect
import importlib.util
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
OUTPUT_DIR = TRAINING_DIR / "outputs"

DATASET_ROOT = PROJECT_ROOT / "dataset"

IMAGE_SIZE = 128
BASE_CHANNELS = 8
BATCH_SIZE = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIGS = {
    "baseline_unet": {
        "module_path": TRAINING_DIR / "unet_baseline.py",
        "checkpoint": OUTPUT_DIR / "unet_baseline_5epochs" / "best_model.pth",
    },
    "attention_unet": {
        "module_path": TRAINING_DIR / "attention_unet.py",
        "checkpoint": OUTPUT_DIR / "attention_unet_5epochs" / "best_model.pth",
    },
}

RESULTS_DIR = OUTPUT_DIR / "full_test_evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD TRAINING MODULE
# ============================================================

def load_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(
        module_name, module_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


# ============================================================
# 3. LOAD DATASET CLASS
# ============================================================

baseline_module = load_module(
    "baseline_training_module",
    MODEL_CONFIGS["baseline_unet"]["module_path"]
)

attention_module = load_module(
    "attention_training_module",
    MODEL_CONFIGS["attention_unet"]["module_path"]
)

# The dataset class was defined in the baseline training module.
if not hasattr(baseline_module, "ArcadeSyntaxDataset"):
    raise AttributeError(
        "ArcadeSyntaxDataset was not found in unet_baseline.py"
    )

ArcadeSyntaxDataset = baseline_module.ArcadeSyntaxDataset


# ============================================================
# 4. CREATE TEST DATASET
# ============================================================

test_dataset = ArcadeSyntaxDataset(
    dataset_root=DATASET_ROOT,
    split="test",
    image_size=IMAGE_SIZE
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nArteryAI - Full Test Set Evaluation")
print("-----------------------------------")
print("Device:", DEVICE)
print("Dataset:", DATASET_ROOT)
print("Test images:", len(test_dataset))


# ============================================================
# 5. CREATE MODEL
# ============================================================

def create_model(module, checkpoint_path):

    model_class = None

    # Find the model class defined in the training module.
    for name, obj in inspect.getmembers(module, inspect.isclass):

        if obj.__module__ == module.__name__ and (
            "unet" in name.lower()
        ):
            model_class = obj
            break

    if model_class is None:
        raise RuntimeError(
            f"Could not find U-Net model class in {module.__file__}"
        )

    signature = inspect.signature(model_class.__init__)
    available_args = {
        "in_channels": 3,
        "input_channels": 3,
        "out_channels": 1,
        "output_channels": 1,
        "base_channels": BASE_CHANNELS,
        "init_features": BASE_CHANNELS,
        "image_size": IMAGE_SIZE,
    }

    model_args = {}

    for parameter in signature.parameters.values():

        if parameter.name in available_args:
            model_args[parameter.name] = available_args[parameter.name]

    model = model_class(**model_args)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    # Support checkpoints saved as either a state_dict or a dictionary.
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Handle checkpoints saved using DataParallel.
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict)

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# 6. CALCULATE METRICS
# ============================================================

def calculate_metrics(prediction, target):

    prediction = prediction.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(prediction, target).sum()
    tn = np.logical_and(~prediction, ~target).sum()
    fp = np.logical_and(prediction, ~target).sum()
    fn = np.logical_and(~prediction, target).sum()

    dice_denominator = (2 * tp) + fp + fn
    iou_denominator = tp + fp + fn

    dice = (
        (2 * tp) / dice_denominator
        if dice_denominator > 0 else 1.0
    )

    iou = (
        tp / iou_denominator
        if iou_denominator > 0 else 1.0
    )

    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0 else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0 else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0 else 0.0
    )

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Dice": float(dice),
        "IoU": float(iou),
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "Specificity": float(specificity),
    }


# ============================================================
# 7. EVALUATE ONE MODEL ON THE COMPLETE TEST SET
# ============================================================

def evaluate_model(model_name, model):

    records = []

    print(f"\nEvaluating: {model_name}")

    with torch.no_grad():

        for index, batch in enumerate(
            tqdm(test_loader, desc=model_name)
        ):

            # Dataset is expected to return image and mask.
            images, masks, *extra = batch

            images = images.to(DEVICE).float()
            masks = masks.to(DEVICE).float()

            # Ensure image tensors are RGB.
            if images.ndim != 4 or images.shape[1] != 3:
                raise ValueError(
                    f"Expected RGB input [B, 3, H, W], "
                    f"but received {tuple(images.shape)}"
                )

            outputs = model(images)

            # Convert logits to probabilities and binary masks.
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities >= 0.5).float()

            prediction_np = predictions.cpu().numpy()
            target_np = masks.cpu().numpy()

            for batch_index in range(prediction_np.shape[0]):

                prediction = prediction_np[batch_index].squeeze()
                target = target_np[batch_index].squeeze()

                if prediction.shape != target.shape:
                    raise ValueError(
                        f"Prediction and mask shapes differ: "
                        f"{prediction.shape} vs {target.shape}"
                    )

                metrics = calculate_metrics(
                    prediction,
                    target
                )

                image_number = (
                    index * BATCH_SIZE + batch_index + 1
                )

                record = {
                    "Model": model_name,
                    "Image_Number": image_number,
                    **metrics
                }

                records.append(record)

    return records


# ============================================================
# 8. EVALUATE BOTH MODELS
# ============================================================

all_records = []

for model_name, config in MODEL_CONFIGS.items():

    print("\nLoading model:", model_name)

    if not config["module_path"].exists():
        raise FileNotFoundError(config["module_path"])

    if not config["checkpoint"].exists():
        raise FileNotFoundError(config["checkpoint"])

    module = (
        baseline_module
        if model_name == "baseline_unet"
        else attention_module
    )

    model = create_model(
        module,
        config["checkpoint"]
    )

    model_records = evaluate_model(
        model_name,
        model
    )

    all_records.extend(model_records)

    del model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"{model_name}: evaluated "
        f"{len(model_records)} images"
    )


# ============================================================
# 9. SAVE IMAGE-WISE RESULTS
# ============================================================

details_path = RESULTS_DIR / "full_test_metrics.csv"

if all_records:

    fieldnames = list(all_records[0].keys())

    with open(
        details_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_records)


# ============================================================
# 10. GENERATE MODEL SUMMARY
# ============================================================

metric_names = [
    "Dice",
    "IoU",
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
]

summary = []

for model_name in MODEL_CONFIGS:

    model_records = [
        record for record in all_records
        if record["Model"] == model_name
    ]

    if not model_records:
        continue

    result = {
        "Model": model_name,
        "Test_Images": len(model_records),
    }

    for metric in metric_names:

        values = [
            record[metric]
            for record in model_records
        ]

        result[f"Mean_{metric}"] = float(
            np.mean(values)
        )

    summary.append(result)


summary_path = RESULTS_DIR / "model_summary.csv"

if summary:

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summary[0].keys())
        )

        writer.writeheader()
        writer.writerows(summary)


# ============================================================
# 11. SAVE JSON RESULTS
# ============================================================

json_path = RESULTS_DIR / "full_test_results.json"

with open(
    json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            "dataset": str(DATASET_ROOT),
            "test_images": len(test_dataset),
            "device": str(DEVICE),
            "model_summary": summary,
            "image_wise_results": all_records,
        },
        file,
        indent=4
    )


# ============================================================
# 12. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FULL TEST SET EVALUATION COMPLETED")
print("=" * 60)

print("Total test images:", len(test_dataset))
print("Total evaluation records:", len(all_records))

for result in summary:

    print("\nModel:", result["Model"])
    print("Images:", result["Test_Images"])

    for metric in metric_names:

        print(
            f"Mean {metric}: "
            f"{result[f'Mean_{metric}']:.4f}"
        )

print("\nSaved files:")
print(details_path)
print(summary_path)
print(json_path)

print("\nEvaluation completed successfully!")
