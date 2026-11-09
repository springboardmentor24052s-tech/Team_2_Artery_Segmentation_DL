
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "stenosis_classification_dataset_v2"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "classifier_v2_error_inspection"
    / "patch_statistics.csv"
)

all_stats = []
missing_files = 0
unreadable_files = 0

for split in ["train", "val", "test"]:

    csv_path = DATASET_DIR / f"{split}_labels.csv"
    df = pd.read_csv(csv_path)

    print(f"\nProcessing {split.upper()}...")

    for _, row in df.iterrows():

        label = int(row["label"])
        class_folder = "positive" if label == 1 else "negative"

        patch_path = (
            DATASET_DIR
            / split
            / class_folder
            / row["filename"]
        )

        if not patch_path.exists():
            missing_files += 1
            continue

        image = cv2.imread(
            str(patch_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            unreadable_files += 1
            continue

        all_stats.append({
            "split": split,
            "label": label,
            "negative_type": row["negative_type"],
            "mean_intensity": float(np.mean(image)),
            "std_intensity": float(np.std(image)),
            "min_intensity": int(np.min(image)),
            "max_intensity": int(np.max(image)),
            "bright_pixel_ratio": float(np.mean(image > 200)),
            "dark_pixel_ratio": float(np.mean(image < 50)),
            "source_image": row["source_image"]
        })

stats = pd.DataFrame(all_stats)

if stats.empty:
    raise RuntimeError("No image patches could be read.")

print("\n========== PATCH STATISTICS ==========\n")

print(
    stats.groupby(["split", "label"])[
        [
            "mean_intensity",
            "std_intensity",
            "bright_pixel_ratio",
            "dark_pixel_ratio"
        ]
    ]
    .agg(["mean", "median", "std"])
    .round(4)
    .to_string()
)

print("\n========== NEGATIVE PATCH STATISTICS ==========\n")

negative_stats = stats[stats["label"] == 0]

print(
    negative_stats.groupby(["split", "negative_type"])[
        [
            "mean_intensity",
            "std_intensity",
            "bright_pixel_ratio",
            "dark_pixel_ratio"
        ]
    ]
    .mean()
    .round(4)
    .to_string()
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
stats.to_csv(OUTPUT_PATH, index=False)

print("\n========== FILE VALIDATION ==========")
print("Total patches analyzed:", len(stats))
print("Missing files:", missing_files)
print("Unreadable files:", unreadable_files)
print("Statistics saved to:", OUTPUT_PATH)

print("\nAnalysis completed.")
