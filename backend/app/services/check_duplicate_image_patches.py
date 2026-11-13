
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "stenosis_classification_dataset_v2"
)

for split, image_name in [
    ("train", "218.png"),
    ("val", "58.png")
]:

    csv_path = DATASET_DIR / f"{split}_labels.csv"
    df = pd.read_csv(csv_path)

    matching = df[df["source_image"].astype(str) == image_name]

    print(f"\n========== {split.upper()} : {image_name} ==========")
    print("Total patches:", len(matching))

    if not matching.empty:
        print(
            matching[
                [
                    "filename",
                    "label",
                    "class_name",
                    "negative_type",
                    "vessel_response_score"
                ]
            ].to_string(index=False)
        )

print("\nAnalysis completed.")
