
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "stenosis_classification_dataset_v2"
)

SPLITS = ["train", "val", "test"]

dataframes = {}

for split in SPLITS:
    csv_path = DATASET_DIR / f"{split}_labels.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)
    df["split"] = split
    dataframes[split] = df

combined = pd.concat(dataframes.values(), ignore_index=True)

print("\n========== 1. DATASET COUNTS ==========\n")

for split, df in dataframes.items():
    print(f"\n{split.upper()}")
    print("Total patches:", len(df))
    print("Positive patches:", int((df["label"] == 1).sum()))
    print("Negative patches:", int((df["label"] == 0).sum()))
    print("Unique source images:", df["source_image"].nunique())

print("\n========== 2. NEGATIVE PATCH TYPE DISTRIBUTION ==========\n")

negative_df = combined[combined["label"] == 0]

print(
    pd.crosstab(
        negative_df["split"],
        negative_df["negative_type"],
        margins=True
    )
)

print("\n========== 3. VESSEL RESPONSE SCORE STATISTICS ==========\n")

if "vessel_response_score" in combined.columns:

    print(
        negative_df.groupby(
            ["split", "negative_type"]
        )["vessel_response_score"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(4)
        .to_string()
    )

else:
    print("vessel_response_score column not found.")

print("\n========== 4. POSITIVE PATCH DISTRIBUTION ==========\n")

positive_df = combined[combined["label"] == 1]

print(
    positive_df.groupby("split")
    .agg(
        total_patches=("filename", "count"),
        unique_source_images=("source_image", "nunique")
    )
    .to_string()
)

print("\n========== 5. PATCHES PER SOURCE IMAGE ==========\n")

source_counts = (
    combined.groupby(["split", "source_image", "label"])
    .size()
    .reset_index(name="patch_count")
)

print(
    source_counts.groupby(["split", "label"])["patch_count"]
    .describe()
    .round(2)
    .to_string()
)

print("\n========== 6. COLUMN INFORMATION ==========\n")

print("Columns:", combined.columns.tolist())

print("\n========== COMPARISON COMPLETED ==========\n")
