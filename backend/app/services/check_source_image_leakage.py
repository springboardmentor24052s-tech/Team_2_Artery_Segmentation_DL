
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "stenosis_classification_dataset_v2"
)

splits = {}

for split in ["train", "val", "test"]:

    csv_path = DATASET_DIR / f"{split}_labels.csv"
    df = pd.read_csv(csv_path)

    splits[split] = set(df["source_image"].dropna().astype(str))

print("\n========== SOURCE IMAGE OVERLAP ANALYSIS ==========\n")

for split, images in splits.items():
    print(f"{split.upper()} unique source images: {len(images)}")

pairs = [
    ("TRAIN", "VAL", "train", "val"),
    ("TRAIN", "TEST", "train", "test"),
    ("VAL", "TEST", "val", "test")
]

for name1, name2, key1, key2 in pairs:

    overlap = splits[key1].intersection(splits[key2])

    print(f"\n{name1} vs {name2}")
    print("Overlapping source images:", len(overlap))

    if overlap:
        print("Examples:", sorted(overlap)[:10])
    else:
        print("No overlapping source images.")

all_overlap = (
    splits["train"].intersection(splits["val"])
    | splits["train"].intersection(splits["test"])
    | splits["val"].intersection(splits["test"])
)

print("\n========== FINAL RESULT ==========")
print("Total overlapping source images:", len(all_overlap))

if len(all_overlap) == 0:
    print("No source-image overlap detected.")
else:
    print("Source-image overlap detected. Investigate split construction.")

print("\nAnalysis completed.")
