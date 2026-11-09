
from pathlib import Path
import hashlib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "data"
    / "arcade"
    / "arcade"
    / "arcade"
    / "stenosis"
)

CLASSIFICATION_DIR = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "stenosis_classification_dataset_v2"
)

def calculate_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


hashes = {}

for split in ["train", "val", "test"]:

    df = pd.read_csv(
        CLASSIFICATION_DIR / f"{split}_labels.csv"
    )

    image_dir = DATASET_ROOT / split / "images"

    for filename in df["source_image"].dropna().astype(str).unique():

        path = image_dir / filename

        if not path.exists():
            continue

        digest = calculate_hash(path)

        hashes.setdefault(digest, []).append(
            (split, filename, str(path))
        )

print("\n========== IDENTICAL SOURCE IMAGES ==========\n")

duplicate_count = 0

for digest, records in hashes.items():

    splits_present = {record[0] for record in records}

    if len(splits_present) > 1:

        duplicate_count += 1

        print("SHA-256:", digest)

        for split, filename, path in records:
            print(f"  {split.upper()}: {filename}")
            print(f"  Path: {path}")

        print()

print("Cross-split duplicate groups:", duplicate_count)
print("\nAnalysis completed.")
