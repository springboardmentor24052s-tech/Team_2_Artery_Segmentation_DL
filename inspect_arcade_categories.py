import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "data"
    / "arcade"
    / "arcade"
    / "arcade"
)

SPLIT = "train"

FILES = {
    "SYNTAX": (
        DATASET_ROOT / "syntax" / SPLIT
        / "annotations" / f"{SPLIT}.json"
    ),
    "STENOSIS": (
        DATASET_ROOT / "stenosis" / SPLIT
        / "annotations" / f"{SPLIT}.json"
    )
}

for task, file_path in FILES.items():

    print(f"\n{'=' * 55}")
    print(f"{task} ANNOTATION DETAILS")
    print(f"{'=' * 55}")

    if not file_path.exists():
        print("File not found:", file_path)
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\nCategories:")

    for category in data.get("categories", []):
        print(category)

    print("\nFirst image metadata:")

    images = data.get("images", [])

    if images:
        print(json.dumps(images[0], indent=4))

    print("\nFirst annotation:")

    annotations = data.get("annotations", [])

    if annotations:
        print(json.dumps(annotations[0], indent=4))

    print("\nImage count:", len(images))
    print("Annotation count:", len(annotations))

print("\nCATEGORY INSPECTION COMPLETE")