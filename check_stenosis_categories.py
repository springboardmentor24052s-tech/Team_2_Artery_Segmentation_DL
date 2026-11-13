
import json
from pathlib import Path
from collections import Counter

ROOT = Path("backend/app/data/arcade/arcade/arcade")

for split in ["train", "val", "test"]:

    annotation_file = (
        ROOT / "stenosis" / split /
        "annotations" / f"{split}.json"
    )

    with open(annotation_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    print("\n==========", split.upper(), "==========")

    print("\nCategories:")
    for category in data["categories"]:
        print(
            "ID:", category["id"],
            "| Name:", category["name"]
        )

    counts = Counter(
        annotation["category_id"]
        for annotation in data["annotations"]
    )

    print("\nAnnotation counts by category:")
    for category_id, count in sorted(counts.items()):
        print("Category ID:", category_id, "| Count:", count)
