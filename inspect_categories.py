
import json
from pathlib import Path
from collections import Counter

root = Path("backend/app/data/arcade/arcade/arcade")

for task in ["syntax", "stenosis"]:
    path = root / task / "train" / "annotations" / "train.json"

    if not path.exists():
        print(f"\n{task}: training annotation file not found")
        continue

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    names = {
        category["id"]: category["name"]
        for category in data["categories"]
    }

    counts = Counter(
        ann["category_id"] for ann in data["annotations"]
    )

    print(f"\n{'=' * 45}")
    print(f"TASK: {task.upper()}")
    print("Annotation counts by category:")

    for category_id, count in sorted(counts.items()):
        print(
            f"ID: {category_id:>2} | "
            f"Name: {names.get(category_id, 'UNKNOWN'):<10} | "
            f"Annotations: {count}"
        )
