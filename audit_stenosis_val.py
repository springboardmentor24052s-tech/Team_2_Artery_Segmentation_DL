
import json
from pathlib import Path
from collections import Counter

ROOT = Path("backend/app/data/arcade/arcade/arcade")

ANNOTATION_FILE = (
    ROOT / "stenosis/val/annotations/val.json"
)

with ANNOTATION_FILE.open("r", encoding="utf-8") as f:
    data = json.load(f)

categories = {
    category["id"]: category["name"]
    for category in data["categories"]
}

counts = Counter(
    annotation["category_id"]
    for annotation in data["annotations"]
)

print("\nSTENOSIS VALIDATION ANNOTATION AUDIT")
print("=" * 45)

for category_id, count in sorted(counts.items()):
    print(
        f"Category ID: {category_id} | "
        f"Name: {categories.get(category_id, 'Unknown')} | "
        f"Annotations: {count}"
    )

stenosis_ids = {
    cid for cid, name in categories.items()
    if name.lower() == "stenosis"
}

stenosis_annotations = [
    a for a in data["annotations"]
    if a["category_id"] in stenosis_ids
]

other_annotations = [
    a for a in data["annotations"]
    if a["category_id"] not in stenosis_ids
]

print("\nSUMMARY")
print("Total images:", len(data["images"]))
print("Total annotations:", len(data["annotations"]))
print("Stenosis annotations:", len(stenosis_annotations))
print("Other category annotations:", len(other_annotations))
print("=" * 45)
