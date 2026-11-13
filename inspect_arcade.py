
import json
from pathlib import Path

DATA_DIR = Path("backend/app/data/arcade/arcade/arcade")

json_files = sorted(DATA_DIR.rglob("*.json"))

if not json_files:
    print("ERROR: No JSON annotation files found.")
    raise SystemExit(1)

print(f"Found {len(json_files)} JSON files\n")

for path in json_files:
    print("=" * 70)
    print("FILE:", path)

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("Could not read JSON:", e)
        continue

    print("Top-level keys:", list(data.keys()))

    categories = data.get("categories", [])
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    print("Images:", len(images))
    print("Annotations:", len(annotations))
    print("Categories:", categories[:20])

    if images:
        print("First image metadata:", images[0])

    if annotations:
        ann = annotations[0]
        print("First annotation keys:", list(ann.keys()))
        print("First annotation category_id:", ann.get("category_id"))
        print("First annotation image_id:", ann.get("image_id"))
        print("First segmentation:", str(ann.get("segmentation"))[:250])

print("\nInspection complete.")
