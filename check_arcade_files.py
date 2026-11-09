
import json
from pathlib import Path

root = Path("backend/app/data/arcade/arcade/arcade")

for task in ["syntax", "stenosis"]:
    print(f"\n--- {task.upper()} ---")

    for split in ["train", "val", "test"]:
        annotation_file = (
            root / task / split / "annotations" / f"{split}.json"
        )

        if not annotation_file.exists():
            print(split, ": annotation file not found")
            continue

        with annotation_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        image_dir = root / task / split / "images"
        image_ids = {img["id"] for img in data["images"]}

        print(f"\nSplit: {split}")
        print("Image count:", len(data["images"]))
        print("Annotation count:", len(data["annotations"]))
        print("Image folder exists:", image_dir.exists())

        if image_dir.exists():
            print("PNG files:", len(list(image_dir.glob("*.png"))))

        print("First image:", data["images"][0]["file_name"])
        print("First image ID:", data["images"][0]["id"])
        print(
            "Unique annotated image IDs:",
            len({ann["image_id"] for ann in data["annotations"]})
        )
