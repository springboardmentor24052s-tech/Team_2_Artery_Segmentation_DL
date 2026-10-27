
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"

def inspect_split(task, split):
    base = DATASET / task / split
    annotation_file = base / "annotations" / f"{split}.json"
    image_dir = base / "images"

    print(f"\n{'=' * 55}")
    print(f"Task: {task} | Split: {split}")

    if not annotation_file.exists():
        print(f"Annotation file not found: {annotation_file}")
        return

    with annotation_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    print("Images in JSON:", len(images))
    print("Annotations:", len(annotations))
    print("Categories:", [c["name"] for c in categories])

    image_names = {img["id"]: img["file_name"] for img in images}
    missing = [
        name for name in image_names.values()
        if not (image_dir / name).exists()
    ]

    polygon_count = sum(
        bool(ann.get("segmentation"))
        for ann in annotations
    )

    print("Annotations with segmentation:", polygon_count)
    print("Image files missing:", len(missing))
    if missing:
        print("Examples:", missing[:5])

    print("Sample image records:", images[:2])
    print("Sample annotation:", annotations[:1])


if __name__ == "__main__":
    for task in ("stenosis", "syntax"):
        for split in ("train", "val", "test"):
            inspect_split(task, split)
