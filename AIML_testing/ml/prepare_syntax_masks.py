
from pathlib import Path
import json

import cv2
import numpy as np

#Generate binary masks for coronary artery segments from the arcade dataset's polygon annotations.
# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

SYNTAX_ROOT = (
    ROOT / "Dataset" / "arcade" / "arcade" / "syntax"
)

OUTPUT_ROOT = ROOT / "data" / "processed" / "syntax"
PREVIEW_ROOT = ROOT / "outputs" / "syntax_mask_previews"

SPLITS = ["train", "val", "test"]


def find_json(annotation_dir, split):
    """Find train.json, val.json or test.json, case-insensitively."""
    if not annotation_dir.exists():
        return None

    for path in annotation_dir.iterdir():
        if (
            path.is_file()
            and path.suffix.lower() == ".json"
            and path.stem.lower() == split.lower()
        ):
            return path

    return None


def rasterize_polygon(mask, polygon):
    """Draw one polygon onto a binary vessel mask."""
    coords = np.asarray(polygon, dtype=np.float32)

    if coords.size < 6 or coords.size % 2 != 0:
        return False

    points = coords.reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0], 0, mask.shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, mask.shape[0] - 1)

    points = np.round(points).astype(np.int32)
    cv2.fillPoly(mask, [points], 255)

    return True


def process_split(split):
    image_dir = SYNTAX_ROOT / split / "images"
    annotation_dir = SYNTAX_ROOT / split / "annotations"

    mask_dir = OUTPUT_ROOT / split / "masks"
    preview_dir = PREVIEW_ROOT / split

    mask_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        print(f"[SKIP] Image folder not found: {image_dir}")
        return

    json_path = find_json(annotation_dir, split)

    if json_path is None:
        print(f"[SKIP] Annotation JSON not found: {annotation_dir}")
        return

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    annotations_by_image = {}

    for annotation in data.get("annotations", []):
        image_id = str(annotation.get("image_id"))
        annotations_by_image.setdefault(image_id, []).append(annotation)

    print(f"\n--- {split.upper()} ---")
    print("JSON:", json_path)
    print("Images listed:", len(data.get("images", [])))
    print("Annotations:", len(data.get("annotations", [])))
    print("Categories:", len(data.get("categories", [])))

    processed = 0
    nonempty = 0
    blank = 0
    polygon_count = 0
    previews_saved = 0

    for image_info in data.get("images", []):
        image_id = str(image_info.get("id"))
        filename = Path(
            str(image_info.get("file_name", ""))
        ).name

        if not filename:
            continue

        image_path = image_dir / filename

        if not image_path.exists():
            print("[WARN] Missing image:", image_path)
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print("[WARN] Unreadable image:", filename)
            continue

        height, width = image.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        # Combine all labelled coronary artery segments
        for annotation in annotations_by_image.get(image_id, []):
            segmentation = annotation.get("segmentation", [])

            # This script expects polygon annotations, not RLE masks.
            if not isinstance(segmentation, list) or not segmentation:
                continue

            # A single polygon can be a flat list of coordinates.
            if isinstance(segmentation[0], (int, float)):
                polygons = [segmentation]
            else:
                polygons = segmentation

            for polygon in polygons:
                if rasterize_polygon(mask, polygon):
                    polygon_count += 1

        output_path = mask_dir / f"{Path(filename).stem}.png"

        if not cv2.imwrite(str(output_path), mask):
            raise RuntimeError(f"Failed to save mask: {output_path}")

        processed += 1

        if np.any(mask):
            nonempty += 1
        else:
            blank += 1

        # Save up to five overlays for visual verification
        if previews_saved < 5 and np.any(mask):
            original = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            overlay = original.copy()
            overlay[mask > 0] = (0, 255, 0)

            blended = cv2.addWeighted(
                original, 0.7, overlay, 0.3, 0
            )

            preview_path = (
                preview_dir / f"{Path(filename).stem}_overlay.png"
            )
            cv2.imwrite(str(preview_path), blended)
            previews_saved += 1

        if processed % 100 == 0:
            print(f"Processed {processed} images")

    print("Processed images:", processed)
    print("Non-empty masks:", nonempty)
    print("Blank masks:", blank)
    print("Polygons rasterized:", polygon_count)
    print("Masks saved to:", mask_dir)
    print("Preview images saved to:", preview_dir)


def main():
    if not SYNTAX_ROOT.exists():
        raise FileNotFoundError(
            f"Syntax dataset folder not found: {SYNTAX_ROOT}"
        )

    print("Project root:", ROOT)
    print("Syntax dataset:", SYNTAX_ROOT)

    for split in SPLITS:
        process_split(split)

    print("\nMask generation finished.")
    print("Check the preview overlays before training.")


if __name__ == "__main__":
    main()
