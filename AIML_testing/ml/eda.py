
from pathlib import Path
import json
from collections import Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. DATASET PATH
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "arcade"
    / "arcade"
    / "stenosis"
)

STENOSIS_CATEGORY_ID = 26

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. FIND ANNOTATION JSON
# --------------------------------------------------

def find_annotation_file(split_path, split_name):
    """Find the COCO annotation JSON for a dataset split."""

    expected = split_path / f"{split_name}.json"

    if expected.exists():
        return expected

    json_files = list(split_path.glob("*.json"))

    if not json_files:
        json_files = list(split_path.rglob("*.json"))

    if json_files:
        return json_files[0]

    return None


# --------------------------------------------------
# 3. INSPECT DATASET SPLIT
# --------------------------------------------------

def inspect_split(split_name):
    split_path = DATASET_ROOT / split_name
    image_dir = split_path / "images"

    print("\n" + "=" * 60)
    print(f"DATASET SPLIT: {split_name.upper()}")
    print("=" * 60)

    if not split_path.exists():
        print(f"Split folder not found: {split_path}")
        return None

    # Read image files
    images = []

    if image_dir.exists():
        images = sorted(
            [
                p for p in image_dir.iterdir()
                if p.suffix.lower() in
                {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            ],
            key=lambda p: (
                0, int(p.stem)
            ) if p.stem.isdigit() else (
                1, p.stem.lower()
            )
        )

    print("Split path:", split_path)
    print("Image folder exists:", image_dir.exists())
    print("Image count:", len(images))

    # Find JSON
    annotation_file = find_annotation_file(
        split_path, split_name
    )

    data = None

    if annotation_file:
        print("Annotation file:", annotation_file)

        with annotation_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        coco_images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])

        print("Images listed in JSON:", len(coco_images))
        print("Total annotations:", len(annotations))
        print("Total categories:", len(categories))

        # Print categories
        print("\nCategories:")

        for category in categories:
            print(
                f"  ID: {category.get('id')} | "
                f"Name: {category.get('name')}"
            )

        # Count annotations per category
        category_counts = Counter(
            ann.get("category_id")
            for ann in annotations
        )

        print("\nAnnotations per category:")

        for category in categories:
            cid = category.get("id")
            name = category.get("name")
            print(f"  {name}: {category_counts[cid]}")

        stenosis_annotations = [
            ann for ann in annotations
            if ann.get("category_id") == STENOSIS_CATEGORY_ID
        ]

        print(
            "\nStenosis annotations (category 26):",
            len(stenosis_annotations)
        )

        # Inspect first annotation
        if annotations:
            first = annotations[0]

            print("\nFirst annotation keys:")
            print(list(first.keys()))

            print(
                "First annotation category:",
                first.get("category_id")
            )

            print(
                "Segmentation type:",
                type(first.get("segmentation")).__name__
            )

            print(
                "Bounding box:",
                first.get("bbox")
            )

            print(
                "Area:",
                first.get("area")
            )

        # Check annotation/image filename matching
        image_names = {p.name for p in images}
        json_names = {
            img.get("file_name")
            for img in coco_images
        }

        print(
            "JSON filenames matching image filenames:",
            len(image_names.intersection(json_names))
        )

        print(
            "Images absent from JSON filename list:",
            len(image_names - json_names)
        )

        # Save split summary
        summary = {
            "split": split_name,
            "image_count": len(images),
            "json_image_count": len(coco_images),
            "annotation_count": len(annotations),
            "category_count": len(categories),
            "stenosis_annotation_count": len(stenosis_annotations),
            "annotation_file": str(annotation_file),
        }

        summary_path = OUTPUT_DIR / f"{split_name}_summary.json"

        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        print("Summary saved:", summary_path)

    else:
        print("No annotation JSON found for this split.")

    # Inspect image dimensions and pixel information
    if images:
        dimensions = Counter()
        unreadable = 0

        for image_path in images:
            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_UNCHANGED
            )

            if image is None:
                unreadable += 1
                continue

            dimensions[
                (image.shape[1], image.shape[0])
            ] += 1

        print("\nImage dimensions (width, height):")

        for dimension, count in dimensions.items():
            print(f"  {dimension}: {count} images")

        print("Unreadable images:", unreadable)

        sample_path = images[0]
        sample = cv2.imread(
            str(sample_path),
            cv2.IMREAD_UNCHANGED
        )

        if sample is not None:
            print("\nSample image:", sample_path.name)
            print("Image shape:", sample.shape)
            print("Data type:", sample.dtype)
            print(
                "Pixel range:",
                int(sample.min()),
                "to",
                int(sample.max())
            )

            if sample.ndim == 2:
                display = sample
                cmap = "gray"
            elif sample.shape[2] == 4:
                display = cv2.cvtColor(
                    sample, cv2.COLOR_BGRA2RGBA
                )
                cmap = None
            else:
                display = cv2.cvtColor(
                    sample, cv2.COLOR_BGR2RGB
                )
                cmap = None

            plt.figure(figsize=(6, 6))
            plt.imshow(display, cmap=cmap)
            plt.title(f"{split_name}: {sample_path.name}")
            plt.axis("off")
            plt.tight_layout()

            sample_output = OUTPUT_DIR / f"{split_name}_sample.png"
            plt.savefig(sample_output, dpi=150)
            plt.show()
            plt.close()

            print("Sample image saved:", sample_output)

    return data


# --------------------------------------------------
# 4. VISUALIZE STENOSIS POLYGON
# --------------------------------------------------

def visualize_stenosis_annotation():
    print("\n" + "=" * 60)
    print("STENOSIS ANNOTATION VISUALIZATION")
    print("=" * 60)

    split_path = DATASET_ROOT / "train"
    image_dir = split_path / "images"

    annotation_file = find_annotation_file(
        split_path, "train"
    )

    if annotation_file is None:
        print("Training annotation JSON not found.")
        return

    with annotation_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = [
        ann for ann in data.get("annotations", [])
        if ann.get("category_id") == STENOSIS_CATEGORY_ID
        and ann.get("segmentation")
    ]

    if not annotations:
        print("No stenosis segmentation annotations found.")
        return

    # Find an annotation with polygon data
    selected = None

    for ann in annotations:
        segmentation = ann.get("segmentation")

        if (
            isinstance(segmentation, list)
            and len(segmentation) > 0
            and isinstance(segmentation[0], list)
            and len(segmentation[0]) >= 6
        ):
            selected = ann
            break

    if selected is None:
        print(
            "No polygon segmentation found in the expected format."
        )
        print("Example segmentation:", annotations[0].get("segmentation"))
        return

    image_info = next(
        (
            img for img in data.get("images", [])
            if img.get("id") == selected.get("image_id")
        ),
        None
    )

    if image_info is None:
        print("Could not match annotation to image metadata.")
        return

    image_path = image_dir / image_info["file_name"]

    if not image_path.exists():
        print("Image file not found:", image_path)
        return

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read image:", image_path)
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a binary mask from the selected polygons
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for polygon in selected["segmentation"]:
        points = np.asarray(
            polygon, dtype=np.float32
        ).reshape(-1, 2)

        points = np.round(points).astype(np.int32)

        cv2.fillPoly(mask, [points], 255)

    # Red overlay for visual inspection
    overlay = image.copy()
    overlay[mask > 0] = [255, 0, 0]

    blended = cv2.addWeighted(
        image, 0.70, overlay, 0.30, 0
    )

    print("Image:", image_info["file_name"])
    print("Category ID:", selected["category_id"])
    print("Polygon count:", len(selected["segmentation"]))
    print("Mask foreground pixels:", int(np.count_nonzero(mask)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Stenosis Mask")

    axes[2].imshow(blended)
    axes[2].set_title("Annotation Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "stenosis_annotation_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    print("Visualization saved:", output_path)


# --------------------------------------------------
# 5. MAIN
# --------------------------------------------------

def main():
    print("ARCADE STENOSIS DATASET EDA")
    print("Dataset path:", DATASET_ROOT)
    print("Dataset exists:", DATASET_ROOT.exists())

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_ROOT}"
        )

    for split_name in ["train", "val", "test"]:
        inspect_split(split_name)

    visualize_stenosis_annotation()

    print("\nEDA completed.")
    print("EDA outputs saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
