
import json
from pathlib import Path

from PIL import Image, ImageDraw


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(r"C:\ArteryAI")
DATASET_DIR = PROJECT_ROOT / "dataset" / "syntax" / "train"
IMAGE_DIR = DATASET_DIR / "images"
ANNOTATION_FILE = DATASET_DIR / "annotations" / "train.json"

OUTPUT_DIR = PROJECT_ROOT / "training" / "mask_previews"

# Set to an image filename, e.g. "922.png".
# Set to None to use the first image in the JSON file.
IMAGE_FILENAME = "922.png"

# Exclude stenosis annotations from the anatomical vessel mask.
EXCLUDED_CATEGORIES = {"stenosis"}


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

def load_dataset():
    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {ANNOTATION_FILE}"
        )

    with ANNOTATION_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    categories = {
        category["id"]: category["name"]
        for category in data.get("categories", [])
    }

    images_by_name = {
        image["file_name"]: image
        for image in data.get("images", [])
    }

    if IMAGE_FILENAME is None:
        if not data.get("images"):
            raise ValueError("The annotation JSON contains no images.")

        image_info = data["images"][0]
    else:
        if IMAGE_FILENAME not in images_by_name:
            raise ValueError(
                f"{IMAGE_FILENAME} was not found in the annotation JSON."
            )

        image_info = images_by_name[IMAGE_FILENAME]

    image_path = IMAGE_DIR / image_info["file_name"]

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return data, categories, image_info, image_path


# --------------------------------------------------
# Build segmentation mask and print diagnostics
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data, categories, image_info, image_path = load_dataset()

    original = Image.open(image_path).convert("RGB")
    width, height = original.size

    print("=" * 60)
    print("ARCADE MASK INSPECTION")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Image ID: {image_info['id']}")
    print(f"Image size: {width} x {height}")
    print(f"Excluded categories: {EXCLUDED_CATEGORIES}")

    if (
        image_info.get("width") != width
        or image_info.get("height") != height
    ):
        print(
            "WARNING: JSON dimensions differ from the actual image size."
        )

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    image_id = image_info["id"]

    # Collect annotations for this image.
    annotations = [
        ann
        for ann in data.get("annotations", [])
        if ann.get("image_id") == image_id
    ]

    print(f"\nTotal annotations for image: {len(annotations)}")
    print("\nAnnotation details:")

    drawn_annotations = 0
    drawn_polygons = 0
    skipped_categories = 0
    missing_segmentations = 0

    for index, ann in enumerate(annotations, start=1):
        category_id = ann.get("category_id")
        category_name = categories.get(
            category_id, f"unknown_category_{category_id}"
        )

        segmentation = ann.get("segmentation", [])
        area = ann.get("area", "unknown")

        # COCO polygon segmentation is represented as a list of
        # flat coordinate arrays. Ignore unsupported RLE dictionaries.
        if isinstance(segmentation, dict):
            polygon_count = 0
        elif isinstance(segmentation, list):
            polygon_count = sum(
                1
                for polygon in segmentation
                if isinstance(polygon, list) and len(polygon) >= 6
            )
        else:
            polygon_count = 0

        print(
            f"{index:02d}. "
            f"category={category_name!r} | "
            f"category_id={category_id} | "
            f"polygons={polygon_count} | "
            f"area={area}"
        )

        if category_name.lower() in EXCLUDED_CATEGORIES:
            skipped_categories += 1
            continue

        if not isinstance(segmentation, list) or not segmentation:
            missing_segmentations += 1
            continue

        annotation_drawn = False

        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6:
                continue

            if len(polygon) % 2 != 0:
                print(
                    "WARNING: Skipping polygon with an odd "
                    f"coordinate count in annotation {ann.get('id')}."
                )
                continue

            points = [
                (polygon[i], polygon[i + 1])
                for i in range(0, len(polygon), 2)
            ]

            # Coordinates should correspond to the image dimensions.
            draw.polygon(points, fill=255)

            drawn_polygons += 1
            annotation_drawn = True

        if annotation_drawn:
            drawn_annotations += 1

    # --------------------------------------------------
    # Save output images
    # --------------------------------------------------

    overlay = original.copy()

    # Red color overlay, with transparency controlled by the mask.
    red_layer = Image.new("RGB", (width, height), (255, 0, 0))
    red_layer.putalpha(mask.point(lambda pixel: 90 if pixel > 0 else 0))

    overlay = Image.alpha_composite(
        overlay.convert("RGBA"),
        red_layer
    ).convert("RGB")

    original_output = OUTPUT_DIR / "original.png"
    mask_output = OUTPUT_DIR / "mask.png"
    overlay_output = OUTPUT_DIR / "overlay.png"

    original.save(original_output)
    mask.save(mask_output)
    overlay.save(overlay_output)

    foreground_pixels = sum(
        pixel > 0 for pixel in mask.getdata()
    )

    print("\n" + "=" * 60)
    print("MASK SUMMARY")
    print("=" * 60)
    print(f"Annotations found: {len(annotations)}")
    print(f"Annotations drawn: {drawn_annotations}")
    print(f"Polygons drawn: {drawn_polygons}")
    print(f"Excluded-category annotations: {skipped_categories}")
    print(f"Annotations without supported polygons: {missing_segmentations}")
    print(f"Mask foreground pixels: {foreground_pixels}")
    print(f"Original: {original_output}")
    print(f"Mask:     {mask_output}")
    print(f"Overlay:  {overlay_output}")
    print("\nPreview generation completed successfully.")


if __name__ == "__main__":
    main()
