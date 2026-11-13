
import json
import hashlib
import shutil
from pathlib import Path
from collections import Counter, defaultdict

# ============================================================
# PATHS AND SETTINGS
# ============================================================

# Project root:
# Team_2_Artery_Segmentation_DL
PROJECT_ROOT = Path(__file__).resolve().parents[3]

BACKEND_ROOT = PROJECT_ROOT / "backend"

SOURCE_ROOT = (
    BACKEND_ROOT
    / "app"
    / "data"
    / "arcade"
    / "arcade"
    / "arcade"
    / "stenosis"
)

OUTPUT_ROOT = (
    BACKEND_ROOT
    / "results"
    / "stenosis_yolo_dataset"
)

CATEGORY_ID = 26
CLASS_ID = 0
CLASS_NAME = "stenosis"

SPLITS = ["train", "val", "test"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_hash(image_path):
    digest = hashlib.sha256()

    with open(image_path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def find_image_path(image_dir, file_name):
    image_path = image_dir / file_name

    if image_path.exists():
        return image_path

    matches = list(
        image_dir.rglob(Path(file_name).name)
    )

    if matches:
        return matches[0]

    return None


def convert_bbox_to_yolo(bbox, image_width, image_height):

    if len(bbox) != 4:
        return None

    x, y, width, height = map(float, bbox)

    if width <= 0 or height <= 0:
        return None

    # Original COCO coordinates
    x1 = max(0.0, min(x, image_width))
    y1 = max(0.0, min(y, image_height))
    x2 = max(0.0, min(x + width, image_width))
    y2 = max(0.0, min(y + height, image_height))

    clipped_width = x2 - x1
    clipped_height = y2 - y1

    if clipped_width <= 0 or clipped_height <= 0:
        return None

    # YOLO normalized center coordinates
    center_x = (x1 + x2) / 2 / image_width
    center_y = (y1 + y2) / 2 / image_height

    normalized_width = clipped_width / image_width
    normalized_height = clipped_height / image_height

    values = [
        center_x,
        center_y,
        normalized_width,
        normalized_height
    ]

    if not all(0 <= value <= 1 for value in values):
        return None

    return values


# ============================================================
# DATASET PREPARATION
# ============================================================

def main():

    print("=" * 60)
    print("ARCADE STENOSIS YOLO DATASET PREPARATION")
    print("=" * 60)

    print("\nProject root:", PROJECT_ROOT)
    print("Source root:", SOURCE_ROOT)
    print("Output root:", OUTPUT_ROOT)

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {SOURCE_ROOT}"
        )

    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"\nOutput folder already exists:\n{OUTPUT_ROOT}\n"
            "Rename or move the existing folder before "
            "running this script again."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    seen_hashes = {}

    report = {
        "class_id": CLASS_ID,
        "class_name": CLASS_NAME,
        "category_id": CATEGORY_ID,
        "splits": {},
        "duplicates": [],
        "invalid_boxes": [],
        "missing_images": [],
        "invalid_dimensions": []
    }

    # ========================================================
    # PROCESS EACH SPLIT
    # ========================================================

    for split in SPLITS:

        print("\n" + "-" * 60)
        print(f"PROCESSING {split.upper()}")
        print("-" * 60)

        split_root = SOURCE_ROOT / split

        annotation_file = (
            split_root / "annotations" / f"{split}.json"
        )

        image_dir = split_root / "images"

        if not annotation_file.exists():
            raise FileNotFoundError(annotation_file)

        if not image_dir.exists():
            raise FileNotFoundError(image_dir)

        with open(
            annotation_file,
            "r",
            encoding="utf-8"
        ) as file:
            coco = json.load(file)

        # Verify category
        categories = {
            item["id"]: item["name"]
            for item in coco["categories"]
        }

        if categories.get(CATEGORY_ID) != CLASS_NAME:
            raise ValueError(
                f"Expected category {CATEGORY_ID}: "
                f"{CLASS_NAME}, found "
                f"{categories.get(CATEGORY_ID)}"
            )

        output_images = (
            OUTPUT_ROOT / "images" / split
        )

        output_labels = (
            OUTPUT_ROOT / "labels" / split
        )

        output_images.mkdir(
            parents=True,
            exist_ok=True
        )

        output_labels.mkdir(
            parents=True,
            exist_ok=True
        )

        # Group stenosis annotations by image ID
        annotations_by_image = defaultdict(list)

        for annotation in coco["annotations"]:

            if annotation["category_id"] == CATEGORY_ID:

                annotations_by_image[
                    annotation["image_id"]
                ].append(annotation)

        stats = Counter()

        # ====================================================
        # PROCESS IMAGES
        # ====================================================

        for image_info in coco["images"]:

            image_id = image_info["id"]
            file_name = image_info["file_name"]

            source_image = find_image_path(
                image_dir,
                file_name
            )

            if source_image is None:

                report["missing_images"].append({
                    "split": split,
                    "file_name": file_name
                })

                stats["missing_images"] += 1
                continue

            # Duplicate image protection
            image_hash = calculate_hash(source_image)

            if image_hash in seen_hashes:

                report["duplicates"].append({
                    "split": split,
                    "file_name": file_name,
                    "duplicate_of": seen_hashes[image_hash]
                })

                stats["duplicate_images"] += 1
                continue

            image_width = int(image_info["width"])
            image_height = int(image_info["height"])

            if image_width <= 0 or image_height <= 0:

                report["invalid_dimensions"].append({
                    "split": split,
                    "file_name": file_name,
                    "width": image_width,
                    "height": image_height
                })

                stats["invalid_dimensions"] += 1
                continue

            seen_hashes[image_hash] = {
                "split": split,
                "file_name": file_name
            }

            label_lines = []

            # Convert all annotations for this image
            for annotation in annotations_by_image[image_id]:

                converted = convert_bbox_to_yolo(
                    annotation["bbox"],
                    image_width,
                    image_height
                )

                if converted is None:

                    report["invalid_boxes"].append({
                        "split": split,
                        "image": file_name,
                        "annotation_id": annotation["id"]
                    })

                    stats["invalid_boxes"] += 1
                    continue

                label_line = (
                    f"{CLASS_ID} "
                    + " ".join(
                        f"{value:.8f}"
                        for value in converted
                    )
                )

                label_lines.append(label_line)

            # Copy original image
            destination_image = (
                output_images / source_image.name
            )

            if destination_image.exists():
                raise FileExistsError(destination_image)

            shutil.copy2(
                source_image,
                destination_image
            )

            # Save YOLO label file.
            # Empty file means no annotated stenosis box.
            label_file = (
                output_labels
                / f"{source_image.stem}.txt"
            )

            label_file.write_text(
                "\n".join(label_lines),
                encoding="utf-8"
            )

            stats["images"] += 1
            stats["annotations"] += len(label_lines)

            if label_lines:
                stats["images_with_annotations"] += 1
            else:
                stats["images_without_annotations"] += 1

        # Save split statistics
        report["splits"][split] = {
            "images": stats["images"],
            "images_with_annotations": (
                stats["images_with_annotations"]
            ),
            "images_without_annotations": (
                stats["images_without_annotations"]
            ),
            "annotations": stats["annotations"],
            "duplicate_images": stats["duplicate_images"],
            "missing_images": stats["missing_images"],
            "invalid_boxes": stats["invalid_boxes"],
            "invalid_dimensions": stats["invalid_dimensions"]
        }

        print("Images:", stats["images"])
        print("Images with annotations:",
              stats["images_with_annotations"])
        print("Images without annotations:",
              stats["images_without_annotations"])
        print("Annotations:", stats["annotations"])
        print("Duplicate images:", stats["duplicate_images"])
        print("Missing images:", stats["missing_images"])
        print("Invalid boxes:", stats["invalid_boxes"])

    # ========================================================
    # CREATE YOLO DATA YAML
    # ========================================================

    yaml_content = f"""path: {OUTPUT_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: {CLASS_NAME}
"""

    yaml_path = OUTPUT_ROOT / "data.yaml"

    yaml_path.write_text(
        yaml_content,
        encoding="utf-8"
    )

    # ========================================================
    # SAVE DATASET REPORT
    # ========================================================

    report_path = (
        OUTPUT_ROOT / "dataset_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("YOLO DATASET PREPARATION COMPLETED")
    print("=" * 60)

    total_images = 0
    total_annotations = 0

    for split, values in report["splits"].items():

        total_images += values["images"]
        total_annotations += values["annotations"]

        print(
            f"{split.upper()}: "
            f"Images={values['images']}, "
            f"Annotations={values['annotations']}, "
            f"Duplicates={values['duplicate_images']}"
        )

    print("\nTOTAL IMAGES:", total_images)
    print("TOTAL ANNOTATIONS:", total_annotations)

    print(
        "TOTAL DUPLICATES:",
        len(report["duplicates"])
    )

    print(
        "TOTAL MISSING IMAGES:",
        len(report["missing_images"])
    )

    print(
        "TOTAL INVALID BOXES:",
        len(report["invalid_boxes"])
    )

    print("\nDataset:", OUTPUT_ROOT)
    print("Configuration:", yaml_path)
    print("Report:", report_path)

    print("\nNext step: Verify YOLO labels before training.")


if __name__ == "__main__":
    main()
