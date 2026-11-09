
import json
from pathlib import Path
from collections import defaultdict

# ============================================================
# PROJECT PATHS
# ============================================================

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

YOLO_ROOT = (
    BACKEND_ROOT
    / "results"
    / "stenosis_yolo_dataset"
)

CATEGORY_ID = 26
CLASS_ID = 0

SPLITS = ["train", "val", "test"]

# Tolerance for comparing pixel coordinates
TOLERANCE = 0.01


# ============================================================
# COCO TO YOLO CONVERSION
# ============================================================

def coco_to_yolo(bbox, image_width, image_height):

    x, y, width, height = map(float, bbox)

    x1 = max(0.0, min(x, image_width))
    y1 = max(0.0, min(y, image_height))

    x2 = max(0.0, min(x + width, image_width))
    y2 = max(0.0, min(y + height, image_height))

    clipped_width = x2 - x1
    clipped_height = y2 - y1

    if clipped_width <= 0 or clipped_height <= 0:
        return None

    return [
        (x1 + x2) / 2 / image_width,
        (y1 + y2) / 2 / image_height,
        clipped_width / image_width,
        clipped_height / image_height
    ]


# ============================================================
# YOLO TO PIXEL CONVERSION
# ============================================================

def yolo_to_pixel(values, image_width, image_height):

    cx, cy, bw, bh = values

    x1 = (cx - bw / 2) * image_width
    y1 = (cy - bh / 2) * image_height

    x2 = (cx + bw / 2) * image_width
    y2 = (cy + bh / 2) * image_height

    return [x1, y1, x2, y2]


def coco_to_pixel(bbox, image_width, image_height):

    x, y, width, height = map(float, bbox)

    x1 = max(0.0, min(x, image_width))
    y1 = max(0.0, min(y, image_height))

    x2 = max(0.0, min(x + width, image_width))
    y2 = max(0.0, min(y + height, image_height))

    return [x1, y1, x2, y2]


# ============================================================
# BOX COMPARISON
# ============================================================

def boxes_match(box1, box2):

    return all(
        abs(a - b) <= TOLERANCE
        for a, b in zip(box1, box2)
    )


# ============================================================
# AUDIT EACH SPLIT
# ============================================================

def audit_split(split):

    print("\n" + "=" * 65)
    print(f"AUDITING {split.upper()} SPLIT")
    print("=" * 65)

    annotation_file = (
        SOURCE_ROOT
        / split
        / "annotations"
        / f"{split}.json"
    )

    yolo_image_dir = YOLO_ROOT / "images" / split
    yolo_label_dir = YOLO_ROOT / "labels" / split

    if not annotation_file.exists():
        print("ERROR: COCO annotation file not found:", annotation_file)
        return

    if not yolo_image_dir.exists() or not yolo_label_dir.exists():
        print("ERROR: YOLO image or label directory not found.")
        return

    with open(annotation_file, "r", encoding="utf-8") as file:
        coco = json.load(file)

    images_by_id = {
        image["id"]: image
        for image in coco["images"]
    }

    annotations_by_image = defaultdict(list)

    for annotation in coco["annotations"]:

        if annotation["category_id"] == CATEGORY_ID:

            annotations_by_image[
                annotation["image_id"]
            ].append(annotation)

    # Map original image names to image information
    original_images = {
        Path(image["file_name"]).name: image
        for image in coco["images"]
    }

    stats = {
        "images_checked": 0,
        "boxes_checked": 0,
        "coordinate_mismatches": 0,
        "missing_label_files": 0,
        "missing_annotations": 0,
        "extra_annotations": 0,
        "dimension_mismatches": 0,
        "skipped_duplicates": 0
    }

    problems = []

    for image_name, image_info in original_images.items():

        yolo_image_path = yolo_image_dir / image_name

        if not yolo_image_path.exists():

            # Images removed by cross-split duplicate filtering
            stats["skipped_duplicates"] += 1
            continue

        stats["images_checked"] += 1

        image_id = image_info["id"]

        width = int(image_info["width"])
        height = int(image_info["height"])

        label_path = (
            yolo_label_dir
            / f"{Path(image_name).stem}.txt"
        )

        if not label_path.exists():

            stats["missing_label_files"] += 1

            problems.append(
                f"Missing label file: {image_name}"
            )

            continue

        # Load original image annotations
        original_boxes = []

        for annotation in annotations_by_image[image_id]:

            converted = coco_to_yolo(
                annotation["bbox"],
                width,
                height
            )

            if converted is not None:
                original_boxes.append(converted)

        # Load YOLO label annotations
        yolo_boxes = []

        lines = label_path.read_text(
            encoding="utf-8"
        ).strip().splitlines()

        for line in lines:

            if not line.strip():
                continue

            values = line.split()

            if len(values) != 5:
                problems.append(
                    f"Invalid label format: {image_name}"
                )
                continue

            class_id = int(values[0])

            if class_id != CLASS_ID:
                problems.append(
                    f"Unexpected class ID: {image_name}"
                )
                continue

            coordinates = list(map(float, values[1:]))

            yolo_boxes.append(coordinates)

        # Compare annotation counts
        if len(original_boxes) > len(yolo_boxes):

            difference = len(original_boxes) - len(yolo_boxes)

            stats["missing_annotations"] += difference

            problems.append(
                f"{image_name}: {difference} missing annotation(s)"
            )

        elif len(yolo_boxes) > len(original_boxes):

            difference = len(yolo_boxes) - len(original_boxes)

            stats["extra_annotations"] += difference

            problems.append(
                f"{image_name}: {difference} extra annotation(s)"
            )

        # Compare coordinates
        unmatched_yolo = list(yolo_boxes)

        for original_box in original_boxes:

            match_found = False

            for index, yolo_box in enumerate(unmatched_yolo):

                if boxes_match(original_box, yolo_box):

                    match_found = True
                    unmatched_yolo.pop(index)
                    break

            stats["boxes_checked"] += 1

            if not match_found:

                stats["coordinate_mismatches"] += 1

                problems.append(
                    f"{image_name}: coordinate mismatch"
                )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\nImages checked:", stats["images_checked"])
    print("Boxes checked:", stats["boxes_checked"])

    print(
        "Coordinate mismatches:",
        stats["coordinate_mismatches"]
    )

    print(
        "Missing label files:",
        stats["missing_label_files"]
    )

    print(
        "Missing annotations:",
        stats["missing_annotations"]
    )

    print(
        "Extra annotations:",
        stats["extra_annotations"]
    )

    print(
        "Skipped images (possible duplicates):",
        stats["skipped_duplicates"]
    )

    print("\nProblems found:", len(problems))

    for problem in problems[:20]:
        print("WARNING:", problem)

    return stats, problems


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("ARCADE COCO vs YOLO ANNOTATION AUDIT")
    print("=" * 65)

    all_results = {}

    for split in SPLITS:

        result = audit_split(split)

        if result is not None:
            stats, problems = result

            all_results[split] = {
                "statistics": stats,
                "problems": problems
            }

    # Save audit report separately
    report_path = YOLO_ROOT / "coco_yolo_audit_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=4)

    print("\n" + "=" * 65)
    print("ANNOTATION AUDIT COMPLETED")
    print("=" * 65)

    print("Audit report saved at:")
    print(report_path)


if __name__ == "__main__":
    main()
