
from pathlib import Path
import numpy as np

# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LABEL_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "stenosis_yolo_dataset"
    / "labels"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "stenosis_yolo_dataset"
    / "distribution_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 512
SPLITS = ["train", "val", "test"]

# COCO-style area thresholds
SMALL_AREA = 32 ** 2
MEDIUM_AREA = 96 ** 2

report_lines = []

def log(message=""):
    print(message)
    report_lines.append(str(message))


# --------------------------------------------------
# ANALYZE EACH DATASET SPLIT
# --------------------------------------------------

for split in SPLITS:

    label_dir = LABEL_ROOT / split

    areas = []
    boxes_per_image = []

    for label_file in label_dir.glob("*.txt"):

        image_box_count = 0

        for line in label_file.read_text().splitlines():

            values = line.strip().split()

            if len(values) != 5:
                continue

            _, x, y, w, h = map(float, values)

            width = w * IMAGE_SIZE
            height = h * IMAGE_SIZE

            area = width * height

            areas.append(area)
            image_box_count += 1

        boxes_per_image.append(image_box_count)

    areas = np.array(areas)
    boxes_per_image = np.array(boxes_per_image)

    total_images = len(boxes_per_image)
    total_boxes = len(areas)

    # --------------------------------------------------
    # BOX SIZE CATEGORIES
    # --------------------------------------------------

    small = np.sum(areas < SMALL_AREA)

    medium = np.sum(
        (areas >= SMALL_AREA) &
        (areas < MEDIUM_AREA)
    )

    large = np.sum(areas >= MEDIUM_AREA)

    # --------------------------------------------------
    # BOXES PER IMAGE
    # --------------------------------------------------

    zero_box_images = np.sum(boxes_per_image == 0)
    one_box_images = np.sum(boxes_per_image == 1)
    two_box_images = np.sum(boxes_per_image == 2)
    three_box_images = np.sum(boxes_per_image == 3)
    four_plus_images = np.sum(boxes_per_image >= 4)

    log("\n" + "=" * 55)
    log(f"{split.upper()} DISTRIBUTION STATISTICS")
    log("=" * 55)

    log(f"Total images: {total_images}")
    log(f"Total boxes: {total_boxes}")

    # --------------------------------------------------
    # AREA STATISTICS
    # --------------------------------------------------

    log("\nBOX AREA STATISTICS")

    log(f"Mean area: {np.mean(areas):.2f} px^2")
    log(f"Median area: {np.median(areas):.2f} px^2")
    log(f"90th percentile: {np.percentile(areas, 90):.2f} px^2")

    log("\nBOX SIZE CATEGORIES")

    categories = [
        ("Small (<32x32)", small),
        ("Medium (32x32 to 96x96)", medium),
        ("Large (>=96x96)", large)
    ]

    for name, count in categories:

        percentage = (count / total_boxes) * 100

        log(
            f"{name}: {count} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------
    # ANNOTATION COUNT PER IMAGE
    # --------------------------------------------------

    log("\nBOXES PER IMAGE")

    count_categories = [
        ("Zero boxes", zero_box_images),
        ("One box", one_box_images),
        ("Two boxes", two_box_images),
        ("Three boxes", three_box_images),
        ("Four or more boxes", four_plus_images)
    ]

    for name, count in count_categories:

        percentage = (count / total_images) * 100

        log(
            f"{name}: {count} "
            f"({percentage:.2f}%)"
        )

    log(
        f"Average boxes per image: "
        f"{np.mean(boxes_per_image):.2f}"
    )


# --------------------------------------------------
# SAVE REPORT
# --------------------------------------------------

REPORT_PATH = OUTPUT_DIR / "distribution_statistics.txt"

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print("\n" + "=" * 55)
print("DISTRIBUTION ANALYSIS COMPLETED")
print("=" * 55)
print("Report saved at:")
print(REPORT_PATH)
