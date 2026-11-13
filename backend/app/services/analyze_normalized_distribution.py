
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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

area_data = {}
count_data = {}

# --------------------------------------------------
# 1. LOAD DATASET DISTRIBUTION
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

            # Convert normalized box dimensions to pixels
            width = w * IMAGE_SIZE
            height = h * IMAGE_SIZE

            area = width * height

            areas.append(area)
            image_box_count += 1

        boxes_per_image.append(image_box_count)

    area_data[split] = np.array(areas)
    count_data[split] = np.array(boxes_per_image)

    total_images = len(boxes_per_image)
    total_boxes = len(areas)

    images_with_boxes = np.count_nonzero(
        np.array(boxes_per_image) > 0
    )

    print(f"\n========== {split.upper()} ==========")
    print("Total images:", total_images)
    print("Images with boxes:", images_with_boxes)
    print("Images without boxes:", total_images - images_with_boxes)
    print("Total stenosis boxes:", total_boxes)

# --------------------------------------------------
# 2. NORMALIZED BOUNDING-BOX AREA DISTRIBUTION
#    COMMON BIN BOUNDARIES FOR ALL SPLITS
# --------------------------------------------------

plt.figure(figsize=(12, 7))

all_areas = np.concatenate([
    area_data[split] for split in SPLITS
])

common_bins = np.linspace(
    0,
    all_areas.max(),
    31
)

for split in SPLITS:

    areas = area_data[split]

    if len(areas) == 0:
        continue

    weights = np.ones(len(areas)) / len(areas)

    plt.hist(
        areas,
        bins=common_bins,
        weights=weights,
        alpha=0.5,
        label=split
    )

plt.title("Normalized Stenosis Bounding Box Size Distribution")
plt.xlabel("Bounding box area (pixels squared)")
plt.ylabel("Proportion of boxes")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "normalized_box_area_distribution.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 3. NORMALIZED BOXES-PER-IMAGE DISTRIBUTION
#    COMMON BIN BOUNDARIES
# --------------------------------------------------

plt.figure(figsize=(12, 7))

max_count = max(
    np.max(count_data[split])
    for split in SPLITS
)

count_bins = np.arange(
    -0.5,
    max_count + 1.5,
    1
)

for split in SPLITS:

    counts = count_data[split]

    if len(counts) == 0:
        continue

    weights = np.ones(len(counts)) / len(counts)

    plt.hist(
        counts,
        bins=count_bins,
        weights=weights,
        alpha=0.5,
        label=split
    )

plt.title("Normalized Stenosis Annotation Count Per Image")
plt.xlabel("Stenosis boxes per image")
plt.ylabel("Proportion of images")

plt.xticks(
    np.arange(0, max_count + 1, 1)
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "normalized_boxes_per_image_distribution.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 4. FINAL OUTPUT
# --------------------------------------------------

print("\n" + "=" * 60)
print("NORMALIZED DISTRIBUTION ANALYSIS COMPLETED")
print("=" * 60)

print("\nGenerated graphs:")

print(
    OUTPUT_DIR / "normalized_box_area_distribution.png"
)

print(
    OUTPUT_DIR / "normalized_boxes_per_image_distribution.png"
)

print("\nOutput folder:", OUTPUT_DIR)
