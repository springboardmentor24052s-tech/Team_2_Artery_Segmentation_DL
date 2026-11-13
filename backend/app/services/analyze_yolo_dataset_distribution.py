
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "stenosis_yolo_dataset"
)

IMAGE_SIZE = 512
SPLITS = ["train", "val", "test"]


def analyze_split(split):

    label_dir = DATASET_ROOT / "labels" / split

    widths = []
    heights = []
    areas = []
    boxes_per_image = []

    for label_file in label_dir.glob("*.txt"):

        image_box_count = 0

        for line in label_file.read_text().splitlines():

            values = line.strip().split()

            if len(values) != 5:
                continue

            _, cx, cy, w, h = map(float, values)

            widths.append(w * IMAGE_SIZE)
            heights.append(h * IMAGE_SIZE)
            areas.append(w * h * IMAGE_SIZE ** 2)

            image_box_count += 1

        boxes_per_image.append(image_box_count)

    return {
        "widths": np.array(widths),
        "heights": np.array(heights),
        "areas": np.array(areas),
        "boxes_per_image": np.array(boxes_per_image)
    }


def main():

    results = {}

    print("=" * 65)
    print("STENOSIS DATASET DISTRIBUTION ANALYSIS")
    print("=" * 65)

    for split in SPLITS:

        data = analyze_split(split)
        results[split] = data

        areas = data["areas"]
        counts = data["boxes_per_image"]

        print(f"\n========== {split.upper()} ==========")

        print("Total images:", len(counts))
        print("Images with boxes:", np.sum(counts > 0))
        print("Images without boxes:", np.sum(counts == 0))
        print("Total stenosis boxes:", len(areas))

        if len(areas):

            print("Average box width:",
                  round(np.mean(data["widths"]), 2))

            print("Average box height:",
                  round(np.mean(data["heights"]), 2))

            print("Average box area:",
                  round(np.mean(areas), 2))

            print("Median box area:",
                  round(np.median(areas), 2))

            print("Area percentiles:",
                  np.percentile(areas, [25, 50, 75, 90]).round(2))

        print("Average boxes per image:",
              round(np.mean(counts), 2))

        print("Maximum boxes in one image:",
              int(np.max(counts)) if len(counts) else 0)

    # --------------------------------------------------------
    # Compare box-size distributions
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for split, data in results.items():

        if len(data["areas"]):

            plt.hist(
                data["areas"],
                bins=30,
                alpha=0.5,
                label=split
            )

    plt.xlabel("Bounding box area (pixels squared)")
    plt.ylabel("Number of boxes")
    plt.title("Stenosis Bounding Box Size Distribution")
    plt.legend()
    plt.tight_layout()

    output_dir = DATASET_ROOT / "distribution_analysis"
    output_dir.mkdir(exist_ok=True)

    plt.savefig(
        output_dir / "box_area_distribution.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Compare boxes per image
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for split, data in results.items():

        plt.hist(
            data["boxes_per_image"],
            bins=np.arange(0, max(10, int(np.max(data["boxes_per_image"])) + 2)) - 0.5,
            alpha=0.5,
            label=split
        )

    plt.xlabel("Stenosis boxes per image")
    plt.ylabel("Number of images")
    plt.title("Stenosis Annotation Count Per Image")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "boxes_per_image_distribution.png",
        dpi=150
    )

    plt.close()

    print("\nAnalysis graphs saved in:")
    print(output_dir)


if __name__ == "__main__":
    main()
