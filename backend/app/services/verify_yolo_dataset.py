
import json
import random
from pathlib import Path
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

DATASET_ROOT = (
    BACKEND_ROOT / "results" / "stenosis_yolo_dataset"
)

SPLITS = ["train", "val", "test"]


def verify_labels():

    print("=" * 60)
    print("YOLO LABEL VERIFICATION")
    print("=" * 60)

    total_images = 0
    total_boxes = 0
    errors = []

    for split in SPLITS:

        image_dir = DATASET_ROOT / "images" / split
        label_dir = DATASET_ROOT / "labels" / split

        image_files = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        )

        stats = Counter()

        for image_path in image_files:

            total_images += 1

            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                errors.append(
                    f"Missing label file: {label_path}"
                )
                continue

            lines = label_path.read_text(
                encoding="utf-8"
            ).strip().splitlines()

            if not lines or lines == [""]:
                stats["empty_labels"] += 1
                continue

            for line_number, line in enumerate(lines, start=1):

                values = line.split()

                if len(values) != 5:
                    errors.append(
                        f"{label_path}: line {line_number} "
                        "must contain exactly 5 values"
                    )
                    continue

                try:
                    class_id = int(values[0])
                    coords = [float(v) for v in values[1:]]
                except ValueError:
                    errors.append(
                        f"Invalid numeric value: {label_path}"
                    )
                    continue

                if class_id != 0:
                    errors.append(
                        f"Unexpected class ID in {label_path}"
                    )

                if not all(0 <= v <= 1 for v in coords):
                    errors.append(
                        f"Coordinates outside [0,1]: {label_path}"
                    )

                if coords[2] <= 0 or coords[3] <= 0:
                    errors.append(
                        f"Invalid box dimensions: {label_path}"
                    )

                stats["boxes"] += 1
                total_boxes += 1

        print(f"\n{split.upper()}")
        print("Images:", len(image_files))
        print("Boxes:", stats["boxes"])
        print("Empty labels:", stats["empty_labels"])

    print("\n" + "=" * 60)
    print("TOTAL IMAGES:", total_images)
    print("TOTAL BOXES:", total_boxes)
    print("ERROR COUNT:", len(errors))

    if errors:
        for error in errors[:30]:
            print("ERROR:", error)
    else:
        print("All YOLO labels passed basic validation.")

    return errors


def create_visual_samples(split="train", count=12):

    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split

    image_files = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        and (label_dir / f"{p.stem}.txt").exists()
        and (label_dir / f"{p.stem}.txt").stat().st_size > 0
    ]

    if not image_files:
        print("No annotated images available.")
        return

    selected = random.Random(42).sample(
        image_files,
        min(count, len(image_files))
    )

    rows = 3
    cols = 4

    fig, axes = plt.subplots(
        rows, cols, figsize=(16, 12)
    )

    axes = axes.flatten()

    for ax, image_path in zip(axes, selected):

        image = cv2.imread(str(image_path))

        if image is None:
            ax.axis("off")
            continue

        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )

        height, width = image.shape[:2]

        ax.imshow(image)

        label_path = (
            label_dir / f"{image_path.stem}.txt"
        )

        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines():

            values = line.split()

            if len(values) != 5:
                continue

            _, cx, cy, bw, bh = map(float, values)

            x = (cx - bw / 2) * width
            y = (cy - bh / 2) * height

            box_width = bw * width
            box_height = bh * height

            rect = patches.Rectangle(
                (x, y),
                box_width,
                box_height,
                linewidth=2,
                edgecolor="red",
                facecolor="none"
            )

            ax.add_patch(rect)

        ax.set_title(image_path.name, fontsize=8)
        ax.axis("off")

    for ax in axes[len(selected):]:
        ax.axis("off")

    plt.tight_layout()

    output_dir = DATASET_ROOT / "verification"
    output_dir.mkdir(exist_ok=True)

    output_path = (
        output_dir / f"{split}_yolo_bbox_samples.png"
    )

    plt.savefig(output_path, dpi=150)
    plt.close()

    print("\nSaved visual verification:", output_path)


if __name__ == "__main__":

    errors = verify_labels()

    if not errors:
        create_visual_samples("train")
        create_visual_samples("val")
