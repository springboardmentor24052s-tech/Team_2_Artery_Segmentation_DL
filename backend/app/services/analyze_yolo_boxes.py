
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LABEL_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "stenosis_yolo_dataset"
    / "labels"
)

for split in ["train", "val", "test"]:
    label_dir = LABEL_ROOT / split

    widths = []
    heights = []
    areas = []

    for label_file in label_dir.glob("*.txt"):
        for line in label_file.read_text().splitlines():
            values = line.strip().split()

            if len(values) != 5:
                continue

            _, x, y, w, h = map(float, values)

            widths.append(w * 512)
            heights.append(h * 512)
            areas.append(w * h * 512 * 512)

    if not areas:
        print(f"\n{split.upper()}: No bounding boxes found")
        continue

    areas = np.array(areas)

    print(f"\n========== {split.upper()} ==========")
    print("Total boxes:", len(areas))
    print("Average width:", round(np.mean(widths), 2), "pixels")
    print("Average height:", round(np.mean(heights), 2), "pixels")
    print("Average area:", round(np.mean(areas), 2), "pixels")

    print("Small boxes (< 32x32 area):", np.sum(areas < 32 * 32))
    print("Medium boxes (32x32 to 96x96 area):",
          np.sum((areas >= 32 * 32) & (areas < 96 * 96)))
    print("Large boxes (>= 96x96 area):", np.sum(areas >= 96 * 96))

    print("Area percentiles:", np.percentile(areas, [25, 50, 75, 90]).round(2))
