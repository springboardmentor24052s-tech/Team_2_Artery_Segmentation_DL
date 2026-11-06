
from pathlib import Path
import csv

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "outputs"
    / "stenosis_postprocessed_masks"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "outputs"
    / "stenosis_vessel_structure"
)

MIN_COMPONENT_SIZE = 20


def analyse_vessel_mask(mask_path):
    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        print(f"Could not read: {mask_path}")
        return None

    _, binary = cv2.threshold(
        mask, 127, 255, cv2.THRESH_BINARY
    )

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    components = []

    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        if area < MIN_COMPONENT_SIZE:
            continue

        center_x, center_y = centroids[label]

        components.append({
            "component_id": label,
            "area_pixels": int(area),
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
            "center_x": round(float(center_x), 2),
            "center_y": round(float(center_y), 2)
        })

    return binary, components


def save_visualization(binary, components, output_path):
    visualization = cv2.cvtColor(
        binary,
        cv2.COLOR_GRAY2BGR
    )

    for component in components:
        x = component["x"]
        y = component["y"]
        width = component["width"]
        height = component["height"]

        cv2.rectangle(
            visualization,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            1
        )

        cv2.putText(
            visualization,
            str(component["component_id"]),
            (x, max(y - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(str(output_path), visualization)


def main():
    if not INPUT_DIR.exists():
        print(f"Input folder not found: {INPUT_DIR}")
        return

    mask_files = sorted(INPUT_DIR.glob("prediction_*.png"))

    if not mask_files:
        print("No processed masks found.")
        return

    component_rows = []
    summary_rows = []

    for mask_path in mask_files:
        result = analyse_vessel_mask(mask_path)

        if result is None:
            continue

        binary, components = result

        total_area = sum(
            item["area_pixels"] for item in components
        )

        summary_rows.append({
            "image": mask_path.name,
            "component_count": len(components),
            "total_component_area": total_area
        })

        for component in components:
            component_rows.append({
                "image": mask_path.name,
                **component
            })

        visualization_path = (
            OUTPUT_DIR
            / "visualizations"
            / f"{mask_path.stem}_structure.png"
        )

        save_visualization(
            binary,
            components,
            visualization_path
        )

        print(
            f"{mask_path.name}: "
            f"{len(components)} components | "
            f"Total component area: {total_area} pixels"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save one summary row per image, including empty masks.
    with open(
        OUTPUT_DIR / "vessel_summary.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "component_count",
                "total_component_area"
            ]
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    # Save detailed information for every component.
    with open(
        OUTPUT_DIR / "vessel_components.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        fieldnames = [
            "image",
            "component_id",
            "area_pixels",
            "x",
            "y",
            "width",
            "height",
            "center_x",
            "center_y"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(component_rows)

    print("\n--------------------------------")
    print(f"Images analysed: {len(summary_rows)}")
    print(f"Components found: {len(component_rows)}")
    print(f"Reports saved in: {OUTPUT_DIR}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
