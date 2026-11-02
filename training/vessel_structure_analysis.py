
from pathlib import Path
import cv2
import csv
import numpy as np

# Input: post-processed prediction masks
INPUT_DIR = Path("training/outputs/postprocessed_masks")

# Output: vessel structure analysis
OUTPUT_DIR = Path("training/outputs/vessel_structure_analysis")

# Ignore extremely small regions
MIN_COMPONENT_SIZE = 20


def analyse_vessel_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Could not read: {mask_path}")
        return None

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(binary, connectivity=8)
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
    visualization = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), visualization)


def main():
    if not INPUT_DIR.exists():
        print(f"Input folder not found: {INPUT_DIR}")
        return

    all_results = []

    for model_folder in sorted(INPUT_DIR.iterdir()):

        if not model_folder.is_dir():
            continue

        print(f"\nAnalysing: {model_folder.name}")

        for mask_path in sorted(model_folder.glob("*.png")):

            result = analyse_vessel_mask(mask_path)

            if result is None:
                continue

            binary, components = result

            image_name = mask_path.stem

            visualization_path = (
                OUTPUT_DIR
                / model_folder.name
                / "visualizations"
                / f"{image_name}_structure.png"
            )

            save_visualization(
                binary,
                components,
                visualization_path
            )

            all_results.append({
                "model": model_folder.name,
                "image": mask_path.name,
                "component_count": len(components),
                "total_component_area": sum(
                    item["area_pixels"] for item in components
                ),
                "components": components
            })

            print(
                f"{mask_path.name}: "
                f"{len(components)} components detected"
            )

    # Save component-level CSV
    csv_path = OUTPUT_DIR / "vessel_components.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "model",
            "image",
            "component_id",
            "area_pixels",
            "x",
            "y",
            "width",
            "height",
            "center_x",
            "center_y"
        ])

        for result in all_results:
            for component in result["components"]:
                writer.writerow([
                    result["model"],
                    result["image"],
                    component["component_id"],
                    component["area_pixels"],
                    component["x"],
                    component["y"],
                    component["width"],
                    component["height"],
                    component["center_x"],
                    component["center_y"]
                ])

    print("\n--------------------------------")
    print(f"Images analysed: {len(all_results)}")
    print(f"CSV saved: {csv_path}")
    print(f"Visualizations saved: {OUTPUT_DIR}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
