
from pathlib import Path
import cv2
import csv
import numpy as np

# Existing input folders
MASK_DIR = Path("training/outputs/postprocessed_masks")
CENTERLINE_DIR = Path("training/outputs/centerline_analysis")

# Output folder
OUTPUT_DIR = Path("training/outputs/vessel_diameter_analysis")

def estimate_diameter(mask_path, centerline_path):
    # Read binary vessel mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Read extracted centerline
    centerline = cv2.imread(
        str(centerline_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None or centerline is None:
        print(f"Could not read files for {mask_path.name}")
        return None

    # Convert images into binary masks
    _, binary_mask = cv2.threshold(
        mask, 127, 255, cv2.THRESH_BINARY
    )

    _, binary_centerline = cv2.threshold(
        centerline, 127, 255, cv2.THRESH_BINARY
    )

    # Calculate distance from each foreground pixel
    # to the nearest background pixel
    distance_map = cv2.distanceTransform(
        binary_mask,
        cv2.DIST_L2,
        5
    )

    # Get centerline pixel coordinates
    y_coords, x_coords = np.where(binary_centerline > 0)

    if len(x_coords) == 0:
        print(f"No centerline found: {mask_path.name}")
        return None

    # Diameter is approximately twice the distance
    # from the centerline to the nearest vessel boundary
    diameters = 2.0 * distance_map[y_coords, x_coords]

    # Remove invalid measurements
    diameters = diameters[diameters > 0]

    if len(diameters) == 0:
        return None

    return {
        "sample_count": int(len(diameters)),
        "mean_diameter_px": round(float(np.mean(diameters)), 3),
        "median_diameter_px": round(float(np.median(diameters)), 3),
        "min_diameter_px": round(float(np.min(diameters)), 3),
        "max_diameter_px": round(float(np.max(diameters)), 3)
    }


def create_visualization(mask_path, centerline_path, output_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    centerline = cv2.imread(
        str(centerline_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None or centerline is None:
        return

    visualization = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Display centreline in red
    visualization[centerline > 0] = (0, 0, 255)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), visualization)


def main():
    if not MASK_DIR.exists() or not CENTERLINE_DIR.exists():
        print("Required input folders are missing.")
        return

    results = []

    for model_folder in sorted(MASK_DIR.iterdir()):

        if not model_folder.is_dir():
            continue

        model_name = model_folder.name
        centerline_folder = CENTERLINE_DIR / model_name

        print(f"\nAnalysing model: {model_name}")

        if not centerline_folder.exists():
            print("Centerline folder not found.")
            continue

        for mask_path in sorted(model_folder.glob("prediction_*.png")):

            centerline_path = (
                centerline_folder /
                f"{mask_path.stem}_centerline.png"
            )

            if not centerline_path.exists():
                print(f"Missing centerline: {mask_path.name}")
                continue

            measurements = estimate_diameter(
                mask_path,
                centerline_path
            )

            if measurements is None:
                continue

            result = {
                "model": model_name,
                "image": mask_path.name,
                **measurements
            }

            results.append(result)

            print(
                f"{mask_path.name}: "
                f"Mean diameter = "
                f"{measurements['mean_diameter_px']} px, "
                f"Median = "
                f"{measurements['median_diameter_px']} px"
            )

            visualization_path = (
                OUTPUT_DIR /
                model_name /
                f"{mask_path.stem}_diameter.png"
            )

            create_visualization(
                mask_path,
                centerline_path,
                visualization_path
            )

    # Save measurements
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "vessel_diameter_measurements.csv"

    fieldnames = [
        "model",
        "image",
        "sample_count",
        "mean_diameter_px",
        "median_diameter_px",
        "min_diameter_px",
        "max_diameter_px"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n--------------------------------")
    print(f"Images analysed: {len(results)}")
    print(f"CSV report: {csv_path}")
    print(f"Visualizations: {OUTPUT_DIR}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
