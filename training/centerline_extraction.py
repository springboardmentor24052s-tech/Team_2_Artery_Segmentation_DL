
from pathlib import Path
import cv2
import csv
import numpy as np

# Input folders
INPUT_DIR = Path("training/outputs/postprocessed_masks")

# Output folder
OUTPUT_DIR = Path("training/outputs/centerline_analysis")

MIN_COMPONENT_SIZE = 20


def skeletonize_mask(binary):
    """
    Extract a one-pixel-wide skeleton using
    morphological skeletonization.
    """

    image = binary.copy()
    skeleton = np.zeros_like(image)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS, (3, 3)
    )

    while cv2.countNonZero(image) > 0:

        eroded = cv2.erode(image, kernel)
        opened = cv2.dilate(eroded, kernel)

        difference = cv2.subtract(image, opened)
        skeleton = cv2.bitwise_or(skeleton, difference)

        image = eroded

    return skeleton


def analyse_centerline(skeleton):
    """
    Count approximate endpoints and branch points
    using the 8-neighbour pixel connectivity.
    """

    binary = (skeleton > 0).astype(np.uint8)

    neighbour_kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)

    neighbour_count = cv2.filter2D(
        binary,
        cv2.CV_16U,
        neighbour_kernel
    )

    endpoints = np.logical_and(
        binary == 1,
        neighbour_count == 1
    )

    branch_points = np.logical_and(
        binary == 1,
        neighbour_count >= 3
    )

    return {
        "skeleton_pixels": int(np.count_nonzero(binary)),
        "endpoint_count": int(np.count_nonzero(endpoints)),
        "branch_point_pixel_count": int(np.count_nonzero(branch_points))
    }


def process_mask(mask_path, model_name):
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

    # Remove very small isolated regions
    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
    )

    cleaned = np.zeros_like(binary)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= MIN_COMPONENT_SIZE:
            cleaned[labels == label] = 255

    # Extract centreline
    skeleton = skeletonize_mask(cleaned)

    # Analyse skeleton
    measurements = analyse_centerline(skeleton)

    # Save skeleton image
    output_folder = OUTPUT_DIR / model_name
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_folder / f"{mask_path.stem}_centerline.png"
    )

    cv2.imwrite(str(output_path), skeleton)

    return measurements


def main():

    if not INPUT_DIR.exists():
        print(f"Input folder not found: {INPUT_DIR}")
        return

    all_results = []

    for model_folder in sorted(INPUT_DIR.iterdir()):

        if not model_folder.is_dir():
            continue

        model_name = model_folder.name

        print(f"\nProcessing model: {model_name}")

        for mask_path in sorted(model_folder.glob("*.png")):

            measurements = process_mask(
                mask_path,
                model_name
            )

            if measurements is None:
                continue

            result = {
                "model": model_name,
                "image": mask_path.name,
                **measurements
            }

            all_results.append(result)

            print(
                f"{mask_path.name}: "
                f"Skeleton pixels = {measurements['skeleton_pixels']}, "
                f"Endpoints = {measurements['endpoint_count']}, "
                f"Branch-point pixels = "
                f"{measurements['branch_point_pixel_count']}"
            )

    # Save measurements to CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "centerline_measurements.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:

        fieldnames = [
            "model",
            "image",
            "skeleton_pixels",
            "endpoint_count",
            "branch_point_pixel_count"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\n--------------------------------")
    print(f"Images processed: {len(all_results)}")
    print(f"CSV report: {csv_path}")
    print(f"Skeleton images: {OUTPUT_DIR}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
