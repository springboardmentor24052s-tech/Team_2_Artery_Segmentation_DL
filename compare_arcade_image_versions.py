
import cv2
import numpy as np

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "data"
    / "arcade"
    / "arcade"
    / "arcade"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "image_comparison"
)

SPLIT = "train"

IMAGE_IDS = [1, 2, 3, 4, 5]


def compare_images(image_id):

    syntax_path = (
        DATASET_ROOT / "syntax" / SPLIT
        / "images" / f"{image_id}.png"
    )

    stenosis_path = (
        DATASET_ROOT / "stenosis" / SPLIT
        / "images" / f"{image_id}.png"
    )

    syntax = cv2.imread(
        str(syntax_path),
        cv2.IMREAD_GRAYSCALE
    )

    stenosis = cv2.imread(
        str(stenosis_path),
        cv2.IMREAD_GRAYSCALE
    )

    if syntax is None or stenosis is None:
        print(f"Image {image_id}: Could not load images.")
        return

    if syntax.shape != stenosis.shape:
        print(f"Image {image_id}: Different dimensions.")
        return

    # Absolute pixel difference.
    difference = cv2.absdiff(syntax, stenosis)

    # Normalize difference for visualization.
    difference_visual = cv2.normalize(
        difference,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # Calculate basic similarity statistics.
    mae = np.mean(difference)

    correlation = np.corrcoef(
        syntax.flatten(),
        stenosis.flatten()
    )[0, 1]

    identical_percentage = (
        np.count_nonzero(syntax == stenosis)
        / syntax.size
    ) * 100

    # Display side-by-side images and difference.
    combined = np.hstack([
        syntax,
        stenosis,
        difference_visual
    ])

    output_path = (
        OUTPUT_DIR
        / f"comparison_{image_id}.png"
    )

    cv2.imwrite(str(output_path), combined)

    print(f"\nImage ID: {image_id}")
    print(f"Mean absolute difference: {mae:.3f}")
    print(f"Pixel correlation: {correlation:.4f}")
    print(f"Identical pixels: {identical_percentage:.2f}%")
    print("Saved:", output_path)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n========== IMAGE VERSION COMPARISON ==========")

    for image_id in IMAGE_IDS:
        compare_images(image_id)

    print("\nComparison completed.")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
