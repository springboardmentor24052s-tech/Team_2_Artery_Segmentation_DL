
import json
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

SPLIT = "train"
CHECK_LIMIT = 10

syntax_json = (
    DATASET_ROOT / "syntax" / SPLIT
    / "annotations" / f"{SPLIT}.json"
)

stenosis_json = (
    DATASET_ROOT / "stenosis" / SPLIT
    / "annotations" / f"{SPLIT}.json"
)

with open(syntax_json, "r", encoding="utf-8") as f:
    syntax_data = json.load(f)

with open(stenosis_json, "r", encoding="utf-8") as f:
    stenosis_data = json.load(f)

syntax_images = {
    image["id"]: image
    for image in syntax_data["images"]
}

stenosis_images = {
    image["id"]: image
    for image in stenosis_data["images"]
}

common_ids = sorted(
    set(syntax_images) & set(stenosis_images)
)

same_filename = 0
same_dimensions = 0
same_pixels = 0
checked = 0

print("\n========== IMAGE ALIGNMENT VALIDATION ==========\n")

for image_id in common_ids:

    if checked >= CHECK_LIMIT:
        break

    syntax_info = syntax_images[image_id]
    stenosis_info = stenosis_images[image_id]

    syntax_path = (
        DATASET_ROOT / "syntax" / SPLIT
        / "images" / syntax_info["file_name"]
    )

    stenosis_path = (
        DATASET_ROOT / "stenosis" / SPLIT
        / "images" / stenosis_info["file_name"]
    )

    print(f"\nImage ID: {image_id}")
    print("Syntax filename:", syntax_info["file_name"])
    print("Stenosis filename:", stenosis_info["file_name"])

    if syntax_info["file_name"] == stenosis_info["file_name"]:
        same_filename += 1
        print("Filename match: YES")
    else:
        print("Filename match: NO")

    syntax_image = cv2.imread(str(syntax_path), cv2.IMREAD_GRAYSCALE)
    stenosis_image = cv2.imread(str(stenosis_path), cv2.IMREAD_GRAYSCALE)

    if syntax_image is None or stenosis_image is None:
        print("Could not read one or both images.")
        checked += 1
        continue

    if syntax_image.shape == stenosis_image.shape:
        same_dimensions += 1
        print("Dimensions match: YES")

        if np.array_equal(syntax_image, stenosis_image):
            same_pixels += 1
            print("Pixel comparison: IDENTICAL")
        else:
            print("Pixel comparison: DIFFERENT")
    else:
        print("Dimensions match: NO")
        print("Syntax shape:", syntax_image.shape)
        print("Stenosis shape:", stenosis_image.shape)

    checked += 1

print("\n========== SUMMARY ==========")
print("Common image IDs:", len(common_ids))
print("Images checked:", checked)
print("Matching filenames:", same_filename)
print("Matching dimensions:", same_dimensions)
print("Pixel-identical images:", same_pixels)

print("\nVALIDATION COMPLETE")
