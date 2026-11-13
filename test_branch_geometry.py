
import cv2
import sys
import json
import csv
from pathlib import Path

sys.path.insert(0, "backend")

from app.services.centerline import extract_centerline
from app.services.branch_geometry import extract_vessel_branches


# Stenosis test image analyzed through the API.
# Vessel geometry is extracted from the Syntax U-Net prediction
# generated on this SAME stenosis image.
IMAGE_ID = "74996d2933414f26a69c7dfa254ad28f"
MODEL_TYPE = "syntax"

# Project directories
RESULTS_DIR = Path("backend/results")
PREDICTIONS_DIR = RESULTS_DIR / "api_predictions"

# Cleaned syntax mask for the selected image
mask_path = (
    PREDICTIONS_DIR /
    f"{IMAGE_ID}_cleaned_mask.png"
)

if not mask_path.exists():
    raise FileNotFoundError(
        f"Cleaned syntax mask not found: {mask_path}\n"
        "Run POST /api/analysis/postprocess/{image_id}"
        " with model_type=syntax first."
    )

mask = cv2.imread(
    str(mask_path),
    cv2.IMREAD_GRAYSCALE
)

if mask is None:
    raise ValueError("Could not read the cleaned syntax mask.")

print("\n========== INPUT INFORMATION ==========")
print("Image ID:", IMAGE_ID)
print("Model type:", MODEL_TYPE)
print("Mask path:", mask_path)
print("Mask dimensions:", mask.shape)
print("Foreground pixels:", int(cv2.countNonZero(mask)))

# Generate centerline from the cleaned syntax mask.
centerline_result = extract_centerline(mask)
centerline_mask = centerline_result["centerline_mask"]

# Extract vessel branches.
result = extract_vessel_branches(centerline_mask)

print("\n========== BRANCH EXTRACTION RESULTS ==========")
print("Skeleton pixels:", result["skeleton_pixel_count"])
print("Connected components:", result["connected_component_count"])
print("Endpoint count:", result["endpoint_count"])
print("Junction candidate pixels:", result["junction_pixel_count"])
print("Junction regions:", result["junction_region_count"])
print("Isolated pixels:", result["isolated_pixel_count"])
print("Branch count:", result["branch_count"])

branches = result["branches"]

if branches:
    lengths = [
        branch["pixel_count"]
        for branch in branches
    ]

    print("\n========== BRANCH LENGTH SUMMARY ==========")
    print("Shortest branch:", min(lengths))
    print("Longest branch:", max(lengths))
    print(
        "Average branch length:",
        round(sum(lengths) / len(lengths), 2)
    )
else:
    print("\nNo branches were extracted.")

print("\n========== INDIVIDUAL BRANCH DETAILS ==========")

for branch in branches:
    print(
        f"Branch {branch['branch_id']}: "
        f"pixels={branch['pixel_count']}, "
        f"start={branch['start_node']}, "
        f"end={branch['end_node']}"
    )

# Save image-specific reports to avoid overwriting
# results from other images.
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

json_path = (
    RESULTS_DIR /
    f"{IMAGE_ID}_branch_geometry.json"
)

csv_path = (
    RESULTS_DIR /
    f"{IMAGE_ID}_branch_geometry.csv"
)

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(
        {
            "image_id": IMAGE_ID,
            "input_image_task": "stenosis",
            "geometry_mask_model": MODEL_TYPE,
            "measurement_unit": "pixels",
            **result
        },
        file,
        indent=4
    )

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Image ID",
        "Geometry Mask Model",
        "Branch ID",
        "Start Node",
        "End Node",
        "Pixel Count"
    ])

    for branch in branches:
        writer.writerow([
            IMAGE_ID,
            MODEL_TYPE,
            branch["branch_id"],
            branch["start_node"],
            branch["end_node"],
            branch["pixel_count"]
        ])

print("\n========== OUTPUT FILES ==========")
print("JSON:", json_path)
print("CSV:", csv_path)

print("\nBranch geometry extraction completed.")
print(
    "Note: Geometry is based on a Syntax U-Net prediction "
    "on a stenosis-task image; it is experimental."
)
