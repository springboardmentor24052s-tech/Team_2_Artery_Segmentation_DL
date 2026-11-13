
import cv2
import csv
from pathlib import Path

from app.services.postprocessing import postprocess_mask
from app.services.centerline import extract_centerline

mask_dir = Path(
    "backend/results/arcade_masks/syntax/test"
)

output_csv = Path(
    "backend/results/geometry_syntax_sample_evaluation.csv"
)

rows = []

for image_number in range(1, 11):

    mask_path = mask_dir / f"{image_number}_mask.png"

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print("Missing:", mask_path)
        continue

    processed = postprocess_mask(
        mask,
        min_component_area=20,
        kernel_size=3
    )

    geometry = extract_centerline(processed["mask"])

    stats = geometry["diameter_statistics"]

    row = {
        "image": image_number,
        "original_pixels": processed["original_foreground_pixels"],
        "cleaned_pixels": processed["cleaned_foreground_pixels"],
        "components": processed["component_count"],
        "centerline_pixels": geometry["centerline_pixel_count"],
        "endpoints": geometry["endpoint_count"],
        "junctions": geometry["junction_count"],
        "mean_diameter": stats["mean_diameter_pixels"],
        "minimum_diameter": stats["minimum_diameter_pixels"],
        "maximum_diameter": stats["maximum_diameter_pixels"]
    }

    rows.append(row)

    print(row)

if rows:
    with open(output_csv, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

print("\nSYNTAX GEOMETRY EVALUATION COMPLETED")
print("Images evaluated:", len(rows))
print("Report saved:", output_csv)
