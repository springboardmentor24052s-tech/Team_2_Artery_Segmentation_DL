
import cv2
from pathlib import Path
from app.services.centerline import extract_centerline

input_dir = Path("backend/results/postprocessing_comparison")
output_dir = Path("backend/results/centerline_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

for mask_path in sorted(input_dir.glob("*.png")):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    result = extract_centerline(mask)
    centerline = result["centerline_mask"]

    output_path = output_dir / mask_path.name
    cv2.imwrite(str(output_path), centerline)

    print(
        f"{mask_path.name}: "
        f"pixels={result['centerline_pixel_count']}, "
        f"endpoints={result['endpoint_count']}, "
        f"junctions={result['junction_count']}, "
        f"mean_diameter={result['diameter_statistics']['mean_diameter_pixels']}"
    )

    print("Saved:", output_path)

print("\nCenterline comparison completed.")
