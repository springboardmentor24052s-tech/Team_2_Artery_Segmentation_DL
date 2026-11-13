
import cv2
from pathlib import Path
from app.services.postprocessing import postprocess_mask

image_path = Path(
    "backend/results/api_predictions/"
    "16c135650bb24194a3abfd836c03237c_mask.png"
)

output_dir = Path(
    "backend/results/postprocessing_comparison"
)
output_dir.mkdir(parents=True, exist_ok=True)

mask = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise FileNotFoundError(image_path)

for kernel_size in (3, 5):
    for min_area in (5, 10, 20):
        result = postprocess_mask(
            mask,
            min_component_area=min_area,
            kernel_size=kernel_size
        )

        output_path = output_dir / (
            f"kernel_{kernel_size}_minarea_{min_area}.png"
        )

        cv2.imwrite(str(output_path), result["mask"])

        print("Saved:", output_path)

print("\nAll six comparison masks generated.")
