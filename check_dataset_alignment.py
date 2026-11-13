
import cv2
import numpy as np
from pathlib import Path

stenosis_image_path = Path(
    "backend/app/data/arcade/arcade/arcade/"
    "stenosis/test/images/1.png"
)

syntax_image_path = Path(
    "backend/app/data/arcade/arcade/arcade/"
    "syntax/test/images/1.png"
)

gt_mask_path = Path(
    "backend/results/arcade_masks/syntax/test/1_mask.png"
)

pred_mask_path = Path(
    "backend/results/api_predictions/"
    "16c135650bb24194a3abfd836c03237c_cleaned_mask.png"
)

stenosis_image = cv2.imread(
    str(stenosis_image_path), cv2.IMREAD_GRAYSCALE
)

syntax_image = cv2.imread(
    str(syntax_image_path), cv2.IMREAD_GRAYSCALE
)

gt_mask = cv2.imread(
    str(gt_mask_path), cv2.IMREAD_GRAYSCALE
)

pred_mask = cv2.imread(
    str(pred_mask_path), cv2.IMREAD_GRAYSCALE
)

for name, image in [
    ("Stenosis image", stenosis_image),
    ("Syntax image", syntax_image),
    ("Ground-truth mask", gt_mask),
    ("Predicted mask", pred_mask)
]:
    if image is None:
        raise FileNotFoundError(f"{name} could not be loaded.")

print("\n========== IMAGE ALIGNMENT ==========")

print("Stenosis image shape:", stenosis_image.shape)
print("Syntax image shape:", syntax_image.shape)

if stenosis_image.shape == syntax_image.shape:

    difference = cv2.absdiff(
        stenosis_image, syntax_image
    )

    print(
        "Mean absolute image difference:",
        round(float(np.mean(difference)), 4)
    )

    print(
        "Identical image pixels:",
        np.array_equal(stenosis_image, syntax_image)
    )

else:
    print("Image dimensions differ.")

print("\n========== MASK INFORMATION ==========")

print("Ground-truth mask shape:", gt_mask.shape)
print("Predicted mask shape:", pred_mask.shape)

print(
    "Ground-truth foreground pixels:",
    int(np.count_nonzero(gt_mask))
)

print(
    "Predicted foreground pixels:",
    int(np.count_nonzero(pred_mask))
)

print(
    "Ground-truth unique values:",
    np.unique(gt_mask)
)

print(
    "Predicted unique values:",
    np.unique(pred_mask)
)
