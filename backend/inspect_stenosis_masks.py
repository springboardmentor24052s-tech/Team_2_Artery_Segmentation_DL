import cv2
import os
import numpy as np

folder = "results/api_predictions"

files = [
    "5f9b3c542b06444493e5c7ad2c3c3e89_stenosis_mask.png",
    "5f9b3c542b06444493e5c7ad2c3c3e89_stenosis_cleaned_mask.png",
    "5f9b3c542b06444493e5c7ad2c3c3e89_stenosis_probability.png"
]

for filename in files:
    path = os.path.join(folder, filename)
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    print("\nFile:", filename)

    if image is None:
        print("ERROR: Could not load file")
        continue

    print("Shape:", image.shape)
    print("Data type:", image.dtype)
    print("Unique values:", np.unique(image)[:30])
    print("Nonzero pixels:", np.count_nonzero(image))
