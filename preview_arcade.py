
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path("backend/app/data/arcade/arcade/arcade")

# Start with the syntax training split.
annotation_file = ROOT / "syntax/train/annotations/train.json"

with annotation_file.open("r", encoding="utf-8") as f:
    data = json.load(f)

print("Categories:")
for category in data["categories"]:
    print(category)

print("\nFirst 5 image paths:")
for image in data["images"][:5]:
    print(image["file_name"])

print("\nFirst 5 annotations:")
for annotation in data["annotations"][:5]:
    print({
        "image_id": annotation["image_id"],
        "category_id": annotation["category_id"],
        "bbox": annotation.get("bbox"),
        "area": annotation.get("area"),
    })

# Display one original image to verify the image folder.
image_info = data["images"][0]
image_name = image_info["file_name"]

candidates = [
    ROOT / "syntax/train" / image_name,
    ROOT / "syntax/train/images" / image_name,
    ROOT / "syntax/train/image" / image_name,
]

image_path = next((p for p in candidates if p.exists()), None)

if image_path is None:
    print("\nImage not found in the expected locations.")
    print("Search for the file named:", image_name)
else:
    image = cv2.imread(str(image_path))
    if image is None:
        print("OpenCV could not read:", image_path)
    else:
        print("\nImage found:", image_path)
        print("Image dimensions:", image.shape)
        cv2.imshow("ARCADE Original Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
