
import json
from pathlib import Path

annotation_path = Path(
    "backend/app/data/arcade/arcade/arcade/"
    "stenosis/test/annotations/test.json"
)

with open(annotation_path, "r") as file:
    data = json.load(file)

# Find image information
images = [
    image for image in data["images"]
    if Path(image["file_name"]).name == "1.png"
]

if not images:
    print("Image 1.png not found in test annotations.")

else:
    for image in images:
        image_id = image["id"]

        print("\nIMAGE DETAILS")
        print("Image ID:", image_id)
        print("Filename:", image["file_name"])
        print("Width:", image["width"])
        print("Height:", image["height"])

        # Find annotations belonging to this image
        annotations = [
            ann for ann in data["annotations"]
            if ann["image_id"] == image_id
        ]

        print("\nGROUND-TRUTH ANNOTATIONS")
        print("Total annotations:", len(annotations))

        for ann in annotations:
            print("\nAnnotation ID:", ann["id"])
            print("Category ID:", ann["category_id"])
            print("Bounding box:", ann.get("bbox"))
            print("Area:", ann.get("area"))
            print(
                "Segmentation type:",
                type(ann.get("segmentation")).__name__
            )

        print("\nCATEGORIES")
        for category in data.get("categories", []):
            print(category)
