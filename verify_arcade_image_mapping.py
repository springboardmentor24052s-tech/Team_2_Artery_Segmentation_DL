
import json
from pathlib import Path

ROOT = Path("backend/app/data/arcade/arcade/arcade")

SPLITS = ["train", "val", "test"]

print("\n========== ARCADE IMAGE MAPPING CHECK ==========\n")

for split in SPLITS:

    syntax_file = (
        ROOT / "syntax" / split /
        "annotations" / f"{split}.json"
    )

    stenosis_file = (
        ROOT / "stenosis" / split /
        "annotations" / f"{split}.json"
    )

    with open(syntax_file, "r", encoding="utf-8") as f:
        syntax_data = json.load(f)

    with open(stenosis_file, "r", encoding="utf-8") as f:
        stenosis_data = json.load(f)

    syntax_images = {
        image["id"]: image["file_name"]
        for image in syntax_data["images"]
    }

    stenosis_images = {
        image["id"]: image["file_name"]
        for image in stenosis_data["images"]
    }

    common_ids = (
        set(syntax_images) & set(stenosis_images)
    )

    matching_names = 0
    mismatching_names = 0

    print(f"\n--- {split.upper()} ---")

    for image_id in sorted(common_ids):

        syntax_name = syntax_images[image_id]
        stenosis_name = stenosis_images[image_id]

        if syntax_name == stenosis_name:
            matching_names += 1
        else:
            mismatching_names += 1

            if mismatching_names <= 5:
                print(
                    f"ID {image_id}: "
                    f"Syntax={syntax_name}, "
                    f"Stenosis={stenosis_name}"
                )

    print("Common numerical IDs:", len(common_ids))
    print("Matching image filenames:", matching_names)
    print("Mismatching image filenames:", mismatching_names)

    print("\nFirst 5 syntax mappings:")
    for image_id, filename in list(syntax_images.items())[:5]:
        print(image_id, "->", filename)

    print("\nFirst 5 stenosis mappings:")
    for image_id, filename in list(stenosis_images.items())[:5]:
        print(image_id, "->", filename)

print("\n========== CHECK COMPLETE ==========\n")
