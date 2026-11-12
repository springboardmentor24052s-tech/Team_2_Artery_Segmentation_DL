
import json
from pathlib import Path

DATASET_ROOT = Path("backend/app/data/arcade/arcade/arcade")

print("\n========== STENOSIS ANNOTATION INSPECTION ==========\n")

json_files = list(DATASET_ROOT.rglob("*.json"))

if not json_files:
    print("No JSON annotation files found.")
    raise SystemExit

for file_path in json_files:
    if "stenosis" not in str(file_path).lower():
        continue

    print(f"\nFile: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        print("Top-level keys:", list(data.keys()))

        categories = data.get("categories", [])
        print("\nCategories:")
        for category in categories:
            print(category)

        annotations = data.get("annotations", [])

        print("\nTotal annotations:", len(annotations))

        if annotations:
            print("\nFirst annotation:")
            print(json.dumps(annotations[0], indent=4))

            all_keys = set()
            for annotation in annotations:
                all_keys.update(annotation.keys())

            print("\nAll annotation fields:")
            print(sorted(all_keys))

            severity_fields = [
                key for key in all_keys
                if any(word in key.lower() for word in
                       ["severity", "grade", "percent", "narrow", "stenosis"])
            ]

            print("\nPossible severity-related fields:")
            print(severity_fields if severity_fields else "None found")

    except Exception as error:
        print("Error reading file:", error)

print("\n========== INSPECTION COMPLETE ==========\n")
