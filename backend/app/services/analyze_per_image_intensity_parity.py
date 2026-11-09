
import json
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = (
    PROJECT_ROOT / "backend" / "app" / "data"
    / "arcade" / "arcade" / "arcade" / "stenosis"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "backend" / "results"
    / "per_image_intensity_parity"
)

SPLITS = ["train", "val", "test"]


def analyze_image(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L"))

    used_levels = np.unique(pixels)

    even_count = int(np.sum(used_levels % 2 == 0))
    odd_count = int(np.sum(used_levels % 2 == 1))
    total = len(used_levels)

    even_percentage = even_count / total * 100 if total else 0
    odd_percentage = odd_count / total * 100 if total else 0

    return {
        "filename": path.name,
        "unique_levels": int(total),
        "even_levels": even_count,
        "odd_levels": odd_count,
        "even_percentage": round(even_percentage, 3),
        "odd_percentage": round(odd_percentage, 3),
        "dominant_parity": (
            "even" if even_count > odd_count
            else "odd" if odd_count > even_count
            else "balanced"
        )
    }


def analyze_split(split):
    folder = DATASET_ROOT / split

    paths = [
        p for p in folder.rglob("*.png")
        if p.is_file() and "annotations" not in p.parts
    ]

    records = []
    failed = []

    print(f"\nAnalyzing {split.upper()} ({len(paths)} images)")

    for index, path in enumerate(paths, start=1):
        try:
            records.append(analyze_image(path))
        except Exception as error:
            failed.append({
                "filename": path.name,
                "error": str(error)
            })

        if index % 100 == 0:
            print(f"Processed {index}/{len(paths)}")

    if not records:
        return {"records": [], "summary": {}, "failed": failed}

    even_values = [r["even_percentage"] for r in records]
    odd_values = [r["odd_percentage"] for r in records]

    dominant_counts = {
        key: sum(r["dominant_parity"] == key for r in records)
        for key in ["even", "odd", "balanced"]
    }

    summary = {
        "images_processed": len(records),
        "failed_images": len(failed),
        "mean_even_level_percentage": round(float(np.mean(even_values)), 3),
        "mean_odd_level_percentage": round(float(np.mean(odd_values)), 3),
        "mean_unique_levels_per_image": round(
            float(np.mean([r["unique_levels"] for r in records])), 3
        ),
        "dominant_parity_counts": dominant_counts,
        "images_even_dominant": dominant_counts["even"],
        "images_odd_dominant": dominant_counts["odd"],
        "images_balanced": dominant_counts["balanced"]
    }

    return {
        "records": records,
        "summary": summary,
        "failed": failed
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for split in SPLITS:
        results[split] = analyze_split(split)

        print("\n" + "=" * 55)
        print(f"{split.upper()} SUMMARY")
        print("=" * 55)

        for key, value in results[split]["summary"].items():
            print(f"{key}: {value}")

    report_path = OUTPUT_DIR / "per_image_parity_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nAnalysis completed.")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
