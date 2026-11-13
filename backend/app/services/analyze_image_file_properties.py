
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, median

from PIL import Image


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

DATASET_ROOT = (
    BACKEND_ROOT
    / "app"
    / "data"
    / "arcade"
    / "arcade"
    / "arcade"
    / "stenosis"
)

OUTPUT_DIR = (
    BACKEND_ROOT
    / "results"
    / "image_file_properties_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp",
    ".tif", ".tiff", ".webp"
}


# ============================================================
# IMAGE DISCOVERY
# ============================================================

def get_image_paths(split):

    split_dir = DATASET_ROOT / split
    image_dir = split_dir / "images"

    if not image_dir.exists():
        image_dir = split_dir

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}"
        )

    return sorted([
        path for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ])


# ============================================================
# IMAGE FILE PROPERTY ANALYSIS
# ============================================================

def analyze_image_file(image_path, split):

    try:

        file_size_bytes = image_path.stat().st_size
        file_size_kb = file_size_bytes / 1024

        with Image.open(image_path) as image:

            image_format = image.format
            width, height = image.size
            mode = image.mode

            metadata = image.info or {}

            # Compression information may not exist
            compression = metadata.get(
                "compression", "not_available"
            )

            # Record metadata field names, not arbitrary metadata contents
            metadata_keys = sorted(
                str(key) for key in metadata.keys()
            )

            has_exif = bool(metadata.get("exif"))

            return {
                "image": image_path.name,
                "split": split,
                "extension": image_path.suffix.lower(),
                "actual_format": image_format,
                "file_size_bytes": file_size_bytes,
                "file_size_kb": round(file_size_kb, 3),
                "width": width,
                "height": height,
                "dimensions": f"{width}x{height}",
                "mode": mode,
                "compression": str(compression),
                "has_exif": has_exif,
                "metadata_keys": metadata_keys,
                "metadata_key_count": len(metadata_keys),
                "metadata_available": len(metadata_keys) > 0,
                "read_status": "success"
            }

    except Exception as error:

        return {
            "image": image_path.name,
            "split": split,
            "extension": image_path.suffix.lower(),
            "read_status": "failed",
            "error": str(error)
        }


# ============================================================
# ANALYZE SPLIT
# ============================================================

def analyze_split(split):

    image_paths = get_image_paths(split)

    print(f"\nAnalyzing {split.upper()} split")
    print(f"Images found: {len(image_paths)}")

    records = []

    for index, image_path in enumerate(image_paths, start=1):

        record = analyze_image_file(image_path, split)

        records.append(record)

        if index % 100 == 0:
            print(f"Processed {index}/{len(image_paths)}")

    return records


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def calculate_summary(records):

    successful = [
        record for record in records
        if record["read_status"] == "success"
    ]

    if not successful:
        return {
            "total_images": len(records),
            "successful_images": 0,
            "failed_images": len(records)
        }

    file_sizes = [
        record["file_size_kb"] for record in successful
    ]

    formats = Counter(
        record["actual_format"] for record in successful
    )

    extensions = Counter(
        record["extension"] for record in successful
    )

    modes = Counter(
        record["mode"] for record in successful
    )

    dimensions = Counter(
        record["dimensions"] for record in successful
    )

    compression = Counter(
        record["compression"] for record in successful
    )

    metadata_counts = Counter(
        record["metadata_available"]
        for record in successful
    )

    exif_counts = Counter(
        record["has_exif"] for record in successful
    )

    return {
        "total_images": len(records),
        "successful_images": len(successful),
        "failed_images": len(records) - len(successful),

        "actual_format_distribution": dict(formats),
        "extension_distribution": dict(extensions),
        "image_mode_distribution": dict(modes),
        "dimension_distribution": dict(dimensions),
        "compression_distribution": dict(compression),

        "file_size_kb": {
            "mean": round(mean(file_sizes), 3),
            "median": round(median(file_sizes), 3),
            "minimum": round(min(file_sizes), 3),
            "maximum": round(max(file_sizes), 3)
        },

        "metadata_available_images": metadata_counts[True],
        "images_with_exif": exif_counts[True]
    }


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(records):

    csv_path = OUTPUT_DIR / "image_file_properties.csv"

    fields = [
        "image",
        "split",
        "extension",
        "actual_format",
        "file_size_bytes",
        "file_size_kb",
        "width",
        "height",
        "dimensions",
        "mode",
        "compression",
        "has_exif",
        "metadata_key_count",
        "metadata_available",
        "read_status"
    ]

    with open(csv_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(records)

    print(f"\nCSV saved: {csv_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("IMAGE FILE FORMAT AND PROPERTY ANALYSIS")
    print("=" * 65)

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_ROOT}"
        )

    all_records = []
    summaries = {}

    for split in SPLITS:

        records = analyze_split(split)

        all_records.extend(records)

        summaries[split] = calculate_summary(records)

    # Save CSV
    save_csv(all_records)

    # Save JSON
    report = {
        "dataset": "ARCADE Stenosis",
        "total_images": len(all_records),
        "total_successful": sum(
            r["read_status"] == "success"
            for r in all_records
        ),
        "total_failed": sum(
            r["read_status"] == "failed"
            for r in all_records
        ),
        "split_summaries": summaries,
        "failed_files": [
            {
                "image": r["image"],
                "split": r["split"],
                "error": r.get("error", "Unknown error")
            }
            for r in all_records
            if r["read_status"] == "failed"
        ]
    }

    json_path = OUTPUT_DIR / "image_file_properties_summary.json"

    with open(json_path, "w") as file:
        json.dump(report, file, indent=4)

    # Display summary
    print("\nIMAGE FILE PROPERTY SUMMARY")
    print("=" * 65)

    for split, summary in summaries.items():

        print(f"\n{split.upper()}")

        print(
            "Successful images:",
            summary["successful_images"]
        )

        print(
            "Failed images:",
            summary["failed_images"]
        )

        print(
            "Actual formats:",
            summary.get("actual_format_distribution", {})
        )

        print(
            "Extensions:",
            summary.get("extension_distribution", {})
        )

        print(
            "Image modes:",
            summary.get("image_mode_distribution", {})
        )

        print(
            "Dimensions:",
            summary.get("dimension_distribution", {})
        )

        print(
            "File size (KB):",
            summary.get("file_size_kb", {})
        )

        print(
            "Compression:",
            summary.get("compression_distribution", {})
        )

        print(
            "Images with EXIF:",
            summary.get("images_with_exif", 0)
        )

    print("\nReports saved:")
    print(json_path)
    print(OUTPUT_DIR / "image_file_properties.csv")


if __name__ == "__main__":
    main()
