
import json
from pathlib import Path
from collections import Counter, defaultdict

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "data"
    / "arcade"
    / "arcade"
    / "arcade"
    / "stenosis"
)

OUTPUT_DIR = PROJECT_ROOT / "backend" / "results" / "original_png_metadata"

SPLITS = ["train", "val", "test"]


def get_png_color_type(image):
    """
    PNG color types:
    0 = Grayscale
    2 = Truecolor RGB
    3 = Indexed color
    4 = Grayscale with alpha
    6 = Truecolor RGBA
    """
    try:
        with open(image, "rb") as file:
            header = file.read(26)

        if header[:8] != b"\x89PNG\r\n\x1a\n":
            return "Not PNG"

        bit_depth = header[24]
        color_type = header[25]

        names = {
            0: "Grayscale",
            2: "RGB",
            3: "Indexed",
            4: "Grayscale + Alpha",
            6: "RGBA"
        }

        return {
            "bit_depth": bit_depth,
            "color_type": color_type,
            "color_type_name": names.get(color_type, "Unknown")
        }

    except Exception as error:
        return {"error": str(error)}


def inspect_split(split):
    split_dir = DATASET_ROOT / split
    records = []

    image_files = [
        path for path in split_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".png"
        and "annotations" not in path.parts
    ]

    print(f"\nInspecting {split.upper()} ({len(image_files)} images)")

    for index, path in enumerate(image_files, start=1):
        try:
            with Image.open(path) as image:
                png_info = get_png_color_type(path)

                record = {
                    "filename": path.name,
                    "mode": image.mode,
                    "format": image.format,
                    "width": image.width,
                    "height": image.height,
                    "file_size_bytes": path.stat().st_size,
                    "bit_depth": png_info.get("bit_depth"),
                    "color_type": png_info.get("color_type"),
                    "color_type_name": png_info.get("color_type_name"),
                    "metadata_keys": sorted(image.info.keys()),
                    "has_gamma": "gamma" in image.info,
                    "has_icc_profile": "icc_profile" in image.info,
                    "has_srgb": "srgb" in image.info,
                    "has_transparency": "transparency" in image.info,
                    "has_dpi": "dpi" in image.info,
                    "has_exif": "exif" in image.info
                }

                records.append(record)

        except Exception as error:
            print(f"Could not inspect {path.name}: {error}")

        if index % 100 == 0:
            print(f"Processed {index}/{len(image_files)}")

    return records


def summarize_split(split, records):
    print("\n" + "=" * 65)
    print(f"{split.upper()} PNG METADATA SUMMARY")
    print("=" * 65)

    if not records:
        print("No images found.")
        return {}

    summary = {
        "total_images": len(records),
        "modes": dict(Counter(r["mode"] for r in records)),
        "formats": dict(Counter(r["format"] for r in records)),
        "dimensions": dict(
            Counter(f'{r["width"]}x{r["height"]}' for r in records)
        ),
        "bit_depths": dict(Counter(str(r["bit_depth"]) for r in records)),
        "color_types": dict(
            Counter(r["color_type_name"] for r in records)
        ),
        "metadata_key_combinations": dict(
            Counter(", ".join(r["metadata_keys"]) or "None" for r in records)
        ),
        "gamma_count": sum(r["has_gamma"] for r in records),
        "icc_profile_count": sum(r["has_icc_profile"] for r in records),
        "srgb_count": sum(r["has_srgb"] for r in records),
        "transparency_count": sum(r["has_transparency"] for r in records),
        "dpi_count": sum(r["has_dpi"] for r in records),
        "exif_count": sum(r["has_exif"] for r in records),
        "file_size_mean_kb": round(
            sum(r["file_size_bytes"] for r in records)
            / len(records) / 1024, 3
        ),
        "file_size_min_kb": round(
            min(r["file_size_bytes"] for r in records) / 1024, 3
        ),
        "file_size_max_kb": round(
            max(r["file_size_bytes"] for r in records) / 1024, 3
        )
    }

    for key, value in summary.items():
        print(f"{key}: {value}")

    return summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_records = {}
    all_summaries = {}

    for split in SPLITS:
        records = inspect_split(split)
        all_records[split] = records
        all_summaries[split] = summarize_split(split, records)

    report = {
        "dataset": "ARCADE stenosis original images",
        "summaries": all_summaries,
        "records": all_records
    }

    report_path = OUTPUT_DIR / "original_png_metadata_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 65)
    print("PNG METADATA INSPECTION COMPLETED")
    print("=" * 65)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
