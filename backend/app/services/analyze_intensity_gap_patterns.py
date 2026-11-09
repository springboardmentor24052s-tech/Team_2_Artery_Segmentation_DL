
from pathlib import Path
import csv
import json

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

DATASET_ROOT = (
    BACKEND_ROOT
    / "results"
    / "stenosis_yolo_dataset"
    / "images"
)

SPLITS = ["train", "val", "test"]

OUTPUT_DIR = (
    BACKEND_ROOT
    / "results"
    / "intensity_gap_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp"
}


# ============================================================
# FIND IMAGES
# ============================================================

def find_images(split):

    image_dir = DATASET_ROOT / split

    if not image_dir.exists():
        raise FileNotFoundError(image_dir)

    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


# ============================================================
# ANALYZE ONE IMAGE
# ============================================================

def analyze_image(path, split):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        return None

    used_levels = np.flatnonzero(
        np.bincount(image.ravel(), minlength=256)
    )

    gaps = np.diff(used_levels)

    if len(gaps) == 0:
        return None

    total_gaps = len(gaps)

    gap_1 = int(np.sum(gaps == 1))
    gap_2 = int(np.sum(gaps == 2))
    gap_3 = int(np.sum(gaps == 3))
    gap_4_plus = int(np.sum(gaps >= 4))

    # Analyze gaps according to the lower intensity value.
    low_mask = used_levels[:-1] < 85
    mid_mask = (
        (used_levels[:-1] >= 85)
        & (used_levels[:-1] < 170)
    )
    high_mask = used_levels[:-1] >= 170

    def region_gap_stats(mask):

        region_gaps = gaps[mask]

        if len(region_gaps) == 0:
            return {
                "count": 0,
                "mean_gap": None,
                "gap_1_percentage": None,
                "gap_2_percentage": None
            }

        return {
            "count": int(len(region_gaps)),
            "mean_gap": round(
                float(np.mean(region_gaps)), 5
            ),
            "gap_1_percentage": round(
                float(np.mean(region_gaps == 1) * 100), 3
            ),
            "gap_2_percentage": round(
                float(np.mean(region_gaps == 2) * 100), 3
            )
        }

    return {
        "image": path.name,
        "split": split,
        "unique_levels": int(len(used_levels)),
        "total_gaps": int(total_gaps),
        "mean_gap": round(float(np.mean(gaps)), 5),
        "median_gap": round(float(np.median(gaps)), 5),
        "maximum_gap": int(np.max(gaps)),
        "gap_1_percentage": round(
            gap_1 / total_gaps * 100, 3
        ),
        "gap_2_percentage": round(
            gap_2 / total_gaps * 100, 3
        ),
        "gap_3_percentage": round(
            gap_3 / total_gaps * 100, 3
        ),
        "gap_4_plus_percentage": round(
            gap_4_plus / total_gaps * 100, 3
        ),
        "low_intensity_gaps": region_gap_stats(low_mask),
        "mid_intensity_gaps": region_gap_stats(mid_mask),
        "high_intensity_gaps": region_gap_stats(high_mask)
    }


# ============================================================
# PROCESS EACH SPLIT
# ============================================================

def process_dataset():

    all_records = []
    processing_status = {}

    for split in SPLITS:

        paths = find_images(split)

        print(f"\nProcessing {split.upper()}...")

        failed = 0

        for index, path in enumerate(paths, start=1):

            result = analyze_image(path, split)

            if result is None:
                failed += 1
                continue

            all_records.append(result)

            if index % 100 == 0:
                print(f"{split}: {index}/{len(paths)}")

        processing_status[split] = {
            "total_images": len(paths),
            "processed_images": len(paths) - failed,
            "failed_images": failed
        }

        print(
            f"{split.upper()} complete: "
            f"{len(paths) - failed} processed, "
            f"{failed} failed"
        )

    return all_records, processing_status


# ============================================================
# GROUP SUMMARY
# ============================================================

def summarize(records):

    summary = {}

    numeric_metrics = [
        "unique_levels",
        "mean_gap",
        "median_gap",
        "maximum_gap",
        "gap_1_percentage",
        "gap_2_percentage",
        "gap_3_percentage",
        "gap_4_plus_percentage"
    ]

    for split in SPLITS:

        group = [
            record for record in records
            if record["split"] == split
        ]

        split_summary = {
            "image_count": len(group)
        }

        for metric in numeric_metrics:

            values = np.array(
                [record[metric] for record in group],
                dtype=float
            )

            split_summary[metric] = {
                "mean": round(float(np.mean(values)), 5),
                "median": round(float(np.median(values)), 5),
                "min": round(float(np.min(values)), 5),
                "max": round(float(np.max(values)), 5)
            }

        for region in [
            "low_intensity_gaps",
            "mid_intensity_gaps",
            "high_intensity_gaps"
        ]:

            region_summary = {}

            for metric in [
                "mean_gap",
                "gap_1_percentage",
                "gap_2_percentage"
            ]:

                values = [
                    record[region][metric]
                    for record in group
                    if record[region][metric] is not None
                ]

                region_summary[metric] = (
                    round(float(np.mean(values)), 5)
                    if values else None
                )

            split_summary[region] = region_summary

        summary[split] = split_summary

    return summary


# ============================================================
# SAVE IMAGE-LEVEL CSV
# ============================================================

def save_csv(records):

    output_path = (
        OUTPUT_DIR / "intensity_gap_image_records.csv"
    )

    rows = []

    for record in records:

        row = {
            key: value
            for key, value in record.items()
            if key not in [
                "low_intensity_gaps",
                "mid_intensity_gaps",
                "high_intensity_gaps"
            ]
        }

        for region in [
            "low_intensity_gaps",
            "mid_intensity_gaps",
            "high_intensity_gaps"
        ]:

            for metric, value in record[region].items():

                row[f"{region}_{metric}"] = value

        rows.append(row)

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("INTENSITY GAP PATTERN ANALYSIS")
    print("=" * 65)

    records, status = process_dataset()

    if not records:
        raise RuntimeError(
            "No images were successfully analyzed."
        )

    summary = summarize(records)

    report = {
        "total_images_analyzed": len(records),
        "processing_status": status,
        "summary": summary
    }

    report_path = (
        OUTPUT_DIR / "intensity_gap_pattern_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    save_csv(records)

    print("\n" + "=" * 65)
    print("INTENSITY GAP ANALYSIS RESULTS")
    print("=" * 65)

    for split, values in summary.items():

        print(f"\n{split.upper()}")
        print("-" * 45)

        print(f"Images: {values['image_count']}")

        for metric in [
            "unique_levels",
            "mean_gap",
            "median_gap",
            "maximum_gap",
            "gap_1_percentage",
            "gap_2_percentage",
            "gap_3_percentage",
            "gap_4_plus_percentage"
        ]:

            stats = values[metric]

            print(
                f"{metric}: "
                f"mean={stats['mean']}, "
                f"median={stats['median']}"
            )

        print("\nIntensity region comparison:")

        for region in [
            "low_intensity_gaps",
            "mid_intensity_gaps",
            "high_intensity_gaps"
        ]:

            print(f"\n{region}")

            print(values[region])

    print("\nProcessing status:")

    for split, result in status.items():
        print(f"{split.upper()}: {result}")

    print("\nReports saved:")
    print(report_path)
    print(
        OUTPUT_DIR / "intensity_gap_image_records.csv"
    )

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
