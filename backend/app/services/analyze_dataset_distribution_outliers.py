
from pathlib import Path
import csv
import json

import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

INPUT_CSV = (
    BACKEND_ROOT
    / "results"
    / "yolo_preprocessing_comparison"
    / "preprocessing_image_records.csv"
)

OUTPUT_DIR = (
    BACKEND_ROOT
    / "results"
    / "dataset_distribution_outliers"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]

METRICS = [
    "brightness",
    "contrast_std",
    "dynamic_range",
    "unique_levels"
]


# ============================================================
# LOAD CSV
# ============================================================

def load_records():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    records = []

    with open(
        INPUT_CSV,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            record = {
                "image": row["image"],
                "split": row["split"]
            }

            for metric in METRICS:
                record[metric] = float(row[metric])

            records.append(record)

    return records


# ============================================================
# PERCENTILE ANALYSIS
# ============================================================

def calculate_percentiles(values):

    percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 100]

    results = {}

    for percentile in percentiles:

        results[f"p{percentile}"] = round(
            float(np.percentile(values, percentile)),
            5
        )

    return results


# ============================================================
# IQR OUTLIER ANALYSIS
# ============================================================

def analyze_outliers(records, metric):

    values = np.array(
        [record[metric] for record in records],
        dtype=float
    )

    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outliers = []

    for record in records:

        value = record[metric]

        if value < lower_bound or value > upper_bound:

            outliers.append({
                "image": record["image"],
                "split": record["split"],
                "value": round(value, 5),
                "outlier_type": (
                    "low" if value < lower_bound else "high"
                )
            })

    return {
        "q1": round(q1, 5),
        "q3": round(q3, 5),
        "iqr": round(iqr, 5),
        "lower_bound": round(lower_bound, 5),
        "upper_bound": round(upper_bound, 5),
        "outlier_count": len(outliers),
        "outlier_percentage": round(
            len(outliers) / len(values) * 100, 2
        ),
        "outliers": outliers
    }


# ============================================================
# PER-SPLIT STATISTICS
# ============================================================

def analyze_split(records, split):

    split_records = [
        record
        for record in records
        if record["split"] == split
    ]

    if not split_records:
        raise ValueError(f"No records found for {split}")

    results = {
        "image_count": len(split_records),
        "metrics": {}
    }

    for metric in METRICS:

        values = np.array(
            [record[metric] for record in split_records],
            dtype=float
        )

        results["metrics"][metric] = {
            "mean": round(float(np.mean(values)), 5),
            "median": round(float(np.median(values)), 5),
            "std": round(float(np.std(values)), 5),
            "percentiles": calculate_percentiles(values)
        }

    return results


# ============================================================
# COMBINED OUTLIER ANALYSIS
# ============================================================

def analyze_all_outliers(records):

    results = {}

    for metric in METRICS:

        results[metric] = analyze_outliers(
            records,
            metric
        )

    return results


# ============================================================
# SAVE OUTLIER CSV
# ============================================================

def save_outliers_csv(outlier_results):

    output_path = (
        OUTPUT_DIR / "dataset_outlier_images.csv"
    )

    rows = []

    for metric, result in outlier_results.items():

        for outlier in result["outliers"]:

            rows.append({
                "metric": metric,
                "image": outlier["image"],
                "split": outlier["split"],
                "value": outlier["value"],
                "outlier_type": outlier["outlier_type"]
            })

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "metric",
            "image",
            "split",
            "value",
            "outlier_type"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nOutlier CSV saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("DATASET DISTRIBUTION AND OUTLIER ANALYSIS")
    print("=" * 65)

    records = load_records()

    print(f"\nTotal records loaded: {len(records)}")

    split_results = {}

    for split in SPLITS:

        split_results[split] = analyze_split(
            records,
            split
        )

    outlier_results = analyze_all_outliers(records)

    report = {
        "dataset_total_images": len(records),
        "split_statistics": split_results,
        "combined_iqr_outlier_analysis": outlier_results
    }

    report_path = (
        OUTPUT_DIR / "dataset_distribution_outlier_report.json"
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

    save_outliers_csv(outlier_results)

    # --------------------------------------------------------
    # PRINT PERCENTILE RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("PERCENTILE COMPARISON")
    print("=" * 65)

    for split in SPLITS:

        print(f"\n{split.upper()}")
        print("-" * 45)

        for metric in METRICS:

            values = (
                split_results[split]["metrics"][metric]
            )

            print(f"\n{metric}")

            for percentile, value in (
                values["percentiles"].items()
            ):

                print(f"  {percentile}: {value}")

    # --------------------------------------------------------
    # PRINT OUTLIER RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("COMBINED IQR OUTLIER ANALYSIS")
    print("=" * 65)

    for metric, result in outlier_results.items():

        print(f"\n{metric}")
        print("-" * 45)

        print(f"Q1: {result['q1']}")
        print(f"Q3: {result['q3']}")
        print(f"IQR: {result['iqr']}")
        print(f"Lower bound: {result['lower_bound']}")
        print(f"Upper bound: {result['upper_bound']}")
        print(f"Outliers: {result['outlier_count']}")
        print(
            f"Outlier percentage: "
            f"{result['outlier_percentage']}%"
        )

        for split in SPLITS:

            count = sum(
                1
                for item in result["outliers"]
                if item["split"] == split
            )

            print(f"  {split}: {count}")

    print("\nReports saved:")
    print(report_path)
    print(OUTPUT_DIR / "dataset_outlier_images.csv")

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
