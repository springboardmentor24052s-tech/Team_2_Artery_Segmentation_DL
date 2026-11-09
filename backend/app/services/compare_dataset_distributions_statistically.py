
from pathlib import Path
import csv
import json

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


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
    / "statistical_distribution_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "brightness",
    "contrast_std",
    "dynamic_range",
    "unique_levels"
]

COMPARISONS = [
    ("train", "val"),
    ("train", "test"),
    ("val", "test")
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    data = {}

    for split in ["train", "val", "test"]:
        data[split] = {
            metric: []
            for metric in METRICS
        }

    with open(
        INPUT_CSV,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            split = row["split"]

            for metric in METRICS:
                data[split][metric].append(
                    float(row[metric])
                )

    return {
        split: {
            metric: np.asarray(values, dtype=float)
            for metric, values in metrics.items()
        }
        for split, metrics in data.items()
    }


# ============================================================
# STATISTICAL COMPARISON
# ============================================================

def compare_distributions(data):

    results = {}

    for first_split, second_split in COMPARISONS:

        comparison_name = (
            f"{first_split}_vs_{second_split}"
        )

        results[comparison_name] = {}

        for metric in METRICS:

            first = data[first_split][metric]
            second = data[second_split][metric]

            ks_result = ks_2samp(
                first,
                second,
                alternative="two-sided",
                method="auto"
            )

            distance = wasserstein_distance(
                first,
                second
            )

            mean_difference = (
                float(np.mean(second))
                - float(np.mean(first))
            )

            median_difference = (
                float(np.median(second))
                - float(np.median(first))
            )

            results[comparison_name][metric] = {
                "first_sample_size": len(first),
                "second_sample_size": len(second),
                "ks_statistic": float(
                    ks_result.statistic
                ),
                "ks_p_value": float(
                    ks_result.pvalue
                ),
                "wasserstein_distance": float(distance),
                "mean_difference_second_minus_first": (
                    mean_difference
                ),
                "median_difference_second_minus_first": (
                    median_difference
                )
            }

    return results


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(results):

    output_path = (
        OUTPUT_DIR / "statistical_distribution_results.csv"
    )

    rows = []

    for comparison, metrics in results.items():

        for metric, values in metrics.items():

            rows.append({
                "comparison": comparison,
                "metric": metric,
                **values
            })

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
    print("STATISTICAL DATASET DISTRIBUTION COMPARISON")
    print("=" * 65)

    data = load_data()

    for split in data:
        print(
            f"{split.upper()} images: "
            f"{len(data[split]['brightness'])}"
        )

    results = compare_distributions(data)

    report_path = (
        OUTPUT_DIR / "statistical_distribution_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    save_csv(results)

    print("\n" + "=" * 65)
    print("STATISTICAL COMPARISON RESULTS")
    print("=" * 65)

    for comparison, metrics in results.items():

        print(f"\n{comparison.upper()}")
        print("-" * 50)

        for metric, values in metrics.items():

            print(f"\n{metric}")

            print(
                f"KS statistic: "
                f"{values['ks_statistic']:.5f}"
            )

            print(
                f"KS p-value: "
                f"{values['ks_p_value']:.8f}"
            )

            print(
                f"Wasserstein distance: "
                f"{values['wasserstein_distance']:.5f}"
            )

            print(
                f"Mean difference: "
                f"{values['mean_difference_second_minus_first']:.5f}"
            )

            print(
                f"Median difference: "
                f"{values['median_difference_second_minus_first']:.5f}"
            )

    print("\nReports saved:")
    print(report_path)
    print(
        OUTPUT_DIR / "statistical_distribution_results.csv"
    )

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
