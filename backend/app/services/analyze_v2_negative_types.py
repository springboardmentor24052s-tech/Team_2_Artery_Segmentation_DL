
import csv
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CSV_PATH = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "classifier_v2_error_inspection"
    / "test_predictions_v2.csv"
)

with open(CSV_PATH, "r", newline="") as file:
    rows = list(csv.DictReader(file))

negative_rows = [
    row for row in rows
    if int(row["actual_label"]) == 0
]

print("=" * 65)
print("V2 TEST NEGATIVE PATCH ANALYSIS")
print("=" * 65)

for category in ["hard_vessel_candidate", "low_vessel_response"]:

    category_rows = [
        row for row in negative_rows
        if row["negative_type"] == category
    ]

    if not category_rows:
        continue

    probabilities = np.array([
        float(row["predicted_probability"])
        for row in category_rows
    ])

    vessel_scores = np.array([
        float(row["vessel_response_score"])
        for row in category_rows
    ])

    false_positives = int(np.sum(probabilities >= 0.5))
    true_negatives = int(np.sum(probabilities < 0.5))

    print(f"\nCATEGORY: {category}")
    print("-" * 45)

    print("Samples:", len(category_rows))
    print("Mean predicted probability:", round(float(np.mean(probabilities)), 4))
    print("Median predicted probability:", round(float(np.median(probabilities)), 4))
    print("Minimum probability:", round(float(np.min(probabilities)), 4))
    print("Maximum probability:", round(float(np.max(probabilities)), 4))

    print("False positives:", false_positives)
    print("Correct negatives:", true_negatives)

    print(
        "Average vessel response score:",
        round(float(np.mean(vessel_scores)), 2)
    )

print("\nAnalysis completed.")
