
import csv
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "classifier_v2_error_inspection"
)

FILES = {
    "VALIDATION": RESULT_DIR / "validation_predictions_v2.csv",
    "TEST": RESULT_DIR / "test_predictions_v2.csv"
}


def load_predictions(path, split):

    if not path.exists():
        print(f"\nFile not found: {path}")
        return None

    with open(path, "r", newline="") as file:
        rows = list(csv.DictReader(file))

    if split == "VALIDATION":
        label_col = "label"
        probability_col = "probability"
    else:
        label_col = "actual_label"
        probability_col = "predicted_probability"

    predictions = []

    for row in rows:
        predictions.append({
            "filename": row["filename"],
            "label": int(row[label_col]),
            "probability": float(row[probability_col]),
            "predicted_label": int(row["predicted_label"])
        })

    return predictions


def summarize(name, rows):

    if not rows:
        return

    labels = np.array([r["label"] for r in rows])
    probs = np.array([r["probability"] for r in rows])

    print(f"\n========== {name} ==========")

    for label, class_name in [(1, "POSITIVE"), (0, "NEGATIVE")]:

        values = probs[labels == label]

        if len(values) == 0:
            continue

        print(f"\n{class_name}")
        print("Samples:", len(values))
        print("Mean probability:", round(float(np.mean(values)), 4))
        print("Median probability:", round(float(np.median(values)), 4))
        print("Minimum probability:", round(float(np.min(values)), 4))
        print("Maximum probability:", round(float(np.max(values)), 4))
        print("Below 0.5:", int(np.sum(values < 0.5)))
        print("At or above 0.5:", int(np.sum(values >= 0.5)))


print("=" * 60)
print("CLASSIFIER V2 - SPLIT PROBABILITY COMPARISON")
print("=" * 60)

for split, path in FILES.items():

    rows = load_predictions(path, split)

    if rows is not None:
        summarize(split, rows)

print("\nComparison completed.")
