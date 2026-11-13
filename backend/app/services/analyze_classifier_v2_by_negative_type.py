
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "classifier_v2_error_inspection"
    / "test_predictions_v2.csv"
)

df = pd.read_csv(PREDICTIONS_PATH)

print("\n========== V2 TEST PERFORMANCE BY PATCH TYPE ==========\n")

# Analyze positive and negative patches separately
for label, label_name in [(0, "NEGATIVE"), (1, "POSITIVE")]:

    subset = df[df["actual_label"] == label]

    print(f"\n--- {label_name} PATCHES ---")
    print("Total:", len(subset))

    if label == 0:
        for patch_type, group in subset.groupby("negative_type"):

            print(f"\nNegative type: {patch_type}")
            print("Samples:", len(group))
            print(
                "Mean probability:",
                round(group["predicted_probability"].mean(), 4)
            )
            print(
                "Median probability:",
                round(group["predicted_probability"].median(), 4)
            )
            print(
                "Correctly predicted negative:",
                int((group["predicted_label"] == 0).sum())
            )
            print(
                "Incorrectly predicted positive:",
                int((group["predicted_label"] == 1).sum())
            )

    else:
        print(
            "Mean probability:",
            round(subset["predicted_probability"].mean(), 4)
        )

print("\n========== SPLIT DISTRIBUTION ==========\n")

print(
    df.groupby(["actual_label", "negative_type"], dropna=False)
    .size()
    .to_string()
)

print("\nAnalysis completed.")
