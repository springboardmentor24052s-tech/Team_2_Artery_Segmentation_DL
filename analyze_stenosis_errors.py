
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CSV_PATH = (
    ROOT / "backend/results/stenosis_validation_evaluation/"
    "per_image_metrics.csv"
)

OUTPUT_DIR = (
    ROOT / "backend/results/stenosis_error_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Sort images by Dice score
df_sorted = df.sort_values("dice", ascending=True)

# Worst 20 images
worst = df_sorted.head(20)

# Best 20 images
best = df_sorted.tail(20).sort_values(
    "dice", ascending=False
)

# Save results
worst.to_csv(
    OUTPUT_DIR / "worst_20_images.csv",
    index=False
)

best.to_csv(
    OUTPUT_DIR / "best_20_images.csv",
    index=False
)

# Print summary
print("\n================================")
print("STENOSIS ERROR ANALYSIS")
print("================================")

print("\nWorst 20 validation images:")
print(
    worst[
        ["image", "ground_truth_pixels",
         "predicted_pixels", "dice", "iou",
         "precision", "recall"]
    ].to_string(index=False)
)

print("\nBest 20 validation images:")
print(
    best[
        ["image", "ground_truth_pixels",
         "predicted_pixels", "dice", "iou",
         "precision", "recall"]
    ].to_string(index=False)
)

print("\nAverage predicted pixels:",
      round(df["predicted_pixels"].mean(), 2))

print("Average ground-truth pixels:",
      round(df["ground_truth_pixels"].mean(), 2))

print("\nResults saved to:", OUTPUT_DIR)
