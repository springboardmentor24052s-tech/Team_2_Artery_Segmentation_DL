
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

INPUT = Path("stenosis_annotation_geometry_report.csv")
OUTPUT = Path("annotation_split_consistency")

OUTPUT.mkdir(exist_ok=True)

df = pd.read_csv(INPUT)

# Derived measurements
df["aspect_ratio"] = (
    df["bbox_width"] /
    df["bbox_height"].replace(0, np.nan)
)

df["area_to_bbox_ratio"] = (
    df["annotation_area"] /
    df["bbox_area"].replace(0, np.nan)
)

print("\nANNOTATION SPLIT CONSISTENCY ANALYSIS")
print("=" * 75)

# 1. Split-wise descriptive statistics
metrics = [
    "annotation_area",
    "bbox_width",
    "bbox_height",
    "aspect_ratio",
    "area_to_bbox_ratio"
]

summary = df.groupby("split")[metrics].agg(
    ["count", "mean", "median", "std", "min", "max"]
)

print("\nSPLIT-WISE STATISTICS")
print(summary.round(3).to_string())

summary.to_csv(OUTPUT / "split_statistics.csv")

# 2. Identify area outliers using IQR within each split
outlier_rows = []

for split in ["train", "val", "test"]:
    subset = df[df["split"] == split]

    q1 = subset["annotation_area"].quantile(0.25)
    q3 = subset["annotation_area"].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = subset[
        (subset["annotation_area"] < lower) |
        (subset["annotation_area"] > upper)
    ].copy()

    print(f"\n{split.upper()} AREA OUTLIERS")
    print(f"IQR lower bound: {lower:.2f}")
    print(f"IQR upper bound: {upper:.2f}")
    print(f"Outlier count: {len(outliers)}")

    outlier_rows.append(outliers)

if outlier_rows:
    pd.concat(outlier_rows).to_csv(
        OUTPUT / "area_outliers.csv",
        index=False
    )

# 3. Compare annotation area distributions
plt.figure()
df.boxplot(column="annotation_area", by="split")
plt.title("Stenosis Annotation Area by Dataset Split")
plt.suptitle("")
plt.xlabel("Dataset split")
plt.ylabel("Annotation area (pixels squared)")
plt.tight_layout()
plt.savefig(OUTPUT / "area_boxplot.png", dpi=200)
plt.close()

# 4. Compare bounding-box dimensions
plt.figure()
df.boxplot(
    column=["bbox_width", "bbox_height"],
    by="split"
)
plt.suptitle("")
plt.title("Bounding Box Dimensions by Dataset Split")
plt.ylabel("Pixels")
plt.tight_layout()
plt.savefig(OUTPUT / "bbox_dimensions.png", dpi=200)
plt.close()

# 5. Compare aspect ratio distributions
plt.figure()
for split in ["train", "val", "test"]:
    df.loc[df["split"] == split, "aspect_ratio"].dropna().plot(
        kind="hist",
        bins=30,
        alpha=0.4,
        label=split
    )

plt.title("Aspect Ratio Distribution by Split")
plt.xlabel("Bounding box width / height")
plt.ylabel("Annotation count")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT / "aspect_ratio_comparison.png", dpi=200)
plt.close()

print("\nAnalysis completed.")
print(f"Results saved in: {OUTPUT.resolve()}")
