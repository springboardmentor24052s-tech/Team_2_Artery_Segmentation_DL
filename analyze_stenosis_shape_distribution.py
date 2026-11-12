
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

INPUT = Path("stenosis_annotation_geometry_report.csv")
OUTPUT = Path("stenosis_shape_analysis")

OUTPUT.mkdir(exist_ok=True)

df = pd.read_csv(INPUT)

# Annotation shape measurements
df["aspect_ratio"] = (
    df["bbox_width"] /
    df["bbox_height"].replace(0, np.nan)
)

df["area_to_bbox_ratio"] = (
    df["annotation_area"] /
    df["bbox_area"].replace(0, np.nan)
)

# Size groups based on annotation area.
# These are exploratory pixel-area groups, not clinical categories.
df["size_group"] = pd.cut(
    df["annotation_area"],
    bins=[-np.inf, 500, 1500, np.inf],
    labels=["Small", "Medium", "Large"]
)

print("\nSTENOSIS SHAPE AND SIZE ANALYSIS")
print("=" * 65)

print("\nSIZE DISTRIBUTION")
print(df["size_group"].value_counts().to_string())

print("\nSHAPE STATISTICS")
print(
    df[
        [
            "annotation_area",
            "bbox_width",
            "bbox_height",
            "aspect_ratio",
            "area_to_bbox_ratio"
        ]
    ].describe().round(3)
)

print("\nSPLIT-WISE SUMMARY")
print(
    df.groupby("split", observed=False)[
        [
            "annotation_area",
            "aspect_ratio",
            "area_to_bbox_ratio"
        ]
    ].agg(["mean", "median"]).round(3)
)

# Save detailed measurements
df.to_csv(
    OUTPUT / "stenosis_shape_details.csv",
    index=False
)

# Size distribution chart
df["size_group"].value_counts().reindex(
    ["Small", "Medium", "Large"]
).plot(kind="bar")

plt.title("Stenosis Annotation Size Distribution")
plt.xlabel("Annotation size group")
plt.ylabel("Number of annotations")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT / "size_distribution.png", dpi=200)
plt.close()

# Aspect ratio distribution
df["aspect_ratio"].dropna().plot(
    kind="hist", bins=30
)

plt.title("Stenosis Bounding Box Aspect Ratio")
plt.xlabel("Width / Height")
plt.ylabel("Number of annotations")
plt.tight_layout()
plt.savefig(OUTPUT / "aspect_ratio_distribution.png", dpi=200)
plt.close()

# Area-to-bounding-box ratio
df["area_to_bbox_ratio"].dropna().plot(
    kind="hist", bins=30
)

plt.title("Annotation Area to Bounding Box Area Ratio")
plt.xlabel("Annotation area / Bounding box area")
plt.ylabel("Number of annotations")
plt.tight_layout()
plt.savefig(OUTPUT / "area_bbox_ratio_distribution.png", dpi=200)
plt.close()

# Split-wise area comparison
df.boxplot(
    column="annotation_area",
    by="split"
)

plt.title("Annotation Area by Dataset Split")
plt.suptitle("")
plt.xlabel("Dataset split")
plt.ylabel("Annotation area (pixels squared)")
plt.tight_layout()
plt.savefig(OUTPUT / "split_area_comparison.png", dpi=200)
plt.close()

print("\nAnalysis completed.")
print(f"Results saved in: {OUTPUT.resolve()}")
