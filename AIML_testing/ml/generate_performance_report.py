
from pathlib import Path
import csv
import statistics
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

UNET_CSV = (
    ROOT / "outputs" / "evaluation_unet" / "test_metrics.csv"
)

ATTENTION_CSV = (
    ROOT / "outputs" / "evaluation_attention" / "test_metrics.csv"
)

REPORT_DIR = ROOT / "outputs" / "performance_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Both models must have been evaluated with this threshold.
THRESHOLD = 0.3


# --------------------------------------------------
# 2. LOAD PER-IMAGE METRICS
# --------------------------------------------------

def read_metrics(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing evaluation file: {csv_path}\n"
            "Run model evaluation first."
        )

    results = {}

    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        required = {"image", "dice", "iou"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{csv_path.name} must contain image, dice and iou columns."
            )

        for row in reader:
            name = row["image"]

            if name in results:
                raise ValueError(f"Duplicate image: {name}")

            results[name] = {
                "dice": float(row["dice"]),
                "iou": float(row["iou"])
            }

    if not results:
        raise ValueError(f"No metrics found in {csv_path}")

    return results


def summarize(results):
    dice = [v["dice"] for v in results.values()]
    iou = [v["iou"] for v in results.values()]

    return {
        "images": len(results),
        "mean_dice": statistics.mean(dice),
        "mean_iou": statistics.mean(iou),
        "median_dice": statistics.median(dice),
        "median_iou": statistics.median(iou),
        "std_dice": statistics.pstdev(dice),
        "std_iou": statistics.pstdev(iou),
        "min_dice": min(dice),
        "max_dice": max(dice)
    }


# --------------------------------------------------
# 3. COMPARE BOTH MODELS
# --------------------------------------------------

def main():
    print("=" * 65)
    print("SEGMENTATION PERFORMANCE COMPARISON")
    print("=" * 65)
    print("Threshold to be reported:", THRESHOLD)
    print("Note: ensure both evaluation CSV files actually use this threshold.")

    unet = read_metrics(UNET_CSV)
    attention = read_metrics(ATTENTION_CSV)

    # Confirm both models were evaluated on the same test images.
    if set(unet) != set(attention):
        raise ValueError(
            "The models were not evaluated on identical test images. "
            "Re-run evaluation using the same test set."
        )

    if len(unet) != 300:
        print(f"WARNING: Expected 300 images; found {len(unet)}.")

    unet_stats = summarize(unet)
    attention_stats = summarize(attention)

    print(f"\nCommon test images: {len(unet)}")

    # Model with the higher mean scores
    if attention_stats["mean_dice"] > unet_stats["mean_dice"]:
        better_dice = "Attention U-Net"
    elif attention_stats["mean_dice"] < unet_stats["mean_dice"]:
        better_dice = "U-Net"
    else:
        better_dice = "Tie"

    if attention_stats["mean_iou"] > unet_stats["mean_iou"]:
        better_iou = "Attention U-Net"
    elif attention_stats["mean_iou"] < unet_stats["mean_iou"]:
        better_iou = "U-Net"
    else:
        better_iou = "Tie"

    dice_diff = (
        attention_stats["mean_dice"] - unet_stats["mean_dice"]
    ) * 100

    iou_diff = (
        attention_stats["mean_iou"] - unet_stats["mean_iou"]
    ) * 100

    # --------------------------------------------------
    # 4. MODEL SUMMARY CSV
    # --------------------------------------------------

    summary_path = REPORT_DIR / "model_summary.csv"

    fields = [
        "model", "images",
        "mean_dice", "mean_iou",
        "median_dice", "median_iou",
        "std_dice", "std_iou",
        "min_dice", "max_dice"
    ]

    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        writer.writerow({"model": "U-Net", **unet_stats})
        writer.writerow({"model": "Attention U-Net", **attention_stats})

    # --------------------------------------------------
    # 5. PER-IMAGE COMPARISON CSV
    # --------------------------------------------------

    comparison_path = REPORT_DIR / "per_image_comparison.csv"

    per_image_fields = [
        "image",
        "unet_dice", "attention_dice", "dice_difference",
        "unet_iou", "attention_iou", "iou_difference"
    ]

    with open(
        comparison_path, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=per_image_fields)
        writer.writeheader()

        for name in sorted(unet):
            writer.writerow({
                "image": name,
                "unet_dice": unet[name]["dice"],
                "attention_dice": attention[name]["dice"],
                "dice_difference": (
                    attention[name]["dice"] - unet[name]["dice"]
                ),
                "unet_iou": unet[name]["iou"],
                "attention_iou": attention[name]["iou"],
                "iou_difference": (
                    attention[name]["iou"] - unet[name]["iou"]
                )
            })

    # --------------------------------------------------
    # 6. COMPARISON GRAPH
    # --------------------------------------------------

    labels = ["Mean Dice Score", "Mean IoU Score"]

    unet_values = [
        unet_stats["mean_dice"] * 100,
        unet_stats["mean_iou"] * 100
    ]

    attention_values = [
        attention_stats["mean_dice"] * 100,
        attention_stats["mean_iou"] * 100
    ]

    x = [0, 1]
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    bars1 = ax.bar(
        [v - width / 2 for v in x],
        unet_values,
        width,
        label="U-Net"
    )

    bars2 = ax.bar(
        [v + width / 2 for v in x],
        attention_values,
        width,
        label="Attention U-Net"
    )

    ax.set_title("U-Net vs Attention U-Net")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend()

    ax.bar_label(bars1, fmt="%.2f%%", padding=3)
    ax.bar_label(bars2, fmt="%.2f%%", padding=3)

    fig.tight_layout()

    graph_path = REPORT_DIR / "performance_comparison.png"
    fig.savefig(graph_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------
    # 7. WRITTEN REPORT
    # --------------------------------------------------

    report_path = REPORT_DIR / "performance_report.md"

    report = f"""# Segmentation Performance Report

## Evaluation Setup

- Dataset: ARCADE processed stenosis test set
- Common test images: {len(unet)}
- Evaluation threshold recorded: {THRESHOLD}
- Models: U-Net and Attention U-Net
- Metrics: Dice Score and Intersection over Union (IoU)

**Verification:** Ensure both source evaluation runs used threshold {THRESHOLD}.
The threshold is not stored in the current per-image CSV files.

## Model Comparison

| Metric | U-Net | Attention U-Net | Difference (Attention - U-Net) |
|---|---:|---:|---:|
| Mean Dice | {unet_stats['mean_dice']*100:.2f}% | {attention_stats['mean_dice']*100:.2f}% | {dice_diff:+.2f} percentage points |
| Mean IoU | {unet_stats['mean_iou']*100:.2f}% | {attention_stats['mean_iou']*100:.2f}% | {iou_diff:+.2f} percentage points |

## Additional Statistics

| Statistic | U-Net Dice | Attention U-Net Dice | U-Net IoU | Attention U-Net IoU |
|---|---:|---:|---:|---:|
| Median | {unet_stats['median_dice']:.4f} | {attention_stats['median_dice']:.4f} | {unet_stats['median_iou']:.4f} | {attention_stats['median_iou']:.4f} |
| Standard deviation | {unet_stats['std_dice']:.4f} | {attention_stats['std_dice']:.4f} | {unet_stats['std_iou']:.4f} | {attention_stats['std_iou']:.4f} |
| Minimum Dice | {unet_stats['min_dice']:.4f} | {attention_stats['min_dice']:.4f} | - | - |
| Maximum Dice | {unet_stats['max_dice']:.4f} | {attention_stats['max_dice']:.4f} | - | - |

## Results

- Higher mean Dice Score: **{better_dice}**
- Higher mean IoU Score: **{better_iou}**

A higher Dice or IoU indicates better overlap with the reference masks
on this test set. Review visual predictions as well. These results alone
do not establish clinical suitability.

## Generated Files

- `model_summary.csv`: summary statistics.
- `per_image_comparison.csv`: metrics for each common test image.
- `performance_comparison.png`: comparison chart.
"""

    report_path.write_text(report, encoding="utf-8")

    # --------------------------------------------------
    # 8. PRINT RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL MODEL COMPARISON")
    print("=" * 65)

    for name, stats in [
        ("U-Net", unet_stats),
        ("Attention U-Net", attention_stats)
    ]:
        print(f"\n{name}")
        print(
            f"Mean Dice: {stats['mean_dice']:.4f} "
            f"({stats['mean_dice']*100:.2f}%)"
        )
        print(
            f"Mean IoU : {stats['mean_iou']:.4f} "
            f"({stats['mean_iou']*100:.2f}%)"
        )

    print(f"\nDice difference (Attention - U-Net): {dice_diff:+.2f} pp")
    print(f"IoU difference (Attention - U-Net): {iou_diff:+.2f} pp")

    print("\nReports saved to:", REPORT_DIR)
    print("Summary CSV:", summary_path)
    print("Per-image CSV:", comparison_path)
    print("Comparison graph:", graph_path)
    print("Written report:", report_path)


if __name__ == "__main__":
    main()
  