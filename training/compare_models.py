
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

MODELS = {
    "Baseline U-Net": OUTPUT_DIR / "unet_baseline_5epochs",
    "Attention U-Net": OUTPUT_DIR / "attention_unet_5epochs",
}


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_csv(path):
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    results = {}
    histories = {}

    for model_name, folder in MODELS.items():
        metrics_path = folder / "test_metrics.json"
        history_path = folder / "training_history.csv"

        if not metrics_path.exists() or not history_path.exists():
            raise FileNotFoundError(
                f"Missing results for {model_name}: {folder}"
            )

        results[model_name] = load_json(metrics_path)
        histories[model_name] = load_csv(history_path)

    # Create comparison table.
    comparison = []

    for model_name, metrics in results.items():
        comparison.append({
            "model": model_name,
            "test_loss": metrics["loss"],
            "test_dice": metrics["dice"],
            "test_iou": metrics["iou"],
        })

    report_dir = OUTPUT_DIR / "model_comparison"
    report_dir.mkdir(parents=True, exist_ok=True)

    with (report_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["model", "test_loss", "test_dice", "test_iou"],
        )
        writer.writeheader()
        writer.writerows(comparison)

    with (report_dir / "comparison.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(comparison, file, indent=2)

    # Compare test metrics visually.
    metric_names = ["test_loss", "test_dice", "test_iou"]
    labels = ["Test Loss", "Test Dice", "Test IoU"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for axis, metric, label in zip(axes, metric_names, labels):
        names = [row["model"] for row in comparison]
        values = [row[metric] for row in comparison]

        axis.bar(names, values)
        axis.set_title(label)
        axis.tick_params(axis="x", rotation=15)

        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.4f}", ha="center", va="bottom")

    fig.suptitle("Baseline U-Net vs Attention U-Net")
    fig.tight_layout()
    fig.savefig(report_dir / "test_metrics_comparison.png", dpi=300)
    plt.close(fig)

    # Compare training and validation Dice across epochs.
    plt.figure(figsize=(9, 5))

    for model_name, rows in histories.items():
        epochs = [int(row["epoch"]) for row in rows]
        train_dice = [float(row["train_dice"]) for row in rows]
        val_dice = [float(row["val_dice"]) for row in rows]

        plt.plot(
            epochs, train_dice,
            marker="o",
            label=f"{model_name} - Train",
        )
        plt.plot(
            epochs, val_dice,
            marker="s",
            linestyle="--",
            label=f"{model_name} - Validation",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Training and Validation Dice Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(report_dir / "dice_training_comparison.png", dpi=300)
    plt.close()

    print("\nMODEL COMPARISON")
    print("-" * 65)

    for row in comparison:
        print(
            f"{row['model']}: "
            f"Loss={row['test_loss']:.4f}, "
            f"Dice={row['test_dice']:.4f}, "
            f"IoU={row['test_iou']:.4f}"
        )

    print(f"\nComparison files saved in: {report_dir.resolve()}")


if __name__ == "__main__":
    main()
