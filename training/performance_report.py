
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIRS = {
    "Baseline U-Net": OUTPUT_DIR / "unet_baseline_5epochs",
    "Attention U-Net": OUTPUT_DIR / "attention_unet_5epochs",
}

REPORT_DIR = OUTPUT_DIR / "performance_report"
PREDICTION_DIR = REPORT_DIR / "prediction_comparison"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD MODEL RESULTS
# --------------------------------------------------

def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_csv(path):
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# --------------------------------------------------
# 3. GENERATE TRAINING GRAPH
# --------------------------------------------------

def create_training_graph(histories):

    plt.figure(figsize=(9, 5))

    for name, history in histories.items():

        epochs = [int(row["epoch"]) for row in history]

        train_dice = [
            float(row["train_dice"]) for row in history
        ]

        val_dice = [
            float(row["val_dice"]) for row in history
        ]

        plt.plot(
            epochs,
            train_dice,
            marker="o",
            label=f"{name} Train",
        )

        plt.plot(
            epochs,
            val_dice,
            marker="s",
            linestyle="--",
            label=f"{name} Validation",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Training and Validation Dice Performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = REPORT_DIR / "training_analysis.png"

    plt.savefig(path, dpi=300)
    plt.close()

    return path


# --------------------------------------------------
# 4. GENERATE TEST METRICS GRAPH
# --------------------------------------------------

def create_test_graph(metrics):

    names = list(metrics.keys())

    metric_keys = ["loss", "dice", "iou"]

    titles = [
        "Test Loss",
        "Test Dice",
        "Test IoU",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for axis, key, title in zip(
        axes,
        metric_keys,
        titles,
    ):

        values = [
            metrics[name][key]
            for name in names
        ]

        axis.bar(names, values)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)

        for index, value in enumerate(values):

            axis.text(
                index,
                value,
                f"{value:.4f}",
                ha="center",
                va="bottom",
            )

    fig.tight_layout()

    path = REPORT_DIR / "test_metrics_comparison.png"

    fig.savefig(path, dpi=300)
    plt.close(fig)

    return path


# --------------------------------------------------
# 5. COLLECT ACTUAL PREDICTION EXAMPLES
# --------------------------------------------------

def collect_predictions():

    copied = []

    for name, folder in MODEL_DIRS.items():

        source = folder / "prediction_examples"

        if not source.is_dir():
            continue

        for index in range(1, 4):

            files = [
                source / f"image_{index:02d}.png",
                source / f"ground_truth_{index:02d}.png",
                source / f"prediction_{index:02d}.png",
            ]

            if not all(path.exists() for path in files):
                continue

            model_label = name.lower().replace(" ", "_")

            destination = PREDICTION_DIR / model_label

            destination.mkdir(parents=True, exist_ok=True)

            copied_files = []

            for path in files:

                target = destination / path.name

                shutil.copy2(path, target)

                copied_files.append(target)

            copied.append((name, index, copied_files))

    return copied


# --------------------------------------------------
# 6. BUILD PDF REPORT
# --------------------------------------------------

def build_report(
    metrics,
    histories,
    training_graph,
    test_graph,
    predictions,
):

    pdf_path = REPORT_DIR / "ArteryAI_Performance_Report.pdf"

    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # FIX: Define the missing Caption style.
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["Normal"],
            fontSize=9,
            alignment=TA_CENTER,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=21,
            leading=26,
            spaceAfter=15,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#174A70"),
            spaceBefore=12,
            spaceAfter=8,
        )
    )

    story = []

    # --------------------------------------------------
    # TITLE
    # --------------------------------------------------

    story.append(
        Paragraph(
            "ArteryAI - Segmentation Performance Report",
            styles["ReportTitle"],
        )
    )

    story.append(
        Paragraph(
            "Comparison of Baseline U-Net and Attention U-Net",
            styles["Heading2"],
        )
    )

    story.append(Spacer(1, 12))

    # --------------------------------------------------
    # PROJECT OBJECTIVE
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Project Objective",
            styles["SectionHeading"],
        )
    )

    story.append(
        Paragraph(
            "This report evaluates two deep-learning architectures "
            "for coronary anatomy segmentation using the ARCADE "
            "syntax dataset. The evaluated models are a baseline "
            "U-Net and an Attention U-Net.",
            styles["BodyText"],
        )
    )

    # --------------------------------------------------
    # EXPERIMENTAL CONFIGURATION
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Experimental Configuration",
            styles["SectionHeading"],
        )
    )

    configuration = [
        ["Setting", "Value"],
        ["Dataset", "ARCADE Syntax"],
        ["Training images", "1000"],
        ["Validation images", "200"],
        ["Test images", "300"],
        ["Image resolution", "128 x 128"],
        ["Batch size", "1"],
        ["Training epochs", "5"],
        ["Optimizer", "Adam"],
        ["Learning rate", "0.001"],
        ["Loss function", "Dice + Binary Cross Entropy"],
        ["Device", "CPU"],
    ]

    table = Table(
        configuration,
        colWidths=[2.3 * inch, 3.5 * inch],
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#174A70"),
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white,
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey,
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7,
            ),
        ])
    )

    story.append(table)

    # --------------------------------------------------
    # TEST METRICS COMPARISON
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Test Performance Comparison",
            styles["SectionHeading"],
        )
    )

    comparison = [
        ["Model", "Test Loss", "Dice", "IoU"]
    ]

    for name, values in metrics.items():

        comparison.append([
            name,
            f"{values['loss']:.4f}",
            f"{values['dice']:.4f}",
            f"{values['iou']:.4f}",
        ])

    table = Table(
        comparison,
        colWidths=[
            2.2 * inch,
            1.2 * inch,
            1.2 * inch,
            1.2 * inch,
        ],
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#174A70"),
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white,
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey,
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7,
            ),
        ])
    )

    story.append(table)
    story.append(Spacer(1, 12))

    story.append(
        Image(
            str(test_graph),
            width=7.0 * inch,
            height=2.5 * inch,
        )
    )

    # --------------------------------------------------
    # TRAINING ANALYSIS
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Training and Validation Analysis",
            styles["SectionHeading"],
        )
    )

    story.append(
        Paragraph(
            "The graph presents the Dice scores recorded during "
            "training and validation for both architectures. "
            "Differences between training and validation performance "
            "may indicate generalization challenges.",
            styles["BodyText"],
        )
    )

    story.append(
        Image(
            str(training_graph),
            width=7.0 * inch,
            height=3.8 * inch,
        )
    )

    # --------------------------------------------------
    # BEST VALIDATION CHECKPOINTS
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Best Validation Checkpoints",
            styles["SectionHeading"],
        )
    )

    checkpoint_rows = [
        ["Model", "Best Epoch", "Validation Dice"]
    ]

    for name, history in histories.items():

        best = max(
            history,
            key=lambda row: float(row["val_dice"]),
        )

        checkpoint_rows.append([
            name,
            str(best["epoch"]),
            f"{float(best['val_dice']):.4f}",
        ])

    table = Table(
        checkpoint_rows,
        colWidths=[
            2.5 * inch,
            1.5 * inch,
            1.5 * inch,
        ],
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#174A70"),
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white,
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey,
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7,
            ),
        ])
    )

    story.append(table)

    # --------------------------------------------------
    # PREDICTION EXAMPLES
    # --------------------------------------------------

    story.append(PageBreak())

    story.append(
        Paragraph(
            "Segmentation Prediction Examples",
            styles["SectionHeading"],
        )
    )

    story.append(
        Paragraph(
            "The following are actual saved outputs from the trained "
            "models. Each set contains the original image, ground-truth "
            "mask and predicted mask.",
            styles["BodyText"],
        )
    )

    for model_name, index, files in predictions:

        story.append(
            Paragraph(
                f"{model_name} - Example {index}",
                styles["Heading3"],
            )
        )

        labels = [
            "Original Image",
            "Ground Truth",
            "Prediction",
        ]

        cells = []

        for label, path in zip(labels, files):

            cells.append([
                Paragraph(label, styles["Caption"]),
                Image(
                    str(path),
                    width=1.65 * inch,
                    height=1.65 * inch,
                ),
            ])

        table = Table(
            [cells],
            colWidths=[1.9 * inch] * 3,
        )

        table.setStyle(
            TableStyle([
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ])
        )

        story.append(table)
        story.append(Spacer(1, 12))

    # --------------------------------------------------
    # OBSERVATIONS
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Observations",
            styles["SectionHeading"],
        )
    )

    baseline = metrics["Baseline U-Net"]
    attention = metrics["Attention U-Net"]

    dice_difference = (
        attention["dice"] - baseline["dice"]
    )

    iou_difference = (
        attention["iou"] - baseline["iou"]
    )

    loss_difference = (
        baseline["loss"] - attention["loss"]
    )

    story.append(
        Paragraph(
            f"Attention U-Net test Dice difference: "
            f"{dice_difference:+.4f}. "
            f"Test IoU difference: {iou_difference:+.4f}. "
            f"Test loss reduction: {loss_difference:+.4f}. "
            "Attention U-Net achieved slightly better results "
            "on all three test metrics in this experiment.",
            styles["BodyText"],
        )
    )

    # --------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Limitations",
            styles["SectionHeading"],
        )
    )

    limitations = [
        "Both models were trained for only five epochs.",
        "Training was performed on CPU at 128 x 128 resolution.",
        "Results are based on one training run and may vary with random seeds.",
        "Validation Dice scores fluctuated across epochs.",
        "The binary mask combines syntax polygons and excludes stenosis. "
        "This target definition should be checked against the dataset protocol.",
        "These experimental results are not clinical validation and "
        "must not be used for diagnosis or treatment decisions.",
    ]

    for item in limitations:

        story.append(
            Paragraph(
                f"- {item}",
                styles["BodyText"],
            )
        )

        story.append(Spacer(1, 4))

    # --------------------------------------------------
    # CONCLUSION
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Conclusion",
            styles["SectionHeading"],
        )
    )

    story.append(
        Paragraph(
            "Both U-Net architectures completed training and held-out "
            "test evaluation. Attention U-Net achieved slightly better "
            "test loss, Dice and IoU in this experiment. Further "
            "experiments with additional epochs, repeated runs and "
            "careful validation are recommended before drawing "
            "stronger conclusions.",
            styles["BodyText"],
        )
    )

    # Generate PDF
    document.build(story)

    return pdf_path


# --------------------------------------------------
# 7. MAIN FUNCTION
# --------------------------------------------------

def main():

    metrics = {}
    histories = {}

    print("Loading model results...")

    for name, folder in MODEL_DIRS.items():

        metrics_file = folder / "test_metrics.json"
        history_file = folder / "training_history.csv"

        if not metrics_file.exists():
            raise FileNotFoundError(
                f"Missing test metrics: {metrics_file}"
            )

        if not history_file.exists():
            raise FileNotFoundError(
                f"Missing training history: {history_file}"
            )

        metrics[name] = read_json(metrics_file)
        histories[name] = read_csv(history_file)

    print("Generating training graph...")

    training_graph = create_training_graph(histories)

    print("Generating test metrics graph...")

    test_graph = create_test_graph(metrics)

    print("Collecting prediction examples...")

    predictions = collect_predictions()

    if not predictions:
        raise RuntimeError(
            "No prediction examples found. Check both "
            "prediction_examples folders."
        )

    print(f"Collected {len(predictions)} prediction sets.")

    print("Building PDF report...")

    pdf_path = build_report(
        metrics,
        histories,
        training_graph,
        test_graph,
        predictions,
    )

    print("\n" + "=" * 55)
    print("PERFORMANCE REPORT GENERATED SUCCESSFULLY!")
    print("=" * 55)

    print(f"PDF: {pdf_path.resolve()}")
    print(f"Report folder: {REPORT_DIR.resolve()}")
    print(f"Prediction sets included: {len(predictions)}")


if __name__ == "__main__":
    main()
