
import json
from pathlib import Path

report_path = Path(
    "backend/results/attention_unet_v3_training.json"
)

with open(report_path, "r") as file:
    report = json.load(file)

history = report.get("history", [])

print("\n========== V3 TRAINING REPORT ==========")

print("Model:", report.get("model"))
print("Loss function:", report.get("loss_function"))
print("Total epochs:", report.get("epochs"))
print("Best validation Dice:", report.get("best_validation_dice"))

print("\n========== EPOCH-WISE HISTORY ==========")

print(
    "Epoch | Loss | Dice | IoU | Precision | Recall | Learning Rate"
)

for row in history:
    print(
        f"{row['epoch']:5} | "
        f"{row['training_loss']:.4f} | "
        f"{row['dice']:.4f} | "
        f"{row['iou']:.4f} | "
        f"{row['precision']:.4f} | "
        f"{row['recall']:.4f} | "
        f"{row['learning_rate']:.6f}"
    )

if history:
    best = max(history, key=lambda x: x["dice"])

    print("\n========== BEST EPOCH ==========")
    print("Epoch:", best["epoch"])
    print("Training loss:", best["training_loss"])
    print("Validation Dice:", best["dice"])
    print("Validation IoU:", best["iou"])
    print("Precision:", best["precision"])
    print("Recall:", best["recall"])

print("\nTraining report inspection completed.")
