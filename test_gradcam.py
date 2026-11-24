
from pathlib import Path

from backend.app.services.explainability import generate_gradcam


def main():
    project_root = Path(__file__).resolve().parent

    image_path = (
        project_root
        / "backend"
        / "app"
        / "data"
        / "arcade"
        / "arcade"
        / "arcade"
        / "stenosis"
        / "test"
        / "images"
        / "1.png"
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Test image not found: {image_path}"
        )

    result = generate_gradcam(image_path)

    print("\nGrad-CAM test completed!")
    print("Method:", result["method"])
    print("Target layer:", result["target_layer"])
    print("Heatmap:", result["heatmap_path"])
    print("Overlay:", result["overlay_path"])
    print("Heatmap shape:", result["heatmap"].shape)
    print("Overlay shape:", result["overlay"].shape)
    print("Heatmap minimum:", result["heatmap"].min())
    print("Heatmap maximum:", result["heatmap"].max())


if __name__ == "__main__":
    main()
