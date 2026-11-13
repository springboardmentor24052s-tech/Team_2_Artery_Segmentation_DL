
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.services.segmentation import (
    stenosis_model,
    preprocess_image
)


def generate_gradcam(image_path, output_dir=None):
    """
    Generate experimental Grad-CAM visualizations
    using the trained Attention U-Net V4 stenosis model.

    Target layer:
        dec2 (256 x 256 feature map).

    Target:
        Mean of the highest stenosis-probability pixels
        (top 1 percent).

    Note:
        This is an experimental segmentation-based
        explanation, not a clinical diagnosis.
    """

    model = stenosis_model
    target_layer = model.dec2

    image, tensor, height, width = preprocess_image(image_path)

    captured_features = {}
    captured_gradients = {}

    # --------------------------------------------------
    # 1. REGISTER HOOKS
    # --------------------------------------------------

    def forward_hook(module, inputs, output):
        captured_features["features"] = output

    def backward_hook(module, grad_input, grad_output):
        captured_gradients["gradients"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:

        # --------------------------------------------------
        # 2. MODEL SETUP
        # --------------------------------------------------

        model.eval()
        model.zero_grad(set_to_none=True)

        tensor = tensor.to(next(model.parameters()).device)
        tensor.requires_grad_(True)

        # --------------------------------------------------
        # 3. FORWARD PASS
        # --------------------------------------------------

        with torch.enable_grad():

            logits = model(tensor)

            probabilities = torch.sigmoid(logits)

            # Select highest probability pixels
            flat_probabilities = probabilities.flatten(1)

            k = max(
                1,
                int(flat_probabilities.shape[1] * 0.01)
            )

            top_values = torch.topk(
                flat_probabilities,
                k=k,
                dim=1
            ).values

            # Experimental region-based target
            target_score = top_values.mean()

            # Backward pass
            target_score.backward()

        # --------------------------------------------------
        # 4. CAPTURE FEATURES AND GRADIENTS
        # --------------------------------------------------

        if "features" not in captured_features:
            raise RuntimeError(
                "Grad-CAM features were not captured."
            )

        if "gradients" not in captured_gradients:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        features = captured_features["features"]
        gradients = captured_gradients["gradients"]

        if not torch.isfinite(features).all():
            raise RuntimeError("Non-finite feature values detected.")

        if not torch.isfinite(gradients).all():
            raise RuntimeError("Non-finite gradient values detected.")

        # --------------------------------------------------
        # 5. DIAGNOSTICS
        # --------------------------------------------------

        print("\n========== GRAD-CAM DIAGNOSTICS ==========")

        print("Input shape:", tensor.shape)
        print("Input min:", tensor.min().item())
        print("Input max:", tensor.max().item())

        print("Logits shape:", logits.shape)
        print("Logits min:", logits.min().item())
        print("Logits max:", logits.max().item())

        print("Probability min:", probabilities.min().item())
        print("Probability max:", probabilities.max().item())

        print("Features shape:", features.shape)
        print("Features min:", features.min().item())
        print("Features max:", features.max().item())

        print("Gradients shape:", gradients.shape)
        print("Gradient min:", gradients.min().item())
        print("Gradient max:", gradients.max().item())

        print(
            "Gradient absolute mean:",
            gradients.abs().mean().item()
        )

        print("Target score:", target_score.item())

        print("==========================================\n")

        # --------------------------------------------------
        # 6. CALCULATE CHANNEL IMPORTANCE
        # --------------------------------------------------

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # --------------------------------------------------
        # 7. GENERATE GRAD-CAM
        # --------------------------------------------------

        cam = (weights * features).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(height, width),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].detach().cpu().numpy()

        if not np.isfinite(cam).all():
            raise RuntimeError(
                "Grad-CAM contains invalid values."
            )

        # --------------------------------------------------
        # 8. NORMALIZE HEATMAP
        # --------------------------------------------------

        cam_min = float(cam.min())
        cam_max = float(cam.max())

        if cam_max - cam_min > 1e-8:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            print(
                "WARNING: Grad-CAM has negligible activation."
            )

            cam = np.zeros_like(
                cam,
                dtype=np.float32
            )

        heatmap = np.uint8(
            np.clip(cam * 255, 0, 255)
        )

        # --------------------------------------------------
        # 9. APPLY COLOUR MAP
        # --------------------------------------------------

        heatmap_color = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        original_color = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

        overlay = cv2.addWeighted(
            original_color,
            0.60,
            heatmap_color,
            0.40,
            0
        )

        # --------------------------------------------------
        # 10. OUTPUT DIRECTORY
        # --------------------------------------------------

        if output_dir is None:

            output_dir = (
                Path(__file__).resolve().parents[2]
                / "results"
                / "explainability"
            )

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        image_name = Path(image_path).stem

        heatmap_path = (
            output_dir /
            f"{image_name}_gradcam.png"
        )

        overlay_path = (
            output_dir /
            f"{image_name}_gradcam_overlay.png"
        )

        # --------------------------------------------------
        # 11. SAVE RESULTS
        # --------------------------------------------------

        if not cv2.imwrite(
            str(heatmap_path),
            heatmap
        ):
            raise IOError(
                "Could not save Grad-CAM heatmap."
            )

        if not cv2.imwrite(
            str(overlay_path),
            overlay
        ):
            raise IOError(
                "Could not save Grad-CAM overlay."
            )

        # --------------------------------------------------
        # 12. RETURN RESULTS
        # --------------------------------------------------

        return {

            "heatmap_path": str(heatmap_path),

            "overlay_path": str(overlay_path),

            "heatmap": heatmap,

            "overlay": overlay,

            "method": "Grad-CAM",

            "target_layer": "dec2",

            "target_method": "top_1_percent_probability",

            "experimental": True,

            "gradient_mean": float(
                gradients.abs().mean().item()
            ),

            "activation_min": cam_min,

            "activation_max": cam_max
        }

    finally:

        forward_handle.remove()

        backward_handle.remove()

        model.zero_grad(set_to_none=True)
