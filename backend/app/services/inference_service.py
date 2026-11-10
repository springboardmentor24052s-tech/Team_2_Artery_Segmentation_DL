import base64
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

from app.ml.attention_unet import AttentionUNet
from app.ml.unet import UNet

logger = logging.getLogger("inference_service")


class SegmentationModelType(str, Enum):
    UNET = "unet"
    ATTENTION_UNET = "attention_unet"


@dataclass
class SegmentationInferenceResult:
    """Encapsulates the structured output of a model inference run."""
    model_name: str
    original_height: int
    original_width: int
    threshold: float
    mask_binary: np.ndarray  # Shape: (H, W), dtype: uint8, values 0 or 1
    segmented_pixels: int
    vessel_area_ratio: float
    overlay_rgb: np.ndarray  # Shape: (H, W, 3), dtype: uint8 RGB
    overlay_base64: Optional[str] = None
    mask_base64: Optional[str] = None


class InferenceService:
    """
    Reusable, dataset-independent segmentation inference service.
    Loads PyTorch models once and runs preprocessing, inference, and postprocessing.
    """

    IMAGE_SIZE: int = 256
    DEFAULT_THRESHOLD: float = 0.3

    def __init__(
        self,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if checkpoint_dir is None:
            # Default to AIML_testing/outputs/models relative to the repository root
            repo_root = Path(__file__).resolve().parent.parent.parent.parent
            self.checkpoint_dir = repo_root / "AIML_testing" / "outputs" / "models"
        else:
            self.checkpoint_dir = Path(checkpoint_dir)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._models: Dict[str, nn.Module] = {}
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load_models(self) -> None:
        """
        Loads all model checkpoints into memory once.
        Supports unet and attention_unet checkpoints.
        """
        logger.info(f"Loading inference models using device: {self.device}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")

        configs = [
            (SegmentationModelType.UNET.value, UNet, "unet_best.pth", "unet_latest.pth"),
            (SegmentationModelType.ATTENTION_UNET.value, AttentionUNet, "attention_unet_best.pth", "attention_unet_latest.pth"),
        ]

        for model_key, model_class, best_name, fallback_name in configs:
            ckpt_path = self.checkpoint_dir / best_name
            if not ckpt_path.exists():
                fallback_path = self.checkpoint_dir / fallback_name
                if fallback_path.exists():
                    ckpt_path = fallback_path
                else:
                    logger.warning(
                        f"Checkpoint not found for {model_key} at {ckpt_path} or {fallback_path}."
                    )
                    continue

            try:
                model = self._load_checkpoint(model_class, ckpt_path)
                self._models[model_key] = model
                logger.info(f"Successfully loaded model '{model_key}' from {ckpt_path.name}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint for {model_key} from {ckpt_path}: {e}")
                raise

        self._is_loaded = len(self._models) > 0
        if not self._is_loaded:
            logger.warning("No segmentation models could be loaded.")

    def _load_checkpoint(self, model_class, checkpoint_path: Path) -> nn.Module:
        model = model_class().to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.eval()
        return model

    def get_model(self, model_type: Union[str, SegmentationModelType]) -> nn.Module:
        key = model_type.value if isinstance(model_type, SegmentationModelType) else str(model_type).lower()
        if key not in self._models:
            available = list(self._models.keys())
            raise ValueError(
                f"Model '{key}' is not available. Available models: {available}. "
                "Ensure load_models() has been called and checkpoints exist."
            )
        return self._models[key]

    # -------------------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------------------

    def preprocess_image(
        self, image_input: Union[str, Path, bytes, np.ndarray]
    ) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
        """
        Loads an angiography image from file path, raw bytes, or numpy array.
        Standardizes to grayscale, resizes to (256, 256), normalizes to [0, 1],
        and packages into a PyTorch tensor with shape (1, 1, 256, 256).

        Returns:
            tensor: torch.Tensor of shape (1, 1, 256, 256) on target device
            original_gray: 2D np.ndarray uint8 of original image
            (orig_h, orig_w): Tuple of original image dimensions
        """
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Unable to read image from path: {path_str}")
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Unable to decode image from provided byte buffer.")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3:
                # Convert color/multichannel image to single channel grayscale
                if image_input.shape[2] == 3:
                    img = cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)
                elif image_input.shape[2] == 4:
                    img = cv2.cvtColor(image_input, cv2.COLOR_RGBA2GRAY)
                else:
                    img = image_input[:, :, 0]
            else:
                img = image_input
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        orig_h, orig_w = img.shape[:2]

        resized = cv2.resize(
            img,
            (self.IMAGE_SIZE, self.IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        normalized = resized.astype(np.float32) / 255.0

        tensor = (
            torch.from_numpy(normalized)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        return tensor, img, (orig_h, orig_w)

    # -------------------------------------------------------------------------
    # Postprocessing
    # -------------------------------------------------------------------------

    @staticmethod
    def postprocess_mask(
        probabilities: np.ndarray,
        original_size: Tuple[int, int],
        threshold: float = 0.3,
    ) -> np.ndarray:
        """
        Applies classification threshold to model probability map and
        interpolates the binary mask back to the original image dimensions.
        """
        orig_h, orig_w = original_size
        binary_small = (probabilities >= threshold).astype(np.uint8)

        restored_mask = cv2.resize(
            binary_small,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        )
        return restored_mask

    @staticmethod
    def create_overlay(
        original_gray: np.ndarray,
        binary_mask: np.ndarray,
        color_rgb: Tuple[int, int, int] = (255, 60, 60),
        alpha: float = 0.55,
    ) -> np.ndarray:
        """
        Blends the segmented mask over the original grayscale angiography image.
        Returns uint8 RGB image of shape (H, W, 3).
        """
        base = np.stack([original_gray, original_gray, original_gray], axis=-1).astype(np.float32)
        result = base.copy()

        selected = binary_mask.astype(bool)
        color_array = np.asarray(color_rgb, dtype=np.float32)

        result[selected] = (1.0 - alpha) * base[selected] + alpha * color_array
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def encode_image_to_base64(image: np.ndarray, format_ext: str = ".png") -> str:
        """Encodes an RGB or grayscale numpy image into a base64 string."""
        if image.ndim == 3:
            # OpenCV expects BGR for encoding
            to_encode = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            to_encode = image
        success, buffer = cv2.imencode(format_ext, to_encode)
        if not success:
            raise RuntimeError("Failed to encode image to buffer.")
        return base64.b64encode(buffer).decode("utf-8")

    # -------------------------------------------------------------------------
    # End-to-end Inference
    # -------------------------------------------------------------------------

    def predict(
        self,
        image_input: Union[str, Path, bytes, np.ndarray],
        model_type: Union[str, SegmentationModelType] = SegmentationModelType.UNET,
        threshold: Optional[float] = None,
        generate_overlays: bool = True,
        encode_base64: bool = False,
    ) -> SegmentationInferenceResult:
        """
        Executes end-to-end segmentation inference on an arbitrary input image.
        """
        model_key = model_type.value if isinstance(model_type, SegmentationModelType) else str(model_type).lower()
        model = self.get_model(model_key)
        th = threshold if threshold is not None else self.DEFAULT_THRESHOLD

        # 1. Preprocess
        tensor, original_gray, (orig_h, orig_w) = self.preprocess_image(image_input)

        # 2. Forward pass
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        # 3. Postprocess
        binary_mask = self.postprocess_mask(probs, (orig_h, orig_w), threshold=th)

        # 4. Analytics
        segmented_pixels = int(np.sum(binary_mask))
        total_pixels = orig_h * orig_w
        vessel_area_ratio = float(segmented_pixels / total_pixels) if total_pixels > 0 else 0.0

        # 5. Overlays
        overlay_rgb = self.create_overlay(original_gray, binary_mask) if generate_overlays else np.empty((0, 0, 3), dtype=np.uint8)

        overlay_b64 = None
        mask_b64 = None
        if encode_base64:
            if generate_overlays:
                overlay_b64 = self.encode_image_to_base64(overlay_rgb)
            mask_b64 = self.encode_image_to_base64(binary_mask * 255)

        return SegmentationInferenceResult(
            model_name=model_key,
            original_height=orig_h,
            original_width=orig_w,
            threshold=th,
            mask_binary=binary_mask,
            segmented_pixels=segmented_pixels,
            vessel_area_ratio=vessel_area_ratio,
            overlay_rgb=overlay_rgb,
            overlay_base64=overlay_b64,
            mask_base64=mask_b64,
        )


# Global singleton instance for application-wide reuse
inference_service = InferenceService()
