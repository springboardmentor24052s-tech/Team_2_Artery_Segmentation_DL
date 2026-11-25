
"""Cached U-Net inference. A locally trained checkpoint is required."""
import os
import cv2
import torch
from models.unet import UNet
from services.preprocessing import preprocess_image

BASE = os.path.dirname(os.path.dirname(__file__))
WEIGHTS = os.getenv("ARTERYAI_MODEL_PATH", os.path.join(BASE, "model_weights", "unet.pth"))
_CACHE = {"mtime": None, "model": None, "device": None, "threshold": 0.5}

def model_status():
    return {
        "ready": os.path.isfile(WEIGHTS),
        "weights_path": WEIGHTS,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

def _load_model():
    if not os.path.isfile(WEIGHTS):
        raise FileNotFoundError(
            "Trained U-Net weights are missing. Train the model and save "
            "backend/model_weights/unet.pth first."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtime = os.path.getmtime(WEIGHTS)
    if (_CACHE["model"] is not None and _CACHE["mtime"] == mtime
            and _CACHE["device"] == str(device)):
        return _CACHE["model"], device, _CACHE["threshold"]
    try:
        checkpoint = torch.load(WEIGHTS, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(WEIGHTS, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        threshold = float(checkpoint.get("threshold", 0.5))
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state, threshold = checkpoint["state_dict"], 0.5
    else:
        state, threshold = checkpoint, 0.5
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    _CACHE.update(mtime=mtime, model=model, device=str(device), threshold=threshold)
    return model, device, threshold

def run_segmentation(image_path, output_path):
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise ValueError("Uploaded image could not be decoded for segmentation.")
    processed = preprocess_image(image_path, size=256)
    model, device, threshold = _load_model()
    tensor = torch.from_numpy(processed.astype("float32") / 255.0)[None, None].to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    small_mask = (probability >= threshold).astype("uint8") * 255
    mask = cv2.resize(
        small_mask, (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    if not cv2.imwrite(output_path, mask):
        raise OSError("Could not save segmentation mask.")
    return mask, (
        f"U-Net segmentation completed (threshold={threshold:.2f}, "
        f"device={device}). Research output only."
    )
