from pathlib import Path
import uuid
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ai.unet import UNet

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
RESULTS = BASE / "results"
WEIGHTS = BASE / "weights"
for folder in (UPLOADS, RESULTS, WEIGHTS):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173", "http://127.0.0.1:4173"]}})
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLOWED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_model(task):
    """Load a trained model. No random-weight predictions are ever presented as results."""
    weights_path = WEIGHTS / f"{task}_unet.pth"
    if not weights_path.exists():
        return None
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def infer(model, image_path, output_path):
    original = Image.open(image_path).convert("L")
    original_size = original.size
    resized = original.resize((256, 256), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.copy()).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    mask = (probability >= 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask).resize(original_size, Image.Resampling.NEAREST)
    mask_img.save(output_path)
    # Overlay is a visual research aid, not a diagnosis.
    rgb = np.asarray(original.convert("RGB")).copy()
    mask_full = np.asarray(mask_img) > 0
    rgb[mask_full] = (0.35 * rgb[mask_full] + 0.65 * np.array([0, 255, 100])).astype(np.uint8)
    Image.fromarray(rgb).save(output_path.with_name(output_path.stem.replace("_mask", "") + "_overlay.png"))
    return float(probability.mean())


@app.get("/api/v1/health")
def health():
    return jsonify({"status": "ok", "device": str(DEVICE), "vessel_model_ready": (WEIGHTS / "vessel_unet.pth").exists(), "lesion_model_ready": (WEIGHTS / "lesion_unet.pth").exists()})


@app.post("/api/v1/ai/analyze")
def analyze():
    if "file" not in request.files:
        return jsonify({"message": "Upload a file using the 'file' form field."}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"message": "No file selected."}), 400
    ext = Path(secure_filename(file.filename)).suffix.lower()
    if ext not in ALLOWED:
        return jsonify({"message": "AI prototype currently accepts PNG, JPG, JPEG, TIFF, and BMP images. Convert DICOM to an image frame first."}), 415
    case_id = request.form.get("caseId", "research-case")
    safe_case = secure_filename(case_id) or "research-case"
    job_id = str(uuid.uuid4())
    try:
        image = Image.open(file.stream).convert("L")
        if image.width < 32 or image.height < 32:
            return jsonify({"message": "Image is too small for analysis."}), 400
        image_path = UPLOADS / f"{safe_case}_{job_id}.png"
        image.save(image_path)
    except (UnidentifiedImageError, OSError):
        return jsonify({"message": "The uploaded file is not a readable image."}), 400

    vessel_model = load_model("vessel")
    lesion_model = load_model("lesion")
    result = {
        "jobId": job_id,
        "caseId": case_id,
        "status": "completed" if vessel_model else "model_missing",
        "originalUrl": f"/api/v1/files/uploads/{image_path.name}",
        "vesselSegmentation": None,
        "stenosisDetection": {"status": "model_missing", "message": "A separately trained lesion model is required. No stenosis diagnosis or percentage is inferred."},
        "notice": "Research prototype only. Not for clinical diagnosis or treatment decisions."
    }
    if vessel_model:
        mask_path = RESULTS / f"{job_id}_vessel_mask.png"
        overlay_path = RESULTS / f"{job_id}_vessel_overlay.png"
        mean_probability = infer(vessel_model, image_path, mask_path)
        result["vesselSegmentation"] = {
            "status": "completed",
            "maskUrl": f"/api/v1/files/results/{mask_path.name}",
            "overlayUrl": f"/api/v1/files/results/{overlay_path.name}",
            "meanPixelProbability": round(mean_probability, 4)
        }
    else:
        result["message"] = "Image accepted, but no trained vessel model was found. Train a model and save backend/weights/vessel_unet.pth before expecting segmentation output."

    if lesion_model:
        lesion_mask_path = RESULTS / f"{job_id}_lesion_mask.png"
        infer(lesion_model, image_path, lesion_mask_path)
        result["stenosisDetection"] = {
            "status": "lesion_mask_generated",
            "maskUrl": f"/api/v1/files/results/{lesion_mask_path.name}",
            "message": "Lesion mask generated by the trained model. A validated reference diameter and calibrated geometry are still required for a stenosis percentage."
        }
    return jsonify(result), 200


@app.get("/api/v1/files/<folder>/<path:filename>")
def get_file(folder, filename):
    if folder not in {"uploads", "results"}:
        return jsonify({"message": "Not found"}), 404
    directory = UPLOADS if folder == "uploads" else RESULTS
    return send_from_directory(directory, filename, as_attachment=False)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
