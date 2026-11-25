import os, uuid
from datetime import timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
from config import Config
from models.database import db, User, Case, Analysis
from services.segmentation import run_segmentation, model_status
from services.stenosis import estimate_narrowing

ALLOWED = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
app = Flask(__name__)
app.config.from_object(Config)
CORS(app, resources={r"/api/*": {"origins": os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")}}, supports_credentials=False)
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
db.init_app(app)
with app.app_context(): db.create_all()
serializer = URLSafeTimedSerializer(app.config["SECRET_KEY"], salt="arteryai-auth-v1")


def issue_token(user):
    return serializer.dumps({"user_id": user.id})


def current_user():
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    try:
        data = serializer.loads(header[7:], max_age=60 * 60 * 24 * 7)
        return db.session.get(User, int(data["user_id"]))
    except (BadSignature, SignatureExpired, KeyError, ValueError, TypeError):
        return None


def require_user():
    user = current_user()
    if user is None:
        return None, (jsonify({"error": "Please sign in again. Your session is missing or expired."}), 401)
    return user, None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


def case_json(case):
    return {
        "id": case.id, "patient_label": case.patient_label, "notes": case.notes or "",
        "image_url": f"/api/uploads/{case.image_filename}", "status": case.status,
        "created_at": case.created_at.isoformat() if case.created_at else None
    }


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "ArteryAI API", "version": "2.0.0"})


@app.get("/api/model/status")
def model_health():
    """Report whether a local trained checkpoint is available."""
    return jsonify(model_status())


@app.post("/api/auth/register")
def register():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not name or not email or len(password) < 8:
        return jsonify({"error": "Name, valid email and password (at least 8 characters) are required."}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "An account with this email already exists."}), 409
    user = User(name=name[:120], email=email[:190], password_hash=generate_password_hash(password))
    db.session.add(user); db.session.commit()
    return jsonify({"user": {"id": user.id, "name": user.name, "email": user.email}, "token": issue_token(user)}), 201


@app.post("/api/auth/login")
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, data.get("password") or ""):
        return jsonify({"error": "Invalid email or password."}), 401
    return jsonify({"user": {"id": user.id, "name": user.name, "email": user.email}, "token": issue_token(user)})


@app.get("/api/auth/me")
def me():
    user, error = require_user()
    if error: return error
    return jsonify({"user": {"id": user.id, "name": user.name, "email": user.email}})


@app.post("/api/cases")
def create_case():
    user, error = require_user()
    if error: return error
    label = (request.form.get("patient_label") or "").strip()
    image_file = request.files.get("image")
    notes = (request.form.get("notes") or "").strip()
    if not label or len(label) > 120 or not image_file or not image_file.filename or not allowed_file(image_file.filename):
        return jsonify({"error": "A case label (up to 120 characters) and a supported image are required."}), 400
    try:
        image_file.stream.seek(0)
        with Image.open(image_file.stream) as image:
            image.verify()
        image_file.stream.seek(0)
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({"error": "The uploaded file is not a valid supported image."}), 400
    filename = f"{uuid.uuid4().hex}_{secure_filename(image_file.filename)}"
    image_file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
    case = Case(user_id=user.id, patient_label=label, notes=notes[:4000], image_filename=filename, status="uploaded")
    db.session.add(case); db.session.commit()
    return jsonify(case_json(case)), 201


@app.get("/api/cases")
def list_cases():
    user, error = require_user()
    if error: return error
    cases = Case.query.filter_by(user_id=user.id).order_by(Case.created_at.desc(), Case.id.desc()).all()
    return jsonify([case_json(case) for case in cases])


@app.get("/api/cases/<int:case_id>")
def get_case(case_id):
    user, error = require_user()
    if error: return error
    case = Case.query.filter_by(id=case_id, user_id=user.id).first_or_404()
    latest = Analysis.query.filter_by(case_id=case.id).order_by(Analysis.id.desc()).first()
    result = case_json(case)
    result["analysis"] = None if latest is None else {
        "status": latest.status,
        "mask_url": f"/api/uploads/{latest.mask_filename}" if latest.mask_filename else None,
        "stenosis_percent": latest.stenosis_percent, "message": latest.message
    }
    return jsonify(result)


@app.delete("/api/cases/<int:case_id>")
def delete_case(case_id):
    user, error = require_user()
    if error: return error
    case = Case.query.filter_by(id=case_id, user_id=user.id).first_or_404()
    filenames = [case.image_filename] + [a.mask_filename for a in case.analyses if a.mask_filename]
    db.session.delete(case); db.session.commit()
    for filename in filenames:
        try: os.remove(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        except OSError: pass
    return jsonify({"message": "Case deleted."})


@app.post("/api/cases/<int:case_id>/analyze")
def analyze(case_id):
    user, error = require_user()
    if error: return error
    case = Case.query.filter_by(id=case_id, user_id=user.id).first_or_404()
    mask_name = f"mask_{uuid.uuid4().hex}.png"
    mask_path = os.path.join(app.config["UPLOAD_FOLDER"], mask_name)
    try:
        mask, message = run_segmentation(os.path.join(app.config["UPLOAD_FOLDER"], case.image_filename), mask_path)
        if mask is None:
            analysis = Analysis(case_id=case.id, status="model_not_ready", message=message)
            case.status = "model_not_ready"
        else:
            estimate, detail = estimate_narrowing(mask)
            analysis = Analysis(case_id=case.id, status="completed", mask_filename=mask_name,
                                stenosis_percent=estimate, message=f"{message} {detail}")
            case.status = "completed"
        db.session.add(analysis); db.session.commit()
        return jsonify({"status": analysis.status,
                        "mask_url": f"/api/uploads/{mask_name}" if analysis.mask_filename else None,
                        "stenosis_percent": analysis.stenosis_percent, "message": analysis.message})
    except Exception as exc:
        app.logger.exception("Analysis failed for case %s", case.id)
        return jsonify({"status": "error", "message": "Analysis failed. Check backend logs and model setup."}), 500


@app.get("/api/uploads/<path:filename>")
def uploaded_file(filename):
    user, error = require_user()
    if error: return error
    # Only expose files belonging to the authenticated user's cases or their analysis masks.
    cases = Case.query.filter_by(user_id=user.id).all()
    allowed_names = {c.image_filename for c in cases}
    for case in cases:
        allowed_names.update(a.mask_filename for a in case.analyses if a.mask_filename)
    if filename not in allowed_names:
        return jsonify({"error": "File not found."}), 404
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, conditional=True)


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Image is too large. Maximum upload size is 15 MB."}), 413


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
