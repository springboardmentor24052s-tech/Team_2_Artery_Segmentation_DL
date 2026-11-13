
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.explainability import generate_gradcam


router = APIRouter(
    prefix="/api/explainability",
    tags=["Explainability"]
)


# Backend results directory
BASE_DIR = Path(__file__).resolve().parents[2]

UPLOAD_DIR = BASE_DIR / "results" / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/health")
def explainability_health():
    return {
        "success": True,
        "message": "Explainability API is ready"
    }


@router.post("/gradcam")
async def generate_gradcam_api(
    file: UploadFile = File(...)
):
    """
    Generate Grad-CAM heatmap and overlay
    for an uploaded coronary angiography image.
    """

    allowed_extensions = {".png", ".jpg", ".jpeg"}

    extension = Path(file.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only PNG, JPG and JPEG images are supported."
        )

    unique_name = f"{uuid4().hex}{extension}"

    image_path = UPLOAD_DIR / unique_name

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        image_path.write_bytes(contents)

        result = generate_gradcam(
            image_path=image_path
        )

        heatmap_filename = Path(
            result["heatmap_path"]
        ).name

        overlay_filename = Path(
            result["overlay_path"]
        ).name

        return {
            "success": True,
            "message": "Grad-CAM generated successfully",
            "method": result["method"],
            "target_layer": result["target_layer"],
            "experimental": result["experimental"],
            "heatmap_url": (
                f"/results/explainability/{heatmap_filename}"
            ),
            "overlay_url": (
                f"/results/explainability/{overlay_filename}"
            )
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Grad-CAM generation failed: {str(e)}"
        )

    finally:
        if image_path.exists():
            image_path.unlink()

        await file.close()
