
from pathlib import Path
from uuid import UUID

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.services.postprocessing import postprocess_mask
from app.services.centerline import extract_centerline


router = APIRouter(
    prefix="/api/analysis",
    tags=["Image Processing"]
)

BACKEND_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = BACKEND_DIR / "results" / "api_predictions"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def validate_image_id(image_id: str):
    try:
        return str(UUID(hex=image_id)).replace("-", "")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image ID."
        )


def get_mask_path(image_id: str, model_type: str):
    if model_type == "syntax":
        return RESULTS_DIR / f"{image_id}_mask.png"

    if model_type == "stenosis":
        return RESULTS_DIR / f"{image_id}_stenosis_mask.png"

    raise HTTPException(
        status_code=400,
        detail="Invalid model type. Choose syntax or stenosis."
    )


def load_mask(mask_path: Path):
    if not mask_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Segmentation mask not found. Run segmentation first."
        )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise HTTPException(
            status_code=500,
            detail="Unable to read segmentation mask."
        )

    return mask


# --------------------------------------------------
# 1. POST-PROCESSING
# --------------------------------------------------

@router.post("/postprocess/{image_id}")
def run_postprocessing(
    image_id: str,
    model_type: str = "syntax"
):
    image_id = validate_image_id(image_id)

    original_mask_path = get_mask_path(
        image_id,
        model_type
    )

    mask = load_mask(original_mask_path)

    try:
        result = postprocess_mask(
            mask,
            min_component_area=20,
            kernel_size=3
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    suffix = "_cleaned_mask.png" if model_type == "syntax" else "_stenosis_cleaned_mask.png"

    cleaned_mask_path = RESULTS_DIR / f"{image_id}{suffix}"

    saved = cv2.imwrite(
        str(cleaned_mask_path),
        result["mask"]
    )

    if not saved:
        raise HTTPException(
            status_code=500,
            detail="Unable to save cleaned mask."
        )

    return {
        "status": "success",
        "message": "Mask post-processing completed.",
        "image_id": image_id,
        "model_type": model_type,
        "cleaned_mask_url": f"/api/analysis/cleaned-mask/{image_id}?model_type={model_type}",
        "statistics": {
            "original_foreground_pixels": result["original_foreground_pixels"],
            "cleaned_foreground_pixels": result["cleaned_foreground_pixels"],
            "removed_pixels": result["removed_pixels"],
            "component_count": result["component_count"],
            "components": result["components"]
        }
    }


# --------------------------------------------------
# 2. RETRIEVE CLEANED MASK
# --------------------------------------------------

@router.get("/cleaned-mask/{image_id}")
def get_cleaned_mask(
    image_id: str,
    model_type: str = "syntax"
):
    image_id = validate_image_id(image_id)

    suffix = "_cleaned_mask.png" if model_type == "syntax" else "_stenosis_cleaned_mask.png"

    if model_type not in ("syntax", "stenosis"):
        raise HTTPException(
            status_code=400,
            detail="Invalid model type."
        )

    mask_path = RESULTS_DIR / f"{image_id}{suffix}"

    if not mask_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Cleaned mask not found. Run post-processing first."
        )

    return FileResponse(
        str(mask_path),
        media_type="image/png"
    )


# --------------------------------------------------
# 3. CENTERLINE EXTRACTION
# --------------------------------------------------

@router.post("/centerline/{image_id}")
def run_centerline_extraction(
    image_id: str,
    model_type: str = "syntax"
):
    image_id = validate_image_id(image_id)

    if model_type not in ("syntax", "stenosis"):
        raise HTTPException(
            status_code=400,
            detail="Invalid model type."
        )

    suffix = "_cleaned_mask.png" if model_type == "syntax" else "_stenosis_cleaned_mask.png"

    cleaned_mask_path = RESULTS_DIR / f"{image_id}{suffix}"

    mask = load_mask(cleaned_mask_path)

    try:
        result = extract_centerline(mask)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    centerline_suffix = "_centerline.png" if model_type == "syntax" else "_stenosis_centerline.png"

    centerline_path = RESULTS_DIR / f"{image_id}{centerline_suffix}"

    saved = cv2.imwrite(
        str(centerline_path),
        result["centerline_mask"]
    )

    if not saved:
        raise HTTPException(
            status_code=500,
            detail="Unable to save centerline image."
        )

    return {
        "status": "success",
        "message": "Centerline extraction and pixel-based diameter estimation completed.",
        "image_id": image_id,
        "model_type": model_type,
        "centerline_url": f"/api/analysis/centerline-image/{image_id}?model_type={model_type}",
        "statistics": {
            "centerline_pixel_count": result["centerline_pixel_count"],
            "endpoint_count": result["endpoint_count"],
            "junction_count": result["junction_count"],
            "diameter_statistics": result["diameter_statistics"],
            "diameter_values_pixels": result["diameter_values_pixels"]
        }
    }


# --------------------------------------------------
# 4. RETRIEVE CENTERLINE IMAGE
# --------------------------------------------------

@router.get("/centerline-image/{image_id}")
def get_centerline_image(
    image_id: str,
    model_type: str = "syntax"
):
    image_id = validate_image_id(image_id)

    if model_type not in ("syntax", "stenosis"):
        raise HTTPException(
            status_code=400,
            detail="Invalid model type."
        )

    suffix = "_centerline.png" if model_type == "syntax" else "_stenosis_centerline.png"

    centerline_path = RESULTS_DIR / f"{image_id}{suffix}"

    if not centerline_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Centerline image not found. Run centerline extraction first."
        )

    return FileResponse(
        str(centerline_path),
        media_type="image/png"
    )
