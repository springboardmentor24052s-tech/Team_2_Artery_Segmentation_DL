
from pathlib import Path
import uuid

import cv2
import numpy as np

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from app.services.segmentation import (
    predict_segmentation,
    predict_stenosis
)

from app.services.candidate_extraction import (
    extract_stenosis_candidates
)

from app.services.localization import (
    localize_stenosis_candidates
)


router = APIRouter(
    prefix="/api/analysis",
    tags=["Segmentation Analysis"]
)


# --------------------------------------------------
# PROJECT DIRECTORIES
# --------------------------------------------------

# backend/app/routes/analysis.py
# parents[0] = routes
# parents[1] = app
# parents[2] = backend

PROJECT_ROOT = Path(__file__).resolve().parents[2]

UPLOAD_DIR = PROJECT_ROOT / "uploads"
RESULT_DIR = PROJECT_ROOT / "results" / "api_predictions"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# --------------------------------------------------
# COMMON IMAGE VALIDATION
# --------------------------------------------------

async def save_uploaded_image(file: UploadFile):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PNG, JPG and JPEG images are supported."
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    image_array = cv2.imdecode(
        np.frombuffer(contents, dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    if image_array is None:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image."
        )

    image_id = uuid.uuid4().hex

    input_path = UPLOAD_DIR / f"{image_id}{extension}"

    input_path.write_bytes(contents)

    return image_id, input_path


# --------------------------------------------------
# IMAGE ID VALIDATION
# --------------------------------------------------

def validate_image_id(image_id: str):

    if (
        len(image_id) != 32
        or not all(c in "0123456789abcdef" for c in image_id)
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid image ID."
        )


# --------------------------------------------------
# COMMON HELPER: FIND ORIGINAL IMAGE
# --------------------------------------------------

def find_original_image(image_id: str):

    for extension in ALLOWED_EXTENSIONS:

        candidate = UPLOAD_DIR / f"{image_id}{extension}"

        if candidate.is_file():
            return candidate

    return None


# --------------------------------------------------
# 1. SYNTAX SEGMENTATION - U-NET
# --------------------------------------------------

@router.post("/segment")
async def segment_image(file: UploadFile = File(...)):

    try:

        image_id, input_path = await save_uploaded_image(file)

        mask_path = RESULT_DIR / f"{image_id}_mask.png"

        prediction = predict_segmentation(input_path)

        mask_saved = cv2.imwrite(
            str(mask_path),
            prediction["mask"]
        )

        if not mask_saved or not mask_path.is_file():
            raise RuntimeError("Unable to save syntax mask.")

        return {
            "status": "success",
            "model": "Syntax U-Net",
            "message": "Syntax segmentation completed successfully.",
            "image_id": image_id,
            "image_dimensions": {
                "width": prediction["width"],
                "height": prediction["height"]
            },
            "foreground_pixels": int(
                np.count_nonzero(prediction["mask"])
            ),
            "mask_url": f"/api/analysis/mask/{image_id}",
            "original_image_url": f"/api/analysis/image/{image_id}",
            "mask_saved": True,
            "mask_file": str(mask_path.resolve())
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Syntax segmentation failed: {str(error)}"
        )

    finally:
        await file.close()


# --------------------------------------------------
# 2. STENOSIS SEGMENTATION - ATTENTION U-NET V3
# --------------------------------------------------

@router.post("/segment/stenosis")
async def segment_stenosis(file: UploadFile = File(...)):

    try:

        image_id, input_path = await save_uploaded_image(file)

        mask_path = RESULT_DIR / f"{image_id}_stenosis_mask.png"

        probability_path = (
            RESULT_DIR / f"{image_id}_stenosis_probability.png"
        )

        prediction = predict_stenosis(input_path)

        predicted_mask = prediction["mask"]
        probability_map = prediction["probability_map"]

        # Save stenosis mask
        mask_saved = cv2.imwrite(
            str(mask_path),
            predicted_mask
        )

        if not mask_saved or not mask_path.is_file():
            raise RuntimeError("Unable to save stenosis mask.")

        # Save probability map
        probability_image = (
            probability_map * 255
        ).clip(0, 255).astype(np.uint8)

        probability_saved = cv2.imwrite(
            str(probability_path),
            probability_image
        )

        if not probability_saved or not probability_path.is_file():
            raise RuntimeError("Unable to save probability map.")

        return {
            "status": "success",
            "model": "Attention U-Net V3",
            "message": "Stenosis segmentation completed successfully.",
            "image_id": image_id,
            "image_dimensions": {
                "width": prediction["width"],
                "height": prediction["height"]
            },
            "foreground_pixels": int(
                np.count_nonzero(predicted_mask)
            ),
            "mask_url": f"/api/analysis/stenosis-mask/{image_id}",
            "probability_map_url": (
                f"/api/analysis/stenosis-probability/{image_id}"
            ),
            "original_image_url": f"/api/analysis/image/{image_id}",
            "mask_saved": True,
            "mask_file": str(mask_path.resolve()),
            "probability_file": str(probability_path.resolve())
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Stenosis segmentation failed: {str(error)}"
        )

    finally:
        await file.close()


# --------------------------------------------------
# 3. RETRIEVE SYNTAX MASK
# --------------------------------------------------

@router.get("/mask/{image_id}")
def get_segmentation_mask(image_id: str):

    validate_image_id(image_id)

    mask_path = RESULT_DIR / f"{image_id}_mask.png"

    if not mask_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Syntax segmentation mask not found."
        )

    return FileResponse(
        path=mask_path,
        media_type="image/png",
        filename=mask_path.name
    )


# --------------------------------------------------
# 4. RETRIEVE STENOSIS MASK
# --------------------------------------------------

@router.get("/stenosis-mask/{image_id}")
def get_stenosis_mask(image_id: str):

    validate_image_id(image_id)

    mask_path = RESULT_DIR / f"{image_id}_stenosis_mask.png"

    if not mask_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Stenosis mask not found."
        )

    return FileResponse(
        path=mask_path,
        media_type="image/png",
        filename=mask_path.name
    )


# --------------------------------------------------
# 5. RETRIEVE STENOSIS PROBABILITY MAP
# --------------------------------------------------

@router.get("/stenosis-probability/{image_id}")
def get_stenosis_probability(image_id: str):

    validate_image_id(image_id)

    probability_path = (
        RESULT_DIR / f"{image_id}_stenosis_probability.png"
    )

    if not probability_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Stenosis probability map not found."
        )

    return FileResponse(
        path=probability_path,
        media_type="image/png",
        filename=probability_path.name
    )


# --------------------------------------------------
# 6. RETRIEVE ORIGINAL ANGIOGRAPHY IMAGE
# --------------------------------------------------

@router.get("/image/{image_id}")
def get_original_image(image_id: str):

    validate_image_id(image_id)

    original_path = find_original_image(image_id)

    if original_path is None:
        raise HTTPException(
            status_code=404,
            detail="Original angiography image not found."
        )

    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg"
    }

    return FileResponse(
        path=original_path,
        media_type=media_types[original_path.suffix.lower()],
        filename=original_path.name
    )


# --------------------------------------------------
# 7. STENOSIS CANDIDATE EXTRACTION
# --------------------------------------------------

@router.post("/detect-candidates/{image_id}")
def detect_stenosis_candidates(image_id: str):

    validate_image_id(image_id)

    mask_path = RESULT_DIR / f"{image_id}_stenosis_mask.png"

    probability_path = (
        RESULT_DIR / f"{image_id}_stenosis_probability.png"
    )

    if not mask_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Stenosis mask not found. Run stenosis segmentation first."
        )

    if not probability_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Stenosis probability map not found."
        )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    probability_image = cv2.imread(
        str(probability_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None or probability_image is None:
        raise HTTPException(
            status_code=500,
            detail="Unable to read segmentation output files."
        )

    if mask.shape != probability_image.shape:
        raise HTTPException(
            status_code=500,
            detail="Mask and probability map dimensions do not match."
        )

    probability_map = (
        probability_image.astype(np.float32) / 255.0
    )

    try:

        results = extract_stenosis_candidates(
            probability_map=probability_map,
            binary_mask=mask,
            min_area=10
        )

        return {
            "status": "success",
            "image_id": image_id,
            "message": "Stenosis candidate extraction completed.",
            "results": results
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Candidate extraction failed: {str(error)}"
        )


# --------------------------------------------------
# 8. STENOSIS CANDIDATE LOCALIZATION
# --------------------------------------------------

@router.post("/localize-candidates/{image_id}")
def localize_candidates(image_id: str):

    validate_image_id(image_id)

    mask_path = RESULT_DIR / f"{image_id}_stenosis_mask.png"

    probability_path = (
        RESULT_DIR / f"{image_id}_stenosis_probability.png"
    )

    original_path = find_original_image(image_id)

    if original_path is None:
        raise HTTPException(
            status_code=404,
            detail="Original angiography image not found."
        )

    if not mask_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Stenosis mask not found. Run segmentation first."
        )

    if not probability_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Stenosis probability map not found."
        )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    probability_image = cv2.imread(
        str(probability_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None or probability_image is None:
        raise HTTPException(
            status_code=500,
            detail="Unable to read segmentation output files."
        )

    if mask.shape != probability_image.shape:
        raise HTTPException(
            status_code=500,
            detail="Mask and probability map dimensions do not match."
        )

    probability_map = (
        probability_image.astype(np.float32) / 255.0
    )

    try:

        # Extract candidate regions
        extraction_results = extract_stenosis_candidates(
            probability_map=probability_map,
            binary_mask=mask,
            min_area=10
        )

        candidates = extraction_results["candidates"]

        # Check image dimensions
        original_image = cv2.imread(
            str(original_path),
            cv2.IMREAD_GRAYSCALE
        )

        if original_image is None:
            raise ValueError(
                "Unable to read original angiography image."
            )

        if original_image.shape != mask.shape:
            raise ValueError(
                "Original image and segmentation mask dimensions do not match."
            )

        # Output path
        output_path = (
            RESULT_DIR / f"{image_id}_stenosis_localized.png"
        )

        # Generate localized image
        localization_results = localize_stenosis_candidates(
            original_image_path=original_path,
            candidates=candidates,
            output_path=output_path
        )

        return {
            "status": "success",
            "image_id": image_id,
            "message": "Stenosis candidate localization completed.",
            "candidate_count": len(candidates),
            "localization_results": localization_results,
            "localized_image_url": (
                f"/api/analysis/localized-image/{image_id}"
            )
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Stenosis localization failed: {str(error)}"
        )


# --------------------------------------------------
# 9. RETRIEVE LOCALIZED STENOSIS IMAGE
# --------------------------------------------------

@router.get("/localized-image/{image_id}")
def get_localized_image(image_id: str):

    validate_image_id(image_id)

    localized_path = (
        RESULT_DIR / f"{image_id}_stenosis_localized.png"
    )

    if not localized_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Localized stenosis image not found."
        )

    return FileResponse(
        path=localized_path,
        media_type="image/png",
        filename=localized_path.name
    )
