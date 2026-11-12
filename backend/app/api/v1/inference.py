from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.dependencies import get_current_user
from app.models.user import User
from app.services.inference_service import (
    SegmentationModelType,
    inference_service,
)

router = APIRouter(prefix="/inference", tags=["ML Inference"])

ALLOWED_IMAGE_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/tiff", "image/bmp"}


def _validate_image_file(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported image type: {file.content_type or 'unknown'}. "
                f"Allowed types: {', '.join(sorted(ALLOWED_IMAGE_CONTENT_TYPES))}"
            ),
        )


@router.get(
    "/models",
    response_model=List[str],
    summary="List available segmentation models",
)
def list_available_models(
    current_user: User = Depends(get_current_user),
):
    """Return the names of loaded segmentation models."""
    if not inference_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference models are not loaded yet.",
        )
    return list(inference_service._models.keys())


@router.post(
    "/predict",
    summary="Run segmentation inference on an uploaded angiography image",
)
def predict_segmentation(
    file: UploadFile = File(..., description="Angiography image (PNG, JPG, TIFF, BMP)"),
    model_type: str = Form(
        default=SegmentationModelType.UNET.value,
        description="Segmentation model: 'unet' or 'attention_unet'",
    ),
    threshold: Optional[float] = Form(
        default=None,
        ge=0.0,
        le=1.0,
        description="Binarization threshold (default 0.3)",
    ),
    current_user: User = Depends(get_current_user),
):
    """Accept an uploaded angiography image, run segmentation inference,
    and return the binary mask, overlay, and metadata as base64-encoded data."""
    _validate_image_file(file)

    if not inference_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference models are not loaded yet.",
        )

    if model_type not in (SegmentationModelType.UNET.value, SegmentationModelType.ATTENTION_UNET.value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid model_type '{model_type}'. "
                f"Allowed values: '{SegmentationModelType.UNET.value}', "
                f"'{SegmentationModelType.ATTENTION_UNET.value}'"
            ),
        )

    try:
        image_bytes = file.file.read()
        result = inference_service.predict(
            image_input=image_bytes,
            model_type=model_type,
            threshold=threshold,
            generate_overlays=True,
            encode_base64=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {type(e).__name__}: {e}",
        )

    return {
        "model_name": result.model_name,
        "original_height": result.original_height,
        "original_width": result.original_width,
        "threshold": result.threshold,
        "segmented_pixels": result.segmented_pixels,
        "vessel_area_ratio": result.vessel_area_ratio,
        "mask_base64": result.mask_base64,
        "overlay_base64": result.overlay_base64,
    }
