from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

from app.api.dependencies import get_current_user
from app.core.config import settings
from app.models.user import User
from app.services.inference_service import (
    SegmentationModelType,
    inference_service,
)
from app.services.storage_service import storage_service, StorageError
from app.services.sql_clinical_service import SQLClinicalService
from app.db.session import get_db
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError, ProgrammingError

from app.schemas.clinical_schemas import (
    ImageUploadResponse,
    PersistedInferenceResponse,
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
    summary="Run segmentation inference on an uploaded angiography image (stateless)",
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
    and return the binary mask, overlay, and metadata as base64-encoded data.
    No database persistence or file storage is performed."""
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


@router.post(
    "/upload-and-predict",
    response_model=PersistedInferenceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload an image, run inference, and persist results to the database",
)
def upload_and_predict(
    file: UploadFile = File(..., description="Angiography image (PNG, JPG, TIFF, BMP)"),
    case_id: int = Form(..., description="Case ID to associate the image with"),
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
    db: Session = Depends(get_db),
):
    """
    End-to-end workflow:
    1. Validate and securely store the uploaded image
    2. Create a medical_images record linked to the case (ownership verified)
    3. Run segmentation inference (U-Net or Attention U-Net)
    4. Store mask and overlay files
    5. Create a segmentation_results record linked to the image
    6. Return the persisted result
    """
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
        # Read file content
        image_bytes = file.file.read()

        # Validate storage
        storage_service.validate_upload_file(file)

        # Compute content hash for deduplication reference
        content_hash = storage_service.compute_content_hash(image_bytes)

        # Generate storage path
        relative_path, full_path = storage_service.generate_storage_path(
            file.filename or "uploaded_image.png", content_hash
        )

        # Determine user_id for ownership check (admin sees all, regular user filtered)
        user_id = None if current_user.role == "administrator" else current_user.id

        # Step 1: Create medical_images record (verifies case ownership)
        try:
            image_record = SQLClinicalService.create_medical_image_record(
                db=db,
                case_id=case_id,
                image_name=file.filename or "uploaded_image.png",
                image_path=relative_path,
                image_format=file.content_type,
                image_size=len(image_bytes),
                user_id=user_id,
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

        image_id = image_record["image_id"]

        # Step 2: Save the original image file
        try:
            storage_service.save_file(image_bytes, full_path)
        except Exception as e:
            # Clean up DB record if file save fails
            SQLClinicalService.delete_medical_image(db, image_id, user_id=user_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to store image file: {type(e).__name__}",
            )

        # Step 3: Run inference
        try:
            result = inference_service.predict(
                image_input=image_bytes,
                model_type=model_type,
                threshold=threshold,
                generate_overlays=True,
                encode_base64=False,  # We store files, not base64, for the persisted version
            )
        except ValueError as e:
            # Clean up stored file and DB record
            storage_service.delete_file(relative_path)
            SQLClinicalService.delete_medical_image(db, image_id, user_id=user_id)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            storage_service.delete_file(relative_path)
            SQLClinicalService.delete_medical_image(db, image_id, user_id=user_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference failed: {type(e).__name__}: {e}",
            )

        # Step 4: Save mask and overlay files
        mask_filename = f"mask_{image_id}.png"
        overlay_filename = f"overlay_{image_id}.png"
        mask_relative, mask_full = storage_service.generate_storage_path(mask_filename, content_hash)
        overlay_relative, overlay_full = storage_service.generate_storage_path(overlay_filename, content_hash)

        try:
            storage_service.save_image_array(result.mask_binary * 255, mask_full, is_rgb=False)
            storage_service.save_image_array(result.overlay_rgb, overlay_full, is_rgb=True)
        except Exception as e:
            # Clean up everything on failure
            storage_service.delete_file(relative_path)
            storage_service.delete_file(mask_relative)
            storage_service.delete_file(overlay_relative)
            SQLClinicalService.delete_medical_image(db, image_id, user_id=user_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to store inference outputs: {type(e).__name__}",
            )

        # Step 5: Create segmentation_results record
        try:
            seg_record = SQLClinicalService.create_segmentation_result_record(
                db=db,
                image_id=image_id,
                mask_path=mask_relative,
                overlay_path=overlay_relative,
                model_name=result.model_name,
                dice_score=None,  # No ground truth available for computed Dice
                iou_score=None,
                precision_score=None,
                recall_score=None,
                f1_score=None,
                mean_vessel_diameter=None,
                processed_mask_path=mask_relative,
                centerline_path=None,
                model_version=None,
                user_id=user_id,
            )
        except ValueError as e:
            # Clean up stored files and DB record
            storage_service.delete_file(relative_path)
            storage_service.delete_file(mask_relative)
            storage_service.delete_file(overlay_relative)
            SQLClinicalService.delete_medical_image(db, image_id, user_id=user_id)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

        segmentation_id = seg_record["segmentation_id"]

        # Encode mask and overlay as base64 for the response
        mask_b64 = inference_service.encode_image_to_base64(result.mask_binary * 255)
        overlay_b64 = inference_service.encode_image_to_base64(result.overlay_rgb)

        return PersistedInferenceResponse(
            image_id=image_id,
            segmentation_id=segmentation_id,
            case_id=case_id,
            model_name=result.model_name,
            threshold=result.threshold,
            original_height=result.original_height,
            original_width=result.original_width,
            segmented_pixels=result.segmented_pixels,
            vessel_area_ratio=result.vessel_area_ratio,
            storage_identifier=relative_path,
            mask_storage_identifier=mask_relative,
            overlay_storage_identifier=overlay_relative,
            mask_base64=mask_b64,
            overlay_base64=overlay_b64,
            created_at=seg_record.get("created_at"),
        )

    except HTTPException:
        raise
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error: {e.orig if hasattr(e, 'orig') else str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {type(e).__name__}: {e}",
        )


@router.get(
    "/images/{image_id}",
    response_model=ImageUploadResponse,
    summary="Get image metadata (owner or admin only)",
)
def get_image_metadata(
    image_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve metadata for a stored image. Ownership is enforced."""
    user_id = None if current_user.role == "administrator" else current_user.id
    try:
        image = SQLClinicalService.get_image_by_id(db, image_id, user_id=user_id)
        if not image:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found.")
        return ImageUploadResponse(
            image_id=image["image_id"],
            case_id=image["case_id"],
            image_name=image["image_name"],
            image_format=image.get("image_format"),
            image_size=image.get("image_size"),
            storage_identifier=image["image_path"],
            uploaded_at=image.get("uploaded_at"),
        )
    except HTTPException:
        raise
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error: {e.orig if hasattr(e, 'orig') else str(e)}",
        )


@router.get(
    "/images/{image_id}/file",
    summary="Download the original stored image file (owner or admin only)",
)
def get_image_file(
    image_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Serve the original image file from storage. Ownership is enforced."""
    user_id = None if current_user.role == "administrator" else current_user.id
    try:
        image = SQLClinicalService.get_image_by_id(db, image_id, user_id=user_id)
        if not image:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found.")

        storage_id = image["image_path"]
        try:
            full_path = storage_service.get_full_path(storage_id)
        except StorageError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file not found on storage.")

        if not full_path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file not found on storage.")

        return FileResponse(
            path=str(full_path),
            media_type=image.get("image_format") or "application/octet-stream",
            filename=image["image_name"],
        )
    except HTTPException:
        raise
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error: {e.orig if hasattr(e, 'orig') else str(e)}",
        )


@router.get(
    "/segmentation-results/{segmentation_id}",
    summary="Get segmentation result metadata (owner or admin only)",
)
def get_segmentation_result(
    segmentation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve metadata for a stored segmentation result. Ownership is enforced."""
    user_id = None if current_user.role == "administrator" else current_user.id
    try:
        result = SQLClinicalService.get_segmentation_result_by_id(
            db, segmentation_id, user_id=user_id
        )
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segmentation result not found.")
        return result
    except HTTPException:
        raise
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error: {e.orig if hasattr(e, 'orig') else str(e)}",
        )


@router.get(
    "/masks/{segmentation_id}",
    summary="Download the stored segmentation mask file (owner or admin only)",
)
def get_mask_file(
    segmentation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Serve the stored segmentation mask file. Ownership is enforced."""
    user_id = None if current_user.role == "administrator" else current_user.id
    try:
        result = SQLClinicalService.get_segmentation_result_by_id(
            db, segmentation_id, user_id=user_id
        )
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segmentation result not found.")

        mask_path = result.get("processed_mask_path") or result.get("mask_path")
        if not mask_path:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mask file not found.")

        try:
            full_path = storage_service.get_full_path(mask_path)
        except StorageError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mask file not found on storage.")

        if not full_path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mask file not found on storage.")

        return FileResponse(
            path=str(full_path),
            media_type="image/png",
            filename=f"mask_{segmentation_id}.png",
        )
    except HTTPException:
        raise
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error: {e.orig if hasattr(e, 'orig') else str(e)}",
        )


@router.get(
    "/overlays/{segmentation_id}",
    summary="Download the stored segmentation overlay file (owner or admin only)",
)
def get_overlay_file(
    segmentation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Serve the stored segmentation overlay file. Ownership is enforced."""
    user_id = None if current_user.role == "administrator" else current_user.id
    try:
        result = SQLClinicalService.get_segmentation_result_by_id(
            db, segmentation_id, user_id=user_id
        )
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segmentation result not found.")

        overlay_path = result.get("overlay_path")
        if not overlay_path:
            # Fallback: try to find overlay via image_id
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Overlay file not found.")

        try:
            full_path = storage_service.get_full_path(overlay_path)
        except StorageError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Overlay file not found on storage.")

        if not full_path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Overlay file not found on storage.")

        return FileResponse(
            path=str(full_path),
            media_type="image/png",
            filename=f"overlay_{segmentation_id}.png",
        )
    except HTTPException:
        raise
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error: {e.orig if hasattr(e, 'orig') else str(e)}",
        )
