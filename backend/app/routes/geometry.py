import json
from pathlib import Path
from uuid import UUID

import cv2
import numpy as np
from scipy.ndimage import median_filter

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.services.postprocessing import postprocess_mask
from app.services.centerline import extract_centerline
from app.services.branch_diameter import calculate_branch_diameters
from app.services.severity_classification import (
    classify_stenosis_severity
)


router = APIRouter(
    prefix="/api/analysis",
    tags=["Image Processing"]
)

BACKEND_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = BACKEND_DIR / "results" / "api_predictions"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# COMMON HELPERS
# --------------------------------------------------

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

    suffix = (
        "_cleaned_mask.png"
        if model_type == "syntax"
        else "_stenosis_cleaned_mask.png"
    )

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
        "cleaned_mask_url": (
            f"/api/analysis/cleaned-mask/{image_id}"
            f"?model_type={model_type}"
        ),
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

    if model_type not in ("syntax", "stenosis"):
        raise HTTPException(
            status_code=400,
            detail="Invalid model type."
        )

    suffix = (
        "_cleaned_mask.png"
        if model_type == "syntax"
        else "_stenosis_cleaned_mask.png"
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

    suffix = (
        "_cleaned_mask.png"
        if model_type == "syntax"
        else "_stenosis_cleaned_mask.png"
    )

    cleaned_mask_path = RESULTS_DIR / f"{image_id}{suffix}"

    mask = load_mask(cleaned_mask_path)

    try:
        result = extract_centerline(mask)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    centerline_suffix = (
        "_centerline.png"
        if model_type == "syntax"
        else "_stenosis_centerline.png"
    )

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
        "message": (
            "Centerline extraction and pixel-based "
            "diameter estimation completed."
        ),
        "image_id": image_id,
        "model_type": model_type,
        "centerline_url": (
            f"/api/analysis/centerline-image/{image_id}"
            f"?model_type={model_type}"
        ),
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

    suffix = (
        "_centerline.png"
        if model_type == "syntax"
        else "_stenosis_centerline.png"
    )

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


# --------------------------------------------------
# 5. STENOSIS PERCENTAGE AND SEVERITY ANALYSIS
# --------------------------------------------------

@router.post("/stenosis-percentage/{image_id}")
def calculate_stenosis_percentage(
    image_id: str,
    model_type: str = "syntax"
):
    image_id = validate_image_id(image_id)

    # Diameter-based narrowing estimation is supported only
    # using the Syntax U-Net vessel geometry mask.
    if model_type != "syntax":
        raise HTTPException(
            status_code=400,
            detail=(
                "Experimental narrowing estimation currently "
                "supports only the Syntax U-Net vessel geometry mask. "
                "Attention U-Net V4 outputs are not used for this "
                "diameter-based measurement."
            )
        )

    mask_path = RESULTS_DIR / f"{image_id}_cleaned_mask.png"

    mask = load_mask(mask_path)

    try:
        # Extract centerline from the cleaned Syntax vessel mask.
        centerline_result = extract_centerline(mask)

        centerline_mask = centerline_result["centerline_mask"]

        # Calculate branch-wise diameter profiles.
        diameter_result = calculate_branch_diameters(
            mask,
            centerline_mask
        )

        results = []

        for branch in diameter_result["branches"]:

            profile = branch["diameter_profile"]

            diameters = np.array(
                [
                    point["diameter_pixels"]
                    for point in profile
                ],
                dtype=float
            )

            # Skip invalid diameter profiles.
            if (
                len(diameters) == 0
                or not np.all(np.isfinite(diameters))
                or np.any(diameters < 0)
            ):
                continue

            reference_window = 10
            smoothing_size = 5

            if len(diameters) < 2 * reference_window + 5:
                continue

            # Smooth diameter fluctuations.
            smoothed = median_filter(
                diameters,
                size=smoothing_size,
                mode="nearest"
            )

            candidates = []

            # Identify local narrowing candidates.
            for i in range(
                reference_window,
                len(smoothed) - reference_window
            ):

                left_reference = float(
                    np.median(
                        smoothed[i-reference_window:i]
                    )
                )

                right_reference = float(
                    np.median(
                        smoothed[i+1:i+1+reference_window]
                    )
                )

                if (
                    left_reference <= 0
                    or right_reference <= 0
                ):
                    continue

                reference = (
                    left_reference + right_reference
                ) / 2.0

                minimum = float(smoothed[i])

                if (
                    minimum < left_reference
                    and minimum < right_reference
                ):

                    # Compare the two reference regions.
                    asymmetry = (
                        abs(left_reference - right_reference)
                        / reference
                    ) * 100.0

                    # Experimental diameter-based narrowing estimate.
                    narrowing = max(
                        0.0,
                        (1.0 - minimum / reference) * 100.0
                    )

                    if asymmetry <= 15:
                        quality = "relatively_consistent"
                    elif asymmetry <= 30:
                        quality = "needs_review"
                    else:
                        quality = "poor"

                    # Experimental severity classification.
                    severity_result = classify_stenosis_severity(
                        narrowing
                    )

                    candidates.append({
                        "position": int(i),

                        "x": int(profile[i]["x"]),
                        "y": int(profile[i]["y"]),

                        "minimum_diameter_pixels": round(
                            minimum, 3
                        ),

                        "left_reference_pixels": round(
                            left_reference, 3
                        ),

                        "right_reference_pixels": round(
                            right_reference, 3
                        ),

                        "reference_diameter_pixels": round(
                            reference, 3
                        ),

                        "reference_asymmetry_percent": round(
                            asymmetry, 2
                        ),

                        "estimated_narrowing_percent": round(
                            narrowing, 2
                        ),

                        "reference_quality": quality,

                        "severity_classification": severity_result
                    })

            if not candidates:
                continue

            # Select the candidate with the highest estimated
            # percentage narrowing.
            best = max(
                candidates,
                key=lambda item: item["estimated_narrowing_percent"]
            )

            results.append({
                "branch_id": int(branch["branch_id"]),
                "branch_length": len(profile),
                "candidate_count": len(candidates),
                "deepest_candidate": best
            })

        # Create report.
        report = {
            "image_id": image_id,

            "geometry_mask_model": "Syntax U-Net",
            "geometry_model_type": model_type,
            "analysis_type": "experimental_pixel_based_narrowing",

            "measurement_unit": "pixels",

            "measurement_method": (
                "2 x Euclidean distance transform"
            ),

            "branches_extracted": diameter_result["branch_count"],

            "branches_evaluated": len(results),

            "experimental": True,

            "clinically_validated": False,

            "severity_classification_enabled": True,

            "severity_thresholds": "provisional",

            "results": results,

            "notice": (
                "Experimental pixel-based narrowing estimates and "
                "provisional severity categories only. The vessel "
                "geometry is predicted by Syntax U-Net. Severity "
                "thresholds are not clinically validated. This output "
                "is not a clinical diagnosis or a validated stenosis "
                "measurement."
            )
        }

        # Save image-specific JSON report.
        output_path = (
            RESULTS_DIR /
            f"{image_id}_stenosis_percentage_analysis.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(report, file, indent=4)

        return {
            "status": "success",
            "message": (
                "Experimental narrowing and severity analysis completed."
            ),
            **report,
            "report_url": (
                f"/api/analysis/stenosis-percentage-report/{image_id}"
            )
        }

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Stenosis percentage analysis failed: "
                f"{str(error)}"
            )
        )


# --------------------------------------------------
# 6. RETRIEVE STENOSIS PERCENTAGE REPORT
# --------------------------------------------------

@router.get("/stenosis-percentage-report/{image_id}")
def get_stenosis_percentage_report(image_id: str):

    image_id = validate_image_id(image_id)

    report_path = (
        RESULTS_DIR /
        f"{image_id}_stenosis_percentage_analysis.json"
    )

    if not report_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Stenosis percentage report not found."
        )

    return FileResponse(
        str(report_path),
        media_type="application/json",
        filename=report_path.name
    )