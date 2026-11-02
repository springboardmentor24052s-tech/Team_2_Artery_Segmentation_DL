from typing import List, Union
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from pydantic import EmailStr
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError, ProgrammingError

from app.api.dependencies import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.clinical_schemas import (
    EmailExistsResponse, UserPublicRecord, UserNameUpdate, UserNameUpdateResponse,
    CaseResponse, MedicalImageResponse,
    SegmentationResultResponse, StenosisResultResponse,
    CountResponse, ModelDiceScore, AnalyticsSummaryResponse
)
from app.services.sql_clinical_service import SQLClinicalService

router = APIRouter(prefix="/sql", tags=["SQL-Backed Clinical & Analytics"])

# =====================================================================
# 1. Endpoints for auth_queries.sql
# =====================================================================

@router.get(
    "/users/email-exists",
    response_model=EmailExistsResponse,
    summary="[auth_queries.sql Query 1] Check whether an email already exists"
)
def check_email_exists(
    email: EmailStr = Query(..., description="Email address to check for existence"),
    db: Session = Depends(get_db)
):
    try:
        exists = SQLClinicalService.check_email_exists(db, email)
        return EmailExistsResponse(email=email, exists=exists)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/users/by-email",
    response_model=UserPublicRecord,
    summary="[auth_queries.sql Query 2] Retrieve user for login verification (Excludes password hash)"
)
def get_user_by_email_verification(
    email: EmailStr = Query(..., description="Email address of user to retrieve"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Executes Query 2 from auth_queries.sql to retrieve user details.
    Password hash is strictly omitted from the API response for security.
    """
    try:
        record = SQLClinicalService.get_user_by_email_for_verification(db, email)
        if not record:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        # Return public profile without password_hash
        return UserPublicRecord(
            user_id=record["user_id"],
            full_name=record.get("full_name"),
            email=record["email"],
            created_at=record.get("created_at")
        )
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/users",
    response_model=List[UserPublicRecord],
    summary="[auth_queries.sql Query 3] View registered users (without password hashes)"
)
def get_registered_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        records = SQLClinicalService.get_registered_users(db)
        return [
            UserPublicRecord(
                user_id=r["user_id"],
                full_name=r.get("full_name"),
                email=r["email"],
                created_at=r.get("created_at")
            )
            for r in records
        ]
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.patch(
    "/users/{user_id}/name",
    response_model=UserNameUpdateResponse,
    summary="[auth_queries.sql Query 4] Update user's name"
)
def update_user_name(
    user_id: str = Path(..., description="User ID to update"),
    name_in: UserNameUpdate = ...,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        updated = SQLClinicalService.update_user_name(db, user_id=user_id, full_name=name_in.full_name)
        if not updated:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found or name unchanged")
        return UserNameUpdateResponse(
            user_id=user_id,
            full_name=name_in.full_name,
            message="User full name updated successfully"
        )
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/cases",
    response_model=List[CaseResponse],
    summary="[auth_queries.sql Query 5] View all cases"
)
def get_sql_cases(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        cases = SQLClinicalService.get_all_cases(db)
        return cases
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: The target table 'cases' may not exist. Details: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/images",
    response_model=List[MedicalImageResponse],
    summary="[auth_queries.sql Query 6] View all medical images"
)
def get_sql_medical_images(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        images = SQLClinicalService.get_all_medical_images(db)
        return images
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: The target table 'medical_images' may not exist. Details: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/segmentation-results",
    response_model=List[SegmentationResultResponse],
    summary="[auth_queries.sql Query 7] View segmentation results"
)
def get_sql_segmentation_results(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        return SQLClinicalService.get_all_segmentation_results(db)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: The target table 'segmentation_results' may not exist. Details: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/stenosis-results",
    response_model=List[StenosisResultResponse],
    summary="[auth_queries.sql Query 8] View stenosis results"
)
def get_sql_stenosis_results(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        return SQLClinicalService.get_all_stenosis_results(db)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: The target table 'stenosis_results' may not exist. Details: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

# =====================================================================
# 2. Endpoints for result_queries.sql
# =====================================================================

@router.get(
    "/analytics/segmentation-count",
    response_model=CountResponse,
    summary="[result_queries.sql Query 1] Count segmentation results"
)
def get_segmentation_count(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        count = SQLClinicalService.get_segmentation_count(db)
        return CountResponse(count=count)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/analytics/stenosis-count",
    response_model=CountResponse,
    summary="[result_queries.sql Query 2] Count stenosis results"
)
def get_stenosis_count(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        count = SQLClinicalService.get_stenosis_count(db)
        return CountResponse(count=count)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/analytics/model-performance",
    response_model=List[ModelDiceScore],
    summary="[result_queries.sql Query 3] Compare segmentation model performance by average Dice score"
)
def get_model_performance(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        return SQLClinicalService.get_model_performance(db)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/analytics/summary",
    response_model=AnalyticsSummaryResponse,
    summary="[result_queries.sql Combined] Model performance & count analytics"
)
def get_sql_analytics_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        seg_count = SQLClinicalService.get_segmentation_count(db)
        sten_count = SQLClinicalService.get_stenosis_count(db)
        performance = SQLClinicalService.get_model_performance(db)
        return AnalyticsSummaryResponse(
            total_segmentation_results=seg_count,
            total_stenosis_results=sten_count,
            model_performance=performance,
            source="sql_database"
        )
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: Analytics tables not available. Details: {e.orig if hasattr(e, 'orig') else str(e)}"
        )
