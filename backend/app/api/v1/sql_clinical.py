from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from pydantic import EmailStr
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError, ProgrammingError

from app.api.dependencies import get_current_user, RoleChecker
from app.db.session import get_db
from app.models.user import User, UserRole
from app.schemas.clinical_schemas import (
    EmailExistsResponse, UserPublicRecord, UserNameUpdate, UserNameUpdateResponse,
    CaseCreateRequest, CaseUpdateRequest, CaseResponse,
    MedicalImageCreateRequest, MedicalImageUpdateRequest, MedicalImageResponse,
    MessageActionResponse,
    SegmentationResultCreateRequest, SegmentationResultResponse,
    StenosisResultCreateRequest, StenosisResultResponse,
    CountResponse, ModelDiceScore, AnalyticsSummaryResponse,
    DashboardAnalyticsResponse,
    DetailedModelPerformanceResponse,
    StenosisSeverityDistributionResponse,
    CaseAnalysisReportItem,
    CaseSummaryCountItem,
    ForeignKeyConstraintInfo,
    DatabaseTableInfo,
    TableColumnInfo,
    DataConsistencyCheckItem
)
from app.services.sql_clinical_service import SQLClinicalService

router = APIRouter(prefix="/sql", tags=["SQL-Backed Clinical & Analytics"])

# =====================================================================
# 1. Endpoints from auth_queries.sql
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
    try:
        record = SQLClinicalService.get_user_by_email_for_verification(db, email)
        if not record:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
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

# =====================================================================
# 2. Case Management Endpoints
# =====================================================================

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
        return SQLClinicalService.get_all_cases(db)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: The target table 'cases' may not exist. Details: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.post(
    "/cases",
    response_model=CaseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new medical case"
)
def create_case(
    case_in: CaseCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        return SQLClinicalService.create_case(db, case_in)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error creating case: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/cases/{case_id}",
    response_model=CaseResponse,
    summary="Retrieve a specific case by case_id"
)
def get_case_by_id(
    case_id: int = Path(..., ge=1),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        case = SQLClinicalService.get_case_by_id(db, case_id)
        if not case:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Case with ID {case_id} not found.")
        return case
    except HTTPException:
        raise
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error retrieving case: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/cases/user/{user_id}",
    response_model=List[CaseResponse],
    summary="Retrieve all cases for a specific user ID"
)
def get_cases_by_user_id(
    user_id: int = Path(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        return SQLClinicalService.get_cases_by_user_id(db, user_id)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error retrieving cases by user: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.patch(
    "/cases/{case_id}",
    response_model=CaseResponse,
    summary="Update case metadata or status"
)
def update_case(
    case_id: int = Path(..., ge=1),
    case_in: CaseUpdateRequest = ...,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        updated = SQLClinicalService.update_case(db, case_id, case_in)
        if not updated:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Case with ID {case_id} not found.")
        return updated
    except HTTPException:
        raise
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error updating case: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

# =====================================================================
# 3. Medical Images Management Endpoints
# =====================================================================

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
        return SQLClinicalService.get_all_medical_images(db)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: The target table 'medical_images' may not exist. Details: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.post(
    "/cases/{case_id}/images",
    response_model=MedicalImageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Insert/register medical image metadata for a case"
)
def insert_medical_image(
    case_id: int = Path(..., ge=1),
    image_in: MedicalImageCreateRequest = ...,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        return SQLClinicalService.insert_medical_image(db, case_id, image_in)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error inserting image: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/images/{image_id}",
    response_model=MedicalImageResponse,
    summary="Retrieve medical image metadata by image_id"
)
def get_image_by_id(
    image_id: int = Path(..., ge=1),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        image = SQLClinicalService.get_image_by_id(db, image_id)
        if not image:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Image with ID {image_id} not found.")
        return image
    except HTTPException:
        raise
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error retrieving image: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/cases/{case_id}/images",
    response_model=List[MedicalImageResponse],
    summary="Retrieve all images associated with a case_id"
)
def get_images_by_case_id(
    case_id: int = Path(..., ge=1),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        return SQLClinicalService.get_images_by_case_id(db, case_id)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error retrieving case images: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.patch(
    "/images/{image_id}",
    response_model=MedicalImageResponse,
    summary="Update medical image metadata"
)
def update_image_metadata(
    image_id: int = Path(..., ge=1),
    image_in: MedicalImageUpdateRequest = ...,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        updated = SQLClinicalService.update_image_metadata(db, image_id, image_in)
        if not updated:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Image with ID {image_id} not found.")
        return updated
    except HTTPException:
        raise
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error updating image: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.delete(
    "/images/{image_id}",
    response_model=MessageActionResponse,
    summary="Delete medical image metadata when permitted (no dependent results)"
)
def delete_medical_image(
    image_id: int = Path(..., ge=1),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        deleted = SQLClinicalService.delete_medical_image(db, image_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Image with ID {image_id} not found.")
        return MessageActionResponse(message="Image deleted successfully", detail=f"Image {image_id} removed.")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except HTTPException:
        raise
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error deleting image: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

# =====================================================================
# 4. Segmentation Results Endpoints
# =====================================================================

@router.get(
    "/segmentation-results",
    response_model=List[SegmentationResultResponse],
    summary="[auth_queries.sql Query 7] View all segmentation results"
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

@router.post(
    "/images/{image_id}/segmentation-results",
    response_model=SegmentationResultResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Insert segmentation result for an image"
)
def insert_segmentation_result(
    image_id: int = Path(..., ge=1),
    seg_in: SegmentationResultCreateRequest = ...,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        return SQLClinicalService.insert_segmentation_result(db, image_id, seg_in)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error inserting segmentation result: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/images/{image_id}/segmentation-results",
    response_model=List[SegmentationResultResponse],
    summary="Retrieve segmentation results for a specific image ID"
)
def get_segmentations_by_image_id(
    image_id: int = Path(..., ge=1),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        return SQLClinicalService.get_segmentations_by_image_id(db, image_id)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error retrieving image segmentation results: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/cases/{case_id}/segmentation-results",
    response_model=List[SegmentationResultResponse],
    summary="Retrieve all segmentation results associated with a case ID"
)
def get_segmentations_by_case_id(
    case_id: int = Path(..., ge=1),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        return SQLClinicalService.get_segmentations_by_case_id(db, case_id)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error retrieving case segmentation results: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/segmentation-results/latest",
    response_model=Optional[SegmentationResultResponse],
    summary="Retrieve the latest segmentation result"
)
def get_latest_segmentation_result(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        result = SQLClinicalService.get_latest_segmentation_result(db)
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No segmentation results found.")
        return result
    except HTTPException:
        raise
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error retrieving latest segmentation result: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

# =====================================================================
# 5. Stenosis Results Endpoints
# =====================================================================

@router.get(
    "/stenosis-results",
    response_model=List[StenosisResultResponse],
    summary="[auth_queries.sql Query 8] View all stenosis results (optional ?sort_by=confidence)"
)
def get_sql_stenosis_results(
    sort_by: Optional[str] = Query(None, description="Set to 'confidence' to sort by confidence score descending"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        if sort_by == "confidence":
            return SQLClinicalService.get_stenosis_sorted_by_confidence(db, descending=True)
        return SQLClinicalService.get_all_stenosis_results(db)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: The target table 'stenosis_results' may not exist. Details: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.post(
    "/images/{image_id}/stenosis-results",
    response_model=StenosisResultResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Insert stenosis detection result for an image"
)
def insert_stenosis_result(
    image_id: int = Path(..., ge=1),
    sten_in: StenosisResultCreateRequest = ...,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        return SQLClinicalService.insert_stenosis_result(db, image_id, sten_in)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error inserting stenosis result: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/images/{image_id}/stenosis-results",
    response_model=List[StenosisResultResponse],
    summary="Retrieve stenosis results for a specific image ID"
)
def get_stenosis_by_image_id(
    image_id: int = Path(..., ge=1),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        return SQLClinicalService.get_stenosis_by_image_id(db, image_id)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error retrieving image stenosis results: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/cases/{case_id}/stenosis-results",
    response_model=List[StenosisResultResponse],
    summary="Retrieve all stenosis results associated with a case ID"
)
def get_stenosis_by_case_id(
    case_id: int = Path(..., ge=1),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        return SQLClinicalService.get_stenosis_by_case_id(db, case_id)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error retrieving case stenosis results: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/stenosis-results/filter",
    response_model=List[StenosisResultResponse],
    summary="Filter stenosis results by severity"
)
def filter_stenosis_by_severity(
    severity: str = Query(..., min_length=1, description="Severity string, e.g. 'Mild', 'Moderate', 'Severe'"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        return SQLClinicalService.filter_stenosis_by_severity(db, severity)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error filtering stenosis results: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/stenosis-results/latest",
    response_model=Optional[StenosisResultResponse],
    summary="Retrieve the latest stenosis detection result"
)
def get_latest_stenosis_result(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        result = SQLClinicalService.get_latest_stenosis_result(db)
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No stenosis results found.")
        return result
    except HTTPException:
        raise
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error retrieving latest stenosis result: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

# =====================================================================
# 6. Analytics & Dashboard Endpoints
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

@router.get(
    "/dashboard/analytics",
    response_model=DashboardAnalyticsResponse,
    summary="Dashboard analytics: users, cases, images, stenosis counts, and recent activity"
)
def get_dashboard_analytics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        return SQLClinicalService.get_dashboard_analytics(db)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

# =====================================================================
# 5. Queries 4 to 11 from result_queries.sql
# =====================================================================

@router.get(
    "/analytics/model-performance/detailed",
    response_model=List[DetailedModelPerformanceResponse],
    summary="[result_queries.sql Query 4] Complete model performance analytics"
)
def get_detailed_model_performance(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Executes Query 4 from result_queries.sql.
    Returns complete segmentation model metrics: total results, Dice, IoU,
    precision, recall, and F1-score grouped by model name.
    """
    try:
        return SQLClinicalService.get_detailed_model_performance(db)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/analytics/stenosis-severity-distribution",
    response_model=List[StenosisSeverityDistributionResponse],
    summary="[result_queries.sql Query 5] Stenosis severity distribution"
)
def get_stenosis_severity_distribution(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Executes Query 5 from result_queries.sql.
    Returns total cases and average confidence score grouped by stenosis severity.
    """
    try:
        return SQLClinicalService.get_stenosis_severity_distribution(db)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/reports/case-analysis",
    response_model=List[CaseAnalysisReportItem],
    summary="[result_queries.sql Query 6] Case-wise medical image and complete analysis report"
)
def get_case_analysis_report(
    case_id: Optional[int] = Query(None, description="Optional case_id filter"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Executes Query 6 from result_queries.sql.
    Performs CTE-based multi-table aggregation across cases, medical images,
    segmentation summaries, and stenosis summaries.
    Avoids duplicate combinations of multiple segmentation and stenosis results per image.
    """
    try:
        return SQLClinicalService.get_case_analysis_report(db, case_id=case_id)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/reports/case-summary-counts",
    response_model=List[CaseSummaryCountItem],
    summary="[result_queries.sql Query 7] Summary counts of images and analysis results per case"
)
def get_case_summary_counts(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Executes Query 7 from result_queries.sql.
    Aggregates distinct image, segmentation, and stenosis counts grouped by case.
    """
    try:
        return SQLClinicalService.get_case_summary_counts(db)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

# --- Database Diagnostic Endpoints (Queries 8 to 11: Restricted to Administrators) ---

@router.get(
    "/diagnostics/foreign-keys",
    response_model=List[ForeignKeyConstraintInfo],
    summary="[result_queries.sql Query 8] Verify database foreign-key relationships (Admin Only)"
)
def diagnose_foreign_keys(
    db: Session = Depends(get_db),
    admin_user: User = Depends(RoleChecker([UserRole.ADMINISTRATOR]))
):
    """
    Executes Query 8 from result_queries.sql.
    Inspects information_schema foreign key constraints in the public schema.
    Restricted to Administrators only.
    """
    try:
        return SQLClinicalService.diagnose_foreign_keys(db)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/diagnostics/tables",
    response_model=List[DatabaseTableInfo],
    summary="[result_queries.sql Query 9] Verify all project database tables (Admin Only)"
)
def diagnose_tables(
    db: Session = Depends(get_db),
    admin_user: User = Depends(RoleChecker([UserRole.ADMINISTRATOR]))
):
    """
    Executes Query 9 from result_queries.sql.
    Verifies existence of the 5 base tables in information_schema.
    Restricted to Administrators only.
    """
    try:
        return SQLClinicalService.diagnose_tables(db)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/diagnostics/columns",
    response_model=List[TableColumnInfo],
    summary="[result_queries.sql Query 10] Verify table columns and data types (Admin Only)"
)
def diagnose_columns(
    table_name: Optional[str] = Query(None, description="Filter by table name"),
    db: Session = Depends(get_db),
    admin_user: User = Depends(RoleChecker([UserRole.ADMINISTRATOR]))
):
    """
    Executes Query 10 from result_queries.sql.
    Inspects table column definitions and types in information_schema.
    Restricted to Administrators only.
    """
    try:
        return SQLClinicalService.diagnose_columns(db, table_name=table_name)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

@router.get(
    "/diagnostics/foreign-key-consistency",
    response_model=List[DataConsistencyCheckItem],
    summary="[result_queries.sql Query 11] Verify foreign-key data consistency / orphan records (Admin Only)"
)
def diagnose_fk_consistency(
    db: Session = Depends(get_db),
    admin_user: User = Depends(RoleChecker([UserRole.ADMINISTRATOR]))
):
    """
    Executes Query 11 from result_queries.sql.
    Runs orphan record checks across cases, images, segmentations, and stenosis tables.
    Restricted to Administrators only.
    """
    try:
        return SQLClinicalService.diagnose_fk_consistency(db)
    except (OperationalError, ProgrammingError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database query error: {e.orig if hasattr(e, 'orig') else str(e)}"
        )

