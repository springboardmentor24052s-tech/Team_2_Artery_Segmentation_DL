import uuid
from datetime import datetime
from typing import Optional, List, Union
from pydantic import BaseModel, EmailStr, Field

# --- Auth Queries Schemas ---
class EmailExistsResponse(BaseModel):
    email: EmailStr
    exists: bool

class UserPublicRecord(BaseModel):
    user_id: uuid.UUID
    name: Optional[str] = None
    email: EmailStr
    created_at: Optional[datetime] = None

class UserNameUpdate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)

class UserNameUpdateResponse(BaseModel):
    user_id: uuid.UUID
    name: str
    message: str

# --- Case Schemas ---
class CaseCreateRequest(BaseModel):
    case_name: str = Field(..., min_length=1, max_length=150)
    imaging_modality: Optional[str] = Field("CTA", max_length=50)
    target_vessel: Optional[str] = Field("LAD", max_length=100)
    status: Optional[str] = Field("pending", max_length=30)

class CaseUpdateRequest(BaseModel):
    case_name: Optional[str] = Field(None, min_length=1, max_length=150)
    imaging_modality: Optional[str] = Field(None, max_length=50)
    target_vessel: Optional[str] = Field(None, max_length=100)
    status: Optional[str] = Field(None, max_length=30)

class CaseResponse(BaseModel):
    case_id: int
    user_id: Optional[uuid.UUID] = None
    case_name: str
    imaging_modality: Optional[str] = None
    target_vessel: Optional[str] = None
    status: str
    created_at: Optional[datetime] = None

# --- Medical Image Schemas ---
class MedicalImageCreateRequest(BaseModel):
    image_name: str = Field(..., min_length=1, max_length=255)
    image_path: str = Field(..., min_length=1)
    image_format: Optional[str] = Field("DICOM", max_length=10)
    image_size: Optional[int] = Field(None, ge=0)

class MedicalImageUpdateRequest(BaseModel):
    image_name: Optional[str] = Field(None, min_length=1, max_length=255)
    image_path: Optional[str] = Field(None, min_length=1)
    image_format: Optional[str] = Field(None, max_length=10)
    image_size: Optional[int] = Field(None, ge=0)

class MedicalImageResponse(BaseModel):
    image_id: int
    case_id: int
    image_name: str
    image_path: str
    image_format: Optional[str] = None
    image_size: Optional[int] = None
    uploaded_at: Optional[datetime] = None

class MessageActionResponse(BaseModel):
    message: str
    detail: Optional[str] = None

# --- Segmentation Results Schemas ---
class SegmentationResultCreateRequest(BaseModel):
    mask_path: Optional[str] = None
    overlay_path: Optional[str] = None
    model_name: Optional[str] = Field(None, max_length=100)
    dice_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    iou_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    processed_mask_path: Optional[str] = None
    centerline_path: Optional[str] = None
    model_version: Optional[str] = Field(None, max_length=50)
    precision_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    recall_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    mean_vessel_diameter: Optional[float] = None

class SegmentationResultResponse(BaseModel):
    segmentation_id: int
    image_id: int
    mask_path: Optional[str] = None
    overlay_path: Optional[str] = None
    model_name: Optional[str] = None
    dice_score: Optional[float] = None
    created_at: Optional[datetime] = None
    iou_score: Optional[float] = None
    processed_mask_path: Optional[str] = None
    centerline_path: Optional[str] = None
    model_version: Optional[str] = None
    precision_score: Optional[float] = None
    recall_score: Optional[float] = None
    f1_score: Optional[float] = None
    mean_vessel_diameter: Optional[float] = None

# --- Stenosis Results Schemas ---
class StenosisResultCreateRequest(BaseModel):
    location: Optional[str] = None
    severity: Optional[str] = Field(None, max_length=30)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    estimated_diameter: Optional[float] = None

class StenosisResultResponse(BaseModel):
    stenosis_id: int
    image_id: int
    location: Optional[str] = None
    severity: Optional[str] = None
    confidence_score: Optional[float] = None
    created_at: Optional[datetime] = None
    estimated_diameter: Optional[float] = None

# --- Analytics Schemas (result_queries.sql) ---
class CountResponse(BaseModel):
    count: int

class ModelDiceScore(BaseModel):
    model_name: str
    average_dice_score: Optional[float] = None

class AnalyticsSummaryResponse(BaseModel):
    total_segmentation_results: int
    total_stenosis_results: int
    model_performance: List[ModelDiceScore]
    source: str = "sql_database"

class RecentActivityItem(BaseModel):
    image_id: int
    image_name: Optional[str] = None
    analysis_type: str  # "Segmentation" or "Stenosis"
    model_or_severity: Optional[str] = None
    analysis_date: Optional[datetime] = None

class DashboardAnalyticsResponse(BaseModel):
    total_users: int
    total_cases: int
    total_images: int
    total_stenosis_detections: int
    recent_activity: List[RecentActivityItem]
    source: str = "sql_database"

# --- result_queries.sql Queries 4 to 11 Schemas ---

class DetailedModelPerformanceResponse(BaseModel):
    model_name: str
    total_results: int
    average_dice: Optional[float] = None
    average_iou: Optional[float] = None
    average_precision: Optional[float] = None
    average_recall: Optional[float] = None
    average_f1: Optional[float] = None

class StenosisSeverityDistributionResponse(BaseModel):
    severity: Optional[str] = None
    total_cases: int
    average_confidence: Optional[float] = None

class CaseAnalysisReportItem(BaseModel):
    case_id: int
    case_name: Optional[str] = None
    imaging_modality: Optional[str] = None
    target_vessel: Optional[str] = None
    image_id: int
    image_name: Optional[str] = None
    total_segmentation_results: int = 0
    model_names: Optional[str] = None
    average_dice: Optional[float] = None
    average_iou: Optional[float] = None
    total_stenosis_results: int = 0
    detected_severities: Optional[str] = None
    average_confidence: Optional[float] = None
    average_estimated_diameter: Optional[float] = None

class CaseSummaryCountItem(BaseModel):
    case_id: int
    case_name: Optional[str] = None
    total_images: int
    total_segmentation_results: int
    total_stenosis_results: int

class ForeignKeyConstraintInfo(BaseModel):
    table_name: str
    column_name: str
    referenced_table: str
    referenced_column: str
    constraint_name: str

class DatabaseTableInfo(BaseModel):
    table_name: str

class TableColumnInfo(BaseModel):
    table_name: str
    column_name: str
    data_type: str
    is_nullable: str
    column_default: Optional[str] = None

class DataConsistencyCheckItem(BaseModel):
    check_name: str
    invalid_records: int


# --- Image Upload & Persisted Inference Schemas ---
class ImageUploadResponse(BaseModel):
    image_id: int
    case_id: int
    image_name: str
    image_format: Optional[str] = None
    image_size: Optional[int] = None
    storage_identifier: str
    uploaded_at: Optional[datetime] = None


class PersistedInferenceResponse(BaseModel):
    image_id: int
    segmentation_id: int
    case_id: int
    model_name: str
    threshold: float
    original_height: int
    original_width: int
    segmented_pixels: int
    vessel_area_ratio: float
    storage_identifier: str
    mask_storage_identifier: Optional[str] = None
    overlay_storage_identifier: Optional[str] = None
    mask_base64: Optional[str] = None
    overlay_base64: Optional[str] = None
    created_at: Optional[datetime] = None

