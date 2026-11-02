from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Union
from datetime import datetime

# --- Auth Queries Schemas ---
class EmailExistsResponse(BaseModel):
    email: EmailStr
    exists: bool

class UserPublicRecord(BaseModel):
    user_id: Union[int, str]
    full_name: Optional[str] = None
    email: EmailStr
    created_at: Optional[datetime] = None

class UserNameUpdate(BaseModel):
    full_name: str = Field(..., min_length=1, max_length=100)

class UserNameUpdateResponse(BaseModel):
    user_id: Union[int, str]
    full_name: str
    message: str

# --- Case & Medical Image Schemas ---
class CaseResponse(BaseModel):
    case_id: int
    user_id: Optional[Union[int, str]] = None
    case_name: str
    imaging_modality: Optional[str] = None
    target_vessel: Optional[str] = None
    status: str
    created_at: Optional[datetime] = None

class MedicalImageResponse(BaseModel):
    image_id: int
    case_id: int
    image_name: str
    image_path: str
    image_format: Optional[str] = None
    image_size: Optional[int] = None
    uploaded_at: Optional[datetime] = None

# --- Results Schemas ---
class SegmentationResultResponse(BaseModel):
    segmentation_id: int
    image_id: int
    mask_path: Optional[str] = None
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
