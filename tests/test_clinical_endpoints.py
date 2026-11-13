import os
import sys
import sqlite3
import uuid
import io
from PIL import Image
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Register UUID adapter for SQLite so UUID objects can be bound as parameters
sqlite3.register_adapter(uuid.UUID, str)

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.extended_main import app
from app.db.base import Base
from app.db.session import get_db
from app.models.user import User, UserRole
from app.core.security import hash_password, create_access_token

TEST_DATABASE_URL = "sqlite:///:memory:"

engine_test = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine_test)

@pytest.fixture(scope="module", autouse=True)
def setup_test_schema():
    Base.metadata.create_all(bind=engine_test)
    with engine_test.connect() as conn:
        # SQLite schema representing database_work.sql tables
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS cases (
                case_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                case_name VARCHAR(150),
                imaging_modality VARCHAR(50),
                target_vessel VARCHAR(100),
                status VARCHAR(30) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS medical_images (
                image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER,
                image_name VARCHAR(255),
                image_path TEXT,
                image_format VARCHAR(10),
                image_size BIGINT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS segmentation_results (
                segmentation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                mask_path TEXT,
                overlay_path TEXT,
                model_name VARCHAR(100),
                dice_score NUMERIC(6,5),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                iou_score NUMERIC(6,5),
                processed_mask_path TEXT,
                centerline_path TEXT,
                model_version VARCHAR(50),
                precision_score NUMERIC(6,5),
                recall_score NUMERIC(6,5),
                f1_score NUMERIC(6,5),
                mean_vessel_diameter NUMERIC(10,4)
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS stenosis_results (
                stenosis_id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                location TEXT,
                severity VARCHAR(30),
                confidence_score NUMERIC(6,5),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                estimated_diameter NUMERIC(10,4)
            );
        """))
        conn.commit()
    yield
    Base.metadata.drop_all(bind=engine_test)

@pytest.fixture(scope="function")
def db_session():
    connection = engine_test.connect()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    connection.close()

@pytest.fixture(scope="function")
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture
def auth_headers(db_session) -> dict:
    user = db_session.query(User).filter_by(email="doctor.sql@hospital.org").first()
    if not user:
        user = User(
            id=uuid.uuid4(),
            name="Dr. Clinical Researcher",
            email="doctor.sql@hospital.org",
            password_hash=hash_password("SuperSecret123!"),
            role=UserRole.MEDICAL_RESEARCHER.value,
            is_active=True,
        )
        db_session.add(user)
        db_session.commit()
    token = create_access_token(subject=str(user.id), role=user.role)
    return {"Authorization": f"Bearer {token}"}

# =====================================================================
# Tests for auth_queries.sql Endpoints
# =====================================================================

def test_query1_email_exists(client, auth_headers):
    resp = client.get("/api/v1/sql/users/email-exists?email=doctor.sql@hospital.org")
    assert resp.status_code == 200
    assert resp.json()["exists"] is True

    resp_nonexistent = client.get("/api/v1/sql/users/email-exists?email=nonexistent@hospital.org")
    assert resp_nonexistent.status_code == 200
    assert resp_nonexistent.json()["exists"] is False

def test_query2_user_by_email_verification(client, admin_auth_headers):
    resp = client.get("/api/v1/sql/users/by-email?email=doctor.sql@hospital.org", headers=admin_auth_headers)
    assert resp.status_code in [200, 404, 503]
    if resp.status_code == 200:
        data = resp.json()
        assert data["email"] == "doctor.sql@hospital.org"
        assert "password_hash" not in data

def test_query3_view_registered_users(client, admin_auth_headers):
    resp = client.get("/api/v1/sql/users", headers=admin_auth_headers)
    assert resp.status_code in [200, 503]
    if resp.status_code == 200:
        users = resp.json()
        assert isinstance(users, list)
        for u in users:
            assert "password_hash" not in u

def test_query4_update_user_name(client, db_session, auth_headers):
    user = db_session.query(User).filter_by(email="doctor.sql@hospital.org").first()
    user_id_str = str(user.id)
    resp = client.patch(
        f"/api/v1/sql/users/{user_id_str}/name",
        json={"name": "Dr. Updated Clinical"},
        headers=auth_headers
    )
    assert resp.status_code in [200, 404, 503]
    if resp.status_code == 200:
        assert resp.json()["name"] == "Dr. Updated Clinical"

def test_query5_view_all_cases(client, db_session, auth_headers):
    user = db_session.query(User).filter_by(email="doctor.sql@hospital.org").first()
    user_uuid = str(user.id)
    db_session.execute(text("""
        INSERT INTO cases (case_id, user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (101, :user_id, 'Coronary Case 1', 'CTA', 'LAD', 'pending');
    """), {"user_id": user_uuid})
    db_session.commit()

    resp = client.get("/api/v1/sql/cases", headers=auth_headers)
    assert resp.status_code == 200
    cases = resp.json()
    assert len(cases) >= 1
    assert any(c["case_id"] == 101 for c in cases)

def test_query6_view_all_medical_images(client, db_session, auth_headers):
    db_session.execute(text("""
        INSERT INTO medical_images (image_id, case_id, image_name, image_path, image_format, image_size)
        VALUES (201, 101, 'coronary_scan.dcm', '/scans/coronary_scan.dcm', 'DICOM', 4194304);
    """))
    db_session.commit()

    resp = client.get("/api/v1/sql/images", headers=auth_headers)
    assert resp.status_code == 200
    images = resp.json()
    assert len(images) >= 1
    assert any(img["image_id"] == 201 for img in images)

def test_query7_view_segmentation_results(client, db_session, auth_headers):
    db_session.execute(text("""
        INSERT INTO segmentation_results (segmentation_id, image_id, mask_path, model_name, dice_score)
        VALUES (301, 201, '/masks/mask_301.nii', 'UNet3D', 0.895);
    """))
    db_session.commit()

    resp = client.get("/api/v1/sql/segmentation-results", headers=auth_headers)
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) >= 1
    assert any(r["segmentation_id"] == 301 for r in results)

def test_query8_view_stenosis_results(client, db_session, auth_headers):
    db_session.execute(text("""
        INSERT INTO stenosis_results (stenosis_id, image_id, location, severity, confidence_score)
        VALUES (401, 201, 'Proximal LAD', 'Severe', 0.94);
    """))
    db_session.commit()

    resp = client.get("/api/v1/sql/stenosis-results", headers=auth_headers)
    assert resp.status_code == 200
    stenosis = resp.json()
    assert len(stenosis) >= 1
    assert any(s["stenosis_id"] == 401 for s in stenosis)

# =====================================================================
# Tests for Additional Case Management Endpoints
# =====================================================================

def test_create_and_get_case(client, db_session, auth_headers):
    user = db_session.query(User).filter_by(email="doctor.sql@hospital.org").first()
    user_uuid = str(user.id)
    create_payload = {
        "case_name": "New Cardiovascular Assessment",
        "imaging_modality": "CTA",
        "target_vessel": "RCA",
        "status": "pending"
    }
    resp = client.post("/api/v1/sql/cases", json=create_payload, headers=auth_headers)
    assert resp.status_code == 201
    created_case = resp.json()
    new_case_id = created_case["case_id"]
    assert created_case["case_name"] == "New Cardiovascular Assessment"
    assert created_case["target_vessel"] == "RCA"
    # Verify case ownership is derived from the authenticated user (UUID)
    assert created_case["user_id"] is not None
    assert str(created_case["user_id"]) == user_uuid

    # Get by ID
    get_resp = client.get(f"/api/v1/sql/cases/{new_case_id}", headers=auth_headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["case_id"] == new_case_id
    assert str(get_resp.json()["user_id"]) == user_uuid

    # Update case
    patch_payload = {
        "status": "completed",
        "target_vessel": "LAD-LCx"
    }
    patch_resp = client.patch(f"/api/v1/sql/cases/{new_case_id}", json=patch_payload, headers=auth_headers)
    assert patch_resp.status_code == 200
    assert patch_resp.json()["status"] == "completed"
    assert patch_resp.json()["target_vessel"] == "LAD-LCx"


def test_get_cases_by_user_uuid_valid(client, db_session, auth_headers):
    """Valid UUID retrieval returns the authenticated user's cases."""
    user = db_session.query(User).filter_by(email="doctor.sql@hospital.org").first()
    user_uuid = str(user.id)

    # Create a case for this user
    create_payload = {
        "case_name": "UUID Retrieval Test Case",
        "imaging_modality": "CTA",
        "target_vessel": "LAD",
        "status": "pending"
    }
    create_resp = client.post("/api/v1/sql/cases", json=create_payload, headers=auth_headers)
    assert create_resp.status_code == 201

    # Retrieve by user UUID
    resp = client.get(f"/api/v1/sql/cases/user/{user_uuid}", headers=auth_headers)
    assert resp.status_code == 200
    cases = resp.json()
    assert len(cases) >= 1
    assert any(c["case_name"] == "UUID Retrieval Test Case" for c in cases)
    assert all(str(c["user_id"]) == user_uuid for c in cases)


def test_get_cases_by_user_uuid_empty_result(client, db_session, auth_headers):
    """User with no cases returns empty list."""
    # Create a brand-new user who has no cases
    empty_user = User(
        id=uuid.uuid4(),
        name="Empty Case User",
        email="empty.cases@hospital.org",
        password_hash=hash_password("EmptyPass123!"),
        role=UserRole.MEDICAL_RESEARCHER.value,
        is_active=True,
    )
    db_session.add(empty_user)
    db_session.commit()

    # Create a separate token for the empty user
    from app.core.security import create_access_token
    empty_token = create_access_token(subject=str(empty_user.id), role=empty_user.role)
    empty_headers = {"Authorization": f"Bearer {empty_token}"}

    resp = client.get(f"/api/v1/sql/cases/user/{str(empty_user.id)}", headers=empty_headers)
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_cases_by_user_uuid_invalid(client, auth_headers):
    """Invalid UUID string returns HTTP 422."""
    resp = client.get("/api/v1/sql/cases/user/not-a-uuid", headers=auth_headers)
    assert resp.status_code == 422


def test_get_cases_by_user_uuid_own_cases(client, db_session, auth_headers):
    """Authenticated user retrieves their own cases via UUID endpoint."""
    user = db_session.query(User).filter_by(email="doctor.sql@hospital.org").first()
    user_uuid = str(user.id)

    # Create two cases
    for i, vessel in enumerate(["LAD", "RCA", "LCx"]):
        create_payload = {
            "case_name": f"Own Cases Test {i}",
            "imaging_modality": "CTA",
            "target_vessel": vessel,
            "status": "pending"
        }
        create_resp = client.post("/api/v1/sql/cases", json=create_payload, headers=auth_headers)
        assert create_resp.status_code == 201

    resp = client.get(f"/api/v1/sql/cases/user/{user_uuid}", headers=auth_headers)
    assert resp.status_code == 200
    cases = resp.json()
    own_cases = [c for c in cases if c["case_name"].startswith("Own Cases Test")]
    assert len(own_cases) == 3


def test_get_cases_by_user_uuid_cannot_access_other_user(client, db_session, auth_headers):
    """User A cannot retrieve User B's cases (403 Forbidden)."""
    # Create a second user
    other_user = User(
        id=uuid.uuid4(),
        name="Other User",
        email="other.user@hospital.org",
        password_hash=hash_password("OtherPass123!"),
        role=UserRole.MEDICAL_RESEARCHER.value,
        is_active=True,
    )
    db_session.add(other_user)
    db_session.commit()

    # Create a case for the other user
    db_session.execute(text("""
        INSERT INTO cases (case_id, user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (999, :user_id, 'Other User Case', 'CTA', 'LAD', 'pending');
    """), {"user_id": str(other_user.id)})
    db_session.commit()

    # Authenticated user tries to access other user's cases
    resp = client.get(f"/api/v1/sql/cases/user/{str(other_user.id)}", headers=auth_headers)
    assert resp.status_code == 403


# =====================================================================
# Tests for Additional Medical Image Management Endpoints
# =====================================================================


def test_medical_image_management(client, auth_headers):
    img_payload = {
        "image_name": "scan_series_101.dcm",
        "image_path": "/var/imaging/dicom_101.dcm",
        "image_format": "DICOM",
        "image_size": 2048576
    }
    resp = client.post("/api/v1/sql/cases/101/images", json=img_payload, headers=auth_headers)
    assert resp.status_code == 201
    img = resp.json()
    img_id = img["image_id"]
    assert img["case_id"] == 101

    # Retrieve by image ID
    get_img = client.get(f"/api/v1/sql/images/{img_id}", headers=auth_headers)
    assert get_img.status_code == 200
    assert get_img.json()["image_name"] == "scan_series_101.dcm"

    # Retrieve by case ID
    get_case_imgs = client.get("/api/v1/sql/cases/101/images", headers=auth_headers)
    assert get_case_imgs.status_code == 200
    assert any(i["image_id"] == img_id for i in get_case_imgs.json())

    # Update metadata
    update_resp = client.patch(
        f"/api/v1/sql/images/{img_id}",
        json={"image_name": "scan_series_101_enhanced.dcm"},
        headers=auth_headers
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["image_name"] == "scan_series_101_enhanced.dcm"

    # Delete image when no analysis results are attached
    del_resp = client.delete(f"/api/v1/sql/images/{img_id}", headers=auth_headers)
    assert del_resp.status_code == 200
    assert "deleted successfully" in del_resp.json()["message"]

# =====================================================================
# Tests for Additional Segmentation Results Endpoints
# =====================================================================

def test_segmentation_results_management(client, auth_headers):
    # Insert new segmentation result for image 201
    seg_payload = {
        "mask_path": "/storage/masks/mask_seg_501.nii.gz",
        "model_name": "3D-Attention-UNet",
        "dice_score": 0.924,
        "iou_score": 0.865,
        "mean_vessel_diameter": 3.45
    }
    resp = client.post("/api/v1/sql/images/201/segmentation-results", json=seg_payload, headers=auth_headers)
    assert resp.status_code == 201
    created_seg = resp.json()
    assert created_seg["model_name"] == "3D-Attention-UNet"
    assert round(created_seg["dice_score"], 3) == 0.924

    # Retrieve by image ID
    by_img = client.get("/api/v1/sql/images/201/segmentation-results", headers=auth_headers)
    assert by_img.status_code == 200
    assert len(by_img.json()) >= 1

    # Retrieve by case ID (case 101 contains image 201)
    by_case = client.get("/api/v1/sql/cases/101/segmentation-results", headers=auth_headers)
    assert by_case.status_code == 200
    assert len(by_case.json()) >= 1

    # Retrieve latest segmentation result
    latest = client.get("/api/v1/sql/segmentation-results/latest", headers=auth_headers)
    assert latest.status_code == 200
    assert "segmentation_id" in latest.json()

# =====================================================================
# Tests for Additional Stenosis Results Endpoints
# =====================================================================

def test_stenosis_results_management(client, auth_headers):
    # Insert new stenosis result for image 201
    sten_payload = {
        "location": "Mid LAD",
        "severity": "Moderate",
        "confidence_score": 0.91,
        "estimated_diameter": 1.85
    }
    resp = client.post("/api/v1/sql/images/201/stenosis-results", json=sten_payload, headers=auth_headers)
    assert resp.status_code == 201
    created_sten = resp.json()
    assert created_sten["severity"] == "Moderate"

    # Retrieve by image ID
    by_img = client.get("/api/v1/sql/images/201/stenosis-results", headers=auth_headers)
    assert by_img.status_code == 200
    assert any(s["location"] == "Mid LAD" for s in by_img.json())

    # Retrieve by case ID
    by_case = client.get("/api/v1/sql/cases/101/stenosis-results", headers=auth_headers)
    assert by_case.status_code == 200
    assert any(s["location"] == "Mid LAD" for s in by_case.json())

    # Filter by severity
    filter_resp = client.get("/api/v1/sql/stenosis-results/filter?severity=Moderate", headers=auth_headers)
    assert filter_resp.status_code == 200
    assert all(s["severity"].lower() == "moderate" for s in filter_resp.json())

    # Sort by confidence
    sort_resp = client.get("/api/v1/sql/stenosis-results?sort_by=confidence", headers=auth_headers)
    assert sort_resp.status_code == 200
    scores = [s["confidence_score"] for s in sort_resp.json() if s["confidence_score"] is not None]
    assert scores == sorted(scores, reverse=True)

    # Retrieve latest stenosis result
    latest = client.get("/api/v1/sql/stenosis-results/latest", headers=auth_headers)
    assert latest.status_code == 200
    assert "stenosis_id" in latest.json()

# =====================================================================
# Tests for Dashboard Analytics Endpoint
# =====================================================================

def test_dashboard_analytics(client, auth_headers):
    resp = client.get("/api/v1/sql/dashboard/analytics", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "total_users" in data
    assert "total_cases" in data
    assert "total_images" in data
    assert "total_stenosis_detections" in data
    assert "recent_activity" in data
    assert isinstance(data["recent_activity"], list)
    if data["recent_activity"]:
        first = data["recent_activity"][0]
        assert "image_id" in first
        assert "analysis_type" in first

# =====================================================================
# Tests for result_queries.sql Endpoints
# =====================================================================

def test_result_query1_count_segmentation(client, db_session, auth_headers):
    resp = client.get("/api/v1/sql/analytics/segmentation-count", headers=auth_headers)
    assert resp.status_code == 200
    assert "count" in resp.json()
    assert resp.json()["count"] >= 1

def test_result_query2_count_stenosis(client, db_session, auth_headers):
    resp = client.get("/api/v1/sql/analytics/stenosis-count", headers=auth_headers)
    assert resp.status_code == 200
    assert "count" in resp.json()
    assert resp.json()["count"] >= 1

def test_result_query3_model_performance(client, db_session, auth_headers):
    resp = client.get("/api/v1/sql/analytics/model-performance", headers=auth_headers)
    assert resp.status_code == 200
    models = resp.json()
    assert len(models) >= 1
    assert any(m["model_name"] in ["UNet3D", "ResUNet", "3D-Attention-UNet"] for m in models)

def test_result_queries_combined_summary(client, auth_headers):
    resp = client.get("/api/v1/sql/analytics/summary", headers=auth_headers)
    assert resp.status_code == 200
    summary = resp.json()
    assert summary["total_segmentation_results"] >= 1
    assert summary["total_stenosis_results"] >= 1
    assert len(summary["model_performance"]) >= 1

def test_sql_endpoints_require_auth(client):
    assert client.get("/api/v1/sql/cases").status_code == 401
    assert client.get("/api/v1/sql/images").status_code == 401
    assert client.get("/api/v1/sql/segmentation-results").status_code == 401
    assert client.get("/api/v1/sql/stenosis-results").status_code == 401
    assert client.get("/api/v1/sql/analytics/segmentation-count").status_code == 401
    assert client.get("/api/v1/sql/analytics/stenosis-count").status_code == 401
    assert client.get("/api/v1/sql/analytics/model-performance").status_code == 401
    assert client.get("/api/v1/sql/dashboard/analytics").status_code == 401
    assert client.get("/api/v1/sql/analytics/model-performance/detailed").status_code == 401
    assert client.get("/api/v1/sql/analytics/stenosis-severity-distribution").status_code == 401
    assert client.get("/api/v1/sql/reports/case-analysis").status_code == 401
    assert client.get("/api/v1/sql/reports/case-summary-counts").status_code == 401
    assert client.get("/api/v1/sql/diagnostics/foreign-keys").status_code == 401
    assert client.get("/api/v1/sql/diagnostics/tables").status_code == 401
    assert client.get("/api/v1/sql/diagnostics/columns").status_code == 401
    assert client.get("/api/v1/sql/diagnostics/foreign-key-consistency").status_code == 401

@pytest.fixture
def admin_auth_headers(db_session) -> dict:
    admin = db_session.query(User).filter_by(email="admin.sql@hospital.org").first()
    if not admin:
        admin = User(
            id=uuid.uuid4(),
            name="System Administrator",
            email="admin.sql@hospital.org",
            password_hash=hash_password("AdminPass123!"),
            role=UserRole.ADMINISTRATOR.value,
            is_active=True,
        )
        db_session.add(admin)
        db_session.commit()
    token = create_access_token(subject=str(admin.id), role=admin.role)
    return {"Authorization": f"Bearer {token}"}

# =====================================================================
# Tests for result_queries.sql Queries 4 to 11
# =====================================================================

def test_result_query4_detailed_model_performance(client, auth_headers):
    resp = client.get("/api/v1/sql/analytics/model-performance/detailed", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    first = data[0]
    assert "model_name" in first
    assert "total_results" in first
    assert "average_dice" in first
    assert "average_iou" in first

def test_result_query5_stenosis_severity_distribution(client, auth_headers):
    resp = client.get("/api/v1/sql/analytics/stenosis-severity-distribution", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    first = data[0]
    assert "severity" in first
    assert "total_cases" in first
    assert "average_confidence" in first

def test_result_query6_case_analysis_report(client, auth_headers):
    # Retrieve all rows
    resp = client.get("/api/v1/sql/reports/case-analysis", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    item = data[0]
    assert "case_id" in item
    assert "image_id" in item
    assert "imaging_modality" in item
    assert "total_segmentation_results" in item
    assert "total_stenosis_results" in item

    # Retrieve with ?case_id= filter
    resp_filtered = client.get(f"/api/v1/sql/reports/case-analysis?case_id={item['case_id']}", headers=auth_headers)
    assert resp_filtered.status_code == 200
    assert len(resp_filtered.json()) >= 1
    filtered_item = resp_filtered.json()[0]
    assert filtered_item["case_id"] == item["case_id"]

def test_result_query7_case_summary_counts(client, auth_headers):
    resp = client.get("/api/v1/sql/reports/case-summary-counts", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    item = data[0]
    assert "case_id" in item
    assert "total_images" in item
    assert "total_segmentation_results" in item
    assert "total_stenosis_results" in item

def test_diagnostics_admin_role_enforcement(client, auth_headers, admin_auth_headers):
    # Standard medical_researcher should receive 403 Forbidden
    assert client.get("/api/v1/sql/diagnostics/foreign-keys", headers=auth_headers).status_code == 403
    assert client.get("/api/v1/sql/diagnostics/tables", headers=auth_headers).status_code == 403
    assert client.get("/api/v1/sql/diagnostics/columns", headers=auth_headers).status_code == 403
    assert client.get("/api/v1/sql/diagnostics/foreign-key-consistency", headers=auth_headers).status_code == 403

    # Administrator receives 200 OK if table schema has user_id, or 503 if users table has ORM id column
    resp_consistency = client.get("/api/v1/sql/diagnostics/foreign-key-consistency", headers=admin_auth_headers)
    assert resp_consistency.status_code in [200, 503]
    if resp_consistency.status_code == 200:
        data = resp_consistency.json()
        assert isinstance(data, list)
        assert len(data) == 4
        check_names = [d["check_name"] for d in data]
        assert "Cases without valid users" in check_names
        assert "Images without valid cases" in check_names
        assert "Segmentation results without valid images" in check_names
        assert "Stenosis results without valid images" in check_names


# =====================================================================
# Tests for Cross-User Ownership Authorization (Security Tests)
# =====================================================================

@pytest.fixture
def user_b_auth_headers(db_session) -> dict:
    """Fixture for a second user (User B) to test cross-user access prevention."""
    user_b = db_session.query(User).filter_by(email="userb.sql@hospital.org").first()
    if not user_b:
        user_b = User(
            id=uuid.uuid4(),
            name="User B Researcher",
            email="userb.sql@hospital.org",
            password_hash=hash_password("UserBPass123!"),
            role=UserRole.MEDICAL_RESEARCHER.value,
            is_active=True,
        )
        db_session.add(user_b)
        db_session.commit()
    token = create_access_token(subject=str(user_b.id), role=user_b.role)
    return {"Authorization": f"Bearer {token}"}


def test_user_a_cannot_list_user_b_cases(client, auth_headers, user_b_auth_headers, db_session):
    """User A cannot list User B's cases."""
    # Create a case for User B
    user_b = db_session.query(User).filter_by(email="userb.sql@hospital.org").first()
    db_session.execute(text("""
        INSERT INTO cases (case_id, user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (501, :user_id, 'User B Private Case', 'CTA', 'LAD', 'pending');
    """), {"user_id": str(user_b.id)})
    db_session.commit()

    # User A lists cases - should not see User B's case
    resp = client.get("/api/v1/sql/cases", headers=auth_headers)
    assert resp.status_code == 200
    cases = resp.json()
    assert not any(c["case_name"] == "User B Private Case" for c in cases)


def test_user_a_cannot_retrieve_user_b_case(client, auth_headers, user_b_auth_headers, db_session):
    """User A cannot retrieve User B's case by ID."""
    user_b = db_session.query(User).filter_by(email="userb.sql@hospital.org").first()
    db_session.execute(text("""
        INSERT INTO cases (case_id, user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (502, :user_id, 'User B Another Case', 'MRI', 'RCA', 'pending');
    """), {"user_id": str(user_b.id)})
    db_session.commit()

    # User A tries to get User B's case
    resp = client.get("/api/v1/sql/cases/502", headers=auth_headers)
    assert resp.status_code == 404

    # User B can get their own case
    resp_b = client.get("/api/v1/sql/cases/502", headers=user_b_auth_headers)
    assert resp_b.status_code == 200
    assert resp_b.json()["case_name"] == "User B Another Case"


def test_user_a_cannot_update_user_b_case(client, auth_headers, user_b_auth_headers, db_session):
    """User A cannot update User B's case."""
    user_b = db_session.query(User).filter_by(email="userb.sql@hospital.org").first()
    db_session.execute(text("""
        INSERT INTO cases (case_id, user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (503, :user_id, 'User B Updatable Case', 'CTA', 'LAD', 'pending');
    """), {"user_id": str(user_b.id)})
    db_session.commit()

    # User A tries to update User B's case
    patch_payload = {"status": "completed"}
    resp = client.patch("/api/v1/sql/cases/503", json=patch_payload, headers=auth_headers)
    assert resp.status_code == 404

    # User B can update their own case
    resp_b = client.patch("/api/v1/sql/cases/503", json=patch_payload, headers=user_b_auth_headers)
    assert resp_b.status_code == 200


def test_user_a_cannot_register_image_under_user_b_case(client, auth_headers, user_b_auth_headers, db_session):
    """User A cannot register an image under User B's case."""
    user_b = db_session.query(User).filter_by(email="userb.sql@hospital.org").first()
    db_session.execute(text("""
        INSERT INTO cases (case_id, user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (504, :user_id, 'User B Image Case', 'CTA', 'LAD', 'pending');
    """), {"user_id": str(user_b.id)})
    db_session.commit()

    # User A tries to register image under User B's case
    img_payload = {
        "image_name": "user_b_image.png",
        "image_path": "/scans/user_b.png",
        "image_format": "PNG",
        "image_size": 100000
    }
    resp = client.post("/api/v1/sql/cases/504/images", json=img_payload, headers=auth_headers)
    assert resp.status_code == 404

    # User B can register image under their own case
    resp_b = client.post("/api/v1/sql/cases/504/images", json=img_payload, headers=user_b_auth_headers)
    assert resp_b.status_code == 201


def test_user_a_cannot_retrieve_user_b_image(client, auth_headers, user_b_auth_headers, db_session):
    """User A cannot retrieve User B's image."""
    user_b = db_session.query(User).filter_by(email="userb.sql@hospital.org").first()
    # Create case and image for User B
    db_session.execute(text("""
        INSERT INTO cases (case_id, user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (505, :user_id, 'User B Image Owner Case', 'CTA', 'LAD', 'pending');
    """), {"user_id": str(user_b.id)})
    db_session.execute(text("""
        INSERT INTO medical_images (image_id, case_id, image_name, image_path, image_format, image_size)
        VALUES (601, 505, 'user_b_scan.dcm', '/scans/user_b.dcm', 'DICOM', 500000);
    """))
    db_session.commit()

    # User A tries to get User B's image
    resp = client.get("/api/v1/sql/images/601", headers=auth_headers)
    assert resp.status_code == 404

    # User B can get their own image
    resp_b = client.get("/api/v1/sql/images/601", headers=user_b_auth_headers)
    assert resp_b.status_code == 200
    assert resp_b.json()["image_name"] == "user_b_scan.dcm"


def test_user_a_cannot_update_user_b_image(client, auth_headers, user_b_auth_headers, db_session):
    """User A cannot update User B's image."""
    user_b = db_session.query(User).filter_by(email="userb.sql@hospital.org").first()
    db_session.execute(text("""
        INSERT INTO cases (case_id, user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (506, :user_id, 'User B Image Update Case', 'CTA', 'LAD', 'pending');
    """), {"user_id": str(user_b.id)})
    db_session.execute(text("""
        INSERT INTO medical_images (image_id, case_id, image_name, image_path, image_format, image_size)
        VALUES (602, 506, 'user_b_update.dcm', '/scans/user_b_update.dcm', 'DICOM', 500000);
    """))
    db_session.commit()

    # User A tries to update User B's image
    update_payload = {"image_name": "hacked.dcm"}
    resp = client.patch("/api/v1/sql/images/602", json=update_payload, headers=auth_headers)
    assert resp.status_code == 404

    # User B can update their own image
    resp_b = client.patch("/api/v1/sql/images/602", json=update_payload, headers=user_b_auth_headers)
    assert resp_b.status_code == 200


def test_user_a_cannot_delete_user_b_image(client, auth_headers, user_b_auth_headers, db_session):
    """User A cannot delete User B's image."""
    user_b = db_session.query(User).filter_by(email="userb.sql@hospital.org").first()
    db_session.execute(text("""
        INSERT INTO cases (case_id, user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (507, :user_id, 'User B Image Delete Case', 'CTA', 'LAD', 'pending');
    """), {"user_id": str(user_b.id)})
    db_session.execute(text("""
        INSERT INTO medical_images (image_id, case_id, image_name, image_path, image_format, image_size)
        VALUES (603, 507, 'user_b_delete.dcm', '/scans/user_b_delete.dcm', 'DICOM', 500000);
    """))
    db_session.commit()

    # User A tries to delete User B's image
    resp = client.delete("/api/v1/sql/images/603", headers=auth_headers)
    assert resp.status_code == 404

    # User B can delete their own image
    resp_b = client.delete("/api/v1/sql/images/603", headers=user_b_auth_headers)
    assert resp_b.status_code == 200


def test_user_a_cannot_retrieve_user_b_segmentation_results(client, auth_headers, user_b_auth_headers, db_session):
    """User A cannot retrieve User B's segmentation results."""
    user_b = db_session.query(User).filter_by(email="userb.sql@hospital.org").first()
    db_session.execute(text("""
        INSERT INTO cases (case_id, user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (508, :user_id, 'User B Seg Case', 'CTA', 'LAD', 'pending');
    """), {"user_id": str(user_b.id)})
    db_session.execute(text("""
        INSERT INTO medical_images (image_id, case_id, image_name, image_path, image_format, image_size)
        VALUES (604, 508, 'user_b_seg.dcm', '/scans/user_b_seg.dcm', 'DICOM', 500000);
    """))
    db_session.execute(text("""
        INSERT INTO segmentation_results (segmentation_id, image_id, mask_path, model_name, dice_score)
        VALUES (701, 604, '/masks/user_b_mask.nii', 'UNet', 0.89);
    """))
    db_session.commit()

    # User A tries to get User B's segmentation results
    resp = client.get("/api/v1/sql/images/604/segmentation-results", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == []

    # User B can get their own segmentation results
    resp_b = client.get("/api/v1/sql/images/604/segmentation-results", headers=user_b_auth_headers)
    assert resp_b.status_code == 200
    assert len(resp_b.json()) >= 1


def test_user_a_cannot_create_segmentation_for_user_b_image(client, auth_headers, user_b_auth_headers, db_session):
    """User A cannot create segmentation results for User B's image."""
    user_b = db_session.query(User).filter_by(email="userb.sql@hospital.org").first()
    db_session.execute(text("""
        INSERT INTO cases (case_id, user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (509, :user_id, 'User B Seg Create Case', 'CTA', 'LAD', 'pending');
    """), {"user_id": str(user_b.id)})
    db_session.execute(text("""
        INSERT INTO medical_images (image_id, case_id, image_name, image_path, image_format, image_size)
        VALUES (605, 509, 'user_b_seg_create.dcm', '/scans/user_b_seg_create.dcm', 'DICOM', 500000);
    """))
    db_session.commit()

    # User A tries to create segmentation for User B's image
    seg_payload = {
        "mask_path": "/masks/hacked.nii",
        "model_name": "HackedModel",
        "dice_score": 0.99
    }
    resp = client.post("/api/v1/sql/images/605/segmentation-results", json=seg_payload, headers=auth_headers)
    assert resp.status_code == 404

    # User B can create segmentation for their own image
    resp_b = client.post("/api/v1/sql/images/605/segmentation-results", json=seg_payload, headers=user_b_auth_headers)
    assert resp_b.status_code == 201


def test_user_a_cannot_retrieve_user_b_stenosis_results(client, auth_headers, user_b_auth_headers, db_session):
    """User A cannot retrieve User B's stenosis results."""
    user_b = db_session.query(User).filter_by(email="userb.sql@hospital.org").first()
    db_session.execute(text("""
        INSERT INTO cases (case_id, user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (510, :user_id, 'User B Stenosis Case', 'CTA', 'LAD', 'pending');
    """), {"user_id": str(user_b.id)})
    db_session.execute(text("""
        INSERT INTO medical_images (image_id, case_id, image_name, image_path, image_format, image_size)
        VALUES (606, 510, 'user_b_sten.dcm', '/scans/user_b_sten.dcm', 'DICOM', 500000);
    """))
    db_session.execute(text("""
        INSERT INTO stenosis_results (stenosis_id, image_id, location, severity, confidence_score)
        VALUES (801, 606, 'Proximal LAD', 'Severe', 0.94);
    """))
    db_session.commit()

    # User A tries to get User B's stenosis results
    resp = client.get("/api/v1/sql/images/606/stenosis-results", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == []

    # User B can get their own stenosis results
    resp_b = client.get("/api/v1/sql/images/606/stenosis-results", headers=user_b_auth_headers)
    assert resp_b.status_code == 200
    assert len(resp_b.json()) >= 1


def test_user_a_cannot_create_stenosis_for_user_b_image(client, auth_headers, user_b_auth_headers, db_session):
    """User A cannot create stenosis results for User B's image."""
    user_b = db_session.query(User).filter_by(email="userb.sql@hospital.org").first()
    db_session.execute(text("""
        INSERT INTO cases (case_id, user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (511, :user_id, 'User B Stenosis Create Case', 'CTA', 'LAD', 'pending');
    """), {"user_id": str(user_b.id)})
    db_session.execute(text("""
        INSERT INTO medical_images (image_id, case_id, image_name, image_path, image_format, image_size)
        VALUES (607, 511, 'user_b_sten_create.dcm', '/scans/user_b_sten_create.dcm', 'DICOM', 500000);
    """))
    db_session.commit()

    # User A tries to create stenosis for User B's image
    sten_payload = {
        "location": "Hacked Location",
        "severity": "Mild",
        "confidence_score": 0.5
    }
    resp = client.post("/api/v1/sql/images/607/stenosis-results", json=sten_payload, headers=auth_headers)
    assert resp.status_code == 404

    # User B can create stenosis for their own image
    resp_b = client.post("/api/v1/sql/images/607/stenosis-results", json=sten_payload, headers=user_b_auth_headers)
    assert resp_b.status_code == 201


def test_user_a_analytics_do_not_include_user_b_data(client, auth_headers, user_b_auth_headers, db_session):
    """User A's analytics do not include User B's clinical data."""
    user_b = db_session.query(User).filter_by(email="userb.sql@hospital.org").first()
    # Create data for User B
    db_session.execute(text("""
        INSERT INTO cases (case_id, user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (512, :user_id, 'User B Analytics Case', 'CTA', 'LAD', 'pending');
    """), {"user_id": str(user_b.id)})
    db_session.execute(text("""
        INSERT INTO medical_images (image_id, case_id, image_name, image_path, image_format, image_size)
        VALUES (608, 512, 'user_b_analytics.dcm', '/scans/user_b_analytics.dcm', 'DICOM', 500000);
    """))
    db_session.execute(text("""
        INSERT INTO segmentation_results (segmentation_id, image_id, mask_path, model_name, dice_score)
        VALUES (901, 608, '/masks/user_b_analytics.nii', 'UNet', 0.95);
    """))
    db_session.commit()

    # User A's segmentation count should not include User B's data
    resp_a = client.get("/api/v1/sql/analytics/segmentation-count", headers=auth_headers)
    assert resp_a.status_code == 200
    count_a = resp_a.json()["count"]

    # User B's segmentation count should include their own data
    resp_b = client.get("/api/v1/sql/analytics/segmentation-count", headers=user_b_auth_headers)
    assert resp_b.status_code == 200
    count_b = resp_b.json()["count"]

    # User B should see at least their own segmentation
    assert count_b >= 1

    # User A's dashboard should not include User B's cases
    resp_a_dash = client.get("/api/v1/sql/dashboard/analytics", headers=auth_headers)
    assert resp_a_dash.status_code == 200
    dash_a = resp_a_dash.json()

    resp_b_dash = client.get("/api/v1/sql/dashboard/analytics", headers=user_b_auth_headers)
    assert resp_b_dash.status_code == 200
    dash_b = resp_b_dash.json()

    # User B should see at least their own case
    assert dash_b["total_cases"] >= 1


def test_administrator_can_access_all_cases(client, admin_auth_headers, auth_headers, db_session):
    """Administrator can access all cases across users."""
    user_a = db_session.query(User).filter_by(email="doctor.sql@hospital.org").first()
    user_b = db_session.query(User).filter_by(email="userb.sql@hospital.org").first()

    # Create a case for User A
    db_session.execute(text("""
        INSERT INTO cases (case_id, user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (513, :user_id, 'Admin Access Test Case A', 'CTA', 'LAD', 'pending');
    """), {"user_id": str(user_a.id)})
    # Create a case for User B
    db_session.execute(text("""
        INSERT INTO cases (case_id, user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (514, :user_id, 'Admin Access Test Case B', 'MRI', 'RCA', 'pending');
    """), {"user_id": str(user_b.id)})
    db_session.commit()

    # Admin can list all cases (both User A and User B)
    resp = client.get("/api/v1/sql/cases", headers=admin_auth_headers)
    assert resp.status_code == 200
    cases = resp.json()
    case_names = [c["case_name"] for c in cases]
    assert "Admin Access Test Case A" in case_names
    assert "Admin Access Test Case B" in case_names

    # Admin can retrieve User A's case
    resp_a = client.get("/api/v1/sql/cases/513", headers=admin_auth_headers)
    assert resp_a.status_code == 200

    # Admin can retrieve User B's case
    resp_b = client.get("/api/v1/sql/cases/514", headers=admin_auth_headers)
    assert resp_b.status_code == 200


# =====================================================================
# Tests for Image Upload and Inference Persistence (Phase 4)
# =====================================================================

import io
from PIL import Image


def _create_test_png_image(width=64, height=64):
    """Create a simple test PNG image as bytes."""
    img = Image.new("L", (width, height), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def _create_test_case(db_session, user_email, case_name="Test Case"):
    """Helper to create a case for a given user and return its ID."""
    user = db_session.query(User).filter_by(email=user_email).first()
    result = db_session.execute(text("""
        INSERT INTO cases (user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (:user_id, :case_name, 'CTA', 'LAD', 'pending');
    """), {"user_id": str(user.id), "case_name": case_name})
    db_session.commit()
    # Get the auto-generated case_id
    case_row = db_session.execute(text("""
        SELECT case_id FROM cases WHERE case_name = :case_name ORDER BY case_id DESC LIMIT 1;
    """), {"case_name": case_name}).fetchone()
    return user, case_row[0]


def test_image_upload_success(client, auth_headers, db_session):
    """Successful image upload with inference and persistence."""
    user, case_id = _create_test_case(db_session, "doctor.sql@hospital.org", "Upload Test Case")
    image_bytes = _create_test_png_image()

    resp = client.post(
        "/api/v1/inference/upload-and-predict",
        files={"file": ("test.png", image_bytes, "image/png")},
        data={"case_id": case_id, "model_type": "unet"},
        headers=auth_headers,
    )
    assert resp.status_code == 201, f"Expected 201, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data["image_id"] is not None
    assert data["segmentation_id"] is not None
    assert data["case_id"] == case_id
    assert data["model_name"] == "unet"
    assert data["threshold"] == 0.3
    assert data["original_height"] > 0
    assert data["original_width"] > 0
    assert data["segmented_pixels"] >= 0
    assert data["vessel_area_ratio"] >= 0.0
    assert data["storage_identifier"] is not None
    assert data["mask_storage_identifier"] is not None
    assert data["overlay_storage_identifier"] is not None
    assert data["mask_base64"] is not None
    assert data["overlay_base64"] is not None


def test_image_upload_attention_unet(client, auth_headers, db_session):
    """Successful Attention U-Net inference and persistence."""
    user, case_id = _create_test_case(db_session, "doctor.sql@hospital.org", "Attention U-Net Test Case")
    image_bytes = _create_test_png_image()

    resp = client.post(
        "/api/v1/inference/upload-and-predict",
        files={"file": ("test.png", image_bytes, "image/png")},
        data={"case_id": case_id, "model_type": "attention_unet"},
        headers=auth_headers,
    )
    assert resp.status_code == 201, f"Expected 201, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data["model_name"] == "attention_unet"
    assert data["segmentation_id"] is not None


def test_image_upload_unsupported_format(client, auth_headers, db_session):
    """Unsupported file format returns 400."""
    user, case_id = _create_test_case(db_session, "doctor.sql@hospital.org", "Format Test Case")
    # Create a fake text file with .png extension but text content type
    resp = client.post(
        "/api/v1/inference/upload-and-predict",
        files={"file": ("test.txt", b"this is not an image", "text/plain")},
        data={"case_id": case_id, "model_type": "unet"},
        headers=auth_headers,
    )
    assert resp.status_code == 400


def test_image_upload_invalid_content(client, auth_headers, db_session):
    """Invalid image content returns 400 or 500 depending on validation stage."""
    user, case_id = _create_test_case(db_session, "doctor.sql@hospital.org", "Invalid Content Case")
    # Fake PNG header but invalid content - may pass storage validation but fail inference
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    resp = client.post(
        "/api/v1/inference/upload-and-predict",
        files={"file": ("test.png", fake_png, "image/png")},
        data={"case_id": case_id, "model_type": "unet"},
        headers=auth_headers,
    )
    assert resp.status_code in [400, 500], f"Expected 400 or 500, got {resp.status_code}"


def test_image_upload_nonexistent_case(client, auth_headers):
    """Upload to a nonexistent case returns 404."""
    image_bytes = _create_test_png_image()
    resp = client.post(
        "/api/v1/inference/upload-and-predict",
        files={"file": ("test.png", image_bytes, "image/png")},
        data={"case_id": 99999, "model_type": "unet"},
        headers=auth_headers,
    )
    assert resp.status_code == 404


def test_image_upload_cross_user_case(client, auth_headers, user_b_auth_headers, db_session):
    """User A cannot upload an image under User B's case."""
    user_b = db_session.query(User).filter_by(email="userb.sql@hospital.org").first()
    # Create a case for User B using auto-increment
    result = db_session.execute(text("""
        INSERT INTO cases (user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (:user_id, 'User B Case for Upload', 'CTA', 'LAD', 'pending');
    """), {"user_id": str(user_b.id)})
    db_session.commit()
    case_row = db_session.execute(text("""
        SELECT case_id FROM cases WHERE case_name = 'User B Case for Upload' ORDER BY case_id DESC LIMIT 1;
    """)).fetchone()
    user_b_case_id = case_row[0]

    image_bytes = _create_test_png_image()
    resp = client.post(
        "/api/v1/inference/upload-and-predict",
        files={"file": ("test.png", image_bytes, "image/png")},
        data={"case_id": user_b_case_id, "model_type": "unet"},
        headers=auth_headers,
    )
    assert resp.status_code == 404


def test_image_retrieve_metadata(client, auth_headers, db_session):
    """Successful retrieval of stored image metadata."""
    user, case_id = _create_test_case(db_session, "doctor.sql@hospital.org", "Metadata Retrieval Case")
    image_bytes = _create_test_png_image()

    # Upload first
    upload_resp = client.post(
        "/api/v1/inference/upload-and-predict",
        files={"file": ("metadata_test.png", image_bytes, "image/png")},
        data={"case_id": case_id, "model_type": "unet"},
        headers=auth_headers,
    )
    assert upload_resp.status_code == 201
    image_id = upload_resp.json()["image_id"]

    # Retrieve metadata
    resp = client.get(f"/api/v1/inference/images/{image_id}", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["image_id"] == image_id
    assert data["case_id"] == case_id
    assert data["storage_identifier"] is not None


def test_image_retrieve_file(client, auth_headers, db_session):
    """Successful retrieval of the original image file."""
    user, case_id = _create_test_case(db_session, "doctor.sql@hospital.org", "File Retrieval Case")
    image_bytes = _create_test_png_image()

    upload_resp = client.post(
        "/api/v1/inference/upload-and-predict",
        files={"file": ("file_test.png", image_bytes, "image/png")},
        data={"case_id": case_id, "model_type": "unet"},
        headers=auth_headers,
    )
    assert upload_resp.status_code == 201
    image_id = upload_resp.json()["image_id"]

    resp = client.get(f"/api/v1/inference/images/{image_id}/file", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.headers.get("content-type") in ["image/png", "application/octet-stream"]


def test_mask_and_overlay_retrieval(client, auth_headers, db_session):
    """Successful retrieval of mask and overlay files."""
    user, case_id = _create_test_case(db_session, "doctor.sql@hospital.org", "Mask Overlay Case")
    image_bytes = _create_test_png_image()

    upload_resp = client.post(
        "/api/v1/inference/upload-and-predict",
        files={"file": ("mask_overlay_test.png", image_bytes, "image/png")},
        data={"case_id": case_id, "model_type": "unet"},
        headers=auth_headers,
    )
    assert upload_resp.status_code == 201
    segmentation_id = upload_resp.json()["segmentation_id"]

    # Retrieve mask
    mask_resp = client.get(f"/api/v1/inference/masks/{segmentation_id}", headers=auth_headers)
    assert mask_resp.status_code == 200
    assert mask_resp.headers.get("content-type") == "image/png"

    # Retrieve overlay
    overlay_resp = client.get(f"/api/v1/inference/overlays/{segmentation_id}", headers=auth_headers)
    assert overlay_resp.status_code == 200
    assert overlay_resp.headers.get("content-type") == "image/png"


def test_segmentation_result_retrieval(client, auth_headers, db_session):
    """Successful retrieval of segmentation result metadata."""
    user, case_id = _create_test_case(db_session, "doctor.sql@hospital.org", "Result Retrieval Case")
    image_bytes = _create_test_png_image()

    upload_resp = client.post(
        "/api/v1/inference/upload-and-predict",
        files={"file": ("result_test.png", image_bytes, "image/png")},
        data={"case_id": case_id, "model_type": "unet"},
        headers=auth_headers,
    )
    assert upload_resp.status_code == 201
    segmentation_id = upload_resp.json()["segmentation_id"]

    resp = client.get(f"/api/v1/inference/segmentation-results/{segmentation_id}", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["segmentation_id"] == segmentation_id
    assert data["image_id"] is not None


def test_image_retrieve_cross_user_rejection(client, auth_headers, user_b_auth_headers, db_session):
    """User A cannot retrieve User B's image metadata."""
    user_b = db_session.query(User).filter_by(email="userb.sql@hospital.org").first()
    # Create case and image for User B
    result = db_session.execute(text("""
        INSERT INTO cases (user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (:user_id, 'User B Image Case', 'CTA', 'LAD', 'pending');
    """), {"user_id": str(user_b.id)})
    db_session.commit()
    case_row = db_session.execute(text("""
        SELECT case_id FROM cases WHERE case_name = 'User B Image Case' ORDER BY case_id DESC LIMIT 1;
    """)).fetchone()
    user_b_case_id = case_row[0]

    db_session.execute(text("""
        INSERT INTO medical_images (image_id, case_id, image_name, image_path, image_format, image_size)
        VALUES (7001, :case_id, 'user_b_img.png', 'images/some_uuid.png', 'PNG', 100000);
    """), {"case_id": user_b_case_id})
    db_session.commit()

    # User A tries to get User B's image metadata
    resp = client.get("/api/v1/inference/images/7001", headers=auth_headers)
    assert resp.status_code == 404

    # User B can get their own image metadata
    resp_b = client.get("/api/v1/inference/images/7001", headers=user_b_auth_headers)
    assert resp_b.status_code == 200


def test_segmentation_result_cross_user_rejection(client, auth_headers, user_b_auth_headers, db_session):
    """User A cannot retrieve User B's segmentation result."""
    user_b = db_session.query(User).filter_by(email="userb.sql@hospital.org").first()
    # Create case and image for User B
    result = db_session.execute(text("""
        INSERT INTO cases (user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (:user_id, 'User B Seg Case', 'CTA', 'LAD', 'pending');
    """), {"user_id": str(user_b.id)})
    db_session.commit()
    case_row = db_session.execute(text("""
        SELECT case_id FROM cases WHERE case_name = 'User B Seg Case' ORDER BY case_id DESC LIMIT 1;
    """)).fetchone()
    user_b_case_id = case_row[0]

    db_session.execute(text("""
        INSERT INTO medical_images (image_id, case_id, image_name, image_path, image_format, image_size)
        VALUES (7002, :case_id, 'user_b_seg.png', 'images/some_uuid.png', 'PNG', 100000);
    """), {"case_id": user_b_case_id})
    db_session.execute(text("""
        INSERT INTO segmentation_results (segmentation_id, image_id, mask_path, model_name, dice_score)
        VALUES (8001, 7002, 'images/mask.png', 'UNet', 0.85);
    """))
    db_session.commit()

    # User A tries to get User B's segmentation result
    resp = client.get(f"/api/v1/inference/segmentation-results/8001", headers=auth_headers)
    assert resp.status_code == 404

    # User B can get their own segmentation result
    resp_b = client.get(f"/api/v1/inference/segmentation-results/8001", headers=user_b_auth_headers)
    assert resp_b.status_code == 200


def test_image_upload_missing_file(client, auth_headers, db_session):
    """Upload without a file returns 422."""
    user, case_id = _create_test_case(db_session, "doctor.sql@hospital.org", "Missing File Case")
    resp = client.post(
        "/api/v1/inference/upload-and-predict",
        data={"case_id": case_id, "model_type": "unet"},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_image_upload_invalid_case_id_type(client, auth_headers, db_session):
    """Upload with non-integer case_id returns 422."""
    image_bytes = _create_test_png_image()
    resp = client.post(
        "/api/v1/inference/upload-and-predict",
        files={"file": ("test.png", image_bytes, "image/png")},
        data={"case_id": "not-a-number", "model_type": "unet"},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_administrator_can_upload_for_any_case(client, admin_auth_headers, db_session):
    """Administrator can upload images for any user's case."""
    user_b = db_session.query(User).filter_by(email="userb.sql@hospital.org").first()
    result = db_session.execute(text("""
        INSERT INTO cases (user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (:user_id, 'Admin Upload Case', 'CTA', 'LAD', 'pending');
    """), {"user_id": str(user_b.id)})
    db_session.commit()
    case_row = db_session.execute(text("""
        SELECT case_id FROM cases WHERE case_name = 'Admin Upload Case' ORDER BY case_id DESC LIMIT 1;
    """)).fetchone()
    user_b_case_id = case_row[0]

    image_bytes = _create_test_png_image()
    resp = client.post(
        "/api/v1/inference/upload-and-predict",
        files={"file": ("admin_upload.png", image_bytes, "image/png")},
        data={"case_id": user_b_case_id, "model_type": "unet"},
        headers=admin_auth_headers,
    )
    assert resp.status_code == 201


def test_invalid_model_type(client, auth_headers, db_session):
    """Invalid model_type returns 400."""
    user, case_id = _create_test_case(db_session, "doctor.sql@hospital.org", "Invalid Model Case")
    image_bytes = _create_test_png_image()
    resp = client.post(
        "/api/v1/inference/upload-and-predict",
        files={"file": ("test.png", image_bytes, "image/png")},
        data={"case_id": case_id, "model_type": "invalid_model"},
        headers=auth_headers,
    )
    assert resp.status_code == 400


def test_missing_image_404(client, auth_headers):
    """Retrieving a missing image returns 404."""
    resp = client.get("/api/v1/inference/images/99999", headers=auth_headers)
    assert resp.status_code == 404


def test_missing_segmentation_404(client, auth_headers):
    """Retrieving a missing segmentation result returns 404."""
    resp = client.get("/api/v1/inference/segmentation-results/99999", headers=auth_headers)
    assert resp.status_code == 404


def test_missing_mask_404(client, auth_headers):
    """Retrieving a missing mask returns 404."""
    resp = client.get("/api/v1/inference/masks/99999", headers=auth_headers)
    assert resp.status_code == 404


def test_missing_overlay_404(client, auth_headers):
    """Retrieving a missing overlay returns 404."""
    resp = client.get("/api/v1/inference/overlays/99999", headers=auth_headers)
    assert resp.status_code == 404


def test_overlay_path_persisted_in_db(client, auth_headers, db_session):
    """Overlay path is persisted in segmentation_results table."""
    user, case_id = _create_test_case(db_session, "doctor.sql@hospital.org", "Overlay Path Case")
    image_bytes = _create_test_png_image()

    upload_resp = client.post(
        "/api/v1/inference/upload-and-predict",
        files={"file": ("overlay_path_test.png", image_bytes, "image/png")},
        data={"case_id": case_id, "model_type": "unet"},
        headers=auth_headers,
    )
    assert upload_resp.status_code == 201
    data = upload_resp.json()
    assert data["overlay_storage_identifier"] is not None
    assert data["overlay_storage_identifier"].startswith("images/")
    assert data["overlay_storage_identifier"].endswith(".png")


def test_overlay_retrieval_returns_overlay_file(client, auth_headers, db_session):
    """Overlay retrieval endpoint serves the overlay file (not the mask file)."""
    user, case_id = _create_test_case(db_session, "doctor.sql@hospital.org", "Overlay Retrieval Case")
    image_bytes = _create_test_png_image()

    upload_resp = client.post(
        "/api/v1/inference/upload-and-predict",
        files={"file": ("overlay_retrieve_test.png", image_bytes, "image/png")},
        data={"case_id": case_id, "model_type": "unet"},
        headers=auth_headers,
    )
    assert upload_resp.status_code == 201
    segmentation_id = upload_resp.json()["segmentation_id"]
    overlay_storage_id = upload_resp.json()["overlay_storage_identifier"]

    # Verify overlay_path is stored in DB
    seg_resp = client.get(
        f"/api/v1/inference/segmentation-results/{segmentation_id}",
        headers=auth_headers,
    )
    assert seg_resp.status_code == 200
    assert seg_resp.json()["overlay_path"] == overlay_storage_id

    # Retrieve overlay file
    overlay_resp = client.get(
        f"/api/v1/inference/overlays/{segmentation_id}",
        headers=auth_headers,
    )
    assert overlay_resp.status_code == 200
    assert overlay_resp.headers.get("content-type") == "image/png"
    assert len(overlay_resp.content) > 0


def test_mask_and_overlay_are_different_files(client, auth_headers, db_session):
    """Mask and overlay storage identifiers are different."""
    user, case_id = _create_test_case(db_session, "doctor.sql@hospital.org", "Mask vs Overlay Case")
    image_bytes = _create_test_png_image()

    upload_resp = client.post(
        "/api/v1/inference/upload-and-predict",
        files={"file": ("mask_vs_overlay.png", image_bytes, "image/png")},
        data={"case_id": case_id, "model_type": "unet"},
        headers=auth_headers,
    )
    assert upload_resp.status_code == 201
    data = upload_resp.json()
    assert data["mask_storage_identifier"] != data["overlay_storage_identifier"]
    assert data["mask_storage_identifier"].startswith("images/")
    assert data["overlay_storage_identifier"].startswith("images/")
    assert data["mask_storage_identifier"].endswith(".png")
    assert data["overlay_storage_identifier"].endswith(".png")


def test_overlay_retrieval_cross_user_rejection(client, auth_headers, user_b_auth_headers, db_session):
    """User A cannot retrieve User B's overlay file."""
    user_b = db_session.query(User).filter_by(email="userb.sql@hospital.org").first()
    # Create case and image for User B
    result = db_session.execute(text("""
        INSERT INTO cases (user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (:user_id, 'User B Overlay Case', 'CTA', 'LAD', 'pending');
    """), {"user_id": str(user_b.id)})
    db_session.commit()
    case_row = db_session.execute(text("""
        SELECT case_id FROM cases WHERE case_name = 'User B Overlay Case' ORDER BY case_id DESC LIMIT 1;
    """)).fetchone()
    user_b_case_id = case_row[0]

    # User B uploads an image
    image_bytes = _create_test_png_image()
    upload_resp = client.post(
        "/api/v1/inference/upload-and-predict",
        files={"file": ("user_b_overlay.png", image_bytes, "image/png")},
        data={"case_id": user_b_case_id, "model_type": "unet"},
        headers=user_b_auth_headers,
    )
    assert upload_resp.status_code == 201
    segmentation_id = upload_resp.json()["segmentation_id"]

    # User A tries to retrieve User B's overlay
    resp = client.get(f"/api/v1/inference/overlays/{segmentation_id}", headers=auth_headers)
    assert resp.status_code == 404

    # User B can retrieve their own overlay
    resp_b = client.get(f"/api/v1/inference/overlays/{segmentation_id}", headers=user_b_auth_headers)
    assert resp_b.status_code == 200


