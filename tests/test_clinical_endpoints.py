import os
import sys
import uuid
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

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
                user_id INTEGER,
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

def test_query2_user_by_email_verification(client, auth_headers):
    resp = client.get("/api/v1/sql/users/by-email?email=doctor.sql@hospital.org", headers=auth_headers)
    assert resp.status_code in [200, 404, 503]
    if resp.status_code == 200:
        data = resp.json()
        assert data["email"] == "doctor.sql@hospital.org"
        assert "password_hash" not in data

def test_query3_view_registered_users(client, auth_headers):
    resp = client.get("/api/v1/sql/users", headers=auth_headers)
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
        json={"full_name": "Dr. Updated Clinical"},
        headers=auth_headers
    )
    assert resp.status_code in [200, 404, 503]

def test_query5_view_all_cases(client, db_session, auth_headers):
    db_session.execute(text("""
        INSERT INTO cases (case_id, user_id, case_name, imaging_modality, target_vessel, status)
        VALUES (101, 1, 'Coronary Case 1', 'CTA', 'LAD', 'pending');
    """))
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

def test_create_and_get_case(client, auth_headers):
    create_payload = {
        "user_id": 5,
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

    # Get by ID
    get_resp = client.get(f"/api/v1/sql/cases/{new_case_id}", headers=auth_headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["case_id"] == new_case_id

    # Get by user ID
    user_cases_resp = client.get("/api/v1/sql/cases/user/5", headers=auth_headers)
    assert user_cases_resp.status_code == 200
    assert any(c["case_id"] == new_case_id for c in user_cases_resp.json())

    # Update case
    patch_payload = {
        "status": "completed",
        "target_vessel": "LAD-LCx"
    }
    patch_resp = client.patch(f"/api/v1/sql/cases/{new_case_id}", json=patch_payload, headers=auth_headers)
    assert patch_resp.status_code == 200
    assert patch_resp.json()["status"] == "completed"
    assert patch_resp.json()["target_vessel"] == "LAD-LCx"

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


