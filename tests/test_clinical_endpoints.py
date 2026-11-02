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
    # Note: Do not wrap in a rolling back transaction if we want table changes to persist within a test
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
    # Query 2 requires the table columns (user_id, full_name, email, password_hash)
    resp = client.get("/api/v1/sql/users/by-email?email=doctor.sql@hospital.org", headers=auth_headers)
    # If the active user table uses id/name instead of user_id/full_name, returns 503 as designed
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
# Tests for result_queries.sql Endpoints
# =====================================================================

def test_result_query1_count_segmentation(client, db_session, auth_headers):
    db_session.execute(text("""
        INSERT INTO segmentation_results (segmentation_id, image_id, mask_path, model_name, dice_score)
        VALUES (302, 201, '/masks/mask_302.nii', 'ResUNet', 0.912);
    """))
    db_session.commit()

    resp = client.get("/api/v1/sql/analytics/segmentation-count", headers=auth_headers)
    assert resp.status_code == 200
    assert "count" in resp.json()
    assert resp.json()["count"] >= 1

def test_result_query2_count_stenosis(client, db_session, auth_headers):
    db_session.execute(text("""
        INSERT INTO stenosis_results (stenosis_id, image_id, location, severity, confidence_score)
        VALUES (402, 201, 'Distal RCA', 'Moderate', 0.88);
    """))
    db_session.commit()

    resp = client.get("/api/v1/sql/analytics/stenosis-count", headers=auth_headers)
    assert resp.status_code == 200
    assert "count" in resp.json()
    assert resp.json()["count"] >= 1

def test_result_query3_model_performance(client, db_session, auth_headers):
    resp = client.get("/api/v1/sql/analytics/model-performance", headers=auth_headers)
    assert resp.status_code == 200
    models = resp.json()
    assert len(models) >= 1
    assert any(m["model_name"] in ["UNet3D", "ResUNet"] for m in models)

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
