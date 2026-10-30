import uuid
import jwt
from datetime import datetime, timedelta, timezone
import pytest
from app.core.config import settings
from app.core.security import create_access_token


def test_get_profile_authenticated(client, test_user, user_access_token):
    response = client.get(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {user_access_token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(test_user.id)
    assert data["name"] == test_user.name
    assert data["email"] == test_user.email
    assert data["role"] == "medical_researcher"
    assert data["is_active"] is True
    assert "created_at" in data
    assert "password_hash" not in data


def test_get_profile_without_token(client):
    response = client.get("/api/v1/users/me")
    assert response.status_code == 401
    assert "credentials were not provided" in response.json()["detail"].lower()


def test_get_profile_expired_token(client, test_user):
    expired_token = create_access_token(
        subject=str(test_user.id),
        role=test_user.role,
        expires_delta=timedelta(seconds=-10),
    )
    response = client.get(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert response.status_code == 401
    assert "expired" in response.json()["detail"].lower()


def test_get_profile_malformed_token(client):
    response = client.get(
        "/api/v1/users/me",
        headers={"Authorization": "Bearer not.a.valid.jwt.token"},
    )
    assert response.status_code == 401
    assert "invalid" in response.json()["detail"].lower()


def test_get_profile_inactive_user(client, inactive_user):
    inactive_token = create_access_token(
        subject=str(inactive_user.id),
        role=inactive_user.role,
    )
    response = client.get(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {inactive_token}"},
    )
    assert response.status_code == 403
    assert "inactive" in response.json()["detail"].lower()


def test_update_profile_name(client, user_access_token):
    response = client.patch(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {user_access_token}"},
        json={"name": "Dr. Jane Updated Doe"},
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Dr. Jane Updated Doe"


def test_update_profile_email(client, user_access_token):
    response = client.patch(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {user_access_token}"},
        json={"email": "dr.jane.new@hospital.org"},
    )
    assert response.status_code == 200
    assert response.json()["email"] == "dr.jane.new@hospital.org"


def test_prevent_role_escalation_via_profile_update(client, test_user, user_access_token):
    response = client.patch(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {user_access_token}"},
        json={"role": "administrator"},
    )
    assert response.status_code == 200
    # The role must remain unchanged
    assert response.json()["role"] == "medical_researcher"

