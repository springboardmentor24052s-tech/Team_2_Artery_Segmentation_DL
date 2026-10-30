import uuid
import pytest
from app.core.security import verify_password, hash_password, create_access_token
from app.models.user import User


def test_register_success(client):
    response = client.post(
        "/api/v1/auth/register",
        json={
            "name": "Dr. John Smith",
            "email": "dr.smith@cardio.clinic",
            "password": "SecurePassword2026!",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Dr. John Smith"
    assert data["email"] == "dr.smith@cardio.clinic"
    assert data["role"] == "medical_researcher"
    assert data["is_active"] is True
    assert "password" not in data
    assert "password_hash" not in data
    assert "id" in data


def test_register_duplicate_email(client, test_user):
    response = client.post(
        "/api/v1/auth/register",
        json={
            "name": "Another User",
            "email": test_user.email,
            "password": "SecurePassword2026!",
        },
    )
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


def test_register_invalid_email(client):
    response = client.post(
        "/api/v1/auth/register",
        json={
            "name": "Invalid User",
            "email": "not-an-email",
            "password": "SecurePassword2026!",
        },
    )
    assert response.status_code == 422


def test_register_short_password(client):
    response = client.post(
        "/api/v1/auth/register",
        json={
            "name": "Invalid User",
            "email": "valid@email.com",
            "password": "short",
        },
    )
    assert response.status_code == 422


def test_register_ignores_role_override(client):
    # Ensure public registration cannot assign role=administrator
    response = client.post(
        "/api/v1/auth/register",
        json={
            "name": "Hacker",
            "email": "hacker@domain.com",
            "password": "HackerPassword123!",
            "role": "administrator",
        },
    )
    assert response.status_code == 201
    assert response.json()["role"] == "medical_researcher"


def test_login_success(client, test_user):
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": test_user.email,
            "password": "SuperSecret123!",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert "password" not in data
    assert "password_hash" not in data


def test_login_incorrect_password(client, test_user):
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": test_user.email,
            "password": "WrongPassword123!",
        },
    )
    assert response.status_code == 401
    assert "Invalid email or password" in response.json()["detail"]


def test_login_nonexistent_email(client):
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "nonexistent@domain.com",
            "password": "WrongPassword123!",
        },
    )
    assert response.status_code == 401
    assert "Invalid email or password" in response.json()["detail"]


def test_password_hash_differs_from_plaintext(test_user):
    assert test_user.password_hash != "SuperSecret123!"
    assert verify_password("SuperSecret123!", test_user.password_hash)


def test_refresh_token_rotation(client, test_user):
    # Step 1: Login to get refresh token
    login_resp = client.post(
        "/api/v1/auth/login",
        json={
            "email": test_user.email,
            "password": "SuperSecret123!",
        },
    )
    refresh_token = login_resp.json()["refresh_token"]

    # Step 2: Call /refresh with the token
    refresh_resp = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert refresh_resp.status_code == 200
    new_data = refresh_resp.json()
    assert "access_token" in new_data
    assert "refresh_token" in new_data
    new_refresh_token = new_data["refresh_token"]
    assert new_refresh_token != refresh_token

    # Step 3: Old refresh token should now be rejected as revoked
    old_reuse_resp = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert old_reuse_resp.status_code == 401


def test_logout_and_revocation(client, test_user):
    # Login
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "SuperSecret123!"},
    )
    refresh_token = login_resp.json()["refresh_token"]

    # Logout
    logout_resp = client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": refresh_token},
    )
    assert logout_resp.status_code == 200

    # Try to use the revoked refresh token
    refresh_resp = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert refresh_resp.status_code == 401


def test_change_password_success(client, test_user):
    # 1. Login to obtain a refresh token
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "SuperSecret123!"},
    )
    access_token = login_resp.json()["access_token"]
    refresh_token = login_resp.json()["refresh_token"]

    # 2. Change password
    change_resp = client.post(
        "/api/v1/auth/change-password",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "current_password": "SuperSecret123!",
            "new_password": "BrandNewSecretPassword456!",
        },
    )
    assert change_resp.status_code == 200

    # 3. Old refresh token should now be revoked
    refresh_resp = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert refresh_resp.status_code == 401

    # 4. Login with old password should fail
    old_login = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "SuperSecret123!"},
    )
    assert old_login.status_code == 401

    # 5. Login with new password should succeed
    new_login = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "BrandNewSecretPassword456!"},
    )
    assert new_login.status_code == 200


def test_change_password_incorrect_current_password(client, user_access_token):
    response = client.post(
        "/api/v1/auth/change-password",
        headers={"Authorization": f"Bearer {user_access_token}"},
        json={
            "current_password": "WrongPassword!",
            "new_password": "BrandNewSecretPassword456!",
        },
    )
    assert response.status_code == 400
    assert "Invalid current password" in response.json()["detail"]


def test_inactive_user_cannot_login(client, inactive_user):
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": inactive_user.email,
            "password": "SuperSecret123!",
        },
    )
    assert response.status_code == 403
    assert "inactive" in response.json()["detail"].lower()

