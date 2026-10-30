from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.core.config import settings
from app.core.security import create_access_token
from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import (
    LoginRequest,
    TokenResponse,
    TokenRefreshRequest,
    PasswordChangeRequest,
    MessageResponse,
)
from app.schemas.user import UserRegister, UserResponse
from app.services.auth_service import auth_service

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new medical researcher user",
)
def register(user_in: UserRegister, db: Session = Depends(get_db)):
    try:
        user = auth_service.register_user(db, user_in)
        return user
    except ValueError as e:
        if "Email already registered" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A user with this email address already exists.",
            )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate user and issue JWT and Refresh tokens",
)
def login(login_in: LoginRequest, db: Session = Depends(get_db)):
    user = auth_service.authenticate_user(db, login_in.email, login_in.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive. Please contact the administrator.",
        )

    access_token = create_access_token(subject=str(user.id), role=user.role)
    refresh_token = auth_service.create_refresh_token(db, user.id)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Rotate refresh token and issue new access token",
)
def refresh(token_in: TokenRefreshRequest, db: Session = Depends(get_db)):
    try:
        user, new_access_token, new_refresh_token = auth_service.rotate_refresh_token(
            db, token_in.refresh_token
        )
        return TokenResponse(
            access_token=new_access_token,
            token_type="bearer",
            refresh_token=new_refresh_token,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Revoke refresh token and end session",
)
def logout(token_in: TokenRefreshRequest, db: Session = Depends(get_db)):
    revoked = auth_service.revoke_refresh_token(db, token_in.refresh_token)
    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or non-existent refresh token",
        )
    return MessageResponse(message="Successfully logged out and revoked session")


@router.post(
    "/change-password",
    response_model=MessageResponse,
    summary="Change user password and revoke all active sessions",
)
def change_password(
    password_data: PasswordChangeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        auth_service.change_password(
            db,
            user=current_user,
            current_password=password_data.current_password,
            new_password=password_data.new_password,
        )
        return MessageResponse(message="Password changed successfully. All sessions revoked.")
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

