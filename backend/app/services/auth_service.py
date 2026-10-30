import secrets
from datetime import datetime, timedelta, timezone
from typing import Tuple, Optional
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import hash_password, verify_password, create_access_token, hash_token
from app.models.user import User, UserRole
from app.models.refresh_token import RefreshToken
from app.schemas.user import UserRegister


class AuthService:
    @staticmethod
    def register_user(db: Session, user_in: UserRegister) -> User:
        # Check if email already exists
        existing_user = db.scalar(
            select(User).where(User.email == user_in.email.lower().strip())
        )
        if existing_user:
            raise ValueError("Email already registered")

        new_user = User(
            name=user_in.name.strip(),
            email=user_in.email.lower().strip(),
            password_hash=hash_password(user_in.password),
            role=UserRole.MEDICAL_RESEARCHER.value,  # Default role
            is_active=True,
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user

    @staticmethod
    def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
        user = db.scalar(select(User).where(User.email == email.lower().strip()))
        if not user:
            return None
        if not verify_password(password, user.password_hash):
            return None
        return user

    @staticmethod
    def create_refresh_token(db: Session, user_id) -> str:
        raw_token = secrets.token_urlsafe(48)
        hashed = hash_token(raw_token)
        expires_at = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

        refresh_token_record = RefreshToken(
            user_id=user_id,
            token_hash=hashed,
            expires_at=expires_at,
        )
        db.add(refresh_token_record)
        db.commit()
        return raw_token

    @staticmethod
    def rotate_refresh_token(db: Session, raw_token: str) -> Tuple[User, str, str]:
        """
        Validates the incoming refresh token, revokes it, and issues a new access token and refresh token.
        Returns: (user, access_token, new_refresh_token)
        """
        token_hash = hash_token(raw_token)
        record = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == token_hash))

        if not record:
            raise ValueError("Invalid refresh token")

        now = datetime.now(timezone.utc)
        if record.revoked_at is not None:
            # Token reuse detection: security precaution, revoke all active sessions for this user
            db.execute(
                update(RefreshToken)
                .where(RefreshToken.user_id == record.user_id, RefreshToken.revoked_at.is_(None))
                .values(revoked_at=now)
            )
            db.commit()
            raise ValueError("Revoked refresh token reused")

        expires_at = record.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        if expires_at <= now:
            raise ValueError("Expired refresh token")

        user = db.scalar(select(User).where(User.id == record.user_id))
        if not user or not user.is_active:
            raise ValueError("User account is inactive or not found")

        # Revoke the current refresh token
        record.revoked_at = now

        # Generate a new refresh token (Rotation)
        new_raw_token = secrets.token_urlsafe(48)
        new_record = RefreshToken(
            user_id=user.id,
            token_hash=hash_token(new_raw_token),
            expires_at=now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        )
        db.add(new_record)
        db.commit()

        # Issue new access token
        access_token = create_access_token(subject=str(user.id), role=user.role)
        return user, access_token, new_raw_token

    @staticmethod
    def revoke_refresh_token(db: Session, raw_token: str) -> bool:
        token_hash = hash_token(raw_token)
        record = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == token_hash))
        if not record:
            return False
        if record.revoked_at is None:
            record.revoked_at = datetime.now(timezone.utc)
            db.commit()
        return True

    @staticmethod
    def revoke_all_user_tokens(db: Session, user_id) -> None:
        db.execute(
            update(RefreshToken)
            .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=datetime.now(timezone.utc))
        )
        db.commit()

    @staticmethod
    def change_password(
        db: Session, user: User, current_password: str, new_password: str
    ) -> None:
        if not verify_password(current_password, user.password_hash):
            raise ValueError("Invalid current password")
        user.password_hash = hash_password(new_password)
        db.commit()
        # Revoke all sessions so the user must log in again
        AuthService.revoke_all_user_tokens(db, user.id)


auth_service = AuthService()
