from typing import Optional
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.user import User
from app.schemas.user import UserUpdate


class UserService:
    @staticmethod
    def get_by_id(db: Session, user_id) -> Optional[User]:
        return db.scalar(select(User).where(User.id == user_id))

    @staticmethod
    def get_by_email(db: Session, email: str) -> Optional[User]:
        return db.scalar(select(User).where(User.email == email.lower().strip()))

    @staticmethod
    def update_profile(db: Session, user: User, update_data: UserUpdate) -> User:
        if update_data.name is not None:
            user.name = update_data.name.strip()
        if update_data.email is not None:
            # Check if email is already taken by another user
            existing = UserService.get_by_email(db, update_data.email)
            if existing and existing.id != user.id:
                raise ValueError("Email is already registered by another user")
            user.email = update_data.email.lower().strip()

        db.commit()
        db.refresh(user)
        return user


user_service = UserService()

