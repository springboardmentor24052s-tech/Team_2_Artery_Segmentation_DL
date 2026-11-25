import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-this-secret-before-deployment")
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", "sqlite:///arteryai_dev.db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.abspath(os.getenv("UPLOAD_FOLDER", "uploads"))
    MAX_CONTENT_LENGTH = 15 * 1024 * 1024
