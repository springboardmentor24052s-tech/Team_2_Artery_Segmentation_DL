from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, status, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.v1 import api_v1_router
from app.core.config import settings
from app.db.base import Base
from app.db.session import engine, get_db
# Import models to ensure they are registered with Base metadata
import app.models  # noqa: F401


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Safe schema creation workflow: create tables if they do not exist
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        # In environments where database is not yet ready during some tests or builds,
        # fail gracefully or log safely without printing secrets
        print(f"Database initialization warning: {type(e).__name__}")
    yield


app = FastAPI(
    title="Artery Segmentation and Stenosis Detection - Auth API",
    description="Authentication and Role-Based Access Control Backend Service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    swagger_ui_parameters={"persistAuthorization": True},
)

# CORS Middleware
if settings.ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Health check endpoints
@app.get("/api/v1/health", tags=["Health"])
def health_check():
    """Basic liveness probe."""
    return {"status": "ok", "service": "artery-segmentation-auth"}


@app.get("/api/v1/health/ready", tags=["Health"])
def readiness_check(response: Response, db: Session = Depends(get_db)):
    """Readiness probe checking database connectivity."""
    try:
        db.execute(text("SELECT 1"))
        return {
            "status": "ready",
            "database": "connected",
            "environment": settings.ENVIRONMENT,
        }
    except Exception:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "not_ready",
            "database": "disconnected",
            "environment": settings.ENVIRONMENT,
        }


# Mount v1 API
app.include_router(api_v1_router, prefix="/api/v1")

