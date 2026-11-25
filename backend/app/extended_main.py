import os
import sys

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.main import app
from app.api.v1.sql_clinical import router as sql_router

# Include only the SQL-backed endpoints under /api/v1
app.include_router(sql_router, prefix="/api/v1")

# Export app for uvicorn: uvicorn app.extended_main:app
