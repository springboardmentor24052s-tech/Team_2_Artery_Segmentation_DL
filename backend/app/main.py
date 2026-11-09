
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.analysis import router as analysis_router
from app.routes.geometry import router as geometry_router


# --------------------------------------------------
# 1. CREATE FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="ArteryAI Backend",
    description="Artery Segmentation and Analysis API",
    version="1.0.0",
)


# --------------------------------------------------
# 2. ENABLE CORS FOR REACT FRONTEND
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 3. REGISTER API ROUTERS
# --------------------------------------------------

# Existing image upload and segmentation APIs
app.include_router(analysis_router)

# New post-processing, centerline and diameter APIs
app.include_router(geometry_router)


# --------------------------------------------------
# 4. HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "ArteryAI backend is running",
        "status": "success",
    }


# --------------------------------------------------
# 5. HEALTH CHECK ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Backend is working properly",
    }
