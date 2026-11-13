
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes.analysis import router as analysis_router
from app.routes.geometry import router as geometry_router
from app.routes.explainability import router as explainability_router


# --------------------------------------------------
# 1. DEFINE DIRECTORIES
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. CREATE FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="ArteryAI Backend",
    description=(
        "Artery Segmentation, Geometry Analysis "
        "and Explainable AI API"
    ),
    version="1.0.0",
)


# --------------------------------------------------
# 3. ENABLE CORS FOR REACT FRONTEND
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 4. SERVE GENERATED RESULT IMAGES
# --------------------------------------------------

app.mount(
    "/results",
    StaticFiles(directory=str(RESULTS_DIR)),
    name="results"
)


# --------------------------------------------------
# 5. REGISTER API ROUTERS
# --------------------------------------------------

# Image upload and segmentation
app.include_router(analysis_router)

# Centerline and diameter analysis
app.include_router(geometry_router)

# Grad-CAM explainability
app.include_router(explainability_router)


# --------------------------------------------------
# 6. HOME ENDPOINT
# --------------------------------------------------

@app.get("/", tags=["Home"])
def home():
    return {
        "message": "ArteryAI backend is running",
        "status": "success",
        "version": "1.0.0",
    }


# --------------------------------------------------
# 7. HEALTH CHECK ENDPOINT
# --------------------------------------------------

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "message": "Backend is working properly",
    }


# --------------------------------------------------
# 8. STARTUP EVENT
# --------------------------------------------------

@app.on_event("startup")
async def startup_event():

    print("====================================")
    print("       ARTERYAI BACKEND STARTED")
    print("====================================")

    print("Analysis API       : Registered")
    print("Geometry API       : Registered")
    print("Explainability API : Registered")
    print("Results Directory  :", RESULTS_DIR)

    print("====================================")
