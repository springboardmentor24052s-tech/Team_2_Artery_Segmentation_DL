# Artery Segmentation and Stenosis Detection — Backend API

Backend REST API service for artery stenosis detection and artery segmentation. Built with **FastAPI**, **SQLAlchemy 2.x**, **PostgreSQL**, **Pydantic 2**, **PyTorch**, and **OpenCV**.

This document describes the backend API only. It covers endpoint contracts, authentication, ML inference integration, database operations, and how to run and test the service.

---

## Table of Contents

1. [Backend Overview](#1-backend-overview)
2. [Technology Stack](#2-technology-stack)
3. [Project Structure](#3-project-structure)
4. [API Documentation](#4-api-documentation)
5. [Authentication Flow](#5-authentication-flow)
6. [ML Inference API](#6-ml-inference-api)
7. [Database and Clinical APIs](#7-database-and-clinical-apis)
8. [Local Setup and Execution](#8-local-setup-and-execution)
9. [Testing the API](#9-testing-the-api)
10. [Important Implementation Notes](#10-important-implementation-notes)

---

## 1. Backend Overview

The backend is a **FastAPI** application that exposes REST endpoints under the `/api/v1` prefix. It handles four areas of responsibility:

### Authentication and user management
- User registration with Argon2 password hashing (`pwdlib[argon2]`)
- JWT access token issuance (HS256, configurable TTL via `ACCESS_TOKEN_EXPIRE_MINUTES`)
- Refresh token rotation with SHA-256 storage and replay detection
- Password change with session revocation
- Role-based access control (`RoleChecker` dependency) with four roles: `radiologist`, `medical_researcher`, `data_scientist`, `administrator`

### PostgreSQL database integration
- SQLAlchemy 2.x ORM with connection pooling (`pool_pre_ping`, `pool_size=10`, `max_overflow=20`)
- ORM tables (`users`, `refresh_tokens`) created automatically on startup via `Base.metadata.create_all()`
- Clinical tables (`cases`, `medical_images`, `segmentation_results`, `stenosis_results`) managed via SQL scripts in `database/database_work.sql`
- SQL-backed endpoints execute parameterized queries against PostgreSQL using `psycopg 3`

### Clinical case and medical image management
- CRUD operations for examination cases, medical image metadata, segmentation results, and stenosis detection results
- Aggregation endpoints for analytics, case reports, and dashboard statistics
- Administrator-only database diagnostic endpoints (foreign keys, table verification, column inspection, orphan record checks)

### ML inference integration
- U-Net (~7.8M params) and Attention U-Net (~34M params) model architectures under `backend/app/ml/`
- `InferenceService` singleton loads both trained checkpoints (`unet_best.pth`, `attention_unet_best.pth`) from `AIML_testing/outputs/models/` during FastAPI startup lifespan
- Two ML inference endpoints: `GET /api/v1/inference/models` and `POST /api/v1/inference/predict`
- Images are uploaded via `multipart/form-data`, preprocessed (grayscale, resize to 256×256, normalize), run through the selected model, and return base64-encoded binary mask and overlay with metadata

---

## 2. Technology Stack

| Component | Technology |
|---|---|
| Web framework | FastAPI |
| ASGI server | Uvicorn `[standard]` |
| Data validation | Pydantic v2 `[email]` |
| Settings management | pydantic-settings |
| ORM | SQLAlchemy 2.x |
| PostgreSQL driver | psycopg 3 `[binary]` |
| Password hashing | pwdlib with Argon2 |
| JWT signing/verification | PyJWT `[crypto]` (HS256) |
| Database migrations | Alembic |
| File upload support | python-multipart |
| ML framework | PyTorch (CPU) |
| Image processing | OpenCV (`opencv-python`) |
| Numerical computing | NumPy |
| Testing | pytest, httpx |
| Runtime | Python 3.12 (tested 3.12.10) |

Dependencies are listed in `requirements.txt`.

---

## 3. Project Structure

```
backend/
├── alembic.ini                 # Alembic migration configuration
├── alembic/
│   ├── env.py                  # Alembic migration runtime environment
│   └── script.py.mako          # Migration revision template
├── app/
│   ├── __init__.py
│   ├── main.py                 # Base entry point: Auth + Users + Health only
│   ├── extended_main.py        # ★ Recommended entry point: all endpoints (48 operations)
│   │
│   ├── api/
│   │   ├── dependencies.py     # get_current_user, RoleChecker RBAC dependency
│   │   └── v1/
│   │       ├── __init__.py     # v1 router aggregator (auth, users, inference)
│   │       ├── auth.py         # 5 authentication routes (/auth/*)
│   │       ├── users.py        # 2 user profile routes (/users/me)
│   │       ├── inference.py    # 2 ML inference routes (/inference/*)
│   │       └── sql_clinical.py # 39 SQL-backed clinical & analytics routes (/sql/*)
│   │
│   ├── core/
│   │   ├── config.py           # pydantic-settings: loads .env, provides Settings singleton
│   │   └── security.py         # Argon2 hash/verify, JWT encode/decode, SHA-256 token hashing
│   │
│   ├── db/
│   │   ├── base.py             # SQLAlchemy DeclarativeBase
│   │   └── session.py          # PostgreSQL engine, SessionLocal, get_db dependency
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py             # User ORM model (UUID PK, name, email, role, is_active)
│   │   └── refresh_token.py    # RefreshToken ORM model (SHA-256 hash, revocation)
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── unet.py             # U-Net architecture (encoder-decoder with skip connections)
│   │   └── attention_unet.py   # Attention U-Net architecture (attention gates)
│   │
│   ├── schemas/
│   │   ├── auth.py             # LoginRequest, TokenResponse, TokenRefreshRequest, PasswordChangeRequest, MessageResponse
│   │   ├── user.py             # UserRegister, UserUpdate, UserResponse
│   │   └── clinical_schemas.py # All clinical request/response Pydantic models
│   │
│   └── services/
│       ├── auth_service.py     # Registration, authentication, token rotation, logout, password change
│       ├── user_service.py     # Profile lookup and update
│       ├── sql_clinical_service.py # All parameterized SQL query methods for clinical endpoints
│       └── inference_service.py # InferenceService singleton: model loading, preprocessing, inference, postprocessing
│
├── .env.example                # Safe configuration template (never commit .env)
├── requirements.txt            # Python dependencies
└── database/
    ├── database_work.sql       # DDL: CREATE TABLE for clinical tables (users, cases, medical_images, segmentation_results, stenosis_results)
    ├── auth_queries.sql        # DML: user, case, image, result SELECT/UPDATE queries
    └── result_queries.sql      # Analytical queries: counts, aggregations, diagnostics
```

### Application entry points

| Entry Point | Uvicorn command | Endpoints exposed |
|---|---|---|
| `app.extended_main:app` ★ | `uvicorn app.extended_main:app` | All routes: Auth + Users + Health + ML Inference + SQL Clinical (48 operations) |
| `app.main:app` | `uvicorn app.main:app` | Auth + Users + Health only (no `/api/v1/sql` router) |

`extended_main.py` imports `app` from `main.py` and additionally mounts the `sql_router` under `/api/v1`. The `api_v1_router` (auth + users + inference) is mounted inside `main.py`.

---

## 4. API Documentation

All endpoints are mounted under `/api/v1`. The FastAPI app title is `"Artery Segmentation and Stenosis Detection - Auth API"`, version `1.0.0`.

### A. Health

| Method | Path | Auth | Description |
|---|---|:---:|---|
| `GET` | `/api/v1/health` | Public | Liveness probe. Always returns `{"status": "ok", "service": "artery-segmentation-auth"}`. |
| `GET` | `/api/v1/health/ready` | Public | Readiness probe. Executes `SELECT 1` against PostgreSQL. Returns `{"status": "ready", "database": "connected", "environment": "..."}` or HTTP 503 if DB unreachable. |

**Response example (health):**
```json
{"status": "ok", "service": "artery-segmentation-auth"}
```

**Response example (ready):**
```json
{"status": "ready", "database": "connected", "environment": "development"}
```

---

### B. Authentication (`/api/v1/auth`)

| Method | Path | Auth | Description |
|---|---|:---:|---|
| `POST` | `/api/v1/auth/register` | Public | Register a new user (default role: `medical_researcher`). Body: `name`, `email`, `password`. Returns `UserResponse`. HTTP 409 if email already registered; HTTP 400 for invalid input. |
| `POST` | `/api/v1/auth/login` | Public | Verify email + password. Returns JWT access token + refresh token. HTTP 401 on bad credentials; HTTP 403 if account inactive. |
| `POST` | `/api/v1/auth/refresh` | Public (token body) | Rotate refresh token. Revokes submitted token, issues new access + refresh pair. Detects replay attacks. HTTP 401 on invalid/expired/revoked token. |
| `POST` | `/api/v1/auth/logout` | Public (token body) | Revoke refresh token. Ends session. HTTP 400 if token invalid or non-existent. |
| `POST` | `/api/v1/auth/change-password` | Bearer JWT | Verify current password, update Argon2 hash, revoke all active sessions. HTTP 400 if current password invalid. |

**Request — Register:**
```json
{
  "name": "Dr. Jane Smith",
  "email": "jane.smith@hospital.example.com",
  "password": "StrongPassword2026!"
}
```

**Response — Register (HTTP 201):**
```json
{
  "id": "<uuid>",
  "name": "Dr. Jane Smith",
  "email": "jane.smith@hospital.example.com",
  "role": "medical_researcher",
  "is_active": true,
  "created_at": "2026-11-12T18:59:43.006781+05:30",
  "updated_at": "2026-11-12T18:59:43.006781+05:30"
}
```

**Request — Login:**
```json
{
  "email": "jane.smith@hospital.example.com",
  "password": "StrongPassword2026!"
}
```

**Response — Login (HTTP 200):**
```json
{
  "access_token": "<signed JWT string>",
  "token_type": "bearer",
  "refresh_token": "<opaque 64-character URL-safe token>",
  "expires_in": 1800
}
```

**Request — Refresh / Logout:**
```json
{
  "refresh_token": "<opaque token>"
}
```

**Request — Change password:**
```json
{
  "current_password": "StrongPassword2026!",
  "new_password": "NewSecurePassword2026!"
}
```

---

### C. Users (`/api/v1/users`)

| Method | Path | Auth | Description |
|---|---|:---:|---|
| `GET` | `/api/v1/users/me` | Bearer JWT | Return authenticated user's profile (`id`, `name`, `email`, `role`, `is_active`, timestamps). |
| `PATCH` | `/api/v1/users/me` | Bearer JWT | Update `name` and/or `email`. HTTP 409 if email already taken by another user. Role is not modifiable via this endpoint. |

**Response — GET /users/me (HTTP 200):**
```json
{
  "id": "<uuid>",
  "name": "Dr. Jane Smith",
  "email": "jane.smith@hospital.example.com",
  "role": "medical_researcher",
  "is_active": true,
  "created_at": "2026-11-12T18:59:43.006781+05:30",
  "updated_at": "2026-11-12T18:59:43.006781+05:30"
}
```

**Request — PATCH /users/me:**
```json
{
  "name": "Dr. Jane Smith, MD",
  "email": "jane.smith@hospital.example.com"
}
```

---

### D. ML Inference (`/api/v1/inference`)

| Method | Path | Auth | Description |
|---|---|:---:|---|
| `GET` | `/api/v1/inference/models` | Bearer JWT | List available segmentation models. Returns `["unet", "attention_unet"]`. HTTP 503 if models not loaded. |
| `POST` | `/api/v1/inference/predict` | Bearer JWT | Upload an angiography image, run segmentation inference, return mask + overlay + metadata. HTTP 400 for invalid file type or model; HTTP 503 if models not loaded; HTTP 500 on inference failure. |

**Request — GET /inference/models (HTTP 200):**
```json
["unet", "attention_unet"]
```

**Request — POST /inference/predict (multipart/form-data):**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | UploadFile | Yes | Angiography image (PNG, JPG, JPEG, TIFF, BMP) |
| `model_type` | string (form field) | No | `"unet"` (default) or `"attention_unet"` |
| `threshold` | float (form field) | No | Binarization threshold 0.0–1.0 (default: 0.3) |

**Response — POST /inference/predict (HTTP 200):**
```json
{
  "model_name": "unet",
  "original_height": 512,
  "original_width": 512,
  "threshold": 0.3,
  "segmented_pixels": 968,
  "vessel_area_ratio": 0.003692626953125,
  "mask_base64": "<base64-encoded PNG of binary mask>",
  "overlay_base64": "<base64-encoded PNG of segmentation overlay>"
}
```

**Error responses:**
- HTTP 400: `{"detail": "Unsupported image type: text/plain. Allowed types: ..."}`
- HTTP 400: `{"detail": "Invalid model_type 'invalid'. Allowed values: 'unet', 'attention_unet'}"`
- HTTP 401: `{"detail": "Authentication credentials were not provided"}`
- HTTP 503: `{"detail": "Inference models are not loaded yet."}`
- HTTP 500: `{"detail": "Inference failed: ..."}`

**Inference flow (implemented in `inference_service.py`):**

1. **Preprocessing** (`preprocess_image`): The uploaded image bytes are decoded by OpenCV (`cv2.imdecode` grayscale). The original dimensions are captured. The image is resized to 256×256 (`cv2.INTER_AREA`), normalized to `[0, 1]`, and converted to a PyTorch tensor of shape `(1, 1, 256, 256)` on the selected device.
2. **Model selection** (`get_model`): The requested model key (`"unet"` or `"attention_unet"`) is looked up in the preloaded `_models` dictionary.
3. **Forward pass**: `torch.sigmoid(model(tensor))` produces a probability map.
4. **Postprocessing** (`postprocess_mask`): Probabilities are binarized using the threshold (default 0.3) and resized back to original dimensions (`cv2.INTER_NEAREST`).
5. **Analytics**: `segmented_pixels` = count of white pixels in the binary mask; `vessel_area_ratio` = `segmented_pixels / (height × width)`.
6. **Overlay** (`create_overlay`): The binary mask is blended over the original grayscale image (red color, alpha 0.55).
7. **Base64 encoding** (`encode_image_to_base64`): Both the binary mask and overlay are encoded as PNG base64 strings.

Models are already trained. Checkpoints (`unet_best.pth`, `attention_unet_best.pth`) are loaded from `AIML_testing/outputs/models/` at application startup. The API executes inference only — it does not train models.

---

### E. SQL-backed Clinical Operations (`/api/v1/sql`)

All endpoints in this group require Bearer JWT authentication unless noted. They execute parameterized SQL against PostgreSQL tables defined in `database/database_work.sql`.

#### Users

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/sql/users/email-exists` | Public. Check whether an email already exists. Query param: `?email=`. Returns `{"email": "...", "exists": true}`. |
| `GET` | `/api/v1/sql/users/by-email` | Bearer JWT. Retrieve user record by email (excludes `password_hash`). |
| `GET` | `/api/v1/sql/users` | Bearer JWT. List all registered users (`user_id`, `full_name`, `email`, `created_at`). |
| `PATCH` | `/api/v1/sql/users/{user_id}/name` | Bearer JWT. Update a user's `full_name` by integer `user_id`. |

#### Cases

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/sql/cases` | List all cases ordered by `case_id DESC`. |
| `POST` | `/api/v1/sql/cases` | Create a case. Body: `user_id` (optional), `case_name`, `imaging_modality` (default `"CTA"`), `target_vessel` (default `"LAD"`), `status` (default `"pending"`). |
| `GET` | `/api/v1/sql/cases/{case_id}` | Retrieve a specific case by `case_id`. HTTP 404 if not found. |
| `GET` | `/api/v1/sql/cases/user/{user_id}` | List all cases for a given `user_id`. |
| `PATCH` | `/api/v1/sql/cases/{case_id}` | Update case metadata fields. |

#### Medical Images

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/sql/images` | List all medical images ordered by `image_id DESC`. |
| `GET` | `/api/v1/sql/images/{image_id}` | Retrieve an image by `image_id`. HTTP 404 if not found. |
| `GET` | `/api/v1/sql/cases/{case_id}/images` | List all images linked to a case. |
| `POST` | `/api/v1/sql/cases/{case_id}/images` | Register image metadata for a case. Body: `image_name`, `image_path`, `image_format` (default `"DICOM"`), `image_size` (optional). |
| `PATCH` | `/api/v1/sql/images/{image_id}` | Update image metadata fields. |
| `DELETE` | `/api/v1/sql/images/{image_id}` | Delete image metadata. Blocked (HTTP 409) if dependent segmentation or stenosis records exist. |

#### Segmentation Results

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/sql/segmentation-results` | List all segmentation results ordered by `segmentation_id DESC`. |
| `GET` | `/api/v1/sql/segmentation-results/latest` | Retrieve the most recent segmentation result by `created_at`. |
| `GET` | `/api/v1/sql/images/{image_id}/segmentation-results` | List all segmentation results for a specific image. |
| `GET` | `/api/v1/sql/cases/{case_id}/segmentation-results` | List all segmentation results linked to a case (via image JOIN). |
| `POST` | `/api/v1/sql/images/{image_id}/segmentation-results` | Insert a segmentation result. Body: `mask_path`, `model_name`, `dice_score`, `iou_score`, `precision_score`, `recall_score`, `f1_score`, `mean_vessel_diameter`, `model_version`, `centerline_path`. |

#### Stenosis Results

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/sql/stenosis-results` | List all stenosis results. Optional `?sort_by=confidence` to order by confidence score descending. |
| `GET` | `/api/v1/sql/stenosis-results/latest` | Retrieve the most recent stenosis result. |
| `GET` | `/api/v1/sql/stenosis-results/filter` | Filter by severity. Query param: `?severity=Mild`. |
| `GET` | `/api/v1/sql/images/{image_id}/stenosis-results` | List all stenosis results for a specific image. |
| `GET` | `/api/v1/sql/cases/{case_id}/stenosis-results` | List all stenosis results linked to a case (via image JOIN). |
| `POST` | `/api/v1/sql/images/{image_id}/stenosis-results` | Insert a stenosis detection result. Body: `location`, `severity`, `confidence_score`, `estimated_diameter`. |

---

### F. Analytics and Reporting (`/api/v1/sql/analytics`, `/api/v1/sql/reports`)

| Method | Path | Auth | Description |
|---|---|:---:|---|
| `GET` | `/api/v1/sql/analytics/segmentation-count` | Bearer JWT | Returns `{"count": <int>}` total segmentation results. |
| `GET` | `/api/v1/sql/analytics/stenosis-count` | Bearer JWT | Returns `{"count": <int>}` total stenosis results. |
| `GET` | `/api/v1/sql/analytics/model-performance` | Bearer JWT | Average Dice score per model, ordered descending. |
| `GET` | `/api/v1/sql/analytics/model-performance/detailed` | Bearer JWT | Full metrics per model: count, avg Dice, IoU, precision, recall, F1. |
| `GET` | `/api/v1/sql/analytics/stenosis-severity-distribution` | Bearer JWT | Total cases and avg confidence grouped by severity. |
| `GET` | `/api/v1/sql/analytics/summary` | Bearer JWT | Combined summary: total counts + model Dice performance. |
| `GET` | `/api/v1/sql/dashboard/analytics` | Bearer JWT | Total counts for users, cases, images, segmentations, stenosis detections, plus recent activity items. |
| `GET` | `/api/v1/sql/reports/case-analysis` | Bearer JWT | Multi-table JOIN report: case → images → segmentation + stenosis results. Optional `?case_id=` filter. |
| `GET` | `/api/v1/sql/reports/case-summary-counts` | Bearer JWT | Per-case count of distinct images, segmentation results, and stenosis results. |

---

### G. Database Diagnostics (`/api/v1/sql/diagnostics`) — Administrator only

These endpoints require the `administrator` role. Any other authenticated user receives HTTP 403.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/sql/diagnostics/foreign-keys` | List all FK constraints in the `public` schema via `information_schema`. |
| `GET` | `/api/v1/sql/diagnostics/tables` | Verify all 5 project tables exist in `information_schema.tables`. |
| `GET` | `/api/v1/sql/diagnostics/columns` | Inspect column names, data types, nullability. Optional `?table_name=` filter. |
| `GET` | `/api/v1/sql/diagnostics/foreign-key-consistency` | Orphan record check across cases, images, segmentations, stenosis results. |

---

## 5. Authentication Flow

### Registration
1. Client sends `POST /api/v1/auth/register` with JSON body `{name, email, password}`.
2. Server checks if email already exists (HTTP 409 if so).
3. Password is hashed with Argon2 via `pwdlib`.
4. User is created with default role `medical_researcher` and `is_active=true`.
5. Response returns the user object (no password hash).

### Login
1. Client sends `POST /api/v1/auth/login` with JSON body `{email, password}`.
2. Server looks up user by email and verifies password with Argon2.
3. If invalid or inactive, returns HTTP 401 or HTTP 403 respectively.
4. On success:
   - A signed JWT access token is created (claims: `sub` = user UUID, `role`, `type` = `"access"`, `iat`, `exp`; TTL = `ACCESS_TOKEN_EXPIRE_MINUTES` minutes, default 30).
   - A 48-byte URL-safe random refresh token is generated and its SHA-256 hash is stored in the `refresh_tokens` table.
5. Response returns `{access_token, token_type: "bearer", refresh_token, expires_in}`.

### Token refresh
1. Client sends `POST /api/v1/auth/refresh` with JSON body `{refresh_token}`.
2. Server hashes the token with SHA-256 and looks it up in `refresh_tokens`.
3. If not found, revoked, or expired → HTTP 401.
4. If revoked token reuse is detected → all active sessions for that user are revoked, then HTTP 401.
5. Otherwise, the old token is revoked, a new access token and new refresh token are issued.

### Logout
1. Client sends `POST /api/v1/auth/logout` with JSON body `{refresh_token}`.
2. Server marks the token as `revoked_at = now`.

### Password change
1. Client sends `POST /api/v1/auth/change-password` with Bearer JWT and body `{current_password, new_password}`.
2. Server verifies current password, updates the Argon2 hash, and revokes all active sessions.

### Bearer token authorization
- Protected endpoints require an `Authorization: Bearer <access_token>` header.
- The `get_current_user` dependency decodes the JWT, validates `type == "access"`, resolves the user by `sub` UUID, and checks `is_active`.
- Expired tokens → HTTP 401 `"Token has expired"`. Invalid tokens → HTTP 401 `"Invalid authentication token"`. Missing credentials → HTTP 401 `"Authentication credentials were not provided"`. Inactive accounts → HTTP 403.

### Authorizing in Swagger UI
1. Open `http://127.0.0.1:8000/docs`.
2. Expand `POST /api/v1/auth/login`, click **Try it out**, enter credentials, click **Execute**.
3. Copy the `access_token` from the response.
4. Click the **Authorize** 🔒 button at the top of the page.
5. Enter `Bearer <your_access_token>` and click **Authorize**.
6. All protected endpoints will now include the authorization header automatically.

### Practical example (PowerShell)
```powershell
$loginBody = @{
    email    = "user@example.com"
    password = "StrongPassword2026!"
} | ConvertTo-Json

$resp  = Invoke-RestMethod -Method Post `
    -Uri "http://127.0.0.1:8000/api/v1/auth/login" `
    -ContentType "application/json" -Body $loginBody

$token = $resp.access_token
$headers = @{ Authorization = "Bearer $token" }

Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/v1/users/me" -Headers $headers
```

---

## 6. ML Inference API

The ML inference API exposes two endpoints under `/api/v1/inference`. It wraps the existing `InferenceService` singleton (`backend/app/services/inference_service.py`) which loads trained PyTorch models once at application startup and reuses them for all inference requests.

### Models available
- `unet` — U-Net encoder-decoder architecture (~7.8M parameters), defined in `backend/app/ml/unet.py`
- `attention_unet` — Attention U-Net with attention gates (~34M parameters), defined in `backend/app/ml/attention_unet.py`

Both checkpoints (`unet_best.pth`, `attention_unet_best.pth`) are already trained and stored in `AIML_testing/outputs/models/`. The API performs inference only — it does not train models.

### GET /api/v1/inference/models

Returns the list of loaded model names:
```json
["unet", "attention_unet"]
```
Requires Bearer JWT. Returns HTTP 503 if models have not been loaded (e.g., checkpoints missing at startup).

### POST /api/v1/inference/predict

Accepts an angiography image via `multipart/form-data` and returns segmentation results.

**Request fields:**
| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file (multipart) | Yes | Image file (PNG, JPG, JPEG, TIFF, BMP) |
| `model_type` | string (form field) | No | `"unet"` (default) or `"attention_unet"` |
| `threshold` | float (form field) | No | Binarization threshold, 0.0–1.0 (default: 0.3) |

**PowerShell example:**
```powershell
$imagePath = "C:\path\to\angiography.png"
$fileBytes = [System.IO.File]::ReadAllBytes($imagePath)
$boundary = [System.Guid]::NewGuid().ToString()
$formData = "--$boundary`r`nContent-Disposition: form-data; name=`"file`"; filename=`"angiography.png`"`r`nContent-Type: image/png`r`n`r`n"
$formData += [System.Text.Encoding]::UTF8.GetString($fileBytes)
$formData += "`r`n--$boundary`r`nContent-Disposition: form-data; name=`"model_type`"`r`n`r`nunet`r`n--$boundary--`r`n"
$contentType = "multipart/form-data; boundary=$boundary"

Invoke-RestMethod -Method Post `
    -Uri "http://127.0.0.1:8000/api/v1/inference/predict?model_type=unet" `
    -Headers $headers -ContentType $contentType -Body $formData
```

**Response (HTTP 200):**
```json
{
  "model_name": "unet",
  "original_height": 512,
  "original_width": 512,
  "threshold": 0.3,
  "segmented_pixels": 968,
  "vessel_area_ratio": 0.003692626953125,
  "mask_base64": "<base64-encoded PNG>",
  "overlay_base64": "<base64-encoded PNG>"
}
```

**Response fields:**
| Field | Type | Description |
|---|---|---|
| `model_name` | string | The model used (`"unet"` or `"attention_unet"`) |
| `original_height` | int | Original image height in pixels |
| `original_width` | int | Original image width in pixels |
| `threshold` | float | Binarization threshold applied |
| `segmented_pixels` | int | Count of foreground (vessel) pixels in the mask |
| `vessel_area_ratio` | float | `segmented_pixels / (height × width)` |
| `mask_base64` | string \| null | Base64-encoded PNG of the binary segmentation mask |
| `overlay_base64` | string \| null | Base64-encoded PNG of the mask overlaid on the original image |

**Error handling:**
- HTTP 400 — unsupported image content type or invalid `model_type`
- HTTP 401 — missing or invalid Bearer token
- HTTP 503 — inference models not loaded at startup
- HTTP 500 — inference failure (e.g., unreadable image bytes)

### How the API router communicates with the inference service

`backend/app/api/v1/inference.py` defines the router and imports the module-level `inference_service` singleton from `app.services.inference_service`. The singleton is initialized at import time and its `load_models()` method is called inside the FastAPI `lifespan` context manager in `main.py`. Because the models are loaded once and cached in `inference_service._models`, every prediction request reuses the same in-memory model instances without reloading checkpoints from disk.

---

## 7. Database and Clinical APIs

### Database entities and relationships

The ORM layer (`backend/app/models/`) manages two tables:

| Table | Primary key | Key columns |
|---|---|---|
| `users` | `id` (UUID) | `name`, `email` (unique), `password_hash`, `role`, `is_active`, `created_at`, `updated_at` |
| `refresh_tokens` | `id` (UUID) | `user_id` (FK → `users.id`), `token_hash` (SHA-256), `expires_at`, `revoked_at`, `created_at` |

User roles: `radiologist`, `medical_researcher`, `data_scientist`, `administrator`.

The clinical tables are defined in `database/database_work.sql` (DDL, not executed by the application) and queried by SQL-backed endpoints:

| Table | Primary key | Key columns | Foreign keys |
|---|---|---|---|
| `users` | `user_id` SERIAL | `full_name`, `email`, `password_hash`, `created_at` | — |
| `cases` | `case_id` SERIAL | `case_name`, `imaging_modality`, `target_vessel`, `status`, `created_at` | `user_id → users.user_id` |
| `medical_images` | `image_id` SERIAL | `image_name`, `image_path`, `image_format`, `image_size`, `uploaded_at` | `case_id → cases.case_id` |
| `segmentation_results` | `segmentation_id` SERIAL | `mask_path`, `model_name`, `dice_score`, `iou_score`, `precision_score`, `recall_score`, `f1_score`, `mean_vessel_diameter`, `model_version`, `centerline_path` | `image_id → medical_images.image_id` |
| `stenosis_results` | `stenosis_id` SERIAL | `location`, `severity`, `confidence_score`, `estimated_diameter`, `created_at` | `image_id → medical_images.image_id` |

> [!IMPORTANT]
> There is a **schema divergence** between the ORM `users` table (`id` UUID, column `name`) and the clinical SQL `users` table (`user_id` SERIAL, column `full_name`). SQL-backed user endpoints operate against the clinical schema version.

### How the API interacts with PostgreSQL

- `backend/app/db/session.py` creates a SQLAlchemy engine using `DATABASE_URL` (psycopg 3 driver) with connection pooling (`pool_pre_ping=True`, `pool_size=10`, `max_overflow=20`).
- The `get_db` dependency yields a `SessionLocal` session per request and closes it afterwards.
- `main.py` calls `Base.metadata.create_all(bind=engine)` in the lifespan handler to create ORM tables (`users`, `refresh_tokens`) automatically.
- Clinical tables (`cases`, `medical_images`, `segmentation_results`, `stenosis_results`) must be created manually by executing `database/database_work.sql` in PostgreSQL before SQL-backed endpoints can return data.
- All SQL-backed endpoints use parameterized queries through `SQLClinicalService` (`backend/app/services/sql_clinical_service.py`), which wraps raw SQL execution with error handling for `OperationalError` and `ProgrammingError` (returning HTTP 503 on database failures).

---

## 8. Local Setup and Execution

### Prerequisites
- Python 3.10–3.12 (3.12.10 verified)
- PostgreSQL 15–18 running locally or reachable over network
- Windows PowerShell 5.1+ (or macOS/Linux bash)

### Step 1 — Navigate to the project directory
```powershell
cd "Artery project"
```

### Step 2 — Create and activate a virtual environment
```powershell
# Create (if not already present)
python -m venv .venv

# Activate on Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

> If PowerShell blocks script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### Step 3 — Install dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Configure environment variables
```powershell
Copy-Item .env.example .env
# Edit .env and set DATABASE_URL, JWT_SECRET_KEY, etc.
```

| Variable | Required | Example |
|---|---:|---|
| `DATABASE_URL` | Yes | `postgresql+psycopg://postgres:your_password@localhost:5432/artery_db` |
| `JWT_SECRET_KEY` | Yes | Random 32+ character string |
| `JWT_ALGORITHM` | No | `HS256` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | `30` |
| `REFRESH_TOKEN_EXPIRE_DAYS` | No | `7` |
| `ENVIRONMENT` | No | `development` |
| `ALLOWED_ORIGINS` | No | `http://localhost:3000,http://127.0.0.1:3000` |

Generate a secure secret:
```powershell
.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Step 5 — PostgreSQL and database setup
1. Verify PostgreSQL is running:
   ```powershell
   Get-Service *postgres*
   ```
2. Create the database:
   ```sql
   CREATE DATABASE artery_db;
   ```
3. Apply the clinical schema by executing `database/database_work.sql` in pgAdmin or psql against `artery_db`. This creates the `cases`, `medical_images`, `segmentation_results`, and `stenosis_results` tables.
   > [!WARNING] Line 53 of `database_work.sql` has a premature semicolon that terminates an `ALTER TABLE` early. Change `;` to `,` on that line before executing.

### Step 6 — Run the API
Set `PYTHONPATH` to `backend` so the `app` package is importable, then start Uvicorn:

```powershell
$env:PYTHONPATH = "backend"
.\.venv\Scripts\python.exe -m uvicorn app.extended_main:app --host 127.0.0.1 --port 8000 --reload
```

Use `app.extended_main:app` (not `app.main:app`) to expose all endpoints including SQL-backed clinical routes.

### Verify the service
| Probe | URL | Expected |
|---|---|---|
| Liveness | `GET http://127.0.0.1:8000/api/v1/health` | `{"status": "ok", "service": "artery-segmentation-auth"}` |
| Readiness | `GET http://127.0.0.1:8000/api/v1/health/ready` | `{"status": "ready", "database": "connected", "environment": "development"}` |

If readiness returns HTTP 503, PostgreSQL is not reachable with the configured `DATABASE_URL`.

### Swagger UI
- Interactive docs: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- OpenAPI schema: `http://127.0.0.1:8000/openapi.json`

---

## 9. Testing the API

### Via Swagger UI
1. Open `http://127.0.0.1:8000/docs`.
2. **Authenticate:** Expand `POST /api/v1/auth/login`, enter valid credentials, click **Execute**, copy the `access_token`.
3. Click **Authorize** 🔒, enter `Bearer <access_token>`, click **Authorize**.
4. **Test ML model availability:** `GET /api/v1/inference/models` → returns `["unet", "attention_unet"]`.
5. **Run inference:** `POST /api/v1/inference/predict`, set `model_type` to `unet` or `attention_unet`, optionally set `threshold` (0.0–1.0), upload an angiography image file. Response includes `mask_base64` and `overlay_base64`.
6. **Test clinical endpoints:** Any `GET /api/v1/sql/*` endpoint with the Bearer token authorized.

### Via PowerShell (Invoke-RestMethod)
1. Register and log in to obtain a Bearer token (see §5 example).
2. Call protected endpoints with `-Headers $headers`.
3. For file upload, construct `multipart/form-data` manually as shown in §6.

### Automated tests
The project includes a pytest suite (45 tests) that runs against an in-memory SQLite database:
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```
Tests cover authentication flows, clinical CRUD, analytics, RBAC enforcement, and user profile operations. SQLite test success does not guarantee live PostgreSQL integration — diagnostic endpoints querying `information_schema` require a live PostgreSQL connection.

---

## 10. Important Implementation Notes

### Separation of concerns
- **API routers** (`auth.py`, `users.py`, `inference.py`, `sql_clinical.py`) handle route definitions, request/response validation, and HTTP status codes.
- **Services** (`auth_service.py`, `user_service.py`, `sql_clinical_service.py`, `inference_service.py`) contain business logic and data access.
- **Database layer** (`session.py`, `base.py`, ORM models) manages the SQLAlchemy engine and table definitions.
- No preprocessing or inference logic is duplicated in the API router — all ML operations delegate to `InferenceService`.

### Model initialization during startup
`main.py` calls `inference_service.load_models()` inside the FastAPI `lifespan` context manager. Both U-Net and Attention U-Net checkpoints are loaded once into memory. If checkpoints are missing, the service starts with an empty model registry and `/api/v1/inference/*` endpoints return HTTP 503.

### Model reuse
The `inference_service` is a module-level singleton. All prediction requests reuse the same loaded model instances — checkpoints are not reloaded from disk on each request.

### Authentication requirements
All endpoints except `/api/v1/health`, `/api/v1/health/ready`, `/api/v1/auth/*`, and `/api/v1/sql/users/email-exists` require a valid Bearer JWT access token. Diagnostic endpoints additionally require the `administrator` role.

### Error handling conventions
- `ValueError` from services → HTTP 400 Bad Request
- Duplicate email → HTTP 409 Conflict
- Missing/invalid/expired JWT → HTTP 401 Unauthorized
- Inactive user → HTTP 403 Forbidden
- Insufficient role → HTTP 403 Forbidden
- Resource not found → HTTP 404 Not Found
- Database connectivity failure → HTTP 503 Service Unavailable
- Unexpected inference failure → HTTP 500 Internal Server Error

### Undocumented or uncertain behaviour
- The `database_work.sql` DDL has a known syntax error on line 53 (premature semicolon) that must be fixed manually before executing.
- The ORM `users` table and clinical SQL `users` table use different primary key column names (`id` UUID vs `user_id` SERIAL) and different name columns (`name` vs `full_name`). This is a known divergence.
- The `/api/v1/sql/users/{user_id}/name` endpoint targets the clinical `user_id` (integer), which does not exist in the ORM-generated `users` table. It works correctly only when the clinical schema version of `users` is present in PostgreSQL.
- Diagnostic endpoints (`/api/v1/sql/diagnostics/*`) query PostgreSQL `information_schema` and cannot return meaningful results against SQLite.

---

*Backend API documentation for the Artery Segmentation and Stenosis Detection project. Covers the FastAPI backend service only.*
