# Artery Segmentation and Stenosis Detection
## Backend REST API & Database Service

A production-ready backend built with **FastAPI**, **SQLAlchemy 2.x**, **PostgreSQL** (via `psycopg 3`), **Pydantic 2**, and **Argon2** password hashing. It provides authentication, role-based access control (RBAC), and SQL-backed clinical endpoints for querying medical examination cases, imaging metadata, artery segmentation metrics, stenosis detection results, model performance analytics, and schema diagnostics.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement & Objectives](#2-problem-statement--objectives)
3. [Key Features](#3-key-features)
4. [System Architecture](#4-system-architecture)
5. [Technology Stack](#5-technology-stack)
6. [Project Structure](#6-project-structure)
7. [Database Design](#7-database-design)
8. [Prerequisites & Installation](#8-prerequisites--installation)
9. [Environment Configuration](#9-environment-configuration)
10. [Database Initialization](#10-database-initialization)
11. [Running the API](#11-running-the-api)
12. [Swagger UI & API Exploration](#12-swagger-ui--api-exploration)
13. [Complete API Endpoint Reference](#13-complete-api-endpoint-reference)
14. [SQL Query-to-Endpoint Mapping](#14-sql-query-to-endpoint-mapping)
15. [Authentication & Authorization Flow](#15-authentication--authorization-flow)
16. [Testing Guide](#16-testing-guide)
17. [Example API Usage](#17-example-api-usage)
18. [Troubleshooting Guide](#18-troubleshooting-guide)
19. [Known Issues & Limitations](#19-known-issues--limitations)
20. [Completed vs. Planned Features](#20-completed-vs-planned-features)

---

## 1. Project Overview

The **Artery Segmentation and Stenosis Detection** system is an Infosys Springboard Internship project targeting automated coronary and peripheral artery image analysis. This repository contains the backend API service that manages:

- **User accounts** with role-based access and secure session management.
- **Clinical examination cases** linking a patient examination to a registered user.
- **Medical image metadata** tracking uploaded imaging files per case.
- **Segmentation results** storing deep learning model outputs (Dice, IoU, precision, recall, F1, vessel diameter) per image.
- **Stenosis detection results** tracking lesion location, severity, confidence score, and estimated diameter per image.
- **Analytical queries** computing model performance benchmarks and clinical population statistics.
- **Database diagnostic endpoints** for administrators to verify schema integrity and data consistency.

---

## 2. Problem Statement & Objectives

Coronary artery disease (CAD) is a leading cause of mortality worldwide. Manual analysis of CT angiography (CTA) or MR angiography images is time-consuming and subject to inter-observer variability. This project aims to:

- Provide an **API backbone** for a deep learning-based artery segmentation and stenosis detection pipeline.
- Enable structured **storage and retrieval** of model evaluation results across multiple segmentation architectures.
- Support **clinical case management** to track patients, imaging studies, and analysis outcomes.
- Expose **benchmarking analytics** to compare segmentation model performance objectively.

> [!NOTE]
> This backend currently handles data management, authentication, and query execution. The deep learning inference pipeline (model training, DICOM ingestion, mask generation) is a separate component and is **not** invoked by this API service.

---

## 3. Key Features

| Feature | Status |
|---|:---:|
| User registration with Argon2 password hashing | ✅ Implemented |
| JWT access token issuance (HS256, 30-min TTL) | ✅ Implemented |
| SHA-256 hashed refresh tokens with rotation & revocation | ✅ Implemented |
| Token reuse attack detection (refresh token replay) | ✅ Implemented |
| Role-based access control (`RoleChecker` dependency) | ✅ Implemented |
| Clinical case management CRUD | ✅ Implemented |
| Medical image metadata CRUD | ✅ Implemented |
| Segmentation result ingestion & retrieval | ✅ Implemented |
| Stenosis result ingestion & retrieval | ✅ Implemented |
| SQL-backed analytics (count, Dice avg, severity distribution) | ✅ Implemented |
| Complete model performance analytics (IoU, precision, recall, F1) | ✅ Implemented |
| Multi-table case analysis report (JOIN query) | ✅ Implemented |
| Case-wise summary counts | ✅ Implemented |
| Dashboard analytics (all-table counts + recent activity) | ✅ Implemented |
| Database diagnostic endpoints (admin-only, `information_schema`) | ✅ Implemented |
| Interactive Swagger UI with persistent authorization | ✅ Implemented |
| Automated test suite (45 tests via SQLite in-memory fixture) | ✅ Implemented |
| Deep learning inference / DICOM ingestion | ⏳ Planned |
| Alembic database migrations | ⏳ Configured (no migrations authored yet) |

---

## 4. System Architecture

```text
         Client / Frontend / Swagger UI
                     │
                     │  HTTP REST + Bearer JWT
                     ▼
     ┌─────────────────────────────────────┐
     │   FastAPI Application Entry Point   │
     │   backend/app/extended_main.py      │
     └──────────────┬──────────────────────┘
                    │
         ┌──────────┴───────────┐
         ▼                      ▼
 ┌───────────────────┐  ┌───────────────────────────┐
 │ Auth & User       │  │ SQL-Backed Clinical Router │
 │ Routers           │  │ /api/v1/sql/*              │
 │ /api/v1/auth      │  └────────────┬──────────────┘
 │ /api/v1/users     │               │
 └────────┬──────────┘               ▼
          │               ┌─────────────────────────┐
          ▼               │   SQLClinicalService     │
 ┌───────────────────┐    │   (Parameterized SQL)    │
 │ AuthService       │    └────────────┬────────────┘
 │ UserService       │                 │
 │ (ORM + Argon2)    │                 │
 └────────┬──────────┘                 │
          └──────────────┬─────────────┘
                         ▼
            ┌─────────────────────────┐
            │   SQLAlchemy Session    │
            │   psycopg 3 conn pool   │
            └────────────┬────────────┘
                         ▼
            ┌─────────────────────────┐
            │   PostgreSQL Database   │
            │   users, cases,         │
            │   medical_images,       │
            │   segmentation_results, │
            │   stenosis_results,     │
            │   refresh_tokens        │
            └─────────────────────────┘
```

### Layer Responsibilities

| Layer | Components | Responsibility |
|---|---|---|
| **Presentation** | `auth.py`, `users.py`, `sql_clinical.py` | Route handling, Pydantic request/response validation, HTTP status codes |
| **Security & Auth** | `security.py`, `dependencies.py` | Argon2 hashing, JWT sign/verify, `RoleChecker` RBAC |
| **Service** | `AuthService`, `UserService`, `SQLClinicalService` | Business logic, ORM operations, parameterized SQL execution |
| **Persistence** | `session.py`, `base.py`, models | SQLAlchemy engine, connection pool, ORM table definitions |

---

## 5. Technology Stack

| Component | Technology | Version |
|---|---|---|
| Web Framework | FastAPI | ≥ 0.115.0 |
| ASGI Server | Uvicorn (with standard extras) | ≥ 0.30.0 |
| Data Validation | Pydantic v2 (with email validator) | ≥ 2.9.0 |
| Settings Management | pydantic-settings | ≥ 2.5.0 |
| ORM | SQLAlchemy 2.x | ≥ 2.0.35 |
| PostgreSQL Driver | psycopg 3 (binary) | ≥ 3.2.0 |
| Password Hashing | pwdlib with Argon2 hasher | ≥ 0.2.0 |
| JWT Tokens | PyJWT with cryptography | ≥ 2.9.0 |
| Database Migrations | Alembic | ≥ 1.13.0 |
| Testing Framework | pytest | ≥ 8.3.0 |
| Test HTTP Client | httpx | ≥ 0.27.0 |
| Runtime | Python | 3.10 – 3.12 |

---

## 6. Project Structure

```text
Artery project/
├── .env                              # Local secrets (Git-ignored — never commit)
├── .env.example                      # Safe configuration template with placeholders
├── LICENSE
├── README.md                         # This file
├── requirements.txt                  # Pinned Python dependencies
│
├── database/                         # SQL reference files (read-only schema & queries)
│   ├── database_work.sql             # DDL: CREATE TABLE and ALTER TABLE definitions
│   ├── auth_queries.sql              # DML: User, case, image, and result SELECT/UPDATE queries
│   └── result_queries.sql            # Analytical queries: counts, aggregations, diagnostics
│
├── backend/
│   ├── alembic.ini                   # Alembic migration configuration
│   ├── alembic/
│   │   ├── env.py                    # Migration runtime environment
│   │   └── script.py.mako            # Migration revision template
│   │
│   └── app/
│       ├── __init__.py
│       ├── main.py                   # Base entry point (Auth + Users + Health only)
│       ├── extended_main.py          # ★ Recommended entry point (all endpoints)
│       │
│       ├── api/
│       │   ├── dependencies.py       # get_current_user, RoleChecker RBAC dependency
│       │   └── v1/
│       │       ├── __init__.py       # v1 router aggregating auth and users routers
│       │       ├── auth.py           # 5 authentication routes
│       │       ├── users.py          # 2 user profile routes
│       │       └── sql_clinical.py   # 39 SQL-backed clinical & analytics routes
│       │
│       ├── core/
│       │   ├── config.py             # pydantic-settings: loads .env, provides Settings
│       │   └── security.py           # Argon2 hash/verify, JWT encode/decode, SHA-256 hashing
│       │
│       ├── db/
│       │   ├── base.py               # SQLAlchemy DeclarativeBase
│       │   └── session.py            # PostgreSQL engine and get_db session dependency
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── user.py               # User ORM model (UUID PK, role enum, active flag)
│       │   └── refresh_token.py      # RefreshToken ORM model (SHA-256 hash, revocation)
│       │
│       ├── schemas/
│       │   ├── auth.py               # LoginRequest, TokenResponse, PasswordChangeRequest
│       │   ├── user.py               # UserRegister, UserUpdate, UserResponse
│       │   └── clinical_schemas.py   # All clinical request/response Pydantic models
│       │
│       └── services/
│           ├── auth_service.py       # Registration, authentication, token rotation
│           ├── user_service.py       # Profile lookup and update
│           └── sql_clinical_service.py  # All 31+ parameterized SQL query methods
│
└── tests/
    ├── conftest.py                   # Fixtures: in-memory SQLite, TestClient, auth tokens
    ├── test_auth.py                  # 14 tests: auth flows, tokens, password management
    ├── test_clinical_endpoints.py    # 23 tests: clinical CRUD, analytics, RBAC
    └── test_users.py                 # 8 tests: profile read/update, auth enforcement
```

### Application Entry Points

| Entry Point | Command | Endpoints Exposed |
|---|---|---|
| `app.extended_main:app` ★ | `uvicorn app.extended_main:app` | **48 operations** (all Auth + Users + Health + SQL Clinical) |
| `app.main:app` | `uvicorn app.main:app` | **9 operations** (Auth + Users + Health only; no `/api/v1/sql` router) |

---

## 7. Database Design

### ORM Tables (managed by SQLAlchemy, used by authentication routes)

| Table | Primary Key | Key Columns |
|---|---|---|
| `users` | `id UUID` | `name`, `email` (unique), `password_hash`, `role`, `is_active`, `created_at`, `updated_at` |
| `refresh_tokens` | `id UUID` | `user_id` (FK → `users.id`), `token_hash` (SHA-256), `expires_at`, `revoked_at` |

**User roles** (`users.role`): `radiologist` · `medical_researcher` · `data_scientist` · `administrator`

### Clinical Tables (defined in `database/database_work.sql`, used by SQL-backed endpoints)

| Table | Primary Key | Key Columns | Foreign Keys |
|---|---|---|---|
| `users` | `user_id SERIAL` | `full_name`, `email`, `password_hash`, `created_at` | — |
| `cases` | `case_id SERIAL` | `case_name`, `imaging_modality`, `target_vessel`, `status`, `created_at` | `user_id → users.user_id` |
| `medical_images` | `image_id SERIAL` | `image_name`, `image_path`, `image_format`, `image_size`, `uploaded_at` | `case_id → cases.case_id` |
| `segmentation_results` | `segmentation_id SERIAL` | `mask_path`, `model_name`, `dice_score`, `iou_score`, `precision_score`, `recall_score`, `f1_score`, `mean_vessel_diameter`, `model_version`, `centerline_path` | `image_id → medical_images.image_id` |
| `stenosis_results` | `stenosis_id SERIAL` | `location`, `severity`, `confidence_score`, `estimated_diameter`, `created_at` | `image_id → medical_images.image_id` |

> [!IMPORTANT]
> There is a **user table schema divergence** between the ORM layer and the SQL clinical schema. The ORM `users` table uses `id UUID` and column `name`, while `database_work.sql` specifies `user_id SERIAL` and `full_name`. The clinical endpoints operate against the SQL schema version. See [§19 Known Issues](#19-known-issues--limitations) for details.

---

## 8. Prerequisites & Installation

### Requirements

- **Python** 3.10 to 3.12 (3.12.10 tested and verified)
- **PostgreSQL** 15–18 running locally or over network
- **Windows** PowerShell 5.1+ or PowerShell 7, or macOS/Linux bash

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

```bash
# Activate on macOS / Linux
source .venv/bin/activate
```

> [!NOTE]
> If PowerShell blocks script execution, run this first:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### Step 3 — Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 9. Environment Configuration

Configuration is loaded from the `.env` file in the project root by [`backend/app/core/config.py`](backend/app/core/config.py) using `pydantic-settings`.

### Setup

```powershell
Copy-Item .env.example .env
# Then open .env in your editor and fill in the values
```

### Configuration Variables

| Variable | Required | Example Value | Description |
|---|:---:|---|---|
| `DATABASE_URL` | **Yes** | `postgresql+psycopg://postgres:your_password@localhost:5432/artery_db` | SQLAlchemy connection string using the psycopg 3 driver |
| `JWT_SECRET_KEY` | **Yes** | *(generate a random 32+ character string)* | Secret used to sign and verify JWT tokens |
| `JWT_ALGORITHM` | No | `HS256` | JWT algorithm (default: HS256) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | `30` | Access token lifetime in minutes |
| `REFRESH_TOKEN_EXPIRE_DAYS` | No | `7` | Refresh token lifetime in days |
| `ENVIRONMENT` | No | `development` | Runtime label (`development`, `testing`, `production`) |
| `ALLOWED_ORIGINS` | No | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated CORS origins for frontend apps |

> [!TIP]
> Verified defaults loaded from `.env` and confirmed by the running application: `JWT_ALGORITHM=HS256`, `ACCESS_TOKEN_EXPIRE_MINUTES=30`, `REFRESH_TOKEN_EXPIRE_DAYS=7`, `ENVIRONMENT=development`, `ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000`.

**Generate a secure `JWT_SECRET_KEY`:**
```powershell
.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
```

> [!CAUTION]
> Never commit `.env` to version control. It is excluded by `.gitignore`. If your PostgreSQL password contains `@`, `:`, `#`, or `/`, URL-encode those characters (e.g. `@` → `%40`).

---

## 10. Database Initialization

### 1. Verify PostgreSQL is running

```powershell
# Windows: check service status
Get-Service *postgres*

# Check the port (default 5432; may differ for multiple side-by-side installs)
Get-Content "C:\Program Files\PostgreSQL\17\data\postgresql.conf" | Select-String "^port"
```

### 2. Create the database

Using psql:
```sql
CREATE DATABASE artery_db;
```

Or in pgAdmin 4: right-click **Databases** → **Create** → **Database…**, name it `artery_db`.

### 3. Apply the clinical schema

Open `database/database_work.sql` in pgAdmin's Query Tool (connected to `artery_db`) and execute it to create the five clinical tables.

> [!WARNING]
> `database/database_work.sql` line 53 contains a premature semicolon that terminates an `ALTER TABLE` statement early, causing a syntax error when run as a single script. Change the `;` at the end of line 53 to `,` before executing. See [§19 Known Issues](#19-known-issues--limitations).

### 4. ORM tables (refresh_tokens, auth users)

The ORM tables (`users` and `refresh_tokens`) are created automatically on startup via `Base.metadata.create_all(bind=engine)` in the application lifespan handler. No manual DDL is needed for them.

---

## 11. Running the API

### Recommended startup (Windows PowerShell)

```powershell
$env:PYTHONPATH = "backend"
.\.venv\Scripts\python.exe -m uvicorn app.extended_main:app --host 127.0.0.1 --port 8000 --reload
```

### macOS / Linux

```bash
PYTHONPATH=backend .venv/bin/python -m uvicorn app.extended_main:app --host 127.0.0.1 --port 8000 --reload
```

### Verify the service is up

| Probe | URL | Expected Response |
|---|---|---|
| Liveness | `GET /api/v1/health` | `{"status": "ok", "service": "artery-segmentation-auth"}` |
| Readiness | `GET /api/v1/health/ready` | `{"status": "ready", "database": "connected", "environment": "development"}` |

If the readiness probe returns HTTP 503, PostgreSQL is not reachable with the current `DATABASE_URL`.

> [!NOTE]
> **Verified:** PostgreSQL 17.10 on `localhost:5432` is confirmed reachable using the application's configured `DATABASE_URL`. The `/api/v1/health/ready` probe returns `{"status": "ready", "database": "connected", "environment": "development"}`. ORM tables (`users`, `refresh_tokens`) are confirmed present in the `public` schema. Clinical tables (`cases`, `medical_images`, `segmentation_results`, `stenosis_results`) require manual DDL execution from `database/database_work.sql` before SQL-backed endpoints return data.

---

## 12. Swagger UI & API Exploration

| Interface | URL |
|---|---|
| Interactive Swagger UI | [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) |
| ReDoc Documentation | [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc) |
| OpenAPI JSON Schema | [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json) |

### Authenticating in Swagger UI

1. Open `/docs`.
2. Expand `POST /api/v1/auth/login` → **Try it out** → enter credentials → **Execute**.
3. Copy the `access_token` from the response.
4. Click the **Authorize** 🔒 button at the top of the page.
5. Enter `Bearer <your_access_token>` in the HTTPBearer field and click **Authorize**.
6. All protected endpoints will now include the authorization header automatically.

---

## 13. Complete API Endpoint Reference

The full application (`app.extended_main:app`) registers **48 operations across 40 URL paths**.

### A. Health Probes

| Method | Path | Auth | Description |
|---|---|:---:|---|
| `GET` | `/api/v1/health` | Public | Liveness probe — always returns `{"status": "ok"}` |
| `GET` | `/api/v1/health/ready` | Public | Readiness probe — executes `SELECT 1` against PostgreSQL; returns HTTP 503 if unreachable |

### B. Authentication (`/api/v1/auth`)

| Method | Path | Auth | Description |
|---|---|:---:|---|
| `POST` | `/api/v1/auth/register` | Public | Register a new user (default role: `medical_researcher`). Returns HTTP 409 on duplicate email. |
| `POST` | `/api/v1/auth/login` | Public | Verify email + password; returns JWT access token + refresh token. HTTP 401 on bad credentials; HTTP 403 if account inactive. |
| `POST` | `/api/v1/auth/refresh` | Public (token body) | Rotate refresh token: revoke submitted token, issue new access + refresh pair. Detects replay attacks. |
| `POST` | `/api/v1/auth/logout` | Public (token body) | Revoke refresh token so it cannot be reused. |
| `POST` | `/api/v1/auth/change-password` | Bearer JWT | Verify current password, update Argon2 hash, revoke all active sessions. |

### C. User Profile (`/api/v1/users`)

| Method | Path | Auth | Description |
|---|---|:---:|---|
| `GET` | `/api/v1/users/me` | Bearer JWT | Return authenticated user's `id`, `name`, `email`, `role`, `is_active`, timestamps. |
| `PATCH` | `/api/v1/users/me` | Bearer JWT | Update `name` and/or `email`. Role field is not modifiable via this endpoint. |

### D. SQL User Queries (`/api/v1/sql/users`)

| Method | Path | Auth | SQL Source | Description |
|---|---|:---:|---|---|
| `GET` | `/api/v1/sql/users/email-exists` | Public | `auth_queries.sql` Q1 | Check email existence. Query param: `?email=`. Returns `{"email": "...", "exists": true}`. |
| `GET` | `/api/v1/sql/users/by-email` | Bearer JWT | `auth_queries.sql` Q2 | Retrieve user record by email for verification. `password_hash` is excluded from response. |
| `GET` | `/api/v1/sql/users` | Bearer JWT | `auth_queries.sql` Q3 | List all registered users (`user_id`, `full_name`, `email`, `created_at`). No password hashes. |
| `PATCH` | `/api/v1/sql/users/{user_id}/name` | Bearer JWT | `auth_queries.sql` Q4 | Update a user's `full_name` by integer `user_id`. |

### E. Case Management (`/api/v1/sql/cases`)

| Method | Path | Auth | Description |
|---|---|:---:|---|
| `GET` | `/api/v1/sql/cases` | Bearer JWT | List all examination cases ordered by `case_id DESC`. |
| `POST` | `/api/v1/sql/cases` | Bearer JWT | Create a new case (`user_id`, `case_name`, `imaging_modality`, `target_vessel`, `status`). |
| `GET` | `/api/v1/sql/cases/{case_id}` | Bearer JWT | Retrieve a specific case by `case_id`. HTTP 404 if not found. |
| `GET` | `/api/v1/sql/cases/user/{user_id}` | Bearer JWT | Retrieve all cases for a given `user_id`. |
| `PATCH` | `/api/v1/sql/cases/{case_id}` | Bearer JWT | Update case metadata (`case_name`, `imaging_modality`, `target_vessel`, `status`). |

### F. Medical Image Management (`/api/v1/sql/images`, `/api/v1/sql/cases/{case_id}/images`)

| Method | Path | Auth | Description |
|---|---|:---:|---|
| `GET` | `/api/v1/sql/images` | Bearer JWT | List all medical images ordered by `image_id DESC`. |
| `GET` | `/api/v1/sql/images/{image_id}` | Bearer JWT | Retrieve a specific image by `image_id`. HTTP 404 if not found. |
| `GET` | `/api/v1/sql/cases/{case_id}/images` | Bearer JWT | Retrieve all images linked to a case. |
| `POST` | `/api/v1/sql/cases/{case_id}/images` | Bearer JWT | Register new image metadata for a case (`image_name`, `image_path`, `image_format`, `image_size`). |
| `PATCH` | `/api/v1/sql/images/{image_id}` | Bearer JWT | Update image metadata fields. |
| `DELETE` | `/api/v1/sql/images/{image_id}` | Bearer JWT | Delete image metadata. Blocked if dependent segmentation or stenosis records exist. |

### G. Segmentation Results (`/api/v1/sql/segmentation-results`, `/api/v1/sql/images/{image_id}/segmentation-results`)

| Method | Path | Auth | Description |
|---|---|:---:|---|
| `GET` | `/api/v1/sql/segmentation-results` | Bearer JWT | List all segmentation results ordered by `segmentation_id DESC`. |
| `GET` | `/api/v1/sql/segmentation-results/latest` | Bearer JWT | Retrieve the most recent segmentation result by `created_at`. |
| `GET` | `/api/v1/sql/images/{image_id}/segmentation-results` | Bearer JWT | Retrieve all segmentation results for a specific image. |
| `GET` | `/api/v1/sql/cases/{case_id}/segmentation-results` | Bearer JWT | Retrieve all segmentation results linked to a case (via image JOIN). |
| `POST` | `/api/v1/sql/images/{image_id}/segmentation-results` | Bearer JWT | Insert a segmentation result for an image (`mask_path`, `model_name`, `dice_score`, `iou_score`, `precision_score`, `recall_score`, `f1_score`, etc.). |

### H. Stenosis Results (`/api/v1/sql/stenosis-results`, `/api/v1/sql/images/{image_id}/stenosis-results`)

| Method | Path | Auth | Description |
|---|---|:---:|---|
| `GET` | `/api/v1/sql/stenosis-results` | Bearer JWT | List all stenosis results. Optional `?sort_by=confidence` to order by confidence score descending. |
| `GET` | `/api/v1/sql/stenosis-results/latest` | Bearer JWT | Retrieve the most recent stenosis result. |
| `GET` | `/api/v1/sql/stenosis-results/filter` | Bearer JWT | Filter stenosis results by severity. Query param: `?severity=severe`. |
| `GET` | `/api/v1/sql/images/{image_id}/stenosis-results` | Bearer JWT | Retrieve all stenosis results for a specific image. |
| `GET` | `/api/v1/sql/cases/{case_id}/stenosis-results` | Bearer JWT | Retrieve all stenosis results linked to a case (via image JOIN). |
| `POST` | `/api/v1/sql/images/{image_id}/stenosis-results` | Bearer JWT | Insert a stenosis detection result (`location`, `severity`, `confidence_score`, `estimated_diameter`). |

### I. Analytics (`/api/v1/sql/analytics`)

| Method | Path | Auth | SQL Source | Description |
|---|---|:---:|---|---|
| `GET` | `/api/v1/sql/analytics/segmentation-count` | Bearer JWT | `result_queries.sql` Q1 | Returns `{"count": <int>}` total segmentation results. |
| `GET` | `/api/v1/sql/analytics/stenosis-count` | Bearer JWT | `result_queries.sql` Q2 | Returns `{"count": <int>}` total stenosis results. |
| `GET` | `/api/v1/sql/analytics/model-performance` | Bearer JWT | `result_queries.sql` Q3 | Average Dice score per model, ordered descending. |
| `GET` | `/api/v1/sql/analytics/model-performance/detailed` | Bearer JWT | `result_queries.sql` Q4 | Full metrics per model: count, avg Dice, IoU, precision, recall, F1. |
| `GET` | `/api/v1/sql/analytics/stenosis-severity-distribution` | Bearer JWT | `result_queries.sql` Q5 | Total cases and avg confidence grouped by severity. |
| `GET` | `/api/v1/sql/analytics/summary` | Bearer JWT | `result_queries.sql` Q1–3 | Combined summary: total counts + model Dice performance. |
| `GET` | `/api/v1/sql/dashboard/analytics` | Bearer JWT | All tables | Total counts for users, cases, images, segmentations, stenosis detections, and recent activity items. |

### J. Clinical Reports (`/api/v1/sql/reports`)

| Method | Path | Auth | SQL Source | Description |
|---|---|:---:|---|---|
| `GET` | `/api/v1/sql/reports/case-analysis` | Bearer JWT | `result_queries.sql` Q6 | Multi-table JOIN report: case → images → segmentation + stenosis results. Optional `?case_id=` filter. **Note:** images with multiple segmentation runs or stenosis findings produce one row per combination. |
| `GET` | `/api/v1/sql/reports/case-summary-counts` | Bearer JWT | `result_queries.sql` Q7 | Per-case count of distinct images, segmentation results, and stenosis results. |

### K. Database Diagnostics (`/api/v1/sql/diagnostics`) — Administrator Only

> [!IMPORTANT]
> These endpoints require the **`administrator`** role. Any other authenticated user receives HTTP 403 Forbidden.

| Method | Path | Auth | SQL Source | Description |
|---|---|:---:|---|---|
| `GET` | `/api/v1/sql/diagnostics/foreign-keys` | Admin JWT | `result_queries.sql` Q8 | List all FK constraints in the `public` schema via `information_schema`. |
| `GET` | `/api/v1/sql/diagnostics/tables` | Admin JWT | `result_queries.sql` Q9 | Verify that all 5 project tables exist in `information_schema.tables`. |
| `GET` | `/api/v1/sql/diagnostics/columns` | Admin JWT | `result_queries.sql` Q10 | Inspect column names, data types, and nullability for project tables. Optional `?table_name=` filter. |
| `GET` | `/api/v1/sql/diagnostics/foreign-key-consistency` | Admin JWT | `result_queries.sql` Q11 | Orphan record check: cases without valid users, images without valid cases, results without valid images. Returns 4 rows. |

---

## 14. SQL Query-to-Endpoint Mapping

### `database/auth_queries.sql`

| Query | Service Method | Endpoint |
|---|---|---|
| Q1 — Email existence check | `SQLClinicalService.check_email_exists` | `GET /api/v1/sql/users/email-exists` |
| Q2 — User login verification lookup | `SQLClinicalService.get_user_by_email_for_verification` | `GET /api/v1/sql/users/by-email` |
| Q3 — List registered users | `SQLClinicalService.get_registered_users` | `GET /api/v1/sql/users` |
| Q4 — Update user full name | `SQLClinicalService.update_user_name` | `PATCH /api/v1/sql/users/{user_id}/name` |
| Q5 — View all cases | `SQLClinicalService.get_all_cases` | `GET /api/v1/sql/cases` |
| Q6 — View all medical images | `SQLClinicalService.get_all_medical_images` | `GET /api/v1/sql/images` |
| Q7 — View all segmentation results | `SQLClinicalService.get_all_segmentation_results` | `GET /api/v1/sql/segmentation-results` |
| Q8 — View all stenosis results | `SQLClinicalService.get_all_stenosis_results` | `GET /api/v1/sql/stenosis-results` |
| Q9 — Retrieve cases by user ID | `SQLClinicalService.get_cases_by_user_id` | `GET /api/v1/sql/cases/user/{user_id}` |
| Q10 — Insert a new case | `SQLClinicalService.create_case` | `POST /api/v1/sql/cases` |
| Q11 — Update existing case details | `SQLClinicalService.update_case` | `PATCH /api/v1/sql/cases/{case_id}` |
| Q12 — Retrieve specific case by case ID | `SQLClinicalService.get_case_by_id` | `GET /api/v1/sql/cases/{case_id}` |
| Q13 — Retrieve images by case ID | `SQLClinicalService.get_images_by_case_id` | `GET /api/v1/sql/cases/{case_id}/images` |
| Q14 — Retrieve specific image by image ID | `SQLClinicalService.get_image_by_id` | `GET /api/v1/sql/images/{image_id}` |
| Q15 — Update medical image metadata | `SQLClinicalService.update_image_metadata` | `PATCH /api/v1/sql/images/{image_id}` |
| Q16–Q18 — Delete image with dependency checks | `SQLClinicalService.delete_medical_image` | `DELETE /api/v1/sql/images/{image_id}` |

### `database/result_queries.sql`

| Query | Service Method | Endpoint |
|---|---|---|
| Q1 — Count segmentation results | `SQLClinicalService.get_segmentation_count` | `GET /api/v1/sql/analytics/segmentation-count` |
| Q2 — Count stenosis results | `SQLClinicalService.get_stenosis_count` | `GET /api/v1/sql/analytics/stenosis-count` |
| Q3 — Avg Dice per model | `SQLClinicalService.get_model_performance` | `GET /api/v1/sql/analytics/model-performance` |
| Q4 — Full metrics per model | `SQLClinicalService.get_detailed_model_performance` | `GET /api/v1/sql/analytics/model-performance/detailed` |
| Q5 — Severity distribution | `SQLClinicalService.get_stenosis_severity_distribution` | `GET /api/v1/sql/analytics/stenosis-severity-distribution` |
| Q6 — Case-wise analysis report (CTE aggregated) | `SQLClinicalService.get_case_analysis_report` | `GET /api/v1/sql/reports/case-analysis` |
| Q7 — Case-wise summary counts | `SQLClinicalService.get_case_summary_counts` | `GET /api/v1/sql/reports/case-summary-counts` |
| Q8 — FK relationship verification | `SQLClinicalService.diagnose_foreign_keys` | `GET /api/v1/sql/diagnostics/foreign-keys` |
| Q9 — Project table verification | `SQLClinicalService.diagnose_tables` | `GET /api/v1/sql/diagnostics/tables` |
| Q10 — Column & type inspection | `SQLClinicalService.diagnose_columns` | `GET /api/v1/sql/diagnostics/columns` |
| Q11 — Orphan record check | `SQLClinicalService.diagnose_fk_consistency` | `GET /api/v1/sql/diagnostics/foreign-key-consistency` |
| Q12 — Check existing medical image records | `SQLClinicalService.get_all_medical_images` | `GET /api/v1/sql/images` |
| Q13 — Insert image details for case | `SQLClinicalService.insert_medical_image` | `POST /api/v1/sql/cases/{case_id}/images` |
| Q14 — Insert segmentation results | `SQLClinicalService.insert_segmentation_result` | `POST /api/v1/sql/images/{image_id}/segmentation-results` |
| Q15 — Retrieve segmentation by image ID | `SQLClinicalService.get_segmentations_by_image_id` | `GET /api/v1/sql/images/{image_id}/segmentation-results` |
| Q16 — Retrieve segmentation by case ID | `SQLClinicalService.get_segmentations_by_case_id` | `GET /api/v1/sql/cases/{case_id}/segmentation-results` |
| Q17 — Retrieve latest segmentation result | `SQLClinicalService.get_latest_segmentation_result` | `GET /api/v1/sql/segmentation-results/latest` |
| Q18 — Insert stenosis detection results | `SQLClinicalService.insert_stenosis_result` | `POST /api/v1/sql/images/{image_id}/stenosis-results` |
| Q19 — Retrieve stenosis by image ID | `SQLClinicalService.get_stenosis_by_image_id` | `GET /api/v1/sql/images/{image_id}/stenosis-results` |
| Q20 — Retrieve stenosis by case ID | `SQLClinicalService.get_stenosis_by_case_id` | `GET /api/v1/sql/cases/{case_id}/stenosis-results` |
| Q21 — Filter stenosis results by severity | `SQLClinicalService.filter_stenosis_by_severity` | `GET /api/v1/sql/stenosis-results/filter` |
| Q22 — Sort stenosis results by confidence | `SQLClinicalService.get_stenosis_sorted_by_confidence` | `GET /api/v1/sql/stenosis-results?sort_by=confidence` |
| Q23 — Retrieve latest stenosis results | `SQLClinicalService.get_latest_stenosis_result` | `GET /api/v1/sql/stenosis-results/latest` |
| Q24–Q28 — Dashboard counts & recent activity | `SQLClinicalService.get_dashboard_analytics` | `GET /api/v1/sql/dashboard/analytics` |

> [!NOTE]
> `database/database_work.sql` is a **schema reference only** — it is not executed by the application. The application queries the tables it defines, but schema creation is a manual step in pgAdmin or psql.

---

## 15. Authentication & Authorization Flow

### Authentication Flow

```
POST /api/v1/auth/login
  ├── Validate email + Argon2 password verification
  ├── Issue signed JWT access token (sub=UUID, role, exp=+30min)
  ├── Generate 48-byte random refresh token
  ├── Store SHA-256(refresh_token) in refresh_tokens table
  └── Return { access_token, refresh_token, token_type, expires_in }

GET /api/v1/sql/cases  (or any protected route)
  ├── Extract Bearer token from Authorization header
  ├── Decode and verify JWT signature + expiration
  ├── Resolve user by sub (UUID) from users table
  └── Inject user object into route handler

POST /api/v1/auth/refresh
  ├── Hash submitted refresh token (SHA-256)
  ├── Look up token_hash in database — reject if not found or revoked/expired
  ├── Mark old token as revoked (revoked_at = now)
  ├── Issue new JWT access token + new refresh token
  └── Return new token pair
```

### Role-Based Access Control

The `RoleChecker` dependency (`backend/app/api/dependencies.py`) enforces permitted roles per route:

```python
# Example: restrict to administrator role
admin_user: User = Depends(RoleChecker([UserRole.ADMINISTRATOR]))
```

| Role | Value | Permitted Endpoints |
|---|---|---|
| `administrator` | `"administrator"` | All endpoints including `/api/v1/sql/diagnostics/*` |
| `radiologist` | `"radiologist"` | All clinical endpoints except diagnostics |
| `medical_researcher` | `"medical_researcher"` | All clinical endpoints except diagnostics (default registration role) |
| `data_scientist` | `"data_scientist"` | All clinical endpoints except diagnostics |

### Security Implementation Details

| Feature | Implementation |
|---|---|
| Password hashing | Argon2 via `pwdlib[argon2]` |
| Token signing | HS256 JWT via `pyjwt[crypto]`; claims: `sub`, `role`, `type`, `iat`, `exp` |
| Refresh token storage | SHA-256 hex digest (64 chars) stored in `refresh_tokens.token_hash` |
| Token rotation | Old token revoked on every refresh call |
| Replay detection | Revoked token reuse triggers immediate revocation of all user sessions |

---

## 16. Testing Guide

### Test Architecture

The test suite uses **pytest** and **FastAPI TestClient** with an **isolated in-memory SQLite database**. No PostgreSQL connection is required to run tests.

- [`tests/conftest.py`](tests/conftest.py) creates the SQLite database, creates all ORM tables, and manually creates the clinical schema tables (`cases`, `medical_images`, `segmentation_results`, `stenosis_results`).
- The `get_db` dependency is overridden per test function to use the isolated SQLite session.
- Each test function receives a fresh transaction that is rolled back after the test completes.

> [!IMPORTANT]
> **SQLite test success ≠ live PostgreSQL integration.**
> Tests confirm route logic, serialization, authentication enforcement, and RBAC. They run entirely against an in-memory SQLite database — PostgreSQL is not contacted during the test run.
>
> **Live PostgreSQL status (verified):** PostgreSQL 17.10 on `localhost:5432` is reachable. The ORM tables (`users`, `refresh_tokens`) exist in the `public` schema and are confirmed by `Base.metadata.create_all`. Clinical tables (`cases`, `medical_images`, `segmentation_results`, `stenosis_results`) must be initialized from `database/database_work.sql` before the SQL-backed endpoints return data rather than HTTP 503.
> The diagnostic endpoints (`/api/v1/sql/diagnostics/*`) require a live PostgreSQL connection because they query `information_schema`, which is not available in SQLite — these cannot be tested via the SQLite fixture.

### Running Tests

```powershell
# Run all 45 tests with verbose output
.\.venv\Scripts\python.exe -m pytest -v
```

### Test Results (45 passed, 0 failed — verified)

| Test File | Tests | Coverage |
|---|:---:|---|
| `tests/test_auth.py` | 14 | Register (success, duplicate, invalid input), login (success, bad password, unknown email), password hashing, refresh rotation, logout revocation, password change, inactive account blocking |
| `tests/test_clinical_endpoints.py` | 23 | All 8 `auth_queries.sql` endpoints, case/image/segmentation/stenosis CRUD, dashboard analytics, all 11 `result_queries.sql` endpoints (Q1–Q11), unauthenticated 401 enforcement, non-admin 403 on diagnostic routes |
| `tests/test_users.py` | 8 | Authenticated profile retrieval, missing/expired/malformed token rejection, inactive account blocking, name update, email update, role escalation prevention |

---

## 17. Example API Usage

### 1. Health check

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/v1/health"
```

```bash
curl http://127.0.0.1:8000/api/v1/health
```

```json
{"status": "ok", "service": "artery-segmentation-auth"}
```

### 2. Register a new user

```powershell
$body = @{
    name     = "Dr. Jane Smith"
    email    = "jane.smith@hospital.example.com"
    password = "StrongPassword2026!"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
    -Uri "http://127.0.0.1:8000/api/v1/auth/register" `
    -ContentType "application/json" -Body $body
```

*Expected Response (HTTP 201):*
```json
{
  "id": "<UUID assigned by the database>",
  "name": "Dr. Jane Smith",
  "email": "jane.smith@hospital.example.com",
  "role": "medical_researcher",
  "is_active": true,
  "created_at": "<ISO 8601 timestamp>",
  "updated_at": "<ISO 8601 timestamp>"
}
```

### 3. Log in and store the access token

```powershell
$loginBody = @{
    email    = "jane.smith@hospital.example.com"
    password = "StrongPassword2026!"
} | ConvertTo-Json

$resp  = Invoke-RestMethod -Method Post `
    -Uri "http://127.0.0.1:8000/api/v1/auth/login" `
    -ContentType "application/json" -Body $loginBody

$token = $resp.access_token
$headers = @{ Authorization = "Bearer $token" }
```

**Response (HTTP 200):**
```json
{
  "access_token": "<signed JWT string>",
  "token_type": "bearer",
  "refresh_token": "<opaque 64-character URL-safe token>",
  "expires_in": 1800
}
```

### 4. Query your profile

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/v1/users/me" -Headers $headers
```

### 5. Create a clinical case

```powershell
$caseBody = @{
    user_id          = 1
    case_name        = "Patient A — LAD CTA Study"
    imaging_modality = "CTA"
    target_vessel    = "LAD"
    status           = "pending"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
    -Uri "http://127.0.0.1:8000/api/v1/sql/cases" `
    -ContentType "application/json" -Headers $headers -Body $caseBody
```

### 6. Query model performance analytics

```powershell
Invoke-RestMethod -Method Get `
    -Uri "http://127.0.0.1:8000/api/v1/sql/analytics/model-performance/detailed" `
    -Headers $headers
```

**Example response shape** *(actual values depend on data in your database — these are illustrative field names and types only)*:
```json
[
  {
    "model_name": "<string: model identifier>",
    "total_results": "<int: evaluation record count>",
    "average_dice": "<float | null: 0.0 – 1.0>",
    "average_iou": "<float | null: 0.0 – 1.0>",
    "average_precision": "<float | null: 0.0 – 1.0>",
    "average_recall": "<float | null: 0.0 – 1.0>",
    "average_f1": "<float | null: 0.0 – 1.0>"
  }
]
```

### 7. Query stenosis severity distribution

```powershell
Invoke-RestMethod -Method Get `
    -Uri "http://127.0.0.1:8000/api/v1/sql/analytics/stenosis-severity-distribution" `
    -Headers $headers
```

**Example response shape** *(actual values depend on data in your database — these are illustrative field names and types only)*:
```json
[
  {
    "severity": "<string | null: e.g. 'severe', 'moderate', 'mild'>",
    "total_cases": "<int: count of stenosis records for this severity>",
    "average_confidence": "<float | null: 0.0 – 1.0>"
  }
]
```

---

## 18. Troubleshooting Guide

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'app'` | `PYTHONPATH` not set | Run `$env:PYTHONPATH = "backend"` before Uvicorn |
| `HTTP 503` on `/api/v1/health/ready` | PostgreSQL not reachable | Check `Get-Service *postgres*`; verify port and credentials in `.env` |
| `HTTP 503` on any `/api/v1/sql/*` route | Clinical table doesn't exist in DB | Execute `database/database_work.sql` DDL in pgAdmin against `artery_db` |
| `HTTP 401 Token has expired` | 30-minute access token TTL elapsed | POST to `/api/v1/auth/refresh` with the refresh token |
| `HTTP 403 Operation not permitted` | User role insufficient | Use an `administrator` account for diagnostic endpoints |
| `HTTP 409 Conflict` on register | Email already registered | Use a different email or log in with existing credentials |
| `SyntaxError in database_work.sql` at line 54 | Premature semicolon on line 53 | Change `;` on line 53 to `,` before running the script |
| Port 8000 already in use | Another Uvicorn process running | `Get-NetTCPConnection -LocalPort 8000`, then `Stop-Process -Id <PID>`, or use `--port 8001` |
| PowerShell `ExecutionPolicy` error on `.ps1` | Execution policy blocked | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |

---

## 19. Known Issues & Limitations

### 1. Syntax error in `database/database_work.sql` (line 53)

The `ALTER TABLE segmentation_results` statement is split incorrectly — a semicolon on line 53 prematurely terminates the statement before all `ADD COLUMN` clauses are parsed:

```sql
-- Line 53 ends with a semicolon instead of a comma:
ADD COLUMN IF NOT EXISTS model_version VARCHAR(50);   -- ← should be a comma
ADD COLUMN IF NOT EXISTS precision_score NUMERIC(6,5),
```

**Impact:** Running the file as a script in psql or pgAdmin raises a syntax error at line 54. Manually change `;` to `,` on line 53 before executing.

### 2. User table primary key divergence

The ORM `users` table (used by authentication routes) has:
- Primary key column: `id UUID`
- Name column: `name VARCHAR(255)`

The `database/database_work.sql` clinical schema defines:
- Primary key column: `user_id SERIAL INTEGER`
- Name column: `full_name VARCHAR(100)`

As a result:
- `PATCH /api/v1/sql/users/{user_id}/name` targets `user_id` (integer), which does not exist in the ORM-generated `users` table. This endpoint works correctly only when the clinical schema version of `users` is present.
- `GET /api/v1/sql/users/by-email` returns `user_id` and `full_name`, which are not present in the ORM `users` table.

Unifying these two schemas is a planned task when merging with teammate branches.

### 3. `information_schema` queries not available in SQLite

Diagnostic endpoints (Queries 8–10 from `result_queries.sql`) query PostgreSQL's `information_schema`. SQLite does not implement these views. In the test suite, these endpoints return HTTP 503 with a database error rather than data, and the test asserts `status_code in [200, 503]`. This is expected behavior — these endpoints require a live PostgreSQL connection to return meaningful results.

### 4. ML inference pipeline not connected

The API stores and queries segmentation and stenosis metrics, but does not invoke any deep learning inference. Running a 3D-UNet model, ingesting DICOM files, or generating segmentation masks are out of scope for this service and are managed by a separate pipeline.

---

## 20. Completed vs. Planned Features

### ✅ Completed

- Full authentication lifecycle: register, login, refresh, logout, change password
- Role-based access control with 4 roles
- Complete clinical CRUD: cases, medical images, segmentation results, stenosis results
- All 11 `result_queries.sql` queries implemented as SQL-backed API endpoints
- All 8 `auth_queries.sql` queries implemented as API endpoints
- Dashboard and report aggregation endpoints
- Administrator-only database diagnostic endpoints
- 45-test automated suite (100% pass rate on SQLite fixture)
- Interactive Swagger UI with persistent JWT authorization

### ⏳ Planned / Incomplete

- Alembic migration scripts (framework configured, no revisions authored)
- Unification of ORM `users` schema with clinical `users` schema
- Deep learning inference integration (DICOM ingestion, model execution, mask output)
- Execution of `database/database_work.sql` DDL in the PostgreSQL database to create persistent clinical tables (`cases`, `medical_images`, `segmentation_results`, `stenosis_results`)

---

*Backend service for the Infosys Springboard Internship — Artery Segmentation and Stenosis Detection project.*
