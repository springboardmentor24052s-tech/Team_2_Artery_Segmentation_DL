# Artery Segmentation and Stenosis Detection Using Advanced Deep Learning Techniques
## Backend REST API & Database Documentation

A modular, production-ready backend service built with **FastAPI**, **SQLAlchemy 2.x**, **PostgreSQL** (via `psycopg 3`), **Pydantic 2**, and **Argon2** password hashing. It provides authentication, role-based access control (RBAC), and SQL-backed clinical endpoints for querying medical cases, imaging metadata, artery segmentation metrics, stenosis detection results, and model performance analytics.

---

## Table of Contents
1. [Project Overview & Architecture](#1-project-overview--architecture)
2. [Project Structure](#2-project-structure)
3. [Prerequisites & Installation](#3-prerequisites--installation)
4. [Environment Configuration](#4-environment-configuration)
5. [Database Setup & Schema Reference](#5-database-setup--schema-reference)
6. [Running the API](#6-running-the-api)
7. [Swagger UI & API Exploration](#7-swagger-ui--api-exploration)
8. [Complete API Endpoint Reference](#8-complete-api-endpoint-reference)
9. [Detailed SQL Query Reference](#9-detailed-sql-query-reference)
10. [SQL Query-to-Endpoint Mapping](#10-sql-query-to-endpoint-mapping)
11. [Authentication & Authorization Flow](#11-authentication--authorization-flow)
12. [Testing Guide](#12-testing-guide)
13. [Example API Usage (PowerShell & cURL)](#13-example-api-usage-powershell--curl)
14. [Troubleshooting Guide](#14-troubleshooting-guide)
15. [Known Issues & Limitations](#15-known-issues--limitations)

---

## 1. Project Overview & Architecture

### Purpose
The **Artery Segmentation and Stenosis Detection** backend serves as the API foundation for automated coronary and peripheral artery image analysis. It manages user authentication, authorization, clinical examination cases, imaging metadata, segmentation mask evaluations (Dice, IoU, centerline, vessel diameter), stenosis severity assessments, and deep learning model benchmarking.

### Implemented Architecture
The backend is structured into clear layers:
- **Presentation Layer (FastAPI Routers):** Mounts versioned routes (`/api/v1/auth`, `/api/v1/users`, `/api/v1/sql`, `/api/v1/health`) with Pydantic request/response validation.
- **Security & Authorization Layer:** Argon2 password hashing via `pwdlib`, signed JWT access tokens with short TTLs (30m), persistent SHA-256 hashed refresh tokens with rotation and revocation, and role-based access dependencies (`RoleChecker`).
- **Service Layer:** Decoupled business logic (`AuthService`, `UserService`, `SQLClinicalService`) isolating route handlers from raw SQL and ORM queries.
- **Persistence Layer:** SQLAlchemy 2.x with connection pooling (`get_db` dependency) and parameterized SQL execution via the PostgreSQL `psycopg 3` driver.

```text
               +-------------------------------------------+
               |     Client / Frontend / Swagger UI        |
               +-------------------------------------------+
                                     |
                                     | (HTTP REST + Bearer JWT)
                                     v
               +-------------------------------------------+
               |        FastAPI Application Entry          |
               |        (backend/app/extended_main.py)     |
               +-------------------------------------------+
                                     |
                     +---------------+---------------+
                     |                               |
                     v                               v
    +----------------------------------+   +----------------------------------+
    | Authentication & User Routers    |   | SQL-Backed Clinical Routers      |
    | (/api/v1/auth, /api/v1/users)    |   | (/api/v1/sql/*)                  |
    +----------------------------------+   +----------------------------------+
                     |                               |
                     v                               v
    +----------------------------------+   +----------------------------------+
    | AuthService & UserService        |   | SQLClinicalService               |
    | (SQLAlchemy ORM + Argon2/JWT)    |   | (Parameterized SQL Queries)      |
    +----------------------------------+   +----------------------------------+
                     |                               |
                     +---------------+---------------+
                                     |
                                     v
               +-------------------------------------------+
               |      SQLAlchemy Session (app.db.session)   |
               |      Engine: psycopg 3 connection pool    |
               +-------------------------------------------+
                                     |
                                     v
               +-------------------------------------------+
               |          PostgreSQL Database              |
               | (users, cases, medical_images, results)   |
               +-------------------------------------------+
```

---

## 2. Project Structure

```text
Artery project/
├── .env                              # Local environment variables (Git-ignored, never committed)
├── .env.example                      # Template with safe configuration placeholders
├── .gitignore                        # Git exclusion rules (.env, .venv, pycache, caches)
├── LICENSE                           # Project open-source license
├── README.md                         # This documentation file
├── requirements.txt                  # Python dependencies pinned with versions
│
├── auth_queries.sql                  # Canonical SQL queries for user checks, cases, images, & results
├── database_work.sql                 # DDL schema definitions and ALTER TABLE column definitions
├── result_queries.sql                # Analytical SQL queries (counts, Dice score averages)
│
├── backend/
│   ├── alembic.ini                   # Alembic database migration configuration
│   ├── alembic/                      # Alembic migration environment
│   │   ├── env.py                    # Migration runtime script
│   │   └── script.py.mako            # Migration revision template
│   │
│   └── app/                          # Core application package
│       ├── __init__.py
│       ├── main.py                   # Base application (Auth, Users, /health probes)
│       ├── extended_main.py          # RECOMMENDED entry point (Auth + SQL-backed clinical APIs)
│       │
│       ├── api/                      # Routing & Security dependencies
│       │   ├── __init__.py
│       │   ├── dependencies.py       # JWT extraction, user validation, RoleChecker RBAC
│       │   └── v1/
│       │       ├── __init__.py       # Base v1 router aggregating auth and users
│       │       ├── auth.py           # 5 auth routes (register, login, refresh, logout, change-password)
│       │       ├── users.py          # 2 user routes (GET /me, PATCH /me)
│       │       └── sql_clinical.py   # 11 SQL-backed clinical & analytics routes
│       │
│       ├── core/                     # Configuration and cryptographic security
│       │   ├── config.py             # Pydantic BaseSettings loading .env with robust path resolution
│       │   └── security.py           # Argon2 password hashing, JWT encode/decode, token hashing
│       │
│       ├── db/                       # Database engine and sessions
│       │   ├── base.py               # SQLAlchemy DeclarativeBase
│       │   └── session.py            # PostgreSQL engine (psycopg 3) and get_db dependency
│       │
│       ├── models/                   # SQLAlchemy ORM models
│       │   ├── __init__.py
│       │   ├── user.py               # User table model (UUID PK, role, active status, timestamps)
│       │   └── refresh_token.py      # RefreshToken model (SHA-256 hashed, revocable sessions)
│       │
│       ├── schemas/                  # Pydantic validation schemas
│       │   ├── auth.py               # LoginRequest, TokenResponse, PasswordChangeRequest
│       │   ├── user.py               # UserRegister, UserUpdate, UserResponse
│       │   └── clinical_schemas.py   # SQL query request and response schemas
│       │
│       └── services/                 # Business logic and query execution
│           ├── auth_service.py       # User registration, token rotation, password change
│           ├── user_service.py       # Profile fetching and email uniqueness validation
│           └── sql_clinical_service.py # Parameterized execution of queries from SQL files
│
└── tests/                            # Automated pytest suite (Runs against in-memory SQLite)
    ├── conftest.py                   # TestClient fixtures, database session overrides, user fixtures
    ├── test_auth.py                  # 14 tests covering authentication, tokens, and password changes
    ├── test_users.py                 # 8 tests covering profiles, token validation, and RBAC
    └── test_clinical_endpoints.py    # 13 tests covering SQL-backed clinical & analytics queries
```

### Application Entry Points Explained
- **`app.extended_main:app` (Recommended):** Starts the complete API exposing **all 20 endpoints**: Authentication, Users, Health probes, and all 11 SQL-backed Clinical and Analytics endpoints.
- **`app.main:app` (Base Auth Only):** Exposes only the 7 base authentication, user profile, and health endpoints (does not mount the `/api/v1/sql` router).

---

## 3. Prerequisites & Installation

### Prerequisites
- **Operating System:** Windows 10/11 (PowerShell 5.1 or PowerShell 7+) or Linux/macOS
- **Python:** Python 3.10 to 3.12 (Python 3.12.10 recommended and verified)
- **PostgreSQL Database:** PostgreSQL 15, 16, 17, or 18 running locally or accessible over network

### Step 1: Clone / Navigate to Project Directory
```powershell
cd "c:\Users\Rohan R\Documents\INFOSYS SPRINGBOARD INTERNSHIP 7\Artery project"
```

### Step 2: Virtual Environment Setup
If `.venv` already exists in the project root, activate it directly. Otherwise, create it first:
```powershell
# Create virtual environment (if not already created)
python -m venv .venv

# Activate on Windows PowerShell
.\.venv\Scripts\Activate.ps1

# (Alternative for macOS/Linux bash)
# source .venv/bin/activate
```

> [!NOTE]
> If PowerShell blocks script execution with an `ExecutionPolicy` error, run:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### Step 3: Install Required Dependencies
All dependencies are specified in [`requirements.txt`](requirements.txt):
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Key packages installed:
- `fastapi` & `uvicorn[standard]`: High-performance ASGI web framework and server
- `sqlalchemy` (2.x): Object Relational Mapper and database toolkit
- `psycopg[binary]` (3.x): Modern PostgreSQL driver
- `pwdlib[argon2]`: Password hashing library implementing Argon2
- `pyjwt[crypto]`: JSON Web Token encoding and cryptographic signature validation
- `pydantic` (2.x) & `pydantic-settings`: Data validation and typed environment settings
- `pytest` & `httpx`: Automated testing framework and ASGI test client

---

## 4. Environment Configuration

Application settings are managed by [`backend/app/core/config.py`](backend/app/core/config.py) using `pydantic-settings`. It discovers the `.env` file from the project root automatically.

### Setting Up `.env`
Copy the template file [`.env.example`](.env.example) to `.env`:
```powershell
Copy-Item .env.example .env
```

### Configuration Variables Reference
Open `.env` in your text editor and configure the following variables:

| Variable | Required | Default / Example | Purpose |
|---|---|---|---|
| `DATABASE_URL` | Yes | `postgresql+psycopg://postgres:<PASSWORD>@localhost:5432/artery_db` | Connection string for SQLAlchemy using psycopg 3 driver |
| `JWT_SECRET_KEY` | Yes | *Random 32+ char string* | Cryptographic secret used to sign and verify JWT tokens |
| `JWT_ALGORITHM` | No | `HS256` | JWT signing algorithm (HMAC with SHA-256) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | `30` | Lifespan of short-lived JWT access tokens |
| `REFRESH_TOKEN_EXPIRE_DAYS` | No | `7` | Lifespan of persistent refresh tokens in database |
| `ENVIRONMENT` | No | `development` | Runtime environment (`development`, `testing`, `production`) |
| `ALLOWED_ORIGINS` | No | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated list of allowed CORS origins for frontend web apps |

> [!IMPORTANT]
> **Password Special Characters:** If your PostgreSQL password contains reserved URI characters such as `@`, `:`, `#`, `%`, or `/`, URL-encode them (e.g. `@` becomes `%40`, `#` becomes `%23`).
> **Security:** Never commit `.env` to Git. The project's [`.gitignore`](.gitignore) excludes `.env`, `.venv/`, `.pytest_cache/`, and `__pycache__/`.

---

## 5. Database Setup & Schema Reference

### 1. PostgreSQL Multi-Instance Port Detection (Windows)
When multiple PostgreSQL versions are installed side-by-side on Windows, each instance listens on a different port:
- `postgresql-x64-17` -> default port **5432**
- `postgresql-x64-18` -> default port **5433**

To verify which service is running and inspect its listening port:
```powershell
Get-Service *postgres*
Get-Content "C:\Program Files\PostgreSQL\17\data\postgresql.conf" | Select-String "port ="
Get-Content "C:\Program Files\PostgreSQL\18\data\postgresql.conf" | Select-String "port ="
```

### 2. Creating the Database in pgAdmin or psql
1. Open **pgAdmin 4** (or connect via `psql -U postgres`).
2. Verify the server connection host (`localhost`), port (`5432`), and username (`postgres`).
3. If `artery_db` does not exist:
   - In pgAdmin: Right-click **Databases** > **Create** > **Database...**, name it `artery_db`, set Owner to `postgres`, and click **Save**.
   - In psql:
     ```sql
     CREATE DATABASE artery_db;
     ```

### 3. Schema Definitions in `database_work.sql`
[`database_work.sql`](database_work.sql) defines the target relational schema:
- **`users`:** `user_id SERIAL PRIMARY KEY`, `full_name VARCHAR(100)`, `email VARCHAR(150) UNIQUE`, `password_hash TEXT`, `created_at TIMESTAMP`.
- **`cases`:** `case_id SERIAL PRIMARY KEY`, `user_id INTEGER REFERENCES users(user_id)`, `case_name VARCHAR(150)`, `imaging_modality VARCHAR(50)`, `target_vessel VARCHAR(100)`, `status VARCHAR(30) DEFAULT 'pending'`, `created_at TIMESTAMP`.
- **`medical_images`:** `image_id SERIAL PRIMARY KEY`, `case_id INTEGER REFERENCES cases(case_id)`, `image_name VARCHAR(255)`, `image_path TEXT`, `image_format VARCHAR(10)`, `image_size BIGINT`, `uploaded_at TIMESTAMP`.
- **`segmentation_results`:** `segmentation_id SERIAL PRIMARY KEY`, `image_id INTEGER REFERENCES medical_images(image_id)`, `mask_path TEXT`, `model_name VARCHAR(100)`, `dice_score NUMERIC(6,5)`, `created_at TIMESTAMP`.
  - Extended metrics: `iou_score`, `processed_mask_path`, `centerline_path`, `model_version`, `precision_score`, `recall_score`, `f1_score`, `mean_vessel_diameter`.
- **`stenosis_results`:** `stenosis_id SERIAL PRIMARY KEY`, `image_id INTEGER REFERENCES medical_images(image_id)`, `location TEXT`, `severity VARCHAR(30)`, `confidence_score NUMERIC(6,5)`, `created_at TIMESTAMP`, `estimated_diameter NUMERIC(10,4)`.

---

## 6. Running the API

### Recommended Startup Command (Windows PowerShell)
To run the complete application exposing both authentication and all SQL-backed clinical endpoints:

```powershell
$env:PYTHONPATH = "backend"
.\.venv\Scripts\python.exe -m uvicorn app.extended_main:app --host 127.0.0.1 --port 8000 --reload
```

### macOS / Linux Alternative
```bash
PYTHONPATH=backend .venv/bin/python -m uvicorn app.extended_main:app --host 127.0.0.1 --port 8000 --reload
```

### Successful Startup Indicators
When launched successfully, Uvicorn outputs:
```text
INFO:     Will watch for changes in these directories: [...]
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [...]
INFO:     Started server process [...]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

- **Liveness probe:** `http://127.0.0.1:8000/api/v1/health` -> returns `{"status": "ok"}`
- **Readiness probe:** `http://127.0.0.1:8000/api/v1/health/ready` -> returns `{"status": "ready", "database": "connected"}` when connected to PostgreSQL.

---

## 7. Swagger UI & API Exploration

FastAPI automatically generates interactive OpenAPI documentation:

- **Interactive Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **OpenAPI 3.1 JSON Schema:** [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

### Authenticating in Swagger UI
1. Open `http://127.0.0.1:8000/docs`.
2. Expand `POST /api/v1/auth/login`, select **Try it out**, enter your registered email and password, and click **Execute**.
3. Copy the returned `access_token` string.
4. Scroll to the top of the Swagger UI page and click the green **Authorize** button (with the padlock icon).
5. Paste the token into the **Value** box in the format:
   ```text
   Bearer YOUR_ACCESS_TOKEN_HERE
   ```
   *(or simply paste the raw token if the dialog prompts for HTTPBearer)*.
6. Click **Authorize** and then **Close**. All protected endpoints will now send the Bearer header automatically.

---

## 8. Complete API Endpoint Reference

The complete application (`app.extended_main:app`) registers **20 endpoints**:

### A. Health & Readiness Probes
| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/api/v1/health` | Public | Liveness probe returning service operational status. |
| `GET` | `/api/v1/health/ready` | Public | Readiness probe verifying PostgreSQL connectivity with `SELECT 1`. Returns HTTP 503 if disconnected. |

### B. Authentication Endpoints (`/api/v1/auth`)
| Method | Path | Auth | Purpose |
|---|---|---|---|
| `POST` | `/api/v1/auth/register` | Public | Registers a new user with default role `medical_researcher`. Hashes password using Argon2. Returns HTTP 409 on duplicate email. |
| `POST` | `/api/v1/auth/login` | Public | Verifies email and password; returns JWT access token, token type, and refresh token. Returns HTTP 401 on bad credentials. |
| `POST` | `/api/v1/auth/refresh` | Public (Token) | Validates refresh token, invalidates old token (rotation), and issues new access/refresh pair. |
| `POST` | `/api/v1/auth/logout` | Public (Token) | Revokes the refresh token record in the database so it cannot be reused. |
| `POST` | `/api/v1/auth/change-password` | Bearer JWT | Verifies current password, updates to new Argon2 hash, and revokes all active sessions for the user. |

### C. User Profile Endpoints (`/api/v1/users`)
| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/api/v1/users/me` | Bearer JWT | Returns the authenticated user's ID, name, email, role, and account creation timestamp. |
| `PATCH` | `/api/v1/users/me` | Bearer JWT | Updates allowed profile fields (`name`, `email`). Blocks role escalation attempts. |

### D. SQL-Backed User Queries (`/api/v1/sql/users`)
| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/api/v1/sql/users/email-exists` | Public | Executes `auth_queries.sql` Query 1. Takes `?email=...` query param and returns `{ "email": "...", "exists": true/false }`. |
| `GET` | `/api/v1/sql/users/by-email` | Bearer JWT | Executes `auth_queries.sql` Query 2. Retrieves user record for verification. Strictly omits `password_hash` from response. |
| `GET` | `/api/v1/sql/users` | Bearer JWT | Executes `auth_queries.sql` Query 3. Lists registered users (`user_id`, `full_name`, `email`, `created_at`) without password hashes. |
| `PATCH` | `/api/v1/sql/users/{user_id}/name` | Bearer JWT | Executes `auth_queries.sql` Query 4. Updates a user's full name via parameterized SQL. |

### E. SQL-Backed Clinical Examination & Results Queries (`/api/v1/sql`)
| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/api/v1/sql/cases` | Bearer JWT | Executes `auth_queries.sql` Query 5. Returns all medical examination cases (`case_id`, `user_id`, `case_name`, `imaging_modality`, `target_vessel`, `status`, `created_at`). |
| `GET` | `/api/v1/sql/images` | Bearer JWT | Executes `auth_queries.sql` Query 6. Returns all medical images (`image_id`, `case_id`, `image_name`, `image_path`, `image_format`, `image_size`, `uploaded_at`). |
| `GET` | `/api/v1/sql/segmentation-results` | Bearer JWT | Executes `auth_queries.sql` Query 7. Returns segmentation outputs and evaluation metrics (`dice_score`, `iou_score`, `precision`, `recall`, `f1_score`, `mean_vessel_diameter`). |
| `GET` | `/api/v1/sql/stenosis-results` | Bearer JWT | Executes `auth_queries.sql` Query 8. Returns stenosis detection records (`location`, `severity`, `confidence_score`, `estimated_diameter`, `created_at`). |

### F. SQL-Backed Model Analytics Queries (`/api/v1/sql/analytics`)
| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/api/v1/sql/analytics/segmentation-count` | Bearer JWT | Executes `result_queries.sql` Query 1. Returns `{ "count": <int> }` for total segmentation outputs. |
| `GET` | `/api/v1/sql/analytics/stenosis-count` | Bearer JWT | Executes `result_queries.sql` Query 2. Returns `{ "count": <int> }` for total stenosis findings. |
| `GET` | `/api/v1/sql/analytics/model-performance` | Bearer JWT | Executes `result_queries.sql` Query 3. Groups segmentation results by `model_name` and returns average Dice score in descending order. |
| `GET` | `/api/v1/sql/analytics/summary` | Bearer JWT | Executes Queries 1, 2, and 3 into a consolidated analytics summary response. |

---

## 9. Detailed SQL Query Reference

| SQL File | Query Purpose | Tables Involved | Operation Type | Used by API Endpoint? |
|---|---|---|---|---|
| `auth_queries.sql` | 1. Check whether an email already exists | `users` | `SELECT EXISTS (...)` | Yes (`GET /api/v1/sql/users/email-exists`) |
| `auth_queries.sql` | 2. Retrieve user for login verification | `users` | `SELECT` | Yes (`GET /api/v1/sql/users/by-email`) |
| `auth_queries.sql` | 3. View registered users without password hashes | `users` | `SELECT` | Yes (`GET /api/v1/sql/users`) |
| `auth_queries.sql` | 4. Update a user's full name | `users` | `UPDATE` | Yes (`PATCH /api/v1/sql/users/{user_id}/name`) |
| `auth_queries.sql` | 5. View all clinical cases | `cases` | `SELECT` | Yes (`GET /api/v1/sql/cases`) |
| `auth_queries.sql` | 6. View all medical images | `medical_images` | `SELECT` | Yes (`GET /api/v1/sql/images`) |
| `auth_queries.sql` | 7. View segmentation results | `segmentation_results` | `SELECT` | Yes (`GET /api/v1/sql/segmentation-results`) |
| `auth_queries.sql` | 8. View stenosis results | `stenosis_results` | `SELECT` | Yes (`GET /api/v1/sql/stenosis-results`) |
| `result_queries.sql` | 1. Count segmentation results | `segmentation_results` | Aggregation (`COUNT`) | Yes (`GET /api/v1/sql/analytics/segmentation-count`) |
| `result_queries.sql` | 2. Count stenosis results | `stenosis_results` | Aggregation (`COUNT`) | Yes (`GET /api/v1/sql/analytics/stenosis-count`) |
| `result_queries.sql` | 3. Compare segmentation model performance | `segmentation_results` | Aggregation (`AVG`, `GROUP BY`, `ORDER BY`) | Yes (`GET /api/v1/sql/analytics/model-performance`) |
| `database_work.sql` | Define core clinical relational tables | `users`, `cases`, `medical_images`, `segmentation_results`, `stenosis_results` | Schema DDL (`CREATE TABLE`, `ALTER TABLE`) | Schema Reference for clinical queries |

### Critical Distinction in SQL Usage:
1. **SQL in `.sql` files:** Reference definitions and diagnostic queries stored on disk.
2. **SQL executed by the application:** Real SQL executed with parameterized inputs (`:email`, `:user_id`, `:full_name`) via SQLAlchemy in [`backend/app/services/sql_clinical_service.py`](backend/app/services/sql_clinical_service.py).
3. **ORM operations:** Authentication routes (`/auth/login`, `/auth/register`, `/users/me`) use SQLAlchemy 2.0 ORM sessions and models rather than raw string queries for transaction safety.

---

## 10. SQL Query-to-Endpoint Mapping

| SQL File & Query | Service Function | Registered API Endpoint | Target Database Table |
|---|---|---|---|
| `auth_queries.sql` (Q1) | `SQLClinicalService.check_email_exists` | `GET /api/v1/sql/users/email-exists` | `users` |
| `auth_queries.sql` (Q2) | `SQLClinicalService.get_user_by_email_for_verification` | `GET /api/v1/sql/users/by-email` | `users` |
| `auth_queries.sql` (Q3) | `SQLClinicalService.get_registered_users` | `GET /api/v1/sql/users` | `users` |
| `auth_queries.sql` (Q4) | `SQLClinicalService.update_user_name` | `PATCH /api/v1/sql/users/{user_id}/name` | `users` |
| `auth_queries.sql` (Q5) | `SQLClinicalService.get_all_cases` | `GET /api/v1/sql/cases` | `cases` |
| `auth_queries.sql` (Q6) | `SQLClinicalService.get_all_medical_images` | `GET /api/v1/sql/images` | `medical_images` |
| `auth_queries.sql` (Q7) | `SQLClinicalService.get_all_segmentation_results` | `GET /api/v1/sql/segmentation-results` | `segmentation_results` |
| `auth_queries.sql` (Q8) | `SQLClinicalService.get_all_stenosis_results` | `GET /api/v1/sql/stenosis-results` | `stenosis_results` |
| `result_queries.sql` (Q1) | `SQLClinicalService.get_segmentation_count` | `GET /api/v1/sql/analytics/segmentation-count` | `segmentation_results` |
| `result_queries.sql` (Q2) | `SQLClinicalService.get_stenosis_count` | `GET /api/v1/sql/analytics/stenosis-count` | `stenosis_results` |
| `result_queries.sql` (Q3) | `SQLClinicalService.get_model_performance` | `GET /api/v1/sql/analytics/model-performance` | `segmentation_results` |
| `result_queries.sql` (Q1-3) | `SQLClinicalService.*` (Aggregated) | `GET /api/v1/sql/analytics/summary` | `segmentation_results`, `stenosis_results` |

---

## 11. Authentication & Authorization Flow

### Concept Distinction:
- **Authentication:** Who are you? (Verified via email and Argon2 hashed password verification; verified on each request via signed JWT bearer token).
- **Authorization:** What are you allowed to do? (Verified via `RoleChecker` dependency enforcing permitted project roles: `radiologist`, `medical_researcher`, `data_scientist`, `administrator`).

### Security Architecture:
1. **Password Hashing:** Passwords are hashed with **Argon2** via `pwdlib`. Plaintext passwords are never stored, logged, or returned.
2. **Access Tokens:** Signed JWT access tokens with subject `sub` (User UUID) and `role`. Tokens expire after 30 minutes (`ACCESS_TOKEN_EXPIRE_MINUTES`).
3. **Refresh Tokens:** Long-lived 48-byte URL-safe tokens stored as SHA-256 digests in the `refresh_tokens` database table.
4. **Token Rotation:** Every call to `/api/v1/auth/refresh` immediately revokes the submitted refresh token and generates a new pair. If a revoked token is reused, all active sessions for that user are revoked immediately (reuse attack detection).
5. **Password Change:** Changing password immediately updates the Argon2 hash and invalidates all existing refresh tokens, forcing re-authentication.

---

## 12. Testing Guide

### Test Architecture
Tests use **pytest** and **FastAPI TestClient**. To ensure complete developer safety and avoid modifying or corrupting real database records:
- The test harness ([`tests/conftest.py`](tests/conftest.py)) provisions an **isolated in-memory SQLite database** with table schemas matching `database_work.sql`.
- Dependency override (`app.dependency_overrides[get_db]`) intercepts database sessions for tests without touching PostgreSQL.

### Running Automated Tests
Run the entire test suite from Windows PowerShell:
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### Verified Test Results (35 Passed, 0 Failed):
- **`tests/test_auth.py` (14 tests):**
  - Registration success, duplicate rejection (409), password length validation (422), role override protection.
  - Login success, invalid password rejection (401), non-existent email rejection (401).
  - Password hashing verification (hash differs from plaintext).
  - Refresh token rotation, token reuse detection, logout revocation.
  - Password change success, session revocation, invalid password rejection.
  - Inactive user login blocking (403).
- **`tests/test_users.py` (8 tests):**
  - Authenticated profile retrieval (`/users/me`), missing token rejection (401), expired token rejection (401), malformed token rejection (401).
  - Inactive account access blocking (403).
  - Name and email updates (`PATCH /users/me`).
  - Role escalation prevention (unauthorized role update attempts are ignored).
- **`tests/test_clinical_endpoints.py` (13 tests):**
  - SQL email existence query (Q1), user verification lookup (Q2), registered users listing (Q3), name update (Q4).
  - SQL cases listing (Q5), medical images listing (Q6), segmentation results (Q7), stenosis results (Q8).
  - Segmentation count (RQ1), stenosis count (RQ2), model performance comparison (RQ3), combined summary.
  - Bearer token authentication enforcement on all clinical endpoints.

---

## 13. Example API Usage (PowerShell & cURL)

### 1. Check Service Health
```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/v1/health"
```
```bash
curl -X GET http://127.0.0.1:8000/api/v1/health
```
*Expected Response:*
```json
{
  "status": "ok",
  "service": "artery-segmentation-auth"
}
```

### 2. Register a New Clinician / Researcher
```powershell
$body = @{
    name = "Dr. Jane Doe"
    email = "jane.doe@cardiology.hospital.org"
    password = "SecurePassword2026!"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/auth/register" `
    -ContentType "application/json" -Body $body
```
*Expected Response (HTTP 201):*
```json
{
  "id": "7f000001-8b20-13f5-818b-201a4e5c0001",
  "name": "Dr. Jane Doe",
  "email": "jane.doe@cardiology.hospital.org",
  "role": "medical_researcher",
  "is_active": true,
  "created_at": "2026-11-02T10:00:00Z",
  "updated_at": "2026-11-02T10:00:00Z"
}
```

### 3. Log In to Receive Access & Refresh Tokens
```powershell
$loginBody = @{
    email = "jane.doe@cardiology.hospital.org"
    password = "SecurePassword2026!"
} | ConvertTo-Json

$loginResponse = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/auth/login" `
    -ContentType "application/json" -Body $loginBody

$token = $loginResponse.access_token
Write-Host "Received JWT Access Token: $token"
```

### 4. Query Current Authenticated Profile
```powershell
$headers = @{ Authorization = "Bearer $token" }
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/v1/users/me" -Headers $headers
```

### 5. Query Clinical Cases via SQL Endpoint
```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/v1/sql/cases" -Headers $headers
```

### 6. Query Model Benchmarking Analytics
```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/v1/sql/analytics/model-performance" -Headers $headers
```
*Example Response:*
```json
[
  {
    "model_name": "3D-ResUNet-Artery",
    "average_dice_score": 0.9124
  },
  {
    "model_name": "3D-UNet",
    "average_dice_score": 0.8850
  }
]
```

---

## 14. Troubleshooting Guide

| Issue | Root Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named 'app'` | `PYTHONPATH` not pointing to `backend/` directory | In PowerShell, run `$env:PYTHONPATH="backend"` before running Uvicorn. |
| `Cannot connect to PostgreSQL on localhost:5432` | PostgreSQL Windows service is stopped or port is 5433 | Run `Get-Service *postgres*` to confirm service is running. If using PostgreSQL 18, update port to `5433` in `.env`. |
| `HTTP 503: Database query error` on `/sql/*` routes | Target table does not exist in the active PostgreSQL database | Run the DDL statements from `database_work.sql` in pgAdmin against `artery_db` to create `cases`, `medical_images`, and result tables. |
| `HTTP 401: Token has expired` | JWT access token lifetime (30m) elapsed | Submit refresh token to `POST /api/v1/auth/refresh` to rotate and obtain a fresh access token. |
| `HTTP 403: Operation not permitted: insufficient privileges` | User role does not meet `RoleChecker` requirement | Contact administrator to adjust role in the database. |
| `WinError 10048: Only one usage of each socket address is permitted` | Port 8000 is occupied by an existing Uvicorn instance | Run `Get-NetTCPConnection -LocalPort 8000` to find the process ID, terminate it with `Stop-Process -Id <PID>`, or start on port 8001. |

---

## 15. Known Issues & Limitations

1. **Syntax Blocker in `database_work.sql` (Line 53):**
   In [`database_work.sql`](database_work.sql#L53), line 53 ends with a premature semicolon `;`:
   ```sql
   ADD COLUMN IF NOT EXISTS model_version VARCHAR(50); -- Premature semicolon terminates statement!
   ADD COLUMN IF NOT EXISTS precision_score NUMERIC(6,5),
   ```
   *Impact:* Executing this script directly in pgAdmin or psql will raise a syntax error at line 54. Per strict non-destructive repository rules, the file has been preserved unchanged. When applying migrations to PostgreSQL, change that semicolon to a comma `,`.
2. **User Identifier Type Divergence:**
   The authentication system uses UUID primary keys (`users.id UUID`), whereas [`database_work.sql`](database_work.sql) specifies integer serial keys (`users.user_id SERIAL`, `cases.user_id INTEGER`). The clinical response schemas bridge this using `Union[int, str]`, but unifying user IDs across the database layer remains a planned integration task when merging teammate branches.
3. **ML Inference Pipeline Not Connected:**
   This API currently serves the database queries and metrics reporting for segmentation and stenosis results. Actual deep learning inference (e.g. running 3D-UNet or PyTorch weights on DICOM files) is part of a separate pipeline and is not yet invoked by this API service.
