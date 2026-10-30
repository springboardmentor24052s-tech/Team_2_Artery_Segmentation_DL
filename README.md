# Artery Segmentation and Stenosis Detection Using Advanced Deep Learning Techniques
## Authentication & Role-Based Access Control Backend Service

This repository provides a modular, production-ready authentication backend service built with **FastAPI**, **SQLAlchemy 2.x**, **PostgreSQL** (via `psycopg 3`), **Pydantic 2**, and **Argon2** password hashing.

---

## 1. Prerequisites

- **Windows OS** (PowerShell 5.1 / PowerShell 7+)
- **Python 3.10+** (Python 3.12 recommended)
- **PostgreSQL 15+** installed and running on localhost

---

## 2. Setup Instructions (PowerShell)

### A. Clone & Navigate to Project Directory
```powershell
cd "c:\Users\Rohan R\Documents\INFOSYS SPRINGBOARD INTERNSHIP 7\Artery project"
```

### B. Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### C. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 3. Database Configuration & Multi-Instance Setup

### Identifying Your Local PostgreSQL Instances
On Windows systems with multiple PostgreSQL versions installed, each service binds to a different port:
- `postgresql-x64-17` -> default port **5432**
- `postgresql-x64-18` -> default port **5433**

You can verify which port each service runs on in PowerShell:
```powershell
Get-Service *postgres*
Get-Content "C:\Program Files\PostgreSQL\17\data\postgresql.conf" | Select-String "port ="
Get-Content "C:\Program Files\PostgreSQL\18\data\postgresql.conf" | Select-String "port ="
```

### Verifying and Creating Database in pgAdmin
1. Open **pgAdmin 4**.
2. Under **Servers**, expand the server node corresponding to your chosen PostgreSQL version (check the port under **Properties > Connection**):
   - Host name/address: `localhost` or `127.0.0.1`
   - Port: `5432` (PostgreSQL 17) or `5433` (PostgreSQL 18)
   - Username: `postgres`
3. Expand **Databases**. Check if `artery_db` is listed:
   - If `artery_db` is **not present**: Right-click **Databases** > **Create** > **Database...**, enter Database name `artery_db`, set Owner to `postgres`, and click **Save**.
   - If resetting password is needed: Right click **Login/Group Roles** > `postgres` > **Properties** > **Definition** tab, enter a new password, and click **Save**.

### Configuring `.env`
1. Copy `.env.example` to `.env` if not already created:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Edit `.env` and set your connection URL:
   ```ini
   # For PostgreSQL 17 on port 5432:
   DATABASE_URL=postgresql+psycopg://postgres:<PASSWORD>@localhost:5432/artery_db

   # (If your password contains special characters like @, #, :, encode them or URL-escape them)
   JWT_SECRET_KEY=generate_a_random_32_char_secret_key
   JWT_ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   REFRESH_TOKEN_EXPIRE_DAYS=7
   ENVIRONMENT=development
   ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
   ```

---

## 4. Run Automated Tests

To execute the test suite (covers all 7 endpoints, security rules, token rotation, and permission guards):
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

---

## 5. Start the Backend API Server

Run the server with Uvicorn:
```powershell
$env:PYTHONPATH="backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Swagger UI Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **OpenAPI Schema:** [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)
- **Liveness Probe:** [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)
- **Readiness Probe:** [http://127.0.0.1:8000/api/v1/health/ready](http://127.0.0.1:8000/api/v1/health/ready)

---

## 6. Endpoints & Sample Requests

### 1. Register (`POST /api/v1/auth/register`)
- Default role assigned: `medical_researcher`
```bash
curl -X POST http://127.0.0.1:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Dr. Sarah Connor",
    "email": "sarah.connor@hospital.org",
    "password": "StrongPassword2026!"
  }'
```

### 2. Login (`POST /api/v1/auth/login`)
```bash
curl -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "sarah.connor@hospital.org",
    "password": "StrongPassword2026!"
  }'
```
Response:
```json
{
  "access_token": "<JWT_ACCESS_TOKEN>",
  "token_type": "bearer",
  "refresh_token": "<SECURE_REFRESH_TOKEN>",
  "expires_in": 1800
}
```

### 3. Get Current User Profile (`GET /api/v1/users/me`)
```bash
curl -X GET http://127.0.0.1:8000/api/v1/users/me \
  -H "Authorization: Bearer <JWT_ACCESS_TOKEN>"
```

### 4. Update Profile (`PATCH /api/v1/users/me`)
```bash
curl -X PATCH http://127.0.0.1:8000/api/v1/users/me \
  -H "Authorization: Bearer <JWT_ACCESS_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Dr. Sarah Connor, MD"
  }'
```

### 5. Refresh Access Token with Token Rotation (`POST /api/v1/auth/refresh`)
```bash
curl -X POST http://127.0.0.1:8000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "<SECURE_REFRESH_TOKEN>"
  }'
```

### 6. Change Password (`POST /api/v1/auth/change-password`)
```bash
curl -X POST http://127.0.0.1:8000/api/v1/auth/change-password \
  -H "Authorization: Bearer <JWT_ACCESS_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "current_password": "StrongPassword2026!",
    "new_password": "EvenStrongerPassword2026!"
  }'
```

### 7. Logout (`POST /api/v1/auth/logout`)
```bash
curl -X POST http://127.0.0.1:8000/api/v1/auth/logout \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "<SECURE_REFRESH_TOKEN>"
  }'
```

