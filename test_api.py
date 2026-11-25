import httpx
import time
import io
from PIL import Image

BASE = "http://127.0.0.1:8001"
results = []

def test(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

print("Waiting for server to start...")
for i in range(30):
    try:
        resp = httpx.get(f"{BASE}/api/v1/health", timeout=2)
        if resp.status_code == 200:
            print("Server is ready!")
            break
    except Exception:
        pass
    time.sleep(1)
else:
    print("ERROR: Server did not start in time")
    exit(1)

# 1. Health endpoint
resp = httpx.get(f"{BASE}/api/v1/health", timeout=10)
test("Health endpoint", resp.status_code == 200, f"status={resp.status_code}")

# 2. Readiness endpoint
resp = httpx.get(f"{BASE}/api/v1/health/ready", timeout=10)
test("Readiness endpoint", resp.status_code == 200, f"status={resp.status_code}")

# 3. Register user
reg_data = {"name": "Docker Tester", "email": f"docker_test_{int(time.time())}@test.com", "password": "TestPass123!"}
resp = httpx.post(f"{BASE}/api/v1/auth/register", json=reg_data, timeout=10)
test("Register user", resp.status_code == 201, f"status={resp.status_code}")
if resp.status_code == 201:
    user_id_a = resp.json().get("id")
    print(f"  User A ID: {user_id_a}")

# 4. Login
login_data = {"email": reg_data["email"], "password": "TestPass123!"}
resp = httpx.post(f"{BASE}/api/v1/auth/login", json=login_data, timeout=10)
test("Login", resp.status_code == 200, f"status={resp.status_code}")
token_a = resp.json().get("access_token") if resp.status_code == 200 else None
headers_a = {"Authorization": f"Bearer {token_a}"} if token_a else {}

# 5. Get profile
resp = httpx.get(f"{BASE}/api/v1/users/me", headers=headers_a, timeout=10)
test("Get profile", resp.status_code == 200, f"status={resp.status_code}")

# 6. Create case
case_data = {"case_name": "Docker Test Case", "imaging_modality": "CTA", "target_vessel": "LAD", "status": "pending"}
resp = httpx.post(f"{BASE}/api/v1/sql/cases", json=case_data, headers=headers_a, timeout=10)
test("Create case", resp.status_code == 201, f"status={resp.status_code}")
case_id_a = resp.json().get("case_id") if resp.status_code == 201 else None

# 7. Get case by ID
resp = httpx.get(f"{BASE}/api/v1/sql/cases/{case_id_a}", headers=headers_a, timeout=10)
test("Get case by ID", resp.status_code == 200, f"status={resp.status_code}")

# 8. List cases
resp = httpx.get(f"{BASE}/api/v1/sql/cases", headers=headers_a, timeout=10)
test("List cases", resp.status_code == 200, f"status={resp.status_code}, count={len(resp.json())}")

# 9. Register image metadata
img_data = {"image_name": "angiography_001.png", "image_path": "uploads/angio_001.png", "image_format": "PNG", "image_size": 204800}
resp = httpx.post(f"{BASE}/api/v1/sql/cases/{case_id_a}/images", json=img_data, headers=headers_a, timeout=10)
test("Register image metadata", resp.status_code == 201, f"status={resp.status_code}")
image_id_a = resp.json().get("image_id") if resp.status_code == 201 else None

# 10. List ML models
resp = httpx.get(f"{BASE}/api/v1/inference/models", headers=headers_a, timeout=10)
test("List ML models", resp.status_code == 200, f"status={resp.status_code}, models={resp.json()}")
models_available = resp.json() if resp.status_code == 200 else []

# 11. Upload and predict (U-Net)
img = Image.new("L", (64, 64), color=128)
buf = io.BytesIO()
img.save(buf, format="PNG")
buf.seek(0)
img_bytes = buf.read()

resp = httpx.post(
    f"{BASE}/api/v1/inference/upload-and-predict",
    files={"file": ("angio_test.png", img_bytes, "image/png")},
    data={"case_id": case_id_a, "model_type": "unet"},
    headers=headers_a,
    timeout=60
)
test("U-Net predict", resp.status_code == 201, f"status={resp.status_code}")
if resp.status_code == 201:
    result = resp.json()
    test("Returns segmentation_id", result.get("segmentation_id") is not None)
    test("Returns image_id", result.get("image_id") is not None)
    test("Returns case_id", result.get("case_id") == case_id_a)
    test("Returns model_name=unet", result.get("model_name") == "unet")
    test("Returns mask_base64", result.get("mask_base64") is not None)
    test("Returns overlay_base64", result.get("overlay_base64") is not None)
    seg_id = result.get("segmentation_id")
    seg_mask_id = result.get("mask_storage_identifier")
    seg_overlay_id = result.get("overlay_base64")
else:
    seg_id = None
    seg_mask_id = None
    seg_overlay_id = None

# 12. Attention U-Net predict
img = Image.new("L", (64, 64), color=100)
buf = io.BytesIO()
img.save(buf, format="PNG")
buf.seek(0)

resp = httpx.post(
    f"{BASE}/api/v1/inference/upload-and-predict",
    files={"file": ("test.png", buf, "image/png")},
    data={"case_id": case_id_a, "model_type": "attention_unet"},
    headers=headers_a,
    timeout=60
)
test("Attention U-Net predict", resp.status_code == 201, f"status={resp.status_code}")
if resp.status_code == 201:
    result = resp.json()
    test("Attention U-Net model_name", result.get("model_name") == "attention_unet")
    test("Attention U-Net mask_base64", result.get("mask_base64") is not None)
    test("Attention U-Net overlay_base64", result.get("overlay_base64") is not None)

# 13. Get segmentation result by ID
if seg_id:
    resp = httpx.get(f"{BASE}/api/v1/inference/segmentation-results/{seg_id}", headers=headers_a, timeout=10)
    test("Get segmentation result by ID", resp.status_code == 200, f"status={resp.status_code}")

# 14. Get mask file
if seg_id:
    resp = httpx.get(f"{BASE}/api/v1/inference/masks/{seg_id}", headers=headers_a, timeout=10)
    test("Get mask file", resp.status_code == 200, f"status={resp.status_code}")

# 15. Get overlay file
if seg_id:
    resp = httpx.get(f"{BASE}/api/v1/inference/overlays/{seg_id}", headers=headers_a, timeout=10)
    test("Get overlay file", resp.status_code == 200, f"status={resp.status_code}")

# 16. Invalid model type
resp = httpx.post(
    f"{BASE}/api/v1/inference/predict",
    files={"file": ("test.png", img_bytes, "image/png")},
    data={"model_type": "invalid_model"},
    headers=headers_a,
    timeout=10
)
test("Invalid model type returns 400", resp.status_code == 400, f"status={resp.status_code}")

# 17. No auth
resp = httpx.get(f"{BASE}/api/v1/users/me", timeout=10)
test("No auth returns 401", resp.status_code == 401, f"status={resp.status_code}")

# Summary
passed = sum(1 for r in results if "[PASS]" in r)
failed = sum(1 for r in results if "[FAIL]" in r)
print(f"\n{'='*60}")
print(f"SUMMARY: {passed} passed, {failed} failed out of {len(results)} tests")
for r in results:
    print(f"  {r}")