# ArteryAI AI backend (research prototype)

This Flask API keeps your existing React app and mock case-management workflow. It adds an optional real-image inference endpoint. **The code does not include trained weights.** A model must be trained on correctly annotated coronary angiography data before its output is meaningful.

## 1. Install (Windows PowerShell)
From the `backend` folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch install fails, select the correct Windows/Pip command at https://pytorch.org/get-started/locally/ . CPU is sufficient for small experiments but training can be slow.

## 2. Dataset format
This training starter expects separate binary vessel and lesion masks. Convert dataset annotations to binary PNG masks only after checking the dataset's annotation semantics and license. Image/mask names must match by stem.

```text
dataset/
  vessel/
    train/images/  train/masks/
    val/images/    val/masks/
  lesion/
    train/images/  train/masks/
    val/images/    val/masks/
backend/weights/
```

Masks must be black background and white foreground. If a source dataset uses COCO polygons, instance labels or multiclass masks, convert them correctly first. Do not use the same vessel masks as lesion masks.

Train models from `backend`:

```powershell
python -m ai.train --task vessel --epochs 30
python -m ai.train --task lesion --epochs 30
```

The first command saves `backend/weights/vessel_unet.pth`; the second saves `backend/weights/lesion_unet.pth`. The starter U-Net is randomly initialized and is **not pretrained**. Training must complete successfully before inference. Split train/validation/test by patient or study where possible to avoid data leakage.

## 3. Run API
From `backend` with the virtual environment active:

```powershell
python app.py
```

Health check: http://localhost:8000/api/v1/health

The endpoint accepts PNG/JPG/TIFF/BMP only. DICOM conversion is intentionally not enabled in this starter; export a valid image frame first. POST `/api/v1/ai/analyze` as multipart/form-data with fields `file` and optional `caseId`.

## 4. Research safety
The API will report `model_missing` if the vessel weights do not exist; it never invents a segmentation. A lesion mask is not itself a stenosis percentage. Accurate percentage stenosis needs validated reference diameter measurement and calibration, plus expert-ground-truth evaluation. This project is not for clinical decisions.
