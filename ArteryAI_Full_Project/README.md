# ArteryAI — Coronary Artery Research Platform

**Project:** Artery Segmentation and Stenosis Detection Using Advanced Deep Learning Techniques

This package contains the React/Vite research dashboard and a Flask/PyTorch inference API. The dashboard has been redesigned to resemble the requested dark coronary angiography workstation, using sky blue, blush pink, cement grey and black accents. Existing routes and case workflows are retained. Existing dashboard, login, registration, case creation, case history, upload and preprocessing pages are retained.

## Important scientific limitation

The code is an end-to-end development starter, not a trained clinical diagnostic system. Model inference requires separately trained weights. The repository intentionally does not fabricate predictions when weights are missing. Stenosis percentage measurement is not implemented as a clinical measurement; a lesion mask alone is not enough. Never use this prototype for patient care.

## Requirements

- Node.js 22+ (Node 24 is fine) and npm
- Python 3.11 recommended
- Internet connection for package installation
- A labeled research dataset for model training

## A. Run the frontend (demo dashboard)

1. Extract the ZIP.
2. In VS Code choose **File → Open Folder** and select the extracted `ArteryAI_Full_Project` folder — the folder that contains `package.json`.
3. Choose **Terminal → New Terminal**.
4. Run:

   ```powershell
   npm install
   npm run dev
   ```

5. Open the local URL printed by Vite, usually `http://localhost:5173`.

The dashboard uses mock case management by default. AI upload requests are enabled by `.env`; the Flask backend must also be running to use the AI endpoint. Restart Vite after editing `.env`.

## B. Run the Flask API

Open a second VS Code terminal in the same project root:

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

If PowerShell blocks activation, use this for the current terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Check the backend by opening `http://127.0.0.1:8000/api/v1/health`. The response reports whether vessel and lesion model weights are available.

## C. Train the models

The dataset must contain expert-annotated binary masks. Put paired image/mask files into this structure (mask files must be PNG and share the image stem):

```text
 dataset/
   vessel/
     train/images/  train/masks/
     val/images/    val/masks/
   lesion/
     train/images/  train/masks/
     val/images/    val/masks/
```

For example, `dataset/vessel/train/images/001.jpg` pairs with `dataset/vessel/train/masks/001.png`. White pixels in each mask represent the target class. Make sure your labels truly correspond to vessel or lesion annotations; do not use arbitrary or synthetic masks for model claims.

From the project root with the backend virtual environment activated:

```powershell
python backend/ai/train.py --task vessel --epochs 30 --batch-size 4
python backend/ai/train.py --task lesion --epochs 30 --batch-size 4
```

The checkpoints are saved to `backend/weights/vessel_unet.pth` and `backend/weights/lesion_unet.pth`. Restart the backend after training. Training time depends on the dataset and hardware; CPU training may be slow.

If PyTorch installation fails, use the official selector at https://pytorch.org/get-started/locally/ to choose the appropriate Windows/Pip/Python build. The requirements file includes CPU-compatible default PyTorch installation but GPU/CUDA choices depend on the machine.

## Supported image formats

PNG, JPG/JPEG, TIFF and BMP. DICOM files need conversion to a suitable image frame before upload in this prototype.

## Troubleshooting

- `ENOENT package.json`: VS Code is opened in the wrong directory. Run `dir` and confirm `package.json` is listed before `npm install`.
- `npm is not recognized`: install Node.js LTS and restart VS Code.
- Frontend cannot connect: confirm Flask is running on port 8000 and Vite is running on port 5173.
- `model_missing`: this is expected until trained model weights exist.
- Do not report model performance unless it was measured on a held-out, annotated test set. A stenosis percentage requires calibrated vessel geometry and validation against expert measurements.
