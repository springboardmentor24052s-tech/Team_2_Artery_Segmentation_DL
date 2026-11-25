# ArteryAI — Full-stack coronary image research workspace

A simplified, full-stack research application. **The six Milestone 3 modules and their navigation/pages have been removed.** The remaining application provides:

- User registration and sign-in with signed, expiring bearer tokens
- A personal case dashboard and case statistics
- Image upload and case metadata storage
- Searchable case library, case detail view and case deletion
- Authenticated original-image and segmentation-mask viewing
- Optional U-Net segmentation inference when a trained checkpoint is installed
- SQLite by default, with optional MySQL configuration
- Responsive React/Vite frontend and Flask REST API

## Project structure

```text
ArteryAI_Milestone3_Dark_v4/
├── backend/
│   ├── app.py                 # Flask API, auth, cases, protected file access
│   ├── config.py
│   ├── models/                # SQLAlchemy models and U-Net
│   ├── services/              # preprocessing, segmentation, narrowing estimate
│   ├── training/              # U-Net training script
│   ├── model_weights/         # add trained unet.pth here
│   └── requirements.txt
├── frontend/
│   ├── src/App.jsx            # dashboard, case library, upload, case review
│   ├── src/api.js             # API client and bearer-token handling
│   └── package.json
├── database/schema.sql
├── dataset/images/            # training images
├── dataset/masks/             # paired binary masks
├── START_BACKEND.bat
└── START_FRONTEND.bat
```

## Requirements

- Windows 10/11, macOS or Linux
- Python 3.11 or 3.13 (the launcher detects either version)
- Node.js 20 LTS or newer and npm
- SQLite (included with Python); MySQL is optional

## One-click full-stack startup (Windows)

1. Extract the ZIP completely.
2. Open the extracted `ArteryAI_Milestone3_Dark_v4` folder in VS Code.
3. Install Python 3.13 or 3.11 and Node.js 20+ if they are not already installed.
4. Double-click `START_ARTERYAI.bat` in the project root. It prepares the Python environment, installs/checks backend dependencies, installs frontend dependencies if needed, starts Flask in a separate terminal window, and starts Vite.
5. Open `http://localhost:5173` when Vite prints its Local URL.

Keep both terminal windows open. To launch from VS Code, use **Terminal → Run Build Task** or press **Ctrl+Shift+B**; the included `.vscode/tasks.json` runs the same launcher.

The first run may take several minutes because Python packages (including PyTorch) and npm packages need to be downloaded. Internet access is required for first-time installation. Later runs skip npm installation when `frontend/node_modules` already exists. The launcher does not create trained model weights; U-Net inference still requires training on paired annotated images and masks.

## Run on Windows

1. Extract the ZIP to a writable folder.
2. Double-click `START_BACKEND.bat`. On first run it creates `backend/venv`, installs the Python requirements and starts Flask.
3. Double-click `START_FRONTEND.bat`. On first run it installs npm dependencies and starts Vite.
4. Open the frontend URL printed by Vite (normally `http://localhost:5173`).
5. Create an account, sign in, upload a de-identified image and manage it from the case library.

Keep both terminal windows open while using the application.

## Run manually

### Backend

```powershell
cd backend
py -3.11 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Health check: `http://127.0.0.1:5000/api/health`

### Frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Configuration

Copy `backend/.env.example` to `backend/.env` and set a long random `SECRET_KEY` before any deployment. SQLite is the default. For MySQL, set for example:

```env
DATABASE_URL=mysql+pymysql://username:password@localhost/arteryai
SECRET_KEY=replace-with-a-long-random-secret
```

For a frontend hosted somewhere other than the local Vite default, set `VITE_API_BASE_URL` to the API base URL, e.g. `http://127.0.0.1:5000/api`, before building/running Vite. Set `CORS_ORIGINS` in the backend environment to the exact frontend origins, comma-separated.

## API overview

| Method | Endpoint | Purpose | Auth |
|---|---|---|---|
| GET | `/api/health` | API health | No |
| POST | `/api/auth/register` | Create account | No |
| POST | `/api/auth/login` | Sign in and receive token | No |
| GET | `/api/auth/me` | Current account | Yes |
| GET | `/api/cases` | List own cases | Yes |
| POST | `/api/cases` | Upload image and create case | Yes |
| GET | `/api/cases/<id>` | Case details and latest analysis | Yes |
| DELETE | `/api/cases/<id>` | Delete own case and files | Yes |
| POST | `/api/cases/<id>/analyze` | Run segmentation if weights exist | Yes |
| GET | `/api/uploads/<filename>` | Retrieve own image/mask | Yes |

Protected endpoints use `Authorization: Bearer <token>`. Tokens expire after seven days. The backend scopes case queries and file access to the authenticated account.

## U-Net training and inference

**A trained checkpoint and annotated dataset are not included.** The repository contains the U-Net architecture, training code and inference integration; it cannot produce genuine segmentation predictions until you train it on paired images and ground-truth masks.

### Dataset layout

Put each source image in `dataset/images/` and its expert-annotated binary mask in `dataset/masks/`, using exactly the same filename:

```text
dataset/
  images/
    case001.png
    case002.png
  masks/
    case001.png
    case002.png
```

Masks must be grayscale/binary: white (pixel value >=128) for vessel pixels and black for background. Masks must be aligned to their source images. Use expert-reviewed annotations and a consistent definition of the target vessel tree. Do not use the generated sample angiogram with a red arrow as a mask.

### Train on Windows / VS Code

1. Start the backend environment once and install `backend/requirements.txt`.
2. Add paired images and masks as above.
3. Open a second VS Code terminal at the project root and run:

```powershell
cd backend
.\\venv\\Scripts\\Activate.ps1
python -m training.train --epochs 60 --batch-size 4
```

Alternatively, double-click `TRAIN_MODEL.bat` from the project root after the backend virtual environment and dependencies have been installed.

The script uses the same CLAHE/blur preprocessing in training and inference, creates a reproducible train/validation split, applies simple flip augmentation to training samples, optimizes BCE + Dice loss, reports validation Dice, and saves the best validation checkpoint to `backend/model_weights/unet.pth`. Training requires at least two paired images; substantially more diverse, expert-annotated data is needed for meaningful research evaluation. Adjust epochs/batch size to available hardware.

### Run inference

After training, restart Flask if it was already running, then open a case in ArteryAI and click **Run segmentation request**. The backend loads the checkpoint on first inference and caches it; it uses CUDA when available and CPU otherwise. The predicted binary mask is resized to the original uploaded image dimensions and shown in the case review page.

Check model availability at `http://127.0.0.1:5000/api/model/status`. The checkpoint can also be placed at `backend/model_weights/unet.pth`; set `ARTERYAI_MODEL_PATH` to use a different checkpoint path.

### Checkpoint and metrics

The saved checkpoint includes model weights, architecture metadata, threshold, training pair count, epoch and best validation Dice. `backend/training_report.json` records the final training summary. Validation Dice is calculated on the held-out split; it is not clinical validation and can be misleading on small or unrepresentative datasets.

## Important limitations and safety

- This is an academic research prototype, not a clinically validated medical device.
- No trained weights or expert-annotated dataset are supplied. Until a valid checkpoint is trained and installed, analysis correctly reports that the model is not ready.
- The included percentage-narrowing calculation is a rudimentary geometry estimate, not a validated stenosis grade. It must not guide diagnosis or treatment.
- Use only synthetic or properly de-identified research images. Do not upload patient-identifiable information.
- Before public deployment, configure HTTPS, a production secret, strict CORS, secure hosting/storage, backups, rate limiting, audit logs, and applicable privacy/regulatory controls. The built-in token scheme is a starter implementation, not a substitute for a security review.
