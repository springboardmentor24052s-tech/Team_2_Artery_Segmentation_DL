# ArteryAI — Coronary Artery Research Platform

**Project Title:** Artery Segmentation and Stenosis Detection Using Advanced Deep Learning Techniques  
**Milestone:** Milestone 1 — Modern Frontend Architecture & Research Dashboard  
**Dataset Context:** ARCADE Benchmark Dataset (2D X-ray Coronary Angiography - XCA)

---

## 1. Project Overview

**ArteryAI** is an AI-assisted medical imaging research platform designed to streamline coronary artery segmentation and stenosis-region analysis for cardiovascular researchers, faculty mentors, and students. 

The platform is designed around the **ARCADE dataset**, which consists of **2D X-ray coronary angiography (XCA)** images and cine sequences. 

> **Important Research Scope Notes:**
> - **Modality:** The platform focuses exclusively on **X-ray coronary angiography (XCA)**. It does **not** use 3D CT angiography.
> - **Status:** In Milestone 1, all AI deep learning inference models (e.g., vessel segmentation masks and stenosis percentage calculations) and automated clinical diagnosis features are represented as **clearly designated placeholders**. No real clinical claims are made.
> - **Demo Mode:** When operating in demo mode, synthetic research cases and simulated workflows are presented without transmitting data to external servers.

---

## 2. Technology Stack

- **React.js 19** with **Vite**
- **JavaScript & JSX**
- **Tailwind CSS** (navy text, clean white/light backgrounds, and medical blue accent palette)
- **React Router DOM v7** (modular routing between auth layouts and authenticated dashboard layouts)
- **Axios** (preconfigured API client with request/response interceptors for future backend deployment)
- **Lucide React** (consistent healthcare and research iconography)

---

## 3. Getting Started

### Prerequisites

- **Node.js**: v18.0.0 or higher (v22.x recommended)
- **npm**: v9.0.0 or higher

### Installation

1. Clone or navigate to the project directory:
   ```bash
   cd "Infosys Project"
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   *(On Windows PowerShell: `Copy-Item .env.example .env`)*

4. Launch the local development server:
   ```bash
   npm run dev
   ```

5. Open your browser and navigate to the local Vite URL (typically `http://localhost:5173`).

---

## 4. Environment Variables (`.env.example`)

The application provides a centralized configuration via Vite environment variables:

| Variable | Default Value | Description |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000/api/v1` | Provisional base URL for the backend API |
| `VITE_USE_MOCK` | `true` | Set to `true` for frontend demo mode; `false` to route calls through `apiClient` |
| `VITE_SUPPORTED_IMAGE_FORMATS` | `.dcm,.png,.jpg,.jpeg,.tif,.tiff` | Permitted angiography image extensions |
| `VITE_MAX_UPLOAD_SIZE_MB` | `50` | Maximum file size allowed in the upload interface |

---

## 5. Mock Data Architecture & Persistence

To enable evaluation prior to backend deployment:

- **Separate Mock Layer:** Mock cases and data stores reside in `src/mocks/mockCases.js` and `src/services/caseService.js`, completely decoupled from UI presentation components.
- **Client-Side Synchronization:** Cases are held in a shared `CaseContext` backed by `localStorage` (`arteryai_mock_cases`). When you create a case or simulate an upload, all views (Dashboard, Case History, Case Details, Preprocessing Status) update immediately without page refresh.
- **Reset Demo Data:** A convenient **"Reset Demo Data"** button in the top navigation bar allows you to restore the initial benchmark cases at any time.
- **Demo Mode Indicator:** A small, clickable **"Demo Mode"** badge is present in the navigation bar, offering an explanatory modal detailing the synthetic dataset characteristics.

---

## 6. Provisional API Specification for Backend Team

The frontend services (`src/services/`) are organized to seamlessly transition to a live REST API by flipping `VITE_USE_MOCK=false`. The backend engineering team should review and confirm the following provisional endpoints:

### Authentication (`src/services/authService.js`)
- `POST /auth/login` — Researcher authentication (returns JWT `token` and `user` object).
- `POST /auth/register` — Researcher registration.
- `GET /auth/me` — Retrieve session details of the currently authenticated researcher.
- `POST /auth/forgot-password` — Dispatch password reset link / token.
- `POST /auth/logout` — Invalidate user token.

### Case Management (`src/services/caseService.js`)
- `GET /cases` — List all study cases (supports query params `?search=...&status=...`).
- `GET /cases/:id` — Retrieve detailed record for a specific research case.
- `POST /cases` — Register a new study case container with synthetic ID and metadata.
- `PATCH /cases/:id/status` — Update case workflow status.

### Scan Intake & Inference (`src/services/uploadService.js`)
- `POST /cases/:id/upload` — Multipart/form-data upload of DICOM or XCA frame.
- `GET /cases/:id/preprocessing` — Retrieve current image preprocessing status and normalization artifacts.
- `GET /cases/:id/segmentation-results` — Retrieve deep learning vessel masks and stenosis localization once models are integrated.

---

## 7. Milestone 1 Demonstration Sequence

Use the following step-by-step walkthrough to present the platform to faculty mentors and review committees:

1. **Researcher Login & Authentication:**
   - Navigate to `/login`.
   - Observe the clean medical research layout, required field validation, and password show/hide toggle.
   - Click **"Auto-fill demo credentials"** to fill sample researcher credentials (`researcher@arteryai.org`), then click **"Sign In to Platform"**.
   - Optional: Click **"Forgot password?"** to demonstrate the simulated password reset modal.
   - Navigate to `/register` to inspect the researcher registration screen and password confirmation checks.

2. **Main Research Dashboard:**
   - Once signed in, you land on `/dashboard`.
   - View the welcome banner and concise project scope statement highlighting the ARCADE XCA dataset.
   - Inspect the **4 Summary Metric Cards** (`Total Cases`, `Awaiting Processing`, `Completed Processing`, `Failed Jobs`). These statistics are dynamically computed from the active cases.
   - Review the **Recent Research Cases** table displaying Case IDs, Vessel Targets, and Status Badges.

3. **Registering a New Study Case:**
   - Click **"Create Case"** in the top navigation or dashboard.
   - Enter a study name (e.g., `ARCADE Study #088 - Diagonal Branch Analysis`), select a Target Vessel (e.g., `Left Circumflex Artery (LCx)`), and choose an Angiography Modality.
   - Notice the automatically calculated next synthetic case ID (`ARC-2025-007`).
   - Click **"Save Research Case"**. You are smoothly redirected to the new case details screen.

4. **Case History & Filtering:**
   - Click **"Case History"** in the left sidebar.
   - Notice that the newly created case is immediately visible in the table.
   - Test the live search input by typing `LAD` or `ARC-2025-001`.
   - Test the status filter dropdown (e.g., filter by `Completed` or `Awaiting Upload`).
   - Clear filters using the **"Reset"** button.

5. **Uploading an Angiography Scan:**
   - Click **"Upload Scan"** from the sidebar or click **"Upload Scan"** inside a case without a file.
   - Select your target case from the dropdown.
   - Drag and drop or browse to select a sample file (e.g., a `.png` or `.dcm` image).
   - Observe the file size and name verification.
   - Click **"Start Simulated Upload"**. Watch the simulated progress bar run. Upon completion, a success confirmation is shown with a link to the case.

6. **Image Preprocessing Pipeline Monitor:**
   - Click **"Preprocessing Status"** in the left sidebar.
   - Review the 5 documented pipeline stages:
     1. *Image Not Uploaded*
     2. *Awaiting Preprocessing*
     3. *Preprocessing In Progress*
     4. *Preprocessing Completed*
     5. *Preprocessing Failed*
   - Test the **"Advance Stage"** button in the case table to interactively demonstrate stage progression.

7. **Case Details & AI Placeholder Notice:**
   - Open any case (e.g., `ARC-2025-001`).
   - View the angiography acquisition metadata and frame representation.
   - Note the right-hand panel clearly marking **AI Artery Segmentation & Stenosis Analysis** as a future milestone feature, complying strictly with research integrity guidelines.

8. **Resetting Demo Data:**
   - Click the **"Reset Demo Data"** button in the top navbar to instantly return the repository to the default benchmark cases.

---

## 8. Folder Structure

```
├── .env.example              # Template environment configuration
├── index.html                # Application root HTML with Inter typography & favicon
├── package.json              # Project scripts and dependencies
├── postcss.config.js         # PostCSS configuration
├── tailwind.config.js        # Medical research theme tokens (navy, brand blue)
├── vite.config.js            # Vite configuration
└── src/
    ├── App.jsx               # React Router route registry
    ├── index.css             # Tailwind base and custom scrollbar
    ├── main.jsx              # Application entry point with Context providers
    ├── components/
    │   ├── Button.jsx        # Accessible buttons with variants and loaders
    │   ├── DemoBadge.jsx     # Demo mode status indicator & modal
    │   ├── EmptyState.jsx    # Standardized empty repository state
    │   ├── FormInput.jsx     # Reusable inputs with password show/hide
    │   ├── LoadingState.jsx  # Medical spinner & progress state
    │   ├── Navbar.jsx        # Top bar with user profile & demo helpers
    │   ├── ProtectedRoute.jsx# Auth route guard
    │   ├── Sidebar.jsx       # Left navigation with active state styling
    │   └── StatusBadge.jsx   # Status pill badge for cases
    ├── context/
    │   ├── AuthContext.jsx   # Researcher authentication & session management
    │   └── CaseContext.jsx   # Shared case store with live sync & derived stats
    ├── layouts/
    │   ├── AuthLayout.jsx    # Centered layout for Login & Register
    │   └── DashboardLayout.jsx# Responsive app layout with sidebar & navbar
    ├── mocks/
    │   └── mockCases.js      # Synthetic ARCADE cases and benchmark metadata
    ├── pages/
    │   ├── CaseDetails.jsx   # Case inspection & model placeholder notices
    │   ├── CaseHistory.jsx   # Case repository with live search and filtering
    │   ├── CreateCase.jsx    # Case registration form with validation
    │   ├── Dashboard.jsx     # Overview with metrics derived from mock cases
    │   ├── Login.jsx         # Sign in with validation and demo auto-fill
    │   ├── NotFound.jsx      # 404 handler
    │   ├── PreprocessingStatus.jsx # 5-stage pipeline monitor
    │   ├── Register.jsx      # Researcher registration form
    │   └── UploadScan.jsx    # Drag-and-drop scan intake with simulated progress
    ├── services/
    │   ├── api.js            # Axios client with interceptors
    │   ├── authService.js    # Provisional auth API service
    │   ├── caseService.js    # Provisional case API service
    │   └── uploadService.js  # Provisional upload & status service
    └── utils/
        ├── constants.js      # Statuses, ARCADE modalities, preprocessing stages
        └── formatters.js     # Date, byte size, and synthetic ID formatters
```
