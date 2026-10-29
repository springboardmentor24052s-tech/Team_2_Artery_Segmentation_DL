/**
 * Application Constants
 * ArteryAI: Artery Segmentation and Stenosis Detection Research Platform
 * Dataset context: ARCADE (X-ray Coronary Angiography)
 */

export const APP_NAME = 'ArteryAI';
export const PROJECT_TITLE = 'Artery Segmentation and Stenosis Detection Using Advanced Deep Learning Techniques';
export const DATASET_NAME = 'ARCADE Dataset (X-ray Coronary Angiography)';

// Case Statuses used in the platform workflow
export const CASE_STATUS = {
  CREATED: 'Created',
  AWAITING_UPLOAD: 'Awaiting Upload',
  UPLOADED: 'Uploaded',
  PROCESSING: 'Processing',
  COMPLETED: 'Completed',
  FAILED: 'Failed',
};

// Color badge variants corresponding to statuses
export const STATUS_STYLES = {
  [CASE_STATUS.CREATED]: {
    bg: 'bg-slate-100',
    text: 'text-slate-700',
    border: 'border-slate-200',
    dot: 'bg-slate-400',
  },
  [CASE_STATUS.AWAITING_UPLOAD]: {
    bg: 'bg-amber-50',
    text: 'text-amber-700',
    border: 'border-amber-200',
    dot: 'bg-amber-500',
  },
  [CASE_STATUS.UPLOADED]: {
    bg: 'bg-blue-50',
    text: 'text-blue-700',
    border: 'border-blue-200',
    dot: 'bg-blue-500',
  },
  [CASE_STATUS.PROCESSING]: {
    bg: 'bg-indigo-50',
    text: 'text-indigo-700',
    border: 'border-indigo-200',
    dot: 'bg-indigo-500',
  },
  [CASE_STATUS.COMPLETED]: {
    bg: 'bg-emerald-50',
    text: 'text-emerald-700',
    border: 'border-emerald-200',
    dot: 'bg-emerald-500',
  },
  [CASE_STATUS.FAILED]: {
    bg: 'bg-rose-50',
    text: 'text-rose-700',
    border: 'border-rose-200',
    dot: 'bg-rose-500',
  },
};

// Modalities tailored specifically to ARCADE X-ray Coronary Angiography
export const IMAGE_MODALITIES = [
  { id: 'XCA-2D', label: 'X-ray Coronary Angiography (2D XCA Frame)', description: 'Standard 2D projection frame from ARCADE coronary sequence' },
  { id: 'DICOM-CINE', label: 'DICOM Cineangiography (Multi-frame X-ray)', description: 'Full cardiac cycle cine sequence from fluoroscopic angiography' },
  { id: 'DSA-2D', label: 'Digital Subtraction Angiogram (DSA)', description: 'Contrast-enhanced 2D angiographic projection with background attenuation' },
];

// Target anatomical vessel regions studied in ARCADE
export const TARGET_REGIONS = [
  'Left Anterior Descending (LAD)',
  'Left Circumflex Artery (LCx)',
  'Right Coronary Artery (RCA)',
  'Left Main Coronary Artery (LMCA)',
  'Bifurcation / Branch Region',
  'Full Coronary Tree (Unspecified)',
];

// Preprocessing pipeline stages
export const PREPROCESSING_STAGES = [
  {
    key: 'NOT_UPLOADED',
    title: 'Image Not Uploaded',
    badgeText: 'Pending Upload',
    statusClass: 'bg-slate-100 text-slate-700 border-slate-200',
    description: 'No angiogram image file has been associated with this study case yet.',
    actionRequired: 'Upload an X-ray coronary angiography scan (DICOM, PNG, or TIFF) to begin.'
  },
  {
    key: 'AWAITING_PREPROCESSING',
    title: 'Awaiting Preprocessing',
    badgeText: 'In Queue',
    statusClass: 'bg-amber-100 text-amber-800 border-amber-200',
    description: 'The scan file is verified and waiting in the pipeline queue for image preparation.',
    actionRequired: 'Queued for automated contrast enhancement and resolution standardization.'
  },
  {
    key: 'PREPROCESSING_IN_PROGRESS',
    title: 'Preprocessing In Progress',
    badgeText: 'Running',
    statusClass: 'bg-blue-100 text-blue-800 border-blue-200',
    description: 'Applying adaptive histogram equalization (CLAHE), frame isolation, and artifact attenuation.',
    actionRequired: 'Pipeline currently executing image normalization routines.'
  },
  {
    key: 'PREPROCESSING_COMPLETED',
    title: 'Preprocessing Completed',
    badgeText: 'Ready for Segmentation',
    statusClass: 'bg-emerald-100 text-emerald-800 border-emerald-200',
    description: 'Angiogram frame successfully normalized, rescaled to 512x512, and ready for model inference.',
    actionRequired: 'Scan prepared. Ready for downstream deep learning segmentation pipeline once connected.'
  },
  {
    key: 'PREPROCESSING_FAILED',
    title: 'Preprocessing Failed',
    badgeText: 'Needs Attention',
    statusClass: 'bg-rose-100 text-rose-800 border-rose-200',
    description: 'Corrupted DICOM headers, unreadable bit depth, or unsupported aspect ratio encountered.',
    actionRequired: 'Please re-upload a clean, uncorrupted angiography image file.'
  }
];

// Configurable upload parameters
export const UPLOAD_CONFIG = {
  supportedFormats: (import.meta.env.VITE_SUPPORTED_IMAGE_FORMATS || '.dcm,.png,.jpg,.jpeg,.tif,.tiff').split(','),
  maxSizeMB: Number(import.meta.env.VITE_MAX_UPLOAD_SIZE_MB) || 50,
  maxSizeBytes: (Number(import.meta.env.VITE_MAX_UPLOAD_SIZE_MB) || 50) * 1024 * 1024,
};
