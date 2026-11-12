/**
 * Mock Data Layer: Synthetic ARCADE Research Cases
 * 
 * IMPORTANT:
 * - All cases are strictly synthetic research identifiers derived from the public ARCADE research dataset.
 * - No real patient identifiers or protected health information (PHI) are included or stored.
 * - ARCADE uses X-ray coronary angiography (XCA).
 * - No 3D models or automated clinical diagnoses are present.
 */

import { CASE_STATUS } from '../utils/constants';

export const INITIAL_MOCK_CASES = [
  {
    id: 'ARC-2025-001',
    name: 'ARCADE Study #014 - LAD Stenosis Investigation',
    description: 'X-ray angiography frame assessing luminal narrowing in proximal and mid Left Anterior Descending artery.',
    targetRegion: 'Left Anterior Descending (LAD)',
    imageType: '2D XCA Frame',
    imageModality: 'X-ray Coronary Angiography (2D XCA Frame)',
    createdAt: '2025-02-10T09:30:00.000Z',
    status: CASE_STATUS.COMPLETED,
    preprocessingStage: 'PREPROCESSING_COMPLETED',
    file: {
      name: 'arcade_patient_014_frame_028.png',
      size: 1428500, // ~1.36 MB
      type: 'image/png',
      uploadedAt: '2025-02-10T09:35:12.000Z',
      dimensions: '512 x 512 px',
      dicomMeta: {
        photometric: 'MONOCHROME2',
        kvp: '78 kV',
        frameRate: '15 fps',
        fieldOfView: '200 mm'
      }
    },
    notes: 'Standard RAO Cranial 30° projection from ARCADE benchmark suite.',
    isSynthetic: true,
  },
  {
    id: 'ARC-2025-002',
    name: 'ARCADE Study #029 - LCx Bifurcation Analysis',
    description: 'Coronary cineangiogram focused on the Left Circumflex artery and first obtuse marginal branch bifurcation.',
    targetRegion: 'Left Circumflex Artery (LCx)',
    imageType: 'DICOM Cine',
    imageModality: 'DICOM Cineangiography (Multi-frame X-ray)',
    createdAt: '2025-02-11T11:15:00.000Z',
    status: CASE_STATUS.PROCESSING,
    preprocessingStage: 'PREPROCESSING_IN_PROGRESS',
    file: {
      name: 'arcade_study_029_cine_seq.dcm',
      size: 14200000, // ~13.5 MB
      type: 'application/dicom',
      uploadedAt: '2025-02-11T11:20:45.000Z',
      dimensions: '512 x 512 px (60 frames)',
      dicomMeta: {
        photometric: 'MONOCHROME2',
        kvp: '82 kV',
        frameRate: '30 fps',
        fieldOfView: '180 mm'
      }
    },
    notes: 'Queued for CLAHE contrast enhancement and cardiac motion compensation.',
    isSynthetic: true,
  },
  {
    id: 'ARC-2025-003',
    name: 'ARCADE Study #038 - Right Coronary Artery Evaluation',
    description: 'Single-frame projection of dominant Right Coronary Artery to test vessel wall boundary delineations.',
    targetRegion: 'Right Coronary Artery (RCA)',
    imageType: '2D XCA Frame',
    imageModality: 'X-ray Coronary Angiography (2D XCA Frame)',
    createdAt: '2025-02-12T14:40:00.000Z',
    status: CASE_STATUS.UPLOADED,
    preprocessingStage: 'AWAITING_PREPROCESSING',
    file: {
      name: 'arcade_rca_projection_038.dcm',
      size: 2650000, // ~2.5 MB
      type: 'application/dicom',
      uploadedAt: '2025-02-12T14:45:10.000Z',
      dimensions: '512 x 512 px',
      dicomMeta: {
        photometric: 'MONOCHROME1',
        kvp: '75 kV',
        frameRate: '15 fps',
        fieldOfView: '220 mm'
      }
    },
    notes: 'LAO 45° projection; verified file integrity, waiting in image preprocessing queue.',
    isSynthetic: true,
  },
  {
    id: 'ARC-2025-004',
    name: 'ARCADE Study #045 - Left Main Coronary Artery Baseline',
    description: 'Candidate study record created for ARCADE validation set image intake.',
    targetRegion: 'Left Main Coronary Artery (LMCA)',
    imageType: '2D XCA Frame',
    imageModality: 'X-ray Coronary Angiography (2D XCA Frame)',
    createdAt: '2025-02-13T08:10:00.000Z',
    status: CASE_STATUS.AWAITING_UPLOAD,
    preprocessingStage: 'NOT_UPLOADED',
    file: null,
    notes: 'Case record created by research coordinator; awaiting XCA scan upload.',
    isSynthetic: true,
  },
  {
    id: 'ARC-2025-005',
    name: 'ARCADE Study #051 - Low Contrast Fluoroscopy Scan',
    description: 'Experimental frame with severe radiation attenuation and contrast dilution.',
    targetRegion: 'Bifurcation / Branch Region',
    imageType: '2D DSA',
    imageModality: 'Digital Subtraction Angiogram (DSA)',
    createdAt: '2025-02-13T16:20:00.000Z',
    status: CASE_STATUS.FAILED,
    preprocessingStage: 'PREPROCESSING_FAILED',
    file: {
      name: 'arcade_corrupt_header_051.dcm',
      size: 980000,
      type: 'application/dicom',
      uploadedAt: '2025-02-13T16:22:00.000Z',
      dimensions: 'Unreadable Header',
      dicomMeta: null
    },
    notes: 'Preprocessing aborted: missing mandatory DICOM Photometric Interpretation tag.',
    isSynthetic: true,
  },
  {
    id: 'ARC-2025-006',
    name: 'ARCADE Study #060 - LAD & Diagonal Branch Protocol',
    description: 'Initial case container configured for multi-observer consensus comparison.',
    targetRegion: 'Left Anterior Descending (LAD)',
    imageType: '2D XCA Frame',
    imageModality: 'X-ray Coronary Angiography (2D XCA Frame)',
    createdAt: '2025-02-14T10:05:00.000Z',
    status: CASE_STATUS.CREATED,
    preprocessingStage: 'NOT_UPLOADED',
    file: null,
    notes: 'Study template initialized.',
    isSynthetic: true,
  }
];

export const MOCK_USER = {
  id: 'usr-research-demo-01',
  name: 'Dr. Jane Chen (Research Fellow)',
  email: 'researcher@arteryai.org',
  role: 'Research Student / Investigator',
  institution: 'Department of Biomedical Engineering',
  isDemoUser: true,
};
