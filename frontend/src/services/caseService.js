
/**
 * Case Service
 *
 * Supports both:
 * - Mock mode: localStorage-backed research cases.
 * - Real mode: FastAPI SQL case management endpoints.
 */

import { apiClient, IS_MOCK_MODE } from './api';
import { INITIAL_MOCK_CASES } from '../mocks/mockCases';
import { generateSyntheticCaseId } from '../utils/formatters';

const STORAGE_KEY = 'arteryai_mock_cases';

// --------------------------------------------------
// Mock storage helpers
// --------------------------------------------------

function getStoredMockCases() {
  const stored = localStorage.getItem(STORAGE_KEY);

  if (stored) {
    try {
      return JSON.parse(stored);
    } catch {
      return INITIAL_MOCK_CASES;
    }
  }

  localStorage.setItem(STORAGE_KEY, JSON.stringify(INITIAL_MOCK_CASES));
  return INITIAL_MOCK_CASES;
}

function saveStoredMockCases(cases) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(cases));
}

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// --------------------------------------------------
// Backend <-> Frontend mapping helpers
// --------------------------------------------------

function mapBackendStatus(status) {
  const normalized = String(status || '').trim().toLowerCase();

  const statusMap = {
    pending: 'Created',
    created: 'Created',
    awaiting_upload: 'Awaiting Upload',
    'awaiting upload': 'Awaiting Upload',
    uploaded: 'Uploaded',
    processing: 'Processing',
    in_progress: 'Processing',
    completed: 'Completed',
    complete: 'Completed',
    failed: 'Failed',
    error: 'Failed',
  };

  return statusMap[normalized] || 'Created';
}

function mapFrontendStatus(status) {
  const normalized = String(status || '').trim().toLowerCase();

  const statusMap = {
    created: 'pending',
    'awaiting upload': 'awaiting_upload',
    uploaded: 'uploaded',
    processing: 'processing',
    completed: 'completed',
    failed: 'failed',
  };

  return statusMap[normalized] || status;
}

function mapBackendImageType(modality) {
  const normalized = String(modality || '').toLowerCase();

  if (normalized.includes('dicom cine')) {
    return 'DICOM Cine';
  }

  if (normalized.includes('dsa')) {
    return '2D DSA';
  }

  return '2D XCA Frame';
}

function mapBackendCase(row) {
  return {
    id: String(row.case_id),
    name: row.case_name || 'Untitled Research Case',
    description: '',
    targetRegion: row.target_vessel || 'Unspecified',
    imageType: mapBackendImageType(row.imaging_modality),
    imageModality: row.imaging_modality || 'CTA',
    createdAt: row.created_at || null,
    status: mapBackendStatus(row.status),

    // These are frontend workflow fields, not fields
    // currently persisted by the backend case schema.
    preprocessingStage: 'NOT_UPLOADED',
    file: null,
    notes: '',

    userId: row.user_id || null,
    isSynthetic: false,
  };
}

function filterCases(cases, filters = {}) {
  let filtered = cases;

  if (filters.search) {
    const query = filters.search.toLowerCase().trim();

    filtered = filtered.filter((c) =>
      String(c.id || '').toLowerCase().includes(query) ||
      String(c.name || '').toLowerCase().includes(query) ||
      String(c.targetRegion || '').toLowerCase().includes(query)
    );
  }

  if (filters.status && filters.status !== 'All') {
    filtered = filtered.filter((c) => c.status === filters.status);
  }

  return filtered;
}

function validateBackendCaseId(id) {
  const numericId = Number(id);

  if (!Number.isInteger(numericId) || numericId < 1) {
    throw new Error(
      `Invalid backend case ID "${id}". Real backend case IDs must be positive integers.`
    );
  }

  return numericId;
}

// --------------------------------------------------
// Case service
// --------------------------------------------------

export const caseService = {
  /**
   * Fetch all research cases.
   *
   * Mock: localStorage
   * Real: GET /sql/cases
   */
  async getCases(filters = {}) {
    if (IS_MOCK_MODE) {
      await delay(250);
      return filterCases(getStoredMockCases(), filters);
    }

    const response = await apiClient.get('/sql/cases');

    const cases = (response.data || []).map(mapBackendCase);

    // Search and status filters are applied locally because
    // the confirmed backend list endpoint does not accept them.
    return filterCases(cases, filters);
  },

  /**
   * Fetch case details by ID.
   *
   * Mock: localStorage
   * Real: GET /sql/cases/{case_id}
   */
  async getCaseById(id) {
    if (IS_MOCK_MODE) {
      await delay(200);

      const cases = getStoredMockCases();
      const found = cases.find((c) => c.id === id);

      if (!found) {
        throw new Error(`Case with identifier "${id}" was not found.`);
      }

      return found;
    }

    const caseId = validateBackendCaseId(id);
    const response = await apiClient.get(`/sql/cases/${caseId}`);

    return mapBackendCase(response.data);
  },

  /**
   * Create a new research case.
   *
   * Mock: generates a synthetic ARCADE identifier.
   * Real: POST /sql/cases
   */
  async createCase(caseData) {
    if (IS_MOCK_MODE) {
      await delay(400);

      const existingCases = getStoredMockCases();
      const newId = generateSyntheticCaseId(existingCases);

      const newCase = {
        id: newId,
        name: caseData.name.trim(),
        description: caseData.description?.trim() || 'No description provided.',
        targetRegion: caseData.targetRegion || 'Full Coronary Tree (Unspecified)',
        imageType: caseData.imageType || '2D XCA Frame',
        imageModality:
          caseData.imageModality || 'X-ray Coronary Angiography (2D XCA Frame)',
        createdAt: new Date().toISOString(),
        status: 'Created',
        preprocessingStage: 'NOT_UPLOADED',
        file: null,
        notes: caseData.notes?.trim() || 'Synthetically registered research case.',
        isSynthetic: true,
      };

      const updated = [newCase, ...existingCases];
      saveStoredMockCases(updated);

      return newCase;
    }

    const payload = {
      case_name: caseData.name.trim(),
      imaging_modality: caseData.imageModality || 'CTA',
      target_vessel: caseData.targetRegion || 'LAD',
      status: 'pending',
    };

    const response = await apiClient.post('/sql/cases', payload);

    return mapBackendCase(response.data);
  },

  /**
   * Update supported case metadata or status.
   *
   * Mock: supports the existing local workflow updates.
   * Real: PATCH /sql/cases/{case_id}
   *
   * The backend case schema does not currently support
   * preprocessingStage or file updates.
   */
  async updateCase(id, updates) {
    if (IS_MOCK_MODE) {
      await delay(300);

      const cases = getStoredMockCases();
      const index = cases.findIndex((c) => c.id === id);

      if (index === -1) {
        throw new Error(`Case ${id} not found.`);
      }

      cases[index] = { ...cases[index], ...updates };
      saveStoredMockCases(cases);

      return cases[index];
    }

    if (Object.prototype.hasOwnProperty.call(updates, 'preprocessingStage')) {
      throw new Error(
        'Preprocessing stage simulation is only available in mock mode. Real preprocessing must be updated by the backend pipeline.'
      );
    }

    const caseId = validateBackendCaseId(id);
    const payload = {};

    if (updates.name !== undefined) {
      payload.case_name = updates.name;
    }

    if (updates.imageModality !== undefined) {
      payload.imaging_modality = updates.imageModality;
    }

    if (updates.targetRegion !== undefined) {
      payload.target_vessel = updates.targetRegion;
    }

    if (updates.status !== undefined) {
      payload.status = mapFrontendStatus(updates.status);
    }

    if (Object.keys(payload).length === 0) {
      return this.getCaseById(caseId);
    }

    const response = await apiClient.patch(
      `/sql/cases/${caseId}`,
      payload
    );

    return mapBackendCase(response.data);
  },

  /**
   * Reset mock cases to the initial benchmark suite.
   */
  resetMockData() {
    if (!IS_MOCK_MODE) {
      throw new Error('Resetting mock cases is only available in mock mode.');
    }

    localStorage.setItem(STORAGE_KEY, JSON.stringify(INITIAL_MOCK_CASES));

    return INITIAL_MOCK_CASES;
  },
};
