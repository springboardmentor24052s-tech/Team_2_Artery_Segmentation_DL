/**
 * Case Service (Provisional Endpoints)
 * 
 * Provides research case management methods.
 * In Milestone 1, operates with client-side mock store that synchronizes
 * across pages via CaseContext / localStorage.
 */

import { apiClient, IS_MOCK_MODE } from './api';
import { INITIAL_MOCK_CASES } from '../mocks/mockCases';
import { generateSyntheticCaseId } from '../utils/formatters';

const STORAGE_KEY = 'arteryai_mock_cases';

// Helper to get cases from localStorage or initial mock data
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

export const caseService = {
  /**
   * Fetch all research cases
   * Provisional Backend Route: GET /cases
   */
  async getCases(filters = {}) {
    if (IS_MOCK_MODE) {
      await delay(250);
      let cases = getStoredMockCases();

      if (filters.search) {
        const query = filters.search.toLowerCase().trim();
        cases = cases.filter(
          (c) =>
            c.id.toLowerCase().includes(query) ||
            c.name.toLowerCase().includes(query) ||
            c.targetRegion?.toLowerCase().includes(query)
        );
      }

      if (filters.status && filters.status !== 'All') {
        cases = cases.filter((c) => c.status === filters.status);
      }

      return cases;
    }

    const response = await apiClient.get('/cases', { params: filters });
    return response.data;
  },

  /**
   * Fetch case details by case ID
   * Provisional Backend Route: GET /cases/:id
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

    const response = await apiClient.get(`/cases/${id}`);
    return response.data;
  },

  /**
   * Create a new research case
   * Provisional Backend Route: POST /cases
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
        imageModality: caseData.imageModality || 'X-ray Coronary Angiography (2D XCA Frame)',
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

    const response = await apiClient.post('/cases', caseData);
    return response.data;
  },

  /**
   * Update case status or file attachment
   * Provisional Backend Route: PATCH /cases/:id/status
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

    const response = await apiClient.patch(`/cases/${id}/status`, updates);
    return response.data;
  },

  /**
   * Reset mock cases back to default benchmark suite
   */
  resetMockData() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(INITIAL_MOCK_CASES));
    return INITIAL_MOCK_CASES;
  }
};
