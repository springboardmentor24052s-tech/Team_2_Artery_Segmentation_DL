import { apiClient } from './api';

const API_ORIGIN = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1').replace(/\/api\/v1\/?$/, '');

export const aiService = {
  async analyzeImage(caseId, file, onProgress = () => {}) {
    const form = new FormData();
    form.append('file', file);
    form.append('caseId', caseId);
    const response = await apiClient.post('/ai/analyze', form, {
      timeout: 120000,
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (event) => {
        if (event.total) onProgress(Math.round(event.loaded * 100 / event.total));
      },
    });
    const result = response.data;
    const absolute = (url) => url ? `${API_ORIGIN}${url}` : null;
    result.originalUrl = absolute(result.originalUrl);
    if (result.vesselSegmentation) {
      result.vesselSegmentation.maskUrl = absolute(result.vesselSegmentation.maskUrl);
      result.vesselSegmentation.overlayUrl = absolute(result.vesselSegmentation.overlayUrl);
    }
    if (result.stenosisDetection) result.stenosisDetection.maskUrl = absolute(result.stenosisDetection.maskUrl);
    localStorage.setItem(`arteryai_ai_result_${caseId}`, JSON.stringify(result));
    return result;
  },
  getResult(caseId) {
    try { return JSON.parse(localStorage.getItem(`arteryai_ai_result_${caseId}`) || 'null'); }
    catch { return null; }
  },
};
