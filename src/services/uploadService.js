/**
 * Upload & Preprocessing Service (Provisional Endpoints)
 * 
 * Manages file uploads, preprocessing pipeline status polling,
 * and future deep learning segmentation result retrieval.
 */

import { apiClient, IS_MOCK_MODE } from './api';
import { caseService } from './caseService';
import { CASE_STATUS } from '../utils/constants';

export const uploadService = {
  /**
   * Upload an angiography scan file associated with a research case
   * Provisional Backend Route: POST /cases/:id/upload
   * Content-Type: multipart/form-data
   */
  async uploadScan(caseId, file, onProgress = () => {}) {
    if (IS_MOCK_MODE) {
      // Simulate stepped upload progress
      for (let percent = 10; percent <= 100; percent += 15) {
        await new Promise((r) => setTimeout(r, 120));
        onProgress(Math.min(percent, 100));
      }

      // Attach file metadata to the mock case
      const fileData = {
        name: file.name,
        size: file.size,
        type: file.type || 'application/dicom',
        uploadedAt: new Date().toISOString(),
        dimensions: '512 x 512 px',
        dicomMeta: {
          photometric: 'MONOCHROME2',
          kvp: '80 kV',
          frameRate: '15 fps',
          fieldOfView: '200 mm'
        }
      };

      await caseService.updateCase(caseId, {
        status: CASE_STATUS.UPLOADED,
        preprocessingStage: 'AWAITING_PREPROCESSING',
        file: fileData,
      });

      return {
        success: true,
        caseId,
        file: fileData,
        message: 'Scan uploaded and verified successfully.',
      };
    }

    // Real API implementation
    const formData = new FormData();
    formData.append('file', file);
    formData.append('caseId', caseId);

    const response = await apiClient.post(`/cases/${caseId}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(percentCompleted);
        }
      },
    });

    return response.data;
  },

  /**
   * Retrieve preprocessing status for a case
   * Provisional Backend Route: GET /cases/:id/preprocessing
   */
  async getPreprocessingStatus(caseId) {
    if (IS_MOCK_MODE) {
      const targetCase = await caseService.getCaseById(caseId);
      return {
        caseId: targetCase.id,
        stage: targetCase.preprocessingStage || 'NOT_UPLOADED',
        lastUpdated: new Date().toISOString(),
        filePresent: Boolean(targetCase.file),
      };
    }

    const response = await apiClient.get(`/cases/${caseId}/preprocessing`);
    return response.data;
  },

  /**
   * Retrieve future AI segmentation outputs
   * Provisional Backend Route: GET /cases/:id/segmentation-results
   * 
   * NOTE: As per project guidelines, automated AI segmentation and stenosis
   * calculations are not yet connected. This function returns an explicit
   * not-available response.
   */
  async getSegmentationResults(caseId) {
    if (IS_MOCK_MODE) {
      return {
        isAvailable: false,
        message: 'AI segmentation model is not yet connected. Deep learning inference will be integrated in a future milestone.',
        caseId,
        masks: null,
        stenosisAnalysis: null,
      };
    }

    const response = await apiClient.get(`/cases/${caseId}/segmentation-results`);
    return response.data;
  }
};
