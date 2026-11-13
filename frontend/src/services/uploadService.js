
/**
 * Scan Upload & Segmentation Inference Service
 *
 * Mock mode:
 *   Stores a local preview and simulates upload progress.
 *
 * Real mode:
 *   Uploads the image, runs the selected segmentation model,
 *   and persists the inference result through the backend.
 */

import { apiClient, IS_MOCK_MODE } from './api';
import { caseService } from './caseService';
import { CASE_STATUS } from '../utils/constants';
import { saveScanAsset } from './scanAssetService';

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const SUPPORTED_IMAGE_TYPES = new Set([
  'image/png',
  'image/jpeg',
  'image/jpg',
  'image/tiff',
  'image/bmp',
]);

export const uploadService = {
  async uploadScan(
    caseId,
    file,
    modelType = 'unet',
    onProgress = () => {}
  ) {
    // Backward compatibility with callers that pass the callback
    // as the third argument.
    if (typeof modelType === 'function') {
      onProgress = modelType;
      modelType = 'unet';
    }

    if (!file) {
      throw new Error('Please select an image before uploading.');
    }

    if (!['unet', 'attention_unet'].includes(modelType)) {
      throw new Error('Please select a supported segmentation model.');
    }

    // ---------------------------------------------------------
    // MOCK MODE
    // ---------------------------------------------------------
    if (IS_MOCK_MODE) {
      try {
        await saveScanAsset(caseId, file);
      } catch (storageError) {
        console.warn(
          'Local scan preview could not be stored:',
          storageError
        );
      }

      for (let percent = 10; percent <= 100; percent += 15) {
        await delay(120);
        onProgress(Math.min(percent, 100));
      }

      const fileData = {
        name: file.name,
        size: file.size,
        type: file.type || 'application/octet-stream',
        uploadedAt: new Date().toISOString(),
        dimensions: '512 x 512 px',
        dicomMeta: {
          photometric: 'MONOCHROME2',
          kvp: '80 kV',
          frameRate: '15 fps',
          fieldOfView: '200 mm',
        },
      };

      await caseService.updateCase(caseId, {
        status: CASE_STATUS.UPLOADED,
        preprocessingStage: 'AWAITING_PREPROCESSING',
        file: fileData,
      });

      return {
        success: true,
        isMock: true,
        caseId: String(caseId),
        file: fileData,
        modelType,
        message: 'Mock scan upload completed successfully.',
      };
    }

    // ---------------------------------------------------------
    // REAL BACKEND MODE
    // ---------------------------------------------------------

    if (!SUPPORTED_IMAGE_TYPES.has(file.type)) {
      throw new Error(
        'Unsupported image format. Real inference currently accepts PNG, JPG, JPEG, TIFF and BMP images. DICOM is not supported by this endpoint.'
      );
    }

    const numericCaseId = Number(caseId);

    if (!Number.isInteger(numericCaseId) || numericCaseId < 1) {
      throw new Error('A valid backend case ID is required.');
    }

    const formData = new FormData();

    formData.append('file', file);
    formData.append('case_id', String(numericCaseId));
    formData.append('model_type', modelType);

    const response = await apiClient.post(
      '/inference/upload-and-predict',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000,
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percent = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );

            onProgress(percent);
          }
        },
      }
    );

    const result = response.data;

    if (
      !result ||
      !result.segmentation_id ||
      !result.image_id
    ) {
      throw new Error(
        'The backend response did not contain a valid segmentation result.'
      );
    }

    const fileData = {
      name: file.name,
      size: file.size,
      type: file.type,
      uploadedAt: result.created_at || new Date().toISOString(),
      dimensions: `${result.original_width} x ${result.original_height} px`,
    };

    return {
      success: true,
      isMock: false,
      caseId: String(result.case_id),
      file: fileData,

      // Persisted backend identifiers and measurements.
      imageId: result.image_id,
      segmentationId: result.segmentation_id,
      modelName: result.model_name,
      threshold: result.threshold,
      originalWidth: result.original_width,
      originalHeight: result.original_height,
      segmentedPixels: result.segmented_pixels,
      vesselAreaRatio: result.vessel_area_ratio,

      // Base64-encoded PNG image data from the backend.
      maskBase64: result.mask_base64,
      overlayBase64: result.overlay_base64,

      // Keep the original backend response available.
      inference: result,

      message: 'Image uploaded and segmentation inference completed successfully.',
    };
  },

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

    throw new Error(
      'A dedicated preprocessing-status endpoint is not available in the backend.'
    );
  },

  async getSegmentationResults(caseId) {
    if (IS_MOCK_MODE) {
      return {
        isAvailable: false,
        message: 'Mock mode does not run real segmentation inference.',
        caseId,
        masks: null,
        stenosisAnalysis: null,
      };
    }

    const numericCaseId = Number(caseId);

    if (!Number.isInteger(numericCaseId) || numericCaseId < 1) {
      throw new Error('A valid backend case ID is required.');
    }

    const response = await apiClient.get(
      `/sql/cases/${numericCaseId}/segmentation-results`
    );

    return response.data;
  },
};
