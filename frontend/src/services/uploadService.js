
/**
 * Scan Upload & Segmentation Inference Service
 *
 * Mock mode:
 *   Stores a local preview and simulates upload progress.
 *
 * Real mode:
 *   Uploads the image, runs the selected segmentation model,
 *   persists the inference result, and retrieves saved files.
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

const SUPPORTED_IMAGE_EXTENSIONS = new Set([
  '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp',
]);

function extractCollection(payload) {
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.value)) return payload.value;
  if (Array.isArray(payload?.items)) return payload.items;
  if (Array.isArray(payload?.results)) return payload.results;
  if (Array.isArray(payload?.data)) return payload.data;
  return [];
}

function mapBackendImage(image) {
  const imageId = image?.image_id ?? image?.id ?? null;
  const width = image?.original_width ?? image?.width ?? null;
  const height = image?.original_height ?? image?.height ?? null;
  const dimensions = width && height ? `${width} x ${height} px` : null;

  return {
    imageId,
    name: image?.original_filename || image?.original_name || image?.file_name || image?.filename || image?.name || (imageId ? `Image ${imageId}` : 'Uploaded image'),
    size: image?.file_size_bytes ?? image?.size_bytes ?? image?.file_size ?? image?.size ?? null,
    type: image?.mime_type || image?.content_type || image?.file_type || image?.type || '',
    uploadedAt: image?.created_at || image?.uploaded_at || null,
    dimensions,
    backendRecord: image,
  };
}

function validateImageFile(file) {
  const name = String(file?.name || '').toLowerCase();
  const extension = name.includes('.') ? name.slice(name.lastIndexOf('.')) : '';
  const mimeType = String(file?.type || '').toLowerCase();

  if (extension && SUPPORTED_IMAGE_EXTENSIONS.has(extension)) return;
  if (!extension && SUPPORTED_IMAGE_TYPES.has(mimeType)) return;

  throw new Error(
    'Unsupported image format. Real inference accepts PNG, JPG, JPEG, TIFF and BMP images. DICOM is not supported by this endpoint.'
  );
}

const validatePositiveId = (value, label) => {
  const id = Number(value);

  if (!Number.isInteger(id) || id < 1) {
    throw new Error(`A valid ${label} is required.`);
  }

  return id;
};

const ensureRealMode = () => {
  if (IS_MOCK_MODE) {
    throw new Error(
      'This operation requires real backend mode. Disable mock mode first.'
    );
  }
};

export const uploadService = {
  // ---------------------------------------------------------
  // UPLOAD SCAN AND RUN SEGMENTATION INFERENCE
  // ---------------------------------------------------------
  async uploadScan(
    caseId,
    file,
    modelType = 'unet',
    onProgress = () => {}
  ) {
    // Backward compatibility with callers passing callback
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

    // -------------------------------------------------------
    // MOCK MODE
    // -------------------------------------------------------
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

    // -------------------------------------------------------
    // REAL BACKEND MODE
    // -------------------------------------------------------
    validateImageFile(file);

    const numericCaseId = validatePositiveId(caseId, 'backend case ID');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('case_id', String(numericCaseId));
    formData.append('model_type', modelType);

    const response = await apiClient.post(
      '/inference/upload-and-predict',
      formData,
      {
        // Let the browser/Axios generate the multipart boundary.
        // Manually setting Content-Type can omit the boundary required by FastAPI.
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

    if (!result?.segmentation_id || !result?.image_id) {
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

      imageId: result.image_id,
      segmentationId: result.segmentation_id,
      modelName: result.model_name,
      threshold: result.threshold,
      originalWidth: result.original_width,
      originalHeight: result.original_height,
      segmentedPixels: result.segmented_pixels,
      vesselAreaRatio: result.vessel_area_ratio,

      maskBase64: result.mask_base64,
      overlayBase64: result.overlay_base64,

      inference: result,

      message:
        'Image uploaded and segmentation inference completed successfully.',
    };
  },

  // ---------------------------------------------------------
  // PREPROCESSING STATUS
  // ---------------------------------------------------------
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

  // ---------------------------------------------------------
  // RETRIEVE SEGMENTATION RESULTS FOR A CASE
  // ---------------------------------------------------------
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

    const numericCaseId = validatePositiveId(
      caseId,
      'backend case ID'
    );

    const response = await apiClient.get(
      `/sql/cases/${numericCaseId}/segmentation-results`
    );

    return extractCollection(response.data);
  },

  // ---------------------------------------------------------
  // RETRIEVE UPLOADED IMAGE RECORDS FOR A CASE
  // ---------------------------------------------------------
  async getCaseImages(caseId) {
    if (IS_MOCK_MODE) return [];

    const numericCaseId = validatePositiveId(caseId, 'backend case ID');
    const response = await apiClient.get(`/sql/cases/${numericCaseId}/images`);
    return extractCollection(response.data).map(mapBackendImage);
  },

  // ---------------------------------------------------------
  // DOWNLOAD ORIGINAL IMAGE
  // ---------------------------------------------------------
  async getOriginalImage(imageId) {
    ensureRealMode();

    const id = validatePositiveId(imageId, 'image ID');

    const response = await apiClient.get(
      `/inference/images/${id}/file`,
      {
        responseType: 'blob',
      }
    );

    return response.data;
  },

  // ---------------------------------------------------------
  // DOWNLOAD SEGMENTATION MASK
  // ---------------------------------------------------------
  async getSegmentationMask(segmentationId) {
    ensureRealMode();

    const id = validatePositiveId(
      segmentationId,
      'segmentation ID'
    );

    const response = await apiClient.get(
      `/inference/masks/${id}`,
      {
        responseType: 'blob',
      }
    );

    return response.data;
  },

  // ---------------------------------------------------------
  // DOWNLOAD SEGMENTATION OVERLAY
  // ---------------------------------------------------------
  async getSegmentationOverlay(segmentationId) {
    ensureRealMode();

    const id = validatePositiveId(
      segmentationId,
      'segmentation ID'
    );

    const response = await apiClient.get(
      `/inference/overlays/${id}`,
      {
        responseType: 'blob',
      }
    );

    return response.data;
  },
};
