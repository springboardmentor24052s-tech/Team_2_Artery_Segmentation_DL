
import axios from 'axios';
import { caseService } from './caseService';
import { CASE_STATUS } from '../utils/constants';

const BACKEND_URL =
  import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

export const uploadService = {
  async uploadScan(caseId, file, onProgress = () => {}, modelType = 'syntax') {
    if (!caseId) {
      throw new Error('Please select a research case.');
    }

    if (!file) {
      throw new Error('Please select an angiography image.');
    }

    const endpoints = {
      syntax: '/api/analysis/segment',
      stenosis: '/api/analysis/segment/stenosis',
    };

    const endpoint = endpoints[modelType];

    if (!endpoint) {
      throw new Error('Invalid segmentation model selected.');
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      onProgress(10);

      const response = await axios.post(
        `${BACKEND_URL}${endpoint}`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: (event) => {
            if (event.total) {
              onProgress(
                Math.min(Math.round((event.loaded / event.total) * 100), 90)
              );
            }
          },
          timeout: 120000,
        }
      );

      const segmentation = response.data;
      onProgress(95);

      const fileData = {
        name: file.name,
        size: file.size,
        type: file.type || 'image/png',
        uploadedAt: new Date().toISOString(),
        dimensions: `${segmentation.image_dimensions.width} x ${segmentation.image_dimensions.height} px`,
      };

      const segmentationData = {
        imageId: segmentation.image_id,
        model: segmentation.model,
        modelType,
        foregroundPixels: segmentation.foreground_pixels,
        maskUrl: segmentation.mask_url,
        probabilityMapUrl: segmentation.probability_map_url || null,
        originalImageUrl: segmentation.original_image_url,
      };

      await caseService.updateCase(caseId, {
        status: CASE_STATUS.UPLOADED,
        preprocessingStage: 'SEGMENTATION_COMPLETED',
        file: fileData,
        segmentation: segmentationData,
      });

      onProgress(100);

      return {
        success: true,
        caseId,
        file: fileData,
        segmentation,
        modelType,
        maskUrl: `${BACKEND_URL}${segmentation.mask_url}`,
        probabilityMapUrl: segmentation.probability_map_url
          ? `${BACKEND_URL}${segmentation.probability_map_url}`
          : null,
        originalImageUrl: `${BACKEND_URL}${segmentation.original_image_url}`,
        message: 'Image uploaded and segmentation completed successfully.',
      };
    } catch (error) {
      const message =
        error.response?.data?.detail ||
        error.message ||
        'Image upload and segmentation failed.';

      throw new Error(message);
    }
  },

  async getPreprocessingStatus(caseId) {
    const targetCase = await caseService.getCaseById(caseId);

    return {
      caseId: targetCase.id,
      stage: targetCase.preprocessingStage || 'NOT_UPLOADED',
      lastUpdated: new Date().toISOString(),
      filePresent: Boolean(targetCase.file),
    };
  },

  async getSegmentationResults(caseId) {
    const targetCase = await caseService.getCaseById(caseId);

    if (!targetCase.segmentation) {
      return {
        isAvailable: false,
        message: 'No segmentation result is available for this case.',
        caseId,
      };
    }

    const segmentation = targetCase.segmentation;

    return {
      isAvailable: true,
      caseId,
      ...segmentation,
      maskUrl: segmentation.maskUrl
        ? `${BACKEND_URL}${segmentation.maskUrl}`
        : null,
      probabilityMapUrl: segmentation.probabilityMapUrl
        ? `${BACKEND_URL}${segmentation.probabilityMapUrl}`
        : null,
      originalImageUrl: segmentation.originalImageUrl
        ? `${BACKEND_URL}${segmentation.originalImageUrl}`
        : null,
    };
  },
};
