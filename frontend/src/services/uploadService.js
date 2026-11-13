
import axios from 'axios';
import { caseService } from './caseService';
import { CASE_STATUS } from '../utils/constants';

const BACKEND_URL =
  import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

const UPLOAD_ENDPOINTS = {
  syntax: '/api/analysis/segment',
  stenosis: '/api/analysis/segment/stenosis',
  combined: '/api/analysis/analyze',
};

const runGeometryAnalysis = async (imageId) => {
  try {
    // Step 1: Generate the cleaned Syntax vessel mask.
    const postprocessResponse = await axios.post(
      `${BACKEND_URL}/api/analysis/postprocess/${imageId}`,
      null,
      {
        params: { model_type: 'syntax' },
        timeout: 60000,
      }
    );

    if (postprocessResponse.data.status !== 'success') {
      throw new Error('Vessel mask post-processing failed.');
    }

    // Step 2: Calculate branch-wise narrowing and severity.
    const analysisResponse = await axios.post(
      `${BACKEND_URL}/api/analysis/stenosis-percentage/${imageId}`,
      null,
      {
        timeout: 120000,
      }
    );

    if (analysisResponse.data.status !== 'success') {
      throw new Error('Stenosis geometry analysis failed.');
    }

    const report = analysisResponse.data;

    return {
      success: true,
      imageId: report.image_id,
      geometryMaskModel: report.geometry_mask_model,
      measurementUnit: report.measurement_unit,
      measurementMethod: report.measurement_method,
      branchesExtracted: report.branches_extracted,
      branchesEvaluated: report.branches_evaluated,
      experimental: report.experimental,
      clinicallyValidated: report.clinically_validated,
      severityClassificationEnabled:
        report.severity_classification_enabled,
      severityThresholds: report.severity_thresholds,
      results: report.results || [],
      notice: report.notice,
      reportUrl: report.report_url
        ? `${BACKEND_URL}${report.report_url}`
        : null,
    };
  } catch (error) {
    console.error('Geometry analysis error:', error);

    return {
      success: false,
      imageId,
      results: [],
      error:
        error.response?.data?.detail ||
        error.message ||
        'Geometry analysis could not be completed.',
    };
  }
};

export const uploadService = {
  async uploadScan(
    caseId,
    file,
    onProgress = () => {},
    modelType = 'syntax'
  ) {
    if (!caseId) {
      throw new Error('Please select a research case.');
    }

    if (!file) {
      throw new Error('Please select an angiography image.');
    }

    const endpoint = UPLOAD_ENDPOINTS[modelType];

    if (!endpoint) {
      throw new Error('Invalid analysis mode selected.');
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      onProgress(10);

      const response = await axios.post(
        `${BACKEND_URL}${endpoint}`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          onUploadProgress: (event) => {
            if (event.total) {
              onProgress(
                Math.min(
                  Math.round((event.loaded / event.total) * 100),
                  90
                )
              );
            }
          },
          timeout: 180000,
        }
      );

      const result = response.data;

      if (result.status !== 'success') {
        throw new Error(
          result.message || 'Image analysis was unsuccessful.'
        );
      }

      onProgress(92);

      // Normalize the three API response formats.
      let segmentation;

      if (modelType === 'combined') {
        segmentation = {
          image_id: result.image_id,
          model: 'Syntax U-Net + Attention U-Net V4',
          image_dimensions: result.image_dimensions,

          foreground_pixels:
            result.syntax_segmentation.foreground_pixels,

          mask_url: result.syntax_segmentation.mask_url,

          stenosis_mask_url:
            result.stenosis_segmentation.mask_url,

          probability_map_url:
            result.stenosis_segmentation.probability_map_url,

          original_image_url: result.original_image_url,

          syntax_segmentation: result.syntax_segmentation,
          stenosis_segmentation: result.stenosis_segmentation,
          candidate_extraction: result.candidate_extraction,
          localization: result.localization,

          notice: result.notice,
        };
      } else {
        segmentation = result;
      }

      // Step 1: Run geometry analysis for Syntax and Combined modes.
      // Standalone Stenosis mode does not use the Syntax geometry mask.
      let geometryAnalysis = null;

      if (
        (modelType === 'syntax' || modelType === 'combined') &&
        segmentation.image_id
      ) {
        onProgress(94);

        geometryAnalysis = await runGeometryAnalysis(
          segmentation.image_id
        );
      }

      const fileData = {
        name: file.name,
        size: file.size,
        type: file.type || 'image/png',
        uploadedAt: new Date().toISOString(),
        dimensions: segmentation.image_dimensions
          ? `${segmentation.image_dimensions.width} x ${segmentation.image_dimensions.height} px`
          : 'Unknown',
      };

      const segmentationData = {
        imageId: segmentation.image_id,
        model: segmentation.model,
        modelType,
        foregroundPixels: segmentation.foreground_pixels,

        // Main segmentation mask.
        maskUrl: segmentation.mask_url,

        // Additional segmentation outputs.
        stenosisMaskUrl:
          segmentation.stenosis_mask_url || null,

        probabilityMapUrl:
          segmentation.probability_map_url || null,

        originalImageUrl:
          segmentation.original_image_url || null,

        syntaxSegmentation:
          segmentation.syntax_segmentation || null,

        stenosisSegmentation:
          segmentation.stenosis_segmentation || null,

        candidateExtraction:
          segmentation.candidate_extraction || null,

        localization:
          segmentation.localization || null,

        localizedImageUrl:
          segmentation.localization?.localized_image_url || null,

        // New geometry and severity analysis.
        geometryAnalysis,

        analysisNotice: segmentation.notice || null,
      };

      // Save upload and analysis data to the research case.
      await caseService.updateCase(caseId, {
        status: CASE_STATUS.UPLOADED,

        preprocessingStage:
          modelType === 'combined'
            ? 'COMBINED_ANALYSIS_COMPLETED'
            : 'SEGMENTATION_COMPLETED',

        file: fileData,
        segmentation: segmentationData,
      });

      onProgress(100);

      return {
        success: true,
        caseId,
        file: fileData,
        segmentation,
        geometryAnalysis,
        modelType,

        maskUrl: segmentation.mask_url
          ? `${BACKEND_URL}${segmentation.mask_url}`
          : null,

        stenosisMaskUrl: segmentation.stenosis_mask_url
          ? `${BACKEND_URL}${segmentation.stenosis_mask_url}`
          : null,

        probabilityMapUrl: segmentation.probability_map_url
          ? `${BACKEND_URL}${segmentation.probability_map_url}`
          : null,

        originalImageUrl: segmentation.original_image_url
          ? `${BACKEND_URL}${segmentation.original_image_url}`
          : null,

        localizedImageUrl:
          segmentation.localization?.localized_image_url
            ? `${BACKEND_URL}${segmentation.localization.localized_image_url}`
            : null,

        message:
          modelType === 'combined'
            ? 'Combined Syntax and Stenosis analysis completed successfully.'
            : modelType === 'syntax'
              ? 'Syntax segmentation and geometry analysis completed.'
              : 'Stenosis segmentation completed successfully.',
      };
    } catch (error) {
      const message =
        error.response?.data?.detail ||
        error.message ||
        'Image upload and analysis failed.';

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

    const makeBackendUrl = (path) => {
      if (!path) return null;
      if (path.startsWith('http')) return path;
      return `${BACKEND_URL}${path}`;
    };

    const geometryAnalysis = segmentation.geometryAnalysis
      ? {
          ...segmentation.geometryAnalysis,
          reportUrl: makeBackendUrl(
            segmentation.geometryAnalysis.reportUrl
          ),
        }
      : null;

    return {
      isAvailable: true,
      caseId,
      ...segmentation,

      maskUrl: makeBackendUrl(segmentation.maskUrl),

      stenosisMaskUrl: makeBackendUrl(
        segmentation.stenosisMaskUrl
      ),

      probabilityMapUrl: makeBackendUrl(
        segmentation.probabilityMapUrl
      ),

      originalImageUrl: makeBackendUrl(
        segmentation.originalImageUrl
      ),

      localizedImageUrl: makeBackendUrl(
        segmentation.localizedImageUrl
      ),

      geometryAnalysis,
    };
  },
};
