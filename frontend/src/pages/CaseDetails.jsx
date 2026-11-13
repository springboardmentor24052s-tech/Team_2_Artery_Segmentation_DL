import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  FileImage,
  UploadCloud,
  Cpu,
  CheckCircle2,
  AlertTriangle,
  Calendar,
  Bot,
  Layers,
  WandSparkles,
  RefreshCw,
  Activity,
  GitBranch,
  MapPin,
  Target,
} from 'lucide-react';

import Button from '../components/Button';
import StatusBadge from '../components/StatusBadge';
import LoadingState from '../components/LoadingState';
import { useCases } from '../context/CaseContext';
import { formatDate, formatBytes } from '../utils/formatters';

const BACKEND_URL =
  import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

const MODEL_LABELS = {
  syntax: 'Syntax U-Net',
  stenosis: 'Attention U-Net V4',
};

const EMPTY_GEOMETRY = {
  syntax: {
    postprocessing: null,
    centerline: null,
    stenosisReport: null,
  },
  stenosis: {
    postprocessing: null,
    centerline: null,
    stenosisReport: null,
  },
};

const getModelTypes = (modelType) => {
  const value = String(modelType || 'syntax').toLowerCase();

  if (
    value === 'combined' ||
    (value.includes('syntax') && value.includes('stenosis'))
  ) {
    return ['syntax', 'stenosis'];
  }

  return [value.includes('stenosis') ? 'stenosis' : 'syntax'];
};

const makeBackendUrl = (path) => {
  if (!path) return null;

  if (/^https?:\/\//i.test(path)) {
    return path;
  }

  return `${BACKEND_URL}${path.startsWith('/') ? path : `/${path}`}`;
};

const formatNumber = (value, digits = 2) => {
  return typeof value === 'number'
    ? value.toFixed(digits)
    : '—';
};

function ImagePanel({
  title,
  icon: Icon = FileImage,
  src,
  alt,
  description,
}) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-3">
      <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
        <Icon className="w-4 h-4 text-blue-600" />
        {title}
      </h3>

      {src ? (
        <div className="rounded-lg overflow-hidden border border-slate-200 bg-black p-2">
          <img
            src={src}
            alt={alt || title}
            className="w-full max-h-[420px] object-contain mx-auto"
          />
        </div>
      ) : (
        <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50 p-10 text-center">
          <FileImage className="w-8 h-8 mx-auto text-slate-400 mb-2" />
          <p className="text-sm text-slate-500">
            Image not available
          </p>
        </div>
      )}

      {description && (
        <p className="text-xs text-slate-500">
          {description}
        </p>
      )}
    </div>
  );
}

function ErrorMessage({ message }) {
  if (!message) return null;

  return (
    <div className="rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
      {message}
    </div>
  );
}

function MetricCard({ label, value, suffix = '' }) {
  return (
    <div className="rounded-lg bg-slate-50 border p-3">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="text-xl font-bold mt-1">
        {value !== null && value !== undefined
          ? `${typeof value === 'number'
              ? value.toLocaleString()
              : value}${suffix}`
          : '—'}
      </p>
    </div>
  );
}

export default function CaseDetails() {
  const { id } = useParams();
  const navigate = useNavigate();

  const { cases, isLoading, updateCase } = useCases();

  const caseData = cases.find((item) => item.id === id);
  const segmentation = caseData?.segmentation;

  const imageId = segmentation?.imageId;

  const savedModelType =
    segmentation?.modelType ||
    segmentation?.model_type ||
    segmentation?.selectedModel ||
    'syntax';

  const modelTypes = getModelTypes(savedModelType);
  const isCombined = modelTypes.length === 2;

  const hasUploadedFile = Boolean(caseData?.file);

  const hasSegmentation = Boolean(
    imageId &&
    (
      segmentation?.maskUrl ||
      segmentation?.syntaxSegmentation?.mask_url ||
      segmentation?.stenosisSegmentation?.mask_url
    )
  );

  // --------------------------------------------------
  // IMAGE URLS
  // --------------------------------------------------

  const originalImageUrl = makeBackendUrl(
    segmentation?.originalImageUrl
  );

  const syntaxMaskUrl = makeBackendUrl(
    segmentation?.syntaxSegmentation?.mask_url ||
    (modelTypes.includes('syntax')
      ? segmentation?.maskUrl
      : null)
  );

  const stenosisMaskUrl = makeBackendUrl(
    segmentation?.stenosisSegmentation?.mask_url ||
    segmentation?.stenosisMaskUrl ||
    (
      modelTypes.length === 1 &&
      modelTypes[0] === 'stenosis'
        ? segmentation?.maskUrl
        : null
    )
  );

  const probabilityMapUrl = makeBackendUrl(
    segmentation?.stenosisSegmentation?.probability_map_url ||
    segmentation?.probabilityMapUrl
  );

  const localizedImageUrl = makeBackendUrl(
    segmentation?.localization?.localized_image_url ||
    segmentation?.localizedImageUrl
  );

  // --------------------------------------------------
  // STATE
  // --------------------------------------------------

  const [geometry, setGeometry] = useState(EMPTY_GEOMETRY);

  const [activeModel, setActiveModel] = useState(
    modelTypes[0] || 'syntax'
  );

  const [loadingAction, setLoadingAction] = useState({});
  const [actionErrors, setActionErrors] = useState({});

  const [gradCamLoading, setGradCamLoading] = useState(false);
  const [gradCamError, setGradCamError] = useState('');
  const [gradCamData, setGradCamData] = useState(null);
  const [gradCamFile, setGradCamFile] = useState(null);

  // --------------------------------------------------
  // RESTORE SAVED GEOMETRY RESULTS
  // --------------------------------------------------

  useEffect(() => {
    if (!caseData) return;

    const saved = caseData.geometryResults || {};

    setGeometry({
      syntax: {
        ...EMPTY_GEOMETRY.syntax,
        ...(saved.syntax || {}),
      },
      stenosis: {
        ...EMPTY_GEOMETRY.stenosis,
        ...(saved.stenosis || {}),
      },
    });
  }, [caseData?.id]);

  // Keep active model valid when case/model changes.
  useEffect(() => {
    if (!modelTypes.includes(activeModel)) {
      setActiveModel(modelTypes[0] || 'syntax');
    }
  }, [savedModelType]);

  // --------------------------------------------------
  // SAVE RESULTS TO CASE
  // --------------------------------------------------

  const saveGeometryResult = async (
    modelType,
    resultKey,
    result
  ) => {
    const nextGeometry = {
      ...geometry,
      [modelType]: {
        ...geometry[modelType],
        [resultKey]: result,
      },
    };

    setGeometry(nextGeometry);

    if (caseData && typeof updateCase === 'function') {
      try {
        await updateCase(caseData.id, {
          geometryResults: nextGeometry,
        });
      } catch (error) {
        console.error(
          'Unable to save geometry result:',
          error
        );

        setActionErrors((previous) => ({
          ...previous,
          save: 'The result was generated but could not be saved to this case.',
        }));
      }
    }
  };

  // --------------------------------------------------
  // API REQUEST HELPER
  // --------------------------------------------------

  const callAnalysisEndpoint = async (
    modelType,
    action,
    endpoint
  ) => {
    if (!imageId) {
      setActionErrors((previous) => ({
        ...previous,
        [`${modelType}-${action}`]:
          'Image ID is not available.',
      }));
      return;
    }

    const key = `${modelType}-${action}`;

    setLoadingAction((previous) => ({
      ...previous,
      [key]: true,
    }));

    setActionErrors((previous) => ({
      ...previous,
      [key]: '',
    }));

    try {
      const response = await fetch(
        `${BACKEND_URL}${endpoint}/${imageId}?model_type=${modelType}`,
        { method: 'POST' }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
          data.message ||
          `${action} failed.`
        );
      }

      const resultKey =
        action === 'postprocessing'
          ? 'postprocessing'
          : 'centerline';

      await saveGeometryResult(
        modelType,
        resultKey,
        data
      );
    } catch (error) {
      setActionErrors((previous) => ({
        ...previous,
        [key]:
          error.message ||
          `${action} failed.`,
      }));
    } finally {
      setLoadingAction((previous) => ({
        ...previous,
        [key]: false,
      }));
    }
  };

  // --------------------------------------------------
  // POST-PROCESSING
  // --------------------------------------------------

  const runPostprocessing = (modelType) => {
    return callAnalysisEndpoint(
      modelType,
      'postprocessing',
      '/api/analysis/postprocess'
    );
  };

  // --------------------------------------------------
  // CENTERLINE
  // --------------------------------------------------

  const runCenterline = (modelType) => {
    return callAnalysisEndpoint(
      modelType,
      'centerline',
      '/api/analysis/centerline'
    );
  };

  // --------------------------------------------------
  // STENOSIS PERCENTAGE ESTIMATION
  // --------------------------------------------------

  const runStenosisReport = async () => {
    if (!imageId) return;

    const key = 'syntax-stenosis-report';

    setLoadingAction((previous) => ({
      ...previous,
      [key]: true,
    }));

    setActionErrors((previous) => ({
      ...previous,
      [key]: '',
    }));

    try {
      const response = await fetch(
        `${BACKEND_URL}/api/analysis/stenosis-percentage/${imageId}?model_type=syntax`,
        { method: 'POST' }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
          data.message ||
          'Stenosis percentage estimation failed.'
        );
      }

      await saveGeometryResult(
        'syntax',
        'stenosisReport',
        data
      );
    } catch (error) {
      setActionErrors((previous) => ({
        ...previous,
        [key]:
          error.message ||
          'Stenosis percentage estimation failed.',
      }));
    } finally {
      setLoadingAction((previous) => ({
        ...previous,
        [key]: false,
      }));
    }
  };

  // --------------------------------------------------
  // GRAD-CAM
  // --------------------------------------------------

  const runGradCam = async () => {
    if (!gradCamFile) {
      setGradCamError(
        'Select the original PNG or JPG image to generate Grad-CAM.'
      );
      return;
    }

    setGradCamLoading(true);
    setGradCamError('');
    setGradCamData(null);

    try {
      const formData = new FormData();

      formData.append('file', gradCamFile);

      const response = await fetch(
        `${BACKEND_URL}/api/explainability/gradcam`,
        {
          method: 'POST',
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok || data.success === false) {
        throw new Error(
          data.detail ||
          data.message ||
          'Grad-CAM generation failed.'
        );
      }

      setGradCamData(data);
    } catch (error) {
      setGradCamError(
        error.message ||
        'Grad-CAM generation failed.'
      );
    } finally {
      setGradCamLoading(false);
    }
  };

  // --------------------------------------------------
  // LOADING AND NOT FOUND
  // --------------------------------------------------

  if (isLoading) {
    return (
      <LoadingState message="Loading research case details..." />
    );
  }

  if (!caseData) {
    return (
      <div className="max-w-2xl mx-auto py-12 text-center space-y-4">
        <AlertTriangle className="w-10 h-10 text-rose-600 mx-auto" />

        <h2 className="text-xl font-bold text-slate-800">
          Case Not Found
        </h2>

        <p className="text-sm text-slate-500">
          The requested research case does not exist.
        </p>

        <Button
          variant="outline"
          size="sm"
          icon={ArrowLeft}
          onClick={() => navigate('/cases')}
        >
          Return to Case History
        </Button>
      </div>
    );
  }

  // --------------------------------------------------
  // CURRENT GEOMETRY RESULTS
  // --------------------------------------------------

  const currentGeometry =
    geometry[activeModel] || EMPTY_GEOMETRY[activeModel];

  const postprocessData =
    currentGeometry?.postprocessing;

  const centerlineData =
    currentGeometry?.centerline;

  const stenosisReport =
    geometry.syntax?.stenosisReport;

  const postprocessStats =
    postprocessData?.statistics;

  const centerlineStats =
    centerlineData?.statistics;

  const diameterStats =
    centerlineStats?.diameter_statistics;

  const cleanedMaskUrl = makeBackendUrl(
    postprocessData?.cleaned_mask_url
  );

  const centerlineImageUrl = makeBackendUrl(
    centerlineData?.centerline_url
  );

  // --------------------------------------------------
  // UI
  // --------------------------------------------------

  return (
    <div className="space-y-6">

      {/* NAVIGATION */}

      <div className="flex flex-wrap items-center justify-between gap-3">
        <button
          type="button"
          onClick={() => navigate('/cases')}
          className="flex items-center gap-2 text-sm text-slate-500 hover:text-slate-900"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Case History
        </button>

        <div className="flex gap-2">
          {!hasUploadedFile && (
            <Button
              variant="primary"
              size="sm"
              icon={UploadCloud}
              onClick={() =>
                navigate(`/upload?caseId=${caseData.id}`)
              }
            >
              Upload Scan
            </Button>
          )}

          <Button
            variant="outline"
            size="sm"
            icon={Cpu}
            onClick={() => navigate('/preprocessing')}
          >
            Preprocessing Status
          </Button>
        </div>
      </div>

      {/* CASE INFORMATION */}

      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
        <div className="flex flex-wrap items-center gap-3 mb-3">
          <span className="text-xs font-bold text-blue-700 bg-blue-50 px-3 py-1 rounded-md">
            {caseData.id}
          </span>

          <StatusBadge status={caseData.status} />

          <span className="text-xs text-slate-500">
            <Calendar className="w-3 h-3 inline mr-1" />
            {formatDate(caseData.createdAt)}
          </span>
        </div>

        <h2 className="text-2xl font-bold text-slate-900">
          {caseData.name}
        </h2>

        <p className="text-sm text-slate-600 mt-2">
          {caseData.description || 'No description provided.'}
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-6 pt-5 border-t border-slate-100">
          <div>
            <p className="text-xs text-slate-400">
              Target Vessel
            </p>
            <p className="text-sm font-medium mt-1">
              {caseData.targetRegion || 'Full Coronary Tree'}
            </p>
          </div>

          <div>
            <p className="text-xs text-slate-400">
              Imaging Modality
            </p>
            <p className="text-sm font-medium mt-1">
              {caseData.imageModality ||
                'X-ray Coronary Angiography'}
            </p>
          </div>

          <div>
            <p className="text-xs text-slate-400">
              File Status
            </p>
            <p className="text-sm font-medium mt-1">
              {hasUploadedFile
                ? 'Scan Attached'
                : 'Awaiting Image'}
            </p>
          </div>

          <div>
            <p className="text-xs text-slate-400">
              Research Notes
            </p>
            <p className="text-sm font-medium mt-1">
              {caseData.notes || 'No notes available'}
            </p>
          </div>
        </div>
      </div>

      {/* ANALYSIS RESULTS */}

      {hasSegmentation ? (
        <>

          {/* ANALYSIS SUMMARY */}

          <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-blue-600" />
                  Analysis Results
                </h3>

                <p className="text-xs text-slate-500 mt-1">
                  Image ID:{' '}
                  <span className="font-mono">
                    {imageId}
                  </span>
                </p>
              </div>

              <span className="text-xs rounded-full bg-slate-100 px-3 py-1 text-slate-700">
                {isCombined
                  ? 'Combined analysis'
                  : MODEL_LABELS[modelTypes[0]]}
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              <div className="rounded-lg bg-slate-50 border p-3">
                <p className="text-xs text-slate-500">
                  Selected workflow
                </p>

                <p className="text-sm font-semibold mt-1">
                  {isCombined
                    ? 'Syntax + Stenosis'
                    : MODEL_LABELS[modelTypes[0]]}
                </p>
              </div>

              <div className="rounded-lg bg-slate-50 border p-3">
                <p className="text-xs text-slate-500">
                  Syntax foreground pixels
                </p>

                <p className="text-sm font-semibold mt-1">
                  {segmentation.syntaxSegmentation
                    ?.foreground_pixels?.toLocaleString?.() ??
                    (
                      modelTypes[0] === 'syntax'
                        ? segmentation.foregroundPixels
                            ?.toLocaleString?.()
                        : '—'
                    )}
                </p>
              </div>

              <div className="rounded-lg bg-slate-50 border p-3">
                <p className="text-xs text-slate-500">
                  Stenosis foreground pixels
                </p>

                <p className="text-sm font-semibold mt-1">
                  {segmentation.stenosisSegmentation
                    ?.foreground_pixels?.toLocaleString?.() ??
                    (
                      modelTypes[0] === 'stenosis'
                        ? segmentation.foregroundPixels
                            ?.toLocaleString?.()
                        : '—'
                    )}
                </p>
              </div>
            </div>
          </div>

          {/* ORIGINAL IMAGE AND MODEL OUTPUTS */}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ImagePanel
              title="Original Coronary Angiogram"
              icon={FileImage}
              src={originalImageUrl}
              alt="Original X-ray coronary angiogram"
              description="Original uploaded X-ray coronary angiography image."
            />

            {syntaxMaskUrl && (
              <ImagePanel
                title="Syntax U-Net Artery Segmentation"
                icon={Bot}
                src={syntaxMaskUrl}
                alt="Syntax U-Net artery segmentation mask"
                description="Predicted artery foreground mask from the Syntax U-Net pipeline."
              />
            )}

            {stenosisMaskUrl && (
              <ImagePanel
                title="Attention U-Net V4 Stenosis Mask"
                icon={Target}
                src={stenosisMaskUrl}
                alt="Stenosis segmentation mask"
                description="Predicted stenosis-related mask from the Attention U-Net V4 pipeline."
              />
            )}

            {probabilityMapUrl && (
              <ImagePanel
                title="Stenosis Probability Map"
                icon={Activity}
                src={probabilityMapUrl}
                alt="Stenosis probability map"
                description="Model probability map. It is not a calibrated clinical confidence score."
              />
            )}

            {localizedImageUrl && (
              <ImagePanel
                title="Stenosis Candidate Localization"
                icon={MapPin}
                src={localizedImageUrl}
                alt="Localized stenosis candidates"
                description="Backend-generated localization visualization for extracted candidates."
              />
            )}
          </div>

          {/* CANDIDATE EXTRACTION */}

          {segmentation.candidateExtraction && (
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-3">
              <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                <Target className="w-5 h-5 text-rose-600" />
                Stenosis Candidate Extraction
              </h3>

              <p className="text-xs text-slate-500">
                Candidate extraction output returned by the backend.
              </p>

              <pre className="max-h-80 overflow-auto rounded-lg bg-slate-950 text-slate-100 p-4 text-xs whitespace-pre-wrap">
                {JSON.stringify(
                  segmentation.candidateExtraction,
                  null,
                  2
                )}
              </pre>
            </div>
          )}

          {/* LOCALIZATION DETAILS */}

          {segmentation.localization && (
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-3">
              <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                <MapPin className="w-5 h-5 text-rose-600" />
                Localization Details
              </h3>

              <p className="text-sm text-slate-700">
                Candidates localized:{' '}
                {segmentation.localization.candidate_count ?? '—'}
              </p>

              {Array.isArray(
                segmentation.localization.results
              ) && (
                <div className="overflow-x-auto border rounded-lg">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-slate-50">
                      <tr>
                        <th className="p-3">Candidate</th>
                        <th className="p-3">
                          Localization output
                        </th>
                      </tr>
                    </thead>

                    <tbody>
                      {segmentation.localization.results.map(
                        (item, index) => (
                          <tr
                            key={item.candidate_id ?? index}
                            className="border-t"
                          >
                            <td className="p-3">
                              {item.candidate_id ?? index + 1}
                            </td>

                            <td className="p-3">
                              <pre className="whitespace-pre-wrap">
                                {JSON.stringify(item, null, 2)}
                              </pre>
                            </td>
                          </tr>
                        )
                      )}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {/* SEGMENTATION OVERLAY */}

          {originalImageUrl && syntaxMaskUrl && (
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-4">
              <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                <Layers className="w-5 h-5 text-rose-600" />
                Artery Segmentation Overlay
              </h3>

              <div className="bg-black rounded-lg border p-3">
                <div className="relative w-full max-w-[600px] aspect-square mx-auto overflow-hidden">
                  <img
                    src={originalImageUrl}
                    alt="Original angiography"
                    className="absolute inset-0 w-full h-full object-contain"
                  />

                  <div
                    className="absolute inset-0 pointer-events-none"
                    style={{
                      backgroundColor: 'rgba(255, 0, 0, 0.60)',
                      WebkitMaskImage: `url("${syntaxMaskUrl}")`,
                      maskImage: `url("${syntaxMaskUrl}")`,
                      WebkitMaskMode: 'luminance',
                      maskMode: 'luminance',
                      WebkitMaskSize: '100% 100%',
                      maskSize: '100% 100%',
                      WebkitMaskRepeat: 'no-repeat',
                      maskRepeat: 'no-repeat',
                      WebkitMaskPosition: 'center',
                      maskPosition: 'center',
                    }}
                  />
                </div>
              </div>

              <p className="text-xs text-slate-500">
                Red highlights predicted artery foreground regions over the original angiogram.
              </p>
            </div>
          )}

          {/* GEOMETRY MODEL SELECTION */}

          {isCombined && (
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
              <h3 className="text-sm font-semibold text-slate-900 mb-3">
                Geometry Analysis Model
              </h3>

              <div className="flex flex-wrap gap-2">
                {modelTypes.map((type) => (
                  <button
                    key={type}
                    type="button"
                    onClick={() => setActiveModel(type)}
                    className={`rounded-lg px-4 py-2 text-sm font-semibold border ${
                      activeModel === type
                        ? 'bg-blue-600 border-blue-600 text-white'
                        : 'bg-white border-slate-200 text-slate-700 hover:bg-slate-50'
                    }`}
                  >
                    {MODEL_LABELS[type]}
                  </button>
                ))}
              </div>

              <p className="text-xs text-slate-500 mt-2">
                Select a model to run its post-processing and centerline
                extraction. Experimental branch-wise narrowing estimation
                is calculated separately using the cleaned Syntax U-Net
                vessel geometry mask only.
              </p>
            </div>
          )}

          {/* POST-PROCESSING */}

          <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                  <WandSparkles className="w-5 h-5 text-purple-600" />
                  Segmentation Post-processing
                </h3>

                <p className="text-xs text-slate-500 mt-1">
                  Morphological closing and small-component filtering for{' '}
                  {MODEL_LABELS[activeModel]}.
                </p>
              </div>

              <button
                type="button"
                onClick={() => runPostprocessing(activeModel)}
                disabled={Boolean(
                  loadingAction[`${activeModel}-postprocessing`]
                )}
                className="flex items-center gap-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white text-sm font-semibold px-4 py-2.5"
              >
                {loadingAction[`${activeModel}-postprocessing`] ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <WandSparkles className="w-4 h-4" />
                    Run Post-processing
                  </>
                )}
              </button>
            </div>

            <ErrorMessage
              message={
                actionErrors[`${activeModel}-postprocessing`]
              }
            />

            {postprocessData && (
              <div className="space-y-4">
                <p className="text-sm font-semibold text-emerald-700 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4" />
                  Post-processing completed for{' '}
                  {MODEL_LABELS[activeModel]}.
                </p>

                <ImagePanel
                  title="Post-processed Segmentation Mask"
                  icon={Layers}
                  src={cleanedMaskUrl}
                  alt="Post-processed segmentation mask"
                  description="Mask after morphological closing and component filtering."
                />

                <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                  <MetricCard
                    label="Components"
                    value={postprocessStats?.component_count}
                  />

                  <MetricCard
                    label="Cleaned Pixels"
                    value={postprocessStats?.cleaned_foreground_pixels}
                  />

                  <MetricCard
                    label="Removed Pixels"
                    value={postprocessStats?.removed_pixels}
                  />

                  <MetricCard
                    label="Original Pixels"
                    value={postprocessStats?.original_foreground_pixels}
                  />
                </div>

                {postprocessStats?.components?.length > 0 && (
                  <div className="overflow-x-auto border border-slate-200 rounded-lg">
                    <table className="w-full text-xs text-left">
                      <thead className="bg-slate-50 text-slate-600">
                        <tr>
                          <th className="p-3">Component</th>
                          <th className="p-3">Area (pixels)</th>
                          <th className="p-3">Bounding Box</th>
                          <th className="p-3">Centroid</th>
                        </tr>
                      </thead>

                      <tbody>
                        {postprocessStats.components.map(
                          (component, index) => (
                            <tr
                              key={
                                component.component_id ?? index
                              }
                              className="border-t border-slate-100"
                            >
                              <td className="p-3">
                                {component.component_id ?? index + 1}
                              </td>

                              <td className="p-3">
                                {component.area_pixels ?? '—'}
                              </td>

                              <td className="p-3 font-mono">
                                {component.bounding_box
                                  ? `${component.bounding_box.x}, ${component.bounding_box.y}, ${component.bounding_box.width} × ${component.bounding_box.height}`
                                  : '—'}
                              </td>

                              <td className="p-3 font-mono">
                                {component.centroid
                                  ? `${component.centroid.x}, ${component.centroid.y}`
                                  : '—'}
                              </td>
                            </tr>
                          )
                        )}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* CENTERLINE EXTRACTION */}

          <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                  <GitBranch className="w-5 h-5 text-indigo-600" />
                  Centerline Extraction
                </h3>

                <p className="text-xs text-slate-500 mt-1">
                  Run post-processing first. Centerline and diameter outputs are pixel-based estimates.
                </p>
              </div>

              <button
                type="button"
                onClick={() => runCenterline(activeModel)}
                disabled={Boolean(
                  loadingAction[`${activeModel}-centerline`]
                )}
                className="flex items-center gap-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-300 text-white text-sm font-semibold px-4 py-2.5"
              >
                {loadingAction[`${activeModel}-centerline`] ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Extracting...
                  </>
                ) : (
                  <>
                    <GitBranch className="w-4 h-4" />
                    Extract Centerline
                  </>
                )}
              </button>
            </div>

            <ErrorMessage
              message={
                actionErrors[`${activeModel}-centerline`]
              }
            />

            {centerlineData && (
              <div className="space-y-4">
                <p className="text-sm font-semibold text-emerald-700 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4" />
                  Centerline extraction completed for{' '}
                  {MODEL_LABELS[activeModel]}.
                </p>

                <ImagePanel
                  title="Extracted Centerline"
                  icon={GitBranch}
                  src={centerlineImageUrl}
                  alt="Extracted vessel centerline"
                  description="Skeleton representation generated from the selected segmentation mask."
                />

                <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                  <MetricCard
                    label="Centerline Pixels"
                    value={centerlineStats?.centerline_pixel_count}
                  />

                  <MetricCard
                    label="Endpoints"
                    value={centerlineStats?.endpoint_count}
                  />

                  <MetricCard
                    label="Junctions"
                    value={centerlineStats?.junction_count}
                  />

                  <MetricCard
                    label="Mean Diameter"
                    value={
                      diameterStats?.mean_diameter_pixels !== undefined
                        ? formatNumber(
                            diameterStats.mean_diameter_pixels
                          )
                        : null
                    }
                    suffix=" px"
                  />

                  <MetricCard
                    label="Minimum Diameter"
                    value={
                      diameterStats?.minimum_diameter_pixels !== undefined
                        ? formatNumber(
                            diameterStats.minimum_diameter_pixels
                          )
                        : null
                    }
                    suffix=" px"
                  />

                  <MetricCard
                    label="Maximum Diameter"
                    value={
                      diameterStats?.maximum_diameter_pixels !== undefined
                        ? formatNumber(
                            diameterStats.maximum_diameter_pixels
                          )
                        : null
                    }
                    suffix=" px"
                  />
                </div>

                <p className="text-xs text-amber-700 bg-amber-50 border border-amber-100 rounded-lg p-3">
                  Diameter values are approximate pixel-based estimates from the segmentation mask. They are not calibrated clinical measurements or a diagnosis.
                </p>
              </div>
            )}
          </div>

          {/* STENOSIS PERCENTAGE ESTIMATION */}

          {modelTypes.includes('syntax') && (
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h3 className="text-sm font-semibold text-slate-900">
                    Experimental Stenosis Percentage Estimation
                  </h3>

                  <p className="text-xs text-slate-500 mt-1">
                    Uses the cleaned Syntax U-Net vessel mask, extracted
                    centerline, and branch-wise pixel diameter profiles.
                    This is an experimental geometry-based estimate,
                    not a clinically validated measurement.
                  </p>
                </div>

                <button
                  type="button"
                  onClick={runStenosisReport}
                  disabled={
                    !geometry.syntax.postprocessing ||
                    Boolean(
                      loadingAction['syntax-stenosis-report']
                    )
                  }
                  className="rounded-lg bg-rose-600 hover:bg-rose-700 disabled:bg-slate-300 text-white text-sm font-semibold px-4 py-2.5"
                >
                  {loadingAction['syntax-stenosis-report']
                    ? 'Calculating...'
                    : 'Estimate Narrowing'}
                </button>
              </div>

              {!geometry.syntax.postprocessing && (
                <p className="text-xs text-slate-500">
                  Run Syntax U-Net post-processing before estimating narrowing.
                </p>
              )}

              <ErrorMessage
                message={actionErrors['syntax-stenosis-report']}
              />

              {stenosisReport && (
                <div className="space-y-3">
                  <p className="text-xs text-slate-600">
                    Geometry source:{' '}
                    <strong>
                      {stenosisReport.geometry_mask_model || 'Syntax U-Net'}
                    </strong>
                  </p>

                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                    <MetricCard
                      label="Branches Extracted"
                      value={stenosisReport.branches_extracted}
                    />

                    <MetricCard
                      label="Branches Evaluated"
                      value={stenosisReport.branches_evaluated}
                    />

                    <MetricCard
                      label="Measurement Unit"
                      value={
                        stenosisReport.measurement_unit || 'pixels'
                      }
                    />
                  </div>

                  {Array.isArray(stenosisReport.results) && (
                    <div className="space-y-3">
                      {stenosisReport.results.map(
                        (branch, index) => (
                          <div
                            key={branch.branch_id ?? index}
                            className="border rounded-lg p-3"
                          >
                            <p className="text-sm font-semibold">
                              Branch {branch.branch_id ?? index + 1}
                            </p>

                            <pre className="mt-2 text-xs whitespace-pre-wrap overflow-auto">
                              {JSON.stringify(branch, null, 2)}
                            </pre>
                          </div>
                        )
                      )}
                    </div>
                  )}

                  <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-lg p-3">
                    {stenosisReport.notice ||
                      'Experimental pixel-based estimate only. This output is not clinically validated and must not be used as a diagnosis.'}
                  </p>
                </div>
              )}
            </div>
          )}

          {/* GRAD-CAM EXPLAINABILITY */}

          <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-3">
            <h3 className="text-sm font-semibold text-slate-900">
              Explainable AI (Grad-CAM)
            </h3>

            <p className="text-xs text-slate-500">
              Select the same angiography image used for this case. The Grad-CAM API accepts an uploaded image file.
            </p>

            <div className="flex flex-wrap items-center gap-3">
              <input
                type="file"
                accept=".png,.jpg,.jpeg,image/png,image/jpeg"
                onChange={(event) => {
                  setGradCamFile(
                    event.target.files?.[0] || null
                  );
                  setGradCamError('');
                  setGradCamData(null);
                }}
                className="block max-w-full text-sm text-slate-600 file:mr-3 file:rounded-lg file:border-0 file:bg-slate-100 file:px-3 file:py-2 file:text-sm file:font-semibold file:text-slate-700"
              />

              <button
                type="button"
                onClick={runGradCam}
                disabled={gradCamLoading || !gradCamFile}
                className="rounded-lg bg-violet-600 px-4 py-2 text-sm font-semibold text-white hover:bg-violet-700 disabled:bg-slate-300"
              >
                {gradCamLoading
                  ? 'Generating...'
                  : 'Generate Grad-CAM'}
              </button>
            </div>

            <ErrorMessage message={gradCamError} />

            {gradCamData && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <ImagePanel
                  title="Grad-CAM Heatmap"
                  src={makeBackendUrl(gradCamData.heatmap_url)}
                  alt="Grad-CAM heatmap"
                />

                <ImagePanel
                  title="Grad-CAM Overlay"
                  src={makeBackendUrl(gradCamData.overlay_url)}
                  alt="Grad-CAM overlay"
                />
              </div>
            )}

            {gradCamData && (
              <div className="rounded-lg bg-slate-50 border p-3 text-xs text-slate-600 space-y-1">
                <p>
                  <strong>Method:</strong>{' '}
                  {gradCamData.method || '—'}
                </p>

                <p>
                  <strong>Target Layer:</strong>{' '}
                  {gradCamData.target_layer || '—'}
                </p>

                <p>
                  <strong>Experimental:</strong>{' '}
                  {gradCamData.experimental ? 'Yes' : 'Not specified'}
                </p>
              </div>
            )}
          </div>
        </>
      ) : (
        <div className="rounded-xl border border-dashed border-slate-300 bg-slate-50 p-10 text-center">
          <FileImage className="w-10 h-10 mx-auto text-slate-400 mb-3" />

          <h3 className="text-base font-semibold text-slate-800">
            No segmentation results yet
          </h3>

          <p className="text-sm text-slate-500 mt-1">
            Upload an X-ray coronary angiography image to start the analysis workflow.
          </p>

          <Button
            variant="primary"
            size="sm"
            icon={UploadCloud}
            onClick={() =>
              navigate(`/upload?caseId=${caseData.id}`)
            }
          >
            Upload Scan
          </Button>
        </div>
      )}

      {/* UPLOADED FILE INFORMATION */}

      {hasUploadedFile && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
          <h3 className="text-sm font-semibold text-slate-900 mb-3">
            Uploaded File Information
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-slate-600">
            <p>
              <strong>File:</strong> {caseData.file.name}
            </p>

            <p>
              <strong>Size:</strong>{' '}
              {formatBytes(caseData.file.size)}
            </p>

            <p>
              <strong>Dimensions:</strong>{' '}
              {caseData.file.dimensions || 'Not recorded'}
            </p>

            <p>
              <strong>Uploaded:</strong>{' '}
              {formatDate(caseData.file.uploadedAt)}
            </p>
          </div>
        </div>
      )}

      {/* RESEARCH DISCLAIMER */}

      <p className="text-xs text-slate-500">
        Research-use notice: outputs are AI-assisted experimental results for X-ray coronary angiography. They are not a standalone clinical diagnosis or clinically validated measurements.
      </p>
    </div>
  );
}