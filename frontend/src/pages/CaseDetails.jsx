
import React, { useState } from 'react';
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
} from 'lucide-react';

import Button from '../components/Button';
import StatusBadge from '../components/StatusBadge';
import LoadingState from '../components/LoadingState';
import { useCases } from '../context/CaseContext';
import { formatDate, formatBytes } from '../utils/formatters';

const BACKEND_URL =
  import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

export default function CaseDetails() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { cases, isLoading } = useCases();

  const caseData = cases.find((c) => c.id === id);
  const segmentation = caseData?.segmentation;

  const imageId = segmentation?.imageId;

  // Identify which segmentation model was used.
  const savedModelType =
    segmentation?.modelType ||
    segmentation?.model_type ||
    segmentation?.selectedModel ||
    'syntax';

  const modelType = String(savedModelType).toLowerCase().includes('stenosis')
    ? 'stenosis'
    : 'syntax';

  const hasUploadedFile = Boolean(caseData?.file);
  const hasSegmentation = Boolean(segmentation?.maskUrl && imageId);

  const makeBackendUrl = (path) => {
    if (!path) return null;
    if (path.startsWith('http')) return path;
    return `${BACKEND_URL}${path}`;
  };

  const originalImageUrl = makeBackendUrl(
    segmentation?.originalImageUrl
  );

  const maskUrl = makeBackendUrl(segmentation?.maskUrl);

  // Post-processing state
  const [postprocessing, setPostprocessing] = useState(false);
  const [postprocessError, setPostprocessError] = useState('');
  const [postprocessData, setPostprocessData] = useState(null);

  // Centerline state
  const [centerlineLoading, setCenterlineLoading] = useState(false);
  const [centerlineError, setCenterlineError] = useState('');
  const [centerlineData, setCenterlineData] = useState(null);

  const cleanedMaskUrl = makeBackendUrl(
    postprocessData?.cleaned_mask_url
  );

  const centerlineImageUrl = makeBackendUrl(
    centerlineData?.centerline_url
  );

  const statistics = postprocessData?.statistics;
  const centerlineStats = centerlineData?.statistics;
  const diameterStats = centerlineStats?.diameter_statistics;

  // Run post-processing
  const runPostprocessing = async () => {
    if (!imageId) {
      setPostprocessError('Image ID is not available.');
      return;
    }

    setPostprocessing(true);
    setPostprocessError('');
    setPostprocessData(null);

    try {
      const response = await fetch(
        `${BACKEND_URL}/api/analysis/postprocess/${imageId}?model_type=${modelType}`,
        { method: 'POST' }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || data.message || 'Post-processing failed.'
        );
      }

      setPostprocessData(data);
    } catch (error) {
      setPostprocessError(error.message);
    } finally {
      setPostprocessing(false);
    }
  };

  // Extract centerline
  const runCenterline = async () => {
    if (!imageId) {
      setCenterlineError('Image ID is not available.');
      return;
    }

    setCenterlineLoading(true);
    setCenterlineError('');
    setCenterlineData(null);

    try {
      const response = await fetch(
        `${BACKEND_URL}/api/analysis/centerline/${imageId}?model_type=${modelType}`,
        { method: 'POST' }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || data.message || 'Centerline extraction failed.'
        );
      }

      setCenterlineData(data);
    } catch (error) {
      setCenterlineError(error.message);
    } finally {
      setCenterlineLoading(false);
    }
  };

  if (isLoading) {
    return <LoadingState message="Loading research case details..." />;
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

  // Reusable image display
  const ImagePanel = ({
    title,
    icon: Icon,
    src,
    alt,
    description,
  }) => (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-3">
      <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
        <Icon className="w-4 h-4 text-blue-600" />
        {title}
      </h3>

      {src ? (
        <div className="rounded-lg overflow-hidden border border-slate-200 bg-black p-2">
          <img
            src={src}
            alt={alt}
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
        <p className="text-xs text-slate-500">{description}</p>
      )}
    </div>
  );

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
              onClick={() => navigate(`/upload?caseId=${caseData.id}`)}
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
            <p className="text-xs text-slate-400">Target Vessel</p>
            <p className="text-sm font-medium mt-1">
              {caseData.targetRegion || 'Full Coronary Tree'}
            </p>
          </div>

          <div>
            <p className="text-xs text-slate-400">Imaging Modality</p>
            <p className="text-sm font-medium mt-1">
              {caseData.imageModality || 'X-ray Coronary Angiography'}
            </p>
          </div>

          <div>
            <p className="text-xs text-slate-400">File Status</p>
            <p className="text-sm font-medium mt-1">
              {hasUploadedFile ? 'Scan Attached' : 'Awaiting Image'}
            </p>
          </div>

          <div>
            <p className="text-xs text-slate-400">Research Notes</p>
            <p className="text-sm font-medium mt-1">
              {caseData.notes || 'No notes available'}
            </p>
          </div>
        </div>
      </div>

      {/* ORIGINAL IMAGE AND PREDICTED MASK */}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ImagePanel
          title="Original Coronary Angiogram"
          icon={FileImage}
          src={originalImageUrl}
          alt="Original angiography"
          description="Original uploaded angiography image."
        />

        <ImagePanel
          title="Predicted Segmentation Mask"
          icon={Bot}
          src={maskUrl}
          alt="Predicted segmentation mask"
          description="White indicates predicted foreground; black indicates background."
        />
      </div>

      {/* SEGMENTATION INFORMATION */}

      {hasSegmentation && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
          <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2 mb-4">
            <Activity className="w-5 h-5 text-blue-600" />
            Segmentation Information
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="bg-slate-50 border rounded-lg p-3">
              <p className="text-xs text-slate-500">Image ID</p>
              <p className="text-xs font-mono break-all mt-1">
                {imageId}
              </p>
            </div>

            <div className="bg-slate-50 border rounded-lg p-3">
              <p className="text-xs text-slate-500">Model Type</p>
              <p className="text-sm font-semibold mt-1 capitalize">
                {modelType}
              </p>
            </div>

            <div className="bg-slate-50 border rounded-lg p-3">
              <p className="text-xs text-slate-500">Foreground Pixels</p>
              <p className="text-sm font-semibold mt-1">
                {segmentation.foregroundPixels?.toLocaleString() ?? '—'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ALIGNED SEGMENTATION OVERLAY */}

      {hasSegmentation && originalImageUrl && maskUrl && (
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
                  WebkitMaskImage: `url("${maskUrl}")`,
                  maskImage: `url("${maskUrl}")`,
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
            Red highlights predicted foreground regions over the original angiography.
          </p>
        </div>
      )}

      {/* POST-PROCESSING */}

      {hasSegmentation && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                <WandSparkles className="w-5 h-5 text-purple-600" />
                Segmentation Post-processing
              </h3>

              <p className="text-xs text-slate-500 mt-1">
                Morphological closing and small-component filtering.
              </p>
            </div>

            <button
              type="button"
              onClick={runPostprocessing}
              disabled={postprocessing}
              className="flex items-center gap-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white text-sm font-semibold px-4 py-2.5"
            >
              {postprocessing ? (
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

          {postprocessError && (
            <div className="rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
              {postprocessError}
            </div>
          )}

          {postprocessData && (
            <div className="space-y-5">
              <p className="text-sm font-semibold text-emerald-700 flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4" />
                Post-processing completed successfully.
              </p>

              <ImagePanel
                title="Post-processed Segmentation Mask"
                icon={Layers}
                src={cleanedMaskUrl}
                alt="Cleaned segmentation mask"
                description="Mask after morphological closing and component filtering."
              />

              <h4 className="text-sm font-semibold text-slate-800">
                Post-processing Statistics
              </h4>

              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                <div className="rounded-lg bg-blue-50 border border-blue-100 p-3">
                  <p className="text-xs text-slate-500">Components</p>
                  <p className="text-xl font-bold text-blue-700 mt-1">
                    {statistics?.component_count ?? 0}
                  </p>
                </div>

                <div className="rounded-lg bg-emerald-50 border border-emerald-100 p-3">
                  <p className="text-xs text-slate-500">Cleaned Pixels</p>
                  <p className="text-xl font-bold text-emerald-700 mt-1">
                    {statistics?.cleaned_foreground_pixels?.toLocaleString() ?? 0}
                  </p>
                </div>

                <div className="rounded-lg bg-amber-50 border border-amber-100 p-3">
                  <p className="text-xs text-slate-500">Removed Pixels</p>
                  <p className="text-xl font-bold text-amber-700 mt-1">
                    {statistics?.removed_pixels?.toLocaleString() ?? 0}
                  </p>
                </div>

                <div className="rounded-lg bg-slate-100 border border-slate-200 p-3">
                  <p className="text-xs text-slate-500">Original Pixels</p>
                  <p className="text-xl font-bold text-slate-800 mt-1">
                    {statistics?.original_foreground_pixels?.toLocaleString() ?? 0}
                  </p>
                </div>
              </div>

              <h4 className="text-sm font-semibold text-slate-800">
                Detected Connected Components
              </h4>

              {statistics?.components?.length > 0 ? (
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
                      {statistics.components.map((component) => (
                        <tr
                          key={component.component_id}
                          className="border-t border-slate-100"
                        >
                          <td className="p-3">
                            {component.component_id}
                          </td>

                          <td className="p-3">
                            {component.area_pixels}
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
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-xs text-slate-500">
                  No connected components were returned.
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {/* CENTERLINE EXTRACTION */}

      {hasSegmentation && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                <GitBranch className="w-5 h-5 text-indigo-600" />
                Centerline Extraction
              </h3>

              <p className="text-xs text-slate-500 mt-1">
                Skeletonizes the segmentation mask and estimates width in pixels.
              </p>
            </div>

            <button
              type="button"
              onClick={runCenterline}
              disabled={centerlineLoading}
              className="flex items-center gap-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-300 text-white text-sm font-semibold px-4 py-2.5"
            >
              {centerlineLoading ? (
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

          {centerlineError && (
            <div className="rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
              {centerlineError}
            </div>
          )}

          {centerlineData && (
            <div className="space-y-5">
              <p className="text-sm font-semibold text-emerald-700 flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4" />
                Centerline extraction completed successfully.
              </p>

              <ImagePanel
                title="Extracted Centerline"
                icon={GitBranch}
                src={centerlineImageUrl}
                alt="Extracted centerline"
                description="Skeleton representation of the predicted segmentation mask."
              />

              <h4 className="text-sm font-semibold text-slate-800">
                Centerline Statistics
              </h4>

              <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                <div className="rounded-lg bg-indigo-50 border border-indigo-100 p-3">
                  <p className="text-xs text-slate-500">Centerline Pixels</p>
                  <p className="text-xl font-bold text-indigo-700 mt-1">
                    {centerlineStats?.centerline_pixel_count ?? '—'}
                  </p>
                </div>

                <div className="rounded-lg bg-slate-50 border p-3">
                  <p className="text-xs text-slate-500">Endpoints</p>
                  <p className="text-xl font-bold mt-1">
                    {centerlineStats?.endpoint_count ?? '—'}
                  </p>
                </div>

                <div className="rounded-lg bg-slate-50 border p-3">
                  <p className="text-xs text-slate-500">Junctions</p>
                  <p className="text-xl font-bold mt-1">
                    {centerlineStats?.junction_count ?? '—'}
                  </p>
                </div>

                <div className="rounded-lg bg-slate-50 border p-3">
                  <p className="text-xs text-slate-500">Mean Diameter</p>
                  <p className="text-xl font-bold mt-1">
                    {diameterStats?.mean_diameter_pixels?.toFixed(2) ?? '—'} px
                  </p>
                </div>

                <div className="rounded-lg bg-slate-50 border p-3">
                  <p className="text-xs text-slate-500">Minimum Diameter</p>
                  <p className="text-xl font-bold mt-1">
                    {diameterStats?.minimum_diameter_pixels?.toFixed(2) ?? '—'} px
                  </p>
                </div>

                <div className="rounded-lg bg-slate-50 border p-3">
                  <p className="text-xs text-slate-500">Maximum Diameter</p>
                  <p className="text-xl font-bold mt-1">
                    {diameterStats?.maximum_diameter_pixels?.toFixed(2) ?? '—'} px
                  </p>
                </div>
              </div>

              <p className="text-xs text-amber-700 bg-amber-50 border border-amber-100 rounded-lg p-3">
                Diameter values are approximate pixel-based estimates from the segmentation mask. They are not calibrated clinical measurements or a diagnosis.
              </p>
            </div>
          )}
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
              <strong>Size:</strong> {formatBytes(caseData.file.size)}
            </p>

            <p>
              <strong>Dimensions:</strong>{' '}
              {caseData.file.dimensions || '512 × 512 px'}
            </p>

            <p>
              <strong>Uploaded:</strong>{' '}
              {formatDate(caseData.file.uploadedAt)}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
