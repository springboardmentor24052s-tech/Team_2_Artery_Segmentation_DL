
import React, { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  Activity,
  ArrowLeft,
  BrainCircuit,
  CheckCircle2,
  CircleHelp,
  FileImage,
  Layers3,
  Microscope,
  ShieldAlert,
  Sparkles,
  Target,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Maximize2,
  RefreshCw,
} from 'lucide-react';

import { useCases } from '../context/CaseContext';
import LoadingState from '../components/LoadingState';
import StatusBadge from '../components/StatusBadge';
import { getScanAsset } from '../services/scanAssetService';
import { uploadService } from '../services/uploadService';
import { IS_MOCK_MODE } from '../services/api';

function StatusPill({ children, tone = 'slate' }) {
  const tones = {
    slate: 'bg-slate-100 text-slate-700 border-slate-200',
    amber: 'bg-amber-50 text-amber-800 border-amber-200',
    blue: 'bg-blue-50 text-blue-800 border-blue-200',
    green: 'bg-emerald-50 text-emerald-800 border-emerald-200',
    red: 'bg-rose-50 text-rose-800 border-rose-200',
  };

  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-semibold ${tones[tone] || tones.slate}`}>
      {children}
    </span>
  );
}

function ImageResultPanel({
  title,
  description,
  imageUrl,
  emptyMessage,
  zoomControls = false,
  fileName,
}) {
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState(0);

  useEffect(() => {
    setZoom(1);
    setRotation(0);
  }, [imageUrl]);

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <FileImage className="h-4 w-4 text-blue-700" />
          <h3 className="text-sm font-bold text-slate-900">{title}</h3>
        </div>

        <StatusPill tone={imageUrl ? 'green' : 'amber'}>
          {imageUrl ? 'Available' : 'Not available'}
        </StatusPill>
      </div>

      <p className="mt-1 text-xs text-slate-500">{description}</p>

      <div className="mt-4 flex min-h-64 max-h-[430px] items-center justify-center overflow-hidden rounded-lg border border-slate-200 bg-slate-950">
        {imageUrl ? (
          <div className="flex h-full w-full flex-col">
            {zoomControls && (
              <div className="flex items-center justify-end gap-1 border-b border-slate-700 bg-slate-900 px-2 py-1">
                <button
                  type="button"
                  title="Zoom out"
                  onClick={() => setZoom((value) => Math.max(0.5, Number((value - 0.1).toFixed(1))))}
                  className="rounded-md p-2 text-slate-200 hover:bg-slate-700"
                >
                  <ZoomOut className="h-4 w-4" />
                </button>

                <span className="min-w-12 text-center text-xs text-slate-300">
                  {Math.round(zoom * 100)}%
                </span>

                <button
                  type="button"
                  title="Zoom in"
                  onClick={() => setZoom((value) => Math.min(2.5, Number((value + 0.1).toFixed(1))))}
                  className="rounded-md p-2 text-slate-200 hover:bg-slate-700"
                >
                  <ZoomIn className="h-4 w-4" />
                </button>

                <button
                  type="button"
                  title="Rotate image"
                  onClick={() => setRotation((value) => (value + 90) % 360)}
                  className="rounded-md p-2 text-slate-200 hover:bg-slate-700"
                >
                  <RotateCcw className="h-4 w-4" />
                </button>

                <button
                  type="button"
                  title="Reset image view"
                  onClick={() => {
                    setZoom(1);
                    setRotation(0);
                  }}
                  className="rounded-md p-2 text-slate-200 hover:bg-slate-700"
                >
                  <Maximize2 className="h-4 w-4" />
                </button>
              </div>
            )}

            <div className="flex min-h-64 flex-1 items-center justify-center overflow-auto bg-black p-3">
              <img
                src={imageUrl}
                alt={title}
                className="max-h-[360px] max-w-full object-contain transition-transform duration-150"
                style={{
                  transform: `scale(${zoom}) rotate(${rotation}deg)`,
                }}
              />
            </div>
          </div>
        ) : (
          <div className="px-5 py-10 text-center">
            <CircleHelp className="mx-auto h-8 w-8 text-slate-400" />
            <p className="mt-3 text-sm font-semibold text-white">
              {emptyMessage}
            </p>
          </div>
        )}
      </div>

      {fileName && (
        <p className="mt-2 truncate text-[11px] text-slate-500" title={fileName}>
          {fileName}
        </p>
      )}
    </section>
  );
}

function MetricCard({ label, value }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-1 text-lg font-bold text-slate-900">{value}</p>
    </div>
  );
}

function OutputPanel({ icon: Icon, title, description, detail }) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-start gap-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-blue-700">
          <Icon className="h-5 w-5" />
        </span>

        <div>
          <h3 className="text-sm font-bold text-slate-900">{title}</h3>
          <p className="mt-0.5 text-xs text-slate-500">{description}</p>
        </div>
      </div>

      <div className="mt-4 rounded-lg border border-dashed border-slate-300 bg-slate-50 px-4 py-6 text-center">
        <p className="text-sm font-semibold text-slate-700">
          Not connected yet
        </p>
        <p className="mx-auto mt-1 max-w-md text-xs leading-5 text-slate-500">
          {detail}
        </p>
      </div>
    </section>
  );
}

export default function AnalysisWorkspace() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { cases, isLoading } = useCases();

  const caseData = cases.find((item) => String(item.id) === String(id));

  const [scanUrl, setScanUrl] = useState('');
  const [maskUrl, setMaskUrl] = useState('');
  const [overlayUrl, setOverlayUrl] = useState('');

  const [analysisResult, setAnalysisResult] = useState(null);
  const [isResultLoading, setIsResultLoading] = useState(false);
  const [resultError, setResultError] = useState('');
  const [scanLoadMessage, setScanLoadMessage] = useState('');

  const isPreviewableImage = useMemo(
    () => Boolean(caseData?.file?.type?.startsWith('image/')),
    [caseData?.file?.type]
  );

  // ---------------------------------------------------------
  // LOAD ORIGINAL IMAGE AND SAVED SEGMENTATION RESULTS
  // ---------------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    const objectUrls = [];

    const createObjectUrl = (blob) => {
      const url = URL.createObjectURL(blob);
      objectUrls.push(url);
      return url;
    };

    const loadWorkspace = async () => {
      setScanUrl('');
      setMaskUrl('');
      setOverlayUrl('');
      setAnalysisResult(null);
      setResultError('');
      setScanLoadMessage('');

      if (!id || !caseData) return;

      if (IS_MOCK_MODE) {
        if (!caseData.file) return;

        try {
          const asset = await getScanAsset(id);

          if (cancelled) return;

          if (asset?.blob && asset.type?.startsWith('image/')) {
            setScanUrl(createObjectUrl(asset.blob));
          } else if (asset?.blob) {
            setScanLoadMessage(
              'Browser preview is unavailable for this file format.'
            );
          } else {
            setScanLoadMessage(
              'The local image preview is unavailable. Please upload the image again.'
            );
          }
        } catch {
          if (!cancelled) {
            setScanLoadMessage(
              'The local image preview could not be loaded.'
            );
          }
        }

        return;
      }

      setIsResultLoading(true);

      try {
        // Step 1: Retrieve all saved results for this case.
        const results = await uploadService.getSegmentationResults(id);

        if (cancelled) return;

        if (!Array.isArray(results) || results.length === 0) {
          setScanLoadMessage(
            'No saved segmentation result is available for this case yet.'
          );
          return;
        }

        // Step 2: Select the most recent result.
        const latestResult = [...results].sort((a, b) => {
          return (
            new Date(b.created_at || 0).getTime() -
            new Date(a.created_at || 0).getTime()
          );
        })[0];

        setAnalysisResult(latestResult);

        // Step 3: Download original scan.
        try {
          const originalBlob = await uploadService.getOriginalImage(
            latestResult.image_id
          );

          if (cancelled) return;

          setScanUrl(createObjectUrl(originalBlob));
        } catch {
          if (!cancelled) {
            setScanLoadMessage(
              'The segmentation result exists, but the original scan could not be downloaded.'
            );
          }
        }

        // Step 4: Download mask and overlay independently.
        const [maskResult, overlayResult] = await Promise.allSettled([
          uploadService.getSegmentationMask(latestResult.segmentation_id),
          uploadService.getSegmentationOverlay(latestResult.segmentation_id),
        ]);

        if (cancelled) return;

        if (maskResult.status === 'fulfilled') {
          setMaskUrl(createObjectUrl(maskResult.value));
        }

        if (overlayResult.status === 'fulfilled') {
          setOverlayUrl(createObjectUrl(overlayResult.value));
        }
      } catch (error) {
        if (!cancelled) {
          setResultError(
            error.message || 'Unable to retrieve segmentation results.'
          );
        }
      } finally {
        if (!cancelled) {
          setIsResultLoading(false);
        }
      }
    };

    loadWorkspace();

    return () => {
      cancelled = true;
      objectUrls.forEach((url) => URL.revokeObjectURL(url));
    };
  }, [id, caseData?.id, caseData?.file?.name, caseData?.file?.uploadedAt]);

  if (isLoading) {
    return <LoadingState message="Loading analysis workspace..." />;
  }

  if (!caseData) {
    return (
      <div className="mx-auto max-w-xl rounded-xl border border-rose-200 bg-white p-8 text-center">
        <ShieldAlert className="mx-auto h-8 w-8 text-rose-600" />
        <h2 className="mt-3 text-lg font-bold text-slate-900">
          Case not found
        </h2>
        <p className="mt-1 text-sm text-slate-500">
          We could not find case {id} in the current case list.
        </p>
        <button
          onClick={() => navigate('/cases')}
          className="mt-4 rounded-lg bg-blue-700 px-4 py-2 text-sm font-semibold text-white"
        >
          Back to cases
        </button>
      </div>
    );
  }

  const hasResult = Boolean(analysisResult);
  const hasFile = Boolean(scanUrl || caseData.file);
  const modelName = analysisResult?.model_name || 'Not supplied';

  const metrics = [
    ['Dice score', analysisResult?.dice_score],
    ['IoU score', analysisResult?.iou_score],
    ['Precision', analysisResult?.precision_score],
    ['Recall', analysisResult?.recall_score],
    ['F1 score', analysisResult?.f1_score],
  ].filter(([, value]) => value !== null && value !== undefined);

  const formatMetric = (value) =>
    `${(Number(value) * 100).toFixed(1)}%`;

  return (
    <div className="space-y-6">
      {/* PAGE HEADER */}
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
        <div>
          <Link
            to={`/cases/${caseData.id}`}
            className="mb-3 inline-flex items-center gap-1.5 text-xs font-semibold text-slate-500 hover:text-blue-700"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Back to case details
          </Link>

          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-md border border-blue-200 bg-blue-50 px-2.5 py-1 font-mono text-xs font-bold text-blue-800">
              {caseData.id}
            </span>

            <StatusBadge status={caseData.status} />

            <StatusPill tone="amber">
              Research preview · not a diagnosis
            </StatusPill>
          </div>

          <h1 className="mt-2 text-2xl font-bold tracking-tight text-slate-900">
            AI Analysis Workspace
          </h1>

          <p className="mt-1 text-sm text-slate-500">
            {caseData.name} · {caseData.targetRegion || 'Full coronary tree'}
          </p>
        </div>

        {!hasFile && !hasResult && (
          <button
            onClick={() => navigate(`/upload?caseId=${caseData.id}`)}
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-700 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-800"
          >
            Upload angiogram
          </button>
        )}
      </div>

      {/* BACKEND STATUS */}
      <div className="rounded-xl border border-blue-200 bg-blue-50/70 p-4">
        <div className="flex items-start gap-3">
          <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-blue-700" />

          <div>
            <p className="text-sm font-bold text-blue-950">
              Model integration status
            </p>

            <p className="mt-1 text-xs leading-5 text-blue-900">
              {IS_MOCK_MODE
                ? 'Mock mode is enabled. This workspace displays locally stored image previews and does not run real segmentation inference.'
                : hasResult
                  ? 'Saved segmentation metadata has been retrieved from the backend. Available original, mask and overlay files are displayed below.'
                  : 'Connected to the backend. Checking for saved segmentation results for this case.'}
            </p>
          </div>
        </div>
      </div>

      {/* LOADING AND ERROR STATES */}
      {isResultLoading && (
        <div className="flex items-center gap-2 rounded-lg border border-blue-200 bg-blue-50 p-3 text-sm text-blue-800">
          <RefreshCw className="h-4 w-4 animate-spin" />
          Retrieving saved segmentation results and image files...
        </div>
      )}

      {resultError && (
        <div className="rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800">
          {resultError}
        </div>
      )}

      {/* ORIGINAL SCAN */}
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <FileImage className="h-4 w-4 text-blue-700" />
            <h2 className="text-sm font-bold text-slate-900">
              Input angiogram
            </h2>
          </div>

          <StatusPill tone={scanUrl ? 'green' : 'amber'}>
            {scanUrl ? 'Image loaded' : 'Image unavailable'}
          </StatusPill>
        </div>

        {scanUrl ? (
          <div className="mt-4">
            <ImageResultPanel
              title="Original angiogram"
              description="Original image retrieved from backend storage or local mock storage."
              imageUrl={scanUrl}
              zoomControls
              fileName={caseData.file?.name}
            />
          </div>
        ) : (
          <div className="mt-4 flex min-h-48 flex-col items-center justify-center rounded-lg bg-slate-950 px-5 py-8 text-center">
            <FileImage className="h-9 w-9 text-slate-400" />

            <p className="mt-3 text-sm font-semibold text-white">
              {caseData.file?.name || 'Angiogram not available'}
            </p>

            <p className="mt-1 max-w-md text-xs leading-5 text-slate-400">
              {scanLoadMessage ||
                (hasFile && isPreviewableImage
                  ? 'Loading image preview...'
                  : 'Upload a supported angiography image to continue.')}
            </p>

            {!hasFile && !isResultLoading && (
              <button
                onClick={() => navigate(`/upload?caseId=${caseData.id}`)}
                className="mt-4 rounded-lg bg-blue-700 px-4 py-2 text-xs font-semibold text-white hover:bg-blue-800"
              >
                Upload angiogram
              </button>
            )}
          </div>
        )}

        {caseData.file && (
          <div className="mt-3 grid grid-cols-2 gap-3 text-xs sm:grid-cols-3">
            <MetricCard
              label="File name"
              value={caseData.file.name}
            />
            <MetricCard
              label="File type"
              value={caseData.file.type || 'Not supplied'}
            />
            <MetricCard
              label="Dimensions"
              value={caseData.file.dimensions || 'Not supplied'}
            />
          </div>
        )}
      </section>

      {/* SEGMENTATION OUTPUTS */}
      <section>
        <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-lg font-bold text-slate-900">
              Segmentation results
            </h2>
            <p className="mt-1 text-xs text-slate-500">
              Backend-generated image outputs for the selected case
            </p>
          </div>

          <StatusPill tone={hasResult ? 'green' : 'amber'}>
            {hasResult ? 'Result retrieved' : 'Awaiting result'}
          </StatusPill>
        </div>

        {hasResult && (
          <div className="mb-4 rounded-xl border border-slate-200 bg-white p-4">
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <MetricCard
                label="Segmentation ID"
                value={analysisResult.segmentation_id}
              />
              <MetricCard
                label="Image ID"
                value={analysisResult.image_id}
              />
              <MetricCard
                label="Model"
                value={modelName}
              />
              <MetricCard
                label="Model version"
                value={analysisResult.model_version || 'Not supplied'}
              />
            </div>
          </div>
        )}

        <div className="grid gap-4 xl:grid-cols-2">
          <ImageResultPanel
            title="Predicted segmentation mask"
            description="Mask generated by the selected segmentation model."
            imageUrl={maskUrl}
            emptyMessage="Segmentation mask is not available."
          />

          <ImageResultPanel
            title="Segmentation overlay"
            description="Backend-generated overlay for visual inspection."
            imageUrl={overlayUrl}
            emptyMessage="Segmentation overlay is not available."
          />
        </div>

        {hasResult && metrics.length > 0 && (
          <div className="mt-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h3 className="text-sm font-bold text-slate-900">
              Available evaluation metrics
            </h3>

            <p className="mt-1 text-xs text-slate-500">
              Values supplied by the backend. Missing metrics are not estimated.
            </p>

            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
              {metrics.map(([label, value]) => (
                <MetricCard
                  key={label}
                  label={label}
                  value={formatMetric(value)}
                />
              ))}
            </div>
          </div>
        )}
      </section>

      {/* ANALYSIS PIPELINE */}
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-center gap-2">
          <Layers3 className="h-4 w-4 text-blue-700" />
          <h2 className="text-sm font-bold text-slate-900">
            Analysis pipeline
          </h2>
        </div>

        <p className="mt-1 text-xs text-slate-500">
          Current backend integration status
        </p>

        <ol className="mt-5 space-y-4">
          {[
            {
              number: '01',
              title: 'Image upload',
              subtitle: 'Original angiography image stored in backend',
              complete: hasResult,
            },
            {
              number: '02',
              title: 'Vessel segmentation',
              subtitle: 'U-Net / Attention U-Net mask generation',
              complete: hasResult && Boolean(maskUrl),
            },
            {
              number: '03',
              title: 'Stenosis localization',
              subtitle: 'Candidate region detection and localization',
              complete: false,
            },
            {
              number: '04',
              title: 'Explainability',
              subtitle: 'Model-supported visual explanation',
              complete: false,
            },
          ].map((step) => (
            <li key={step.number} className="flex gap-3">
              <span
                className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-bold ${
                  step.complete
                    ? 'bg-emerald-100 text-emerald-700'
                    : 'bg-slate-100 text-slate-500'
                }`}
              >
                {step.complete ? (
                  <CheckCircle2 className="h-4 w-4" />
                ) : (
                  step.number
                )}
              </span>

              <div className="min-w-0 flex-1 border-b border-slate-100 pb-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="text-sm font-semibold text-slate-800">
                    {step.title}
                  </p>

                  <StatusPill tone={step.complete ? 'green' : 'slate'}>
                    {step.complete ? 'Completed' : 'Not connected'}
                  </StatusPill>
                </div>

                <p className="mt-1 text-xs text-slate-500">
                  {step.subtitle}
                </p>
              </div>
            </li>
          ))}
        </ol>

        <button
          onClick={() => navigate('/preprocessing')}
          className="mt-4 inline-flex items-center gap-2 text-xs font-semibold text-blue-700 hover:text-blue-900"
        >
          View preprocessing monitor
          <Activity className="h-3.5 w-3.5" />
        </button>
      </section>

      {/* FUTURE MILESTONE 3 OUTPUTS */}
      <div className="grid gap-4 xl:grid-cols-2">
        <OutputPanel
          icon={Target}
          title="Stenosis detection & localization"
          description="Candidate region and vessel-level analysis"
          detail="Stenosis detection, localization and severity require their dedicated backend model outputs. Segmentation measurements are not presented as clinical narrowing percentages."
        />

        <OutputPanel
          icon={BrainCircuit}
          title="Explainable AI"
          description="Visual explanation of model-supported outputs"
          detail="A heatmap or attribution overlay will be displayed only after an explanation method compatible with the trained model produces it."
        />

        <OutputPanel
          icon={Microscope}
          title="Research interpretation"
          description="Model metadata and review notes"
          detail="This section will present validated model information and available research metrics without generating an automated clinical diagnosis."
        />

        <OutputPanel
          icon={Sparkles}
          title="Analysis summary"
          description="Consolidated model outputs"
          detail="Additional analysis will appear when the corresponding backend pipelines are connected and return verified results."
        />
      </div>

      <div className="flex items-start gap-2 rounded-lg border border-slate-200 bg-slate-50 p-3 text-[11px] leading-5 text-slate-600">
        <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-slate-500" />
        <p>
          <strong>Research-use notice:</strong> ArteryAI is a research
          prototype. Outputs require validation and qualified review and must
          not be used as a standalone clinical decision or diagnosis.
        </p>
      </div>
    </div>
  );
}
