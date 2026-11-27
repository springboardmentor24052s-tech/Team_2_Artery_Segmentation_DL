
import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  FileImage,
  UploadCloud,
  Cpu,
  Clock,
  CheckCircle2,
  AlertTriangle,
  Calendar,
  Sparkles,
  Bot,
  Activity,
} from 'lucide-react';
import Button from '../components/Button';
import StatusBadge from '../components/StatusBadge';
import LoadingState from '../components/LoadingState';
import { useCases } from '../context/CaseContext';
import { formatDate, formatBytes } from '../utils/formatters';
import { uploadService } from '../services/uploadService';
import { IS_MOCK_MODE } from '../services/api';

export default function CaseDetails() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { cases, isLoading } = useCases();

  const caseData = cases.find((c) => String(c.id) === String(id));

  const [hasPersistedResults, setHasPersistedResults] = useState(false);
  const [persistedImages, setPersistedImages] = useState([]);
  const [isCheckingResults, setIsCheckingResults] = useState(false);
  const [resultCheckError, setResultCheckError] = useState('');

  useEffect(() => {
    let active = true;

    if (IS_MOCK_MODE || !id) {
      setHasPersistedResults(false);
      setPersistedImages([]);
      setIsCheckingResults(false);
      setResultCheckError('');
      return () => { active = false; };
    }

    const loadSavedRecords = async () => {
      setIsCheckingResults(true);
      setResultCheckError('');

      const [resultsResponse, imagesResponse] = await Promise.allSettled([
        uploadService.getSegmentationResults(id),
        uploadService.getCaseImages(id),
      ]);

      if (!active) return;

      const results = resultsResponse.status === 'fulfilled'
        ? resultsResponse.value : [];
      const images = imagesResponse.status === 'fulfilled'
        ? imagesResponse.value : [];

      setHasPersistedResults(results.length > 0);
      setPersistedImages(images);

      if (resultsResponse.status === 'rejected' && imagesResponse.status === 'rejected') {
        setResultCheckError(
          resultsResponse.reason?.message || imagesResponse.reason?.message ||
          'Saved image and segmentation records could not be checked.'
        );
      } else if (resultsResponse.status === 'rejected') {
        setResultCheckError(resultsResponse.reason?.message || 'Segmentation results could not be checked.');
      }

      setIsCheckingResults(false);
    };

    loadSavedRecords();
    return () => { active = false; };
  }, [id]);

  if (isLoading) {
    return <LoadingState message="Loading research case details..." />;
  }

  if (!caseData) {
    return (
      <div className="max-w-2xl mx-auto py-12 text-center space-y-4">
        <div className="w-12 h-12 rounded-full bg-rose-50 text-rose-600 flex items-center justify-center mx-auto border border-rose-200">
          <AlertTriangle className="w-6 h-6" />
        </div>

        <h2 className="text-xl font-bold text-slate-800">
          Case Not Found
        </h2>

        <p className="text-sm text-slate-500">
          The requested research case with identifier{' '}
          <strong className="font-mono text-slate-700">{id}</strong>{' '}
          does not exist in the current session.
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

  // Prefer current-session file metadata; otherwise use persisted image records.
  const latestPersistedImage = [...persistedImages].sort((a, b) =>
    new Date(b.uploadedAt || 0).getTime() - new Date(a.uploadedAt || 0).getTime()
  )[0];
  const fileMetadata = caseData.file || latestPersistedImage || null;
  const hasUploadedFile = Boolean(fileMetadata);

  const hasPersistedResultsWithoutFile =
    hasPersistedResults && !hasUploadedFile;

  const hasAnalysisAccess =
    hasPersistedResults || (IS_MOCK_MODE && hasUploadedFile);

  return (
    <div className="space-y-6">
      {/* Top breadcrumb & navigation */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <button
          type="button"
          onClick={() => navigate('/cases')}
          className="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-500 hover:text-slate-800 transition-colors w-fit"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Case History</span>
        </button>

        <div className="flex items-center gap-2">
          {!hasUploadedFile && (
            <Button
              variant="primary"
              size="sm"
              icon={UploadCloud}
              onClick={() => navigate(`/upload?caseId=${caseData.id}`)}
            >
              Upload Scan File
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

          {hasAnalysisAccess && (
            <Button
              variant="primary"
              size="sm"
              icon={Activity}
              onClick={() => navigate(`/cases/${caseData.id}/analysis`)}
            >
              AI Analysis Workspace
            </Button>
          )}
        </div>
      </div>

      {/* Case Header Card */}
      <div className="bg-white rounded-xl border border-slate-200/90 shadow-2xs p-5 sm:p-6">
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2.5">
              <span className="font-mono text-xs font-bold text-blue-700 bg-blue-50 border border-blue-200 px-2.5 py-0.5 rounded-md">
                {caseData.id}
              </span>

              <StatusBadge status={caseData.status} />

              <span className="text-xs text-slate-400">&bull;</span>

              <span className="text-xs text-slate-500 flex items-center gap-1">
                <Calendar className="w-3.5 h-3.5 text-slate-400" />
                Created {formatDate(caseData.createdAt)}
              </span>
            </div>

            <h2 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
              {caseData.name}
            </h2>

            <p className="text-sm text-slate-600 max-w-3xl leading-relaxed">
              {caseData.description || 'No description provided for this study.'}
            </p>
          </div>
        </div>

        {/* Metadata Details Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-6 pt-5 border-t border-slate-100">
          <div>
            <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
              Target Vessel
            </p>
            <p className="text-sm font-medium text-slate-800 mt-1">
              {caseData.targetRegion || 'Full Coronary Tree'}
            </p>
          </div>

          <div>
            <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
              Imaging Modality
            </p>
            <p
              className="text-sm font-medium text-slate-800 mt-1 truncate"
              title={caseData.imageModality}
            >
              {caseData.imageModality || 'X-ray Coronary Angiography (2D XCA)'}
            </p>
          </div>

          <div>
            <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
              File Status
            </p>
            <p className="text-sm font-medium text-slate-800 mt-1">
              {hasUploadedFile || hasPersistedResults ? (
                <span className="text-emerald-700 flex items-center gap-1 font-semibold text-xs">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  {hasUploadedFile ? 'Scan Attached' : 'Saved Result Available'}
                </span>
              ) : (
                <span className="text-amber-700 flex items-center gap-1 font-semibold text-xs">
                  <Clock className="w-3.5 h-3.5" />
                  Awaiting Image
                </span>
              )}
            </p>
          </div>

          <div>
            <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
              Internal Research Notes
            </p>
            <p
              className="text-sm text-slate-700 mt-1 truncate"
              title={caseData.notes}
            >
              {caseData.notes || 'Synthetic benchmark frame'}
            </p>
          </div>
        </div>
      </div>

      {/* Image / File Details and AI Segmentation */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Left Column: Image Scan File Information */}
        <div className="bg-white rounded-xl border border-slate-200 shadow-2xs p-5 space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100">
            <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
              <FileImage className="w-4 h-4 text-blue-600" />
              <span>Coronary Angiogram Scan File</span>
            </h3>

            {hasUploadedFile && (
              <span className="text-xs text-slate-500 font-mono">
                {formatBytes(fileMetadata.size)}
              </span>
            )}
          </div>

          {hasUploadedFile ? (
            <div className="space-y-4">
              {/* Scan Preview Canvas / Visualizer */}
              <div className="relative rounded-lg overflow-hidden border border-slate-200 bg-slate-950 flex flex-col items-center justify-center p-6 text-center min-h-[220px]">
                <div className="w-32 h-32 rounded-full border border-slate-800/80 bg-radial from-slate-800 to-slate-950 flex items-center justify-center relative shadow-inner">
                  <div className="w-24 h-24 rounded-full border border-dashed border-blue-500/30 flex items-center justify-center">
                    <FileImage className="w-10 h-10 text-slate-400 animate-pulse" />
                  </div>

                  <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                    <span className="text-[10px] font-mono text-blue-400/80 bg-slate-900/80 px-2 py-0.5 rounded">
                      {caseData.imageType}
                    </span>
                  </div>
                </div>

                <div className="mt-4 text-xs text-slate-300">
                  <p className="font-semibold text-white">
                    {fileMetadata.name}
                  </p>

                  <p className="text-slate-400 text-[11px] mt-0.5">
                    Frame Resolution: {fileMetadata.dimensions || '512 x 512 px'}
                    {' '}&bull; Uploaded {formatDate(fileMetadata.uploadedAt)}
                  </p>
                </div>
              </div>

              {/* DICOM / File Headers */}
              {fileMetadata.dicomMeta ? (
                <div className="bg-slate-50 rounded-lg p-3 border border-slate-200 text-xs space-y-1.5">
                  <p className="font-semibold text-slate-700 text-[11px] uppercase tracking-wider">
                    Angiography Acquisition Metadata
                  </p>

                  <div className="grid grid-cols-2 gap-2 text-slate-600 text-[11px] font-mono">
                    <div>Photometric: {fileMetadata.dicomMeta.photometric}</div>
                    <div>Tube Voltage: {fileMetadata.dicomMeta.kvp}</div>
                    <div>Frame Rate: {fileMetadata.dicomMeta.frameRate}</div>
                    <div>Field of View: {fileMetadata.dicomMeta.fieldOfView}</div>
                  </div>
                </div>
              ) : (
                <div className="bg-slate-50 rounded-lg p-3 border border-slate-200 text-xs text-slate-500">
                  Standard format verification complete.
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              {hasPersistedResultsWithoutFile && (
                <div className="rounded-lg border border-blue-200 bg-blue-50 p-4">
                  <div className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 text-blue-600 mt-0.5" />

                    <div>
                      <p className="text-sm font-semibold text-blue-900">
                        Saved Segmentation Result Available
                      </p>

                      <p className="text-xs text-blue-800 mt-1">
                        A segmentation result exists in backend storage.
                        Open the Analysis Workspace to retrieve the original
                        angiography image, segmentation mask and overlay.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50/60 p-8 text-center space-y-3">
                <div className="w-10 h-10 rounded-full bg-slate-200 text-slate-500 mx-auto flex items-center justify-center">
                  <FileImage className="w-5 h-5" />
                </div>

                <h4 className="text-sm font-semibold text-slate-800">
                  {hasPersistedResultsWithoutFile
                    ? 'Original Scan Stored in Backend'
                    : 'No Scan Uploaded Yet'}
                </h4>

                <p className="text-xs text-slate-500 max-w-sm mx-auto leading-relaxed">
                  {hasPersistedResultsWithoutFile
                    ? 'The original scan is retrieved through its saved backend image reference in the Analysis Workspace.'
                    : 'Associate a supported angiography image (PNG, JPG, JPEG, TIFF or BMP) with this case to run segmentation.'}
                </p>

                {!hasPersistedResultsWithoutFile && (
                  <Button
                    variant="primary"
                    size="sm"
                    icon={UploadCloud}
                    onClick={() => navigate(`/upload?caseId=${caseData.id}`)}
                  >
                    Upload Scan for {caseData.id}
                  </Button>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Segmentation Results */}
        <div className="bg-white rounded-xl border border-slate-200 shadow-2xs p-5 space-y-4">
          <div className="flex items-center justify-between gap-3 pb-3 border-b border-slate-100">
            <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
              <Bot className="w-4 h-4 text-blue-600" />
              <span>AI Artery Segmentation &amp; Stenosis Analysis</span>
            </h3>

            <span
              className={`text-[11px] font-medium px-2 py-0.5 rounded-full border ${
                isCheckingResults
                  ? 'bg-sky-50 text-sky-800 border-sky-200'
                  : hasPersistedResults || (IS_MOCK_MODE && hasUploadedFile)
                    ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                    : 'bg-amber-50 text-amber-800 border-amber-200'
              }`}
            >
              {isCheckingResults
                ? 'Checking'
                : hasPersistedResults
                  ? 'Result available'
                  : IS_MOCK_MODE && hasUploadedFile
                    ? 'Preview available'
                    : 'Awaiting scan'}
            </span>
          </div>

          {hasAnalysisAccess ? (
            <div className="rounded-xl border border-emerald-200 bg-emerald-50/60 p-6 text-center space-y-3">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full border border-emerald-200 bg-emerald-100 text-emerald-700">
                <CheckCircle2 className="h-6 w-6" />
              </div>

              <div>
                <h4 className="text-sm font-bold text-slate-900">
                  {hasPersistedResults
                    ? 'Segmentation result is available'
                    : 'Scan preview is available'}
                </h4>

                <p className="mx-auto mt-1 max-w-md text-xs leading-relaxed text-slate-600">
                  {hasPersistedResults
                    ? 'Open the analysis workspace to review the original angiography image, saved segmentation mask, overlay and available model measurements.'
                    : 'This is a local demo preview. Mock mode does not run U-Net inference or generate a segmentation mask.'}
                </p>
              </div>

              <Button
                variant="primary"
                size="sm"
                icon={Activity}
                onClick={() => navigate(`/cases/${caseData.id}/analysis`)}
              >
                {hasPersistedResults
                  ? 'Open Analysis Workspace'
                  : 'Open Preview Workspace'}
              </Button>
            </div>
          ) : (
            <div className="rounded-xl border border-dashed border-amber-300 bg-amber-50/50 p-6 text-center space-y-3">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full border border-amber-200 bg-amber-100 text-amber-700">
                <Sparkles className="h-6 w-6" />
              </div>

              <div>
                <h4 className="text-sm font-bold text-slate-900">
                  {isCheckingResults
                    ? 'Checking saved analysis results'
                    : 'No segmentation result yet'}
                </h4>

                <p className="mx-auto mt-1 max-w-md text-xs leading-relaxed text-slate-600">
                  {resultCheckError
                    ? resultCheckError
                    : IS_MOCK_MODE
                      ? 'Upload a sample angiography image to preview the mock workflow. Mock mode does not run real model inference.'
                      : 'Upload a supported angiography image to run the connected U-Net or Attention U-Net inference pipeline.'}
                </p>
              </div>

              {resultCheckError && !IS_MOCK_MODE && (
                <p className="text-[11px] text-slate-500">
                  The result lookup may be unavailable while the backend database
                  schema is being initialized.
                </p>
              )}

              <Button
                variant="primary"
                size="sm"
                icon={UploadCloud}
                onClick={() => navigate(`/upload?caseId=${caseData.id}`)}
              >
                Upload Scan
              </Button>

              <p className="text-[11px] text-slate-500">
                Research-use only. Segmentation output is not a clinical diagnosis.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
