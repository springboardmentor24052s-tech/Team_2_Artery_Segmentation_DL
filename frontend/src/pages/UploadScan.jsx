import React, { useState, useRef, useEffect } from 'react';
import { useSearchParams, useNavigate, Link } from 'react-router-dom';
import {
  UploadCloud,
  AlertCircle,
  X,
  FileImage,
  CheckCircle2,
  ArrowRight,
} from 'lucide-react';

import Button from '../components/Button';
import { useCases } from '../context/CaseContext';
import { UPLOAD_CONFIG } from '../utils/constants';
import { formatBytes } from '../utils/formatters';

const SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg'];

const getSeverityStyles = (severity) => {
  switch (String(severity || '').toLowerCase()) {
    case 'normal':
      return 'bg-emerald-100 text-emerald-800';
    case 'mild':
      return 'bg-yellow-100 text-yellow-800';
    case 'moderate':
      return 'bg-orange-100 text-orange-800';
    case 'severe':
      return 'bg-rose-100 text-rose-800';
    case 'critical':
      return 'bg-red-100 text-red-800';
    default:
      return 'bg-slate-100 text-slate-700';
  }
};

const formatPercentage = (value) => {
  const numericValue = Number(value);
  return Number.isFinite(numericValue)
    ? `${numericValue.toFixed(2)}%`
    : '—';
};

export default function UploadScan() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const { cases, uploadCaseScan } = useCases();

  const preselectedCaseId = searchParams.get('caseId') || '';

  const [selectedCaseId, setSelectedCaseId] =
    useState(preselectedCaseId);

  const [selectedModel, setSelectedModel] = useState('syntax');

  const [selectedFile, setSelectedFile] = useState(null);
  const [originalPreview, setOriginalPreview] = useState(null);

  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState('');

  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const [uploadResult, setUploadResult] = useState(null);

  const fileInputRef = useRef(null);

  // 1. CASE SELECTION
  useEffect(() => {
    if (preselectedCaseId) {
      setSelectedCaseId(preselectedCaseId);
    } else if (cases.length > 0 && !selectedCaseId) {
      const pendingCase = cases.find((c) => !c.file);

      setSelectedCaseId(
        pendingCase ? pendingCase.id : cases[0].id
      );
    }
  }, [preselectedCaseId, cases, selectedCaseId]);

  // 2. ORIGINAL IMAGE PREVIEW
  useEffect(() => {
    if (!selectedFile) {
      setOriginalPreview(null);
      return;
    }

    const previewUrl = URL.createObjectURL(selectedFile);
    setOriginalPreview(previewUrl);

    return () => {
      URL.revokeObjectURL(previewUrl);
    };
  }, [selectedFile]);

  // 3. FILE VALIDATION
  const validateFile = (file) => {
    if (!file) return false;

    if (file.size > UPLOAD_CONFIG.maxSizeBytes) {
      setError(
        `File size exceeds the limit of ${UPLOAD_CONFIG.maxSizeMB} MB.`
      );
      return false;
    }

    const extension =
      '.' + file.name.split('.').pop().toLowerCase();

    if (!SUPPORTED_FORMATS.includes(extension)) {
      setError(
        `Unsupported file format "${extension}". Supported formats: PNG, JPG, JPEG.`
      );
      return false;
    }

    setError('');
    return true;
  };

  // 4. DRAG AND DROP
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();

    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();

    setDragActive(false);

    if (e.dataTransfer.files?.[0]) {
      const file = e.dataTransfer.files[0];

      if (validateFile(file)) {
        setSelectedFile(file);
        setUploadResult(null);
      }
    }
  };

  // 5. FILE SELECTION
  const handleFileChange = (e) => {
    if (e.target.files?.[0]) {
      const file = e.target.files[0];

      if (validateFile(file)) {
        setSelectedFile(file);
        setUploadResult(null);
      }

      e.target.value = '';
    }
  };

  // 6. REMOVE SELECTED FILE
  const handleRemoveFile = () => {
    setSelectedFile(null);
    setUploadProgress(0);
    setUploadResult(null);
    setError('');

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // 7. UPLOAD AND SEGMENTATION
  const handleUpload = async (e) => {
    e.preventDefault();

    if (!selectedCaseId) {
      setError('Please select an associated research case.');
      return;
    }

    if (!selectedFile) {
      setError('Please select an angiography image.');
      return;
    }

    try {
      setIsUploading(true);
      setError('');
      setUploadResult(null);
      setUploadProgress(5);

      const result = await uploadCaseScan(
        selectedCaseId,
        selectedFile,
        (progress) => setUploadProgress(progress),
        selectedModel
      );

      setUploadProgress(100);
      setUploadResult(result);

    } catch (err) {
      setError(
        err.message || 'Image upload and segmentation failed.'
      );
    } finally {
      setIsUploading(false);
    }
  };

  // 8. RESET AFTER SUCCESSFUL UPLOAD
  const handleUploadAnother = () => {
    setUploadResult(null);
    setSelectedFile(null);
    setUploadProgress(0);
    setError('');

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const isStenosis = selectedModel === 'stenosis';
  const isCombined = selectedModel === 'combined';

  // 9. USER INTERFACE
  return (
    <div className="max-w-4xl mx-auto space-y-6">

      {/* PAGE HEADER */}
      <div>
        <div className="flex items-center gap-2 text-slate-500 text-xs font-semibold uppercase tracking-wider mb-1">
          <UploadCloud className="w-4 h-4 text-blue-600" />
          <span>Image Intake</span>
        </div>

        <h2 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
          Upload Coronary Angiogram Scan
        </h2>

        <p className="text-xs sm:text-sm text-slate-500 mt-1">
          Upload a 2D X-ray coronary angiography image
          for AI-powered artery segmentation.
        </p>
      </div>

      {/* ERROR MESSAGE */}
      {error && (
        <div className="p-3.5 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-700 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rose-500 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* SUCCESS AND SEGMENTATION RESULTS */}
      {uploadResult && uploadResult.segmentation && (
        <div className="p-5 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-900 space-y-4">

          <div className="flex items-center gap-2 text-emerald-800 font-semibold text-sm">
            <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0" />
            <span>AI Segmentation Completed Successfully</span>
          </div>

          <p className="text-xs text-emerald-800">
            File <strong>{uploadResult.file.name}</strong> was
            processed successfully for case{' '}
            <strong>{uploadResult.caseId}</strong>.
          </p>

          <p className="text-xs font-semibold text-emerald-800">
            Model used: {uploadResult.segmentation.model ||
              (isStenosis
                ? 'Attention U-Net V4'
                : isCombined
                  ? 'Syntax U-Net + Attention U-Net V4'
                  : 'Syntax U-Net')}
          </p>

          {/* ORIGINAL AND PREDICTED MASK */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

            {/* ORIGINAL IMAGE */}
            <div className="bg-white rounded-xl border border-slate-200 p-3 space-y-3">
              <h3 className="text-sm font-semibold text-slate-800">
                Original Angiography Image
              </h3>

              {originalPreview && (
                <img
                  src={originalPreview}
                  alt="Original angiography image"
                  className="w-full aspect-square object-contain bg-black rounded-lg"
                />
              )}

              <p className="text-xs text-slate-500">
                Uploaded input image
              </p>
            </div>

            {/* SEGMENTATION MASK */}
            <div className="bg-white rounded-xl border border-slate-200 p-3 space-y-3">
              <h3 className="text-sm font-semibold text-slate-800">
                {isStenosis
                  ? 'Stenosis Segmentation Mask'
                  : 'Artery Segmentation Mask'}
              </h3>

              <img
                src={uploadResult.maskUrl}
                alt="Predicted segmentation mask"
                className="w-full aspect-square object-contain bg-black rounded-lg"
                onError={() =>
                  setError('Unable to load the generated segmentation mask.')
                }
              />

              <p className="text-xs text-slate-500">
                White: Detected regions | Black: Background
              </p>
            </div>

          </div>

          {/* SECOND MODEL MASK FOR COMBINED ANALYSIS */}
          {isCombined && uploadResult.stenosisMaskUrl && (
            <div className="bg-white rounded-xl border border-slate-200 p-3 space-y-3">
              <h3 className="text-sm font-semibold text-slate-800">
                Stenosis Segmentation Mask — Attention U-Net V4
              </h3>

              <img
                src={uploadResult.stenosisMaskUrl}
                alt="Stenosis segmentation mask"
                className="w-full max-h-96 object-contain bg-black rounded-lg"
                onError={() =>
                  setError('Unable to load the stenosis segmentation mask.')
                }
              />

              <p className="text-xs text-slate-500">
                White: Predicted stenosis regions | Black: Background
              </p>
            </div>
          )}

          {/* LOCALIZED STENOSIS CANDIDATES */}
          {isCombined && uploadResult.localizedImageUrl && (
            <div className="bg-white rounded-xl border border-slate-200 p-3 space-y-3">
              <h3 className="text-sm font-semibold text-slate-800">
                Stenosis Candidate Localization
              </h3>

              <img
                src={uploadResult.localizedImageUrl}
                alt="Localized stenosis candidates"
                className="w-full max-h-96 object-contain bg-black rounded-lg"
                onError={() =>
                  setError('Unable to load the localized candidate image.')
                }
              />

              <p className="text-xs text-slate-500">
                Candidate regions identified by the experimental localization pipeline.
              </p>
            </div>
          )}

          {/* STENOSIS PROBABILITY MAP */}
          {(isStenosis || isCombined) && uploadResult.probabilityMapUrl && (
            <div className="bg-white rounded-xl border border-slate-200 p-3 space-y-3">
              <h3 className="text-sm font-semibold text-slate-800">
                Stenosis Probability Map
              </h3>

              <img
                src={uploadResult.probabilityMapUrl}
                alt="Stenosis probability map"
                className="w-full max-h-96 object-contain bg-black rounded-lg"
              />

              <p className="text-xs text-slate-500">
                Displays the model's predicted probability values for
                stenosis regions.
              </p>
            </div>
          )}

          {isCombined && uploadResult.segmentation.notice && (
            <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-lg p-3">
              {uploadResult.segmentation.notice}
            </p>
          )}

          {/* EXPERIMENTAL GEOMETRY-BASED STENOSIS ANALYSIS */}
          {uploadResult.geometryAnalysis && (
            <div className="bg-white rounded-xl border border-slate-200 p-4 sm:p-5 space-y-4">

              {uploadResult.geometryAnalysis.success ? (
                <>
                  <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2">
                    <div>
                      <h3 className="text-sm font-bold text-slate-900">
                        Geometry-Based Stenosis Analysis
                      </h3>

                      <p className="text-xs text-slate-500 mt-1">
                        Experimental narrowing estimates from the Syntax U-Net artery mask.
                      </p>
                    </div>

                    {uploadResult.geometryAnalysis.reportUrl && (
                      <a
                        href={uploadResult.geometryAnalysis.reportUrl}
                        target="_blank"
                        rel="noreferrer"
                        className="text-xs font-semibold text-blue-600 hover:text-blue-800 hover:underline shrink-0"
                      >
                        View JSON Report
                      </a>
                    )}
                  </div>

                  {(() => {
                    const geometry = uploadResult.geometryAnalysis;

                    const geometryResults =
                      geometry.results || geometry.report?.results || [];

                    const branchCount =
                      geometry.totalBranches ??
                      geometry.branchCount ??
                      geometryResults.length;

                    const severityCounts = geometryResults.reduce(
                      (counts, item) => {
                        const candidate =
                          item.deepest_candidate ||
                          item.deepestCandidate ||
                          item;

                        const severity =
                          candidate.severity_classification?.severity ||
                          candidate.severity ||
                          'unclassified';

                        const key = String(severity).toLowerCase();

                        counts[key] = (counts[key] || 0) + 1;

                        return counts;
                      },
                      {}
                    );

                    return (
                      <>
                        {/* GEOMETRY SUMMARY */}
                        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">

                          <div className="rounded-lg bg-slate-50 border border-slate-200 p-3">
                            <p className="text-[11px] text-slate-500">
                              Branches Evaluated
                            </p>

                            <p className="text-xl font-bold text-slate-900 mt-1">
                              {branchCount}
                            </p>
                          </div>

                          <div className="rounded-lg bg-slate-50 border border-slate-200 p-3">
                            <p className="text-[11px] text-slate-500">
                              Results Returned
                            </p>

                            <p className="text-xl font-bold text-slate-900 mt-1">
                              {geometryResults.length}
                            </p>
                          </div>

                          <div className="col-span-2 sm:col-span-1 rounded-lg bg-slate-50 border border-slate-200 p-3">
                            <p className="text-[11px] text-slate-500">
                              Classification Status
                            </p>

                            <p className="text-sm font-bold text-amber-700 mt-2">
                              Provisional
                            </p>
                          </div>

                        </div>

                        {/* BRANCH-WISE RESULTS TABLE */}
                        {geometryResults.length > 0 ? (
                          <div className="overflow-x-auto rounded-lg border border-slate-200">
                            <table className="w-full min-w-[620px] text-xs text-left">

                              <thead className="bg-slate-50 text-slate-600">
                                <tr>
                                  <th className="px-3 py-3 font-semibold">
                                    Branch
                                  </th>

                                  <th className="px-3 py-3 font-semibold">
                                    Estimated Narrowing
                                  </th>

                                  <th className="px-3 py-3 font-semibold">
                                    Provisional Category
                                  </th>

                                  <th className="px-3 py-3 font-semibold">
                                    Reference Quality
                                  </th>
                                </tr>
                              </thead>

                              <tbody className="divide-y divide-slate-100">
                                {geometryResults.map((item, index) => {
                                  const candidate =
                                    item.deepest_candidate ||
                                    item.deepestCandidate ||
                                    item;

                                  const branchId =
                                    item.branch_id ??
                                    item.branchId ??
                                    candidate.branch_id ??
                                    index + 1;

                                  const narrowing =
                                    candidate.estimated_narrowing_percent ??
                                    candidate.narrowing_percentage ??
                                    candidate.narrowing_percent ??
                                    candidate.stenosis_percentage;

                                  const severity =
                                    candidate.severity_classification?.severity ||
                                    candidate.severity ||
                                    'Unclassified';

                                  const quality =
                                    candidate.reference_quality ||
                                    candidate.quality_label ||
                                    item.reference_quality ||
                                    item.quality_label ||
                                    'Not available';

                                  return (
                                    <tr
                                      key={`${branchId}-${index}`}
                                      className="bg-white"
                                    >
                                      <td className="px-3 py-3 font-semibold text-slate-800">
                                        {branchId}
                                      </td>

                                      <td className="px-3 py-3 text-slate-700">
                                        {formatPercentage(narrowing)}
                                      </td>

                                      <td className="px-3 py-3">
                                        <span
                                          className={`inline-flex px-2 py-1 rounded-full text-[11px] font-semibold capitalize ${getSeverityStyles(severity)}`}
                                        >
                                          {severity}
                                        </span>
                                      </td>

                                      <td className="px-3 py-3 text-slate-600">
                                        {String(quality).replaceAll('_', ' ')}
                                      </td>
                                    </tr>
                                  );
                                })}
                              </tbody>

                            </table>
                          </div>
                        ) : (
                          <p className="text-xs text-slate-500 bg-slate-50 rounded-lg p-3">
                            Geometry analysis completed, but no branch-wise results were returned.
                          </p>
                        )}

                        {/* SEVERITY DISTRIBUTION */}
                        <div className="space-y-2">
                          <p className="text-xs font-semibold text-slate-700">
                            Provisional Severity Distribution
                          </p>

                          <div className="flex flex-wrap gap-2">
                            {[
                              'normal',
                              'mild',
                              'moderate',
                              'severe',
                              'critical',
                            ].map((level) =>
                              severityCounts[level] ? (
                                <span
                                  key={level}
                                  className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold capitalize ${getSeverityStyles(level)}`}
                                >
                                  {level}: {severityCounts[level]}
                                </span>
                              ) : null
                            )}
                          </div>
                        </div>
                      </>
                    );
                  })()}

                  {/* EXPERIMENTAL DISCLAIMER */}
                  <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 space-y-1">
                    <p className="text-xs font-semibold text-amber-900">
                      Research / experimental output — not a clinical diagnosis
                    </p>

                    <p className="text-[11px] leading-relaxed text-amber-800">
                      Narrowing values are estimated from predicted 2D vessel-mask geometry.
                      Severity categories use provisional thresholds and have not been clinically
                      validated. Results may be affected by segmentation quality and image conditions;
                      they must not be used for clinical decisions.
                    </p>
                  </div>
                </>
              ) : (
                <div className="rounded-lg border border-amber-200 bg-amber-50 p-3">
                  <p className="text-xs font-semibold text-amber-900">
                    Geometry analysis was not completed
                  </p>

                  <p className="text-xs text-amber-800 mt-1">
                    The image segmentation succeeded, but the additional geometry-based analysis
                    could not be completed.
                    {uploadResult.geometryAnalysis.error
                      ? ` ${uploadResult.geometryAnalysis.error}`
                      : ''}
                  </p>
                </div>
              )}
            </div>
          )}

          {/* SEGMENTATION METRICS */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">

            <div className="bg-white p-4 rounded-lg border border-slate-200">
              <p className="text-xs text-slate-500">
                Image Dimensions
              </p>

              <p className="text-lg font-bold text-slate-800 mt-1">
                {uploadResult.segmentation.image_dimensions.width}
                {' × '}
                {uploadResult.segmentation.image_dimensions.height}
              </p>
            </div>

            <div className="bg-white p-4 rounded-lg border border-slate-200">
              <p className="text-xs text-slate-500">
                Foreground Pixels
              </p>

              <p className="text-lg font-bold text-slate-800 mt-1">
                {uploadResult.segmentation.foreground_pixels.toLocaleString()}
              </p>
            </div>

          </div>

          {/* IMAGE ID */}
          <div className="text-xs text-slate-600 break-all">
            <strong>Image ID:</strong>{' '}
            {uploadResult.segmentation.image_id}
          </div>

          {/* ACTION BUTTONS */}
          <div className="flex flex-wrap items-center gap-3 pt-2">

            <Button
              variant="primary"
              size="sm"
              icon={ArrowRight}
              onClick={() =>
                navigate(`/cases/${uploadResult.caseId}`)
              }
            >
              View Updated Case
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={handleUploadAnother}
            >
              Upload Another Scan
            </Button>

          </div>
        </div>
      )}

      {/* UPLOAD FORM */}
      <form
        onSubmit={handleUpload}
        className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-6"
      >

        {/* STEP 1: SELECT CASE */}
        <div className="space-y-1.5">
          <label
            htmlFor="upload-case-select"
            className="block text-xs font-semibold text-slate-700 uppercase tracking-wide"
          >
            1. Select Associated Research Case
            <span className="text-rose-500"> *</span>
          </label>

          <select
            id="upload-case-select"
            value={selectedCaseId}
            onChange={(e) => setSelectedCaseId(e.target.value)}
            disabled={isUploading}
            className="w-full rounded-lg border border-slate-300 text-sm py-2.5 px-3 bg-white text-slate-800 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100 cursor-pointer"
          >
            {cases.length === 0 ? (
              <option value="">No cases created yet</option>
            ) : (
              cases.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.id} — {c.name} ({c.status})
                </option>
              ))
            )}
          </select>

          <p className="text-[11px] text-slate-500">
            Need a new case?{' '}
            <Link
              to="/cases/create"
              className="text-blue-600 hover:underline font-medium"
            >
              Create a new research case first
            </Link>
          </p>
        </div>

        {/* STEP 2: SELECT MODEL */}
        <div className="space-y-2">
          <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wide">
            2. Select Analysis Mode
            <span className="text-rose-500"> *</span>
          </label>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">

            <button
              type="button"
              disabled={isUploading}
              onClick={() => setSelectedModel('syntax')}
              className={`text-left p-4 rounded-xl border-2 transition-all ${
                selectedModel === 'syntax'
                  ? 'border-blue-600 bg-blue-50'
                  : 'border-slate-200 bg-white hover:border-slate-300'
              }`}
            >
              <p className="text-sm font-semibold text-slate-800">
                Syntax Segmentation
              </p>

              <p className="text-xs text-slate-500 mt-1">
                Syntax U-Net – detects coronary artery structures.
              </p>

              {selectedModel === 'syntax' && (
                <span className="text-xs font-semibold text-blue-700 mt-2 block">
                  Selected
                </span>
              )}
            </button>

            <button
              type="button"
              disabled={isUploading}
              onClick={() => setSelectedModel('stenosis')}
              className={`text-left p-4 rounded-xl border-2 transition-all ${
                selectedModel === 'stenosis'
                  ? 'border-blue-600 bg-blue-50'
                  : 'border-slate-200 bg-white hover:border-slate-300'
              }`}
            >
              <p className="text-sm font-semibold text-slate-800">
                Stenosis Segmentation
              </p>

              <p className="text-xs text-slate-500 mt-1">
                Attention U-Net V4 – predicts potential stenosis regions.
              </p>

              {selectedModel === 'stenosis' && (
                <span className="text-xs font-semibold text-blue-700 mt-2 block">
                  Selected
                </span>
              )}
            </button>

            <button
              type="button"
              disabled={isUploading}
              onClick={() => setSelectedModel('combined')}
              className={`text-left p-4 rounded-xl border-2 transition-all sm:col-span-2 ${
                selectedModel === 'combined'
                  ? 'border-emerald-600 bg-emerald-50'
                  : 'border-slate-200 bg-white hover:border-slate-300'
              }`}
            >
              <p className="text-sm font-semibold text-slate-800">
                Complete AI Analysis (Both Models)
              </p>

              <p className="text-xs text-slate-500 mt-1">
                Runs Syntax U-Net and Attention U-Net V4 on the same image.
                Provides artery segmentation, stenosis prediction,
                probability map and candidate localization.
              </p>

              {selectedModel === 'combined' && (
                <span className="text-xs font-semibold text-emerald-700 mt-2 block">
                  Selected
                </span>
              )}
            </button>

          </div>
        </div>

        {/* STEP 3: SELECT IMAGE */}
        <div className="space-y-1.5">
          <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wide">
            3. Choose Angiography Scan File
            <span className="text-rose-500"> *</span>
          </label>

          <input
            ref={fileInputRef}
            type="file"
            id="file-upload"
            onChange={handleFileChange}
            accept={SUPPORTED_FORMATS.join(',')}
            className="hidden"
            disabled={isUploading}
          />

          {!selectedFile ? (
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
                dragActive
                  ? 'border-blue-500 bg-blue-50/50 scale-[1.005]'
                  : 'border-slate-300 hover:border-slate-400 bg-slate-50/50 hover:bg-slate-50'
              }`}
            >
              <div className="w-12 h-12 rounded-full bg-blue-50 text-blue-600 mx-auto flex items-center justify-center mb-3 border border-blue-100">
                <UploadCloud className="w-6 h-6" />
              </div>

              <p className="text-sm font-semibold text-slate-800">
                Drag and drop your scan here, or{' '}
                <span className="text-blue-600 underline">
                  browse files
                </span>
              </p>

              <p className="text-xs text-slate-500 mt-1.5">
                Supported formats: PNG, JPG, JPEG
                {' '}• Max size: {UPLOAD_CONFIG.maxSizeMB} MB
              </p>

              <p className="text-[11px] text-slate-400 mt-1">
                Optimized for 2D X-ray coronary angiography images
              </p>
            </div>
          ) : (
            <div className="p-4 rounded-xl border border-slate-200 bg-slate-50 flex items-center justify-between gap-3">

              <div className="flex items-center gap-3 min-w-0">
                <div className="w-10 h-10 rounded-lg bg-blue-100 text-blue-700 flex items-center justify-center shrink-0">
                  <FileImage className="w-5 h-5" />
                </div>

                <div className="min-w-0">
                  <p className="text-xs font-semibold text-slate-900 truncate">
                    {selectedFile.name}
                  </p>

                  <p className="text-[11px] text-slate-500">
                    {formatBytes(selectedFile.size)}
                    {' • '}
                    {selectedFile.type || 'Angiography image'}
                  </p>
                </div>
              </div>

              {!isUploading && (
                <div className="flex items-center gap-2">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    Replace
                  </Button>

                  <button
                    type="button"
                    onClick={handleRemoveFile}
                    className="text-slate-400 hover:text-rose-600 p-1.5 rounded-md hover:bg-slate-200 transition-colors"
                    title="Remove selected file"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {/* UPLOAD PROGRESS */}
        {isUploading && (
          <div className="space-y-2 p-4 rounded-xl bg-blue-50/50 border border-blue-100">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-blue-900">
                {isCombined
                  ? 'Running combined Syntax and Stenosis analysis...'
                  : `Running ${isStenosis ? 'Stenosis' : 'Syntax'} segmentation...`}
              </span>

              <span className="font-mono font-medium text-blue-700">
                {uploadProgress}%
              </span>
            </div>

            <div className="w-full h-2 rounded-full bg-blue-100 overflow-hidden">
              <div
                className="h-full bg-blue-600 transition-all duration-150 rounded-full"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          </div>
        )}

        {/* FORM ACTIONS */}
        <div className="flex items-center justify-end gap-3 pt-4 border-t border-slate-100">
          <Button
            type="button"
            variant="outline"
            size="md"
            onClick={() => navigate('/cases')}
            disabled={isUploading}
          >
            Cancel
          </Button>

          <Button
            type="submit"
            variant="primary"
            size="md"
            icon={UploadCloud}
            isLoading={isUploading}
            disabled={!selectedFile || isUploading || !selectedCaseId}
          >
            {isUploading
              ? 'Processing Image...'
              : isCombined
                ? 'Upload & Run Complete AI Analysis'
                : `Upload & Run ${isStenosis ? 'Stenosis' : 'Syntax'} Segmentation`}
          </Button>
        </div>

      </form>
    </div>
  );
}