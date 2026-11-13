
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
import { IS_MOCK_MODE } from '../services/api';

const REAL_SUPPORTED_FORMATS = [
  '.png',
  '.jpg',
  '.jpeg',
  '.tif',
  '.tiff',
  '.bmp',
];

export default function UploadScan() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { cases, uploadCaseScan } = useCases();

  const preselectedCaseId = searchParams.get('caseId') || '';

  const [selectedCaseId, setSelectedCaseId] = useState(preselectedCaseId);
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedModel, setSelectedModel] = useState('unet');
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadResult, setUploadResult] = useState(null);

  const fileInputRef = useRef(null);

  const supportedFormats = IS_MOCK_MODE
    ? UPLOAD_CONFIG.supportedFormats
    : REAL_SUPPORTED_FORMATS;

  // Sync selected case with URL or select the first available case.
  useEffect(() => {
    if (preselectedCaseId) {
      setSelectedCaseId(preselectedCaseId);
    } else if (cases.length > 0 && !selectedCaseId) {
      const pendingCase = cases.find((c) => !c.file);
      setSelectedCaseId(pendingCase ? pendingCase.id : cases[0].id);
    }
  }, [preselectedCaseId, cases, selectedCaseId]);

  const validateFile = (file) => {
    if (!file) return false;

    if (file.size > UPLOAD_CONFIG.maxSizeBytes) {
      setError(
        `File size exceeds the limit of ${UPLOAD_CONFIG.maxSizeMB} MB.`
      );
      return false;
    }

    const extension = '.' + file.name.split('.').pop().toLowerCase();

    const isSupported = supportedFormats.some(
      (format) => format.trim().toLowerCase() === extension
    );

    if (!isSupported) {
      setError(
        `Unsupported file format "${extension}". Supported formats: ${supportedFormats.join(', ')}`
      );
      return false;
    }

    setError('');
    return true;
  };

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

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];

      if (validateFile(file)) {
        setSelectedFile(file);
        setUploadResult(null);
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];

      if (validateFile(file)) {
        setSelectedFile(file);
        setUploadResult(null);
      }
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setUploadProgress(0);
    setError('');

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();

    if (!selectedCaseId) {
      setError('Please select an associated research case.');
      return;
    }

    if (!selectedFile) {
      setError('Please select an angiography image file to upload.');
      return;
    }

    try {
      setIsUploading(true);
      setError('');
      setUploadProgress(5);

      const result = await uploadCaseScan(
        selectedCaseId,
        selectedFile,
        selectedModel,
        (progress) => setUploadProgress(progress)
      );

      setUploadResult(result);
    } catch (err) {
      setError(err.message || 'Upload operation failed.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-slate-500 text-xs font-semibold uppercase tracking-wider mb-1">
          <UploadCloud className="w-4 h-4 text-blue-600" />
          <span>Image Intake</span>
        </div>

        <h2 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
          Upload Coronary Angiogram Scan
        </h2>

        <p className="text-xs sm:text-sm text-slate-500 mt-0.5">
          Select an angiography image, associate it with a study case,
          and choose a segmentation model.
        </p>
      </div>

      {/* Upload Success */}
      {uploadResult && (
        <div className="p-5 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-900 space-y-3 animate-in fade-in">
          <div className="flex items-center gap-2 text-emerald-800 font-semibold text-sm">
            <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0" />

            <span>
              {uploadResult.isMock
                ? 'Mock Scan Upload Completed'
                : 'Scan Uploaded & Segmentation Completed'}
            </span>
          </div>

          <p className="text-xs text-emerald-800 leading-relaxed">
            File{' '}
            <strong>
              {uploadResult.file?.name || 'Uploaded scan'}
            </strong>{' '}
            has been associated with case{' '}
            <strong>{uploadResult.caseId}</strong>.

            {uploadResult.isMock ? (
              <>
                {' '}The mock upload completed successfully. No real AI
                inference was performed.
              </>
            ) : (
              <>
                {' '}Segmentation completed using{' '}
                <strong>{uploadResult.modelName}</strong>.
                {' '}Segmentation result ID:{' '}
                <strong>{uploadResult.segmentationId}</strong>.
              </>
            )}
          </p>

          <div className="flex items-center gap-3 pt-1">
            <Button
              variant="primary"
              size="sm"
              icon={ArrowRight}
              onClick={() => navigate(`/cases/${uploadResult.caseId}`)}
            >
              View Updated Case
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setUploadResult(null);
                setSelectedFile(null);
                setUploadProgress(0);

                if (fileInputRef.current) {
                  fileInputRef.current.value = '';
                }
              }}
            >
              Upload Another Scan
            </Button>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="p-3.5 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-700 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rose-500 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Upload Form */}
      <form
        onSubmit={handleUpload}
        className="bg-white rounded-xl border border-slate-200 shadow-2xs p-6 space-y-6"
      >
        {/* Step 1: Select Case */}
        <div className="space-y-1.5">
          <label
            htmlFor="upload-case-select"
            className="block text-xs font-semibold text-slate-700 uppercase tracking-wide"
          >
            1. Select Associated Research Case{' '}
            <span className="text-rose-500">*</span>
          </label>

          <div className="relative">
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
          </div>

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

        {/* Step 2: Select Segmentation Model */}
        <div className="space-y-2">
          <label
            htmlFor="segmentation-model"
            className="block text-xs font-semibold text-slate-700 uppercase tracking-wide"
          >
            2. Select Segmentation Model
          </label>

          <select
            id="segmentation-model"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={isUploading}
            className="w-full rounded-lg border border-slate-300 text-sm py-2.5 px-3 bg-white text-slate-800 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100 cursor-pointer"
          >
            <option value="unet">U-Net</option>
            <option value="attention_unet">Attention U-Net</option>
          </select>

          <p className="text-[11px] text-slate-500">
            {IS_MOCK_MODE
              ? 'Demo mode: model selection is recorded, but real inference is not executed.'
              : 'Choose the trained deep-learning architecture for segmentation inference.'}
          </p>
        </div>

        {/* Step 3: Upload Scan */}
        <div className="space-y-1.5">
          <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wide">
            3. Choose Angiography Scan File{' '}
            <span className="text-rose-500">*</span>
          </label>

          <input
            ref={fileInputRef}
            type="file"
            id="file-upload"
            onChange={handleFileChange}
            accept={supportedFormats.join(',')}
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
                Supported formats:{' '}
                <strong className="text-slate-700">
                  {supportedFormats.join(', ')}
                </strong>{' '}
                &bull; Max size: {UPLOAD_CONFIG.maxSizeMB} MB
              </p>

              <p className="text-[11px] text-slate-400 mt-1">
                Optimized for 2D X-ray coronary angiography frames
                (ARCADE standard)
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
                    {formatBytes(selectedFile.size)} &bull;{' '}
                    {selectedFile.type || 'Angiogram image'}
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

        {/* Upload Progress */}
        {isUploading && (
          <div className="space-y-2 p-4 rounded-xl bg-blue-50/50 border border-blue-100">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-blue-900">
                Uploading scan and running segmentation...
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

        {/* Submit Actions */}
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
            {isUploading ? 'Processing Scan...' : 'Upload & Segment Scan'}
          </Button>
        </div>
      </form>
    </div>
  );
}
