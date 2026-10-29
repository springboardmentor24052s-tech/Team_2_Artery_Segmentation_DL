import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import {
  ArrowLeft,
  FileImage,
  UploadCloud,
  Cpu,
  Clock,
  CheckCircle2,
  AlertTriangle,
  Info,
  Calendar,
  Layers,
  Sparkles,
  FileText,
  ShieldAlert,
  Bot,
  ExternalLink,
} from 'lucide-react';
import Button from '../components/Button';
import StatusBadge from '../components/StatusBadge';
import LoadingState from '../components/LoadingState';
import { useCases } from '../context/CaseContext';
import { formatDate, formatBytes } from '../utils/formatters';
import { CASE_STATUS } from '../utils/constants';

export default function CaseDetails() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { cases, isLoading } = useCases();

  const caseData = cases.find((c) => c.id === id);

  if (isLoading) {
    return <LoadingState message="Loading research case details..." />;
  }

  if (!caseData) {
    return (
      <div className="max-w-2xl mx-auto py-12 text-center space-y-4">
        <div className="w-12 h-12 rounded-full bg-rose-50 text-rose-600 flex items-center justify-center mx-auto border border-rose-200">
          <AlertTriangle className="w-6 h-6" />
        </div>
        <h2 className="text-xl font-bold text-slate-800">Case Not Found</h2>
        <p className="text-sm text-slate-500">
          The requested research case with identifier <strong className="font-mono text-slate-700">{id}</strong> does not exist in the current session.
        </p>
        <Button variant="outline" size="sm" icon={ArrowLeft} onClick={() => navigate('/cases')}>
          Return to Case History
        </Button>
      </div>
    );
  }

  const hasUploadedFile = Boolean(caseData.file);

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
            <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Target Vessel</p>
            <p className="text-sm font-medium text-slate-800 mt-1">
              {caseData.targetRegion || 'Full Coronary Tree'}
            </p>
          </div>

          <div>
            <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Imaging Modality</p>
            <p className="text-sm font-medium text-slate-800 mt-1 truncate" title={caseData.imageModality}>
              {caseData.imageModality || 'X-ray Coronary Angiography (2D XCA)'}
            </p>
          </div>

          <div>
            <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">File Status</p>
            <p className="text-sm font-medium text-slate-800 mt-1">
              {hasUploadedFile ? (
                <span className="text-emerald-700 flex items-center gap-1 font-semibold text-xs">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Scan Attached
                </span>
              ) : (
                <span className="text-amber-700 flex items-center gap-1 font-semibold text-xs">
                  <Clock className="w-3.5 h-3.5" /> Awaiting Image
                </span>
              )}
            </p>
          </div>

          <div>
            <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Internal Research Notes</p>
            <p className="text-sm text-slate-700 mt-1 truncate" title={caseData.notes}>
              {caseData.notes || 'Synthetic benchmark frame'}
            </p>
          </div>
        </div>
      </div>

      {/* Grid: Image / File Details on Left, AI Segmentation Output Placeholder on Right */}
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
                {formatBytes(caseData.file.size)}
              </span>
            )}
          </div>

          {hasUploadedFile ? (
            <div className="space-y-4">
              {/* Scan Preview Canvas/Visualizer placeholder */}
              <div className="relative rounded-lg overflow-hidden border border-slate-200 bg-slate-950 flex flex-col items-center justify-center p-6 text-center min-h-[220px]">
                {/* Simulated Angiogram Grid Representation */}
                <div className="w-32 h-32 rounded-full border border-slate-800/80 bg-radial from-slate-800 to-slate-950 flex items-center justify-center relative shadow-inner">
                  <div className="w-24 h-24 rounded-full border border-dashed border-blue-500/30 flex items-center justify-center">
                    <FileImage className="w-10 h-10 text-slate-400 animate-pulse" />
                  </div>
                  {/* Anatomical coronary branch illustration cue */}
                  <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                    <span className="text-[10px] font-mono text-blue-400/80 bg-slate-900/80 px-2 py-0.5 rounded">
                      {caseData.imageType}
                    </span>
                  </div>
                </div>

                <div className="mt-4 text-xs text-slate-300">
                  <p className="font-semibold text-white">{caseData.file.name}</p>
                  <p className="text-slate-400 text-[11px] mt-0.5">
                    Frame Resolution: {caseData.file.dimensions || '512 x 512 px'} &bull; Uploaded {formatDate(caseData.file.uploadedAt)}
                  </p>
                </div>
              </div>

              {/* DICOM / File Headers */}
              {caseData.file.dicomMeta ? (
                <div className="bg-slate-50 rounded-lg p-3 border border-slate-200 text-xs space-y-1.5">
                  <p className="font-semibold text-slate-700 text-[11px] uppercase tracking-wider">
                    Angiography Acquisition Metadata
                  </p>
                  <div className="grid grid-cols-2 gap-2 text-slate-600 text-[11px] font-mono">
                    <div>Photometric: {caseData.file.dicomMeta.photometric}</div>
                    <div>Tube Voltage: {caseData.file.dicomMeta.kvp}</div>
                    <div>Frame Rate: {caseData.file.dicomMeta.frameRate}</div>
                    <div>Field of View: {caseData.file.dicomMeta.fieldOfView}</div>
                  </div>
                </div>
              ) : (
                <div className="bg-slate-50 rounded-lg p-3 border border-slate-200 text-xs text-slate-500">
                  Standard format verification complete.
                </div>
              )}
            </div>
          ) : (
            <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50/60 p-8 text-center space-y-3">
              <div className="w-10 h-10 rounded-full bg-slate-200 text-slate-500 mx-auto flex items-center justify-center">
                <FileImage className="w-5 h-5" />
              </div>
              <h4 className="text-sm font-semibold text-slate-800">No Scan Uploaded Yet</h4>
              <p className="text-xs text-slate-500 max-w-sm mx-auto leading-relaxed">
                Associate an X-ray coronary angiography scan (DICOM or standard image format) to this case to enable preprocessing.
              </p>
              <Button
                variant="primary"
                size="sm"
                icon={UploadCloud}
                onClick={() => navigate(`/upload?caseId=${caseData.id}`)}
              >
                Upload Scan for {caseData.id}
              </Button>
            </div>
          )}
        </div>

        {/* Right Column: AI Segmentation Output Placeholder (Requirement F) */}
        <div className="bg-white rounded-xl border border-slate-200 shadow-2xs p-5 space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100">
            <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
              <Bot className="w-4 h-4 text-blue-600" />
              <span>AI Artery Segmentation &amp; Stenosis Analysis</span>
            </h3>
            <span className="text-[11px] font-medium bg-amber-50 text-amber-800 border border-amber-200 px-2 py-0.5 rounded-full">
              Model Pending
            </span>
          </div>

          {/* Explicit Placeholder Notice (Mandated by Section 4.F and Section 3) */}
          <div className="rounded-xl border border-dashed border-amber-300 bg-amber-50/50 p-6 text-center space-y-3">
            <div className="w-12 h-12 rounded-full bg-amber-100 text-amber-700 flex items-center justify-center mx-auto border border-amber-200">
              <Sparkles className="w-6 h-6" />
            </div>

            <div className="space-y-1">
              <h4 className="text-sm font-bold text-slate-900">
                AI Segmentation Not Yet Connected
              </h4>
              <p className="text-xs text-slate-600 max-w-md mx-auto leading-relaxed">
                Deep learning inference routines (coronary artery boundary segmentation and luminal stenosis-region localization) will be connected when the deep learning backend is integrated in Milestone 2.
              </p>
            </div>

            <div className="bg-white/80 rounded-lg p-3 border border-amber-200 text-left text-xs text-slate-600 max-w-md mx-auto space-y-1.5">
              <p className="font-semibold text-slate-800 text-[11px] uppercase tracking-wider flex items-center gap-1.5">
                <ShieldAlert className="w-3.5 h-3.5 text-amber-600" />
                <span>Planned Milestone 2 Output Artifacts</span>
              </p>
              <ul className="list-disc pl-4 space-y-1 text-[11px] text-slate-500">
                <li>Binary vessel segmentation mask overlay (ARCADE ground-truth comparison)</li>
                <li>Stenosis region bounding box delineations</li>
                <li>Lumen diameter profile across vessel centerline</li>
              </ul>
            </div>

            <p className="text-[11px] text-slate-400 italic">
              Note: No automated clinical diagnosis or unverified percentage calculation is displayed in this milestone.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
