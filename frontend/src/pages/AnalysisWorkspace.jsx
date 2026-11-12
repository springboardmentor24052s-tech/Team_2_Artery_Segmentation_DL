import React, { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  Activity, ArrowLeft, BrainCircuit, CheckCircle2, CircleHelp,
  FileImage, Layers3, Microscope, ShieldAlert, Sparkles, Target, ZoomIn, ZoomOut, RotateCcw, Maximize2,
} from 'lucide-react';
import { useCases } from '../context/CaseContext';
import LoadingState from '../components/LoadingState';
import StatusBadge from '../components/StatusBadge';
import { getScanAsset } from '../services/scanAssetService';

function StatusPill({ children, tone = 'slate' }) {
  const tones = {
    slate: 'bg-slate-100 text-slate-700 border-slate-200',
    amber: 'bg-amber-50 text-amber-800 border-amber-200',
    blue: 'bg-blue-50 text-blue-800 border-blue-200',
    green: 'bg-emerald-50 text-emerald-800 border-emerald-200',
  };
  return <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-semibold ${tones[tone]}`}>{children}</span>;
}

function OutputPanel({ icon: Icon, title, description, detail }) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-50 text-blue-700"><Icon className="h-5 w-5" /></span>
          <div>
            <h3 className="text-sm font-bold text-slate-900">{title}</h3>
            <p className="mt-0.5 text-xs text-slate-500">{description}</p>
          </div>
        </div>
        <StatusPill tone="amber">Awaiting model output</StatusPill>
      </div>
      <div className="mt-4 rounded-lg border border-dashed border-slate-300 bg-slate-50 px-4 py-7 text-center">
        <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-full bg-white text-slate-400 shadow-sm"><CircleHelp className="h-5 w-5" /></div>
        <p className="mt-3 text-sm font-semibold text-slate-700">No verified result available</p>
        <p className="mx-auto mt-1 max-w-md text-xs leading-5 text-slate-500">{detail}</p>
      </div>
    </section>
  );
}

export default function AnalysisWorkspace() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { cases, isLoading } = useCases();
  const caseData = cases.find((item) => item.id === id);
  const [scanUrl, setScanUrl] = useState('');
  const [scanLoadMessage, setScanLoadMessage] = useState('');
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState(0);
  const isPreviewableImage = useMemo(() => Boolean(caseData?.file?.type?.startsWith('image/')), [caseData?.file?.type]);

  useEffect(() => {
    let objectUrl = '';
    let cancelled = false;
    setScanUrl('');
    setScanLoadMessage('');
    setZoom(1);
    setRotation(0);
    if (!caseData?.file) return undefined;
    getScanAsset(id).then((asset) => {
      if (cancelled) return;
      if (asset?.blob && asset.type?.startsWith('image/')) {
        objectUrl = URL.createObjectURL(asset.blob);
        setScanUrl(objectUrl);
      } else if (asset?.blob) {
        setScanLoadMessage('This file is stored locally, but browser preview is not available for this format. DICOM decoding is not enabled in this frontend yet.');
      } else {
        setScanLoadMessage('Only file metadata is available for this case. Re-upload the image from this browser session to enable its local preview.');
      }
    }).catch(() => setScanLoadMessage('The local preview could not be loaded. The uploaded file metadata is still available.'));
    return () => { cancelled = true; if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [id, caseData?.file?.name, caseData?.file?.uploadedAt]);

  if (isLoading) return <LoadingState message="Loading analysis workspace..." />;

  if (!caseData) {
    return (
      <div className="mx-auto max-w-xl rounded-xl border border-rose-200 bg-white p-8 text-center">
        <ShieldAlert className="mx-auto h-8 w-8 text-rose-600" />
        <h2 className="mt-3 text-lg font-bold text-slate-900">Case not found</h2>
        <p className="mt-1 text-sm text-slate-500">We could not find case {id} in the current case list.</p>
        <button onClick={() => navigate('/cases')} className="mt-4 rounded-lg bg-blue-700 px-4 py-2 text-sm font-semibold text-white">Back to cases</button>
      </div>
    );
  }

  const hasFile = Boolean(caseData.file);

  return (
    <div className="space-y-6">
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
        <div>
          <Link to={`/cases/${caseData.id}`} className="mb-3 inline-flex items-center gap-1.5 text-xs font-semibold text-slate-500 hover:text-blue-700"><ArrowLeft className="h-3.5 w-3.5" /> Back to case details</Link>
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-md border border-blue-200 bg-blue-50 px-2.5 py-1 font-mono text-xs font-bold text-blue-800">{caseData.id}</span>
            <StatusBadge status={caseData.status} />
            <StatusPill tone="amber">Research preview · not a diagnosis</StatusPill>
          </div>
          <h1 className="mt-2 text-2xl font-bold tracking-tight text-slate-900">AI Analysis Workspace</h1>
          <p className="mt-1 text-sm text-slate-500">{caseData.name} · {caseData.targetRegion || 'Full coronary tree'}</p>
        </div>
        {!hasFile && <button onClick={() => navigate(`/upload?caseId=${caseData.id}`)} className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-700 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-800">Upload angiogram</button>}
      </div>

      <div className="rounded-xl border border-blue-200 bg-blue-50/70 p-4">
        <div className="flex items-start gap-3">
          <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-blue-700" />
          <div>
            <p className="text-sm font-bold text-blue-950">Model integration status</p>
            <p className="mt-1 text-xs leading-5 text-blue-900">This workspace is connected to the current case record, but verified segmentation, stenosis detection, and explainability outputs are not yet returned by the frontend service. No clinical measurements or predictions are being generated on this page.</p>
          </div>
        </div>
      </div>

      <section className="grid gap-4 lg:grid-cols-[1.15fr_0.85fr]">
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2"><FileImage className="h-4 w-4 text-blue-700" /><h2 className="text-sm font-bold text-slate-900">Input angiogram</h2></div>
            <StatusPill tone={hasFile ? 'green' : 'amber'}>{hasFile ? 'File attached' : 'No file attached'}</StatusPill>
          </div>
          <div className="mt-4 overflow-hidden rounded-lg border border-slate-300 bg-slate-950">
            {scanUrl ? (
              <>
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-700 bg-slate-900 px-3 py-2">
                  <p className="text-xs font-semibold text-slate-200">Local image preview <span className="ml-1 font-normal text-slate-400">· original upload</span></p>
                  <div className="flex items-center gap-1">
                    <button type="button" title="Zoom out" onClick={() => setZoom((value) => Math.max(0.5, Number((value - 0.1).toFixed(1))))} className="rounded-md p-2 text-slate-200 hover:bg-slate-700"><ZoomOut className="h-4 w-4" /></button>
                    <span className="min-w-12 text-center text-xs tabular-nums text-slate-300">{Math.round(zoom * 100)}%</span>
                    <button type="button" title="Zoom in" onClick={() => setZoom((value) => Math.min(2.5, Number((value + 0.1).toFixed(1))))} className="rounded-md p-2 text-slate-200 hover:bg-slate-700"><ZoomIn className="h-4 w-4" /></button>
                    <button type="button" title="Rotate image" onClick={() => setRotation((value) => (value + 90) % 360)} className="rounded-md p-2 text-slate-200 hover:bg-slate-700"><RotateCcw className="h-4 w-4" /></button>
                    <button type="button" title="Reset view" onClick={() => { setZoom(1); setRotation(0); }} className="rounded-md p-2 text-slate-200 hover:bg-slate-700"><Maximize2 className="h-4 w-4" /></button>
                  </div>
                </div>
                <div className="flex min-h-72 max-h-[560px] items-center justify-center overflow-auto bg-black p-4">
                  <img src={scanUrl} alt={`Original uploaded angiography image: ${caseData.file.name}`} className="max-h-[500px] max-w-full object-contain transition-transform duration-150" style={{ transform: `scale(${zoom}) rotate(${rotation}deg)` }} />
                </div>
                <div className="border-t border-slate-700 bg-slate-900 px-3 py-2 text-[11px] text-slate-300">Original uploaded file · local browser preview only · no segmentation overlay or AI result is implied</div>
              </>
            ) : (
              <div className="flex min-h-64 flex-col items-center justify-center px-5 py-8 text-center">
                <FileImage className="h-9 w-9 text-slate-400" />
                <p className="mt-3 text-sm font-semibold text-white">{hasFile ? caseData.file.name : 'Angiogram not uploaded'}</p>
                <p className="mt-1 max-w-md text-xs leading-5 text-slate-400">{hasFile ? (scanLoadMessage || (isPreviewableImage ? 'Loading your local image preview…' : 'This file format needs a dedicated medical-image viewer.')) : 'Upload a supported X-ray angiography image to continue.'}</p>
                {!hasFile && <button onClick={() => navigate(`/upload?caseId=${caseData.id}`)} className="mt-4 rounded-lg bg-blue-700 px-4 py-2 text-xs font-semibold text-white hover:bg-blue-800">Upload angiogram</button>}
              </div>
            )}
          </div>
          {hasFile && <div className="mt-3 grid grid-cols-2 gap-3 text-xs sm:grid-cols-3">
            <div className="rounded-lg bg-slate-50 p-3"><p className="text-slate-500">File name</p><p className="mt-1 truncate font-semibold text-slate-800" title={caseData.file.name}>{caseData.file.name}</p></div>
            <div className="rounded-lg bg-slate-50 p-3"><p className="text-slate-500">File type</p><p className="mt-1 font-semibold text-slate-800">{caseData.file.type || 'Not supplied'}</p></div>
            <div className="rounded-lg bg-slate-50 p-3"><p className="text-slate-500">Dimensions</p><p className="mt-1 font-semibold text-slate-800">{caseData.file.dimensions || 'Not supplied'}</p></div>
          </div>}
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2"><Layers3 className="h-4 w-4 text-blue-700" /><h2 className="text-sm font-bold text-slate-900">Analysis pipeline</h2></div>
          <p className="mt-1 text-xs text-slate-500">Expected outputs for this research workflow</p>
          <ol className="mt-5 space-y-4">
            {[
              ['01', 'Image preprocessing', 'Contrast enhancement and normalization', Boolean(caseData.preprocessingStage === 'PREPROCESSING_COMPLETED')],
              ['02', 'Vessel segmentation', 'U-Net / Attention U-Net mask generation', false],
              ['03', 'Stenosis localization', 'Candidate region detection and localization', false],
              ['04', 'Explainability', 'Model-supported visual explanation', false],
            ].map(([number, title, subtitle, complete]) => <li key={number} className="flex gap-3">
              <span className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-bold ${complete ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-500'}`}>{complete ? <CheckCircle2 className="h-4 w-4" /> : number}</span>
              <div className="min-w-0 flex-1 border-b border-slate-100 pb-3"><div className="flex flex-wrap items-center justify-between gap-2"><p className="text-sm font-semibold text-slate-800">{title}</p><StatusPill tone={complete ? 'green' : 'slate'}>{complete ? 'Case marked ready' : 'Not connected'}</StatusPill></div><p className="mt-1 text-xs text-slate-500">{subtitle}</p></div>
            </li>)}
          </ol>
          <button onClick={() => navigate('/preprocessing')} className="mt-4 inline-flex items-center gap-2 text-xs font-semibold text-blue-700 hover:text-blue-900">View preprocessing monitor <Activity className="h-3.5 w-3.5" /></button>
        </div>
      </section>

      <div className="grid gap-4 xl:grid-cols-2">
        <OutputPanel icon={Microscope} title="Artery segmentation" description="Baseline U-Net and Attention U-Net" detail="The model output images and evaluation metrics are not yet exposed through the current frontend API. Once connected, this panel will display the original image, predicted mask, overlay, and model comparison." />
        <OutputPanel icon={Target} title="Stenosis detection & localization" description="Candidate region and vessel-level analysis" detail="No verified detection boxes, vessel measurements, or stenosis percentages are available from the current case service. These values will only be shown when supplied by the trained inference pipeline." />
        <OutputPanel icon={BrainCircuit} title="Explainable AI" description="Visual explanation of model-supported outputs" detail="A heatmap or attribution overlay will be displayed only after an explanation method compatible with the connected model produces it. No illustrative heatmap is presented as a real explanation." />
        <OutputPanel icon={Sparkles} title="Research interpretation" description="Model metadata, limitations, and review notes" detail="This area will summarize the model version, inference settings, confidence or evaluation metrics where available, and known limitations. It will not generate an automated clinical diagnosis." />
      </div>

      <div className="flex items-start gap-2 rounded-lg border border-slate-200 bg-slate-50 p-3 text-[11px] leading-5 text-slate-600"><ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-slate-500" /><p><strong>Research-use notice:</strong> ArteryAI is a research prototype. Outputs require validation and qualified review and must not be used as a standalone clinical decision or diagnosis.</p></div>
    </div>
  );
}
