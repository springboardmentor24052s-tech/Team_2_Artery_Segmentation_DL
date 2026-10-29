import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Activity, UploadCloud, FolderPlus, ScanLine, Clock3, CheckCircle2,
  AlertTriangle, ChevronRight, Play, Pause, ZoomIn, ZoomOut, Maximize2,
  Settings2, Bell, UserRound, Waves, CircleHelp
} from 'lucide-react';
import { useCases } from '../context/CaseContext';
import { useAuth } from '../context/AuthContext';
import heroImage from '../assets/hero.png';
import StatusBadge from '../components/StatusBadge';

const panel = 'rounded-xl border border-slate-700/80 bg-[#151f2d] shadow-lg shadow-black/10';
const muted = 'text-slate-400';

export default function Dashboard() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { cases = [], stats = {} } = useCases();
  const recent = cases.slice(0, 6);
  const selected = recent[0];

  return (
    <div className="min-h-[calc(100vh-9rem)] -m-4 sm:-m-6 lg:-m-8 p-4 sm:p-5 bg-[#0b1220] text-slate-100">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-sky-400/30 bg-sky-400/10 text-sky-300"><Activity size={23}/></div>
          <div>
            <h1 className="text-lg font-semibold tracking-tight">ArteryInsight AI <span className="text-slate-300">· Coronary Artery Analysis</span></h1>
            <p className={`text-xs ${muted}`}>Research workspace · Coronary angiography viewer</p>
          </div>
          <span className="hidden rounded-full border border-sky-400/30 bg-sky-400/10 px-3 py-1 text-xs text-sky-200 md:inline-flex">AI model · U-Net research pipeline</span>
        </div>
        <div className="flex items-center gap-2 text-slate-300">
          <span className="hidden text-xs sm:inline">Researcher: {user?.name || 'Researcher'}</span>
          <button title="Notifications" className="rounded-lg border border-slate-700 p-2 hover:bg-slate-800"><Bell size={16}/></button>
          <button title="Settings" className="rounded-lg border border-slate-700 p-2 hover:bg-slate-800"><Settings2 size={16}/></button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[245px_minmax(360px,1fr)_270px]">
        <section className={`${panel} overflow-hidden`}>
          <div className="flex items-center justify-between border-b border-slate-700 px-4 py-3">
            <div><h2 className="text-sm font-semibold">Patient Studies</h2><p className={`mt-0.5 text-[11px] ${muted}`}>{cases.length} research cases</p></div>
            <button onClick={() => navigate('/cases/create')} title="Create case" className="rounded-lg bg-sky-400/10 p-2 text-sky-300 hover:bg-sky-400/20"><FolderPlus size={16}/></button>
          </div>
          <div className="space-y-2 p-2">
            {recent.length ? recent.map((c, i) => (
              <button key={c.id} onClick={() => navigate(`/cases/${c.id}`)} className={`w-full rounded-lg border p-3 text-left transition ${i === 0 ? 'border-sky-300/70 bg-sky-400/10' : 'border-slate-700 bg-slate-900/50 hover:border-slate-500'}`}>
                <div className="flex gap-3">
                  <div className="flex h-14 w-14 shrink-0 items-center justify-center overflow-hidden rounded-md border border-slate-600 bg-black">
                    <img src={heroImage} alt="" className="h-full w-full object-cover opacity-80 grayscale"/>
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-xs font-semibold">{c.name || `Case ${c.id}`}</p>
                    <p className={`mt-1 text-[10px] ${muted}`}>ID: {c.id}</p>
                    <p className={`mt-1 text-[10px] ${muted}`}>{c.targetRegion || c.imageType || 'Coronary angiography'}</p>
                    <div className="mt-1"><StatusBadge status={c.status}/></div>
                  </div>
                </div>
              </button>
            )) : <div className="p-5 text-center"><ScanLine className="mx-auto mb-2 text-slate-500"/><p className="text-sm">No studies yet</p><p className={`mt-1 text-xs ${muted}`}>Create a case to begin.</p></div>}
          </div>
          <div className="border-t border-slate-700 p-3">
            <button onClick={() => navigate('/cases')} className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-xs font-medium text-sky-300 hover:bg-slate-800">View all studies <ChevronRight size={15}/></button>
            <button onClick={() => navigate('/upload')} className="mt-1 flex w-full items-center justify-center gap-2 rounded-lg bg-sky-400 px-3 py-2.5 text-xs font-bold text-slate-950 hover:bg-sky-300"><UploadCloud size={15}/> Upload angiogram</button>
          </div>
        </section>

        <section className={`${panel} min-w-0 overflow-hidden`}>
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-700 px-4 py-3">
            <div><h2 className="text-sm font-semibold">Coronary Angiogram Viewer</h2><p className={`mt-0.5 text-[11px] ${muted}`}>{selected ? `Selected study: ${selected.id}` : 'Illustrative preview'}</p></div>
            <div className="flex gap-1">
              {[ZoomIn, ZoomOut, Maximize2].map((Icon, i) => <button key={i} title={['Zoom in','Zoom out','Fit view'][i]} className="rounded-md border border-slate-700 p-2 text-slate-300 hover:bg-slate-800"><Icon size={15}/></button>)}
            </div>
          </div>
          <div className="relative flex min-h-[300px] items-center justify-center overflow-hidden bg-black p-4 sm:min-h-[390px]">
            <img src={heroImage} alt="Illustrative angiography research preview" className="max-h-[390px] w-full object-contain opacity-80 grayscale"/>
            <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-slate-950/60 via-transparent to-slate-950/10"/>
            <div className="absolute left-[28%] top-[32%] h-20 w-28 rounded border-2 border-pink-400/80 bg-pink-400/10"/>
            <div className="absolute left-[31%] top-[25%] rounded-md border border-pink-300/60 bg-slate-950/90 px-3 py-2 text-xs text-pink-200">
              <span className="block font-semibold">Segmentation preview</span><span className="text-[10px] text-slate-400">Illustrative overlay only</span>
            </div>
            <div className="absolute bottom-3 left-3 rounded-md border border-slate-700 bg-slate-950/80 px-2.5 py-1.5 text-[10px] text-slate-300">Demo image · Not a patient result</div>
          </div>
          <div className="border-t border-slate-700 bg-[#101927] px-4 py-3">
            <div className="mb-2 flex items-center justify-between text-[11px] text-slate-400"><span>Frame preview</span><span>Static image · no cine sequence loaded</span></div>
            <div className="h-1.5 rounded-full bg-slate-700"><div className="h-1.5 w-[38%] rounded-full bg-sky-400"/></div>
            <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
              <button onClick={() => navigate('/upload')} className="inline-flex items-center gap-2 rounded-lg bg-sky-400 px-3 py-2 text-xs font-semibold text-slate-950 hover:bg-sky-300"><UploadCloud size={14}/> Analyze new image</button>
              <button onClick={() => navigate('/preprocessing')} className="inline-flex items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-xs text-slate-300 hover:bg-slate-800"><Clock3 size={14}/> Processing status</button>
            </div>
          </div>
        </section>

        <aside className="space-y-4">
          <section className={`${panel} p-4`}>
            <div className="flex items-center justify-between"><h2 className="text-sm font-semibold">AI Analytics & Insights</h2><Waves size={17} className="text-sky-300"/></div>
            <p className={`mt-1 text-[11px] ${muted}`}>Research pipeline status</p>
            <div className="mt-4 rounded-lg border border-pink-300/20 bg-pink-300/5 p-3">
              <div className="flex items-center justify-between"><span className="text-xs text-slate-300">Vessel segmentation</span><span className="rounded-full bg-pink-300/10 px-2 py-1 text-[10px] text-pink-200">Model setup needed</span></div>
              <p className="mt-2 text-xs text-slate-400">A trained model is required before masks can be generated.</p>
            </div>
            <div className="mt-3 rounded-lg border border-sky-300/20 bg-sky-300/5 p-3">
              <div className="flex items-center justify-between"><span className="text-xs text-slate-300">Stenosis detection</span><span className="rounded-full bg-slate-700 px-2 py-1 text-[10px] text-slate-300">Not validated</span></div>
              <p className="mt-2 text-xs text-slate-400">No stenosis percentage is estimated from this preview.</p>
            </div>
          </section>

          <section className={`${panel} p-4`}>
            <h2 className="text-sm font-semibold">Vascular Research Metrics</h2>
            <div className="mt-3 space-y-3">
              {[['Total studies', stats.total ?? cases.length, 'text-sky-300'],['Awaiting processing', stats.awaitingProcessing ?? '—','text-pink-300'],['Completed workflows', stats.completed ?? '—','text-emerald-300'],['Failed workflows', stats.failed ?? '—','text-slate-300']].map(([label,value,color]) => <div key={label} className="flex items-center justify-between border-b border-slate-700/70 pb-2 last:border-0 last:pb-0"><span className="text-xs text-slate-400">{label}</span><strong className={`text-sm ${color}`}>{value}</strong></div>)}
            </div>
          </section>

          <section className={`${panel} p-4`}>
            <h2 className="text-sm font-semibold">Research Safety</h2>
            <div className="mt-3 flex gap-2 rounded-lg border border-slate-700 bg-slate-900/60 p-3">
              <CircleHelp size={17} className="shrink-0 text-sky-300"/>
              <p className="text-[11px] leading-relaxed text-slate-400">This is a research prototype. Displayed image overlays are illustrative and are not diagnostic results. Validate models against expert annotations.</p>
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}
