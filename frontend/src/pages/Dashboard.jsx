import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  FolderPlus,
  UploadCloud,
  Layers,
  Clock3,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Eye,
  FileImage,
  Sparkles,
  Activity,
  ShieldCheck,
  Microscope,
  Database,
  ScanLine,
  Brain,
  FlaskConical,
} from 'lucide-react';

import Button from '../components/Button';
import StatusBadge from '../components/StatusBadge';
import EmptyState from '../components/EmptyState';
import LoadingState from '../components/LoadingState';
import { useCases } from '../context/CaseContext';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';
import { formatDateShort } from '../utils/formatters';
import { PROJECT_TITLE, DATASET_NAME } from '../utils/constants';

const panelBase = 'rounded-2xl border shadow-sm transition-colors duration-200';
const mutedText = 'text-sm leading-6';
const headingText = 'font-semibold tracking-tight';

function MetricCard({ label, value, note, icon: Icon, tone, dark }) {
  const tones = {
    blue: dark
      ? 'border-sky-400/20 bg-sky-400/10 text-sky-300'
      : 'border-sky-200 bg-sky-50 text-sky-700',
    amber: dark
      ? 'border-amber-400/20 bg-amber-400/10 text-amber-300'
      : 'border-amber-200 bg-amber-50 text-amber-700',
    green: dark
      ? 'border-emerald-400/20 bg-emerald-400/10 text-emerald-300'
      : 'border-emerald-200 bg-emerald-50 text-emerald-700',
    rose: dark
      ? 'border-rose-400/20 bg-rose-400/10 text-rose-300'
      : 'border-rose-200 bg-rose-50 text-rose-700',
  };

  return (
    <article className={`${panelBase} ${dark ? 'border-slate-700 bg-slate-900/80 hover:border-slate-600' : 'border-slate-200 bg-white hover:border-slate-300'} p-5 sm:p-6`}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className={`text-sm font-medium ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
            {label}
          </p>
          <p className={`mt-3 text-3xl font-bold tracking-tight tabular-nums sm:text-4xl ${dark ? 'text-white' : 'text-slate-950'}`}>
            {value ?? 0}
          </p>
        </div>
        <div className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border ${tones[tone] || tones.blue}`}>
          <Icon className="h-5 w-5" aria-hidden="true" />
        </div>
      </div>
      <p className={`mt-3 ${mutedText} ${dark ? 'text-slate-400' : 'text-slate-600'}`}>
        {note}
      </p>
    </article>
  );
}

function InfoRow({ icon: Icon, title, description, tone = 'green', dark }) {
  const dotTones = {
    green: 'bg-emerald-500',
    amber: 'bg-amber-500',
    blue: 'bg-sky-500',
  };

  return (
    <div className="flex items-start gap-3">
      <span className={`mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full ${dotTones[tone]}`} aria-hidden="true" />
      <div className="min-w-0">
        <p className={`text-sm font-semibold ${dark ? 'text-slate-100' : 'text-slate-900'}`}>{title}</p>
        <p className={`mt-1 text-sm leading-5 ${dark ? 'text-slate-400' : 'text-slate-600'}`}>{description}</p>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { cases = [], stats = {}, isLoading } = useCases();
  const { theme } = useTheme();
  const dark = theme === 'dark';

  const recentCases = cases.slice(0, 5);
  const fullName = typeof user?.name === 'string' ? user.name.trim() : '';
  const displayName = fullName || 'Researcher';

  const pageText = dark ? 'text-slate-100' : 'text-slate-900';
  const secondaryText = dark ? 'text-slate-400' : 'text-slate-600';
  const panel = dark ? 'border-slate-700 bg-slate-900/80' : 'border-slate-200 bg-white';
  const divider = dark ? 'border-slate-700' : 'border-slate-200';
  const tableHeader = dark ? 'bg-slate-800/80 text-slate-300' : 'bg-slate-50 text-slate-600';
  const tableRowHover = dark ? 'hover:bg-sky-400/5' : 'hover:bg-sky-50/70';
  const linkColor = dark ? 'text-teal-300 hover:text-teal-200' : 'text-teal-700 hover:text-teal-900';

  return (
    <main className={`mx-auto w-full max-w-[1600px] space-y-7 pb-8 sm:space-y-8 ${pageText}`}>
      {/* Welcome banner */}
      <section className={`${panelBase} ${panel} overflow-hidden`}>
        <div className="grid gap-8 p-6 sm:p-8 lg:grid-cols-[minmax(0,1fr)_220px] lg:items-center">
          <div className="min-w-0">
            <div className={`mb-5 inline-flex items-center gap-2 rounded-full border px-3 py-2 text-sm font-semibold ${
              dark
                ? 'border-teal-400/30 bg-teal-400/10 text-teal-200'
                : 'border-teal-200 bg-teal-50 text-teal-800'
            }`}>
              <Sparkles className="h-4 w-4 shrink-0" aria-hidden="true" />
              Clinical research workspace
            </div>

            <p className={`text-sm font-semibold uppercase tracking-[0.12em] ${dark ? 'text-slate-400' : 'text-slate-500'}`}>
              Your research overview
            </p>

            <h1 className={`mt-2 text-3xl font-bold leading-tight tracking-tight sm:text-4xl ${dark ? 'text-white' : 'text-slate-950'}`}>
              Welcome back, {displayName}
            </h1>

            <p className={`mt-4 max-w-3xl text-base leading-7 ${secondaryText}`}>
              Organize coronary angiography research cases, review scan workflows,
              and monitor segmentation model development in one workspace.
            </p>

            <div className="mt-7 flex flex-wrap gap-3">
              <Button
                variant="primary"
                size="md"
                icon={UploadCloud}
                onClick={() => navigate('/upload')}
              >
                Upload a scan
              </Button>
              <Button
                variant="outline"
                size="md"
                icon={FolderPlus}
                onClick={() => navigate('/cases/create')}
              >
                Create a case
              </Button>
            </div>
          </div>

          <div className="hidden items-center justify-center lg:flex">
            <div className={`flex h-44 w-44 items-center justify-center rounded-full border ${
              dark ? 'border-teal-400/30 bg-teal-400/10' : 'border-teal-200 bg-teal-50'
            }`}>
              <div className={`flex h-32 w-32 items-center justify-center rounded-full border shadow-sm ${
                dark ? 'border-teal-400/30 bg-slate-950 text-teal-300' : 'border-teal-200 bg-white text-teal-700'
              }`}>
                <Activity className="h-16 w-16" strokeWidth={1.4} aria-hidden="true" />
              </div>
            </div>
          </div>
        </div>

        <div className={`flex flex-col gap-3 border-t px-6 py-4 text-sm sm:flex-row sm:items-center sm:justify-between sm:px-8 ${divider} ${dark ? 'bg-slate-800/50' : 'bg-slate-50'}`}>
          <span className={`flex items-center gap-2 ${secondaryText}`}>
            <Database className={`h-5 w-5 shrink-0 ${dark ? 'text-teal-300' : 'text-teal-700'}`} aria-hidden="true" />
            {DATASET_NAME}
          </span>
          <span className={`flex items-center gap-2 ${dark ? 'text-amber-300' : 'text-amber-700'}`}>
            <ShieldCheck className="h-5 w-5 shrink-0" aria-hidden="true" />
            Research prototype · Model integration in progress
          </span>
        </div>
      </section>

      {/* Case statistics */}
      <section aria-labelledby="metrics-heading">
        <div className="mb-5">
          <h2 id="metrics-heading" className={`text-xl font-bold sm:text-2xl ${dark ? 'text-white' : 'text-slate-950'}`}>
            Case activity
          </h2>
          <p className={`mt-2 text-sm leading-6 sm:text-base ${secondaryText}`}>
            An overview of the cases registered in this workspace.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard label="Total cases" value={stats.total} note="Cases registered in this workspace" icon={Layers} tone="blue" dark={dark} />
          <MetricCard label="Awaiting processing" value={stats.awaitingProcessing} note="Cases waiting for the next workflow step" icon={Clock3} tone="amber" dark={dark} />
          <MetricCard label="Completed processing" value={stats.completed} note="Cases marked complete in the current workflow" icon={CheckCircle2} tone="green" dark={dark} />
          <MetricCard label="Failed jobs" value={stats.failed} note="Cases marked as failed and needing review" icon={AlertTriangle} tone="rose" dark={dark} />
        </div>
      </section>

      {/* Recent cases and project information */}
      <section className="grid min-w-0 gap-5 xl:grid-cols-[minmax(0,1fr)_320px]">
        <div className={`${panelBase} ${panel} min-w-0 overflow-hidden`}>
          <div className={`flex flex-col gap-3 border-b px-5 py-5 sm:flex-row sm:items-center sm:justify-between sm:px-6 ${divider}`}>
            <div>
              <h2 className={`text-xl font-bold ${dark ? 'text-white' : 'text-slate-950'}`}>Recent research cases</h2>
              <p className={`mt-2 text-sm leading-6 ${secondaryText}`}>The latest cases registered in your workspace.</p>
            </div>
            <Link to="/cases" className={`inline-flex w-fit items-center gap-2 rounded-md text-sm font-semibold focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 ${linkColor}`}>
              View all cases <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </Link>
          </div>

          {isLoading ? (
            <LoadingState message="Loading research cases..." />
          ) : recentCases.length === 0 ? (
            <div className="p-6">
              <EmptyState
                title="No research cases yet"
                description="Create a case to start organizing your angiography research workflow."
                actionLabel="Create your first case"
                actionIcon={FolderPlus}
                onAction={() => navigate('/cases/create')}
              />
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[700px] text-left text-sm">
                <thead className={`${tableHeader} text-xs font-bold uppercase tracking-wider`}>
                  <tr>
                    <th scope="col" className="px-5 py-4">Case</th>
                    <th scope="col" className="px-4 py-4">Image type</th>
                    <th scope="col" className="px-4 py-4">Date added</th>
                    <th scope="col" className="px-4 py-4">Status</th>
                    <th scope="col" className="px-5 py-4 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className={`divide-y ${dark ? 'divide-slate-700' : 'divide-slate-200'}`}>
                  {recentCases.map((item) => (
                    <tr key={item.id} className={`transition-colors ${tableRowHover}`}>
                      <td className="px-5 py-4">
                        <p className={`text-sm font-semibold leading-6 ${dark ? 'text-slate-100' : 'text-slate-900'}`}>{item.name}</p>
                        <p className={`mt-1.5 text-xs leading-5 ${secondaryText}`}>
                          <span className={`font-mono font-medium ${dark ? 'text-teal-300' : 'text-teal-700'}`}>{item.id}</span>
                          {' · '}
                          {item.targetRegion}
                        </p>
                      </td>
                      <td className={`px-4 py-4 text-sm ${dark ? 'text-slate-300' : 'text-slate-700'}`}>
                        <span className="inline-flex items-center gap-2">
                          <FileImage className={`h-4 w-4 shrink-0 ${secondaryText}`} aria-hidden="true" />
                          {item.imageType}
                        </span>
                      </td>
                      <td className={`whitespace-nowrap px-4 py-4 text-sm ${secondaryText}`}>
                        {formatDateShort(item.createdAt)}
                      </td>
                      <td className="px-4 py-4"><StatusBadge status={item.status} /></td>
                      <td className="px-5 py-4 text-right">
                        <Button variant="outline" size="sm" icon={Eye} onClick={() => navigate(`/cases/${item.id}`)}>
                          Details
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <aside className="space-y-5">
          <section className={`${panelBase} ${panel} p-5 sm:p-6`}>
            <div className={`flex h-11 w-11 items-center justify-center rounded-xl ${
              dark ? 'bg-teal-400/10 text-teal-300' : 'bg-teal-50 text-teal-700'
            }`}>
              <Microscope className="h-6 w-6" aria-hidden="true" />
            </div>
            <h2 className={`mt-4 text-lg font-bold ${dark ? 'text-white' : 'text-slate-950'}`}>Project status</h2>
            <p className={`mt-2 ${mutedText} ${secondaryText}`}>
              Research workspace and segmentation development progress.
            </p>
            <div className="mt-5 space-y-4">
              <InfoRow title="Case workflow" description="Frontend prototype available" tone="green" dark={dark} />
              <InfoRow title="Baseline U-Net" description="Initial training experiment completed" tone="green" dark={dark} />
              <InfoRow title="Attention U-Net" description="Implementation pending" tone="amber" dark={dark} />
              <InfoRow title="Model comparison" description="Pending evaluation of both models" tone="amber" dark={dark} />
            </div>
          </section>

          <section className={`${panelBase} p-5 sm:p-6 ${
            dark ? 'border-sky-400/20 bg-sky-400/5' : 'border-sky-200 bg-sky-50'
          }`}>
            <p className={`text-xs font-bold uppercase tracking-[0.12em] ${dark ? 'text-sky-300' : 'text-sky-800'}`}>
              Research focus
            </p>
            <h2 className={`mt-3 text-base font-bold leading-6 ${dark ? 'text-slate-100' : 'text-slate-900'}`}>
              {PROJECT_TITLE}
            </h2>
            <p className={`mt-3 text-sm leading-6 ${dark ? 'text-slate-300' : 'text-slate-700'}`}>
              This workspace supports research case organization and model development.
              Segmentation outputs must be evaluated before being treated as validated results.
              This prototype is not a clinical diagnostic tool.
            </p>
            <div className={`mt-4 flex items-start gap-2 text-sm font-semibold leading-5 ${dark ? 'text-sky-300' : 'text-sky-800'}`}>
              <ScanLine className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
              <span>Research evaluation in progress</span>
            </div>
          </section>

          <section className={`${panelBase} ${panel} p-5`}>
            <div className="flex items-center gap-3">
              <div className={`flex h-10 w-10 items-center justify-center rounded-xl ${
                dark ? 'bg-violet-400/10 text-violet-300' : 'bg-violet-50 text-violet-700'
              }`}>
                <Brain className="h-5 w-5" aria-hidden="true" />
              </div>
              <div>
                <h2 className={`text-sm font-bold ${dark ? 'text-white' : 'text-slate-900'}`}>Model development</h2>
                <p className={`mt-1 text-xs ${secondaryText}`}>Current research stage</p>
              </div>
            </div>
            <p className={`mt-3 text-sm leading-6 ${secondaryText}`}>
              Baseline U-Net has completed an initial experiment. More training,
              visual inspection and validation are required before final reporting.
            </p>
          </section>
        </aside>
      </section>
    </main>
  );
}
