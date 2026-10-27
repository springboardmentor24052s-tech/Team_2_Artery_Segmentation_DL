
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
} from 'lucide-react';

import Button from '../components/Button';
import StatusBadge from '../components/StatusBadge';
import EmptyState from '../components/EmptyState';
import LoadingState from '../components/LoadingState';
import { useCases } from '../context/CaseContext';
import { useAuth } from '../context/AuthContext';
import { formatDateShort } from '../utils/formatters';
import { PROJECT_TITLE, DATASET_NAME } from '../utils/constants';

function MetricCard({ label, value, note, icon: Icon, tone }) {
  const styles = {
    blue: 'bg-blue-50 text-blue-700 border-blue-100',
    amber: 'bg-amber-50 text-amber-700 border-amber-100',
    green: 'bg-emerald-50 text-emerald-700 border-emerald-100',
    rose: 'bg-rose-50 text-rose-700 border-rose-100',
  };

  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-5 sm:p-6 shadow-sm transition-shadow hover:shadow-md">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm sm:text-base font-medium text-slate-600">
            {label}
          </p>
          <p className="mt-3 text-3xl sm:text-4xl font-bold tracking-tight text-slate-900 tabular-nums">
            {value}
          </p>
        </div>

        <div
          className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border ${styles[tone]}`}
        >
          <Icon className="h-5 w-5" aria-hidden="true" />
        </div>
      </div>

      <p className="mt-3 text-sm leading-6 text-slate-600">
        {note}
      </p>
    </article>
  );
}

export default function Dashboard() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { cases, stats, isLoading } = useCases();

  const recentCases = cases.slice(0, 5);
  const firstName = user?.name
    ? user.name.trim().split(' ')[0]
    : 'Researcher';

  return (
    <main className="space-y-7 sm:space-y-8 text-slate-900">
      {/* Welcome banner */}
      <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
        <div className="grid gap-6 p-6 sm:p-8 lg:grid-cols-[minmax(0,1fr)_200px] lg:items-center">
          <div>
            <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-blue-200 bg-blue-50 px-3 py-2 text-sm font-semibold text-blue-800">
              <Sparkles className="h-4 w-4 shrink-0" aria-hidden="true" />
              Clinical research workspace
            </div>

            <p className="text-base font-medium text-slate-600">
              Your research overview
            </p>

            <h1 className="mt-2 text-3xl sm:text-4xl font-bold tracking-tight text-slate-950 leading-tight">
              Welcome back, {firstName}
            </h1>

            <p className="mt-4 max-w-2xl text-base leading-7 text-slate-600">
              Organize coronary angiography research cases and prepare
              scans for the ArteryAI workflow. Keep your case activity
              organized as the segmentation project develops.
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

          <div className="hidden lg:flex items-center justify-center">
            <div className="flex h-40 w-40 items-center justify-center rounded-full border border-blue-200 bg-blue-50">
              <div className="flex h-28 w-28 items-center justify-center rounded-full border border-blue-200 bg-white text-blue-700 shadow-sm">
                <Activity className="h-14 w-14" strokeWidth={1.5} />
              </div>
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-3 border-t border-slate-200 bg-slate-50 px-6 py-4 text-sm text-slate-600 sm:flex-row sm:items-center sm:justify-between sm:px-8">
          <span className="flex items-center gap-2">
            <Database className="h-5 w-5 shrink-0 text-blue-700" aria-hidden="true" />
            {DATASET_NAME}
          </span>

          <span className="flex items-center gap-2">
            <ShieldCheck className="h-5 w-5 shrink-0 text-amber-600" aria-hidden="true" />
            Research prototype · AI model not connected
          </span>
        </div>
      </section>

      {/* Case statistics */}
      <section aria-labelledby="metrics-heading">
        <div className="mb-5">
          <h2
            id="metrics-heading"
            className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900"
          >
            Case activity
          </h2>
          <p className="mt-2 text-base leading-6 text-slate-600">
            An overview of the cases registered in this workspace.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard
            label="Total cases"
            value={stats.total}
            note="Cases registered in this workspace"
            icon={Layers}
            tone="blue"
          />

          <MetricCard
            label="Awaiting processing"
            value={stats.awaitingProcessing}
            note="Cases waiting for the next workflow step"
            icon={Clock3}
            tone="amber"
          />

          <MetricCard
            label="Completed processing"
            value={stats.completed}
            note="Cases marked complete in the current workflow"
            icon={CheckCircle2}
            tone="green"
          />

          <MetricCard
            label="Failed jobs"
            value={stats.failed}
            note="Cases marked as failed and needing review"
            icon={AlertTriangle}
            tone="rose"
          />
        </div>
      </section>

      {/* Recent cases and project information */}
      <section className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_300px]">
        {/* Recent cases */}
        <div className="min-w-0 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
          <div className="flex flex-col gap-3 border-b border-slate-200 px-5 py-5 sm:flex-row sm:items-center sm:justify-between sm:px-6">
            <div>
              <h2 className="text-xl font-bold tracking-tight text-slate-900">
                Recent research cases
              </h2>
              <p className="mt-2 text-base leading-6 text-slate-600">
                The latest cases registered in your workspace.
              </p>
            </div>

            <Link
              to="/cases"
              className="inline-flex w-fit items-center gap-2 rounded-md text-base font-semibold text-blue-700 hover:text-blue-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-blue-600"
            >
              View all cases
              <ArrowRight className="h-5 w-5" aria-hidden="true" />
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
                <thead className="bg-slate-50 text-xs font-bold uppercase tracking-wider text-slate-600">
                  <tr>
                    <th scope="col" className="px-5 py-4">Case</th>
                    <th scope="col" className="px-4 py-4">Image type</th>
                    <th scope="col" className="px-4 py-4">Date added</th>
                    <th scope="col" className="px-4 py-4">Status</th>
                    <th scope="col" className="px-5 py-4 text-right">Action</th>
                  </tr>
                </thead>

                <tbody className="divide-y divide-slate-200">
                  {recentCases.map((item) => (
                    <tr
                      key={item.id}
                      className="transition-colors hover:bg-blue-50/40"
                    >
                      <td className="px-5 py-4">
                        <p className="text-sm sm:text-base font-semibold leading-6 text-slate-900">
                          {item.name}
                        </p>
                        <p className="mt-1.5 text-sm leading-5 text-slate-600">
                          <span className="font-mono font-medium text-blue-700">
                            {item.id}
                          </span>
                          {' · '}
                          {item.targetRegion}
                        </p>
                      </td>

                      <td className="px-4 py-4 text-sm leading-6 text-slate-700">
                        <span className="inline-flex items-center gap-2">
                          <FileImage className="h-4 w-4 shrink-0 text-slate-500" aria-hidden="true" />
                          {item.imageType}
                        </span>
                      </td>

                      <td className="whitespace-nowrap px-4 py-4 text-sm text-slate-600">
                        {formatDateShort(item.createdAt)}
                      </td>

                      <td className="px-4 py-4">
                        <StatusBadge status={item.status} />
                      </td>

                      <td className="px-5 py-4 text-right">
                        <Button
                          variant="outline"
                          size="sm"
                          icon={Eye}
                          onClick={() => navigate(`/cases/${item.id}`)}
                        >
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

        {/* Project information */}
        <aside className="space-y-5">
          <section className="rounded-2xl border border-slate-200 bg-white p-5 sm:p-6 shadow-sm">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-blue-50 text-blue-700">
              <Microscope className="h-6 w-6" aria-hidden="true" />
            </div>

            <h2 className="mt-4 text-lg font-bold text-slate-900">
              Project status
            </h2>

            <p className="mt-2 text-base leading-7 text-slate-600">
              The frontend supports case organization while the AI model
              integration is being developed.
            </p>

            <div className="mt-5 space-y-4">
              <div className="flex items-start gap-3">
                <span className="mt-2 h-2.5 w-2.5 shrink-0 rounded-full bg-emerald-500" />
                <div>
                  <p className="text-base font-semibold text-slate-900">
                    Case workflow
                  </p>
                  <p className="mt-1 text-sm leading-5 text-slate-600">
                    Prototype functionality
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <span className="mt-2 h-2.5 w-2.5 shrink-0 rounded-full bg-amber-500" />
                <div>
                  <p className="text-base font-semibold text-slate-900">
                    Segmentation models
                  </p>
                  <p className="mt-1 text-sm leading-5 text-slate-600">
                    Integration pending
                  </p>
                </div>
              </div>
            </div>
          </section>

          <section className="rounded-2xl border border-blue-200 bg-blue-50/70 p-5 sm:p-6">
            <p className="text-sm font-bold uppercase tracking-wide text-blue-800">
              Research focus
            </p>

            <h2 className="mt-3 text-base font-bold leading-6 text-slate-900">
              {PROJECT_TITLE}
            </h2>

            <p className="mt-3 text-sm leading-6 text-slate-700">
              This workspace helps organize research cases. It does not
              currently generate validated stenosis predictions and must
              not be used as a clinical diagnostic tool.
            </p>

            <div className="mt-4 flex items-start gap-2 text-sm font-semibold leading-5 text-blue-800">
              <ScanLine className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
              Segmentation workflow in development
            </div>
          </section>
        </aside>
      </section>
    </main>
  );
}
