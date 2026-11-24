import React, { useMemo } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  FolderPlus,
  UploadCloud,
  Layers,
  Clock,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Eye,
  FileImage,
  Sparkles,
  ScanLine,
  GitBranch,
  Activity,
  BrainCircuit,
  Microscope,
  ChartColumn,
  ShieldAlert,
  CircleDot,
} from 'lucide-react';

import Button from '../components/Button';
import StatusBadge from '../components/StatusBadge';
import EmptyState from '../components/EmptyState';
import LoadingState from '../components/LoadingState';

import { useCases } from '../context/CaseContext';
import { useAuth } from '../context/AuthContext';
import { formatDateShort } from '../utils/formatters';
import { PROJECT_TITLE, DATASET_NAME } from '../utils/constants';

// ============================================================
// REUSABLE METRIC CARD
// ============================================================

function MetricCard({
  title,
  value,
  description,
  icon: Icon,
  color = 'bg-blue-50 text-blue-600 border-blue-100',
  valueColor = 'text-slate-900',
}) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-2xs flex items-center justify-between gap-3">
      <div className="min-w-0">
        <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">
          {title}
        </p>

        <p className={`text-2xl font-bold mt-1 ${valueColor}`}>
          {value}
        </p>

        <p className="text-[11px] text-slate-400 mt-0.5">
          {description}
        </p>
      </div>

      <div className={`w-10 h-10 rounded-lg flex items-center justify-center border shrink-0 ${color}`}>
        <Icon className="w-5 h-5" />
      </div>
    </div>
  );
}

// ============================================================
// MODEL USAGE BAR
// ============================================================

function ModelUsageBar({ label, count, total, color }) {
  const percentage = total > 0
    ? (count / total) * 100
    : 0;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-medium text-slate-700">
          {label}
        </span>

        <span className="text-sm font-semibold text-slate-900 tabular-nums">
          {count}
          <span className="text-xs font-normal text-slate-400 ml-1">
            ({percentage.toFixed(0)}%)
          </span>
        </span>
      </div>

      <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

// ============================================================
// CASE STATUS DISTRIBUTION
// ============================================================

function StatusDistribution({ label, count, total, color }) {
  const percentage = total > 0
    ? (count / total) * 100
    : 0;

  return (
    <div className="space-y-1.5">
      <div className="flex justify-between items-center text-xs">
        <span className="text-slate-600 font-medium">{label}</span>

        <span className="text-slate-900 font-semibold tabular-nums">
          {count}
        </span>
      </div>

      <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

// ============================================================
// MAIN DASHBOARD
// ============================================================

export default function Dashboard() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { cases, stats, isLoading } = useCases();

  // ==========================================================
  // DYNAMIC RESEARCH ANALYTICS
  // ==========================================================

  const analytics = useMemo(() => {
    const safeCases = Array.isArray(cases) ? cases : [];

    // Cases with actual segmentation records
    const segmentationCases = safeCases.filter(
      (c) => Boolean(c.segmentation?.imageId)
    );

    // Successful geometry reports
    const geometryCases = safeCases.filter(
      (c) => c.segmentation?.geometryAnalysis?.success === true
    );

    // Model usage
    const syntaxCount = safeCases.filter(
      (c) => c.segmentation?.modelType === 'syntax'
    ).length;

    const stenosisCount = safeCases.filter(
      (c) => c.segmentation?.modelType === 'stenosis'
    ).length;

    const combinedCount = safeCases.filter(
      (c) => c.segmentation?.modelType === 'combined'
    ).length;

    // Geometry statistics
    const branchesExtracted = geometryCases.reduce(
      (total, c) =>
        total +
        (Number(c.segmentation?.geometryAnalysis?.branchesExtracted) || 0),
      0
    );

    const branchesEvaluated = geometryCases.reduce(
      (total, c) =>
        total +
        (Number(c.segmentation?.geometryAnalysis?.branchesEvaluated) || 0),
      0
    );

    const experimentalReports = geometryCases.filter(
      (c) =>
        c.segmentation?.geometryAnalysis?.experimental === true
    ).length;

    // Stenosis and combined model usage
    const stenosisAnalysisCases = stenosisCount + combinedCount;

    // Case status distribution
    const uploaded = safeCases.filter(
      (c) => c.status === 'Uploaded'
    ).length;

    const created = safeCases.filter(
      (c) => c.status === 'Created'
    ).length;

    const awaitingUpload = safeCases.filter(
      (c) => c.status === 'Awaiting Upload'
    ).length;

    const processing = safeCases.filter(
      (c) => c.status === 'Processing'
    ).length;

    const completed = safeCases.filter(
      (c) => c.status === 'Completed'
    ).length;

    const failed = safeCases.filter(
      (c) => c.status === 'Failed'
    ).length;

    return {
      total: safeCases.length,
      segmentationCount: segmentationCases.length,
      geometryCount: geometryCases.length,

      syntaxCount,
      stenosisCount,
      combinedCount,

      branchesExtracted,
      branchesEvaluated,
      experimentalReports,

      stenosisAnalysisCases,

      uploaded,
      created,
      awaitingUpload,
      processing,
      completed,
      failed,
    };
  }, [cases]);

  // Latest five research cases
  const recentCases = cases.slice(0, 5);

  const processedPercentage = analytics.total > 0
    ? Math.round(
        (analytics.segmentationCount / analytics.total) * 100
      )
    : 0;

  // ==========================================================
  // DASHBOARD UI
  // ==========================================================

  return (
    <div className="space-y-6">

      {/* =====================================================
          WELCOME BANNER
      ====================================================== */}

      <div className="bg-white rounded-xl border border-slate-200/90 p-5 sm:p-6 shadow-2xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">

          <div className="space-y-1.5 max-w-3xl">

            <div className="inline-flex items-center gap-1.5 text-xs font-semibold text-blue-700 bg-blue-50 px-2.5 py-0.5 rounded-full border border-blue-200">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Research Analytics Dashboard</span>
            </div>

            <h2 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
              Welcome back, {user?.name ? user.name.split(' ')[0] : 'Researcher'}
            </h2>

            <p className="text-sm text-slate-600 leading-relaxed">
              <strong>{PROJECT_TITLE}:</strong> An AI-assisted medical imaging platform for managing and analyzing 2D X-ray coronary angiography (XCA) scans using the <em>{DATASET_NAME}</em>.
            </p>

          </div>

          <div className="flex items-center gap-2.5 shrink-0 flex-wrap">

            <Button
              variant="outline"
              size="md"
              icon={FolderPlus}
              onClick={() => navigate('/cases/create')}
            >
              Create Case
            </Button>

            <Button
              variant="primary"
              size="md"
              icon={UploadCloud}
              onClick={() => navigate('/upload')}
            >
              Upload Scan
            </Button>

          </div>
        </div>
      </div>

      {/* =====================================================
          SECTION 1: RESEARCH OVERVIEW
      ====================================================== */}

      <section className="space-y-3">

        <div className="flex items-center justify-between">

          <div>
            <h3 className="text-base font-semibold text-slate-900">
              Research Overview
            </h3>

            <p className="text-xs text-slate-500 mt-0.5">
              Live statistics from registered research cases
            </p>
          </div>

          <span className="inline-flex items-center gap-1.5 text-[11px] font-medium text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-full border border-emerald-100">
            <CircleDot className="w-3 h-3" />
            Live Data
          </span>

        </div>

        {isLoading ? (
          <LoadingState message="Calculating research statistics..." />
        ) : (

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">

            <MetricCard
              title="Total Cases"
              value={stats.total}
              description="Registered research cases"
              icon={Layers}
            />

            <MetricCard
              title="Processed Scans"
              value={analytics.segmentationCount}
              description={`${processedPercentage}% of registered cases`}
              icon={ScanLine}
              color="bg-blue-50 text-blue-600 border-blue-100"
            />

            <MetricCard
              title="Geometry Reports"
              value={analytics.geometryCount}
              description="Successful geometry analyses"
              icon={GitBranch}
              color="bg-violet-50 text-violet-600 border-violet-100"
            />

            <MetricCard
              title="Branches Evaluated"
              value={analytics.branchesEvaluated}
              description={`${analytics.branchesExtracted} branches extracted`}
              icon={Activity}
              color="bg-emerald-50 text-emerald-600 border-emerald-100"
              valueColor="text-emerald-700"
            />

          </div>
        )}

      </section>

      {/* =====================================================
          SECTION 2: MODEL PIPELINE
      ====================================================== */}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

        {/* MODEL USAGE */}

        <section className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs">

          <div className="flex items-center justify-between mb-5">

            <div>
              <h3 className="text-base font-semibold text-slate-900 flex items-center gap-2">
                <BrainCircuit className="w-4 h-4 text-blue-600" />
                Model Usage Overview
              </h3>

              <p className="text-xs text-slate-500 mt-1">
                Distribution of recorded analysis modes
              </p>
            </div>

            <span className="text-xs font-semibold text-slate-500 bg-slate-100 px-2 py-1 rounded-md">
              {analytics.segmentationCount} scans
            </span>

          </div>

          {analytics.segmentationCount === 0 ? (

            <div className="py-8 text-center text-sm text-slate-500">
              No segmentation records available yet.
            </div>

          ) : (

            <div className="space-y-5">

              <ModelUsageBar
                label="Syntax U-Net"
                count={analytics.syntaxCount}
                total={analytics.segmentationCount}
                color="bg-blue-500"
              />

              <ModelUsageBar
                label="Attention U-Net V4"
                count={analytics.stenosisCount}
                total={analytics.segmentationCount}
                color="bg-violet-500"
              />

              <ModelUsageBar
                label="Combined Analysis"
                count={analytics.combinedCount}
                total={analytics.segmentationCount}
                color="bg-emerald-500"
              />

            </div>
          )}

          <div className="mt-5 pt-4 border-t border-slate-100">
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Model usage is calculated from saved segmentation records.
              Combined analysis uses both Syntax U-Net and Attention U-Net V4.
            </p>
          </div>

        </section>

        {/* ANALYSIS PIPELINE */}

        <section className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs">

          <div className="mb-5">

            <h3 className="text-base font-semibold text-slate-900 flex items-center gap-2">
              <Layers className="w-4 h-4 text-blue-600" />
              Analysis Pipeline
            </h3>

            <p className="text-xs text-slate-500 mt-1">
              Overview of the research processing workflow
            </p>

          </div>

          <div className="space-y-3">

            <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg border border-slate-100">

              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-lg bg-blue-100 text-blue-600 flex items-center justify-center">
                  <FileImage className="w-4 h-4" />
                </div>

                <div>
                  <p className="text-sm font-semibold text-slate-800">
                    Image Segmentation
                  </p>

                  <p className="text-[11px] text-slate-500">
                    Syntax and stenosis model outputs
                  </p>
                </div>
              </div>

              <span className="text-sm font-bold text-slate-900">
                {analytics.segmentationCount}
              </span>

            </div>

            <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg border border-slate-100">

              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-lg bg-violet-100 text-violet-600 flex items-center justify-center">
                  <GitBranch className="w-4 h-4" />
                </div>

                <div>
                  <p className="text-sm font-semibold text-slate-800">
                    Geometry Analysis
                  </p>

                  <p className="text-[11px] text-slate-500">
                    Vessel branch measurements
                  </p>
                </div>
              </div>

              <span className="text-sm font-bold text-slate-900">
                {analytics.geometryCount}
              </span>

            </div>

            <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg border border-slate-100">

              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-lg bg-emerald-100 text-emerald-600 flex items-center justify-center">
                  <Microscope className="w-4 h-4" />
                </div>

                <div>
                  <p className="text-sm font-semibold text-slate-800">
                    Stenosis / Combined Analysis
                  </p>

                  <p className="text-[11px] text-slate-500">
                    Cases using stenosis analysis mode
                  </p>
                </div>
              </div>

              <span className="text-sm font-bold text-slate-900">
                {analytics.stenosisAnalysisCases}
              </span>

            </div>

          </div>

        </section>

      </div>

      {/* =====================================================
          SECTION 3: GEOMETRY ANALYSIS INSIGHTS
      ====================================================== */}

      <section className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs">

        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-5">

          <div>
            <h3 className="text-base font-semibold text-slate-900 flex items-center gap-2">
              <Activity className="w-4 h-4 text-blue-600" />
              Geometry Analysis Insights
            </h3>

            <p className="text-xs text-slate-500 mt-1">
              Branch-level statistics from successful geometry reports
            </p>
          </div>

          <span className="inline-flex items-center gap-1.5 text-[11px] font-medium text-amber-700 bg-amber-50 px-2.5 py-1 rounded-full border border-amber-200">
            <ShieldAlert className="w-3.5 h-3.5" />
            Experimental Research
          </span>

        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">

          <div className="rounded-lg border border-slate-200 p-4">
            <p className="text-xs font-medium text-slate-500">
              Total Branches Extracted
            </p>

            <p className="text-2xl font-bold text-slate-900 mt-2">
              {analytics.branchesExtracted}
            </p>

            <p className="text-[11px] text-slate-400 mt-1">
              Across successful reports
            </p>
          </div>

          <div className="rounded-lg border border-slate-200 p-4">
            <p className="text-xs font-medium text-slate-500">
              Branches Evaluated
            </p>

            <p className="text-2xl font-bold text-blue-700 mt-2">
              {analytics.branchesEvaluated}
            </p>

            <p className="text-[11px] text-slate-400 mt-1">
              Branch-level measurement results
            </p>
          </div>

          <div className="rounded-lg border border-slate-200 p-4">
            <p className="text-xs font-medium text-slate-500">
              Experimental Reports
            </p>

            <p className="text-2xl font-bold text-amber-700 mt-2">
              {analytics.experimentalReports}
            </p>

            <p className="text-[11px] text-slate-400 mt-1">
              Reports marked experimental
            </p>
          </div>

        </div>

        <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">

          <div className="flex gap-2 items-start">

            <AlertTriangle className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />

            <p className="text-xs text-amber-800 leading-relaxed">
              <strong>Research limitation:</strong> Geometry measurements
              are experimental and pixel-based. They are not clinically
              validated and must not be interpreted as a medical diagnosis.
              Extracted branch counts and evaluated branch counts are
              operational statistics, not segmentation accuracy metrics.
            </p>

          </div>

        </div>

      </section>

      {/* =====================================================
          SECTION 4: RESEARCH ACTIVITY
      ====================================================== */}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

        {/* CASE STATUS DISTRIBUTION */}

        <section className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs">

          <div className="mb-5">

            <h3 className="text-base font-semibold text-slate-900 flex items-center gap-2">
              <ChartColumn className="w-4 h-4 text-blue-600" />
              Research Activity
            </h3>

            <p className="text-xs text-slate-500 mt-1">
              Distribution of research case statuses
            </p>

          </div>

          <div className="space-y-4">

            <StatusDistribution
              label="Uploaded"
              count={analytics.uploaded}
              total={analytics.total}
              color="bg-blue-500"
            />

            <StatusDistribution
              label="Completed"
              count={analytics.completed}
              total={analytics.total}
              color="bg-emerald-500"
            />

            <StatusDistribution
              label="Processing"
              count={analytics.processing}
              total={analytics.total}
              color="bg-violet-500"
            />

            <StatusDistribution
              label="Awaiting Upload"
              count={analytics.awaitingUpload}
              total={analytics.total}
              color="bg-amber-500"
            />

            <StatusDistribution
              label="Created"
              count={analytics.created}
              total={analytics.total}
              color="bg-slate-400"
            />

            <StatusDistribution
              label="Failed"
              count={analytics.failed}
              total={analytics.total}
              color="bg-rose-500"
            />

          </div>

          <div className="mt-5 pt-4 border-t border-slate-100 flex justify-between items-center">

            <span className="text-xs text-slate-500">
              Total research cases
            </span>

            <span className="text-sm font-bold text-slate-900">
              {analytics.total}
            </span>

          </div>

        </section>

        {/* PROCESSING SUMMARY */}

        <section className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs">

          <div className="mb-5">

            <h3 className="text-base font-semibold text-slate-900 flex items-center gap-2">
              <Activity className="w-4 h-4 text-blue-600" />
              Processing Summary
            </h3>

            <p className="text-xs text-slate-500 mt-1">
              Current research workflow status
            </p>

          </div>

          <div className="grid grid-cols-2 gap-3">

            <div className="p-3 rounded-lg bg-blue-50 border border-blue-100">
              <p className="text-xs text-blue-700 font-medium">
                Awaiting Processing
              </p>

              <p className="text-2xl font-bold text-blue-900 mt-1">
                {stats.awaitingProcessing}
              </p>
            </div>

            <div className="p-3 rounded-lg bg-violet-50 border border-violet-100">
              <p className="text-xs text-violet-700 font-medium">
                In Progress
              </p>

              <p className="text-2xl font-bold text-violet-900 mt-1">
                {stats.processing}
              </p>
            </div>

            <div className="p-3 rounded-lg bg-emerald-50 border border-emerald-100">
              <p className="text-xs text-emerald-700 font-medium">
                Completed
              </p>

              <p className="text-2xl font-bold text-emerald-900 mt-1">
                {stats.completed}
              </p>
            </div>

            <div className="p-3 rounded-lg bg-rose-50 border border-rose-100">
              <p className="text-xs text-rose-700 font-medium">
                Failed
              </p>

              <p className="text-2xl font-bold text-rose-900 mt-1">
                {stats.failed}
              </p>
            </div>

          </div>

          <div className="mt-5 p-3 rounded-lg border border-slate-200">

            <div className="flex items-center gap-2 mb-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />

              <p className="text-sm font-semibold text-slate-800">
                Analysis Availability
              </p>
            </div>

            {/* Corrected spacing between the geometry count and "have" */}
            <p className="text-xs text-slate-500 leading-relaxed">
              {analytics.segmentationCount} of {analytics.total} research
              cases have stored segmentation results.{" "}
              {analytics.geometryCount} have successful geometry reports.
            </p>

          </div>

        </section>

      </div>

      {/* =====================================================
          SECTION 5: MODEL EVALUATION NOTICE
      ====================================================== */}

      <section className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs">

        <div className="flex items-start gap-3">

          <div className="w-10 h-10 rounded-lg bg-indigo-50 text-indigo-600 border border-indigo-100 flex items-center justify-center shrink-0">
            <BrainCircuit className="w-5 h-5" />
          </div>

          <div>

            <h3 className="text-sm font-semibold text-slate-900">
              Model Evaluation Metrics
            </h3>

            <p className="text-xs text-slate-500 mt-1 leading-relaxed">
              Quantitative model-performance metrics such as Dice
              coefficient, IoU, precision, recall, and F1-score are
              intentionally not calculated from case counts or geometry
              reports. They should be displayed here only when verified
              evaluation results from the benchmark dataset are available.
            </p>

            <div className="mt-3 inline-flex items-center gap-1.5 text-[11px] font-medium text-slate-600 bg-slate-100 px-2.5 py-1 rounded-full">
              <Clock className="w-3 h-3" />
              Awaiting verified evaluation results
            </div>

          </div>

        </div>

      </section>

      {/* =====================================================
          SECTION 6: RECENT RESEARCH CASES
      ====================================================== */}

      <section className="bg-white rounded-xl border border-slate-200 shadow-2xs overflow-hidden">

        <div className="p-4 sm:p-5 border-b border-slate-200 flex items-center justify-between">

          <div>
            <h3 className="text-base font-semibold text-slate-900">
              Recent Research Cases
            </h3>

            <p className="text-xs text-slate-500 mt-0.5">
              Latest coronary angiography studies registered in the platform
            </p>
          </div>

          <Link
            to="/cases"
            className="text-xs font-semibold text-blue-600 hover:text-blue-800 flex items-center gap-1 group"
          >
            <span>View All Cases</span>

            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
          </Link>

        </div>

        {isLoading ? (

          <LoadingState message="Retrieving recent cases..." />

        ) : recentCases.length === 0 ? (

          <div className="p-6">

            <EmptyState
              title="No research cases registered yet"
              description="Create a research case to begin logging coronary angiography images and tracking preprocessing."
              actionLabel="Create Your First Case"
              actionIcon={FolderPlus}
              onAction={() => navigate('/cases/create')}
            />

          </div>

        ) : (

          <div className="overflow-x-auto">

            <table className="w-full text-left text-sm text-slate-700">

              <thead className="bg-slate-50/80 text-[11px] uppercase tracking-wider font-semibold text-slate-500 border-b border-slate-200">

                <tr>
                  <th scope="col" className="py-3 px-4">Case ID</th>
                  <th scope="col" className="py-3 px-4">Case Name</th>
                  <th scope="col" className="py-3 px-4">Image Type</th>
                  <th scope="col" className="py-3 px-4">Date Added</th>
                  <th scope="col" className="py-3 px-4">Status</th>
                  <th scope="col" className="py-3 px-4 text-right">Action</th>
                </tr>

              </thead>

              <tbody className="divide-y divide-slate-100">

                {recentCases.map((c) => (

                  <tr
                    key={c.id}
                    className="hover:bg-slate-50/70 transition-colors"
                  >

                    <td className="py-3.5 px-4 font-mono font-medium text-blue-600 text-xs">
                      {c.id}
                    </td>

                    <td className="py-3.5 px-4">

                      <div className="font-medium text-slate-900 truncate max-w-xs">
                        {c.name}
                      </div>

                      <div className="text-xs text-slate-400 truncate max-w-xs">
                        {c.targetRegion}
                      </div>

                    </td>

                    <td className="py-3.5 px-4 text-xs text-slate-600">

                      <span className="inline-flex items-center gap-1 bg-slate-100 px-2 py-0.5 rounded text-slate-700 font-medium">

                        <FileImage className="w-3 h-3 text-slate-400" />

                        {c.imageType}

                      </span>

                    </td>

                    <td className="py-3.5 px-4 text-xs text-slate-500">
                      {formatDateShort(c.createdAt)}
                    </td>

                    <td className="py-3.5 px-4">
                      <StatusBadge status={c.status} />
                    </td>

                    <td className="py-3.5 px-4 text-right">

                      <Button
                        variant="outline"
                        size="sm"
                        icon={Eye}
                        onClick={() => navigate(`/cases/${c.id}`)}
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

      </section>

    </div>
  );
}