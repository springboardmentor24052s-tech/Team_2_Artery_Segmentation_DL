
import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Search,
  FolderPlus,
  Eye,
  FileImage,
  X,
  History,
  Download,
  FileText,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import Button from '../components/Button';
import StatusBadge from '../components/StatusBadge';
import EmptyState from '../components/EmptyState';
import LoadingState from '../components/LoadingState';
import { useCases } from '../context/CaseContext';
import { formatDateShort } from '../utils/formatters';
import { CASE_STATUS } from '../utils/constants';

const STATUS_FILTER_OPTIONS = [
  'All',
  CASE_STATUS.CREATED,
  CASE_STATUS.AWAITING_UPLOAD,
  CASE_STATUS.UPLOADED,
  CASE_STATUS.PROCESSING,
  CASE_STATUS.COMPLETED,
  CASE_STATUS.FAILED,
];

// Supports reports from both Case Details and Upload Scan.
const getSavedStenosisReport = (researchCase) => {
  // Reports generated from the Case Details page.
  const caseDetailsReport =
    researchCase?.geometryResults?.syntax?.stenosisReport;

  if (caseDetailsReport) {
    return caseDetailsReport;
  }

  // Reports generated automatically during Upload Scan.
  const uploadReport =
    researchCase?.segmentation?.geometryAnalysis;

  if (uploadReport?.success) {
    return {
      ...uploadReport,
      image_id:
        uploadReport.image_id ||
        uploadReport.imageId ||
        researchCase?.segmentation?.imageId,
      results:
        uploadReport.results ||
        uploadReport.report?.results ||
        [],
    };
  }

  return null;
};

const getCandidate = (branch) =>
  branch?.deepest_candidate ||
  branch?.deepestCandidate ||
  branch ||
  {};

const getNarrowingValue = (candidate) =>
  candidate?.estimated_narrowing_percent ??
  candidate?.narrowing_percentage ??
  candidate?.narrowing_percent ??
  candidate?.stenosis_percentage;

const getSeverity = (candidate) =>
  candidate?.severity_classification?.severity ||
  candidate?.severity ||
  candidate?.severity_classification?.classification_type ||
  'Not classified';

const formatPercentage = (value) => {
  if (value === null || value === undefined || value === '') {
    return 'Not available';
  }

  const number = Number(value);

  return Number.isFinite(number)
    ? `${number.toFixed(2)}%`
    : String(value);
};

const downloadJsonReport = (report, caseId) => {
  if (!report || typeof report !== 'object') return;

  const blob = new Blob(
    [JSON.stringify(report, null, 2)],
    { type: 'application/json;charset=utf-8;' }
  );

  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');

  const safeId = String(
    report.image_id || report.imageId || caseId || 'case'
  ).replace(/[^a-zA-Z0-9_-]/g, '_');

  link.href = url;
  link.download = `${safeId}_stenosis_research_report.json`;

  document.body.appendChild(link);
  link.click();
  link.remove();

  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
};

const downloadCsvReport = (report, caseId) => {
  const results = Array.isArray(report?.results)
    ? report.results
    : [];

  if (!results.length) return;

  const columns = [
    'case_id',
    'image_id',
    'branch_id',
    'branch_length',
    'candidate_count',
    'candidate_position',
    'x',
    'y',
    'minimum_diameter_pixels',
    'left_reference_pixels',
    'right_reference_pixels',
    'reference_diameter_pixels',
    'reference_asymmetry_percent',
    'estimated_narrowing_percent',
    'severity',
    'classification_type',
    'thresholds',
    'clinically_validated',
    'reference_quality',
  ];

  const escapeCsv = (value) => {
    if (value === null || value === undefined) return '';

    const text = String(value);

    return /[",\r\n]/.test(text)
      ? `"${text.replace(/"/g, '""')}"`
      : text;
  };

  const rows = results.map((branch) => {
    const candidate = getCandidate(branch);
    const classification =
      candidate.severity_classification || {};

    return [
      caseId,
      report.image_id || report.imageId || '',
      branch.branch_id ?? branch.branchId,
      branch.branch_length ?? branch.branchLength,
      branch.candidate_count ?? branch.candidateCount,
      candidate.position,
      candidate.x,
      candidate.y,
      candidate.minimum_diameter_pixels,
      candidate.left_reference_pixels,
      candidate.right_reference_pixels,
      candidate.reference_diameter_pixels,
      candidate.reference_asymmetry_percent,
      getNarrowingValue(candidate),
      classification.severity ?? candidate.severity,
      classification.classification_type,
      typeof classification.thresholds === 'object'
        ? JSON.stringify(classification.thresholds)
        : classification.thresholds,
      classification.clinically_validated,
      candidate.reference_quality ??
        candidate.quality_label ??
        branch.reference_quality ??
        branch.quality_label,
    ].map(escapeCsv);
  });

  const csv = [
    columns.map(escapeCsv).join(','),
    ...rows.map((row) => row.join(',')),
  ].join('\r\n');

  const blob = new Blob(['\uFEFF', csv], {
    type: 'text/csv;charset=utf-8;',
  });

  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');

  const safeId = String(
    report.image_id || report.imageId || caseId || 'case'
  ).replace(/[^a-zA-Z0-9_-]/g, '_');

  link.href = url;
  link.download = `${safeId}_stenosis_research_report.csv`;

  document.body.appendChild(link);
  link.click();
  link.remove();

  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
};

export default function CaseHistory() {
  const navigate = useNavigate();
  const { cases, isLoading } = useCases();

  const [searchTerm, setSearchTerm] = useState('');
  const [selectedStatus, setSelectedStatus] = useState('All');
  const [expandedReports, setExpandedReports] = useState({});

  // Filter cases based on search and status.
  const filteredCases = useMemo(() => {
    return cases.filter((c) => {
      const normalizedSearch = searchTerm.toLowerCase().trim();

      const matchesSearch =
        normalizedSearch === '' ||
        String(c.id || '').toLowerCase().includes(normalizedSearch) ||
        String(c.name || '').toLowerCase().includes(normalizedSearch) ||
        String(c.targetRegion || '')
          .toLowerCase()
          .includes(normalizedSearch);

      const matchesStatus =
        selectedStatus === 'All' || c.status === selectedStatus;

      return matchesSearch && matchesStatus;
    });
  }, [cases, searchTerm, selectedStatus]);

  const hasActiveFilters =
    searchTerm !== '' || selectedStatus !== 'All';

  const clearFilters = () => {
    setSearchTerm('');
    setSelectedStatus('All');
  };

  const toggleReport = (caseId) => {
    setExpandedReports((previous) => ({
      ...previous,
      [caseId]: !previous[caseId],
    }));
  };

  return (
    <div className="space-y-6">

      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-slate-500 text-xs font-semibold uppercase tracking-wider mb-1">
            <History className="w-4 h-4 text-blue-600" />
            <span>Case Repository</span>
          </div>

          <h2 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
            Research Case History
          </h2>

          <p className="text-xs sm:text-sm text-slate-500 mt-0.5">
            View, search, and manage all registered ARCADE coronary angiography studies.
          </p>
        </div>

        <Button
          variant="primary"
          size="md"
          icon={FolderPlus}
          onClick={() => navigate('/cases/create')}
        >
          Create New Case
        </Button>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-2xs space-y-3">
        <div className="flex flex-col md:flex-row gap-3">

          <div className="relative flex-1">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
              <Search className="w-4 h-4" />
            </div>

            <input
              type="text"
              placeholder="Search by Case ID (e.g. ARC-2025-001), name, or target vessel..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-9 py-2 rounded-lg border border-slate-300 text-sm focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100 bg-white"
            />

            {searchTerm && (
              <button
                type="button"
                onClick={() => setSearchTerm('')}
                className="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-600"
                aria-label="Clear search"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>

          <div className="flex items-center gap-2">
            <div className="relative w-full sm:w-48">
              <select
                value={selectedStatus}
                onChange={(e) => setSelectedStatus(e.target.value)}
                className="w-full py-2 px-3 rounded-lg border border-slate-300 text-sm focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100 bg-white text-slate-700 cursor-pointer"
              >
                {STATUS_FILTER_OPTIONS.map((status) => (
                  <option key={status} value={status}>
                    {status === 'All' ? 'All Statuses' : status}
                  </option>
                ))}
              </select>
            </div>

            {hasActiveFilters && (
              <Button
                variant="ghost"
                size="sm"
                icon={X}
                onClick={clearFilters}
                className="text-slate-500"
              >
                Reset
              </Button>
            )}
          </div>
        </div>

        <div className="text-xs text-slate-500 flex items-center justify-between pt-1 border-t border-slate-100">
          <span>
            Showing{' '}
            <strong className="text-slate-800">
              {filteredCases.length}
            </strong>{' '}
            of{' '}
            <strong className="text-slate-800">
              {cases.length}
            </strong>{' '}
            total study cases
          </span>

          {hasActiveFilters && (
            <span className="text-blue-600 font-medium">
              Filter active
            </span>
          )}
        </div>
      </div>

      {/* Cases Table */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-2xs overflow-hidden">

        {isLoading ? (
          <LoadingState message="Loading case history..." />
        ) : cases.length === 0 ? (
          <div className="p-8">
            <EmptyState
              title="No research cases found"
              description="Your repository does not have any research cases registered yet. Register your first study case to start."
              actionLabel="Create First Case"
              actionIcon={FolderPlus}
              onAction={() => navigate('/cases/create')}
            />
          </div>
        ) : filteredCases.length === 0 ? (
          <div className="p-8">
            <EmptyState
              title="No matching research cases"
              description={`No study cases match your search term "${searchTerm}" and status filter "${selectedStatus}". Try adjusting your keywords or clearing the filter.`}
              actionLabel="Clear All Filters"
              actionIcon={X}
              onAction={clearFilters}
            />
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-slate-700">

              <thead className="bg-slate-50/80 text-[11px] uppercase tracking-wider font-semibold text-slate-500 border-b border-slate-200">
                <tr>
                  <th scope="col" className="py-3 px-4">Case ID</th>
                  <th scope="col" className="py-3 px-4">Case Name</th>
                  <th scope="col" className="py-3 px-4">Target Vessel</th>
                  <th scope="col" className="py-3 px-4">Image Type</th>
                  <th scope="col" className="py-3 px-4">Date Added</th>
                  <th scope="col" className="py-3 px-4">Status</th>
                  <th scope="col" className="py-3 px-4">Stenosis Report</th>
                  <th scope="col" className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>

              <tbody className="divide-y divide-slate-100">
                {filteredCases.map((c) => {
                  const stenosisReport = getSavedStenosisReport(c);

                  const hasReport =
                    stenosisReport &&
                    typeof stenosisReport === 'object';

                  const reportResults = Array.isArray(
                    stenosisReport?.results
                  )
                    ? stenosisReport.results
                    : [];

                  const hasCsvRows = reportResults.length > 0;
                  const isExpanded = Boolean(expandedReports[c.id]);

                  return (
                    <React.Fragment key={c.id}>
                      <tr className="hover:bg-slate-50/70 transition-colors">
                        <td className="py-3.5 px-4 font-mono font-medium text-blue-600 text-xs">
                          {c.id}
                        </td>

                        <td className="py-3.5 px-4">
                          <div className="font-semibold text-slate-900 truncate max-w-xs">
                            {c.name}
                          </div>
                          <div className="text-xs text-slate-400 truncate max-w-xs">
                            {c.description}
                          </div>
                        </td>

                        <td className="py-3.5 px-4 text-xs text-slate-700">
                          {c.targetRegion || 'Unspecified'}
                        </td>

                        <td className="py-3.5 px-4 text-xs text-slate-600">
                          <span className="inline-flex items-center gap-1 bg-slate-100 px-2 py-0.5 rounded text-slate-700 font-medium text-xs">
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

                        <td className="py-3.5 px-4">
                          {hasReport ? (
                            <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 border border-emerald-200 px-2.5 py-1 text-xs font-semibold text-emerald-700 whitespace-nowrap">
                              <FileText className="w-3.5 h-3.5" />
                              Available
                            </span>
                          ) : (
                            <span className="inline-flex items-center rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-500 whitespace-nowrap">
                              Not generated
                            </span>
                          )}
                        </td>

                        <td className="py-3.5 px-4">
                          <div className="flex flex-wrap items-center justify-end gap-2">

                            {hasReport && (
                              <>
                                <button
                                  type="button"
                                  onClick={() => toggleReport(c.id)}
                                  className="inline-flex items-center gap-1.5 rounded-lg border border-indigo-200 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 text-xs font-semibold px-2.5 py-2"
                                  title="View stenosis report"
                                >
                                  <FileText className="w-3.5 h-3.5" />
                                  {isExpanded ? 'Hide Report' : 'View Report'}
                                  {isExpanded ? (
                                    <ChevronUp className="w-3.5 h-3.5" />
                                  ) : (
                                    <ChevronDown className="w-3.5 h-3.5" />
                                  )}
                                </button>

                                <button
                                  type="button"
                                  onClick={() =>
                                    downloadJsonReport(stenosisReport, c.id)
                                  }
                                  className="inline-flex items-center gap-1.5 rounded-lg border border-blue-200 bg-blue-50 hover:bg-blue-100 text-blue-700 text-xs font-semibold px-2.5 py-2"
                                  title="Download stenosis report as JSON"
                                >
                                  <Download className="w-3.5 h-3.5" />
                                  JSON
                                </button>

                                <button
                                  type="button"
                                  onClick={() =>
                                    downloadCsvReport(stenosisReport, c.id)
                                  }
                                  disabled={!hasCsvRows}
                                  className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-200 bg-emerald-50 hover:bg-emerald-100 disabled:bg-slate-100 disabled:text-slate-400 disabled:border-slate-200 text-emerald-700 text-xs font-semibold px-2.5 py-2"
                                  title="Download stenosis report as CSV"
                                >
                                  <Download className="w-3.5 h-3.5" />
                                  CSV
                                </button>
                              </>
                            )}

                            <Button
                              variant="outline"
                              size="sm"
                              icon={Eye}
                              onClick={() => navigate(`/cases/${c.id}`)}
                            >
                              View Details
                            </Button>

                          </div>
                        </td>
                      </tr>

                      {/* Expanded Stenosis Report */}
                      {hasReport && isExpanded && (
                        <tr className="bg-slate-50/70">
                          <td colSpan={8} className="px-4 py-5">
                            <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-4">

                              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                                <div>
                                  <div className="flex items-center gap-2">
                                    <FileText className="w-5 h-5 text-blue-600" />
                                    <h3 className="font-semibold text-slate-900">
                                      Stenosis Research Report
                                    </h3>
                                  </div>
                                  <p className="text-xs text-slate-500 mt-1">
                                    Saved analysis for case {c.id}
                                  </p>
                                </div>

                                <span className="inline-flex items-center w-fit rounded-full bg-emerald-50 border border-emerald-200 px-3 py-1 text-xs font-semibold text-emerald-700">
                                  Report available
                                </span>
                              </div>

                              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                                  <p className="text-xs text-slate-500">
                                    Image ID
                                  </p>
                                  <p className="text-sm font-semibold text-slate-800 mt-1 break-all">
                                    {stenosisReport.image_id ||
                                      stenosisReport.imageId ||
                                      c.segmentation?.imageId ||
                                      'Not available'}
                                  </p>
                                </div>

                                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                                  <p className="text-xs text-slate-500">
                                    Analysed branches / results
                                  </p>
                                  <p className="text-xl font-bold text-slate-900 mt-1">
                                    {reportResults.length}
                                  </p>
                                </div>

                                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                                  <p className="text-xs text-slate-500">
                                    Report status
                                  </p>
                                  <p className="text-sm font-semibold text-emerald-700 mt-1">
                                    {stenosisReport.success === false
                                      ? 'Check report'
                                      : 'Successfully generated'}
                                  </p>
                                </div>
                              </div>

                              {reportResults.length > 0 ? (
                                <div className="space-y-2">
                                  <h4 className="text-sm font-semibold text-slate-800">
                                    Stenosis Analysis Results
                                  </h4>

                                  <div className="overflow-x-auto border border-slate-200 rounded-lg">
                                    <table className="w-full text-left text-xs">
                                      <thead className="bg-slate-50 text-slate-500 uppercase tracking-wide">
                                        <tr>
                                          <th className="px-3 py-3 font-semibold">
                                            Branch
                                          </th>
                                          <th className="px-3 py-3 font-semibold">
                                            Position
                                          </th>
                                          <th className="px-3 py-3 font-semibold">
                                            Narrowing
                                          </th>
                                          <th className="px-3 py-3 font-semibold">
                                            Severity
                                          </th>
                                          <th className="px-3 py-3 font-semibold">
                                            Reference Quality
                                          </th>
                                        </tr>
                                      </thead>

                                      <tbody className="divide-y divide-slate-100">
                                        {reportResults.map((branch, index) => {
                                          const candidate =
                                            getCandidate(branch);

                                          const narrowing =
                                            getNarrowingValue(candidate);

                                          const severity =
                                            getSeverity(candidate);

                                          const referenceQuality =
                                            candidate.reference_quality ||
                                            candidate.quality_label ||
                                            branch.reference_quality ||
                                            branch.quality_label ||
                                            'Not available';

                                          return (
                                            <tr
                                              key={`${branch.branch_id || branch.branchId || 'branch'}-${index}`}
                                              className="hover:bg-slate-50"
                                            >
                                              <td className="px-3 py-3 font-medium text-slate-800">
                                                {branch.branch_id ??
                                                  branch.branchId ??
                                                  `Branch ${index + 1}`}
                                              </td>

                                              <td className="px-3 py-3 text-slate-600">
                                                {candidate.position ??
                                                  'Not available'}
                                              </td>

                                              <td className="px-3 py-3 font-semibold text-slate-800">
                                                {formatPercentage(narrowing)}
                                              </td>

                                              <td className="px-3 py-3 text-slate-700">
                                                {severity}
                                              </td>

                                              <td className="px-3 py-3 text-slate-600">
                                                {referenceQuality}
                                              </td>
                                            </tr>
                                          );
                                        })}
                                      </tbody>
                                    </table>
                                  </div>
                                </div>
                              ) : (
                                <div className="rounded-lg border border-amber-200 bg-amber-50 p-3">
                                  <p className="text-sm font-medium text-amber-800">
                                    The report is saved, but no branch result rows were found.
                                  </p>
                                  <p className="text-xs text-amber-700 mt-1">
                                    You can still download the complete JSON report to inspect its contents.
                                  </p>
                                </div>
                              )}

                              <div className="flex flex-wrap justify-end gap-2 pt-2 border-t border-slate-100">
                                <button
                                  type="button"
                                  onClick={() =>
                                    downloadJsonReport(stenosisReport, c.id)
                                  }
                                  className="inline-flex items-center gap-1.5 rounded-lg border border-blue-200 bg-blue-50 hover:bg-blue-100 text-blue-700 text-xs font-semibold px-3 py-2"
                                >
                                  <Download className="w-3.5 h-3.5" />
                                  Download JSON
                                </button>

                                <button
                                  type="button"
                                  onClick={() =>
                                    downloadCsvReport(stenosisReport, c.id)
                                  }
                                  disabled={!hasCsvRows}
                                  className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-200 bg-emerald-50 hover:bg-emerald-100 disabled:bg-slate-100 disabled:text-slate-400 disabled:border-slate-200 text-emerald-700 text-xs font-semibold px-3 py-2"
                                >
                                  <Download className="w-3.5 h-3.5" />
                                  Download CSV
                                </button>

                                <Button
                                  variant="outline"
                                  size="sm"
                                  icon={Eye}
                                  onClick={() => navigate(`/cases/${c.id}`)}
                                >
                                  Open Full Case
                                </Button>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
