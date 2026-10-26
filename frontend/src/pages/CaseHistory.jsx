import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Search,
  Filter,
  FolderPlus,
  Eye,
  FileImage,
  Calendar,
  X,
  History,
  Layers,
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

export default function CaseHistory() {
  const navigate = useNavigate();
  const { cases, isLoading } = useCases();

  const [searchTerm, setSearchTerm] = useState('');
  const [selectedStatus, setSelectedStatus] = useState('All');

  // Filter cases based on search and status
  const filteredCases = useMemo(() => {
    return cases.filter((c) => {
      const matchesSearch =
        searchTerm === '' ||
        c.id.toLowerCase().includes(searchTerm.toLowerCase().trim()) ||
        c.name.toLowerCase().includes(searchTerm.toLowerCase().trim()) ||
        c.targetRegion?.toLowerCase().includes(searchTerm.toLowerCase().trim());

      const matchesStatus =
        selectedStatus === 'All' || c.status === selectedStatus;

      return matchesSearch && matchesStatus;
    });
  }, [cases, searchTerm, selectedStatus]);

  const hasActiveFilters = searchTerm !== '' || selectedStatus !== 'All';

  const clearFilters = () => {
    setSearchTerm('');
    setSelectedStatus('All');
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
          {/* Search Input */}
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
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>

          {/* Status Dropdown */}
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

        {/* Results count counter */}
        <div className="text-xs text-slate-500 flex items-center justify-between pt-1 border-t border-slate-100">
          <span>
            Showing <strong className="text-slate-800">{filteredCases.length}</strong> of{' '}
            <strong className="text-slate-800">{cases.length}</strong> total study cases
          </span>
          {hasActiveFilters && (
            <span className="text-blue-600 font-medium">Filter active</span>
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
                  <th scope="col" className="py-3 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {filteredCases.map((c) => (
                  <tr key={c.id} className="hover:bg-slate-50/70 transition-colors">
                    <td className="py-3.5 px-4 font-mono font-medium text-blue-600 text-xs">
                      {c.id}
                    </td>
                    <td className="py-3.5 px-4">
                      <div className="font-semibold text-slate-900 truncate max-w-xs">{c.name}</div>
                      <div className="text-xs text-slate-400 truncate max-w-xs">{c.description}</div>
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
                    <td className="py-3.5 px-4 text-right">
                      <Button
                        variant="outline"
                        size="sm"
                        icon={Eye}
                        onClick={() => navigate(`/cases/${c.id}`)}
                      >
                        View Details
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
