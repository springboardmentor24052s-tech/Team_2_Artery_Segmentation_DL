import React from 'react';
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
} from 'lucide-react';
import Button from '../components/Button';
import StatusBadge from '../components/StatusBadge';
import EmptyState from '../components/EmptyState';
import LoadingState from '../components/LoadingState';
import { useCases } from '../context/CaseContext';
import { useAuth } from '../context/AuthContext';
import { formatDateShort } from '../utils/formatters';
import { PROJECT_TITLE, DATASET_NAME } from '../utils/constants';

export default function Dashboard() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { cases, stats, isLoading } = useCases();

  // Get top 5 recent cases
  const recentCases = cases.slice(0, 5);

  return (
    <div className="space-y-6">
      {/* Welcome & Research Overview Banner */}
      <div className="bg-white rounded-xl border border-slate-200/90 p-5 sm:p-6 shadow-2xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1.5 max-w-3xl">
            <div className="inline-flex items-center gap-1.5 text-xs font-semibold text-blue-700 bg-blue-50 px-2.5 py-0.5 rounded-full border border-blue-200">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Research Dashboard</span>
            </div>
            <h2 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
              Welcome back, {user?.name ? user.name.split(' ')[0] : 'Researcher'}
            </h2>
            <p className="text-sm text-slate-600 leading-relaxed">
              <strong>{PROJECT_TITLE}:</strong> An AI-assisted medical imaging platform for managing and analyzing 2D X-ray coronary angiography (XCA) scans using the <em>{DATASET_NAME}</em>.
            </p>
          </div>

          {/* Quick Action Buttons */}
          <div className="flex items-center gap-2.5 shrink-0">
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

      {/* Summary Stat Cards (Strictly derived from mock cases) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Cases */}
        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-2xs flex items-center justify-between">
          <div>
            <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">Total Cases</p>
            <p className="text-2xl font-bold text-slate-900 mt-1">{stats.total}</p>
            <p className="text-[11px] text-slate-400 mt-0.5">Active research cases</p>
          </div>
          <div className="w-10 h-10 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center border border-blue-100">
            <Layers className="w-5 h-5" />
          </div>
        </div>

        {/* Awaiting Processing */}
        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-2xs flex items-center justify-between">
          <div>
            <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">Awaiting Processing</p>
            <p className="text-2xl font-bold text-amber-700 mt-1">{stats.awaitingProcessing}</p>
            <p className="text-[11px] text-slate-400 mt-0.5">Pending upload or preprocessing</p>
          </div>
          <div className="w-10 h-10 rounded-lg bg-amber-50 text-amber-600 flex items-center justify-center border border-amber-100">
            <Clock className="w-5 h-5" />
          </div>
        </div>

        {/* Completed Processing Jobs */}
        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-2xs flex items-center justify-between">
          <div>
            <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">Completed Processing</p>
            <p className="text-2xl font-bold text-emerald-700 mt-1">{stats.completed}</p>
            <p className="text-[11px] text-slate-400 mt-0.5">Normalized scans prepared</p>
          </div>
          <div className="w-10 h-10 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center border border-emerald-100">
            <CheckCircle2 className="w-5 h-5" />
          </div>
        </div>

        {/* Failed Processing Jobs */}
        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-2xs flex items-center justify-between">
          <div>
            <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">Failed Jobs</p>
            <p className="text-2xl font-bold text-rose-700 mt-1">{stats.failed}</p>
            <p className="text-[11px] text-slate-400 mt-0.5">Header or quality errors</p>
          </div>
          <div className="w-10 h-10 rounded-lg bg-rose-50 text-rose-600 flex items-center justify-center border border-rose-100">
            <AlertTriangle className="w-5 h-5" />
          </div>
        </div>
      </div>

      {/* Recent Cases Section */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-2xs overflow-hidden">
        <div className="p-4 sm:p-5 border-b border-slate-200 flex items-center justify-between">
          <div>
            <h3 className="text-base font-semibold text-slate-900">Recent Research Cases</h3>
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

        {/* Table Content */}
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
                  <tr key={c.id} className="hover:bg-slate-50/70 transition-colors">
                    <td className="py-3.5 px-4 font-mono font-medium text-blue-600 text-xs">
                      {c.id}
                    </td>
                    <td className="py-3.5 px-4">
                      <div className="font-medium text-slate-900 truncate max-w-xs">{c.name}</div>
                      <div className="text-xs text-slate-400 truncate max-w-xs">{c.targetRegion}</div>
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
      </div>
    </div>
  );
}
