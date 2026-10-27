import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Cpu,
  CheckCircle2,
  Clock,
  AlertTriangle,
  FileX2,
  Sparkles,
  ArrowRight,
  Info,
  Layers,
  RefreshCw,
  Eye,
  SlidersHorizontal,
} from 'lucide-react';
import Button from '../components/Button';
import StatusBadge from '../components/StatusBadge';
import { useCases } from '../context/CaseContext';
import { PREPROCESSING_STAGES, CASE_STATUS } from '../utils/constants';

export default function PreprocessingStatus() {
  const navigate = useNavigate();
  const { cases, updateCase } = useCases();

  const [simulatingCaseId, setSimulatingCaseId] = useState(null);

  // Helper to count cases per stage
  const getCasesCountForStage = (stageKey) => {
    return cases.filter((c) => c.preprocessingStage === stageKey).length;
  };

  // Allow live interactive simulation of pipeline progression for demonstration!
  const handleSimulateNextStep = async (caseId, currentStage) => {
    setSimulatingCaseId(caseId);
    let nextStage = 'PREPROCESSING_COMPLETED';
    let nextStatus = CASE_STATUS.COMPLETED;

    if (currentStage === 'NOT_UPLOADED') {
      nextStage = 'AWAITING_PREPROCESSING';
      nextStatus = CASE_STATUS.UPLOADED;
    } else if (currentStage === 'AWAITING_PREPROCESSING') {
      nextStage = 'PREPROCESSING_IN_PROGRESS';
      nextStatus = CASE_STATUS.PROCESSING;
    } else if (currentStage === 'PREPROCESSING_IN_PROGRESS') {
      nextStage = 'PREPROCESSING_COMPLETED';
      nextStatus = CASE_STATUS.COMPLETED;
    } else if (currentStage === 'PREPROCESSING_FAILED') {
      nextStage = 'AWAITING_PREPROCESSING';
      nextStatus = CASE_STATUS.UPLOADED;
    }

    setTimeout(async () => {
      await updateCase(caseId, {
        preprocessingStage: nextStage,
        status: nextStatus,
      });
      setSimulatingCaseId(null);
    }, 600);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <div className="flex items-center gap-2 text-slate-500 text-xs font-semibold uppercase tracking-wider mb-1">
          <Cpu className="w-4 h-4 text-blue-600" />
          <span>Pipeline Monitor</span>
        </div>
        <h2 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
          Image Preprocessing Pipeline
        </h2>
        <p className="text-xs sm:text-sm text-slate-500 mt-0.5">
          Standardization, contrast adaptation (CLAHE), and frame validation stages for X-ray coronary angiography scans.
        </p>
      </div>


      {/* 5 Standard Preprocessing Stages Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {PREPROCESSING_STAGES.map((stage, idx) => {
          const count = getCasesCountForStage(stage.key);
          const iconMap = {
            NOT_UPLOADED: FileX2,
            AWAITING_PREPROCESSING: Clock,
            PREPROCESSING_IN_PROGRESS: RefreshCw,
            PREPROCESSING_COMPLETED: CheckCircle2,
            PREPROCESSING_FAILED: AlertTriangle,
          };
          const StageIcon = iconMap[stage.key] || Cpu;

          return (
            <div
              key={stage.key}
              className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs flex flex-col justify-between space-y-3"
            >
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono font-semibold text-slate-400">
                    Stage 0{idx + 1}
                  </span>
                  <span className={`text-[11px] font-semibold px-2 py-0.5 rounded-full border ${stage.statusClass}`}>
                    {stage.badgeText}
                  </span>
                </div>

                <div className="flex items-center gap-2 pt-1">
                  <div className="w-8 h-8 rounded-lg bg-slate-50 text-slate-700 flex items-center justify-center border border-slate-200 shrink-0">
                    <StageIcon className="w-4 h-4 text-blue-600" />
                  </div>
                  <h3 className="text-sm font-bold text-slate-900">{stage.title}</h3>
                </div>

                <p className="text-xs text-slate-600 leading-relaxed">
                  {stage.description}
                </p>
              </div>

              <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
                <span className="text-slate-500">
                  Active Cases:{' '}
                  <strong className="text-slate-900 font-semibold">{count}</strong>
                </span>
                <span className="text-[11px] text-slate-400">
                  {stage.key === 'PREPROCESSING_COMPLETED' ? 'Ready for Model' : 'In Progress'}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Case-by-case Pipeline Status Table */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-2xs overflow-hidden">
        <div className="p-4 sm:p-5 border-b border-slate-200 flex items-center justify-between">
          <div>
            <h3 className="text-base font-semibold text-slate-900">
              Live Case Preprocessing Log
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Review current preparation stage for each ARCADE study scan
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            icon={SlidersHorizontal}
            onClick={() => navigate('/upload')}
          >
            Upload New Scan
          </Button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-700">
            <thead className="bg-slate-50/80 text-[11px] uppercase tracking-wider font-semibold text-slate-500 border-b border-slate-200">
              <tr>
                <th scope="col" className="py-3 px-4">Case ID</th>
                <th scope="col" className="py-3 px-4">Case Name</th>
                <th scope="col" className="py-3 px-4">Current Pipeline Stage</th>
                <th scope="col" className="py-3 px-4">Attached Scan</th>
                <th scope="col" className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {cases.map((c) => {
                const stageObj = PREPROCESSING_STAGES.find((s) => s.key === c.preprocessingStage) || PREPROCESSING_STAGES[0];
                const isSimulating = simulatingCaseId === c.id;

                return (
                  <tr key={c.id} className="hover:bg-slate-50/70 transition-colors">
                    <td className="py-3.5 px-4 font-mono font-medium text-blue-600 text-xs">
                      {c.id}
                    </td>
                    <td className="py-3.5 px-4">
                      <div className="font-semibold text-slate-900 text-xs truncate max-w-xs">{c.name}</div>
                      <div className="text-[11px] text-slate-400">{c.targetRegion}</div>
                    </td>
                    <td className="py-3.5 px-4">
                      <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${stageObj.statusClass}`}>
                        <span>{stageObj.title}</span>
                      </span>
                    </td>
                    <td className="py-3.5 px-4 text-xs text-slate-600">
                      {c.file ? (
                        <span className="font-mono text-slate-700 text-[11px]">
                          {c.file.name}
                        </span>
                      ) : (
                        <span className="text-slate-400 italic text-xs">No file</span>
                      )}
                    </td>
                    <td className="py-3.5 px-4 text-right space-x-2">
                      {/* Stage processing trigger */}
                      <Button
                        variant="secondary"
                        size="sm"
                        isLoading={isSimulating}
                        onClick={() => handleSimulateNextStep(c.id, c.preprocessingStage)}
                        title="Process case to next pipeline stage"
                      >
                        Process Next Stage
                      </Button>
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
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
