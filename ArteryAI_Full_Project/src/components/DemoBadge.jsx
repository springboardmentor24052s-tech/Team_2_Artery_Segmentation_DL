import React, { useState } from 'react';
import { Info, Sparkles, X } from 'lucide-react';
import { IS_MOCK_MODE } from '../services/api';

export default function DemoBadge({ className = '' }) {
  const [showModal, setShowModal] = useState(false);

  if (!IS_MOCK_MODE) return null;

  return (
    <>
      <button
        type="button"
        onClick={() => setShowModal(true)}
        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-amber-50 text-amber-800 border border-amber-200 hover:bg-amber-100 transition-colors shadow-xs ${className}`}
        title="Running in Demo Mode with synthetic research data. Click for details."
      >
        <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
        <span>Demo Mode</span>
        <Info className="w-3 h-3 text-amber-600 ml-0.5" />
      </button>

      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs">
          <div className="bg-white rounded-xl shadow-xl max-w-md w-full p-6 border border-slate-200 animate-in fade-in zoom-in-95 duration-150">
            <div className="flex items-start justify-between gap-3 mb-4">
              <div className="flex items-center gap-2 text-amber-700 font-semibold">
                <Sparkles className="w-5 h-5 text-amber-600" />
                <span>Demo Mode Active</span>
              </div>
              <button
                type="button"
                onClick={() => setShowModal(false)}
                className="text-slate-400 hover:text-slate-600 rounded-lg p-1 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <p className="text-sm text-slate-600 mb-4 leading-relaxed">
              The application is currently operating using a client-side mock data layer for development and demonstration:
            </p>

            <ul className="text-xs text-slate-600 space-y-2 mb-5 list-disc pl-5">
              <li>
                <strong className="text-slate-800">Synthetic Data:</strong> Case IDs and records are synthetic representations based on the ARCADE X-ray coronary angiography benchmark. No real patient data is used.
              </li>
              <li>
                <strong className="text-slate-800">Browser Storage:</strong> Newly created cases and simulated uploads are stored in your browser session for live demonstration without page refresh.
              </li>
              <li>
                <strong className="text-slate-800">No AI Diagnosis:</strong> Automated deep learning segmentation and stenosis percentages are reserved for future backend milestones.
              </li>
            </ul>

            <div className="flex justify-end">
              <button
                type="button"
                onClick={() => setShowModal(false)}
                className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-lg text-sm font-medium transition-colors"
              >
                Understood
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
