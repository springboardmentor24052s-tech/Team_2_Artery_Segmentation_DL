import React, { useState } from 'react';
import { Menu, RotateCcw, User, LogOut, CheckCircle2 } from 'lucide-react';
import DemoBadge from './DemoBadge';
import { useAuth } from '../context/AuthContext';
import { useCases } from '../context/CaseContext';
import { APP_NAME } from '../utils/constants';

export default function Navbar({ onToggleSidebar }) {
  const { user, logout } = useAuth();
  const { resetCases } = useCases();
  const [resetToast, setResetToast] = useState(false);

  const handleReset = () => {
    resetCases();
    setResetToast(true);
    setTimeout(() => setResetToast(false), 3000);
  };

  return (
    <header className="sticky top-0 z-30 h-16 bg-[#111b29] border-b border-slate-700 px-4 sm:px-6 flex items-center justify-between shadow-2xs">
      {/* Left: Mobile hamburger + Project branding text */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={onToggleSidebar}
          className="lg:hidden p-2 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-slate-300"
          aria-label="Toggle navigation menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="hidden sm:block">
          <h1 className="text-sm font-semibold text-slate-100 leading-tight">
            Coronary Artery Segmentation &amp; Stenosis Research
          </h1>
          <p className="text-xs text-slate-500">
            ARCADE X-ray Coronary Angiography (XCA) Benchmark Platform
          </p>
        </div>
        <div className="sm:hidden font-semibold text-slate-100 text-sm">
          {APP_NAME}
        </div>
      </div>

      {/* Right: Demo indicator, reset mock button, user info & actions */}
      <div className="flex items-center gap-2 sm:gap-3">
        {/* Reset feedback toast notification */}
        {resetToast && (
          <div className="hidden md:flex items-center gap-1.5 text-xs text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-md border border-emerald-200 animate-in fade-in">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
            <span>Mock cases reset to default</span>
          </div>
        )}

        {/* Reset Mock Data button */}
        <button
          type="button"
          onClick={handleReset}
          title="Reset dataset back to benchmark cases"
          className="hidden md:inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-md transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span>Reset Sample Data</span>
        </button>

        {/* Demo Mode Badge */}
        <DemoBadge />

        {/* User Pill / Action */}
        <div className="flex items-center gap-2 pl-2 border-l border-slate-700">
          <div className="w-8 h-8 rounded-full bg-sky-400/10 text-sky-200 border border-sky-400/30 flex items-center justify-center text-xs font-semibold">
            <User className="w-4 h-4 text-sky-300" />
          </div>
          <div className="hidden xl:block text-left">
            <p className="text-xs font-medium text-slate-100 leading-none">
              {user?.name || 'Researcher'}
            </p>
            <p className="text-[10px] text-slate-400 mt-0.5 leading-none">
              Cardiovascular Research
            </p>
          </div>
        </div>
      </div>
    </header>
  );
}
