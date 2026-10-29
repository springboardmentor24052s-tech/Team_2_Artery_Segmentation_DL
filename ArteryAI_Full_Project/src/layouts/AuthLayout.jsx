import React from 'react';
import { Outlet, Link } from 'react-router-dom';
import { HeartPulse, ShieldAlert, Sparkles } from 'lucide-react';
import { APP_NAME, PROJECT_TITLE } from '../utils/constants';
import DemoBadge from '../components/DemoBadge';

export default function AuthLayout() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-blue-50/30 flex flex-col justify-between p-4 sm:p-6 lg:p-8">
      {/* Top bar with logo and demo indicator */}
      <header className="max-w-6xl w-full mx-auto flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2.5 group">
          <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center text-white shadow-md shadow-blue-500/20 group-hover:scale-105 transition-transform">
            <HeartPulse className="w-6 h-6 text-white" />
          </div>
          <div>
            <span className="font-bold text-lg text-slate-900 tracking-tight flex items-center gap-2">
              {APP_NAME}
              <span className="text-xs font-normal text-blue-600 bg-blue-50 px-2 py-0.5 rounded-full border border-blue-200">
                Research Portal
              </span>
            </span>
            <p className="text-xs text-slate-500">ARCADE X-ray Coronary Angiography</p>
          </div>
        </Link>

        <DemoBadge />
      </header>

      {/* Main card container */}
      <main className="my-auto py-8">
        <div className="max-w-md w-full mx-auto">
          <Outlet />
        </div>
      </main>

      {/* Footer disclaimers */}
      <footer className="max-w-3xl w-full mx-auto text-center space-y-2 pt-4 border-t border-slate-200/60">
        <div className="flex items-center justify-center gap-1.5 text-xs text-slate-500">
          <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0" />
          <span>
            Research prototype for experimental validation only. Not approved for diagnostic or clinical decision-making.
          </span>
        </div>
        <p className="text-[11px] text-slate-400">
          {PROJECT_TITLE} &bull; ARCADE Benchmark Framework
        </p>
      </footer>
    </div>
  );
}
