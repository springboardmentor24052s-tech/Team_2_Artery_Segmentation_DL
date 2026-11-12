
import React from 'react';
import { Outlet, Link } from 'react-router-dom';
import { HeartPulse, ShieldCheck, Activity } from 'lucide-react';
import { APP_NAME, PROJECT_TITLE } from '../utils/constants';

function VesselBackdrop() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 800 600"
      preserveAspectRatio="xMaxYMid slice"
      className="pointer-events-none absolute inset-0 h-full w-full text-teal-800 opacity-[0.055]"
    >
      <g
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path
          strokeWidth="12"
          d="M820 20 C680 80 600 130 550 240 S430 440 280 510"
        />
        <path strokeWidth="7" d="M550 240 C610 280 680 300 780 360" />
        <path strokeWidth="5" d="M650 140 C690 180 730 190 800 185" />
        <path strokeWidth="5" d="M430 425 C370 390 330 330 275 310" />
        <path strokeWidth="4" d="M780 360 C755 410 770 465 800 500" />
        <path strokeWidth="3" d="M280 510 C220 530 170 565 100 590" />
      </g>
    </svg>
  );
}

export default function AuthLayout() {
  return (
    <div className="relative isolate flex min-h-screen flex-col overflow-hidden bg-slate-50 px-4 py-5 sm:px-6 sm:py-7 lg:px-10">
      <VesselBackdrop />

      <a
        href="#auth-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-white focus:px-4 focus:py-2 focus:shadow-lg"
      >
        Skip to content
      </a>

      <header className="relative z-10 mx-auto flex w-full max-w-7xl items-center justify-between gap-4">
        <Link
          to="/"
          aria-label={`${APP_NAME} home`}
          className="flex min-w-0 items-center gap-3 rounded-xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600"
        >
          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-slate-900 text-white shadow-lg shadow-slate-900/15">
            <HeartPulse className="h-7 w-7" aria-hidden="true" />
          </div>

          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xl font-bold tracking-tight text-slate-950">
                {APP_NAME}
              </span>
              <span className="rounded-full border border-teal-200 bg-teal-50 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-teal-800">
                Research Portal
              </span>
            </div>
            <p className="mt-0.5 text-xs text-slate-500 sm:text-sm">
              Coronary imaging research workspace
            </p>
          </div>
        </Link>

        <div className="hidden items-center gap-2 rounded-full border border-slate-200 bg-white/90 px-3 py-2 text-xs font-medium text-slate-600 shadow-sm sm:flex">
          <Activity className="h-4 w-4 text-teal-700" />
          <span>Research Workspace</span>
        </div>
      </header>

      <main
        id="auth-content"
        tabIndex={-1}
        className="relative z-10 flex flex-1 items-center justify-center py-8 sm:py-10"
      >
        <div className="w-full max-w-md">
          <Outlet />
        </div>
      </main>

      <footer className="relative z-10 mx-auto w-full max-w-4xl space-y-3 pt-2">
        <div
          role="note"
          className="flex items-start justify-center gap-2 rounded-xl border border-amber-200/80 bg-amber-50/90 px-4 py-3 text-xs leading-relaxed text-amber-950 sm:items-center"
        >
          <ShieldCheck
            className="mt-0.5 h-4 w-4 shrink-0 text-amber-700 sm:mt-0"
            aria-hidden="true"
          />
          <p>
            Research prototype for experimental validation only. Not approved
            for diagnostic or clinical decision-making.
          </p>
        </div>

        <p className="text-center text-[11px] leading-relaxed text-slate-500">
          {PROJECT_TITLE} · ARCADE Benchmark Framework
        </p>
      </footer>
    </div>
  );
}
