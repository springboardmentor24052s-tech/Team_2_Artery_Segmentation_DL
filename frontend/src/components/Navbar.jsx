
import React, { useState } from 'react';
import {
  Menu,
  RotateCcw,
  User,
  CheckCircle2,
  Sun,
  Moon,
  Palette,
  Check,
  ChevronDown,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { useCases } from '../context/CaseContext';
import { useTheme } from '../context/ThemeContext';
import { APP_NAME } from '../utils/constants';



export default function Navbar({ onToggleSidebar }) {
  const { user } = useAuth();
  const { resetCases } = useCases();
  const { theme, setTheme } = useTheme();

  const [resetToast, setResetToast] = useState(false);
  const [appearanceOpen, setAppearanceOpen] = useState(false);

  const isDark = theme === 'dark';

  const handleReset = () => {
    resetCases();
    setResetToast(true);
    window.setTimeout(() => setResetToast(false), 3000);
  };

  return (
    <header className="app-navbar sticky top-0 z-30 min-h-16 bg-white dark:bg-slate-950 border-b border-slate-200 dark:border-slate-800 px-4 sm:px-6 flex items-center justify-between gap-3 shadow-sm">
      {/* Left: Mobile menu and branding */}
      <div className="flex items-center gap-3 min-w-0">
        <button
          type="button"
          onClick={onToggleSidebar}
          className="lg:hidden p-2 rounded-lg text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-400"
          aria-label="Toggle navigation menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="hidden sm:block min-w-0">
          <h1 className="text-sm font-semibold text-slate-900 dark:text-slate-100 leading-tight">
            Coronary Artery Segmentation &amp; Stenosis Research
          </h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            ARCADE X-ray Coronary Angiography (XCA) Benchmark Platform
          </p>
        </div>

        <div className="sm:hidden font-semibold text-slate-900 dark:text-slate-100 text-sm">
          {APP_NAME}
        </div>
      </div>

      {/* Right: Appearance, reset feedback, and profile */}
      <div className="flex items-center gap-2 sm:gap-3">
        {resetToast && (
          <div
            role="status"
            className="hidden md:flex items-center gap-1.5 text-xs text-emerald-700 dark:text-emerald-300 bg-emerald-50 dark:bg-emerald-950 px-2.5 py-1.5 rounded-md border border-emerald-200 dark:border-emerald-800"
          >
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>Sample cases reset</span>
          </div>
        )}

        <button
          type="button"
          onClick={handleReset}
          title="Reset sample cases to the default benchmark data"
          className="hidden md:inline-flex items-center gap-1.5 px-2.5 py-2 text-xs font-medium text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white bg-slate-50 dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span>Reset Sample Data</span>
        </button>

        {/* Appearance settings */}
        <div className="relative">
          <button
            type="button"
            onClick={() => setAppearanceOpen((open) => !open)}
            aria-expanded={appearanceOpen}
            aria-haspopup="dialog"
            aria-label="Appearance settings"
            title="Customize appearance"
            className="inline-flex items-center gap-2 p-2 sm:px-3 sm:py-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-400"
          >
            <Palette className="w-4 h-4" />
            <span className="hidden sm:inline text-xs font-medium">
              Appearance
            </span>
            <ChevronDown className="hidden sm:block w-3.5 h-3.5" />
          </button>

          {appearanceOpen && (
            <>
              <button
                type="button"
                aria-label="Close appearance settings"
                className="fixed inset-0 z-40 cursor-default"
                onClick={() => setAppearanceOpen(false)}
              />

              <div
                role="dialog"
                aria-label="Appearance settings"
                className="absolute right-0 top-full mt-3 z-50 w-72 p-4 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl"
              >
                <div className="mb-4">
                  <h2 className="text-sm font-semibold text-slate-900 dark:text-white">
                    Appearance
                  </h2>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                    Personalize your research workspace.
                  </p>
                </div>

                {/* Complete workspace theme presets */}
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-2">
                  Workspace theme
                </p>
                <div className="space-y-2">
                  {[
                    { id: 'light', label: 'Clinical White', description: 'Pure white workspace and crisp white cards', icon: Sun },
                    { id: 'blue', label: 'Clinical Blue', description: 'Clearly blue workspace with white cards', icon: Palette },
                    { id: 'dark', label: 'Midnight Navy', description: 'Near-black workspace and charcoal panels', icon: Moon },
                  ].map((option) => {
                    const Icon = option.icon;
                    const selected = theme === option.id;
                    return (
                      <button
                        key={option.id}
                        type="button"
                        onClick={() => setTheme(option.id)}
                        aria-pressed={selected}
                        className={`w-full flex items-center gap-3 rounded-lg border p-3 text-left transition-colors ${
                          selected
                            ? 'border-teal-600 bg-teal-50 text-slate-900 dark:border-teal-400 dark:bg-teal-950 dark:text-white'
                            : 'border-slate-200 text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800'
                        }`}
                      >
                        <Icon className="w-4 h-4 shrink-0" />
                        <span className="min-w-0 flex-1">
                          <span className="block text-sm font-semibold">{option.label}</span>
                          <span className="block text-xs opacity-75">{option.description}</span>
                        </span>
                        {selected && <Check className="w-4 h-4 shrink-0" />}
                      </button>
                    );
                  })}
                </div>

                <p className="mt-4 text-[11px] leading-relaxed text-slate-500 dark:text-slate-400">
                  Your appearance preferences are saved in this browser.
                </p>
              </div>
            </>
          )}
        </div>

        {/* User profile */}
        <div className="flex items-center gap-2 pl-2 border-l border-slate-200 dark:border-slate-700">
          <div className="w-8 h-8 shrink-0 rounded-full bg-blue-50 dark:bg-blue-950 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800 flex items-center justify-center text-xs font-semibold">
            <User className="w-4 h-4" />
          </div>

          <div className="hidden xl:block text-left">
            <p className="text-xs font-medium text-slate-800 dark:text-slate-100 leading-none">
              {user?.name || 'Researcher'}
            </p>
            <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1 leading-none">
              Cardiovascular Research
            </p>
          </div>
        </div>
      </div>
    </header>
  );
}
