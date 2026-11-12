import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  FolderPlus,
  History,
  UploadCloud,
  Cpu,
  HeartPulse,
  LogOut,
  X,
  FileCode2,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { APP_NAME } from '../utils/constants';

const navItems = [
  {
    name: 'Dashboard',
    path: '/dashboard',
    icon: LayoutDashboard,
    description: 'Overview and recent activity'
  },
  {
    name: 'Create Case',
    path: '/cases/create',
    icon: FolderPlus,
    description: 'Register a new research study'
  },
  {
    name: 'Case History',
    path: '/cases',
    icon: History,
    description: 'Browse and filter all cases'
  },
  {
    name: 'Upload Scan',
    path: '/upload',
    icon: UploadCloud,
    description: 'Add X-ray angiography scans'
  },
  {
    name: 'Preprocessing Status',
    path: '/preprocessing',
    icon: Cpu,
    description: 'Image preparation pipeline'
  },
];

export default function Sidebar({ isOpen, onClose }) {
  const { user, logout } = useAuth();

  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          onClick={onClose}
          className="fixed inset-0 z-40 bg-slate-900/50 backdrop-blur-xs lg:hidden"
          aria-hidden="true"
        />
      )}

      {/* Sidebar container */}
      <aside
        className={`fixed top-0 bottom-0 left-0 z-40 w-64 bg-slate-900 text-slate-100 flex flex-col border-r border-slate-800 transition-transform duration-200 ease-in-out lg:translate-x-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Brand Header */}
        <div className="h-16 flex items-center justify-between px-5 border-b border-slate-800/80 bg-slate-950/40">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center text-white shadow-md shadow-blue-500/20">
              <HeartPulse className="w-5 h-5 text-white" />
            </div>
            <div>
              <span className="font-bold text-base tracking-tight text-white flex items-center gap-1.5">
                {APP_NAME}
                <span className="text-[10px] font-medium bg-blue-500/20 text-blue-300 px-1.5 py-0.5 rounded border border-blue-400/30">
                  XCA
                </span>
              </span>
              <p className="text-[11px] text-slate-400 font-normal truncate max-w-[140px]">
                Angiography Research
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="lg:hidden text-slate-400 hover:text-white p-1 rounded-md"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Dataset Context Callout */}
        <div className="px-4 pt-3 pb-1">
          <div className="px-3 py-2 rounded-lg bg-slate-800/60 border border-slate-700/60 text-xs">
            <p className="text-slate-400 text-[10px] uppercase font-semibold tracking-wider">Research Dataset</p>
            <p className="text-slate-200 font-medium text-xs truncate">ARCADE 2D Angiography</p>
          </div>
        </div>

        {/* Navigation Links */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          <div className="px-3 pb-2 text-[10px] font-bold uppercase tracking-wider text-slate-400">
            Platform Navigation
          </div>
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                onClick={() => {
                  if (window.innerWidth < 1024) onClose();
                }}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all group ${
                    isActive
                      ? 'bg-blue-600 text-white shadow-sm font-semibold'
                      : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    <Icon
                      className={`w-4 h-4 transition-transform group-hover:scale-105 shrink-0 ${
                        isActive ? 'text-white' : 'text-slate-400 group-hover:text-blue-400'
                      }`}
                    />
                    <span className="truncate">{item.name}</span>
                  </>
                )}
              </NavLink>
            );
          })}
        </nav>

        {/* User Session & Logout Footer */}
        <div className="p-3 border-t border-slate-800 bg-slate-950/30">
          <div className="flex items-center justify-between p-2 rounded-lg bg-slate-800/40">
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="w-8 h-8 rounded-full bg-blue-900/60 border border-blue-500/30 text-blue-300 flex items-center justify-center font-bold text-xs shrink-0">
                {user?.name ? user.name.charAt(0).toUpperCase() : 'R'}
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-xs font-medium text-slate-200 truncate">
                  {user?.name || 'Researcher'}
                </p>
                <p className="text-[11px] text-slate-400 truncate">
                  {user?.email || 'researcher@arteryai.org'}
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={logout}
              title="Sign out of demo session"
              className="text-slate-400 hover:text-rose-400 p-1.5 rounded-md hover:bg-slate-800 transition-colors shrink-0"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}
