import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from '../components/Sidebar';
import Navbar from '../components/Navbar';

export default function DashboardLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen bg-[#0b1220] text-slate-100">
      {/* Sidebar navigation */}
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      {/* Main content wrapper */}
      <div className="lg:pl-64 flex flex-col min-h-screen">
        <Navbar onToggleSidebar={() => setSidebarOpen(!sidebarOpen)} />

        <main className="flex-1 w-full min-w-0 p-0">
          <Outlet />
        </main>

        <footer className="py-4 px-6 border-t border-slate-800 text-center text-xs text-slate-400 bg-[#0b1220]">
          <div className="w-full flex flex-col sm:flex-row items-center justify-between gap-2">
            <span>
              ArteryAI &bull; Coronary Artery Segmentation &amp; Stenosis Detection Platform
            </span>
            <span className="text-slate-500">
              ARCADE Dataset Research &bull; Milestone 1 Frontend Architecture
            </span>
          </div>
        </footer>
      </div>
    </div>
  );
}
