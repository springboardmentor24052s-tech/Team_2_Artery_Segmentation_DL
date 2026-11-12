
import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import AuthLayout from './layouts/AuthLayout';
import DashboardLayout from './layouts/DashboardLayout';
import ProtectedRoute from './components/ProtectedRoute';
import { ThemeProvider } from './context/ThemeContext';

import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import CreateCase from './pages/CreateCase';
import CaseHistory from './pages/CaseHistory';
import CaseDetails from './pages/CaseDetails';
import UploadScan from './pages/UploadScan';
import PreprocessingStatus from './pages/PreprocessingStatus';
import NotFound from './pages/NotFound';

export default function App() {
  return (
    <ThemeProvider>
      <Routes>
        {/* Authentication Routes */}
        <Route element={<AuthLayout />}>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
        </Route>

        {/* Main Research Application Routes */}
        <Route
          element={
            <ProtectedRoute>
              <DashboardLayout />
            </ProtectedRoute>
          }
        >
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/cases" element={<CaseHistory />} />
          <Route path="/cases/create" element={<CreateCase />} />
          <Route path="/cases/:id" element={<CaseDetails />} />
          <Route path="/upload" element={<UploadScan />} />
          <Route path="/preprocessing" element={<PreprocessingStatus />} />
        </Route>

        {/* 404 Catch-All Route */}
        <Route path="*" element={<NotFound />} />
      </Routes>
    </ThemeProvider>
  );
}
