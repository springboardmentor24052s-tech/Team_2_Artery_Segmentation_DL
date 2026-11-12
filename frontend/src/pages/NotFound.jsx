import React from 'react';
import { useNavigate } from 'react-router-dom';
import { FileQuestion, Home, ArrowLeft } from 'lucide-react';
import Button from '../components/Button';

export default function NotFound() {
  const navigate = useNavigate();

  return (
    <div className="min-h-[70vh] flex flex-col items-center justify-center p-6 text-center">
      <div className="w-16 h-16 rounded-2xl bg-blue-50 text-blue-600 flex items-center justify-center mb-4 border border-blue-100 shadow-2xs">
        <FileQuestion className="w-8 h-8" />
      </div>
      <span className="font-mono text-xs font-bold text-blue-600 bg-blue-50 px-2.5 py-1 rounded-full mb-2">
        Error 404
      </span>
      <h2 className="text-2xl font-bold text-slate-900 tracking-tight">
        Page Not Found
      </h2>
      <p className="text-sm text-slate-500 max-w-md mt-2 mb-6 leading-relaxed">
        The research portal route you navigated to does not exist or has been relocated in this milestone.
      </p>

      <div className="flex items-center gap-3">
        <Button
          variant="outline"
          size="md"
          icon={ArrowLeft}
          onClick={() => navigate(-1)}
        >
          Go Back
        </Button>
        <Button
          variant="primary"
          size="md"
          icon={Home}
          onClick={() => navigate('/dashboard')}
        >
          Return to Dashboard
        </Button>
      </div>
    </div>
  );
}
