import React from 'react';
import { Loader2 } from 'lucide-react';

export default function LoadingState({
  message = 'Loading data...',
  subtext = 'Please wait while we retrieve the research records.',
  className = '',
}) {
  return (
    <div className={`flex flex-col items-center justify-center py-12 px-4 text-center ${className}`}>
      <div className="w-12 h-12 rounded-full bg-blue-50 flex items-center justify-center text-blue-600 mb-3 border border-blue-100">
        <Loader2 className="w-6 h-6 animate-spin" />
      </div>
      <h3 className="text-sm font-semibold text-slate-800 mb-1">{message}</h3>
      {subtext && <p className="text-xs text-slate-500 max-w-sm">{subtext}</p>}
    </div>
  );
}
