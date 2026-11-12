import React from 'react';
import { STATUS_STYLES, CASE_STATUS } from '../utils/constants';

export default function StatusBadge({ status, className = '' }) {
  const currentStatus = status || CASE_STATUS.CREATED;
  const config = STATUS_STYLES[currentStatus] || {
    bg: 'bg-slate-100',
    text: 'text-slate-700',
    border: 'border-slate-200',
    dot: 'bg-slate-400',
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium border ${config.bg} ${config.text} ${config.border} ${className}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${config.dot}`} />
      <span>{currentStatus}</span>
    </span>
  );
}
