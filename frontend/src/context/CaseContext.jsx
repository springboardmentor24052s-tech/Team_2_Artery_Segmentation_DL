import React, { createContext, useContext, useState, useEffect, useMemo, useCallback } from 'react';
import { caseService } from '../services/caseService';
import { uploadService } from '../services/uploadService';
import { CASE_STATUS } from '../utils/constants';

const CaseContext = createContext(null);

export function CaseProvider({ children }) {
  const [cases, setCases] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchCases = useCallback(async (filters = {}) => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await caseService.getCases(filters);
      setCases(data);
    } catch (err) {
      setError(err.message || 'Failed to fetch research cases');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCases();
  }, [fetchCases]);

  const createCase = async (caseData) => {
    const newCase = await caseService.createCase(caseData);
    setCases((prev) => [newCase, ...prev]);
    return newCase;
  };

  const updateCase = async (id, updates) => {
    const updated = await caseService.updateCase(id, updates);
    setCases((prev) => prev.map((c) => (c.id === id ? updated : c)));
    return updated;
  };

  const uploadCaseScan = async (caseId, file, onProgress) => {
    const result = await uploadService.uploadScan(caseId, file, onProgress);
    // Refresh cases from storage to stay completely in sync
    const freshCases = await caseService.getCases();
    setCases(freshCases);
    return result;
  };

  const resetCases = () => {
    const initial = caseService.resetMockData();
    setCases(initial);
  };

  // Derive all statistics strictly from the current mock cases
  const stats = useMemo(() => {
    const total = cases.length;
    const awaitingProcessing = cases.filter(
      (c) =>
        c.status === CASE_STATUS.CREATED ||
        c.status === CASE_STATUS.AWAITING_UPLOAD ||
        c.status === CASE_STATUS.UPLOADED
    ).length;
    const processing = cases.filter((c) => c.status === CASE_STATUS.PROCESSING).length;
    const completed = cases.filter((c) => c.status === CASE_STATUS.COMPLETED).length;
    const failed = cases.filter((c) => c.status === CASE_STATUS.FAILED).length;

    return {
      total,
      awaitingProcessing,
      processing,
      completed,
      failed,
    };
  }, [cases]);

  return (
    <CaseContext.Provider
      value={{
        cases,
        stats,
        isLoading,
        error,
        fetchCases,
        createCase,
        updateCase,
        uploadCaseScan,
        resetCases,
      }}
    >
      {children}
    </CaseContext.Provider>
  );
}

export function useCases() {
  const context = useContext(CaseContext);
  if (!context) {
    throw new Error('useCases must be used within a CaseProvider');
  }
  return context;
}
