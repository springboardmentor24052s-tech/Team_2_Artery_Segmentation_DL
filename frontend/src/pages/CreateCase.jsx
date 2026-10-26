import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  FolderPlus,
  ArrowLeft,
  Save,
  Info,
  CheckCircle2,
  AlertCircle,
  FileText,
  Activity,
  Heart,
} from 'lucide-react';
import FormInput from '../components/FormInput';
import Button from '../components/Button';
import { useCases } from '../context/CaseContext';
import { IMAGE_MODALITIES, TARGET_REGIONS } from '../utils/constants';
import { generateSyntheticCaseId } from '../utils/formatters';

export default function CreateCase() {
  const navigate = useNavigate();
  const { cases, createCase } = useCases();

  const proposedCaseId = generateSyntheticCaseId(cases);

  const [formData, setFormData] = useState({
    name: '',
    description: '',
    targetRegion: TARGET_REGIONS[0],
    imageModality: IMAGE_MODALITIES[0].label,
    imageType: '2D XCA Frame',
    notes: '',
  });

  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => {
      const updated = { ...prev, [name]: value };
      if (name === 'imageModality') {
        if (value.includes('DICOM Cine')) {
          updated.imageType = 'DICOM Cine';
        } else if (value.includes('DSA')) {
          updated.imageType = '2D DSA';
        } else {
          updated.imageType = '2D XCA Frame';
        }
      }
      return updated;
    });

    if (errors[name]) {
      setErrors((prev) => ({ ...prev, [name]: null }));
    }
  };

  const validate = () => {
    const newErrors = {};
    if (!formData.name.trim()) {
      newErrors.name = 'Case name or study identifier is required.';
    } else if (formData.name.trim().length < 4) {
      newErrors.name = 'Case name should be at least 4 characters for research traceability.';
    }

    if (!formData.imageModality) {
      newErrors.imageModality = 'Please select a valid angiography image modality.';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validate()) return;

    try {
      setIsSubmitting(true);
      const created = await createCase({
        ...formData,
      });

      setSuccessMessage(`Research Case ${created.id} registered successfully!`);
      setTimeout(() => {
        navigate(`/cases/${created.id}`);
      }, 900);
    } catch (err) {
      setErrors((prev) => ({ ...prev, submit: err.message || 'Failed to create case.' }));
      setIsSubmitting(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* Back button and page title */}
      <div className="flex items-center justify-between">
        <button
          type="button"
          onClick={() => navigate(-1)}
          className="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-500 hover:text-slate-800 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to previous page</span>
        </button>
      </div>

      <div className="bg-white rounded-xl border border-slate-200/90 shadow-2xs p-6 sm:p-8">
        {/* Title & Research Guidelines */}
        <div className="mb-6 pb-5 border-b border-slate-100">
          <div className="flex items-center gap-2.5 text-blue-600 mb-1.5">
            <FolderPlus className="w-5 h-5" />
            <span className="text-xs font-semibold uppercase tracking-wider text-blue-700">
              Case Registration
            </span>
          </div>
          <h2 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
            Create Research Study Case
          </h2>
          <p className="text-xs sm:text-sm text-slate-500 mt-1 leading-relaxed">
            Register a synthetic research study container for coronary artery angiography analysis before associating scan files.
          </p>
        </div>

        {/* Study Identifier Tag */}
        <div className="mb-6 p-3.5 rounded-lg bg-slate-50 border border-slate-200 flex items-center justify-between text-xs text-slate-700">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-600">Assigned Study Identifier:</span>
            <span className="font-mono font-bold text-blue-700 bg-white px-2.5 py-1 rounded border border-slate-200">{proposedCaseId}</span>
          </div>
          <span className="text-[11px] text-slate-400 hidden sm:inline">ARCADE Study Sequence</span>
        </div>

        {successMessage && (
          <div className="mb-6 p-3.5 rounded-lg bg-emerald-50 border border-emerald-200 text-xs text-emerald-800 flex items-center gap-2 animate-in fade-in">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
            <span className="font-medium">{successMessage}</span>
          </div>
        )}

        {errors.submit && (
          <div className="mb-6 p-3.5 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-700 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-500 shrink-0" />
            <span>{errors.submit}</span>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-5">
          <FormInput
            label="Case Study Name / Identifier"
            id="case-name"
            name="name"
            placeholder="e.g. ARCADE Study #072 - Proximal LAD Stenosis Assessment"
            value={formData.name}
            onChange={handleChange}
            error={errors.name}
            helperText="A descriptive title for your research evaluation."
            required
            icon={FileText}
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Target Vessel Region */}
            <div className="space-y-1.5">
              <label htmlFor="target-region" className="block text-xs font-semibold text-slate-700 uppercase tracking-wide">
                Target Coronary Vessel <span className="text-rose-500">*</span>
              </label>
              <div className="relative">
                <select
                  id="target-region"
                  name="targetRegion"
                  value={formData.targetRegion}
                  onChange={handleChange}
                  className="w-full rounded-lg border border-slate-300 text-sm py-2.5 px-3 bg-white text-slate-800 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
                >
                  {TARGET_REGIONS.map((region) => (
                    <option key={region} value={region}>
                      {region}
                    </option>
                  ))}
                </select>
              </div>
              <p className="text-[11px] text-slate-500">Coronary branch investigated in this sequence</p>
            </div>

            {/* Image Modality (Strictly ARCADE 2D XCA, no 3D CT) */}
            <div className="space-y-1.5">
              <label htmlFor="image-modality" className="block text-xs font-semibold text-slate-700 uppercase tracking-wide">
                Angiography Modality <span className="text-rose-500">*</span>
              </label>
              <div className="relative">
                <select
                  id="image-modality"
                  name="imageModality"
                  value={formData.imageModality}
                  onChange={handleChange}
                  className="w-full rounded-lg border border-slate-300 text-sm py-2.5 px-3 bg-white text-slate-800 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
                >
                  {IMAGE_MODALITIES.map((modality) => (
                    <option key={modality.id} value={modality.label}>
                      {modality.label}
                    </option>
                  ))}
                </select>
              </div>
              <p className="text-[11px] text-slate-500">ARCADE uses 2D X-ray fluoroscopic angiography</p>
            </div>
          </div>

          <FormInput
            label="Study Description (Optional)"
            id="case-description"
            name="description"
            as="textarea"
            rows={3}
            placeholder="Document experimental rationale, projection angles (e.g. RAO 30° / Cranial 25°), or specific research goals..."
            value={formData.description}
            onChange={handleChange}
            helperText="Optional notes on projection angle or imaging protocol."
          />

          <FormInput
            label="Internal Notes / Research Tags (Optional)"
            id="case-notes"
            name="notes"
            placeholder="e.g. Baseline control frame, multi-observer evaluation, ARCADE validation fold"
            value={formData.notes}
            onChange={handleChange}
          />

          {/* Form Actions */}
          <div className="flex items-center justify-end gap-3 pt-4 border-t border-slate-100">
            <Button
              type="button"
              variant="outline"
              size="md"
              onClick={() => navigate('/cases')}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              variant="primary"
              size="md"
              icon={Save}
              isLoading={isSubmitting}
            >
              Save Research Case
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
