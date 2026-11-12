
import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  UserRound,
  Mail,
  LockKeyhole,
  Building2,
  UserPlus,
  Eye,
  EyeOff,
  ShieldCheck,
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
} from 'lucide-react';

import FormInput from '../components/FormInput';
import Button from '../components/Button';
import { useAuth } from '../context/AuthContext';
import { APP_NAME } from '../utils/constants';

const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export default function Register() {
  const navigate = useNavigate();
  const { register } = useAuth();

  const [formData, setFormData] = useState({
    fullName: '',
    email: '',
    institution: '',
    password: '',
    confirmPassword: '',
  });

  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [errors, setErrors] = useState({});
  const [serverError, setServerError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (event) => {
    const { name, value } = event.target;

    setFormData((previous) => ({
      ...previous,
      [name]: value,
    }));

    setErrors((previous) => ({
      ...previous,
      [name]: null,
    }));

    setServerError('');
  };

  const validate = () => {
    const nextErrors = {};

    if (formData.fullName.trim().length < 2) {
      nextErrors.fullName = 'Enter your full name.';
    }

    if (!emailPattern.test(formData.email.trim())) {
      nextErrors.email = 'Enter a valid email address.';
    }

    if (!formData.institution.trim()) {
      nextErrors.institution = 'Enter your institution or research organisation.';
    }

    if (formData.password.length < 8) {
      nextErrors.password = 'Use at least 8 characters for your password.';
    }

    if (formData.password !== formData.confirmPassword) {
      nextErrors.confirmPassword = 'The passwords do not match.';
    }

    setErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setServerError('');

    if (!validate()) return;

    try {
      setIsLoading(true);

      await register({
        fullName: formData.fullName.trim(),
        email: formData.email.trim(),
        institution: formData.institution.trim(),
        password: formData.password,
      });

      navigate('/dashboard', { replace: true });
    } catch (error) {
      setServerError(
        error.message ||
          'We could not create the account. Review your details and try again.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-xl shadow-slate-900/[0.06]">
      <div className="h-1.5 bg-gradient-to-r from-slate-900 via-teal-700 to-cyan-500" />

      <div className="p-6 sm:p-8">
        <Link
          to="/login"
          className="mb-6 inline-flex items-center gap-1.5 text-xs font-semibold text-slate-500 transition hover:text-teal-800"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Back to sign in
        </Link>

        <div className="mb-6">
          <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-teal-50 text-teal-800 ring-1 ring-teal-100">
            <UserPlus className="h-6 w-6" />
          </div>

          <p className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-teal-800">
            Researcher registration
          </p>

          <h1 className="text-2xl font-bold tracking-tight text-slate-950 sm:text-3xl">
            Create your account
          </h1>

          <p className="mt-2 text-sm leading-6 text-slate-500">
            Set up your profile to access the {APP_NAME} research workspace.
          </p>
        </div>

        {serverError && (
          <div
            role="alert"
            className="mb-5 flex items-start gap-3 rounded-xl border border-rose-200 bg-rose-50 p-3.5 text-sm text-rose-800"
          >
            <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />
            <p>{serverError}</p>
          </div>
        )}

        <form onSubmit={handleSubmit} noValidate className="space-y-4">
          <FormInput
            label="Full name"
            id="reg-name"
            name="fullName"
            type="text"
            placeholder="Enter your full name"
            icon={UserRound}
            value={formData.fullName}
            onChange={handleChange}
            error={errors.fullName}
            required
            autoComplete="name"
          />

          <FormInput
            label="Institutional or research email"
            id="reg-email"
            name="email"
            type="email"
            placeholder="you@institution.edu"
            icon={Mail}
            value={formData.email}
            onChange={handleChange}
            error={errors.email}
            required
            autoComplete="email"
          />

          <FormInput
            label="Institution / organisation"
            id="reg-institution"
            name="institution"
            type="text"
            placeholder="University, hospital or research institute"
            icon={Building2}
            value={formData.institution}
            onChange={handleChange}
            error={errors.institution}
            required
            autoComplete="organization"
          />

          <div className="space-y-1.5">
            <label
              htmlFor="reg-password"
              className="block text-xs font-semibold uppercase tracking-wide text-slate-700"
            >
              Password <span className="text-rose-500">*</span>
            </label>

            <div className="relative">
              <LockKeyhole className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />

              <input
                id="reg-password"
                name="password"
                type={showPassword ? 'text' : 'password'}
                value={formData.password}
                onChange={handleChange}
                placeholder="Create a strong password"
                autoComplete="new-password"
                required
                aria-invalid={Boolean(errors.password)}
                className={`w-full rounded-xl border bg-white py-3 pl-10 pr-12 text-sm text-slate-900 outline-none transition placeholder:text-slate-400 focus:ring-4 ${
                  errors.password
                    ? 'border-rose-400 focus:border-rose-500 focus:ring-rose-100'
                    : 'border-slate-300 focus:border-teal-600 focus:ring-teal-50'
                }`}
              />

              <button
                type="button"
                onClick={() => setShowPassword((value) => !value)}
                aria-label={showPassword ? 'Hide password' : 'Show password'}
                className="absolute right-3 top-1/2 -translate-y-1/2 rounded-md p-1 text-slate-500 hover:text-slate-900 focus-visible:ring-2 focus-visible:ring-teal-600"
              >
                {showPassword ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </button>
            </div>

            {errors.password ? (
              <p className="text-xs font-medium text-rose-600">
                {errors.password}
              </p>
            ) : (
              <p className="text-xs text-slate-500">
                Use at least 8 characters.
              </p>
            )}
          </div>

          <div className="space-y-1.5">
            <label
              htmlFor="reg-confirm-password"
              className="block text-xs font-semibold uppercase tracking-wide text-slate-700"
            >
              Confirm password <span className="text-rose-500">*</span>
            </label>

            <div className="relative">
              <LockKeyhole className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />

              <input
                id="reg-confirm-password"
                name="confirmPassword"
                type={showConfirmPassword ? 'text' : 'password'}
                value={formData.confirmPassword}
                onChange={handleChange}
                placeholder="Enter the password again"
                autoComplete="new-password"
                required
                aria-invalid={Boolean(errors.confirmPassword)}
                className={`w-full rounded-xl border bg-white py-3 pl-10 pr-12 text-sm text-slate-900 outline-none transition placeholder:text-slate-400 focus:ring-4 ${
                  errors.confirmPassword
                    ? 'border-rose-400 focus:border-rose-500 focus:ring-rose-100'
                    : 'border-slate-300 focus:border-teal-600 focus:ring-teal-50'
                }`}
              />

              <button
                type="button"
                onClick={() =>
                  setShowConfirmPassword((value) => !value)
                }
                aria-label={
                  showConfirmPassword
                    ? 'Hide confirmation password'
                    : 'Show confirmation password'
                }
                className="absolute right-3 top-1/2 -translate-y-1/2 rounded-md p-1 text-slate-500 hover:text-slate-900 focus-visible:ring-2 focus-visible:ring-teal-600"
              >
                {showConfirmPassword ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </button>
            </div>

            {errors.confirmPassword && (
              <p className="text-xs font-medium text-rose-600">
                {errors.confirmPassword}
              </p>
            )}
          </div>

          <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
            <div className="flex items-start gap-2.5">
              <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-teal-800" />
              <p className="text-xs leading-5 text-slate-600">
                This prototype is for research and educational use. Do not
                enter real patient information or confidential clinical data.
              </p>
            </div>
          </div>

          <Button
            type="submit"
            variant="primary"
            size="lg"
            className="w-full !rounded-xl !bg-slate-900 py-3 hover:!bg-slate-800 focus:!ring-slate-500"
            isLoading={isLoading}
            icon={CheckCircle2}
          >
            Create researcher account
          </Button>
        </form>

        <div className="mt-6 border-t border-slate-100 pt-5 text-center text-sm text-slate-600">
          Already registered?{' '}
          <Link
            to="/login"
            className="font-semibold text-teal-800 hover:text-teal-950 hover:underline"
          >
            Sign in
          </Link>
        </div>
      </div>
    </section>
  );
}
