
import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Mail,
  LockKeyhole,
  ArrowRight,
  ShieldCheck,
  Eye,
  EyeOff,
  KeyRound,
  X,
  AlertCircle,
  CheckCircle2,
  Activity,
} from 'lucide-react';

import FormInput from '../components/FormInput';
import Button from '../components/Button';
import { useAuth } from '../context/AuthContext';
import { authService } from '../services/authService';
import { APP_NAME } from '../utils/constants';

const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export default function Login() {
  const navigate = useNavigate();
  const { login } = useAuth();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const [errors, setErrors] = useState({});
  const [serverError, setServerError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const [showForgotModal, setShowForgotModal] = useState(false);
  const [forgotEmail, setForgotEmail] = useState('');
  const [forgotError, setForgotError] = useState('');
  const [forgotMessage, setForgotMessage] = useState('');
  const [forgotLoading, setForgotLoading] = useState(false);

  useEffect(() => {
    const savedEmail = localStorage.getItem('arteryai_remember_email');

    if (savedEmail) {
      setEmail(savedEmail);
      setRememberMe(true);
    }
  }, []);

  const validate = () => {
    const nextErrors = {};

    if (!email.trim()) {
      nextErrors.email = 'Enter your email address.';
    } else if (!emailPattern.test(email.trim())) {
      nextErrors.email = 'Enter a valid email address.';
    }

    if (!password) {
      nextErrors.password = 'Enter your password.';
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

      await login({
        email: email.trim(),
        password,
        rememberMe,
      });

      navigate('/dashboard', { replace: true });
    } catch (error) {
      setServerError(
        error.message || 'Unable to sign in. Check your details and try again.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  const openForgotModal = () => {
    setForgotEmail(email);
    setForgotError('');
    setForgotMessage('');
    setShowForgotModal(true);
  };

  const handleForgotSubmit = async (event) => {
    event.preventDefault();
    setForgotError('');
    setForgotMessage('');

    if (!emailPattern.test(forgotEmail.trim())) {
      setForgotError('Enter a valid email address.');
      return;
    }

    try {
      setForgotLoading(true);

      const result = await authService.requestPasswordReset(
        forgotEmail.trim()
      );

      setForgotMessage(
        result.message ||
          'If the account is eligible, follow the password reset instructions.'
      );
    } catch (error) {
      setForgotError(
        error.message || 'Unable to process the password reset request.'
      );
    } finally {
      setForgotLoading(false);
    }
  };

  return (
    <>
      <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-xl shadow-slate-900/[0.06]">
        <div className="h-1.5 bg-gradient-to-r from-slate-900 via-teal-700 to-cyan-500" />

        <div className="p-6 sm:p-9">
          <div className="mb-7">
            <div className="mb-5 flex h-12 w-12 items-center justify-center rounded-2xl bg-teal-50 text-teal-800 ring-1 ring-teal-100">
              <ShieldCheck className="h-6 w-6" />
            </div>

            <p className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-teal-800">
              Researcher access
            </p>

            <h1 className="text-2xl font-bold tracking-tight text-slate-950 sm:text-3xl">
              Welcome back
            </h1>

            <p className="mt-2 text-sm leading-6 text-slate-500">
              Sign in to access your {APP_NAME} research workspace and case
              records.
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

          <form onSubmit={handleSubmit} noValidate className="space-y-5">
            <FormInput
              label="Email address"
              id="login-email"
              name="email"
              type="email"
              placeholder="you@institution.edu"
              icon={Mail}
              value={email}
              onChange={(event) => {
                setEmail(event.target.value);
                setServerError('');
                setErrors((previous) => ({
                  ...previous,
                  email: null,
                }));
              }}
              error={errors.email}
              required
              autoComplete="username"
            />

            <div className="space-y-1.5">
              <div className="flex items-center justify-between gap-3">
                <label
                  htmlFor="login-password"
                  className="block text-xs font-semibold uppercase tracking-wide text-slate-700"
                >
                  Password <span className="text-rose-500">*</span>
                </label>

                <button
                  type="button"
                  onClick={openForgotModal}
                  className="text-xs font-semibold text-teal-800 hover:text-teal-950 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600"
                >
                  Forgot password?
                </button>
              </div>

              <div className="relative">
                <LockKeyhole
                  aria-hidden="true"
                  className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
                />

                <input
                  id="login-password"
                  name="password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(event) => {
                    setPassword(event.target.value);
                    setServerError('');
                    setErrors((previous) => ({
                      ...previous,
                      password: null,
                    }));
                  }}
                  placeholder="Enter your password"
                  autoComplete="current-password"
                  required
                  aria-invalid={Boolean(errors.password)}
                  aria-describedby={
                    errors.password ? 'login-password-error' : undefined
                  }
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
                  className="absolute right-3 top-1/2 -translate-y-1/2 rounded-md p-1 text-slate-500 hover:text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600"
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </button>
              </div>

              {errors.password && (
                <p
                  id="login-password-error"
                  className="text-xs font-medium text-rose-600"
                >
                  {errors.password}
                </p>
              )}
            </div>

            <label className="flex cursor-pointer items-center gap-2.5 text-sm text-slate-600">
              <input
                type="checkbox"
                checked={rememberMe}
                onChange={(event) => setRememberMe(event.target.checked)}
                className="h-4 w-4 rounded border-slate-300 text-teal-700 focus:ring-teal-600"
              />
              Remember my email on this device
            </label>

            <Button
              type="submit"
              variant="primary"
              size="lg"
              className="w-full !rounded-xl !bg-slate-900 py-3 hover:!bg-slate-800 focus:!ring-slate-500"
              isLoading={isLoading}
              icon={ArrowRight}
            >
              Sign in to workspace
            </Button>
          </form>

          <div className="my-6 flex items-center gap-3">
            <div className="h-px flex-1 bg-slate-200" />
            <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">
              Research access
            </span>
            <div className="h-px flex-1 bg-slate-200" />
          </div>

          <p className="text-center text-sm text-slate-600">
            New to the research workspace?{' '}
            <Link
              to="/register"
              className="font-semibold text-teal-800 hover:text-teal-950 hover:underline"
            >
              Create an account
            </Link>
          </p>
        </div>

        <div className="flex items-start gap-2.5 border-t border-slate-100 bg-slate-50/80 px-6 py-4 sm:px-9">
          <Activity className="mt-0.5 h-4 w-4 shrink-0 text-teal-700" />
          <p className="text-xs leading-5 text-slate-500">
            Access is intended for research and experimental validation. Do not
            submit real patient information to this demo application.
          </p>
        </div>
      </section>

      {showForgotModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/50 p-4 backdrop-blur-sm"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              setShowForgotModal(false);
            }
          }}
        >
          <section
            role="dialog"
            aria-modal="true"
            aria-labelledby="forgot-title"
            className="w-full max-w-md rounded-2xl border border-slate-200 bg-white p-6 shadow-2xl sm:p-7"
          >
            <div className="mb-5 flex items-start justify-between gap-3">
              <div className="flex gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-teal-50 text-teal-800">
                  <KeyRound className="h-5 w-5" />
                </div>

                <div>
                  <h2
                    id="forgot-title"
                    className="text-lg font-bold text-slate-950"
                  >
                    Reset password
                  </h2>
                  <p className="mt-1 text-xs leading-5 text-slate-500">
                    Request password reset instructions.
                  </p>
                </div>
              </div>

              <button
                type="button"
                onClick={() => setShowForgotModal(false)}
                aria-label="Close password reset dialog"
                className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-800"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {forgotMessage ? (
              <div
                role="status"
                className="rounded-xl border border-teal-200 bg-teal-50 p-4"
              >
                <CheckCircle2 className="mb-2 h-6 w-6 text-teal-800" />
                <p className="text-sm leading-6 text-teal-950">
                  {forgotMessage}
                </p>
                <Button
                  type="button"
                  className="mt-4 w-full"
                  onClick={() => setShowForgotModal(false)}
                >
                  Close
                </Button>
              </div>
            ) : (
              <form onSubmit={handleForgotSubmit} className="space-y-4">
                <FormInput
                  label="Email address"
                  id="forgot-email"
                  name="forgotEmail"
                  type="email"
                  icon={Mail}
                  value={forgotEmail}
                  onChange={(event) => {
                    setForgotEmail(event.target.value);
                    setForgotError('');
                  }}
                  placeholder="you@institution.edu"
                  required
                  autoComplete="email"
                />

                {forgotError && (
                  <p role="alert" className="text-sm text-rose-700">
                    {forgotError}
                  </p>
                )}

                <p className="text-xs leading-5 text-slate-500">
                  In the research prototype, this request is simulated. No
                  reset email is sent until a backend email service is
                  configured.
                </p>

                <div className="flex gap-3 pt-1">
                  <Button
                    type="button"
                    variant="outline"
                    className="flex-1"
                    onClick={() => setShowForgotModal(false)}
                  >
                    Cancel
                  </Button>
                  <Button
                    type="submit"
                    className="flex-1"
                    isLoading={forgotLoading}
                  >
                    Submit request
                  </Button>
                </div>
              </form>
            )}
          </section>
        </div>
      )}
    </>
  );
}
