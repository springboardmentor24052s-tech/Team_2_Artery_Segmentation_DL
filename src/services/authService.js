
/**
 * ArteryAI Authentication Service
 *
 * Demo mode is intended only for local development and demonstrations.
 * Production authentication must use a properly secured backend.
 */

import { apiClient, IS_MOCK_MODE } from './api';
import { MOCK_USER } from '../mocks/mockCases';

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const USERS_KEY = 'arteryai_demo_accounts';
const USER_KEY = 'arteryai_user';
const TOKEN_KEY = 'arteryai_auth_token';
const REMEMBER_KEY = 'arteryai_remember_email';

const normalizeEmail = (email = '') => email.trim().toLowerCase();

const isValidEmail = (email) =>
  /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

function readAccounts() {
  try {
    const parsed = JSON.parse(localStorage.getItem(USERS_KEY) || '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveAccounts(accounts) {
  localStorage.setItem(USERS_KEY, JSON.stringify(accounts));
}

function saveSession(user, token) {
  localStorage.setItem(USER_KEY, JSON.stringify(user));
  localStorage.setItem(TOKEN_KEY, token);
}

function clearSession() {
  localStorage.removeItem(USER_KEY);
  localStorage.removeItem(TOKEN_KEY);
}

// Demo-only hash. Client-side password storage is not secure enough
// for real accounts; production password handling belongs on the server.
async function hashDemoPassword(password) {
  if (!globalThis.crypto?.subtle) {
    throw new Error(
      'Browser cryptography is unavailable. Open the application on localhost.'
    );
  }

  const bytes = new TextEncoder().encode(password);
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);

  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

function publicUser(account) {
  const { passwordHash, ...user } = account;
  return user;
}

export const authService = {
  async login({
    email = '',
    password = '',
    rememberMe = false,
  }) {
    const normalizedEmail = normalizeEmail(email);

    if (!normalizedEmail || !password) {
      throw new Error('Please enter both email and password.');
    }

    if (!isValidEmail(normalizedEmail)) {
      throw new Error('Please enter a valid email address.');
    }

    if (IS_MOCK_MODE) {
      await delay(350);

      const accounts = readAccounts();
      const account = accounts.find(
        (item) => item.email === normalizedEmail
      );

      let user;

      if (account) {
        const submittedHash = await hashDemoPassword(password);

        if (submittedHash !== account.passwordHash) {
          throw new Error('Incorrect email or password.');
        }

        user = publicUser(account);
      } else if (
        normalizedEmail === normalizeEmail(MOCK_USER.email) &&
        password === 'Demo@123'
      ) {
        // Optional fixed demo account for demonstrations only.
        user = { ...MOCK_USER };
      } else {
        throw new Error(
          'Account not found. Please register before signing in.'
        );
      }

      const token = 'arteryai-demo-token';
      saveSession(user, token);

      if (rememberMe) {
        localStorage.setItem(REMEMBER_KEY, normalizedEmail);
      } else {
        localStorage.removeItem(REMEMBER_KEY);
      }

      return { user, token };
    }

    const response = await apiClient.post('/auth/login', {
      email: normalizedEmail,
      password,
    });

    const result = response.data;

    if (!result?.user || !result?.token) {
      throw new Error('The authentication server returned an invalid response.');
    }

    saveSession(result.user, result.token);

    if (rememberMe) {
      localStorage.setItem(REMEMBER_KEY, normalizedEmail);
    } else {
      localStorage.removeItem(REMEMBER_KEY);
    }

    return result;
  },

  async register({
    fullName = '',
    email = '',
    institution = '',
    password = '',
  }) {
    const name = fullName.trim();
    const normalizedEmail = normalizeEmail(email);
    const organization = institution.trim();

    if (!name || !normalizedEmail || !organization || !password) {
      throw new Error('Please complete all required fields.');
    }

    if (!isValidEmail(normalizedEmail)) {
      throw new Error('Please enter a valid email address.');
    }

    if (password.length < 8) {
      throw new Error('Password must contain at least 8 characters.');
    }

    if (IS_MOCK_MODE) {
      await delay(350);

      const accounts = readAccounts();

      const emailAlreadyExists =
        accounts.some((item) => item.email === normalizedEmail) ||
        normalizeEmail(MOCK_USER.email) === normalizedEmail;

      if (emailAlreadyExists) {
        throw new Error(
          'An account with this email already exists. Please sign in.'
        );
      }

      const user = {
        id: `usr-demo-${Date.now()}`,
        name,
        email: normalizedEmail,
        role: 'Research Student / Investigator',
        institution: organization,
        isDemoUser: true,
      };

      const passwordHash = await hashDemoPassword(password);

      saveAccounts([
        ...accounts,
        { ...user, passwordHash },
      ]);

      const token = 'arteryai-demo-token';
      saveSession(user, token);

      return { user, token };
    }

    const response = await apiClient.post('/auth/register', {
      fullName: name,
      email: normalizedEmail,
      institution: organization,
      password,
    });

    const result = response.data;

    if (result?.user && result?.token) {
      saveSession(result.user, result.token);
    }

    return result;
  },

  async getCurrentUser() {
    if (IS_MOCK_MODE) {
      const token = localStorage.getItem(TOKEN_KEY);
      const storedUser = localStorage.getItem(USER_KEY);

      if (!token || !storedUser) {
        clearSession();
        return null;
      }

      try {
        return JSON.parse(storedUser);
      } catch {
        clearSession();
        return null;
      }
    }

    try {
      const response = await apiClient.get('/auth/me');
      return response.data?.user ?? null;
    } catch (error) {
      clearSession();
      throw error;
    }
  },

  async requestPasswordReset(email) {
    const normalizedEmail = normalizeEmail(email);

    if (!isValidEmail(normalizedEmail)) {
      throw new Error('Please enter a valid email address.');
    }

    if (IS_MOCK_MODE) {
      await delay(350);

      return {
        success: true,
        message:
          'This prototype does not send email or change passwords. ' +
          'A backend password-reset service must be configured to complete this process.',
      };
    }

    const response = await apiClient.post('/auth/forgot-password', {
      email: normalizedEmail,
    });

    return response.data;
  },

  async logout() {
    try {
      if (!IS_MOCK_MODE) {
        await apiClient.post('/auth/logout');
      }
    } finally {
      clearSession();
    }

    return true;
  },
};
