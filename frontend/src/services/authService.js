
/**
 * ArteryAI Authentication Service
 *
 * Mock mode is for local development and demonstrations.
 * Real mode uses the FastAPI authentication endpoints.
 */

import { apiClient, IS_MOCK_MODE } from './api';
import { MOCK_USER } from '../mocks/mockCases';

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const USERS_KEY = 'arteryai_demo_accounts';
const USER_KEY = 'arteryai_user';
const TOKEN_KEY = 'arteryai_auth_token';
const REFRESH_TOKEN_KEY = 'arteryai_refresh_token';
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

function saveSession(user, accessToken, refreshToken = null) {
  localStorage.setItem(USER_KEY, JSON.stringify(user));
  localStorage.setItem(TOKEN_KEY, accessToken);

  if (refreshToken) {
    localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
  } else {
    localStorage.removeItem(REFRESH_TOKEN_KEY);
  }
}

function clearSession() {
  localStorage.removeItem(USER_KEY);
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(REFRESH_TOKEN_KEY);
}

function publicUser(account) {
  const { passwordHash, ...user } = account;
  return user;
}

// Demo-only password hashing. Real password handling belongs on the server.
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

// Uses the actual backend profile endpoint.
async function fetchBackendUser() {
  const response = await apiClient.get('/users/me');
  return response.data;
}

// Completes backend login and stores both tokens.
async function authenticateBackend(email, password) {
  const response = await apiClient.post('/auth/login', {
    email,
    password,
  });

  const result = response.data;

  if (!result?.access_token || !result?.refresh_token) {
    throw new Error(
      'The authentication server returned an invalid token response.'
    );
  }

  // Save access token before requesting the protected profile endpoint.
  localStorage.setItem(TOKEN_KEY, result.access_token);
  localStorage.setItem(REFRESH_TOKEN_KEY, result.refresh_token);

  try {
    const user = await fetchBackendUser();

    saveSession(user, result.access_token, result.refresh_token);

    return {
      user,
      token: result.access_token,
    };
  } catch (error) {
    clearSession();
    throw error;
  }
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

    const result = await authenticateBackend(
      normalizedEmail,
      password
    );

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

    // Backend accepts name, email and password.
    // Institution is currently a frontend-only field.
    await apiClient.post('/auth/register', {
      name,
      email: normalizedEmail,
      password,
    });

    // Registration does not return tokens, so sign in separately.
    return authenticateBackend(normalizedEmail, password);
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

    const token = localStorage.getItem(TOKEN_KEY);

    if (!token) {
      clearSession();
      return null;
    }

    try {
      const user = await fetchBackendUser();

      localStorage.setItem(USER_KEY, JSON.stringify(user));

      return user;
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

    throw new Error(
      'Password reset is not yet available. Please contact your administrator.'
    );
  },

  async logout() {
    try {
      if (!IS_MOCK_MODE) {
        const refreshToken = localStorage.getItem(REFRESH_TOKEN_KEY);

        if (refreshToken) {
          await apiClient.post('/auth/logout', {
            refresh_token: refreshToken,
          });
        }
      }
    } finally {
      clearSession();
    }

    return true;
  },
};
