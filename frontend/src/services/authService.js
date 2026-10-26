/**
 * Authentication Service (Provisional Endpoints)
 * 
 * In Milestone 1, this service operates in Demo Mode by default.
 * When a real authentication server is deployed, switch VITE_USE_MOCK=false
 * in .env to point to the backend endpoints.
 */

import { apiClient, IS_MOCK_MODE } from './api';
import { MOCK_USER } from '../mocks/mockCases';

// Simulated delay helper for realistic demo UX
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export const authService = {
  /**
   * Login user with credentials
   * Provisional Backend Route: POST /auth/login
   */
  async login({ email, password, rememberMe }) {
    if (IS_MOCK_MODE) {
      await delay(600); // simulate network latency
      
      // Accept demo credentials or any validly formatted email/password in demo mode
      if (!email || !password) {
        throw new Error('Please enter both email and password.');
      }
      
      const user = {
        ...MOCK_USER,
        email: email.trim().toLowerCase(),
      };

      const token = 'demo-jwt-token-arcade-research-milestone-1';
      if (rememberMe) {
        localStorage.setItem('arteryai_remember_email', email);
      } else {
        localStorage.removeItem('arteryai_remember_email');
      }

      localStorage.setItem('arteryai_auth_token', token);
      localStorage.setItem('arteryai_user', JSON.stringify(user));

      return { user, token };
    }

    // Real backend call
    const response = await apiClient.post('/auth/login', { email, password });
    if (response.data.token) {
      localStorage.setItem('arteryai_auth_token', response.data.token);
      localStorage.setItem('arteryai_user', JSON.stringify(response.data.user));
    }
    return response.data;
  },

  /**
   * Register a new researcher account
   * Provisional Backend Route: POST /auth/register
   */
  async register({ fullName, email, password }) {
    if (IS_MOCK_MODE) {
      await delay(700);

      const newUser = {
        id: `usr-demo-${Date.now().toString().slice(-4)}`,
        name: fullName.trim(),
        email: email.trim().toLowerCase(),
        role: 'Research Student / Investigator',
        institution: 'Medical Research Department',
        isDemoUser: true,
      };

      const token = 'demo-jwt-token-arcade-research-milestone-1';
      localStorage.setItem('arteryai_auth_token', token);
      localStorage.setItem('arteryai_user', JSON.stringify(newUser));

      return { user: newUser, token };
    }

    const response = await apiClient.post('/auth/register', {
      fullName,
      email,
      password,
    });
    return response.data;
  },

  /**
   * Retrieve currently authenticated user session
   * Provisional Backend Route: GET /auth/me
   */
  async getCurrentUser() {
    if (IS_MOCK_MODE) {
      const stored = localStorage.getItem('arteryai_user');
      if (stored) {
        try {
          return JSON.parse(stored);
        } catch {
          return MOCK_USER;
        }
      }
      return null;
    }

    const response = await apiClient.get('/auth/me');
    return response.data.user;
  },

  /**
   * Request password reset instructions
   * Provisional Backend Route: POST /auth/forgot-password
   */
  async requestPasswordReset(email) {
    if (IS_MOCK_MODE) {
      await delay(500);
      return {
        success: true,
        message: `In demo mode, password reset link is simulated for ${email}. When the email service is connected, an actual reset token will be dispatched.`,
      };
    }

    const response = await apiClient.post('/auth/forgot-password', { email });
    return response.data;
  },

  /**
   * Terminate user session
   * Provisional Backend Route: POST /auth/logout
   */
  async logout() {
    if (IS_MOCK_MODE) {
      localStorage.removeItem('arteryai_auth_token');
      localStorage.removeItem('arteryai_user');
      return true;
    }

    try {
      await apiClient.post('/auth/logout');
    } finally {
      localStorage.removeItem('arteryai_auth_token');
      localStorage.removeItem('arteryai_user');
    }
    return true;
  },
};
