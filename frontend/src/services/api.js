import axios from 'axios';

/**
 * Centralized Axios API client instance
 * 
 * Configured with provisional base URL from environment variables.
 * Ready for future backend integration without modifying frontend components.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
});

// Request interceptor to attach JWT token when backend authentication is connected
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('arteryai_auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for consistent error normalization
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Graceful handling of common API responses
    const customError = {
      message: error.response?.data?.message || error.message || 'An unexpected error occurred',
      status: error.response?.status,
      data: error.response?.data,
      isNetworkError: !error.response,
    };
    return Promise.reject(customError);
  }
);

export const IS_MOCK_MODE = import.meta.env.VITE_USE_MOCK !== 'false';

export default apiClient;
