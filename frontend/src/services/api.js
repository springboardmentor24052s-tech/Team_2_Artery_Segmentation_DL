
import axios from 'axios';

/**
 * Centralized Axios API client instance.
 * Supports both JSON requests and multipart FormData uploads.
 */

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    Accept: 'application/json',
  },
});

// Attach JWT token to authenticated requests.
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('arteryai_auth_token');

    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    return config;
  },
  (error) => Promise.reject(error)
);

// Normalize API errors.
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const customError = {
      message:
        error.response?.data?.message ||
        error.message ||
        'An unexpected error occurred',
      status: error.response?.status,
      data: error.response?.data,
      isNetworkError: !error.response,
    };

    return Promise.reject(customError);
  }
);

export const IS_MOCK_MODE = import.meta.env.VITE_USE_MOCK !== 'false';

export default apiClient;
