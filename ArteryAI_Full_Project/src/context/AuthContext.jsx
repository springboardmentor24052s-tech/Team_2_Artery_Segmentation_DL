import React, { createContext, useContext, useState, useEffect } from 'react';
import { authService } from '../services/authService';
import { MOCK_USER } from '../mocks/mockCases';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // Initialize session on mount
  useEffect(() => {
    async function initAuth() {
      try {
        const currentUser = await authService.getCurrentUser();
        // Default to demo user if not logged in to make demonstration smooth,
        // or allow user to log out to test login/registration screens
        if (currentUser) {
          setUser(currentUser);
        } else {
          // Check if previously logged out explicitly
          const hasExplicitLogout = sessionStorage.getItem('arteryai_logged_out');
          if (!hasExplicitLogout) {
            setUser(MOCK_USER);
            localStorage.setItem('arteryai_user', JSON.stringify(MOCK_USER));
            localStorage.setItem('arteryai_auth_token', 'demo-token');
          }
        }
      } catch (err) {
        console.error('Auth initialization error:', err);
      } finally {
        setIsLoading(false);
      }
    }
    initAuth();
  }, []);

  const login = async (credentials) => {
    sessionStorage.removeItem('arteryai_logged_out');
    const result = await authService.login(credentials);
    setUser(result.user);
    return result;
  };

  const register = async (userData) => {
    sessionStorage.removeItem('arteryai_logged_out');
    const result = await authService.register(userData);
    setUser(result.user);
    return result;
  };

  const logout = async () => {
    await authService.logout();
    sessionStorage.setItem('arteryai_logged_out', 'true');
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: Boolean(user),
        isLoading,
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
