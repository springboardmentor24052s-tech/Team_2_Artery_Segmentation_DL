import React, { createContext, useContext, useEffect, useState } from 'react';

const ThemeContext = createContext(null);
const VALID_THEMES = ['light', 'blue', 'dark'];

function readTheme() {
  try {
    const saved = localStorage.getItem('arteryai-theme');
    // Keep existing users' saved light/dark preferences working.
    return VALID_THEMES.includes(saved) ? saved : 'light';
  } catch {
    return 'light';
  }
}

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(readTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('arteryai-theme', theme); } catch { /* Storage is optional. */ }
  }, [theme]);

  const value = {
    theme,
    setTheme,
    // Backwards-compatible helper for any existing component using it.
    toggleTheme: () => setTheme((current) => current === 'dark' ? 'light' : 'dark'),
  };

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) throw new Error('useTheme must be used inside ThemeProvider');
  return context;
}
