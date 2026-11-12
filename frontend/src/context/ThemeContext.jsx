
import React, {
  createContext,
  useContext,
  useEffect,
  useState,
} from 'react';

const ThemeContext = createContext(null);

const VALID_THEMES = ['light', 'dark'];
const VALID_ACCENTS = ['blue', 'teal', 'purple', 'green'];

function readPreference(key, allowedValues, fallback) {
  try {
    const savedValue = localStorage.getItem(key);
    return allowedValues.includes(savedValue) ? savedValue : fallback;
  } catch {
    return fallback;
  }
}

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(() =>
    readPreference('arteryai-theme', VALID_THEMES, 'light')
  );

  const [accent, setAccent] = useState(() =>
    readPreference('arteryai-accent', VALID_ACCENTS, 'blue')
  );

  useEffect(() => {
    const root = document.documentElement;

    root.dataset.theme = theme;
    root.dataset.accent = accent;

    try {
      localStorage.setItem('arteryai-theme', theme);
      localStorage.setItem('arteryai-accent', accent);
    } catch {
      // The selected theme still works if browser storage is unavailable.
    }
  }, [theme, accent]);

  const value = {
    theme,
    accent,
    setTheme,
    setAccent,
    toggleTheme: () =>
      setTheme((current) => current === 'light' ? 'dark' : 'light'),
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);

  if (!context) {
    throw new Error('useTheme must be used inside ThemeProvider');
  }

  return context;
}
