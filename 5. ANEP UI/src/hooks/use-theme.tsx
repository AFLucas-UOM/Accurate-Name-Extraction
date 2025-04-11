
import { useEffect, useState } from 'react';

export function useTheme() {
  const [theme, setTheme] = useState<'light' | 'dark'>(
    () => {
      const storedTheme = localStorage.getItem('anep-theme');
      
      if (storedTheme) {
        return storedTheme as 'light' | 'dark';
      }
      
      return window.matchMedia('(prefers-color-scheme: dark)').matches 
        ? 'dark'
        : 'light';
    }
  );

  const toggleTheme = () => {
    setTheme(prevTheme => prevTheme === 'dark' ? 'light' : 'dark');
  };

  useEffect(() => {
    localStorage.setItem('anep-theme', theme);
    
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  return { theme, toggleTheme };
}
