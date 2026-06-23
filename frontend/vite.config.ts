import { defineConfig, configDefaults } from 'vitest/config';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: '/PSDtoSVG/',
  test: {
    // The Playwright accessibility suite under e2e/ uses @playwright/test and must not be run
    // by vitest (it throws "test.describe() ... did not expect to be called here"). Vitest's
    // default include matches *.spec.ts, so exclude e2e explicitly. Unit tests use *.test.ts.
    exclude: [...configDefaults.exclude, 'e2e/**'],
  },
});
