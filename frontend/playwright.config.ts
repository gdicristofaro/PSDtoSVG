import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright config for the accessibility (axe) end-to-end suite.
 * Tests live in ./e2e and run against the Vite dev server.
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? 'github' : 'list',
  use: {
    // Vite serves the app under the configured base ('/PSDtoSVG/', see vite.config.ts).
    baseURL: 'http://localhost:5173/PSDtoSVG/',
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:5173/PSDtoSVG/',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
