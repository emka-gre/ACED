// `vitest/config` rather than `vite`: it is the same defineConfig widened to accept the
// `test` block, so the dev server and the test runner stay described in one file.
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

// The dev server proxies /analyses and /health to the API so the browser sees one origin.
// Without it every request would be cross-origin and the API would need CORS opened up for
// a case that only exists in development.
const API = process.env.ACED_API_URL ?? 'http://localhost:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/analyses': { target: API, changeOrigin: true },
      '/health': { target: API, changeOrigin: true },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
  },
});
