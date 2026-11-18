import { defineConfig, devices } from '@playwright/test'

// Boots the real stack: FastAPI backend (seeded knowledge base, MiniLM embeddings,
// offline heuristic reasoner) + the production frontend build served by `vite preview`,
// which proxies /ws, /sse and /health to the backend (WebSocket upgrade included).
const BACKEND_PORT = Number(process.env.E2E_BACKEND_PORT ?? 8765)
const FRONTEND_PORT = Number(process.env.E2E_FRONTEND_PORT ?? 4173)
const backendUrl = `http://127.0.0.1:${BACKEND_PORT}`

export default defineConfig({
  testDir: './e2e',
  outputDir: './e2e-artifacts/results',
  fullyParallel: false,
  workers: 1,
  timeout: 60_000,
  expect: { timeout: 20_000 },
  reporter: [['list']],
  use: {
    baseURL: `http://127.0.0.1:${FRONTEND_PORT}`,
    trace: 'retain-on-failure',
    ...devices['Desktop Chrome'],
    viewport: { width: 1400, height: 900 },
  },
  webServer: [
    {
      command: `.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port ${BACKEND_PORT}`,
      cwd: '..',
      url: `${backendUrl}/readyz`,
      timeout: 120_000,
      reuseExistingServer: false,
      env: {
        SEED_PATH: 'samples',
        EMBEDDING_PROVIDER: process.env.E2E_EMBEDDING_PROVIDER ?? 'sentence-transformers',
        INFERENCE_BACKEND: 'mock',
        LLM_PROVIDER: 'heuristic',
        QDRANT_URL: '',
        ELASTICSEARCH_URL: '',
        REDIS_URL: '',
        LOG_FORMAT: 'json',
        LOG_LEVEL: 'INFO',
      },
      stdout: 'ignore',
      stderr: 'pipe',
    },
    {
      command: `npm run build && npx vite preview --host 127.0.0.1 --port ${FRONTEND_PORT} --strictPort`,
      url: `http://127.0.0.1:${FRONTEND_PORT}`,
      timeout: 120_000,
      reuseExistingServer: false,
      env: { OMNI_BACKEND_URL: backendUrl },
    },
  ],
})
