/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Backend for the dev server and `vite preview`. In Kubernetes the Ingress (or the
// frontend's nginx) routes these paths to the backend instead.
const backend = process.env.OMNI_BACKEND_URL ?? 'http://127.0.0.1:8000'
const proxy = {
  '/ws': { target: backend.replace(/^http/, 'ws'), ws: true, changeOrigin: true },
  '/sse': { target: backend, changeOrigin: true },
  '/health': { target: backend, changeOrigin: true },
  '/search': { target: backend, changeOrigin: true },
  '/ingest': { target: backend, changeOrigin: true },
  '/rooms': { target: backend, changeOrigin: true },
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy },
  preview: { proxy },
  test: {
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
    environment: 'node',
  },
})
