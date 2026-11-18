# OmniAgent UI

React 19 + TypeScript (strict) + Tailwind v4, built with Vite.

- `src/lib/gatewayClient.ts`: WebSocket transport. Handles reconnects with backoff,
  heartbeat pongs, per-session `seq` gap detection, and per-response token-index and
  SHA-256 verification.
- `src/lib/conversation.ts`: a pure reducer from gateway events to turns (attempts,
  discarded generations, steps, sources).
- `src/components/`: the chat panel, plus an inspector showing tool routing, relevance
  and grounding scores, the reasoning timeline, token rate and connection integrity.

```bash
npm ci
OMNI_BACKEND_URL=http://127.0.0.1:8000 npm run dev   # proxies /ws, /sse, /health
npm run typecheck && npm test && npm run build
npx playwright install chromium && npm run e2e        # boots backend + built UI
```

The app connects to `ws(s)://<same host>/ws/agent`. In Kubernetes, the Ingress (or this
image's nginx) routes `/ws`, `/sse` and `/health` to the backend.
