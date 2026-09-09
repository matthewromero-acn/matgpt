#!/bin/bash
# MatGPT — start both the Python backend and Next.js frontend
set -e

REPO="$(cd "$(dirname "$0")" && pwd)"

echo "🚀  Starting MatGPT..."

# ── Python backend ──────────────────────────────────────────────────────────
echo "▸ Starting backend on http://localhost:8000"
python -m matgpt.server &
BACKEND_PID=$!

# ── Wait for backend to be ready ────────────────────────────────────────────
echo "▸ Waiting for backend..."
for i in $(seq 1 20); do
  if curl -sf http://localhost:8000/models > /dev/null 2>&1; then
    echo "  ✓ Backend ready"
    break
  fi
  sleep 1
done

# ── Next.js frontend ────────────────────────────────────────────────────────
echo "▸ Starting frontend on http://localhost:3000"
cd "$REPO/frontend"
pnpm dev &
FRONTEND_PID=$!

echo ""
echo "  MatGPT running at → http://localhost:3000"
echo "  Press Ctrl+C to stop both services."
echo ""

# ── Cleanup on exit ─────────────────────────────────────────────────────────
trap "echo ''; echo 'Stopping…'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
