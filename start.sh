#!/bin/bash
# MatGPT — start backend + frontend, then open browser
set -e

REPO="$(cd "$(dirname "$0")" && pwd)"

echo "🤖  MatGPT starting..."

# ── Find Python ──────────────────────────────────────────────────────────────
if [ -f "$REPO/.venv/bin/python" ]; then
  PYTHON="$REPO/.venv/bin/python"
elif command -v python3 &>/dev/null; then
  PYTHON="$(command -v python3)"
else
  echo "✗ Python 3 not found. Install from https://python.org" >&2
  exit 1
fi
echo "▸ Python: $PYTHON"

# ── Find pnpm ────────────────────────────────────────────────────────────────
if command -v pnpm &>/dev/null; then
  PNPM="$(command -v pnpm)"
elif [ -f "$HOME/Library/pnpm/pnpm" ]; then
  PNPM="$HOME/Library/pnpm/pnpm"
elif [ -f "$HOME/.local/share/pnpm/pnpm" ]; then
  PNPM="$HOME/.local/share/pnpm/pnpm"
else
  echo "✗ pnpm not found. Install: npm install -g pnpm" >&2
  exit 1
fi
echo "▸ pnpm:   $PNPM"

# ── Backend ──────────────────────────────────────────────────────────────────
echo "▸ Starting backend on http://localhost:8000"
"$PYTHON" -m matgpt.server &
BACKEND_PID=$!

# Wait for backend to be healthy
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/models >/dev/null 2>&1; then
    echo "  ✓ Backend ready"
    break
  fi
  sleep 1
done

# ── Frontend ─────────────────────────────────────────────────────────────────
echo "▸ Starting frontend"
cd "$REPO/frontend"
"$PNPM" dev &
FRONTEND_PID=$!

# Wait for frontend then open browser
sleep 4
echo "  ✓ Opening http://localhost:3000"
open http://localhost:3000 2>/dev/null || xdg-open http://localhost:3000 2>/dev/null || true

echo ""
echo "  MatGPT is running. Press Ctrl+C to stop."
echo ""

# ── Cleanup on exit ──────────────────────────────────────────────────────────
trap 'echo ""; echo "Stopping MatGPT..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit' INT TERM
wait
