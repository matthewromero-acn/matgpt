#!/bin/bash
# MatGPT — start backend + frontend, open as a native-looking app window
set -e

REPO="$(cd "$(dirname "$0")" && pwd)"

echo "🤖  MatGPT starting..."

# ── Kill any stale MatGPT processes ─────────────────────────────────────────
pkill -f "matgpt.server" 2>/dev/null || true
pkill -f "next dev"       2>/dev/null || true
sleep 1

# ── Find Python ──────────────────────────────────────────────────────────────
if [ -f "$REPO/.venv/bin/python" ]; then
  PYTHON="$REPO/.venv/bin/python"
elif command -v python3 &>/dev/null; then
  PYTHON="$(command -v python3)"
else
  echo "✗ Python 3 not found." >&2; exit 1
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
  echo "✗ pnpm not found. Run: npm install -g pnpm" >&2; exit 1
fi
echo "▸ pnpm:   $PNPM"

# ── Backend ──────────────────────────────────────────────────────────────────
echo "▸ Starting backend on http://localhost:8000"
"$PYTHON" -m matgpt.server &
BACKEND_PID=$!

for i in $(seq 1 30); do
  curl -sf http://localhost:8000/models >/dev/null 2>&1 && { echo "  ✓ Backend ready"; break; }
  sleep 1
done

# ── Frontend — capture whatever port Next.js actually binds to ───────────────
NEXT_LOG=$(mktemp /tmp/matgpt-next-XXXX.log)
echo "▸ Starting frontend..."
cd "$REPO/frontend"
"$PNPM" dev > "$NEXT_LOG" 2>&1 &
FRONTEND_PID=$!

# Tail log to terminal in background so user can see Next.js output
tail -f "$NEXT_LOG" &
TAIL_PID=$!

# Wait until Next.js prints its local URL, extract the real port
PORT=""
for i in $(seq 1 30); do
  sleep 1
  PORT=$(grep -oE 'localhost:[0-9]+' "$NEXT_LOG" 2>/dev/null | head -1 | cut -d: -f2)
  [ -n "$PORT" ] && break
done
PORT=${PORT:-3000}
URL="http://localhost:$PORT"

echo "  ✓ Frontend at $URL"

# ── Open as native app window ────────────────────────────────────────────────
sleep 1

# 1. Try Chrome app mode (no address bar, no tabs — looks native)
if open -na "Google Chrome" --args --app="$URL" --window-size=1280,800 2>/dev/null; then
  echo "  ✓ Opened in Chrome app window"
# 2. Try Chrome Beta
elif open -na "Google Chrome Beta" --args --app="$URL" --window-size=1280,800 2>/dev/null; then
  echo "  ✓ Opened in Chrome Beta app window"
# 3. Try Arc
elif open -na "Arc" --args --app="$URL" 2>/dev/null; then
  echo "  ✓ Opened in Arc"
# 4. Fallback: default browser
else
  open "$URL"
  echo "  ✓ Opened in default browser"
fi

echo ""
echo "  MatGPT is running at $URL"
echo "  Press Ctrl+C to stop."
echo ""

# ── Cleanup on exit ──────────────────────────────────────────────────────────
trap 'echo ""; echo "Stopping MatGPT..."; kill $BACKEND_PID $FRONTEND_PID $TAIL_PID 2>/dev/null; rm -f "$NEXT_LOG"; exit' INT TERM
wait $BACKEND_PID $FRONTEND_PID
