#!/bin/bash
# MatGPT one-time setup — run this once after cloning the repo.
set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
echo "🤖  MatGPT setup..."

# ── Python venv + deps ───────────────────────────────────────────────────────
if [ ! -d "$REPO/.venv" ]; then
  echo "▸ Creating Python venv..."
  python3 -m venv "$REPO/.venv"
fi
echo "▸ Installing Python dependencies..."
"$REPO/.venv/bin/pip" install -q -r "$REPO/requirements.txt"
"$REPO/.venv/bin/pip" install -q -e "$REPO"

# ── Frontend deps ────────────────────────────────────────────────────────────
echo "▸ Installing frontend dependencies..."
cd "$REPO/frontend" && pnpm install

# ── Electron deps + binary ───────────────────────────────────────────────────
echo "▸ Installing Electron..."
cd "$REPO/electron"
npm install --legacy-peer-deps

# Download Electron binary if install script was skipped (pnpm issue)
EPATH=$(node -e "console.log(require('path').dirname(require.resolve('electron')))")
if [ ! -f "$EPATH/path.txt" ]; then
  echo "▸ Downloading Electron binary..."
  ARCH=$(uname -m | sed 's/x86_64/x64/' | sed 's/arm64/arm64/')
  VERSION=$(node -e "console.log(require('./node_modules/electron/package.json').version)")
  ZIP="/tmp/electron-v${VERSION}-darwin-${ARCH}.zip"
  curl -L --progress-bar \
    "https://github.com/electron/electron/releases/download/v${VERSION}/electron-v${VERSION}-darwin-${ARCH}.zip" \
    -o "$ZIP"
  rm -rf "$EPATH/dist"
  mkdir -p "$EPATH/dist"
  unzip -q "$ZIP" -d "$EPATH/dist"
  echo -n "Electron.app/Contents/MacOS/Electron" > "$EPATH/path.txt"
  rm "$ZIP"
fi

# ── Replace Electron robot icon with MatGPT pixel face ───────────────────────
echo "▸ Setting app icon..."
ELECTRON_RESOURCES=$(find "$REPO/electron/node_modules" -name "electron.icns" -path "*/Electron.app/*" 2>/dev/null | head -1)
if [ -n "$ELECTRON_RESOURCES" ]; then
  cp "$REPO/electron/AppIcon.icns" "$ELECTRON_RESOURCES"
fi
killall Dock 2>/dev/null || true

echo ""
echo "  ✓ Setup complete!"
echo "  Double-click MatGPT.app to launch, or run: cd electron && npx electron ."
echo ""
