/**
 * MatGPT Electron main process.
 *
 * On launch:
 *  1. Shows a loading splash window
 *  2. Kills stale backend / frontend processes
 *  3. Starts Python backend (matgpt.server) as a child process
 *  4. Starts Next.js frontend (pnpm dev) as a child process
 *  5. Detects whatever port Next.js actually binds to
 *  6. Opens the main BrowserWindow at that URL
 *  7. On quit, kills both child processes cleanly
 */

const { app, BrowserWindow, dialog, shell, nativeImage } = require('electron')
const { spawn, execSync }                                 = require('child_process')
const http                                                = require('http')
const path                                                = require('path')
const fs                                                  = require('fs')

// Set dock icon immediately — before any window is created
if (process.platform === 'darwin') {
  const icon = nativeImage.createFromPath(path.join(__dirname, 'AppIcon.icns'))
  if (!icon.isEmpty()) app.dock.setIcon(icon)
}

const REPO     = path.resolve(__dirname, '..')
const FRONTEND = path.join(REPO, 'frontend')

// All child processes — killed on app quit
const children = []

// ── Path helpers ──────────────────────────────────────────────────────────────

function findPython () {
  const venv = path.join(REPO, '.venv', 'bin', 'python')
  return fs.existsSync(venv) ? venv : 'python3'
}

function findPnpm () {
  const candidates = [
    '/opt/homebrew/bin/pnpm',
    '/usr/local/bin/pnpm',
    path.join(process.env.HOME, 'Library', 'pnpm', 'pnpm'),
    path.join(process.env.HOME, '.local', 'share', 'pnpm', 'pnpm'),
  ]
  for (const p of candidates) if (fs.existsSync(p)) return p
  try { return execSync('which pnpm').toString().trim() } catch {}
  return 'pnpm'
}

// ── Process helpers ───────────────────────────────────────────────────────────

function spawnChild (cmd, args, opts = {}) {
  const proc = spawn(cmd, args, {
    cwd: REPO,
    env: { ...process.env, PATH: `/opt/homebrew/bin:/usr/local/bin:${process.env.PATH}` },
    ...opts,
  })
  children.push(proc)
  const tag = path.basename(cmd)
  proc.stdout?.on('data', d => process.stdout.write(`[${tag}] ${d}`))
  proc.stderr?.on('data', d => process.stderr.write(`[${tag}] ${d}`))
  return proc
}

function killStale () {
  for (const pattern of ['matgpt.server', 'next dev']) {
    try { execSync(`pkill -f "${pattern}"`) } catch {}
  }
}

// ── Wait helpers ──────────────────────────────────────────────────────────────

function httpGet (url) {
  return new Promise(resolve => {
    http.get(url, () => resolve(true)).on('error', () => resolve(false))
  })
}

async function waitForBackend (timeout = 30_000) {
  const deadline = Date.now() + timeout
  while (Date.now() < deadline) {
    if (await httpGet('http://localhost:8000/models')) return
    await new Promise(r => setTimeout(r, 500))
  }
  throw new Error('Backend did not start in time.')
}

// Returns the port Next.js actually bound to
async function waitForFrontend (proc, timeout = 60_000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Frontend timed out.')), timeout)

    const scan = (data) => {
      const text = data.toString()
      const m = text.match(/localhost:(\d+)/)
      if (m) {
        clearTimeout(timer)
        // Small extra pause so Next.js finishes compiling
        setTimeout(() => resolve(Number(m[1])), 1500)
      }
    }

    proc.stdout?.on('data', scan)
    proc.stderr?.on('data', scan)
  })
}

// ── Windows ───────────────────────────────────────────────────────────────────

function createSplash () {
  const win = new BrowserWindow({
    width: 360,
    height: 260,
    frame: false,
    transparent: true,
    resizable: false,
    alwaysOnTop: true,
    webPreferences: { contextIsolation: true },
  })

  win.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(`
    <!DOCTYPE html>
    <html>
    <body style="
      margin:0; padding:0;
      background:#3f6247;
      border-radius:18px;
      display:flex; flex-direction:column;
      align-items:center; justify-content:center;
      height:100vh;
      font-family:'Courier New',monospace;
      color:#d9e6d8;
      -webkit-app-region:drag;
    ">
      <div style="font-size:56px; margin-bottom:12px">🤖</div>
      <div style="font-size:20px; font-weight:900; letter-spacing:-1px">MatGPT</div>
      <div style="font-size:11px; opacity:.6; margin-top:10px">Starting up…</div>
    </body>
    </html>
  `)}`)

  return win
}

function createMainWindow (port) {
  const win = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 860,
    minHeight: 600,
    titleBarStyle: 'hiddenInset',   // native macOS traffic lights, no title bar
    trafficLightPosition: { x: 18, y: 18 },
    backgroundColor: '#f4f2e9',
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  win.loadURL(`http://localhost:${port}`)

  // Open <a target="_blank"> links in the system browser, not a new Electron window
  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })

  return win
}

// ── App lifecycle ─────────────────────────────────────────────────────────────

let mainWindow = null

app.whenReady().then(async () => {
  const splash = createSplash()

  try {
    console.log('[matgpt] Killing stale processes…')
    killStale()
    await new Promise(r => setTimeout(r, 600))

    console.log('[matgpt] Starting backend…')
    spawnChild(findPython(), ['-m', 'matgpt.server'])
    await waitForBackend()
    console.log('[matgpt] Backend ready.')

    console.log('[matgpt] Starting frontend…')
    const frontendProc = spawnChild(findPnpm(), ['dev'], { cwd: FRONTEND })
    const port = await waitForFrontend(frontendProc)
    console.log(`[matgpt] Frontend ready on port ${port}.`)

    splash.close()
    mainWindow = createMainWindow(port)

    mainWindow.on('closed', () => { mainWindow = null })

  } catch (err) {
    console.error('[matgpt] Startup error:', err)
    splash.close()
    dialog.showErrorBox('MatGPT failed to start', err.message)
    app.quit()
  }
})

// macOS: re-open window when dock icon is clicked and no windows are open
app.on('activate', () => {
  if (mainWindow === null && BrowserWindow.getAllWindows().length === 0) {
    app.emit('ready')
  }
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

// Kill child processes before quitting
app.on('before-quit', () => {
  children.forEach(p => { try { p.kill('SIGTERM') } catch {} })
})
