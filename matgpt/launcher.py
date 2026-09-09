"""
LM Studio health check and launcher.

Usage:
    from matgpt.launcher import ensure_lmstudio_running
    ensure_lmstudio_running()  # no-op if already up; starts it if not
"""

from __future__ import annotations

import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

# Path to the lms CLI — LM Studio installs it here but doesn't add it to PATH
_LMS_DEFAULT_PATH = Path.home() / ".lmstudio" / "bin" / "lms"
_APP_NAME = "LM Studio"
_DEFAULT_BASE_URL = "http://localhost:1234"
_POLL_INTERVAL = 2   # seconds between health checks
_TIMEOUT = 60        # max seconds to wait for server to come up


def _lms_bin() -> str | None:
    """Return path to lms CLI, or None if not found."""
    if _LMS_DEFAULT_PATH.exists():
        return str(_LMS_DEFAULT_PATH)
    found = shutil.which("lms")
    return found  # None if not on PATH either


def is_server_running(base_url: str = _DEFAULT_BASE_URL) -> bool:
    """Return True if the LM Studio HTTP server is responding."""
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _start_via_lms_cli(lms: str, port: int = 1234) -> bool:
    """
    Use the lms CLI to start the server in the background.
    Returns True if the command was issued without error.
    """
    try:
        result = subprocess.run(
            [lms, "server", "start", "--port", str(port)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _open_lmstudio_app() -> None:
    """Open the LM Studio GUI app via macOS `open`."""
    subprocess.Popen(["open", "-a", _APP_NAME])


def _wait_for_server(base_url: str, timeout: int, verbose: bool) -> bool:
    """Poll until server is up or timeout expires. Returns True if it came up."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_server_running(base_url):
            return True
        if verbose:
            remaining = int(deadline - time.time())
            print(f"  Waiting for LM Studio server... ({remaining}s remaining)", flush=True)
        time.sleep(_POLL_INTERVAL)
    return False


def ensure_lmstudio_running(
    base_url: str = _DEFAULT_BASE_URL,
    port: int = 1234,
    timeout: int = _TIMEOUT,
    verbose: bool = True,
) -> None:
    """
    Ensure the LM Studio server is running.

    Strategy:
      1. If server already responds → done.
      2. Try `lms server start` (fast, headless).
      3. If lms unavailable or fails → open the GUI app and wait.

    Raises RuntimeError if the server doesn't come up within `timeout` seconds.
    """
    if is_server_running(base_url):
        if verbose:
            print("✓ LM Studio server already running.")
        return

    if verbose:
        print("LM Studio server not detected. Attempting to start...")

    lms = _lms_bin()
    started_headless = False

    if lms:
        if verbose:
            print(f"  Using lms CLI at: {lms}")
        started_headless = _start_via_lms_cli(lms, port)
        if started_headless and verbose:
            print("  lms server start issued.")

    if not started_headless:
        if verbose:
            print(f"  Opening {_APP_NAME} GUI...")
        _open_lmstudio_app()

    if verbose:
        print(f"  Waiting up to {timeout}s for server on {base_url}...")

    if _wait_for_server(base_url, timeout, verbose):
        if verbose:
            print(f"✓ LM Studio server is up at {base_url}")
    else:
        raise RuntimeError(
            f"LM Studio server did not come up within {timeout}s at {base_url}.\n"
            "If you're using the GUI app, make sure to enable the server in "
            "LM Studio → Developer → Start Server."
        )
