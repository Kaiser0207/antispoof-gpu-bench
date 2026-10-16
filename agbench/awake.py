"""Keep the PC from going to sleep while a suite runs (best effort, never raises).

A benchmark gets no keyboard or mouse input for 30-50 minutes, which is exactly when an idle
timer suspends the PC; a sleep in the middle of a scenario turns it into a timeout or skews it.

* Windows: ``SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`` on the runner thread
  (released at the end, and automatically by Windows when the process exits).
* Linux: a ``systemd-inhibit --what=sleep:idle ... sleep N`` helper process holds an inhibitor lock
  for at most ``max_s`` seconds; it is killed at the end and dies with the runner (PDEATHSIG).
  Without systemd (or without permission) nothing happens.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _pdeathsig() -> None:  # runs in the helper child before exec (Linux only)
    try:
        import ctypes
        import signal

        ctypes.CDLL(None).prctl(1, int(signal.SIGTERM), 0, 0, 0)  # PR_SET_PDEATHSIG
    except Exception:
        pass


class KeepAwake:
    def __init__(self, max_s: float = 4 * 3600, why: str = "GPU benchmark running", enabled: bool = True):
        self.max_s = max(60, int(max_s))
        self.why = why
        self.enabled = enabled
        self.method: str | None = None
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> "KeepAwake":
        try:
            if not self.enabled:
                pass
            elif sys.platform.startswith("win"):
                import ctypes

                if ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
                    self.method = "SetThreadExecutionState"
            elif sys.platform.startswith("linux") and shutil.which("systemd-inhibit"):
                self._proc = subprocess.Popen(
                    ["systemd-inhibit", "--what=sleep:idle", "--who=agbench", f"--why={self.why}",
                     "--mode=block", "sleep", str(self.max_s)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    preexec_fn=_pdeathsig)
                try:
                    self._proc.wait(timeout=1.0)  # exits at once when the lock is refused
                    self._proc = None
                except subprocess.TimeoutExpired:
                    self.method = "systemd-inhibit"
        except Exception:
            self._proc = None
        return self

    def __exit__(self, *exc) -> None:
        try:
            if self.method == "SetThreadExecutionState":
                import ctypes

                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            if self._proc is not None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        except Exception:
            pass
        self._proc = None


def keep_awake(max_s: float = 4 * 3600) -> KeepAwake:
    """``with keep_awake(): ...``; set AGBENCH_NO_KEEP_AWAKE=1 to turn it off."""
    return KeepAwake(max_s=max_s, enabled=not os.environ.get("AGBENCH_NO_KEEP_AWAKE"))
