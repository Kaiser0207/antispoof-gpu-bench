#!/usr/bin/env bash
# One-command setup and run on Linux:  bash scripts/bench.sh [full|quick|doctor]
#
# Does exactly the steps of AGENTS.md: install uv if missing, pick the PyTorch build from the
# NVIDIA driver version (>= 580 -> cu130, 570-579 -> cu128), uv sync, doctor, fetch, run, zip results.
# Default suite: full (all 49 scenarios, about 35-50 min; it runs the quick subset first).
# Override the build with AGBENCH_EXTRA=cu130|cu128. Extra arguments are passed to agbench.
set -euo pipefail

SUITE="${1:-full}"
shift || true
case "$SUITE" in
  quick|full|doctor) ;;
  *) echo "usage: bash scripts/bench.sh [full|quick|doctor] [extra agbench options]"; exit 2 ;;
esac

cd "$(dirname "$0")/.."

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found: install the NVIDIA driver (580 or newer) first, then rerun."
  exit 2
fi
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d ' ')"
MAJOR="${DRIVER%%.*}"
echo "NVIDIA driver: ${DRIVER}"

EXTRA="${AGBENCH_EXTRA:-}"
if [ -z "$EXTRA" ]; then
  if [ "$MAJOR" -ge 580 ] 2>/dev/null; then
    EXTRA=cu130
  elif [ "$MAJOR" -ge 570 ] 2>/dev/null; then
    EXTRA=cu128
  else
    echo "Driver ${DRIVER} is too old: install NVIDIA driver 580 or newer (570 at the very least), then rerun."
    exit 2
  fi
fi
echo "PyTorch build: ${EXTRA}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (https://astral.sh/uv) ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --frozen --extra "$EXTRA"
uv run --no-sync python -m agbench doctor
if [ "$SUITE" = "doctor" ]; then
  exit 0
fi
uv run --no-sync python -m agbench fetch

STATUS=0
uv run --no-sync python -m agbench "$SUITE" "$@" || STATUS=$?

uv run --no-sync python -m zipfile -c agbench_results.zip results/ && \
  echo "Results zipped: agbench_results.zip (send this file back)"
echo "Exit code: ${STATUS} (0 = all scenarios ok/oom/skipped, 1 = some error/timeout/crash, 2 = setup problem)"
exit "$STATUS"
