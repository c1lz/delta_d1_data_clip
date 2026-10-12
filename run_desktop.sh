#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/video_clip"
ACTIVATE_PATH="$VENV_DIR/bin/activate"
if [[ ! -f "$ACTIVATE_PATH" ]]; then
  echo "未找到 video_clip。请先运行 ./install_desktop_linux.sh" >&2
  exit 1
fi

# Activate explicitly so Qt, MediaPipe and all Python imports are always read
# from this project's environment, irrespective of the caller's shell state.
# shellcheck disable=SC1090
source "$ACTIVATE_PATH"
cd "$SCRIPT_DIR"
export MPLCONFIGDIR="$SCRIPT_DIR/.runtime/matplotlib"
export PYTHONUNBUFFERED=1
RUNTIME_DIR="$SCRIPT_DIR/.runtime"
LOG_FILE="$RUNTIME_DIR/desktop.log"
mkdir -p "$MPLCONFIGDIR" "$RUNTIME_DIR"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date --iso-8601=seconds)] starting data_clip with $(command -v python)"
echo "DISPLAY=${DISPLAY:-<unset>} QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-xcb}"
exec python "$SCRIPT_DIR/run_desktop.py" "$@"
