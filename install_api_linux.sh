#!/usr/bin/env bash
# Install the API dependencies into the project's existing Python 3.12 venv.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/video_clip"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if [[ ! -x "$VENV_DIR/bin/pip" ]]; then
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "需要 Python 3.12，或先运行 ./install_desktop_linux.sh。" >&2
    exit 1
  fi
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "需要 FFmpeg（ffmpeg 与 ffprobe）。" >&2
  exit 1
fi

"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0+cpu
"$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.inference.txt" -r "$SCRIPT_DIR/requirements.api.txt"
echo "安装完成。请设置 DATA_CLIP_API_KEY 后运行 ./run_api.sh"
