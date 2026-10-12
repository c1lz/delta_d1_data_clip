#!/usr/bin/env bash
# Set PYTHON_BIN if Python 3.12 is installed under another executable name.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="$SCRIPT_DIR/video_clip"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "需要 Python 3.12。请安装它，或用 PYTHON_BIN 指定解释器。" >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "需要 FFmpeg（ffmpeg 与 ffprobe）。Debian/Ubuntu 可执行：sudo apt install ffmpeg libxcb-cursor0" >&2
  exit 1
fi
if ! ldconfig -p 2>/dev/null | grep -q 'libxcb-cursor\.so\.0'; then
  echo "需要 Qt X11 运行库 libxcb-cursor0。Debian/Ubuntu 可执行：sudo apt install libxcb-cursor0" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0+cpu
"$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.inference.txt" -r "$SCRIPT_DIR/requirements.app.txt"
echo "安装完成：./run_desktop.sh"
