#!/usr/bin/env bash
# Start the HTTP API without starting Qt or the desktop application.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/video_clip"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "未找到 video_clip。请先运行 ./install_api_linux.sh" >&2
  exit 1
fi
if [[ -z "${DATA_CLIP_API_KEY:-}" ]]; then
  echo "必须设置 DATA_CLIP_API_KEY，拒绝以无鉴权模式启动。" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
cd "$SCRIPT_DIR"
RUNTIME_DIR="$SCRIPT_DIR/.runtime"
mkdir -p "$RUNTIME_DIR"
exec python "$SCRIPT_DIR/run_api.py" "$@"
