#!/usr/bin/env python3
"""Launch the authenticated data_clip HTTP API."""
from __future__ import annotations

import logging
from pathlib import Path
import sys


def configure_logging() -> None:
    runtime = Path(__file__).resolve().parent / ".runtime"
    runtime.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(runtime / "api.log", encoding="utf-8"), logging.StreamHandler()],
    )


def main() -> int:
    configure_logging()
    try:
        import uvicorn
        from video_cutter.api import ApiConfig, create_app
        config = ApiConfig.from_environment()
        uvicorn.run(create_app(config), host=config.host, port=config.port, log_level="info")
    except BaseException:
        logging.getLogger("data_clip.api").exception("API 未能启动")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
