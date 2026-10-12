#!/usr/bin/env python3
"""Launch the desktop application and persist startup failures to a log."""
from __future__ import annotations

import logging
from pathlib import Path
import sys


def configure_logging() -> logging.Logger:
    runtime = Path(__file__).resolve().parent / ".runtime"
    runtime.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(runtime / "desktop-python.log", encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger("data_clip.desktop")


def launch() -> int:
    logger = configure_logging()
    logger.info("Python executable: %s", sys.executable)
    try:
        from video_cutter.gui import main
        return main()
    except BaseException:
        logger.exception("桌面程序未能启动")
        return 1

if __name__ == "__main__":
    raise SystemExit(launch())
