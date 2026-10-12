#!/usr/bin/env python3
"""Validate and register a frozen model package for the API.

The candidate directory must contain ``best.pt``, ``frozen_inference.json``,
``package_manifest.json`` and ``assets/hand_landmarker.task``. It is copied
immutably to ``model/releases/<version>``; selecting it still requires an API
restart with ``DATA_CLIP_MODEL_VERSION=<version>``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil

from video_cutter.service import InferenceService


ROOT = Path(__file__).resolve().parent
VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Immutable release identifier, e.g. v1.4")
    parser.add_argument("candidate", type=Path, help="Directory containing a complete frozen model package")
    args = parser.parse_args()
    if not VERSION.fullmatch(args.version):
        parser.error("version 只能包含字母、数字、点、下划线和连字符")
    candidate = args.candidate.expanduser().resolve()
    destination = ROOT / "model/releases" / args.version
    if not candidate.is_dir():
        parser.error("candidate 必须是目录")
    if destination.exists():
        parser.error(f"版本已存在，拒绝覆盖：{destination}")
    required = ("best.pt", "frozen_inference.json", "package_manifest.json", "assets/hand_landmarker.task")
    missing = [name for name in required if not (candidate / name).is_file()]
    if missing:
        parser.error("候选模型包缺少：" + ", ".join(missing))
    # Verify hashes and checkpoint provenance before anything is copied.
    InferenceService(package_root=ROOT, model_root=candidate).verify_package()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(candidate, destination)
    # Verify the copied release as the API will verify it at startup.
    InferenceService(package_root=ROOT, model_root=destination).verify_package()
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
