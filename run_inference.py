#!/usr/bin/env python3
"""Inference-only runner for the packaged frozen BiLSTM gesture cutter."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
MODEL_ROOT = PACKAGE / "model"
os.environ["SUPERCUT_EXPERIMENT"] = str(MODEL_ROOT)
sys.path.insert(0, str(PACKAGE))

import cv2
import numpy as np

from core import prediction_payload, read_json, sha256, write_json
from data import EXTRACTOR, extract_video
from learning import load_checkpoint, predict_probabilities
from core import normalize


def policy_and_model(model_root: Path | None = None):
    """Load a verified model package, optionally from a registered release."""
    active_model_root = Path(model_root or MODEL_ROOT).resolve()
    policy = read_json(active_model_root / "frozen_inference.json")
    checkpoint = active_model_root / policy["checkpoint"]
    asset = active_model_root / "assets/hand_landmarker.task"
    if sha256(checkpoint) != policy["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match packaged frozen policy")
    if sha256(asset) != policy["model_asset_sha256"]:
        raise ValueError("MediaPipe asset does not match packaged frozen policy")
    model, normalizer, metadata, device = load_checkpoint(checkpoint)
    if metadata["architecture"] != policy["architecture"]:
        raise ValueError("Checkpoint architecture differs from packaged policy")
    if metadata["provenance"]["extractor"] != EXTRACTOR:
        raise ValueError("Feature extractor differs from checkpoint provenance")
    if metadata["provenance"]["model_asset_sha256"] != policy["model_asset_sha256"]:
        raise ValueError("Checkpoint expects a different MediaPipe asset")
    return policy, checkpoint, model, normalizer, metadata, device


def export_frames(prediction: Path, output_dir: Path) -> None:
    payload = read_json(prediction)
    cap = cv2.VideoCapture(payload["source_video"])
    if not cap.isOpened():
        raise ValueError("Cannot reopen input video for frame export")
    exported = []
    try:
        for boundary in payload["boundaries"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, boundary["frame_index"])
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {boundary['frame_index']}")
            path = output_dir / f"cut_{boundary['boundary_index']:03d}_frame_{boundary['frame_index']:06d}.jpg"
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"Cannot write {path}")
            exported.append({**boundary, "image": path.name})
    finally:
        cap.release()
    write_json(output_dir / "manifest.json", {"prediction": str(prediction.resolve()), "frames": exported})


def infer(video: Path, output: Path, export: bool) -> None:
    video, output = video.resolve(), output.resolve()
    if not video.is_file() or output.suffix != ".json":
        raise ValueError("--video must be a file and --output must end in .json")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    policy, checkpoint_path, model, normalizer, checkpoint, device = policy_and_model()
    feature_path = output.with_suffix(".features.npz")
    extract_video(video, feature_path, model=MODEL_ROOT / "assets/hand_landmarker.task")
    with np.load(feature_path, allow_pickle=False) as raw:
        sequence = {key: raw[key] for key in raw.files}
    cfg = checkpoint["config"]
    probabilities = predict_probabilities(model, normalize(sequence, normalizer), device,
                                          cfg["window"], cfg["stride"], cfg["batch_size"])
    payload = prediction_payload(video, float(sequence["fps"]), probabilities, policy["threshold"],
                                 policy["merge_gap"], policy.get("boundary_policy", "peak"),
                                 policy.get("event_policies"), policy.get("event_search_fractions"))
    payload["inference"] = {
        "architecture": policy["architecture"], "checkpoint_sha256": sha256(checkpoint_path),
        "threshold": policy["threshold"], "merge_gap": policy["merge_gap"],
        "boundary_policy": policy.get("boundary_policy", "peak"), "uses_gt": False,
        "package_manifest": "model/package_manifest.json",
    }
    write_json(output, payload)
    np.savez_compressed(output.with_suffix(".probabilities.npz"), probabilities=probabilities)
    if export:
        out_dir = output.with_name(output.stem + "_cut_frames")
        out_dir.mkdir(parents=True, exist_ok=False)
        export_frames(output, out_dir)
    print(json.dumps({"output": str(output), "events": len(payload["boundaries"]),
                      "threshold": policy["threshold"]}, ensure_ascii=False))


def verify() -> None:
    manifest = read_json(MODEL_ROOT / "package_manifest.json")
    bad = []
    for rel, expected in manifest["files"].items():
        path = PACKAGE / rel
        if not path.is_file() or sha256(path) != expected:
            bad.append(rel)
    if bad:
        raise ValueError("Package checksum mismatch: " + ", ".join(bad))
    policy_and_model()
    print(json.dumps({"status": "ok", "model": manifest["model"], "policy": manifest["policy"]}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    infer_parser = commands.add_parser("infer")
    infer_parser.add_argument("--video", type=Path, required=True)
    infer_parser.add_argument("--output", type=Path, required=True)
    infer_parser.add_argument("--export-frames", action="store_true")
    args = parser.parse_args()
    verify() if args.command == "verify" else infer(args.video, args.output, args.export_frames)


if __name__ == "__main__":
    main()
