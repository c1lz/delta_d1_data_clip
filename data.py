"""Immutable raw-session manifest and independent MediaPipe feature extraction."""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

from core import ROOT, dataset_config, read_json, sha256, validate_intervals, write_json

EXTRACTOR = {"version": 1, "fps": 30.0, "process_width": 1280,
             "roi": [0.30, 0.30, 0.70, 0.72], "confidence": 0.55,
             "feature_dim": 272, "coordinate_dim": 134,
             "hand_order": "image palm x then y", "running_mode": "VIDEO"}


def prepare():
    config = dataset_config()
    splits = config["splits"]
    if set(splits) != {"train", "val", "test"} or any(not names for names in splits.values()):
        raise ValueError("dataset.json must define nonempty train, val and test session lists")
    path = ROOT / "splits/manifest.json"
    if path.exists():
        manifest = read_json(path)
        verify_manifest(manifest, hashes=True)
        print("Existing immutable manifest verified", flush=True)
        return manifest
    gt_path = ROOT / "annotations/gt.json"
    gt = read_json(gt_path)["tasks"]
    records = []
    for split, names in splits.items():
        for name in names:
            folder = Path(config["root"]) / config["batch"] / name
            video = folder / "cam0/cam0-00000.mp4"
            session = read_json(folder / "session.json")
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise ValueError(f"Unreadable video: {video}")
            n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            if fps != 30 or n < 1:
                raise ValueError(f"Unexpected video properties: {video}: {n}, {fps}")
            validate_intervals(gt[name], n)
            record = {"name": name, "split": split, "session_id": session["session_id"],
                      "device_id": session["device"]["device_id"], "started_at": session["started_at"],
                      "video": str(video), "sha256": sha256(video), "frames": n, "fps": fps,
                      "gt_intervals": gt[name]}
            records.append(record)
            print(f"Frozen {split}: {name}, {n} frames, {len(gt[name])} GT", flush=True)
    manifest = {"schema": "gesture-temporal.split.v1", "frame_index_base": 0,
                "intervals": "inclusive", "gt_sha256": sha256(gt_path), "records": records}
    verify_manifest(manifest)
    write_json(path, manifest)
    for split in splits:
        write_json(ROOT / f"splits/{split}.json", [r for r in records if r["split"] == split])
    return manifest


def verify_manifest(manifest, hashes=False):
    seen = {k: set() for k in ("session_id", "video", "sha256", "name")}
    for r in manifest["records"]:
        for key, values in seen.items():
            if r[key] in values:
                raise ValueError(f"Duplicate source {key}: {r[key]}")
            values.add(r[key])
        validate_intervals(r["gt_intervals"], r["frames"])
        if hashes and sha256(r["video"]) != r["sha256"]:
            raise ValueError(f"Source changed: {r['video']}")
    splits = dataset_config()["splits"]
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("dataset.json must define train, val and test splits")
    for split, names in splits.items():
        if {r["name"] for r in manifest["records"] if r["split"] == split} != set(names):
            raise ValueError(f"Frozen split changed: {split}")
    gt_path = ROOT / "annotations/gt.json"
    if sha256(gt_path) != manifest["gt_sha256"]:
        raise ValueError("GT file changed after split freeze")
    gt = read_json(gt_path)["tasks"]
    if any(r["gt_intervals"] != gt[r["name"]] for r in manifest["records"]):
        raise ValueError("Manifest annotations differ from frozen GT")


def hand_descriptor(points):
    axis = points[9, :2] - points[0, :2]
    scale = float(np.linalg.norm(axis))
    if scale < 1e-6:
        return None
    ux = axis / scale
    uy = np.array([-ux[1], ux[0]])
    relative = points - points[0]
    shape = np.column_stack((relative[:, :2] @ ux / scale,
                             relative[:, :2] @ uy / scale, relative[:, 2] / scale))
    return np.concatenate([shape.flatten(), points[0, :2], [scale]]).astype(np.float32)


def assemble_features(coordinates, presence, fps):
    n = len(coordinates)
    valid = np.zeros((n, 134), dtype=bool)
    valid[:, :66] = presence[:, 0:1]
    valid[:, 66:132] = presence[:, 1:2]
    valid[:, 132:] = presence.all(1, keepdims=True)
    velocity_valid = np.zeros_like(valid)
    velocity_valid[1:] = valid[1:] & valid[:-1]
    velocity = np.zeros_like(coordinates)
    velocity[1:] = (coordinates[1:] - coordinates[:-1]) * fps
    velocity *= velocity_valid
    motion_present = np.zeros_like(presence)
    motion_present[1:] = presence[1:] & presence[:-1]
    x = np.concatenate([coordinates, velocity, presence, motion_present], axis=1).astype(np.float32)
    masks = np.concatenate([valid, velocity_valid, np.ones((n, 4), dtype=bool)], axis=1)
    x *= masks
    return x, masks


def extract_video(video, output, model=None, expected_sha=None):
    """No GT or split information is read by this function."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    video, output = Path(video), Path(output)
    model = Path(model or ROOT / "assets/hand_landmarker.task")
    digest = sha256(video)
    if expected_sha and digest != expected_sha:
        raise ValueError(f"Source checksum mismatch: {video}")
    metadata = {"extractor": EXTRACTOR, "video_sha256": digest,
                "model_sha256": sha256(model), "mediapipe_version": mp.__version__,
                "opencv_version": cv2.__version__, "source_video": str(video.resolve())}
    meta_path = output.with_suffix(".json")
    if output.exists() and meta_path.exists():
        old = read_json(meta_path)
        if all(old.get(k) == v for k, v in metadata.items()):
            if old.get("feature_sha256") != sha256(output):
                raise ValueError(f"Feature cache checksum mismatch: {output}")
            print(f"Cached: {video.parent.parent.name}", flush=True)
            return old
        raise ValueError(f"Cache incompatible; use a new output path: {output}")
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {video}")
    n, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), float(capture.get(cv2.CAP_PROP_FPS))
    if n < 1 or abs(fps - EXTRACTOR["fps"]) > 0.01:
        capture.release()
        raise ValueError("This checkpoint requires nonempty 30 FPS video")
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.55, min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55)
    coordinates = np.zeros((n, 134), dtype=np.float32)
    presence = np.zeros((n, 2), dtype=bool)
    decoded = np.zeros(n, dtype=bool)
    try:
        with vision.HandLandmarker.create_from_options(options) as detector:
            for i in range(n):
                ok, frame = capture.read()
                if not ok:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, i + 1)
                    continue  # Preserve original frame index and explicit missing mask.
                decoded[i] = True
                h, w = frame.shape[:2]
                if w > EXTRACTOR["process_width"]:
                    frame = cv2.resize(frame, (EXTRACTOR["process_width"], round(h * EXTRACTOR["process_width"] / w)))
                h, w = frame.shape[:2]
                x0, y0, x1, y1 = EXTRACTOR["roi"]
                ix0, iy0, ix1, iy1 = round(x0*w), round(y0*h), round(x1*w), round(y1*h)
                crop = frame[iy0:iy1, ix0:ix1]
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), round(i * 1000 / fps))
                hands = []
                for raw in result.hand_landmarks:
                    points = np.array([[p.x, p.y, p.z] for p in raw], dtype=np.float32)
                    points[:, 0] = (points[:, 0] * (ix1-ix0) + ix0) / w
                    points[:, 1] = (points[:, 1] * (iy1-iy0) + iy0) / h
                    points[:, 2] *= (ix1-ix0) / w
                    descriptor = hand_descriptor(points)
                    if descriptor is not None:
                        palm = points[[0, 5, 9, 13, 17], :2].mean(0)
                        hands.append((palm, descriptor))
                hands.sort(key=lambda item: (float(item[0][0]), float(item[0][1])))
                for j, (_, descriptor) in enumerate(hands[:2]):
                    coordinates[i, j*66:(j+1)*66] = descriptor
                    presence[i, j] = True
                if presence[i].all():
                    coordinates[i, 132:] = coordinates[i, 129:131] - coordinates[i, 63:65]
                if i % 600 == 0:
                    print(f"{video.parent.parent.name}: {i}/{n}", flush=True)
    finally:
        capture.release()
    if not decoded.any():
        raise ValueError(f"No frames could be decoded: {video}")
    x, valid = assemble_features(coordinates, presence, fps)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite extracted features")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".tmp").open("wb") as stream:
        np.savez_compressed(stream, x=x, valid=valid, presence=presence, decoded=decoded,
                            frame_indices=np.arange(n), fps=np.array(fps))
    output.with_suffix(".tmp").replace(output)
    metadata.update({"frames": n, "fps": fps, "decoded_frames": int(decoded.sum()),
                     "both_hands_frames": int(presence.all(1).sum()),
                     "any_hand_frames": int(presence.any(1).sum()), "feature_sha256": sha256(output)})
    write_json(meta_path, metadata)
    print(f"Finished {video.parent.parent.name}: {metadata['both_hands_frames']}/{n} two-hand frames", flush=True)
    return metadata


def extract_record(record):
    return extract_video(record["video"], ROOT / f"features/{record['name']}.npz", expected_sha=record["sha256"])


def load_features(record):
    path = ROOT / f"features/{record['name']}.npz"
    meta = read_json(path.with_suffix(".json"))
    if meta["video_sha256"] != record["sha256"] or meta["extractor"] != EXTRACTOR or sha256(path) != meta["feature_sha256"]:
        raise ValueError(f"Feature provenance mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k] for k in data.files}
    if len(result["x"]) != record["frames"] or not np.array_equal(result["frame_indices"], np.arange(record["frames"])):
        raise ValueError("Feature frame indices changed")
    return result
