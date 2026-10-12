"""Shared, label-independent sequence processing and strict event evaluation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
_experiment = Path(os.environ.get("SUPERCUT_EXPERIMENT", "experiments/fridge_v1")).expanduser()
ROOT = (_experiment if _experiment.is_absolute() else PROJECT_ROOT / _experiment).resolve()


def experiment_path(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def dataset_config():
    config = read_json(ROOT / "dataset.json")
    configured_root = Path(os.environ.get("SUPERCUT_DATA_ROOT", config["root"])).expanduser()
    config["root"] = str(configured_root.resolve())
    return config


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_intervals(intervals, n):
    previous = -1
    for start, end in intervals:
        if not (isinstance(start, int) and isinstance(end, int) and previous < start <= end < n):
            raise ValueError(f"Invalid, overlapping or unordered GT interval: {start, end}; frames={n}")
        previous = end


def frame_labels(n, intervals):
    validate_intervals(intervals, n)
    labels = np.zeros(n, dtype=np.float32)
    for start, end in intervals:
        labels[start:end + 1] = 1
    return labels


def window_starts(n, window=96, stride=24):
    if n <= 0 or window <= 0 or not 0 < stride <= window:
        raise ValueError("Invalid window dimensions")
    starts = list(range(0, max(1, n - window + 1), stride))
    if n > window and starts[-1] != n - window:
        starts.append(n - window)
    return starts


def decode_events(probabilities, threshold, merge_gap, boundary_policy="peak"):
    """Merge positive runs and choose a cut frame from each run.

    ``third_event_run_end`` preserves peak selection for the first two events
    and emits the end of the third positive run, immediately before the
    separately annotated task-completion hand raise.
    """
    if boundary_policy not in ("peak", "third_event_run_end"):
        raise ValueError(f"Unsupported boundary policy: {boundary_policy}")
    p = np.asarray(probabilities)
    if p.ndim != 1 or not np.isfinite(p).all():
        raise ValueError("Probabilities must be a finite vector")
    indices = np.flatnonzero(p >= threshold)
    if len(indices) == 0:
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > merge_gap + 1) + 1)
    events = [int(group[np.argmax(p[group])]) for group in groups]
    if boundary_policy == "third_event_run_end" and len(groups) >= 3:
        events[2] = int(groups[2][-1])
    return events


def decode_ordered_events(probabilities, threshold, merge_gap, event_policies, event_search_fractions=None):
    """Decode one highest-confidence run per ordered event head.

    A three-head model represents the three cuts separately. This prevents an
    early false-positive run from being mistaken for the semantic third cut.
    """
    p = np.asarray(probabilities)
    if p.ndim != 2 or p.shape[1] != len(event_policies) or not np.isfinite(p).all():
        raise ValueError("Ordered probabilities must have one finite column per event policy")
    if event_search_fractions is None:
        event_search_fractions = [(0.0, 1.0)] * len(event_policies)
    if len(event_search_fractions) != len(event_policies):
        raise ValueError("Each event head requires a search fraction range")
    selected = []
    for event_index, policy in enumerate(event_policies):
        if policy not in ("peak", "run_end"):
            raise ValueError(f"Unsupported ordered event policy: {policy}")
        values = p[:, event_index]
        lower, upper = event_search_fractions[event_index]
        if not (0.0 <= lower <= upper <= 1.0):
            raise ValueError("Search fractions must be ordered values in [0, 1]")
        first = int(np.ceil(len(values) * lower))
        last = int(np.floor((len(values) - 1) * upper))
        indices = np.flatnonzero(values >= threshold)
        indices = indices[(indices >= first) & (indices <= last)]
        if len(indices) == 0:
            continue
        groups = np.split(indices, np.flatnonzero(np.diff(indices) > merge_gap + 1) + 1)
        group = max(groups, key=lambda item: float(values[item].max()))
        frame = int(group[-1] if policy == "run_end" else group[np.argmax(values[group])])
        selected.append((event_index, frame))
    return selected


def decode_configured_events(probabilities, threshold, merge_gap, boundary_policy="peak", event_policies=None,
                             event_search_fractions=None):
    p = np.asarray(probabilities)
    if p.ndim == 1:
        return decode_events(p, threshold, merge_gap, boundary_policy)
    if event_policies is None:
        raise ValueError("Ordered probabilities require event policies")
    return [frame for _, frame in decode_ordered_events(p, threshold, merge_gap, event_policies, event_search_fractions)]


def evaluate_events(predictions, intervals, frames, fps):
    validate_intervals(intervals, frames)
    if any(not 0 <= int(p) < frames for p in predictions):
        raise ValueError("Prediction outside source video")
    matched, details = set(), []
    for frame in sorted(int(p) for p in predictions):
        candidates = [i for i, (a, b) in enumerate(intervals) if a <= frame <= b and i not in matched]
        hit = candidates[0] if candidates else None
        if hit is not None:
            matched.add(hit)
        distances = [frame - a if frame < a else frame - b if frame > b else 0 for a, b in intervals]
        nearest = min(range(len(distances)), key=lambda i: abs(distances[i])) if distances else None
        details.append({"frame_index": frame, "matched_gt": hit,
                        "nearest_gt": nearest,
                        "signed_distance_frames": distances[nearest] if nearest is not None else None})
    tp = len(matched)
    return {"tp": tp, "fp": len(predictions) - tp, "fn": len(intervals) - tp,
            "duration_seconds": frames / fps, "predictions": details,
            "missed_gt": [{"index": i, "interval": v} for i, v in enumerate(intervals) if i not in matched]}


def aggregate(results):
    values = list(results.values()) if isinstance(results, dict) else list(results)
    tp, fp, fn = (sum(r[k] for r in values) for k in ("tp", "fp", "fn"))
    duration = sum(r["duration_seconds"] for r in values)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    distances = [abs(p["signed_distance_frames"]) for r in values for p in r["predictions"]
                 if p["signed_distance_frames"] is not None]
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "false_positives_per_minute": fp * 60 / duration if duration else 0.0,
            "mean_nearest_gt_distance_frames": float(np.mean(distances)) if distances else None,
            "p95_nearest_gt_distance_frames": float(np.percentile(distances, 95)) if distances else None,
            "duration_seconds": duration}


def selection_key(metrics):
    r, p = metrics["recall"], metrics["precision"]
    return (1, p, r) if r >= 0.95 else (0, r, p)


def fit_normalizer(sequences):
    """Fit only on supplied training sequences; missing coordinates contribute nothing."""
    x = np.concatenate([s["x"] for s in sequences])
    valid = np.concatenate([s["valid"] for s in sequences])
    count = valid.sum(0)
    mean = np.where(count > 0, (x * valid).sum(0) / np.maximum(count, 1), 0)
    var = (((x - mean) ** 2) * valid).sum(0) / np.maximum(count, 1)
    scale = np.maximum(np.sqrt(var), 1e-3)
    mean[-4:], scale[-4:] = 0, 1  # Presence masks retain binary semantics.
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32)}


def normalize(sequence, normalizer):
    return (((sequence["x"] - normalizer["mean"]) / normalizer["scale"])
            * sequence["valid"]).astype(np.float32)


def prediction_payload(video, fps, probabilities, threshold, merge_gap, boundary_policy="peak", event_policies=None,
                       event_search_fractions=None):
    p = np.asarray(probabilities)
    if p.ndim == 1:
        candidates = [(None, frame) for frame in decode_events(p, threshold, merge_gap, boundary_policy)]
    else:
        if event_policies is None:
            raise ValueError("Ordered probabilities require event policies")
        candidates = decode_ordered_events(p, threshold, merge_gap, event_policies, event_search_fractions)
    candidates.sort(key=lambda item: item[1])
    points = [frame for _, frame in candidates]
    n = len(probabilities)
    # Clip endpoints are metadata, never gesture detections or scored events.
    starts = sorted(set([0] + points))
    return {"schema": "gesture-temporal.prediction.v1", "source_video": str(Path(video).resolve()),
            "fps": fps, "frame_count": n,
            "boundaries": [{"boundary_index": i,
                            "kind": "learned_gesture" if event_index is None else f"learned_gesture_{event_index + 1}",
                            "frame_index": f, "timestamp_seconds": f / fps,
                            "score": float(p[f] if p.ndim == 1 else p[f, event_index])}
                           for i, (event_index, f) in enumerate(candidates)],
            "clip_endpoints": {"start_frame": 0, "end_frame": n - 1},
            "segments": [{"start_frame": a, "end_frame": b - 1,
                          "start_seconds": a / fps, "end_seconds_exclusive": b / fps}
                         for a, b in zip(starts, starts[1:] + [n])]}
