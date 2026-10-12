"""Small offline models. Training and selection never load test features."""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from core import (ROOT, PROJECT_ROOT, aggregate, decode_configured_events, evaluate_events, fit_normalizer,
                  frame_labels, normalize, read_json, selection_key, sha256,
                  window_starts, write_json)
from data import EXTRACTOR, load_features, verify_manifest


class TemporalModel(nn.Module):
    def __init__(self, architecture, features=272, hidden=64, dropout=0.2, event_count=1):
        super().__init__()
        self.architecture = architecture
        self.event_count = event_count
        if event_count < 1:
            raise ValueError("event_count must be positive")
        if architecture == "bilstm":
            self.encoder = nn.LSTM(features, hidden, num_layers=2, batch_first=True,
                                   dropout=dropout, bidirectional=True)
            self.head = nn.Linear(hidden * 2, event_count)
        elif architecture == "tcn":
            layers = []
            for dilation in (1, 2, 4, 8):
                layers += [nn.Conv1d(features, hidden, 3, padding=dilation, dilation=dilation),
                           nn.ReLU(), nn.Dropout(dropout)]
                features = hidden
            self.encoder = nn.Sequential(*layers)
            self.head = nn.Conv1d(hidden, event_count, 1)
        else:
            raise ValueError(architecture)

    def forward(self, x, lengths=None):
        if self.architecture == "bilstm":
            if lengths is not None and bool((lengths != x.shape[1]).any()):
                packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
                packed, _ = self.encoder(packed)
                output, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, total_length=x.shape[1])
            else:
                output, _ = self.encoder(x)
            output = self.head(output)
        else:
            output = self.head(self.encoder(x.transpose(1, 2))).transpose(1, 2)
        return output.squeeze(-1) if self.event_count == 1 else output


class Windows(Dataset):
    def __init__(self, sequences, records, normalizer, window=96, stride=24, event_count=1):
        self.window = window
        self.event_count = event_count
        self.x = [normalize(s, normalizer) for s in sequences]
        self.y = [self.labels(r) for r in records]
        self.indices = [(i, start) for i, x in enumerate(self.x) for start in window_starts(len(x), window, stride)]

    def labels(self, record):
        if self.event_count == 1:
            return frame_labels(record["frames"], record["gt_intervals"])
        if len(record["gt_intervals"]) != self.event_count:
            raise ValueError("Ordered training requires one interval per event head")
        y = np.zeros((record["frames"], self.event_count), np.float32)
        for event_index, (start, end) in enumerate(record["gt_intervals"]):
            y[start:end + 1, event_index] = 1
        return y

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        sequence, start = self.indices[i]
        length = min(self.window, len(self.x[sequence]) - start)
        x = np.zeros((self.window, self.x[sequence].shape[1]), np.float32)
        target_shape = (self.window,) if self.event_count == 1 else (self.window, self.event_count)
        y = np.zeros(target_shape, np.float32)
        mask = np.zeros(target_shape, np.float32)
        x[:length] = self.x[sequence][start:start + length]
        y[:length] = self.y[sequence][start:start + length]
        mask[:length] = 1
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask), length


def setup(config):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    torch.use_deterministic_algorithms(True)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def predict_probabilities(model, x, device, window=96, stride=24, batch_size=16):
    model.eval()
    starts = window_starts(len(x), window, stride)
    sums = np.zeros((len(x), model.event_count), np.float64) if model.event_count > 1 else np.zeros(len(x), np.float64)
    counts = np.zeros(len(x), np.int32)
    for offset in range(0, len(starts), batch_size):
        chunk = starts[offset:offset + batch_size]
        batch = np.zeros((len(chunk), window, x.shape[1]), np.float32)
        lengths = []
        for i, start in enumerate(chunk):
            length = min(window, len(x) - start)
            batch[i, :length] = x[start:start + length]
            lengths.append(length)
        probs = model(torch.from_numpy(batch).to(device), torch.tensor(lengths)).sigmoid().cpu().numpy()
        for start, length, p in zip(chunk, lengths, probs):
            sums[start:start + length] += p[:length]
            counts[start:start + length] += 1
    if not (counts > 0).all():
        raise RuntimeError("Incomplete window coverage")
    return (sums / counts[:, None] if model.event_count > 1 else sums / counts).astype(np.float32)


def load_checkpoint(path, device=None):
    # Checkpoints are locally generated. No arbitrary external checkpoints are loaded.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    device = setup(config) if device is None else device
    model = TemporalModel(checkpoint["architecture"], hidden=config["hidden"], dropout=config["dropout"],
                          event_count=config.get("event_count", 1))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    normalizer = {k: checkpoint["normalizer"][k].numpy() for k in ("mean", "scale")}
    return model, normalizer, checkpoint, device


def train(architecture):
    config = read_json(ROOT / "config.json")
    manifest = read_json(ROOT / "splits/manifest.json")
    verify_manifest(manifest)
    run = ROOT / "runs" / architecture
    if (ROOT / "runs/selection.json").exists():
        raise RuntimeError("Model selection is frozen; use a separate workspace for another experiment")
    run.mkdir(parents=True, exist_ok=True)
    if (run / "best.pt").exists():
        raise RuntimeError("Existing run must not be silently overwritten")
    device = setup(config)
    train_records = [r for r in manifest["records"] if r["split"] == "train"]
    val_records = [r for r in manifest["records"] if r["split"] == "val"]
    train_sequences = [load_features(r) for r in train_records]
    val_sequences = [load_features(r) for r in val_records]
    normalizer = fit_normalizer(train_sequences)
    event_count = config.get("event_count", 1)
    train_set = Windows(train_sequences, train_records, normalizer, config["window"], config["stride"], event_count)
    val_set = Windows(val_sequences, val_records, normalizer, config["window"], config["stride"], event_count)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"])
    if event_count == 1:
        positives = sum(float(frame_labels(r["frames"], r["gt_intervals"]).sum()) for r in train_records)
        positive_weight = (sum(r["frames"] for r in train_records) - positives) / positives
    else:
        positives = np.array([sum(r["gt_intervals"][i][1] - r["gt_intervals"][i][0] + 1
                                  for r in train_records) for i in range(event_count)], dtype=np.float32)
        positive_weight = (sum(r["frames"] for r in train_records) - positives) / positives
    loss_weight = torch.tensor(positive_weight, device=device) if event_count > 1 else torch.tensor(float(positive_weight), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=loss_weight, reduction="none")
    model = TemporalModel(architecture, hidden=config["hidden"], dropout=config["dropout"], event_count=event_count).to(device)
    warm_start = config.get("warm_start_checkpoint")
    warm_start_provenance = None
    if warm_start:
        warm_start_path = Path(warm_start).expanduser().resolve()
        if not warm_start_path.is_file():
            raise FileNotFoundError(f"Warm-start checkpoint missing: {warm_start_path}")
        base = torch.load(warm_start_path, map_location="cpu", weights_only=True)
        if base.get("architecture") != architecture:
            raise ValueError("Warm-start architecture differs from requested model")
        base_event_count = base.get("config", {}).get("event_count", 1)
        if base_event_count != event_count:
            raise ValueError("Warm-start event head count differs from current configuration")
        model.load_state_dict(base["state_dict"])
        warm_start_provenance = {
            "checkpoint": str(warm_start_path), "checkpoint_sha256": sha256(warm_start_path),
            "architecture": base["architecture"], "epoch": base.get("epoch"),
            "normalizer_refit_on_current_train_split": True,
        }
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    best_loss, stale = float("inf"), 0
    started = time.monotonic()
    provenance = {"manifest_sha256": sha256(ROOT / "splits/manifest.json"),
                  "config_sha256": sha256(ROOT / "config.json"),
                  "model_asset_sha256": sha256(ROOT / "assets/hand_landmarker.task"),
                  "normalizer_fit_sessions": [r["session_id"] for r in train_records],
                  "validation_sessions": [r["session_id"] for r in val_records],
                  "positive_weight": positive_weight.tolist() if event_count > 1 else positive_weight,
                  "event_count": event_count, "device": str(device),
                  "torch_version": str(torch.__version__), "extractor": EXTRACTOR,
                  "warm_start": warm_start_provenance,
                  "code_sha256": {p.name: sha256(p) for p in [PROJECT_ROOT / "core.py", PROJECT_ROOT / "data.py", PROJECT_ROOT / "learning.py"]}}
    write_json(run / "provenance.json", provenance)
    log_path = run / "epochs.jsonl"
    with log_path.open("x") as log:
        for epoch in range(1, config["max_epochs"] + 1):
            model.train()
            total_loss, total_frames = 0.0, 0
            for x, y, mask, lengths in train_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                losses = loss_fn(model(x, lengths), y) * mask
                loss = losses.sum() / mask.sum()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(losses.detach().sum())
                total_frames += int(mask.sum())
            model.eval()
            val_loss, val_frames = 0.0, 0
            with torch.inference_mode():
                for x, y, mask, lengths in val_loader:
                    x, y, mask = x.to(device), y.to(device), mask.to(device)
                    val_loss += float((loss_fn(model(x, lengths), y) * mask).sum())
                    val_frames += int(mask.sum())
            val_loss /= val_frames
            improved = val_loss < best_loss - 1e-6
            if improved:
                best_loss, stale = val_loss, 0
                checkpoint = {"schema": "gesture-temporal.checkpoint.v1", "architecture": architecture,
                              "config": config, "epoch": epoch, "val_loss": val_loss,
                              "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                              "normalizer": {k: torch.from_numpy(v) for k, v in normalizer.items()},
                              "provenance": provenance}
                torch.save(checkpoint, run / "best.tmp")
                (run / "best.tmp").replace(run / "best.pt")
            else:
                stale += 1
            entry = {"epoch": epoch, "train_loss": total_loss / total_frames, "val_loss": val_loss,
                     "best": improved, "elapsed_seconds": time.monotonic() - started}
            import json
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(f"{architecture} {epoch:03d}: train={entry['train_loss']:.5f} val={val_loss:.5f} best={improved}", flush=True)
            if stale >= config["patience"]:
                break
    model, normalizer, checkpoint, device = load_checkpoint(run / "best.pt", device)
    probabilities = {r["name"]: predict_probabilities(model, normalize(s, normalizer), device,
                     config["window"], config["stride"], config["batch_size"])
                     for r, s in zip(val_records, val_sequences)}
    np.savez_compressed(run / "validation_probabilities.npz", **probabilities)
    write_json(run / "training_summary.json", {"architecture": architecture, "best_epoch": checkpoint["epoch"],
               "epochs": epoch, "best_val_loss": best_loss, "elapsed_seconds": time.monotonic() - started,
               "parameters": sum(p.numel() for p in model.parameters()), "device": str(device)})


def select():
    destination = ROOT / "runs/selection.json"
    if destination.exists():
        raise RuntimeError("Selection is already frozen")
    config = read_json(ROOT / "config.json")
    configured_architectures = config.get("selection_architectures")
    architectures = tuple(configured_architectures if configured_architectures is not None else
                           (name for name in ("bilstm", "tcn") if (ROOT / "runs" / name / "best.pt").is_file()))
    if not architectures or any(name not in ("bilstm", "tcn") for name in architectures):
        raise ValueError("selection_architectures must be a nonempty subset of bilstm and tcn")
    boundary_policy = config.get("boundary_policy", "peak")
    event_policies = config.get("event_policies")
    event_search_fractions = config.get("event_search_fractions")
    manifest = read_json(ROOT / "splits/manifest.json")
    verify_manifest(manifest)
    records = [r for r in manifest["records"] if r["split"] == "val"]
    all_results, best = [], None
    for architecture in architectures:
        run = ROOT / "runs" / architecture
        provenance = read_json(run / "provenance.json")
        if provenance["manifest_sha256"] != sha256(ROOT / "splits/manifest.json") or provenance["config_sha256"] != sha256(ROOT / "config.json"):
            raise ValueError("Training inputs changed before selection")
        with np.load(run / "validation_probabilities.npz") as raw:
            probabilities = {k: raw[k] for k in raw.files}
        for threshold in config["thresholds"]:
            for merge_gap in config["merge_gaps"]:
                results = {r["name"]: evaluate_events(decode_configured_events(probabilities[r["name"]], threshold, merge_gap,
                                                                                boundary_policy, event_policies, event_search_fractions),
                           r["gt_intervals"], r["frames"], r["fps"]) for r in records}
                metrics = aggregate(results)
                candidate = {"architecture": architecture, "threshold": threshold, "merge_gap": merge_gap,
                             "boundary_policy": boundary_policy, "event_policies": event_policies,
                             "event_search_fractions": event_search_fractions,
                             "validation_metrics": metrics}
                all_results.append(candidate)
                # Ties retain the first entry in fixed architecture/threshold/gap order.
                if best is None or selection_key(metrics) > selection_key(best["validation_metrics"]):
                    best = candidate
    checkpoint = ROOT / "runs" / best["architecture"] / "best.pt"
    best = dict(best, schema="gesture-temporal.selection.v1", checkpoint=str(checkpoint.relative_to(ROOT)),
                checkpoint_sha256=sha256(checkpoint), manifest_sha256=sha256(ROOT / "splits/manifest.json"),
                config_sha256=sha256(ROOT / "config.json"), target_recall=0.95,
                target_met_on_validation=best["validation_metrics"]["recall"] >= 0.95,
                test_used_for_selection=False,
                boundary_policy=boundary_policy,
                event_policies=event_policies,
                event_search_fractions=event_search_fractions,
                tie_break="first in fixed order: bilstm then tcn; ascending threshold then gap")
    per_model = {}
    for architecture in architectures:
        options = [r for r in all_results if r["architecture"] == architecture]
        per_model[architecture] = max(options, key=lambda x: selection_key(x["validation_metrics"]))
        per_model[architecture] = dict(per_model[architecture], checkpoint_sha256=sha256(ROOT / f"runs/{architecture}/best.pt"))
    best["per_model"] = per_model
    write_json(ROOT / "reports/validation_search.json", all_results)
    write_json(destination, best)
    print(best, flush=True)
    return best
