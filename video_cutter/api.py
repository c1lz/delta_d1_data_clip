"""Authenticated, serial HTTP API around the frozen video cutter service.

The desktop GUI and this module deliberately share :class:`InferenceService`.
This module owns only HTTP validation, durable queue state and result delivery.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import asynccontextmanager
import csv
from datetime import UTC, datetime
from io import StringIO
import json
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
import threading
import time
from typing import Any, Callable
from uuid import uuid4
import zipfile

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .service import (DEFAULT_DATASET_ROOT, DEFAULT_EXPORT_ROOT, Boundary, CutSettings, CutterError,
                      InferenceService, VideoJob, deduplicate_boundaries, discover_delta_sessions, file_digest)


LOGGER_NAME = "data_clip.api"
TERMINAL_ITEM_STATES = {"completed", "failed", "cancelled", "interrupted"}
QUEUEABLE_STATES = {"queued", "export_queued"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def json_load(value: str | None, fallback: Any) -> Any:
    return json.loads(value) if value else fallback


def resolved_within(value: str | Path, root: Path, description: str) -> Path:
    path, trusted_root = Path(value).expanduser().resolve(), Path(root).expanduser().resolve()
    try:
        path.relative_to(trusted_root)
    except ValueError as error:
        raise ValueError(f"{description} 必须位于 {trusted_root} 下") from error
    return path


@dataclass(frozen=True)
class ApiConfig:
    api_key: str
    dataset_root: Path = DEFAULT_DATASET_ROOT
    export_root: Path = DEFAULT_EXPORT_ROOT
    database_path: Path = Path(".runtime/api_jobs.sqlite")
    host: str = "127.0.0.1"
    port: int = 18082
    cors_origins: tuple[str, ...] = ()
    model_version: str | None = None

    @classmethod
    def from_environment(cls) -> "ApiConfig":
        api_key = os.environ.get("DATA_CLIP_API_KEY", "")
        if not api_key:
            raise RuntimeError("必须设置 DATA_CLIP_API_KEY")
        project_root = Path(__file__).resolve().parents[1]
        database = Path(os.environ.get("DATA_CLIP_API_DATABASE", project_root / ".runtime/api_jobs.sqlite"))
        origins = tuple(value.strip() for value in os.environ.get("DATA_CLIP_CORS_ORIGINS", "").split(",") if value.strip())
        return cls(
            api_key=api_key,
            dataset_root=Path(os.environ.get("DATA_CLIP_DATASET_ROOT", DEFAULT_DATASET_ROOT)),
            export_root=Path(os.environ.get("DATA_CLIP_EXPORT_ROOT", DEFAULT_EXPORT_ROOT)),
            database_path=database,
            host=os.environ.get("DATA_CLIP_API_HOST", "127.0.0.1"),
            port=int(os.environ.get("DATA_CLIP_API_PORT", "18082")),
            cors_origins=origins,
            model_version=os.environ.get("DATA_CLIP_MODEL_VERSION") or None,
        )

    def normalised(self) -> "ApiConfig":
        if not self.api_key:
            raise ValueError("API Key 不能为空")
        if not 1 <= self.port <= 65535:
            raise ValueError("API 端口必须在 1 到 65535 之间")
        return ApiConfig(self.api_key, self.dataset_root.expanduser().resolve(), self.export_root.expanduser().resolve(),
                         self.database_path.expanduser().resolve(), self.host, self.port, self.cors_origins, self.model_version)


class SettingsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    threshold: float | None = Field(default=None, ge=0, le=1)
    merge_gap: int | None = Field(default=None, ge=0)
    minimum_segment_seconds: float | None = Field(default=None, ge=0)
    export_keyframes: bool | None = None
    export_root: str | None = None


class CreateJobPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date_task_path: str
    sequence_start: int = Field(ge=0)
    sequence_end: int = Field(ge=0)
    settings: SettingsPayload = Field(default_factory=SettingsPayload)


class BoundaryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_index: int = Field(gt=0)
    score: float | None = Field(default=None, ge=0, le=1)
    origin: str = "manual"
    kind: str = "split"
    pair_id: str | None = None


class BoundariesPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=0)
    boundaries: list[BoundaryPayload]


class ApiStore:
    """Small SQLite repository. Connections are short-lived and thread-safe."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)

    def connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialise(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS api_jobs (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    date_task_path TEXT NOT NULL, sequence_start INTEGER NOT NULL, sequence_end INTEGER NOT NULL,
                    settings_json TEXT NOT NULL, status TEXT NOT NULL, error TEXT
                );
                CREATE TABLE IF NOT EXISTS api_items (
                    id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES api_jobs(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL, session_name TEXT NOT NULL, source_video TEXT,
                    stage TEXT NOT NULL, state TEXT NOT NULL, progress TEXT NOT NULL DEFAULT '',
                    error TEXT, revision INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    state_json TEXT NOT NULL, UNIQUE(job_id, sequence)
                );
                CREATE INDEX IF NOT EXISTS api_items_pending_idx ON api_items(state, created_at);
            """)
            # A process died while owning the model/ffmpeg call. Do not silently
            # rerun it; expose a deterministic retry action to the caller.
            now = utc_now()
            connection.execute("UPDATE api_items SET state='interrupted', progress='', error='服务重启前任务被中断', updated_at=? WHERE state='running'", (now,))
            connection.execute("UPDATE api_jobs SET updated_at=? WHERE status='running'", (now,))

    def create_job(self, job: dict, items: list[dict]) -> None:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("""INSERT INTO api_jobs
                (id,created_at,updated_at,date_task_path,sequence_start,sequence_end,settings_json,status,error)
                VALUES (:id,:created_at,:updated_at,:date_task_path,:sequence_start,:sequence_end,:settings_json,:status,:error)""", job)
            connection.executemany("""INSERT INTO api_items
                (id,job_id,sequence,session_name,source_video,stage,state,progress,error,revision,created_at,updated_at,state_json)
                VALUES (:id,:job_id,:sequence,:session_name,:source_video,:stage,:state,:progress,:error,:revision,:created_at,:updated_at,:state_json)""", items)
            connection.commit()

    @staticmethod
    def row(value: sqlite3.Row | None) -> dict | None:
        if value is None:
            return None
        item = dict(value)
        if "settings_json" in item:
            item["settings"] = json_load(item.pop("settings_json"), {})
        # ``state`` is already the queue-state SQL column. Keep the durable
        # VideoJob representation under a distinct key instead of overwriting
        # it when decoding ``state_json``.
        if "state_json" in item:
            item["snapshot"] = json_load(item.pop("state_json"), {})
        return item

    def job(self, job_id: str) -> dict | None:
        with self.connection() as connection:
            return self.row(connection.execute("SELECT * FROM api_jobs WHERE id=?", (job_id,)).fetchone())

    def item(self, job_id: str, sequence: int) -> dict | None:
        with self.connection() as connection:
            return self.row(connection.execute("SELECT * FROM api_items WHERE job_id=? AND sequence=?", (job_id, sequence)).fetchone())

    def items(self, job_id: str) -> list[dict]:
        with self.connection() as connection:
            return [self.row(row) for row in connection.execute("SELECT * FROM api_items WHERE job_id=? ORDER BY sequence", (job_id,))]

    def next_work(self) -> dict | None:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM api_items WHERE state IN ('queued','export_queued') ORDER BY created_at,id LIMIT 1").fetchone()
            if row is None:
                return None
            now = utc_now()
            cursor = connection.execute("UPDATE api_items SET state='running', progress='等待工作线程', updated_at=? WHERE id=? AND state IN ('queued','export_queued')", (now, row["id"]))
            if cursor.rowcount != 1:
                return None
            connection.execute("UPDATE api_jobs SET status='running', updated_at=? WHERE id=?", (now, row["job_id"]))
            return self.row(connection.execute("SELECT * FROM api_items WHERE id=?", (row["id"],)).fetchone())

    def update_item(self, item_id: str, *, state: str | None = None, stage: str | None = None,
                    progress: str | None = None, error: str | None = None, state_json: dict | None = None,
                    revision: int | None = None) -> None:
        values: dict[str, Any] = {"updated_at": utc_now(), "id": item_id}
        fields = ["updated_at=:updated_at"]
        for key, value in (("state", state), ("stage", stage), ("progress", progress), ("error", error), ("revision", revision)):
            if value is not None:
                fields.append(f"{key}=:{key}")
                values[key] = value
        if state_json is not None:
            fields.append("state_json=:state_json")
            values["state_json"] = json_dump(state_json)
        with self.connection() as connection:
            connection.execute(f"UPDATE api_items SET {', '.join(fields)} WHERE id=:id", values)
            row = connection.execute("SELECT job_id FROM api_items WHERE id=?", (item_id,)).fetchone()
            if row:
                self.refresh_job_status(connection, row["job_id"])

    def refresh_job_status(self, connection: sqlite3.Connection, job_id: str) -> None:
        states = [row[0] for row in connection.execute("SELECT state FROM api_items WHERE job_id=?", (job_id,))]
        if not states:
            aggregate = "failed"
        elif any(item in {"queued", "running", "export_queued"} for item in states):
            aggregate = "running"
        elif any(item == "awaiting_review" for item in states):
            aggregate = "awaiting_review"
        elif all(item == "completed" for item in states):
            aggregate = "completed"
        elif any(item == "interrupted" for item in states):
            aggregate = "interrupted"
        elif all(item in TERMINAL_ITEM_STATES for item in states):
            aggregate = "finished_with_errors" if any(item == "failed" for item in states) else "cancelled"
        else:
            aggregate = "running"
        connection.execute("UPDATE api_jobs SET status=?, updated_at=? WHERE id=?", (aggregate, utc_now(), job_id))

    def queue_exports(self, job_id: str, sequence: int | None = None) -> int:
        clauses, args = ["job_id=?", "state='awaiting_review'"], [job_id]
        if sequence is not None:
            clauses.append("sequence=?")
            args.append(sequence)
        with self.connection() as connection:
            cursor = connection.execute(f"UPDATE api_items SET state='export_queued', stage='export', progress='等待导出', error=NULL, updated_at=? WHERE {' AND '.join(clauses)}", (utc_now(), *args))
            self.refresh_job_status(connection, job_id)
            return cursor.rowcount

    def retry(self, job_id: str, sequence: int) -> bool:
        with self.connection() as connection:
            item = connection.execute("SELECT * FROM api_items WHERE job_id=? AND sequence=?", (job_id, sequence)).fetchone()
            if item is None or item["state"] not in {"failed", "interrupted", "cancelled"}:
                return False
            state = "export_queued" if item["stage"] == "export" else "queued"
            connection.execute("UPDATE api_items SET state=?, progress='等待重试', error=NULL, updated_at=? WHERE id=?", (state, utc_now(), item["id"]))
            self.refresh_job_status(connection, job_id)
            return True

    def replace_boundaries(self, item_id: str, expected_revision: int, snapshot: dict) -> int | None:
        with self.connection() as connection:
            cursor = connection.execute("UPDATE api_items SET state='awaiting_review', stage='review', progress='已保存人工边界，等待导出', revision=revision+1, state_json=?, updated_at=? WHERE id=? AND revision=? AND state IN ('awaiting_review','completed')",
                                        (json_dump(snapshot), utc_now(), item_id, expected_revision))
            if cursor.rowcount != 1:
                return None
            row = connection.execute("SELECT revision,job_id FROM api_items WHERE id=?", (item_id,)).fetchone()
            self.refresh_job_status(connection, row["job_id"])
            return int(row["revision"])


def boundary_data(boundary: Boundary) -> dict:
    return {"frame_index": boundary.frame_index, "score": boundary.score, "origin": boundary.origin,
            "kind": boundary.kind, "pair_id": boundary.pair_id}


def pack_video_job(job: VideoJob) -> dict:
    return {
        "source": str(job.source), "output_root": str(job.output_root), "identifier": job.identifier,
        "session_name": job.session_name, "status": job.status, "progress": job.progress, "error": job.error,
        "duration_seconds": job.duration_seconds, "source_fps": job.source_fps, "frame_count": job.frame_count,
        "work_video": str(job.work_video) if job.work_video else None, "work_fps": job.work_fps,
        "output_dir": str(job.output_dir) if job.output_dir else None,
        "features_path": str(job.features_path) if job.features_path else None,
        "probabilities_path": str(job.probabilities_path) if job.probabilities_path else None,
        "inference_path": str(job.inference_path) if job.inference_path else None,
        "effective_settings": job.effective_settings,
        "boundaries": [boundary_data(item) for item in job.boundaries],
        "original_boundaries": [boundary_data(item) for item in job.original_boundaries],
        "discard_pair_metadata": job.discard_pair_metadata, "tail_crop_metadata": job.tail_crop_metadata,
        "warnings": job.warnings,
    }


def unpack_boundaries(values: list[dict]) -> list[Boundary]:
    return [Boundary(int(item["frame_index"]), item.get("score"), item.get("origin", "auto"),
                     item.get("kind", "split"), item.get("pair_id")) for item in values]


def unpack_video_job(snapshot: dict) -> VideoJob:
    job = VideoJob(Path(snapshot["source"]), Path(snapshot["output_root"]), snapshot["identifier"], snapshot.get("session_name"))
    for name in ("status", "progress", "error", "duration_seconds", "source_fps", "frame_count", "work_fps",
                 "effective_settings", "discard_pair_metadata", "tail_crop_metadata", "warnings"):
        setattr(job, name, snapshot.get(name))
    for name in ("work_video", "output_dir", "features_path", "probabilities_path", "inference_path"):
        value = snapshot.get(name)
        setattr(job, name, Path(value) if value else None)
    job.boundaries = unpack_boundaries(snapshot.get("boundaries", []))
    job.original_boundaries = unpack_boundaries(snapshot.get("original_boundaries", []))
    return job


class ApiRuntime:
    def __init__(self, config: ApiConfig, service_factory: Callable[[], InferenceService] = InferenceService) -> None:
        self.config = config.normalised()
        self.store = ApiStore(self.config.database_path)
        self.service_factory = service_factory
        self.service: InferenceService | None = None
        self.model_root = self.resolve_model_root()
        self.model_error: str | None = None
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.worker: threading.Thread | None = None

    def resolve_model_root(self) -> Path:
        project_root = Path(__file__).resolve().parents[1]
        active = project_root / "model"
        if not self.config.model_version:
            return active
        release = active / "releases" / self.config.model_version
        if not release.is_dir():
            raise ValueError(f"未注册模型版本：{self.config.model_version}")
        return release.resolve()

    def available_models(self) -> list[dict]:
        roots = [Path(__file__).resolve().parents[1] / "model"]
        release_root = roots[0] / "releases"
        if release_root.is_dir():
            roots.extend(sorted((path for path in release_root.iterdir() if path.is_dir()), key=lambda path: path.name))
        results = []
        for root in roots:
            policy_path = root / "frozen_inference.json"
            if not policy_path.is_file():
                continue
            try:
                policy = json.loads(policy_path.read_text(encoding="utf-8"))
                results.append({"id": root.name if root.parent.name == "releases" else policy.get("model_version", "active"),
                                "model_version": policy.get("model_version"), "architecture": policy.get("architecture"),
                                "checkpoint_sha256": policy.get("checkpoint_sha256"), "active": root.resolve() == self.model_root})
            except json.JSONDecodeError:
                continue
        return results

    def start(self) -> None:
        self.store.initialise()
        self.service = (InferenceService(model_root=self.model_root) if self.service_factory is InferenceService
                        else self.service_factory())
        try:
            self.service.verify_package()
        except Exception as error:  # Health reports this; queued work remains retryable after repair.
            self.model_error = str(error)
        self.worker = threading.Thread(target=self.run, name="data-clip-api-worker", daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.wake_event.set()
        if self.worker:
            self.worker.join(timeout=5)

    def wake(self) -> None:
        self.wake_event.set()

    def run(self) -> None:
        while not self.stop_event.is_set():
            work = self.store.next_work()
            if work is None:
                self.wake_event.wait(1)
                self.wake_event.clear()
                continue
            if self.model_error:
                self.store.update_item(work["id"], state="failed", progress="", error=f"模型包不可用：{self.model_error}")
                continue
            try:
                self.process(work)
            except Exception as error:
                self.store.update_item(work["id"], state="failed", progress="", error=str(error))

    def process(self, item: dict) -> None:
        if self.service is None:
            raise RuntimeError("服务尚未初始化")
        job = unpack_video_job(item["snapshot"])
        stored_settings = self.store.job(item["job_id"])["settings"]
        settings = CutSettings(**{key: stored_settings[key] for key in (
            "threshold", "merge_gap", "minimum_segment_seconds", "export_keyframes")})

        def update(updated: VideoJob) -> None:
            self.store.update_item(item["id"], state="running", stage=item["stage"], progress=updated.progress,
                                   error=updated.error, state_json=pack_video_job(updated))

        if item["stage"] == "infer":
            self.service.load_model()
            result = self.service.infer(job, settings, update)
            final = "awaiting_review" if result.status == "等待复核" else "failed"
        else:
            result = self.service.export(job, settings, update)
            final = "completed" if result.status == "已导出" else "failed"
        self.store.update_item(item["id"], state=final, stage="review" if final == "awaiting_review" else item["stage"],
                               progress=result.progress, error=result.error, state_json=pack_video_job(result))

    def create_job(self, payload: CreateJobPayload) -> dict:
        if payload.sequence_start > payload.sequence_end:
            raise ValueError("sequence_start 不能大于 sequence_end")
        folder = resolved_within(payload.date_task_path, self.config.dataset_root, "日期任务目录")
        date, task_name, sessions = discover_delta_sessions(folder)
        settings, output_root = self.settings(payload.settings), self.output_root(payload.settings)
        selected = [session for session in sessions if payload.sequence_start <= session.sequence_number <= payload.sequence_end]
        if not selected:
            raise ValueError("所选序号范围没有任务")
        now, job_id = utc_now(), uuid4().hex
        job = {"id": job_id, "created_at": now, "updated_at": now, "date_task_path": str(folder),
               "sequence_start": payload.sequence_start, "sequence_end": payload.sequence_end,
               "settings_json": json_dump(asdict(settings) | {"export_root": str(output_root)}), "status": "queued", "error": None}
        items = []
        for session in selected:
            item_id = uuid4().hex
            if session.cam0_video is None:
                snapshot = {"source": "", "output_root": str(output_root), "identifier": "",
                            "session_name": session.name, "boundaries": [], "original_boundaries": [],
                            "discard_pair_metadata": {}, "tail_crop_metadata": None, "warnings": []}
                state, error, source = "failed", session.issue or "cam0 视频不可用", None
            else:
                video = VideoJob(session.cam0_video, output_root, session_name=session.name)
                snapshot, state, error, source = pack_video_job(video), "queued", None, str(session.cam0_video)
            items.append({"id": item_id, "job_id": job_id, "sequence": session.sequence_number,
                          "session_name": session.name, "source_video": source, "stage": "infer", "state": state,
                          "progress": "等待推理" if state == "queued" else "", "error": error, "revision": 0,
                          "created_at": now, "updated_at": now, "state_json": json_dump(snapshot)})
        self.store.create_job(job, items)
        self.wake()
        return self.describe_job(job_id) or {"id": job_id}

    def output_root(self, payload: SettingsPayload) -> Path:
        if payload.export_root is None:
            return self.config.export_root
        return resolved_within(payload.export_root, self.config.export_root, "导出目录")

    def settings(self, payload: SettingsPayload) -> CutSettings:
        defaults = CutSettings()
        settings = CutSettings(defaults.threshold if payload.threshold is None else payload.threshold,
                               defaults.merge_gap if payload.merge_gap is None else payload.merge_gap,
                               defaults.minimum_segment_seconds if payload.minimum_segment_seconds is None else payload.minimum_segment_seconds,
                               defaults.export_keyframes if payload.export_keyframes is None else payload.export_keyframes)
        return settings.validate()

    def describe_item(self, item: dict, include_curve: bool = False, max_points: int = 2000) -> dict:
        snapshot = item["snapshot"]
        value = {"sequence": item["sequence"], "session_name": item["session_name"], "state": item["state"],
                 "stage": item["stage"], "progress": item["progress"], "error": item["error"],
                 "revision": item["revision"], "source_video": item["source_video"],
                 "frame_count": snapshot.get("frame_count"), "fps": snapshot.get("work_fps"),
                 "warnings": snapshot.get("warnings", []), "boundaries": [boundary_data(x) for x in unpack_boundaries(snapshot.get("boundaries", []))],
                 "automatic_boundaries": [boundary_data(x) for x in unpack_boundaries(snapshot.get("original_boundaries", []))],
                 "tail_crop": snapshot.get("tail_crop_metadata"), "result_directory": snapshot.get("output_dir")}
        if include_curve:
            path = snapshot.get("probabilities_path")
            if not path or not Path(path).is_file():
                raise FileNotFoundError("该任务尚未生成概率曲线")
            import numpy as np
            values = np.load(path, allow_pickle=False)["probabilities"].astype(float)
            count = min(max_points, len(values))
            indexes = np.linspace(0, len(values) - 1, count, dtype=int) if count else np.array([], dtype=int)
            value["probabilities"] = [{"frame_index": int(index), "score": float(values[index])} for index in indexes]
        return value

    def describe_job(self, job_id: str) -> dict | None:
        job = self.store.job(job_id)
        if job is None:
            return None
        return {"id": job["id"], "created_at": job["created_at"], "updated_at": job["updated_at"],
                "status": job["status"], "error": job["error"], "date_task_path": job["date_task_path"],
                "sequence_start": job["sequence_start"], "sequence_end": job["sequence_end"], "settings": job["settings"],
                "items": [self.describe_item(item) for item in self.store.items(job_id)]}

    def replace_boundaries(self, job_id: str, sequence: int, payload: BoundariesPayload) -> dict:
        item = self.store.item(job_id, sequence)
        if item is None:
            raise KeyError("任务序号不存在")
        snapshot, frame_count = item["snapshot"], item["snapshot"].get("frame_count")
        if not frame_count:
            raise ValueError("该视频尚未完成推理")
        existing_auto = {(value["frame_index"], value["kind"], value.get("pair_id"))
                         for value in snapshot.get("boundaries", []) if value.get("origin") == "auto"}
        values: list[Boundary] = []
        for value in payload.boundaries:
            if value.kind not in {"split", "discard_start", "discard_end"}:
                raise ValueError("未知边界类型")
            if value.origin not in {"auto", "manual"}:
                raise ValueError("origin 只能是 auto 或 manual")
            identity = (value.frame_index, value.kind, value.pair_id)
            if value.origin == "auto" and identity not in existing_auto:
                raise ValueError("新增或移动的边界必须标记为 manual")
            values.append(Boundary(value.frame_index, value.score, value.origin, value.kind, value.pair_id))
        resolved = deduplicate_boundaries(values, int(frame_count))
        if len(resolved) != len(values):
            raise ValueError("边界重复或超出有效帧范围")
        pairs: dict[str, list[Boundary]] = {}
        for value in resolved:
            if value.kind != "split":
                if not value.pair_id:
                    raise ValueError("删除端点必须提供 pair_id")
                pairs.setdefault(value.pair_id, []).append(value)
        for pair_id, endpoints in pairs.items():
            if len(endpoints) != 2 or {value.kind for value in endpoints} != {"discard_start", "discard_end"}:
                raise ValueError(f"删除区间 {pair_id} 必须包含一对起止端点")
            if endpoints[0].frame_index == endpoints[1].frame_index:
                raise ValueError(f"删除区间 {pair_id} 不能为零长度")
        snapshot["boundaries"] = [boundary_data(value) for value in resolved]
        snapshot["warnings"] = list(snapshot.get("warnings", []))
        revision = self.store.replace_boundaries(item["id"], payload.revision, snapshot)
        if revision is None:
            raise RuntimeError("边界已被其他请求修改，请重新读取后再提交")
        return {"revision": revision, "boundaries": [boundary_data(value) for value in resolved]}

    def artifact_path(self, item: dict, artifact: str) -> tuple[Path, str]:
        snapshot = item["snapshot"]
        output_value = snapshot.get("output_dir")
        if not output_value:
            raise FileNotFoundError("该任务尚无结果目录")
        output = resolved_within(output_value, self.config.export_root, "结果目录")
        if not output.is_dir():
            raise FileNotFoundError("该任务尚无结果目录")
        files = {"result": output / "result.json", "inference": output / "inference.json",
                 "probabilities": output / "probabilities.npz", "features": output / "features.npz",
                 "job": output / "job.json", "keyframes": output / "cut_frames/manifest.json"}
        if artifact == "archive":
            target = output / "api_result.zip"
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in output.rglob("*"):
                    if path.is_file() and path != target and "working_30fps.mp4" not in path.name:
                        archive.write(path, path.relative_to(output))
            return target, f"{item['session_name']}_result.zip"
        if artifact not in files:
            raise KeyError("未知下载工件")
        return files[artifact], files[artifact].name

    def csv(self, job_id: str) -> str:
        stream = StringIO()
        writer = csv.DictWriter(stream, fieldnames=("task_sequence", "task_name", "boundary_set", "frame_index",
                                                     "timestamp_seconds", "score", "origin", "kind", "pair_id"))
        writer.writeheader()
        for item in self.store.items(job_id):
            snapshot, fps = item["snapshot"], snapshot_fps(item["snapshot"])
            for group, values in (("automatic", snapshot.get("original_boundaries", [])),
                                  ("final", snapshot.get("boundaries", []))):
                for boundary in values:
                    frame = int(boundary["frame_index"])
                    writer.writerow({"task_sequence": item["sequence"], "task_name": item["session_name"],
                                     "boundary_set": group, "frame_index": frame,
                                     "timestamp_seconds": frame / fps if fps else "", "score": boundary.get("score"),
                                     "origin": boundary.get("origin"), "kind": boundary.get("kind"),
                                     "pair_id": boundary.get("pair_id")})
        return stream.getvalue()


def snapshot_fps(snapshot: dict) -> float:
    return float(snapshot.get("work_fps") or 0)


def create_app(config: ApiConfig | None = None, service_factory: Callable[[], InferenceService] = InferenceService) -> FastAPI:
    runtime = ApiRuntime(config or ApiConfig.from_environment(), service_factory)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        runtime.start()
        try:
            yield
        finally:
            runtime.stop()

    app = FastAPI(title="data_clip API", version="1.0.0", description="冻结 v1.3 视频裁剪模型的串行任务 API",
                  lifespan=lifespan)
    if runtime.config.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(runtime.config.cors_origins), allow_credentials=False,
                           allow_methods=["GET", "POST", "PUT"], allow_headers=["X-API-Key", "Content-Type"])

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if x_api_key is None or not secrets.compare_digest(x_api_key, runtime.config.api_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的 API Key")

    @app.get("/healthz")
    def health() -> dict:
        return {"status": "ok" if runtime.model_error is None else "degraded", "model_ready": runtime.service is not None and runtime.model_error is None,
                "model_error": runtime.model_error, "worker_alive": bool(runtime.worker and runtime.worker.is_alive()),
                "database": str(runtime.config.database_path)}

    @app.get("/v1/model", dependencies=[Depends(require_key)])
    def model() -> dict:
        policy_path = runtime.model_root / "frozen_inference.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        return {"model_version": policy.get("model_version"), "architecture": policy.get("architecture"),
                "checkpoint_sha256": policy.get("checkpoint_sha256"), "frozen_policy": policy,
                "runtime_defaults": asdict(CutSettings())}

    @app.get("/v1/models", dependencies=[Depends(require_key)])
    def models() -> dict:
        return {"active_model_root": str(runtime.model_root), "models": runtime.available_models(),
                "switching": "设置 DATA_CLIP_MODEL_VERSION 后重启 API；新版本必须先通过 register_model_release.py 校验。"}

    @app.post("/v1/jobs", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_key)])
    def create_job(payload: CreateJobPayload) -> dict:
        try:
            return runtime.create_job(payload)
        except (ValueError, CutterError) as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
    def get_job(job_id: str) -> dict:
        value = runtime.describe_job(job_id)
        if value is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return value

    @app.get("/v1/jobs/{job_id}/items/{sequence}", dependencies=[Depends(require_key)])
    def get_item(job_id: str, sequence: int, curve: bool = False, max_points: int = Query(default=2000, ge=1, le=20000)) -> dict:
        item = runtime.store.item(job_id, sequence)
        if item is None:
            raise HTTPException(status_code=404, detail="任务序号不存在")
        try:
            return runtime.describe_item(item, curve, max_points)
        except FileNotFoundError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.put("/v1/jobs/{job_id}/items/{sequence}/boundaries", dependencies=[Depends(require_key)])
    def put_boundaries(job_id: str, sequence: int, payload: BoundariesPayload) -> dict:
        try:
            return runtime.replace_boundaries(job_id, sequence, payload)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v1/jobs/{job_id}/items/{sequence}/export", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_key)])
    def export_item(job_id: str, sequence: int) -> dict:
        if runtime.store.item(job_id, sequence) is None:
            raise HTTPException(status_code=404, detail="任务序号不存在")
        if runtime.store.queue_exports(job_id, sequence) != 1:
            raise HTTPException(status_code=409, detail="任务尚不可导出")
        runtime.wake()
        return {"job_id": job_id, "sequence": sequence, "state": "export_queued"}

    @app.post("/v1/jobs/{job_id}/export", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_key)])
    def export_job(job_id: str) -> dict:
        if runtime.store.job(job_id) is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        queued = runtime.store.queue_exports(job_id)
        if not queued:
            raise HTTPException(status_code=409, detail="没有可导出的任务")
        runtime.wake()
        return {"job_id": job_id, "queued_items": queued}

    @app.post("/v1/jobs/{job_id}/items/{sequence}/retry", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_key)])
    def retry_item(job_id: str, sequence: int) -> dict:
        if not runtime.store.retry(job_id, sequence):
            raise HTTPException(status_code=409, detail="只有失败、取消或中断的任务可重试")
        runtime.wake()
        return {"job_id": job_id, "sequence": sequence, "state": "queued"}

    @app.get("/v1/jobs/{job_id}/items/{sequence}/downloads/{artifact}", dependencies=[Depends(require_key)])
    def download(job_id: str, sequence: int, artifact: str):
        item = runtime.store.item(job_id, sequence)
        if item is None:
            raise HTTPException(status_code=404, detail="任务序号不存在")
        try:
            path, filename = runtime.artifact_path(item, artifact)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if not path.is_file():
            raise HTTPException(status_code=404, detail="请求的结果文件尚不存在")
        return FileResponse(path, filename=filename)

    @app.get("/v1/jobs/{job_id}/cut-points.csv", dependencies=[Depends(require_key)])
    def cut_points(job_id: str) -> Response:
        if runtime.store.job(job_id) is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return Response(runtime.csv(job_id), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{job_id}_cut_points.csv"'})

    app.state.runtime = runtime
    return app
