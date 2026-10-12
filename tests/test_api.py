from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np

from video_cutter.api import (ApiConfig, ApiRuntime, BoundariesPayload, CreateJobPayload,
                              SettingsPayload, resolved_within)
from video_cutter.service import Boundary, VideoJob


class FakeInferenceService:
    loads = 0

    def verify_package(self) -> None:
        return None

    def load_model(self):
        type(self).loads += 1
        return ()

    def infer(self, job: VideoJob, settings, update):
        job.output_dir = job.output_root / job.identifier
        job.output_dir.mkdir(parents=True, exist_ok=True)
        job.frame_count, job.work_fps = 100, 30.0
        job.work_video = job.source
        job.probabilities_path = job.output_dir / "probabilities.npz"
        np.savez_compressed(job.probabilities_path, probabilities=np.linspace(0, 1, 100, dtype=np.float32))
        job.features_path = job.output_dir / "features.npz"
        np.savez_compressed(job.features_path, x=np.zeros((100, 1), dtype=np.float32))
        job.inference_path = job.output_dir / "inference.json"
        job.inference_path.write_text(json.dumps({"boundaries": []}), encoding="utf-8")
        job.boundaries = [Boundary(10, .1), Boundary(90, .9)]
        job.original_boundaries = list(job.boundaries)
        job.status, job.progress = "等待复核", "可在时间轴调整边界后导出"
        update(job)
        return job

    def export(self, job: VideoJob, settings, update):
        (job.output_dir / "segments").mkdir(exist_ok=True)
        (job.output_dir / "segments/segment_001.mp4").write_bytes(b"fake")
        (job.output_dir / "result.json").write_text(json.dumps({"segments": ["segment_001.mp4"]}), encoding="utf-8")
        job.status, job.progress = "已导出", "已导出 1 个片段"
        update(job)
        return job


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.dataset = root / "datasets"
        self.folder = self.dataset / "2026-09-16-Kitchen Table"
        video = self.folder / "Kitchen_Table_205/cam0/cam0-00000.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video")
        self.config = ApiConfig("test-key", self.dataset, root / "exports", root / "api.sqlite")
        FakeInferenceService.loads = 0
        self.runtime = ApiRuntime(self.config, FakeInferenceService)
        self.runtime.start()

    def tearDown(self) -> None:
        self.runtime.stop()
        self.temporary.cleanup()

    def wait_for(self, job_id: str, expected: str) -> dict:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            payload = self.runtime.describe_job(job_id)
            if payload["items"][0]["state"] == expected:
                return payload
            time.sleep(.03)
        self.fail(f"job {job_id} did not become {expected}")

    def test_serial_two_phase_workflow_and_artifacts(self) -> None:
        created = self.runtime.create_job(CreateJobPayload(
            date_task_path=str(self.folder), sequence_start=205, sequence_end=205,
            settings=SettingsPayload(threshold=.8),
        ))
        job_id = created["id"]
        inferred = self.wait_for(job_id, "awaiting_review")
        self.assertEqual(FakeInferenceService.loads, 1)
        item = self.runtime.store.item(job_id, 205)
        detail = self.runtime.describe_item(item, include_curve=True, max_points=11)
        self.assertEqual(len(detail["probabilities"]), 11)

        boundaries = detail["boundaries"] + [{"frame_index": 50, "origin": "manual", "kind": "split"}]
        saved = self.runtime.replace_boundaries(job_id, 205, BoundariesPayload(
            revision=inferred["items"][0]["revision"], boundaries=boundaries))
        self.assertEqual(saved["revision"], 1)
        with self.assertRaises(RuntimeError):
            self.runtime.replace_boundaries(job_id, 205, BoundariesPayload(revision=0, boundaries=boundaries))

        self.assertEqual(self.runtime.store.queue_exports(job_id, 205), 1)
        self.runtime.wake()
        self.wait_for(job_id, "completed")
        path, _ = self.runtime.artifact_path(self.runtime.store.item(job_id, 205), "result")
        self.assertTrue(path.is_file())
        self.assertIn("task_sequence", self.runtime.csv(job_id))

    def test_rejects_paths_outside_configured_dataset_root(self) -> None:
        with self.assertRaises(ValueError):
            resolved_within("/tmp/not-delta", self.dataset, "日期任务目录")


if __name__ == "__main__":
    unittest.main()
