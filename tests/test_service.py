from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from video_cutter.service import (Boundary, CutSettings, VideoJob, atomic_json, deduplicate_boundaries,
                                  discover_delta_sessions, edge_peak_boundaries, enforce_minimum_segments, safe_name,
                                  enforce_cut_gap, export_intervals, scan_videos, DEFAULT_CUT_THRESHOLD,
                                  MINIMUM_CUT_GAP_SECONDS, EDGE_WINDOW_SECONDS, discard_ranges, intersects_tail_edge_window,
                                  high_score_regions, nearest_valley, valley_pair, reduce_short_video_middle_nodes,
                                  SHORT_VIDEO_MIDDLE_NODE_LIMIT_SECONDS, resolve_short_export_segments)


class SettingsTests(unittest.TestCase):
    def test_default_threshold_is_point_eight(self) -> None:
        self.assertEqual(CutSettings().threshold, DEFAULT_CUT_THRESHOLD)
        self.assertEqual(DEFAULT_CUT_THRESHOLD, .80)

    def test_default_edge_window_is_two_seconds(self) -> None:
        self.assertEqual(EDGE_WINDOW_SECONDS, 2.0)

    def test_validation_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            CutSettings(threshold=1.01).validate()
        with self.assertRaises(ValueError):
            CutSettings(merge_gap=-1).validate()
        with self.assertRaises(ValueError):
            CutSettings(minimum_segment_seconds=-0.1).validate()


class BoundaryTests(unittest.TestCase):
    def test_four_second_gap_keeps_earlier_cut_including_exact_gap(self) -> None:
        points = [Boundary(100, .8), Boundary(220, .9), Boundary(221, .7)]
        kept, warnings = enforce_cut_gap(points, fps=30)
        self.assertEqual(MINIMUM_CUT_GAP_SECONDS, 4)
        self.assertEqual([item.frame_index for item in kept], [100, 221])
        self.assertEqual(len(warnings), 1)
        self.assertIn("第 220 帧", warnings[0])

    def test_export_trims_head_and_tail(self) -> None:
        boundaries = [Boundary(100, .8), Boundary(400, .9), Boundary(700, .85)]
        self.assertEqual(export_intervals(boundaries, 900), [(100, 400), (400, 700)])
        self.assertEqual(export_intervals(boundaries, 900, trim_head_tail=False),
                         [(0, 100), (100, 400), (400, 700), (700, 900)])

    def test_trimmed_export_requires_two_boundaries(self) -> None:
        self.assertEqual(export_intervals([Boundary(100, .8)], 900), [])

    def test_valley_pair_uses_nearest_low_plateau_frames(self) -> None:
        scores = [.1, .1, .2, .6, .9, .6, .2, .1, .1]
        self.assertEqual(nearest_valley(scores, 4, -1, 4), 1)
        self.assertEqual(nearest_valley(scores, 4, 1, 4), 7)
        self.assertEqual(valley_pair(scores, 4, fps=30, smoothing_frames=1, search_seconds=.2), (1, 7))

    def test_valley_pair_uses_search_edge_when_no_local_valley_exists(self) -> None:
        scores = [0, .1, .2, .3, .9, .8, .7, .6, .5]
        self.assertEqual(valley_pair(scores, 4, fps=30, smoothing_frames=1, search_seconds=.1), (1, 7))

    def test_valley_requires_meaningful_drop_from_peak(self) -> None:
        scores = [.05, .05, .1, .8, .99, .98, .99, .5, .1, .1]
        # Frame 5 is a tiny local dip on the .99 plateau. It must not win;
        # frame 8 is the first local low that dropped at least .20.
        self.assertEqual(nearest_valley(scores, 4, 1, 5, minimum_drop=.20), 8)

    def test_tail_high_score_region_must_intersect_final_edge_window(self) -> None:
        scores = [0.0] * 100
        scores[70], scores[71], scores[76] = .9, .95, .85
        regions = high_score_regions(scores, threshold=.8, merge_gap=6)
        self.assertEqual(regions, [{"start_frame": 70, "end_frame": 76,
                                    "peak_frame": 71, "peak_score": .95}])
        # At 30 FPS, the final two seconds begin at frame 40.
        self.assertTrue(intersects_tail_edge_window(regions[0], 100, 60))
        self.assertFalse(intersects_tail_edge_window(
            {"start_frame": 30, "end_frame": 39, "peak_frame": 35, "peak_score": .95}, 100, 60))

    def test_paired_discard_endpoints_are_exempt_from_four_second_gap(self) -> None:
        boundaries = [Boundary(100, .2, "auto", "discard_start", "one"),
                      Boundary(130, .1, "auto", "discard_end", "one")]
        ranges, warnings = discard_ranges(boundaries, fps=30)
        self.assertEqual([(item["start"].frame_index, item["end"].frame_index) for item in ranges], [(100, 130)])
        self.assertEqual(warnings, [])

    def test_later_automatic_discard_range_is_removed_when_retained_gap_is_short(self) -> None:
        boundaries = [Boundary(100, .2, "auto", "discard_start", "first"),
                      Boundary(160, .1, "auto", "discard_end", "first"),
                      Boundary(200, .2, "auto", "discard_start", "second"),
                      Boundary(260, .1, "auto", "discard_end", "second")]
        ranges, warnings = discard_ranges(boundaries, fps=30)
        self.assertEqual([item["pair_id"] for item in ranges], ["first"])
        self.assertIn("second", warnings[0])

    def test_manual_discard_range_overrides_conflicting_automatic_range(self) -> None:
        boundaries = [Boundary(100, .2, "auto", "discard_start", "first"),
                      Boundary(160, .1, "auto", "discard_end", "first"),
                      Boundary(200, .2, "manual", "discard_start", "second"),
                      Boundary(260, .1, "auto", "discard_end", "second")]
        ranges, warnings = discard_ranges(boundaries, fps=30)
        self.assertEqual([item["pair_id"] for item in ranges], ["second"])
        self.assertIn("人工", warnings[0])

    def test_short_video_keeps_only_central_automatic_discard_pair_and_edge_splits(self) -> None:
        boundaries = [Boundary(300, .1, "auto", "discard_start", "left"),
                      Boundary(360, .1, "auto", "discard_end", "left"),
                      Boundary(570, .1, "auto", "discard_start", "central"),
                      Boundary(630, .1, "auto", "discard_end", "central")]
        ranges, _ = discard_ranges(boundaries, fps=30)
        splits, ranges, warnings = reduce_short_video_middle_nodes(
            [Boundary(30, .9), Boundary(400, .8), Boundary(1170, .9)], ranges, frame_count=1200, fps=30)
        self.assertEqual(SHORT_VIDEO_MIDDLE_NODE_LIMIT_SECONDS, 55)
        self.assertEqual([item.frame_index for item in splits], [30, 1170])
        self.assertEqual([item["pair_id"] for item in ranges], ["central"])
        self.assertEqual(len(warnings), 2)

    def test_fifty_five_second_video_is_not_reduced(self) -> None:
        boundaries = [Boundary(400, .1, "auto", "discard_start", "one"),
                      Boundary(460, .1, "auto", "discard_end", "one"),
                      Boundary(900, .1, "auto", "discard_start", "two"),
                      Boundary(960, .1, "auto", "discard_end", "two")]
        ranges, _ = discard_ranges(boundaries, fps=30)
        splits, ranges, warnings = reduce_short_video_middle_nodes(
            [Boundary(30, .9), Boundary(600, .8), Boundary(1620, .9)], ranges, frame_count=1650, fps=30)
        self.assertEqual([item.frame_index for item in splits], [30, 600, 1620])
        self.assertEqual([item["pair_id"] for item in ranges], ["one", "two"])
        self.assertEqual(warnings, [])

    def test_export_discards_paired_range_but_keeps_other_content(self) -> None:
        boundaries = [Boundary(30, .8), Boundary(210, .8, "manual"), Boundary(270, .8),
                      Boundary(90, .1, "auto", "discard_start", "one"),
                      Boundary(150, .1, "auto", "discard_end", "one")]
        self.assertEqual(export_intervals(boundaries, 300, discard_intervals=[(90, 150)]),
                         [(30, 90), (150, 210), (210, 270)])

    def test_short_export_segments_drop_second_and_penultimate_then_merge_others(self) -> None:
        # 1/20 of 200 frames is 10 frames. The second and penultimate short
        # outputs are discarded. The first short output is merged instead.
        groups, warnings = resolve_short_export_segments([(0, 5), (5, 12), (12, 50), (50, 57), (57, 200)], 200)
        self.assertEqual(groups, [[(0, 5), (12, 50)], [(57, 200)]])
        self.assertEqual(len(warnings), 3)

    def test_short_export_segment_merges_with_shorter_neighbor(self) -> None:
        groups, warnings = resolve_short_export_segments([(0, 40), (40, 60), (60, 65), (65, 75), (75, 130)], 130)
        self.assertEqual(groups, [[(0, 40)], [(40, 60)], [(60, 65), (65, 75)], [(75, 130)]])
        self.assertEqual(len(warnings), 1)

    def test_edge_fallback_adds_only_missing_edge_peaks(self) -> None:
        scores = [0.0] * 360  # 12 seconds at 30 FPS
        scores[20] = .65  # First five seconds: below frozen threshold.
        scores[250] = .95  # Final five seconds: threshold hit, so no fallback.
        result = edge_peak_boundaries(scores, fps=30, threshold=.70, edge_seconds=5)
        self.assertEqual([(item.frame_index, item.origin) for item in result], [(20, "edge_peak_start")])

    def test_edge_fallback_adds_both_peaks_and_never_clip_endpoint(self) -> None:
        scores = [0.0] * 360
        scores[0], scores[1], scores[40] = .69, .40, .60
        scores[-1], scores[-2], scores[-30] = .69, .50, .65
        result = edge_peak_boundaries(scores, fps=30, threshold=.70, edge_seconds=5)
        self.assertEqual([(item.frame_index, item.origin) for item in result],
                         [(40, "edge_peak_start"), (330, "edge_peak_end")])

    def test_deduplication_prefers_manual_and_drops_endpoints(self) -> None:
        result = deduplicate_boundaries([Boundary(0, .2), Boundary(10, .1), Boundary(10, .9, "manual"),
                                         Boundary(100, .2)], 100)
        self.assertEqual([(item.frame_index, item.origin) for item in result], [(10, "manual")])

    def test_minimum_duration_removes_low_score_automatic_boundary(self) -> None:
        result, warnings = enforce_minimum_segments([Boundary(10, .1), Boundary(12, .9)], 100, 10)
        self.assertEqual([item.frame_index for item in result], [12])
        self.assertEqual(len(warnings), 1)

    def test_manual_conflict_is_kept_and_reported(self) -> None:
        result, warnings = enforce_minimum_segments([Boundary(10, .5, "manual"), Boundary(12, .6, "manual")], 100, 10)
        self.assertEqual([item.frame_index for item in result], [10, 12])
        self.assertIn("人工边界", warnings[0])


class FilesystemTests(unittest.TestCase):
    def test_delta_session_discovery_uses_cam0_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "2026-09-02-Oven"
            (root / "Oven_100/cam0").mkdir(parents=True)
            (root / "Oven_100/cam1").mkdir()
            (root / "Oven_100/cam0/cam0-00000.mp4").touch()
            (root / "Oven_100/cam1/cam1-00000.mp4").touch()
            (root / "Oven_102/cam0").mkdir(parents=True)
            (root / "Oven_102/cam0/alternate.mov").touch()
            (root / "Oven_103").mkdir()
            date, task, sessions = discover_delta_sessions(root)
            self.assertEqual((date, task), ("2026-09-02", "Oven"))
            self.assertEqual([item.sequence_number for item in sessions], [100, 102, 103])
            self.assertEqual(sessions[0].cam0_video.name, "cam0-00000.mp4")
            self.assertEqual(sessions[1].cam0_video.name, "alternate.mov")
            self.assertIsNone(sessions[2].cam0_video)
            self.assertEqual(sessions[2].issue, "缺少 cam0 目录")

    def test_scan_is_recursive_and_extension_aware(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            (root / "one.MP4").touch()
            (root / "nested/two.mov").touch()
            (root / "ignored.txt").touch()
            self.assertEqual([item.name for item in scan_videos(root)], ["one.MP4", "two.mov"])

    def test_scan_excludes_result_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "data_clip_results"
            results.mkdir()
            (root / "source.mp4").touch()
            (results / "segment_001.mp4").touch()
            self.assertEqual([item.name for item in scan_videos(root, [results])], ["source.mp4"])

    def test_atomic_report_and_job_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            source.write_bytes(b"test video")
            job = VideoJob(source, root / "results")
            atomic_json(root / "report.json", {"ok": True})
            self.assertEqual(json.loads((root / "report.json").read_text()), {"ok": True})
            self.assertTrue(job.identifier.startswith("source_"))
            self.assertEqual(safe_name("a / b"), "a_b")


if __name__ == "__main__":
    unittest.main()
