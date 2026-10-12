"""Media, inference, and export services for the desktop application.

This module deliberately does not alter the frozen inference policy.  It
loads the packaged model once, and applies user settings only while decoding
that model's frame probabilities for an individual job.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256 as hashlib_sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Callable, Iterable, Sequence


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
FPS_EPSILON = 0.01
DEFAULT_DATASET_ROOT = Path("/mnt/Datasets/Delta-D1")
DEFAULT_EXPORT_ROOT = Path("/home/descfly/segment_data")
DEFAULT_CUT_THRESHOLD = 0.80
MINIMUM_CUT_GAP_SECONDS = 4.0
SHORT_VIDEO_MIDDLE_NODE_LIMIT_SECONDS = 55.0
EDGE_WINDOW_SECONDS = 2.0
VALLEY_MINIMUM_DROP = 0.20
VALLEY_SMOOTHING_FRAMES = 15
VALLEY_SEARCH_SECONDS = 5.0
DATE_TASK_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)$")
SESSION_PATTERN = re.compile(r"^.+_(\d+)$")


class CutterError(RuntimeError):
    """An actionable failure while preparing, inferring, or exporting media."""


@dataclass(frozen=True)
class CutSettings:
    """Per-run options.  They never modify model/frozen_inference.json."""

    threshold: float = DEFAULT_CUT_THRESHOLD
    merge_gap: int = 6
    minimum_segment_seconds: float = 0.0
    export_keyframes: bool = True

    def validate(self) -> "CutSettings":
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("阈值必须在 0 到 1 之间")
        if self.merge_gap < 0:
            raise ValueError("合并间隔不能小于 0 帧")
        if self.minimum_segment_seconds < 0:
            raise ValueError("最短片段时长不能小于 0")
        return self


@dataclass
class Boundary:
    frame_index: int
    score: float | None
    origin: str = "auto"  # auto or manual
    kind: str = "split"  # split, discard_start, or discard_end
    pair_id: str | None = None

    def as_dict(self, fps: float) -> dict:
        return {"frame_index": self.frame_index, "timestamp_seconds": self.frame_index / fps,
                "score": self.score, "origin": self.origin, "kind": self.kind, "pair_id": self.pair_id}


@dataclass(frozen=True)
class DeltaSession:
    """One numbered Delta-D1 task session and its sole eligible camera video."""

    date: str
    task_name: str
    directory: Path
    sequence_number: int
    cam0_video: Path | None
    issue: str | None = None

    @property
    def name(self) -> str:
        return self.directory.name


@dataclass
class VideoJob:
    source: Path
    output_root: Path
    identifier: str = ""
    session_name: str | None = None
    status: str = "等待处理"
    progress: str = ""
    error: str | None = None
    duration_seconds: float | None = None
    source_fps: float | None = None
    frame_count: int | None = None
    work_video: Path | None = None
    work_fps: float | None = None
    output_dir: Path | None = None
    features_path: Path | None = None
    probabilities_path: Path | None = None
    inference_path: Path | None = None
    effective_settings: dict | None = None
    boundaries: list[Boundary] = field(default_factory=list)
    original_boundaries: list[Boundary] = field(default_factory=list)
    discard_pair_metadata: dict[str, dict] = field(default_factory=dict)
    tail_crop_metadata: dict | None = None
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.source = Path(self.source).resolve()
        self.output_root = Path(self.output_root).resolve()
        if not self.identifier:
            # Constructed on the Qt UI thread when users load a range. Full
            # SHA-256 checks stay in the background inference pipeline.
            fingerprint = quick_file_fingerprint(self.source) if self.source.is_file() else "missing"
            self.identifier = f"{safe_name(self.session_name or self.source.stem)}_{fingerprint}"

    @property
    def job_path(self) -> Path:
        if self.output_dir is None:
            self.output_dir = self.output_root / self.identifier
        return self.output_dir / "job.json"

    def to_status_dict(self) -> dict:
        return {"schema": "data-clip.desktop-job.v2", "source_video": str(self.source),
                "identifier": self.identifier, "status": self.status, "progress": self.progress,
                "session_name": self.session_name,
                "error": self.error, "duration_seconds": self.duration_seconds,
                "source_fps": self.source_fps, "frame_count": self.frame_count,
                "working_video": str(self.work_video) if self.work_video else None,
                "working_fps": self.work_fps, "features": str(self.features_path) if self.features_path else None,
                "probabilities": str(self.probabilities_path) if self.probabilities_path else None,
                "inference": str(self.inference_path) if self.inference_path else None,
                "effective_settings": self.effective_settings,
                "boundaries": [item.as_dict(self.work_fps or 30.0) for item in self.boundaries],
                "discard_ranges": serialize_discard_ranges(self.boundaries, self.work_fps or 30.0,
                                                            self.discard_pair_metadata),
                "tail_crop": self.tail_crop_metadata,
                "warnings": self.warnings}


def safe_name(value: str) -> str:
    """Produce a stable, readable directory component without path characters."""
    return re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._") or "video"


def file_digest(path: Path) -> str:
    h = hashlib_sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def quick_file_fingerprint(path: Path) -> str:
    """Return a stat-based output identity without reading the video bytes."""
    try:
        metadata = path.stat()
    except OSError:
        return "missing"
    return f"{metadata.st_size:x}_{metadata.st_mtime_ns:x}"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_program(name: str) -> str:
    program = shutil.which(name)
    if not program:
        raise CutterError(f"未找到 {name}。请安装 FFmpeg，并确保 {name} 位于 PATH 中。")
    return program


def run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or error.stdout.strip() or "未知媒体处理错误"
        raise CutterError(message) from error


def parse_fps(value: str) -> float:
    try:
        numerator, denominator = value.split("/", 1)
        result = float(numerator) / float(denominator)
    except (AttributeError, ValueError, ZeroDivisionError) as error:
        raise CutterError(f"无法解析视频帧率：{value!r}") from error
    if result <= 0:
        raise CutterError(f"视频帧率必须为正数：{value!r}")
    return result


def probe_video(path: Path) -> dict:
    """Return the first video stream's properties using ffprobe."""
    ffprobe = require_program("ffprobe")
    result = run_command([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    try:
        payload = json.loads(result.stdout)
        stream = next(item for item in payload["streams"] if item.get("codec_type") == "video")
        fps = parse_fps(stream.get("avg_frame_rate") or stream["r_frame_rate"])
        duration = float(stream.get("duration") or payload["format"]["duration"])
    except (KeyError, StopIteration, TypeError, ValueError) as error:
        raise CutterError(f"无法读取视频流信息：{path}") from error
    if duration <= 0:
        raise CutterError(f"视频时长必须大于 0：{path}")
    return {"fps": fps, "duration_seconds": duration, "width": int(stream["width"]),
            "height": int(stream["height"]), "has_audio": any(item.get("codec_type") == "audio" for item in payload["streams"])}


def scan_videos(folder: Path, excluded_roots: Iterable[Path] = ()) -> list[Path]:
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError("请选择一个存在的视频文件夹")
    excluded = [Path(item).expanduser().resolve() for item in excluded_roots]

    def included(path: Path) -> bool:
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            return False
        return not any(root == path or root in path.parents for root in excluded)

    return sorted((path for path in folder.rglob("*") if included(path)),
                  key=lambda path: (path.name.lower(), str(path).lower()))


def discover_delta_sessions(date_task_folder: Path) -> tuple[str, str, list[DeltaSession]]:
    """Index Delta-D1's ``date-task/session_number/cam0`` layout only.

    ``cam0-00000.mp4`` is preferred. A different supported video directly in
    ``cam0`` is accepted as a compatibility fallback; all other cameras and
    nested result directories are intentionally ignored.
    """
    folder = Path(date_task_folder).expanduser().resolve()
    match = DATE_TASK_PATTERN.fullmatch(folder.name)
    if not folder.is_dir() or not match:
        raise ValueError("请选择 Delta-D1 中形如“YYYY-MM-DD-任务名”的日期-任务文件夹")
    date, task_name = match.groups()
    sessions: list[DeltaSession] = []
    for child in folder.iterdir():
        if not child.is_dir():
            continue
        number_match = SESSION_PATTERN.fullmatch(child.name)
        if not number_match:
            continue
        sequence_number = int(number_match.group(1))
        cam0 = child / "cam0"
        preferred = cam0 / "cam0-00000.mp4"
        if preferred.is_file():
            video, issue = preferred, None
        elif not cam0.is_dir():
            video, issue = None, "缺少 cam0 目录"
        else:
            choices = sorted((item for item in cam0.iterdir()
                              if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS),
                             key=lambda item: item.name.lower())
            video, issue = (choices[0], None) if choices else (None, "cam0 中没有支持的视频文件")
        sessions.append(DeltaSession(date, task_name, child, sequence_number, video, issue))
    sessions.sort(key=lambda item: (item.sequence_number, item.name))
    if not sessions:
        raise ValueError("该日期-任务文件夹中没有编号任务目录（例如 Oven_100）")
    return date, task_name, sessions


def normalize_to_30fps(source: Path, target: Path) -> Path:
    """Create an H.264/AAC constant-frame-rate working copy for the model."""
    ffmpeg = require_program("ffmpeg")
    target.parent.mkdir(parents=True, exist_ok=True)
    run_command([ffmpeg, "-y", "-i", str(source), "-map", "0:v:0", "-map", "0:a?", "-vf", "fps=30",
                 "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-movflags", "+faststart", str(target)])
    return target


def deduplicate_boundaries(boundaries: Iterable[Boundary], frame_count: int) -> list[Boundary]:
    """Keep one internal boundary per frame, preferring manual edits."""
    selected: dict[int, Boundary] = {}
    for item in boundaries:
        if not 0 < int(item.frame_index) < frame_count:
            continue
        if item.kind not in {"split", "discard_start", "discard_end"}:
            raise ValueError(f"未知边界类型：{item.kind}")
        candidate = Boundary(int(item.frame_index), item.score, item.origin, item.kind, item.pair_id)
        existing = selected.get(candidate.frame_index)
        if existing is None or (candidate.origin == "manual" and existing.origin != "manual"):
            selected[candidate.frame_index] = candidate
    return [selected[index] for index in sorted(selected)]


def smoothed_scores(probabilities: Sequence[float], window_frames: int = 15) -> list[float]:
    """Return a centered, edge-replicated moving average for valley detection."""
    if window_frames < 1 or window_frames % 2 == 0:
        raise ValueError("平滑窗口必须是正奇数")
    values = [float(value) for value in probabilities]
    if not values:
        return []
    radius = window_frames // 2
    padded = [values[0]] * radius + values + [values[-1]] * radius
    return [sum(padded[index:index + window_frames]) / window_frames for index in range(len(values))]


def high_score_regions(probabilities: Sequence[float], threshold: float, merge_gap: int) -> list[dict]:
    """Decode threshold-hit regions using the same merge rule as the model decoder."""
    if not 0 <= threshold <= 1 or merge_gap < 0:
        raise ValueError("高分区域参数无效")
    values = [float(value) for value in probabilities]
    hits = [index for index, value in enumerate(values) if value >= threshold]
    if not hits:
        return []
    groups: list[list[int]] = [[hits[0]]]
    for index in hits[1:]:
        if index - groups[-1][-1] > merge_gap + 1:
            groups.append([index])
        else:
            groups[-1].append(index)
    return [{"start_frame": group[0], "end_frame": group[-1],
             "peak_frame": max(group, key=lambda index: values[index]),
             "peak_score": float(max(values[index] for index in group))}
            for group in groups]


def intersects_tail_edge_window(region: dict, frame_count: int, edge_frames: int) -> bool:
    """Return whether a high-score region overlaps the final edge window."""
    if frame_count < 1:
        raise ValueError("frame_count 必须为正数")
    if edge_frames < 0:
        raise ValueError("edge_frames 不能为负数")
    tail_window_start = max(0, frame_count - edge_frames)
    return (int(region["start_frame"]) < frame_count
            and int(region["end_frame"]) >= tail_window_start)


def nearest_valley(values: Sequence[float], peak_frame: int, direction: int, search_frames: int,
                   minimum_drop: float = VALLEY_MINIMUM_DROP) -> int:
    """Find the nearest local minimum away from a peak, or the search edge.

    ``direction`` is -1 for the left valley and +1 for the right. A candidate
    must fall at least ``minimum_drop`` below the smoothed peak, preventing
    tiny ripples on a high-score plateau from becoming a false valley. On a
    flat eligible valley, the closest frame to the peak is selected.
    """
    if direction not in {-1, 1} or search_frames < 1 or not 0 <= minimum_drop <= 1:
        raise ValueError("谷底搜索方向或范围无效")
    n = len(values)
    if not 0 < peak_frame < n - 1:
        raise ValueError("峰值必须是内部帧")
    limit = max(0, peak_frame - search_frames) if direction < 0 else min(n - 1, peak_frame + search_frames)
    peak_score = float(values[peak_frame])
    accepted_score = peak_score - minimum_drop
    frame = peak_frame + direction
    while frame != limit + direction:
        if (0 < frame < n - 1 and values[frame] <= accepted_score
                and values[frame] <= values[frame - 1] and values[frame] <= values[frame + 1]):
            return frame
        frame += direction
    return limit


def valley_pair(probabilities: Sequence[float], peak_frame: int, fps: float,
                smoothing_frames: int = VALLEY_SMOOTHING_FRAMES,
                search_seconds: float = VALLEY_SEARCH_SECONDS,
                minimum_drop: float = VALLEY_MINIMUM_DROP) -> tuple[int, int]:
    """Return the left/right valley frames surrounding a middle peak."""
    if fps <= 0 or search_seconds <= 0:
        raise ValueError("fps 和谷底搜索秒数必须为正数")
    smoothed = smoothed_scores(probabilities, smoothing_frames)
    search_frames = max(1, round(fps * search_seconds))
    left = nearest_valley(smoothed, peak_frame, -1, search_frames, minimum_drop)
    right = nearest_valley(smoothed, peak_frame, 1, search_frames, minimum_drop)
    if not left < peak_frame < right:
        raise ValueError("无法构成有效的峰值谷底区间")
    return left, right


def discard_ranges(boundaries: Iterable[Boundary], fps: float, metadata: dict[str, dict] | None = None) -> tuple[list[dict], list[str]]:
    """Build paired discard ranges and resolve automatic conflicts.

    A manually moved endpoint makes its whole range manual.  Automatic ranges
    that overlap, or leave at most four seconds of retained video, retain the
    earlier range.  A manual range displaces a conflicting automatic range;
    two conflicting manual ranges are retained with a warning for review.
    """
    pairs: dict[str, dict[str, Boundary]] = {}
    warnings: list[str] = []
    for item in boundaries:
        if item.kind not in {"discard_start", "discard_end"}:
            continue
        if not item.pair_id:
            warnings.append(f"第 {item.frame_index} 帧删除端点缺少配对标识，已忽略。")
            continue
        values = pairs.setdefault(item.pair_id, {})
        side = "start" if item.kind == "discard_start" else "end"
        if side in values:
            warnings.append(f"删除区间 {item.pair_id} 存在重复{side}端点，已忽略重复项。")
            continue
        values[side] = item
    ranges: list[dict] = []
    for pair_id, endpoints in pairs.items():
        start, end = endpoints.get("start"), endpoints.get("end")
        if start is None or end is None or start.frame_index >= end.frame_index:
            warnings.append(f"删除区间 {pair_id} 端点无效，已忽略。")
            continue
        ranges.append({"pair_id": pair_id, "start": start, "end": end,
                       "manual": start.origin == "manual" or end.origin == "manual",
                       **(metadata or {}).get(pair_id, {})})
    ranges.sort(key=lambda item: (item["start"].frame_index, item["end"].frame_index, item["pair_id"]))
    kept: list[dict] = []
    minimum_frames = round(MINIMUM_CUT_GAP_SECONDS * fps)
    for candidate in ranges:
        if not kept:
            kept.append(candidate)
            continue
        previous = kept[-1]
        retained = candidate["start"].frame_index - previous["end"].frame_index
        if retained > minimum_frames:
            kept.append(candidate)
            continue
        if candidate["manual"] and not previous["manual"]:
            kept[-1] = candidate
            warnings.append(f"自动删除区间 {previous['pair_id']} 与人工调整区间冲突，已保留人工区间。")
        elif previous["manual"] and not candidate["manual"]:
            warnings.append(f"已舍弃自动删除区间 {candidate['pair_id']}：与人工调整区间相交或间隔不大于 {MINIMUM_CUT_GAP_SECONDS:g} 秒。")
        elif candidate["manual"] and previous["manual"]:
            kept.append(candidate)
            warnings.append(f"两个人工调整删除区间相交或间隔不大于 {MINIMUM_CUT_GAP_SECONDS:g} 秒，保留以供复核。")
        else:
            warnings.append(f"已舍弃自动删除区间 {candidate['pair_id']}：与前一删除区间相交或间隔不大于 {MINIMUM_CUT_GAP_SECONDS:g} 秒。")
    return kept, warnings


def serialize_discard_ranges(boundaries: Iterable[Boundary], fps: float,
                             metadata: dict[str, dict] | None = None) -> list[dict]:
    """Serialize valid paired deletion ranges without changing them."""
    ranges, _ = discard_ranges(boundaries, fps, metadata)
    return [{"pair_id": item["pair_id"], "start": item["start"].as_dict(fps),
             "end": item["end"].as_dict(fps), "manual": item["manual"],
             **{key: value for key, value in item.items()
                if key not in {"pair_id", "start", "end", "manual"}}}
            for item in ranges]


def reduce_short_video_middle_nodes(splits: Iterable[Boundary], ranges: Iterable[dict], frame_count: int,
                                    fps: float, maximum_seconds: float = SHORT_VIDEO_MIDDLE_NODE_LIMIT_SECONDS) -> tuple[list[Boundary], list[dict], list[str]]:
    """Limit automatic middle deletions for a short, single-action video.

    For videos shorter than ``maximum_seconds``, a pair of ordinary split
    boundaries forms the head/tail crop.  If there are multiple automatic
    middle nodes (paired deletion ranges or ordinary split points), retain at
    most one automatic deletion range: the range whose centre is nearest the
    video midpoint.  Ordinary automatic middle cuts and every other automatic
    range are ignored.  Manual edits are intentionally exempt because they
    are the final authority during review.
    """
    if frame_count < 1 or fps <= 0 or maximum_seconds <= 0:
        raise ValueError("视频帧数、fps 和短视频时长阈值必须为正数")
    resolved_splits = sorted((Boundary(item.frame_index, item.score, item.origin, item.kind, item.pair_id)
                              for item in splits if item.kind == "split"), key=lambda item: item.frame_index)
    resolved_ranges = sorted((dict(item) for item in ranges),
                             key=lambda item: (item["start"].frame_index, item["end"].frame_index, item["pair_id"]))
    if frame_count / fps >= maximum_seconds or len(resolved_splits) < 2:
        return resolved_splits, resolved_ranges, []

    head, tail = resolved_splits[0], resolved_splits[-1]
    middle_splits = resolved_splits[1:-1]
    middle_ranges = [item for item in resolved_ranges
                     if item["end"].frame_index > head.frame_index and item["start"].frame_index < tail.frame_index]
    automatic_splits = [item for item in middle_splits if item.origin != "manual"]
    automatic_ranges = [item for item in middle_ranges if not item["manual"]]
    if len(automatic_splits) + len(automatic_ranges) <= 1:
        return resolved_splits, resolved_ranges, []

    midpoint = frame_count / 2
    keep_pair_id: str | None = None
    if automatic_ranges:
        selected = min(automatic_ranges,
                       key=lambda item: (abs((item["start"].frame_index + item["end"].frame_index) / 2 - midpoint),
                                         item["start"].frame_index, item["pair_id"]))
        keep_pair_id = selected["pair_id"]

    removed_splits = {id(item) for item in automatic_splits}
    removed_pairs = {item["pair_id"] for item in automatic_ranges if item["pair_id"] != keep_pair_id}
    warnings: list[str] = []
    if removed_splits:
        warnings.append(f"短视频小于 {maximum_seconds:g} 秒：已忽略 {len(removed_splits)} 个自动中间切分点，仅保留头尾裁剪点。")
    if removed_pairs:
        warnings.append(f"短视频小于 {maximum_seconds:g} 秒：已忽略 {len(removed_pairs)} 个非中心自动删除区间，仅保留最接近视频中点的删除区间 {keep_pair_id}。")
    return ([item for item in resolved_splits if id(item) not in removed_splits],
            [item for item in resolved_ranges if item["pair_id"] not in removed_pairs], warnings)


def edge_peak_boundaries(probabilities: Sequence[float], fps: float, threshold: float,
                         edge_seconds: float = EDGE_WINDOW_SECONDS) -> list[Boundary]:
    """Add a peak cut at either clip edge when that edge has no threshold hit.

    The model's normal decoder remains responsible for every run at or above
    the selected UI threshold. This fallback deliberately checks the *frozen*
    effective decoding threshold: for the first and final edge-window
    seconds independently, it emits the highest-scoring internal frame only
    when no frame in that entire edge window reaches the effective decoding
    threshold.
    Clip endpoints themselves are excluded because they are metadata rather
    than meaningful segment boundaries.
    """
    if fps <= 0 or edge_seconds <= 0:
        raise ValueError("fps 和 edge_seconds 必须为正数")
    values = [float(value) for value in probabilities]
    frame_count = len(values)
    if frame_count < 3:
        return []
    edge_frames = max(1, round(fps * edge_seconds))
    windows = (("edge_peak_start", 0, min(frame_count, edge_frames)),
               ("edge_peak_end", max(0, frame_count - edge_frames), frame_count))
    selected: dict[int, Boundary] = {}
    for origin, window_start, window_end in windows:
        if any(value >= threshold for value in values[window_start:window_end]):
            continue
        # First/last clip frames cannot form an actual division, so select an
        # internal peak while retaining the requested edge-window test.
        candidate_start, candidate_end = max(1, window_start), min(frame_count - 1, window_end)
        if candidate_start >= candidate_end:
            continue
        frame = max(range(candidate_start, candidate_end), key=lambda index: values[index])
        selected.setdefault(frame, Boundary(frame, values[frame], origin))
    return [selected[index] for index in sorted(selected)]


def enforce_cut_gap(boundaries: Iterable[Boundary], fps: float,
                    minimum_seconds: float = MINIMUM_CUT_GAP_SECONDS) -> tuple[list[Boundary], list[str]]:
    """Keep the earlier point whenever adjacent cuts are too close together."""
    if fps <= 0 or minimum_seconds < 0:
        raise ValueError("fps 必须为正数且 minimum_seconds 不能为负数")
    ordered = sorted((Boundary(item.frame_index, item.score, item.origin, item.kind, item.pair_id) for item in boundaries),
                     key=lambda item: item.frame_index)
    kept: list[Boundary] = []
    warnings: list[str] = []
    for item in ordered:
        if not kept:
            kept.append(item)
            continue
        previous = kept[-1]
        interval = (item.frame_index - previous.frame_index) / fps
        # The product requirement is strictly greater than the configured gap.
        if interval > minimum_seconds:
            kept.append(item)
        else:
            warnings.append(f"已舍弃第 {item.frame_index} 帧切分点：与前一切分点相隔 {interval:.3f} 秒，不大于 {minimum_seconds:g} 秒。")
    return kept, warnings


def enforce_minimum_segments(boundaries: Iterable[Boundary], frame_count: int, minimum_frames: int) -> tuple[list[Boundary], list[str]]:
    """Remove low-confidence automatic cuts that create a too-short segment.

    Manually inserted cuts are never removed. If two manual cuts conflict, the
    result remains editable and a warning is returned to the caller.
    """
    if frame_count < 1:
        raise ValueError("frame_count 必须为正数")
    if minimum_frames < 0:
        raise ValueError("minimum_frames 不能为负数")
    current = deduplicate_boundaries(boundaries, frame_count)
    warnings: list[str] = []
    while minimum_frames and len(current) >= 0:
        anchors: list[Boundary | None] = [None, *current, None]
        frames = [0, *(item.frame_index for item in current), frame_count]
        conflict = next(((index, frames[index + 1] - frames[index]) for index in range(len(frames) - 1)
                         if frames[index + 1] - frames[index] < minimum_frames), None)
        if conflict is None:
            break
        index, length = conflict
        removable = [item for item in (anchors[index], anchors[index + 1])
                     if item is not None and item.origin != "manual"]
        if not removable:
            warnings.append(f"存在 {length} 帧的过短片段；两侧均为人工边界，未自动删除。")
            break
        discard = min(removable, key=lambda item: (float("inf") if item.score is None else item.score, item.frame_index))
        current.remove(discard)
        warnings.append(f"已移除第 {discard.frame_index} 帧的自动边界，以满足最短片段时长。")
    return current, warnings


def export_intervals(boundaries: Iterable[Boundary], frame_count: int, trim_head_tail: bool = True,
                     discard_intervals: Iterable[tuple[int, int]] = ()) -> list[tuple[int, int]]:
    """Return retained frame intervals after crop, ordinary splits, and deletions.

    Only ``split`` boundaries divide exported files.  Paired discard endpoints
    remove their half-open range ``[start, end)`` from those files.
    """
    splits = [item.frame_index for item in deduplicate_boundaries(boundaries, frame_count)
              if item.kind == "split"]
    if trim_head_tail:
        if len(splits) < 2:
            return []
        base = list(zip(splits, splits[1:]))
    else:
        anchors = [0, *splits, frame_count]
        base = list(zip(anchors, anchors[1:]))
    discarded = sorted((max(0, int(start)), min(frame_count, int(end)))
                       for start, end in discard_intervals if int(start) < int(end))
    retained: list[tuple[int, int]] = []
    for start, end in base:
        pieces = [(start, end)]
        for discard_start, discard_end in discarded:
            next_pieces: list[tuple[int, int]] = []
            for piece_start, piece_end in pieces:
                if discard_end <= piece_start or discard_start >= piece_end:
                    next_pieces.append((piece_start, piece_end))
                    continue
                if piece_start < discard_start:
                    next_pieces.append((piece_start, discard_start))
                if discard_end < piece_end:
                    next_pieces.append((discard_end, piece_end))
            pieces = next_pieces
        retained.extend(piece for piece in pieces if piece[0] < piece[1])
    return retained


def resolve_short_export_segments(intervals: Iterable[tuple[int, int]], frame_count: int) -> tuple[list[list[tuple[int, int]]], list[str]]:
    """Apply the final short-export-segment rule without restoring deletions.

    A retained interval shorter than one twentieth of the original video is
    discarded when it was originally the second or penultimate output. Other
    short intervals are joined to the shorter adjacent output.  A joined
    output may contain multiple source intervals when a discard range lies
    between them; the exporter concatenates those retained pieces instead of
    putting the deleted material back.
    """
    if frame_count < 1:
        raise ValueError("视频帧数必须为正数")
    source = [(int(start), int(end)) for start, end in intervals if int(start) < int(end)]
    groups: list[list[tuple[int, int]]] = [[item] for item in source]
    warnings: list[str] = []
    threshold = frame_count / 20
    discard_indexes = {index for index, (start, end) in enumerate(source)
                       if end - start < threshold and (index == 1 or index == len(source) - 2)}
    if discard_indexes:
        groups = [group for index, group in enumerate(groups) if index not in discard_indexes]
        for index in sorted(discard_indexes):
            warnings.append(f"已舍弃第 {index + 1} 个导出片段：{source[index][1] - source[index][0]} 帧，小于视频总时长的 1/20。")

    def duration(group: list[tuple[int, int]]) -> int:
        return sum(end - start for start, end in group)

    while len(groups) > 1:
        index = next((position for position, group in enumerate(groups) if duration(group) < threshold), None)
        if index is None:
            break
        choices = [position for position in (index - 1, index + 1) if 0 <= position < len(groups)]
        target = min(choices, key=lambda position: (duration(groups[position]), position))
        short_duration, target_duration = duration(groups[index]), duration(groups[target])
        if target < index:
            groups[target].extend(groups[index])
        else:
            groups[target][:0] = groups[index]
        groups.pop(index)
        warnings.append(f"已将 {short_duration} 帧的短导出片段并入相邻较短片段（原 {target_duration} 帧）。")
    return groups, warnings


class InferenceService:
    """Stateful inference service; call :meth:`load_model` once per queue."""

    def __init__(self, package_root: Path | None = None, model_root: Path | None = None) -> None:
        self.package_root = Path(package_root or Path(__file__).resolve().parents[1]).resolve()
        self.model_root = Path(model_root or self.package_root / "model").resolve()
        self.loaded: tuple | None = None

    def verify_package(self) -> None:
        """Validate every file listed in the immutable model package manifest."""
        manifest_path = self.model_root / "package_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = manifest["files"]
        except (OSError, KeyError, json.JSONDecodeError) as error:
            raise CutterError("无法读取模型包清单") from error
        def manifest_file(rel: str) -> Path:
            # Frozen manifests name model files as ``model/...`` while a
            # registered release stores that model package in its own folder.
            return self.model_root / rel.removeprefix("model/") if rel.startswith("model/") else self.package_root / rel

        bad = [rel for rel, expected in files.items()
               if not manifest_file(rel).is_file() or file_digest(manifest_file(rel)) != expected]
        if bad:
            raise CutterError("模型包校验失败：" + ", ".join(bad))

    def load_model(self) -> tuple:
        if self.loaded is None:
            self.verify_package()
            # Imported lazily so scanner/settings unit tests do not require ML dependencies.
            from run_inference import policy_and_model
            self.loaded = policy_and_model(self.model_root)
        return self.loaded

    def persist(self, job: VideoJob) -> None:
        atomic_json(job.job_path, job.to_status_dict())

    def prepare(self, job: VideoJob, update: Callable[[VideoJob], None] | None = None) -> None:
        job.status, job.progress, job.error = "预处理中", "读取媒体信息", None
        self.persist(job)
        if update:
            update(job)
        metadata = probe_video(job.source)
        job.duration_seconds, job.source_fps = metadata["duration_seconds"], metadata["fps"]
        job.output_dir = job.output_root / job.identifier
        job.output_dir.mkdir(parents=True, exist_ok=True)
        if abs(job.source_fps - 30.0) <= FPS_EPSILON:
            job.work_video, job.work_fps = job.source, job.source_fps
        else:
            job.progress = "转换为模型所需的 30 FPS"
            self.persist(job)
            if update:
                update(job)
            job.work_video = normalize_to_30fps(job.source, job.output_dir / "working_30fps.mp4")
            job.work_fps = 30.0
        self.persist(job)

    def infer(self, job: VideoJob, settings: CutSettings, update: Callable[[VideoJob], None] | None = None) -> VideoJob:
        settings.validate()
        try:
            job.effective_settings = asdict(settings)
            self.prepare(job, update)
            policy, checkpoint_path, model, normalizer, checkpoint, device = self.load_model()
            job.status, job.progress = "推理中", "提取手部特征"
            self.persist(job)
            if update:
                update(job)
            from data import extract_video
            feature_path = job.output_dir / "features.npz"
            extract_video(job.work_video, feature_path, model=self.model_root / "assets/hand_landmarker.task")
            job.features_path = feature_path
            job.progress = "计算切分概率"
            self.persist(job)
            if update:
                update(job)
            import numpy as np
            from core import normalize
            from learning import predict_probabilities
            with np.load(feature_path, allow_pickle=False) as raw:
                sequence = {key: raw[key] for key in raw.files}
            config = checkpoint["config"]
            probabilities = predict_probabilities(model, normalize(sequence, normalizer), device,
                                                  config["window"], config["stride"], config["batch_size"])
            probability_path = job.output_dir / "probabilities.npz"
            np.savez_compressed(probability_path, probabilities=probabilities)
            job.frame_count = int(len(probabilities))
            job.work_fps = float(sequence["fps"])
            regions = high_score_regions(probabilities, settings.threshold, settings.merge_gap)
            edge_frames = round(job.work_fps * EDGE_WINDOW_SECONDS)
            normal_boundaries: list[Boundary] = []
            job.discard_pair_metadata = {}
            job.tail_crop_metadata = None
            for index, region in enumerate(regions, start=1):
                frame = int(region["peak_frame"])
                # Only a high-score region that overlaps the final edge window
                # is a tail crop; every earlier region remains a middle peak.
                if intersects_tail_edge_window(region, job.frame_count, edge_frames):
                    cut_frame = min(job.frame_count - 1, region["end_frame"] + 1)
                    normal_boundaries.append(Boundary(cut_frame, float(probabilities[cut_frame]),
                                                      "tail_high_region_end"))
                    if job.tail_crop_metadata is None:
                        job.tail_crop_metadata = {
                            "reason": "high_score_region_intersects_tail_edge_window",
                            "edge_window_seconds": EDGE_WINDOW_SECONDS,
                            "region": region, "crop_boundary_frame": cut_frame,
                            "intersects_tail_edge_window": True,
                        }
                    continue
                # The first edge window remains an ordinary crop boundary.
                if frame <= edge_frames:
                    normal_boundaries.append(Boundary(frame, float(probabilities[frame]), "auto"))
                    continue
                start, end = valley_pair(probabilities, frame, job.work_fps)
                pair_id = f"auto_discard_{index:03d}"
                normal_boundaries.extend([
                    Boundary(start, float(probabilities[start]), "auto", "discard_start", pair_id),
                    Boundary(end, float(probabilities[end]), "auto", "discard_end", pair_id),
                ])
                job.discard_pair_metadata[pair_id] = {
                    "peak_frame": frame, "peak_score": float(probabilities[frame]),
                    "smoothing_frames": VALLEY_SMOOTHING_FRAMES,
                    "search_seconds": VALLEY_SEARCH_SECONDS,
                    "minimum_score_drop": VALLEY_MINIMUM_DROP,
                }
            fallback_boundaries = edge_peak_boundaries(probabilities, job.work_fps, settings.threshold)
            decoded_boundaries = deduplicate_boundaries([*normal_boundaries, *fallback_boundaries], job.frame_count)
            ranges, range_warnings = discard_ranges(decoded_boundaries, job.work_fps, job.discard_pair_metadata)
            split_boundaries, split_warnings = enforce_cut_gap(
                [item for item in decoded_boundaries if item.kind == "split"], job.work_fps
            )
            split_boundaries, ranges, short_video_warnings = reduce_short_video_middle_nodes(
                split_boundaries, ranges, job.frame_count, job.work_fps
            )
            kept_pair_ids = {item["pair_id"] for item in ranges}
            job.original_boundaries = sorted([
                *split_boundaries,
                *(item for item in decoded_boundaries
                  if item.kind != "split" and item.pair_id in kept_pair_ids),
            ], key=lambda item: item.frame_index)
            job.discard_pair_metadata = {key: value for key, value in job.discard_pair_metadata.items()
                                         if key in kept_pair_ids}
            job.warnings = [*split_warnings, *range_warnings, *short_video_warnings]
            job.boundaries = list(job.original_boundaries)
            job.probabilities_path = probability_path
            job.inference_path = job.output_dir / "inference.json"
            atomic_json(job.inference_path, {"schema": "data-clip.desktop-inference.v2",
                                              "source_video": str(job.source), "working_video": str(job.work_video),
                                              "frame_count": job.frame_count, "fps": job.work_fps,
                                              "settings": asdict(settings), "model_checkpoint_sha256": file_digest(checkpoint_path),
                                              "edge_peak_fallback": {"window_seconds": EDGE_WINDOW_SECONDS,
                                                                     "threshold": settings.threshold,
                                                                     "frozen_package_threshold": float(policy["threshold"])},
                                              "minimum_cut_gap_seconds": MINIMUM_CUT_GAP_SECONDS,
                                              "short_video_middle_node_policy": {
                                                  "maximum_duration_seconds": SHORT_VIDEO_MIDDLE_NODE_LIMIT_SECONDS,
                                                  "enabled": True,
                                                  "manual_nodes_preserved": True,
                                              },
                                              "high_score_region_policy": {
                                                  "threshold": settings.threshold, "merge_gap_frames": settings.merge_gap,
                                                  "tail_edge_window_seconds": EDGE_WINDOW_SECONDS,
                                                  "tail_high_score_intersects_edge_window_only": True,
                                                  "tail_crop_after_region": True,
                                              },
                                              "tail_crop": job.tail_crop_metadata,
                                              "valley_pair_policy": {"edge_seconds": EDGE_WINDOW_SECONDS,
                                                                     "smoothing_frames": VALLEY_SMOOTHING_FRAMES,
                                                                     "search_seconds": VALLEY_SEARCH_SECONDS,
                                                                     "minimum_score_drop": VALLEY_MINIMUM_DROP,
                                                                     "interior_peak_only": True},
                                              "discard_ranges": serialize_discard_ranges(
                                                  job.original_boundaries, job.work_fps, job.discard_pair_metadata),
                                              "boundaries": [item.as_dict(job.work_fps) for item in job.original_boundaries]})
            job.status, job.progress = "等待复核", "可在时间轴调整边界后导出"
            self.persist(job)
        except Exception as error:
            job.status, job.progress, job.error = "失败", "", str(error)
            self.persist(job)
        if update:
            update(job)
        return job

    def export(self, job: VideoJob, settings: CutSettings, update: Callable[[VideoJob], None] | None = None) -> VideoJob:
        settings.validate()
        if not job.work_video or not job.frame_count or not job.work_fps:
            raise CutterError("该任务尚未完成推理")
        try:
            job.status, job.progress = "导出中", "整理最终边界"
            minimum_frames = round(settings.minimum_segment_seconds * job.work_fps)
            split_boundaries = [item for item in job.boundaries if item.kind == "split"]
            gap_resolved, gap_warnings = enforce_cut_gap(split_boundaries, job.work_fps)
            resolved_splits, minimum_warnings = enforce_minimum_segments(gap_resolved, job.frame_count, minimum_frames)
            ranges, range_warnings = discard_ranges(job.boundaries, job.work_fps, job.discard_pair_metadata)
            resolved_splits, ranges, short_video_warnings = reduce_short_video_middle_nodes(
                resolved_splits, ranges, job.frame_count, job.work_fps
            )
            kept_pair_ids = {item["pair_id"] for item in ranges}
            resolved = sorted([
                *resolved_splits,
                *(item for item in job.boundaries
                  if item.kind != "split" and item.pair_id in kept_pair_ids),
            ], key=lambda item: item.frame_index)
            warnings = [*gap_warnings, *minimum_warnings, *range_warnings, *short_video_warnings]
            job.boundaries, job.warnings = resolved, warnings
            self.persist(job)
            if update:
                update(job)
            ffmpeg = require_program("ffmpeg")
            segments_dir = job.output_dir / "segments"
            segments_dir.mkdir(parents=True, exist_ok=True)
            intervals = export_intervals(
                resolved, job.frame_count, trim_head_tail=True,
                discard_intervals=[(item["start"].frame_index, item["end"].frame_index) for item in ranges],
            )
            segment_groups, short_export_warnings = resolve_short_export_segments(intervals, job.frame_count)
            job.warnings.extend(short_export_warnings)
            segments = []
            if not segment_groups:
                job.warnings.append("掐头去尾需要至少两个普通切分点；未导出保留视频片段。")
            for index, source_intervals in enumerate(segment_groups, start=1):
                target = segments_dir / f"segment_{index:03d}.mp4"
                start, end = source_intervals[0][0], source_intervals[-1][1]
                start_seconds, end_seconds = start / job.work_fps, end / job.work_fps
                job.progress = f"导出片段 {index}/{len(segment_groups)}"
                self.persist(job)
                if update:
                    update(job)
                self.export_segment_group(ffmpeg, job.work_video, source_intervals, job.work_fps, target, segments_dir)
                segments.append({"index": index, "file": str(target.relative_to(job.output_dir)), "start_frame": start,
                                 "end_frame_exclusive": end, "start_seconds": start_seconds, "end_seconds_exclusive": end_seconds,
                                 "duration_seconds": sum(stop - begin for begin, stop in source_intervals) / job.work_fps,
                                 "source_intervals": [{"start_frame": begin, "end_frame_exclusive": stop}
                                                      for begin, stop in source_intervals]})
            keyframes = self.export_keyframes(job, resolved) if settings.export_keyframes else []
            inference_metadata = json.loads(job.inference_path.read_text(encoding="utf-8")) if job.inference_path else {}
            trim_points = [item.frame_index for item in resolved_splits]
            report = {"schema": "data-clip.desktop-result.v2", "source_video": str(job.source),
                      "source_video_sha256": file_digest(job.source),
                      "working_video": str(job.work_video), "fps": job.work_fps, "frame_count": job.frame_count,
                      "model_checkpoint_sha256": inference_metadata.get("model_checkpoint_sha256"),
                      "inference_settings": job.effective_settings, "export_settings": asdict(settings),
                      "tail_crop": inference_metadata.get("tail_crop"),
                      "automatic_boundaries": [item.as_dict(job.work_fps) for item in job.original_boundaries],
                      "automatic_discard_ranges": serialize_discard_ranges(
                          job.original_boundaries, job.work_fps, job.discard_pair_metadata),
                      "final_boundaries": [item.as_dict(job.work_fps) for item in resolved], "warnings": job.warnings,
                      "discard_ranges": serialize_discard_ranges(resolved, job.work_fps, job.discard_pair_metadata),
                      "trim_policy": {"drop_head_before_first_boundary": True,
                                      "drop_tail_after_last_boundary": True,
                                      "discarded_head": {"start_frame": 0, "end_frame_exclusive": trim_points[0]
                                                         if len(trim_points) >= 2 else job.frame_count},
                                      "discarded_tail": {"start_frame": trim_points[-1] if len(trim_points) >= 2 else 0,
                                                         "end_frame_exclusive": job.frame_count}},
                      "minimum_cut_gap_seconds": MINIMUM_CUT_GAP_SECONDS,
                      "short_video_middle_node_policy": {
                          "maximum_duration_seconds": SHORT_VIDEO_MIDDLE_NODE_LIMIT_SECONDS,
                          "enabled": True,
                          "manual_nodes_preserved": True,
                      },
                      "short_export_segment_policy": {"maximum_fraction_of_video": 0.05,
                                                      "second_and_penultimate": "discard",
                                                      "other_positions": "merge_into_shorter_adjacent_segment"},
                      "segments": segments, "keyframes": keyframes}
            atomic_json(job.output_dir / "result.json", report)
            job.status, job.progress = "已导出", f"已导出 {len(segments)} 个片段"
            self.persist(job)
        except Exception as error:
            job.status, job.progress, job.error = "失败", "", str(error)
            self.persist(job)
        if update:
            update(job)
        return job

    @staticmethod
    def export_segment_group(ffmpeg: str, video: Path, source_intervals: list[tuple[int, int]], fps: float,
                             target: Path, temporary_root: Path) -> None:
        """Encode one retained interval or concatenate several retained pieces."""
        def encode(source: tuple[int, int], destination: Path) -> None:
            start, end = source
            run_command([ffmpeg, "-y", "-i", str(video), "-ss", f"{start / fps:.6f}", "-to", f"{end / fps:.6f}",
                         "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                         "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(destination)])

        if len(source_intervals) == 1:
            encode(source_intervals[0], target)
            return
        with tempfile.TemporaryDirectory(prefix=".concat_", dir=temporary_root) as directory:
            work = Path(directory)
            pieces = []
            for index, interval in enumerate(source_intervals, start=1):
                piece = work / f"piece_{index:03d}.mp4"
                encode(interval, piece)
                pieces.append(piece)
            listing = work / "inputs.txt"
            listing.write_text("".join(f"file '{str(piece).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\\n"
                                       for piece in pieces), encoding="utf-8")
            run_command([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
                         "-movflags", "+faststart", str(target)])

    def export_keyframes(self, job: VideoJob, boundaries: list[Boundary]) -> list[dict]:
        ffmpeg = require_program("ffmpeg")
        target_dir = job.output_dir / "cut_frames"
        target_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for index, item in enumerate(boundaries, start=1):
            target = target_dir / f"cut_{index:03d}_frame_{item.frame_index:06d}.jpg"
            run_command([ffmpeg, "-y", "-ss", f"{item.frame_index / job.work_fps:.6f}", "-i", str(job.work_video),
                         "-frames:v", "1", "-q:v", "2", str(target)])
            results.append({**item.as_dict(job.work_fps), "image": str(target.relative_to(job.output_dir))})
        atomic_json(target_dir / "manifest.json", {"frames": results})
        return results
