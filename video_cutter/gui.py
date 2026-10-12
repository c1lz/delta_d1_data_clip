"""Chinese PySide6 desktop UI for reviewing and exporting gesture cuts."""
from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path
import sys

from PySide6.QtCore import QObject, QThread, Qt, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QPainter, QPen
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QPushButton, QSpinBox, QSplitter, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from .service import (DEFAULT_CUT_THRESHOLD, DEFAULT_DATASET_ROOT, DEFAULT_EXPORT_ROOT, Boundary, CutSettings, CutterError, DeltaSession,
                      InferenceService, VideoJob, discover_delta_sessions)


LOGGER = logging.getLogger("data_clip.desktop")


class TimelineWidget(QWidget):
    """Editable score timeline with ordinary cuts and paired discard ranges."""

    boundariesChanged = Signal(object)
    seekRequested = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        LOGGER.info("Building main window")
        self.setMinimumHeight(150)
        self.setToolTip("左键空白处新增普通切分点；拖动端点调整；右键删除普通点或整个删除区间。")
        self.probabilities = []
        self.fps = 30.0
        self.frame_count = 0
        self.boundaries: list[Boundary] = []
        self.selected: int | None = None

    def set_data(self, probabilities, fps: float, boundaries: list[Boundary]) -> None:
        self.probabilities = probabilities.tolist() if hasattr(probabilities, "tolist") else list(probabilities)
        self.frame_count, self.fps = len(self.probabilities), fps
        self.boundaries = [replace(item) for item in boundaries]
        self.selected = None
        self.update()

    def x_for_frame(self, frame: int) -> float:
        return 8 + (self.width() - 16) * frame / max(self.frame_count - 1, 1)

    def frame_for_x(self, x: float) -> int:
        return max(1, min(self.frame_count - 1, round((x - 8) * max(self.frame_count - 1, 1) / max(self.width() - 16, 1))))

    def closest(self, x: float, threshold: float = 10) -> int | None:
        candidates = [(abs(self.x_for_frame(item.frame_index) - x), index) for index, item in enumerate(self.boundaries)]
        return min(candidates)[1] if candidates and min(candidates)[0] <= threshold else None

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API name
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#fbfbfb"))
        plot = self.rect().adjusted(8, 12, -8, -28)
        painter.setPen(QPen(QColor("#d1d5db"), 1))
        painter.drawLine(plot.left(), plot.bottom(), plot.right(), plot.bottom())
        painter.drawLine(plot.left(), plot.top(), plot.left(), plot.bottom())
        if self.frame_count:
            painter.setPen(QPen(QColor("#2563eb"), 1.2))
            previous = None
            step = max(1, self.frame_count // max(plot.width() * 2, 1))
            for frame in range(0, self.frame_count, step):
                x = self.x_for_frame(frame)
                y = plot.bottom() - float(self.probabilities[frame]) * plot.height()
                if previous:
                    painter.drawLine(previous[0], previous[1], x, y)
                previous = (x, y)
        paired: dict[str, dict[str, Boundary]] = {}
        for item in self.boundaries:
            if item.pair_id and item.kind in {"discard_start", "discard_end"}:
                paired.setdefault(item.pair_id, {})["start" if item.kind == "discard_start" else "end"] = item
        for endpoints in paired.values():
            start, end = endpoints.get("start"), endpoints.get("end")
            if start is None or end is None:
                continue
            left, right = sorted((self.x_for_frame(start.frame_index), self.x_for_frame(end.frame_index)))
            painter.fillRect(round(left), plot.top(), max(1, round(right - left)), plot.height(), QColor("#fecaca"))
        for index, item in enumerate(self.boundaries):
            if item.origin == "manual":
                color = QColor("#dc2626")
            elif item.kind in {"discard_start", "discard_end"}:
                color = QColor("#059669")
            else:
                color = QColor("#f59e0b")
            if index == self.selected:
                color = QColor("#7c3aed")
            painter.setPen(QPen(color, 2))
            x = self.x_for_frame(item.frame_index)
            painter.drawLine(x, plot.top(), x, plot.bottom())
        painter.setPen(QPen(QColor("#4b5563"), 1))
        duration = self.frame_count / self.fps if self.fps else 0
        painter.drawText(8, self.height() - 8, "0:00")
        painter.drawText(self.width() - 90, self.height() - 8, f"{duration:.2f} 秒")

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if not self.frame_count:
            return
        selected = self.closest(event.position().x())
        if event.button() == Qt.RightButton:
            if selected is not None:
                item = self.boundaries[selected]
                if item.pair_id and item.kind in {"discard_start", "discard_end"}:
                    self.boundaries = [value for value in self.boundaries if value.pair_id != item.pair_id]
                else:
                    self.boundaries.pop(selected)
                self.selected = None
                self.boundariesChanged.emit([replace(item) for item in self.boundaries])
                self.update()
            return
        if selected is None:
            frame = self.frame_for_x(event.position().x())
            score = self.probabilities[frame] if frame < len(self.probabilities) else None
            self.boundaries.append(Boundary(frame, float(score) if score is not None else None, "manual"))
            self.boundaries.sort(key=lambda item: item.frame_index)
            selected = next(index for index, item in enumerate(self.boundaries) if item.frame_index == frame)
            self.boundariesChanged.emit([replace(item) for item in self.boundaries])
        self.selected = selected
        self.seekRequested.emit(self.boundaries[selected].frame_index / self.fps)
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self.selected is None or not self.frame_count:
            return
        frame = self.frame_for_x(event.position().x())
        item = self.boundaries[self.selected]
        item.frame_index, item.origin = frame, "manual"
        item.score = self.probabilities[frame] if frame < len(self.probabilities) else item.score
        self.boundaries = sorted(self.boundaries, key=lambda value: value.frame_index)
        self.selected = next(index for index, value in enumerate(self.boundaries) if value is item)
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self.selected is not None:
            self.boundariesChanged.emit([replace(item) for item in self.boundaries])


class QueueWorker(QObject):
    jobUpdated = Signal(object)
    completed = Signal()

    def __init__(self, jobs: list[VideoJob], settings: CutSettings) -> None:
        super().__init__()
        self.jobs, self.settings = jobs, settings

    def run(self) -> None:
        service = InferenceService()
        active_jobs = [job for job in self.jobs if job.status == "等待处理"]
        for job in active_jobs:
            job.status, job.progress, job.error = "准备模型", "校验并加载冻结模型", None
            service.persist(job)
            self.jobUpdated.emit(job)
        LOGGER.info("Queue worker loading model for %d jobs", len(active_jobs))
        try:
            service.load_model()
        except Exception as error:
            LOGGER.exception("Queue worker could not load model")
            for job in active_jobs:
                job.status, job.progress, job.error = "失败", "", str(error)
                service.persist(job)
                self.jobUpdated.emit(job)
            self.completed.emit()
            return
        for job in active_jobs:
            LOGGER.info("Starting inference for %s", job.source)
            service.infer(job, self.settings, self.jobUpdated.emit)
        self.completed.emit()


class ExportWorker(QObject):
    jobUpdated = Signal(object)
    completed = Signal()

    def __init__(self, jobs: list[VideoJob], settings: CutSettings) -> None:
        super().__init__()
        self.jobs, self.settings = jobs, settings

    def run(self) -> None:
        service = InferenceService()
        for job in self.jobs:
            if job.status in {"等待复核", "已导出"}:
                LOGGER.info("Exporting %s", job.source)
                service.export(job, self.settings, self.jobUpdated.emit)
        self.completed.emit()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("data_clip Video Cutter")
        self.resize(1280, 820)
        self.jobs: list[VideoJob] = []
        self.sessions: list[DeltaSession] = []
        self.selected_date_task: Path | None = None
        self.worker_thread: QThread | None = None
        self.worker: QObject | None = None
        self.timeline = TimelineWidget()
        # Qt's audio backend can block on machines without a working
        # PulseAudio/PipeWire session. Preview is video-only, so defer the
        # multimedia backend until the user actually selects a completed job.
        self.player: QMediaPlayer | None = None
        self.video_widget = QVideoWidget()
        self.build_ui()
        LOGGER.info("Main window ready")

    def verify_model_package(self) -> bool:
        try:
            InferenceService().verify_package()
            return True
        except CutterError as error:
            QMessageBox.critical(self, "模型包校验失败", str(error))
            return False

    def build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        toolbar = QHBoxLayout()
        import_button = QPushButton("选择日期-任务文件夹")
        import_button.clicked.connect(self.import_folder)
        self.task_label = QLabel("未选择日期-任务文件夹")
        self.start_sequence = QSpinBox()
        self.end_sequence = QSpinBox()
        for spin in (self.start_sequence, self.end_sequence):
            spin.setRange(0, 0)
            spin.setEnabled(False)
        self.load_range_button = QPushButton("载入所选序号")
        self.load_range_button.setEnabled(False)
        self.load_range_button.clicked.connect(self.load_selected_range)
        self.start_button = QPushButton("开始推理")
        self.start_button.clicked.connect(self.start_inference)
        self.export_current_button = QPushButton("导出当前")
        self.export_current_button.clicked.connect(lambda: self.start_export(False))
        self.export_all_button = QPushButton("导出全部")
        self.export_all_button.clicked.connect(lambda: self.start_export(True))
        toolbar.addWidget(import_button)
        toolbar.addWidget(self.task_label, 1)
        toolbar.addWidget(QLabel("序号范围"))
        toolbar.addWidget(self.start_sequence)
        toolbar.addWidget(QLabel("至"))
        toolbar.addWidget(self.end_sequence)
        toolbar.addWidget(self.load_range_button)
        toolbar.addWidget(self.start_button)
        toolbar.addWidget(self.export_current_button)
        toolbar.addWidget(self.export_all_button)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Vertical)
        top = QWidget()
        top_layout = QHBoxLayout(top)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["任务序号 / cam0", "时长", "源帧率", "状态", "切分点", "提示 / 失败原因"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self.load_selected_job)
        self.table.horizontalHeader().setStretchLastSection(True)
        top_layout.addWidget(self.table, 4)
        top_layout.addWidget(self.settings_box(), 1)
        splitter.addWidget(top)
        preview = QWidget()
        preview_layout = QGridLayout(preview)
        self.video_widget.setMinimumHeight(270)
        self.preview_label = QLabel("选择已完成推理的视频以预览、编辑切分点。")
        self.preview_label.setWordWrap(True)
        preview_layout.addWidget(self.video_widget, 0, 0)
        preview_layout.addWidget(self.preview_label, 0, 1)
        preview_layout.addWidget(self.timeline, 1, 0, 1, 2)
        splitter.addWidget(preview)
        splitter.setSizes([350, 440])
        layout.addWidget(splitter)
        self.timeline.boundariesChanged.connect(self.boundaries_changed)
        self.timeline.seekRequested.connect(self.seek_preview)
        quit_action = QAction("退出", self)
        quit_action.triggered.connect(self.close)
        self.menuBar().addMenu("文件").addAction(quit_action)

    def settings_box(self) -> QGroupBox:
        box = QGroupBox("本次任务设置")
        form = QFormLayout(box)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0, 1)
        self.threshold.setDecimals(3)
        self.threshold.setSingleStep(0.01)
        self.threshold.setValue(DEFAULT_CUT_THRESHOLD)
        self.merge_gap = QSpinBox()
        self.merge_gap.setRange(0, 300)
        self.merge_gap.setValue(6)
        self.minimum_duration = QDoubleSpinBox()
        self.minimum_duration.setRange(0, 60)
        self.minimum_duration.setDecimals(2)
        self.minimum_duration.setSuffix(" 秒")
        self.keyframes = QCheckBox("导出关键帧图片")
        self.keyframes.setChecked(True)
        self.output_directory = QLabel(f"默认：{DEFAULT_EXPORT_ROOT}")
        self.output_directory.setWordWrap(True)
        choose_output = QPushButton("选择导出目录")
        choose_output.clicked.connect(self.choose_output)
        form.addRow("阈值", self.threshold)
        form.addRow("合并间隔", self.merge_gap)
        form.addRow("最短片段", self.minimum_duration)
        form.addRow(self.keyframes)
        form.addRow("导出目录", self.output_directory)
        form.addRow(choose_output)
        return box

    def settings(self) -> CutSettings:
        return CutSettings(self.threshold.value(), self.merge_gap.value(), self.minimum_duration.value(), self.keyframes.isChecked()).validate()

    def output_root(self) -> Path | None:
        value = self.output_directory.property("path")
        return Path(value).resolve() if value else DEFAULT_EXPORT_ROOT

    def choose_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if folder:
            self.output_directory.setProperty("path", folder)
            self.output_directory.setText(folder)
            for job in self.jobs:
                if job.status == "等待处理":
                    job.output_root, job.output_dir = Path(folder).resolve(), None

    def import_folder(self) -> None:
        initial = DEFAULT_DATASET_ROOT if DEFAULT_DATASET_ROOT.is_dir() else Path.home()
        folder = QFileDialog.getExistingDirectory(self, "选择 Delta-D1 的日期-任务文件夹", str(initial))
        if not folder:
            return
        try:
            date, task_name, sessions = discover_delta_sessions(Path(folder))
        except ValueError as error:
            QMessageBox.warning(self, "无法导入", str(error))
            return
        self.selected_date_task, self.sessions = Path(folder).resolve(), sessions
        numbers = [item.sequence_number for item in sessions]
        for spin in (self.start_sequence, self.end_sequence):
            spin.setRange(min(numbers), max(numbers))
            spin.setEnabled(True)
        self.start_sequence.setValue(min(numbers))
        self.end_sequence.setValue(max(numbers))
        self.load_range_button.setEnabled(True)
        self.task_label.setText(f"日期：{date}　任务：{task_name}　可选 {min(numbers)}–{max(numbers)}（{len(sessions)} 个）")
        self.jobs = []
        self.refresh_table()

    def load_selected_range(self) -> None:
        """Build jobs only for the selected numeric range's cam0 video."""
        if not self.sessions:
            return
        first, last = self.start_sequence.value(), self.end_sequence.value()
        if first > last:
            QMessageBox.warning(self, "序号范围无效", "起始序号不能大于结束序号。")
            return
        output_root = self.output_root()
        if self.output_directory.property("path") is None and self.selected_date_task:
            # Different recording dates can reuse the same task sequence name.
            output_root = output_root / self.selected_date_task.name
        self.jobs = []
        for session in self.sessions:
            if not first <= session.sequence_number <= last:
                continue
            source = session.cam0_video or session.directory / "cam0" / "cam0-00000.mp4"
            job = VideoJob(source, output_root, session_name=session.name)
            if session.issue:
                job.status, job.error = "不可用", session.issue
            self.jobs.append(job)
        self.refresh_table()

    def refresh_table(self) -> None:
        selected = self.current_job()
        self.table.setUpdatesEnabled(False)
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(self.jobs))
            for row, job in enumerate(self.jobs):
                values = [f"{job.session_name or job.source.parents[1].name} / cam0",
                          f"{job.duration_seconds:.2f} 秒" if job.duration_seconds else "待检测",
                          f"{job.source_fps:.3f}" if job.source_fps else "待检测", job.status,
                          str(len(job.boundaries)), job.error or job.progress or "; ".join(job.warnings)]
                for column, value in enumerate(values):
                    cell = QTableWidgetItem(value)
                    cell.setData(Qt.UserRole, job.identifier)
                    cell.setFlags(cell.flags() & ~Qt.ItemIsEditable)
                    self.table.setItem(row, column, cell)
                if selected is job:
                    self.table.selectRow(row)
            self.table.resizeColumnsToContents()
        finally:
            self.table.blockSignals(False)
            self.table.setUpdatesEnabled(True)

    def current_job(self) -> VideoJob | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return self.jobs[rows[0].row()] if rows else None

    def update_job(self, changed: VideoJob) -> None:
        self.refresh_table()
        if self.current_job() is changed:
            self.load_selected_job()

    def start_thread(self, worker: QObject) -> None:
        if self.worker_thread is not None:
            QMessageBox.information(self, "任务进行中", "请等待当前任务完成。")
            return
        self.start_button.setEnabled(False)
        self.export_current_button.setEnabled(False)
        self.export_all_button.setEnabled(False)
        thread = QThread(self)
        self.worker_thread = thread
        self.worker = worker  # Keep Python ownership for the whole QThread lifetime.
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.jobUpdated.connect(self.update_job)
        worker.completed.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self.worker_finished)
        thread.start()

    def worker_finished(self) -> None:
        LOGGER.info("Background worker finished")
        self.worker_thread = None
        self.worker = None
        self.start_button.setEnabled(True)
        self.export_current_button.setEnabled(True)
        self.export_all_button.setEnabled(True)
        self.refresh_table()

    def start_inference(self) -> None:
        if not self.jobs:
            QMessageBox.information(self, "请先选择任务", "请选择日期-任务文件夹，再载入需要切分的序号范围。")
            return
        LOGGER.info("Inference requested for %d displayed jobs", len(self.jobs))
        self.start_thread(QueueWorker(self.jobs, self.settings()))

    def start_export(self, all_jobs: bool) -> None:
        current = self.current_job()
        jobs = self.jobs if all_jobs else ([current] if current else [])
        jobs = [job for job in jobs if job is not None and job.status in {"等待复核", "已导出"}]
        if not jobs:
            QMessageBox.information(self, "暂无结果", "请先完成推理，然后选择一个可导出的任务。")
            return
        self.start_thread(ExportWorker(jobs, self.settings()))

    def load_selected_job(self) -> None:
        job = self.current_job()
        if not job or not job.probabilities_path or not job.work_video or not job.work_fps:
            return
        try:
            import numpy as np
            with np.load(job.probabilities_path, allow_pickle=False) as payload:
                probabilities = payload["probabilities"]
            self.timeline.set_data(probabilities, job.work_fps, job.boundaries)
            self.ensure_player().setSource(QUrl.fromLocalFile(str(job.work_video)))
            self.preview_label.setText("橙色：模型裁剪点；绿色端点与红色阴影：模型删除区间；红色：人工修改。\n"
                                       "左键空白处新增普通切分点，拖动可调整；右键删除普通点或整个删除区间。")
        except Exception as error:
            self.preview_label.setText(f"无法加载预览：{error}")

    def boundaries_changed(self, boundaries: list[Boundary]) -> None:
        job = self.current_job()
        if not job:
            return
        job.boundaries = boundaries
        InferenceService().persist(job)
        self.refresh_table()

    def ensure_player(self) -> QMediaPlayer:
        """Create a video-only preview backend on first use, not at startup."""
        if self.player is None:
            self.player = QMediaPlayer(self)
            self.player.setVideoOutput(self.video_widget)
        return self.player

    def seek_preview(self, seconds: float) -> None:
        if self.player is not None:
            self.player.setPosition(round(seconds * 1000))


def main() -> int:
    LOGGER.info("Creating Qt application")
    app = QApplication(sys.argv)
    app.setApplicationName("data_clip 视频切分")
    window = MainWindow()
    window.show()
    LOGGER.info("Main window shown")
    exit_code = app.exec()
    LOGGER.info("Qt event loop exited with code %s", exit_code)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
