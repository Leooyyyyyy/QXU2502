import argparse
import queue
import statistics
import sys
import threading
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_inference import Phase12InferenceEngine
from core.runtime_inference import RuntimeConfig


PANEL_WIDTH = 360
WINDOW_NAME = "Yoga Posture Demo"


def _to_title_text(value: str | None, default: str = "None") -> str:
    if not value:
        return default
    text = value.replace("_", " ").title()
    roman_tokens = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}
    return " ".join(word.upper() if word.lower() in roman_tokens else word for word in text.split())


def _get_tracking_status(result) -> str:
    status = result.get("status")
    if status == "ok":
        return "Tracking"
    if status == "low_confidence":
        return "Stabilizing"
    if status == "out_of_frame":
        return "Adjust position"
    if status == "no_person":
        return "No person detected"
    return "Tracking"


def _get_helper_message(result) -> str:
    status = result.get("status")
    if status == "out_of_frame":
        return "Move fully into frame"
    if status == "no_person":
        return "Step into view"
    if status == "low_confidence":
        return "Hold still for a clearer reading"
    if status == "ok":
        return result.get("status_message", "Prediction stable")
    return result.get("status_message", "")


def _format_elapsed_mmss(elapsed_seconds):
    total_seconds = max(int(elapsed_seconds), 0)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def _get_display_status_text_and_color(result):
    status = result.get("status")
    if status != "ok":
        if status == "low_confidence":
            return "Stabilizing", (0, 215, 255)
        if status == "out_of_frame":
            return "Adjust Position", (190, 190, 190)
        if status == "no_person":
            return "No Person", (190, 190, 190)
        return "Waiting", (190, 190, 190)

    # Prefer stabilized/hierarchical status when available.
    final_feedback = result.get("stable_feedback") or result.get("pred_feedback")
    if final_feedback == "Correct":
        return "Correct", (70, 200, 70)
    if final_feedback == "Incorrect":
        return "Incorrect", (60, 60, 220)
    return str(final_feedback or "-"), (190, 190, 190)


def _wrap_text(text: str, font, scale: float, thickness: int, max_width: int) -> list[str]:
    if not text:
        return []
    words = text.split()
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        width = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the webcam demo with the frozen Phase 1.2 model.")
    parser.add_argument("--checkpoint", default="artifacts/experiments/phase1.2_baseline_refined/best_checkpoint.pth")
    parser.add_argument("--config", default="artifacts/experiments/phase1.2_baseline_refined/config.json")
    parser.add_argument("--feature-mode", default="landmarks_v1")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--posture-margin-threshold", type=float, default=0.08)
    parser.add_argument("--correctness-threshold", type=float, default=0.50)
    parser.add_argument("--stabilization-window-size", type=int, default=5)
    parser.add_argument("--stabilization-min-votes", type=int, default=3)
    parser.add_argument("--runtime-mode", choices=("baseline", "optimized"), default="baseline")
    parser.add_argument("--posture-window-size", type=int, default=4)
    parser.add_argument("--posture-min-votes", type=int, default=2)
    parser.add_argument("--correctness-window-size", type=int, default=5)
    parser.add_argument("--correctness-min-votes", type=int, default=3)
    parser.add_argument("--subtype-window-size", type=int, default=7)
    parser.add_argument("--subtype-min-votes", type=int, default=4)
    parser.add_argument("--subtype-confidence-threshold", type=float, default=0.60)
    parser.add_argument("--subtype-margin-threshold", type=float, default=0.10)
    parser.add_argument("--save-video", type=str, default=None)
    parser.add_argument("--display-smoothing", action="store_true")
    parser.add_argument("--display-smoothing-alpha", type=float, default=0.18)
    return parser


def draw_demo_ui(frame, result, fps, elapsed_seconds):
    height, width = frame.shape[:2]
    panel_x0 = max(width - PANEL_WIDTH, 0)
    tracking_status = _get_tracking_status(result)
    status_text, status_color = _get_display_status_text_and_color(result)
    correction_focus = _to_title_text(result.get("display_negative_subtype"))
    show_correction_focus = status_text == "Incorrect"
    helper_lines = _wrap_text(
        _get_helper_message(result),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        2,
        PANEL_WIDTH - 48,
    )
    helper_y0 = 138
    helper_line_gap = 24
    block_shift = max(0, len(helper_lines) - 1) * helper_line_gap

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x0, 0), (width, height), (248, 248, 248), thickness=-1)
    frame[:] = cv2.addWeighted(overlay, 0.86, frame, 0.14, 0)
    cv2.line(frame, (panel_x0, 0), (panel_x0, height), (170, 170, 170), 2)

    # Session timer in top-left of the live video area.
    timer_text = f"Timer: {_format_elapsed_mmss(elapsed_seconds)}"
    cv2.rectangle(frame, (16, 16), (220, 58), (40, 40, 40), thickness=-1)
    cv2.putText(frame, timer_text, (24, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)

    lines = [
        ("Yoga Posture Demo", (40, 40, 40), 0.95, 36),
        ("Tracking status", (120, 120, 120), 0.55, 78),
        (tracking_status, (40, 40, 40), 0.78, 113),
        ("Detected posture", (120, 120, 120), 0.55, 160 + block_shift),
        (_to_title_text(result.get("pred_posture"), default="-"), (25, 25, 25), 0.9, 195 + block_shift),
        ("Posture status", (120, 120, 120), 0.55, 245 + block_shift),
        (status_text, status_color, 0.82, 280 + block_shift),
        ("Correction focus", (120, 120, 120), 0.55, 330 + block_shift),
        ((correction_focus if show_correction_focus else "None"), (25, 25, 25), 0.76, 365 + block_shift),
        ("Posture confidence", (120, 120, 120), 0.55, 415 + block_shift),
        (f"{result.get('posture_confidence', 0.0):.2f}", (25, 25, 25), 0.82, 450 + block_shift),
        ("Stability", (120, 120, 120), 0.55, 500 + block_shift),
        (f"{result.get('stability', 0.0):.2f}", (25, 25, 25), 0.82, 535 + block_shift),
        ("FPS", (120, 120, 120), 0.55, 585 + block_shift),
        (f"{fps:.1f}", (25, 25, 25), 0.82, 620 + block_shift),
        ("Frame age (ms)", (120, 120, 120), 0.55, 670 + block_shift),
        (f"{result.get('frame_age_ms', 0.0):.1f}", (25, 25, 25), 0.82, 705 + block_shift),
    ]

    x = panel_x0 + 24
    for text, color, scale, y in lines:
        cv2.putText(frame, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    for idx, helper_line in enumerate(helper_lines):
        y = helper_y0 + idx * helper_line_gap
        cv2.putText(frame, helper_line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (110, 110, 110), 2, cv2.LINE_AA)

    return frame


def _format_metric(value, precision: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{precision}f}"


def _compute_percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


class RuntimeMetrics:
    def __init__(self, runtime_mode: str):
        self.runtime_mode = runtime_mode
        self.start_time = time.time()
        self.end_time: float | None = None
        self.fps_samples: list[float] = []
        self.frame_age_samples_ms: list[float] = []
        self.inference_samples_ms: list[float] = []
        self.processed_frames = 0
        self.dropped_frames = 0
        self.correctness_switch_count = 0
        self.subtype_visible_count = 0
        self.subtype_switch_count = 0
        self.flip_samples_ms: list[float] = []
        self.previous_feedback: str | None = None
        self.previous_visible_subtype: str | None = None

    def record_frame(self, result: dict, fps: float, frame_age_ms: float, inference_time_ms: float) -> None:
        self.processed_frames += 1
        self.fps_samples.append(fps)
        self.frame_age_samples_ms.append(frame_age_ms)
        self.inference_samples_ms.append(inference_time_ms)

        current_feedback = result.get("pred_feedback")
        if self.previous_feedback is not None and current_feedback != self.previous_feedback:
            self.correctness_switch_count += 1
        self.previous_feedback = current_feedback

        current_visible_subtype = result.get("display_negative_subtype")
        if current_visible_subtype is not None:
            self.subtype_visible_count += 1
        if self.previous_visible_subtype is not None and current_visible_subtype != self.previous_visible_subtype:
            self.subtype_switch_count += 1
        self.previous_visible_subtype = current_visible_subtype

    def finish(self) -> None:
        if self.end_time is None:
            self.end_time = time.time()

    def summary(self) -> list[tuple[str, str]]:
        self.finish()
        total_runtime_s = max((self.end_time or self.start_time) - self.start_time, 0.0)
        dropped_ratio = (self.dropped_frames / (self.processed_frames + self.dropped_frames)) if (self.processed_frames + self.dropped_frames) else 0.0
        subtype_visible_ratio = (self.subtype_visible_count / self.processed_frames) if self.processed_frames else 0.0

        return [
            ("runtime_mode", self.runtime_mode),
            ("average_fps", _format_metric(statistics.fmean(self.fps_samples) if self.fps_samples else None)),
            ("median_fps", _format_metric(statistics.median(self.fps_samples) if self.fps_samples else None)),
            ("average_frame_age_ms", _format_metric(statistics.fmean(self.frame_age_samples_ms) if self.frame_age_samples_ms else None)),
            ("median_frame_age_ms", _format_metric(statistics.median(self.frame_age_samples_ms) if self.frame_age_samples_ms else None)),
            ("p95_frame_age_ms", _format_metric(_compute_percentile(self.frame_age_samples_ms, 0.95))),
            ("max_frame_age_ms", _format_metric(max(self.frame_age_samples_ms) if self.frame_age_samples_ms else None)),
            ("total_processed_frames", _format_metric(self.processed_frames)),
            ("total_runtime_s", _format_metric(total_runtime_s)),
            ("average_inference_time_ms", _format_metric(statistics.fmean(self.inference_samples_ms) if self.inference_samples_ms else None)),
            ("dropped_frame_count", _format_metric(self.dropped_frames)),
            ("dropped_frame_ratio", _format_metric(dropped_ratio)),
            ("correctness_switch_count", _format_metric(self.correctness_switch_count)),
            ("subtype_visible_ratio", _format_metric(subtype_visible_ratio)),
            ("subtype_switch_count", _format_metric(self.subtype_switch_count)),
        ]

    def print_summary(self) -> None:
        print("\n=== Webcam Runtime Summary ===")
        for key, value in self.summary():
            print(f"{key}: {value}")
        flip_average_ms = statistics.fmean(self.flip_samples_ms) if self.flip_samples_ms else None
        print(f"average_display_flip_ms: {_format_metric(flip_average_ms, precision=4)}")


class DemoVideoRecorder:
    def __init__(self, output_path: str | None):
        self.output_path = output_path
        self.writer: cv2.VideoWriter | None = None

    def write(self, frame, fps: float) -> None:
        if not self.output_path:
            return
        if self.writer is None:
            height, width = frame.shape[:2]
            output_fps = max(fps, 1.0)
            output_dir = Path(self.output_path).expanduser().resolve().parent
            output_dir.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(Path(self.output_path).expanduser().resolve()), fourcc, output_fps, (width, height))
            if not self.writer.isOpened():
                self.writer = None
                raise RuntimeError(f"Unable to open video writer for {self.output_path}")
        self.writer.write(frame)

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class DisplaySmoother:
    def __init__(self, enabled: bool = False, alpha: float = 0.18):
        self.enabled = enabled
        self.alpha = max(0.0, min(alpha, 0.5))
        self.previous_frame = None

    def apply(self, frame):
        current_frame = frame.copy()
        if not self.enabled:
            self.previous_frame = current_frame
            return current_frame

        if self.previous_frame is None or self.previous_frame.shape != current_frame.shape:
            self.previous_frame = current_frame
            return current_frame

        smoothed = cv2.addWeighted(current_frame, 1.0 - self.alpha, self.previous_frame, self.alpha, 0.0)
        self.previous_frame = current_frame
        return smoothed


class LatestFrameBuffer:
    def __init__(self):
        self.queue: queue.Queue[tuple[int, float, object]] = queue.Queue(maxsize=1)
        self.frame_id = 0
        self.stopped = threading.Event()
        self.dropped_frames = 0

    def push(self, frame) -> None:
        payload = (self.frame_id, time.time(), frame)
        self.frame_id += 1
        if self.queue.full():
            try:
                self.queue.get_nowait()
                self.dropped_frames += 1
            except queue.Empty:
                pass
        self.queue.put_nowait(payload)

    def get(self, timeout: float = 0.25):
        while not self.stopped.is_set():
            try:
                return self.queue.get(timeout=timeout)
            except queue.Empty:
                continue
        return None

    def stop(self) -> None:
        self.stopped.set()


def run_baseline_loop(
    capture,
    engine,
    metrics: RuntimeMetrics,
    recorder: DemoVideoRecorder,
    display_smoother: DisplaySmoother,
):
    last_time = time.time()

    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break

        frame_timestamp = time.time()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inference_start = time.time()
        result = engine.predict_rgb_frame(frame_rgb, use_stabilization=True)
        inference_time_ms = max((time.time() - inference_start) * 1000.0, 0.0)
        result["frame_age_ms"] = max((time.time() - frame_timestamp) * 1000.0, 0.0)

        now = time.time()
        fps = 1.0 / max(now - last_time, 1e-6)
        last_time = now
        metrics.record_frame(result, fps, result["frame_age_ms"], inference_time_ms)

        flip_start = time.time()
        display_frame = cv2.flip(frame_bgr, 1)
        metrics.flip_samples_ms.append(max((time.time() - flip_start) * 1000.0, 0.0))
        display_frame = display_smoother.apply(display_frame)
        elapsed_seconds = time.time() - metrics.start_time
        demo_frame = draw_demo_ui(display_frame, result, fps, elapsed_seconds)
        recorder.write(demo_frame, fps)
        cv2.imshow(WINDOW_NAME, demo_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break


def run_optimized_loop(
    capture,
    engine,
    metrics: RuntimeMetrics,
    recorder: DemoVideoRecorder,
    display_smoother: DisplaySmoother,
):
    frame_buffer = LatestFrameBuffer()

    def capture_worker():
        while not frame_buffer.stopped.is_set():
            ok, frame_bgr = capture.read()
            if not ok:
                frame_buffer.stop()
                break
            frame_buffer.push(frame_bgr)

    worker = threading.Thread(target=capture_worker, daemon=True)
    worker.start()
    last_time = time.time()

    try:
        while True:
            payload = frame_buffer.get()
            if payload is None:
                break

            _, frame_timestamp, frame_bgr = payload
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            inference_start = time.time()
            result = engine.predict_rgb_frame(frame_rgb, use_hierarchical_stabilization=True)
            inference_time_ms = max((time.time() - inference_start) * 1000.0, 0.0)
            result["frame_age_ms"] = max((time.time() - frame_timestamp) * 1000.0, 0.0)

            now = time.time()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            metrics.record_frame(result, fps, result["frame_age_ms"], inference_time_ms)
            metrics.dropped_frames = frame_buffer.dropped_frames

            flip_start = time.time()
            display_frame = cv2.flip(frame_bgr, 1)
            metrics.flip_samples_ms.append(max((time.time() - flip_start) * 1000.0, 0.0))
            display_frame = display_smoother.apply(display_frame)
            elapsed_seconds = time.time() - metrics.start_time
            demo_frame = draw_demo_ui(display_frame, result, fps, elapsed_seconds)
            recorder.write(demo_frame, fps)
            cv2.imshow(WINDOW_NAME, demo_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        frame_buffer.stop()
        worker.join(timeout=1.0)


def main() -> None:
    args = build_parser().parse_args()
    runtime_config = RuntimeConfig(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        feature_mode=args.feature_mode,
        camera_index=args.camera_index,
        confidence_threshold=args.confidence_threshold,
        posture_margin_threshold=args.posture_margin_threshold,
        correctness_threshold=args.correctness_threshold,
        stabilization_window_size=args.stabilization_window_size,
        stabilization_min_votes=args.stabilization_min_votes,
        posture_window_size=args.posture_window_size,
        posture_min_votes=args.posture_min_votes,
        correctness_window_size=args.correctness_window_size,
        correctness_min_votes=args.correctness_min_votes,
        subtype_window_size=args.subtype_window_size,
        subtype_min_votes=args.subtype_min_votes,
        subtype_confidence_threshold=args.subtype_confidence_threshold,
        subtype_margin_threshold=args.subtype_margin_threshold,
        static_image_mode=False,
    )

    capture = cv2.VideoCapture(runtime_config.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open camera index {runtime_config.camera_index}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    engine = Phase12InferenceEngine(runtime_config)
    metrics = RuntimeMetrics(args.runtime_mode)
    recorder = DemoVideoRecorder(args.save_video)
    display_smoother = DisplaySmoother(
        enabled=args.display_smoothing,
        alpha=args.display_smoothing_alpha,
    )

    try:
        try:
            if args.runtime_mode == "baseline":
                run_baseline_loop(capture, engine, metrics, recorder, display_smoother)
            else:
                run_optimized_loop(capture, engine, metrics, recorder, display_smoother)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received. Closing webcam loop.")
    finally:
        metrics.print_summary()
        recorder.release()
        engine.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
