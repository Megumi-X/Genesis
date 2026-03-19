from __future__ import annotations

import base64
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory


class VideoSamplingError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class SampledFrame:
    index: int
    data_url: str


def _encode_jpeg_data_url(raw: bytes) -> str:
    return f"data:image/jpeg;base64,{base64.b64encode(raw).decode('ascii')}"


def _run_cmd(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise VideoSamplingError(f"Command not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr if stderr else stdout
        raise VideoSamplingError(f"Command failed: {' '.join(cmd)}\n{detail}") from exc
    return (proc.stdout or "").strip()


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _probe_video_duration_ffprobe(video_path: Path) -> float:
    out = _run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ]
    )
    try:
        parsed = json.loads(out)
        duration = float(parsed["format"]["duration"])
    except Exception as exc:  # noqa: BLE001
        raise VideoSamplingError(f"Unable to parse duration from ffprobe output: {out[:500]}") from exc
    if duration <= 0:
        raise VideoSamplingError(f"Invalid video duration: {duration}")
    return duration


def _sample_frames_ffmpeg(
    video_path: Path,
    *,
    sample_every_sec: float,
    max_frames: int,
    max_width: int,
) -> list[SampledFrame]:
    fps = 1.0 / sample_every_sec

    with TemporaryDirectory(prefix="rigid_critic_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        frame_pattern = tmp_path / "frame_%03d.jpg"
        _run_cmd(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(video_path),
                "-vf",
                f"fps={fps},scale='min({max_width},iw)':-2",
                "-frames:v",
                str(max_frames),
                "-q:v",
                "3",
                str(frame_pattern),
            ]
        )

        frame_paths = sorted(tmp_path.glob("frame_*.jpg"))
        if not frame_paths:
            raise VideoSamplingError("No frames were sampled from the input video.")

        sampled: list[SampledFrame] = []
        for index, frame_path in enumerate(frame_paths):
            sampled.append(SampledFrame(index=index, data_url=_encode_jpeg_data_url(frame_path.read_bytes())))
        return sampled


def _build_interval_indices(total: int, fps: float, sample_every_sec: float, max_frames: int) -> list[int]:
    if total <= 0 or fps <= 0 or sample_every_sec <= 0 or max_frames <= 0:
        return []
    step_frames = max(1, int(round(sample_every_sec * fps)))
    indices = list(range(0, total, step_frames))
    if not indices:
        indices = [0]
    if len(indices) > max_frames:
        indices = indices[:max_frames]
    return indices


def _sample_frames_cv2(
    video_path: Path,
    *,
    sample_every_sec: float,
    max_frames: int,
) -> list[SampledFrame]:
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        raise VideoSamplingError("OpenCV is not available for video fallback sampling.") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoSamplingError(f"OpenCV failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    indices = _build_interval_indices(frame_count, fps, sample_every_sec, max_frames)
    if not indices:
        cap.release()
        raise VideoSamplingError(f"Unable to get valid frame indices from video: {video_path}")

    sampled: list[SampledFrame] = []
    try:
        for out_index, frame_index in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            ok_jpg, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok_jpg:
                continue
            sampled.append(SampledFrame(index=out_index, data_url=_encode_jpeg_data_url(encoded.tobytes())))
    finally:
        cap.release()

    if not sampled:
        raise VideoSamplingError("OpenCV fallback failed to sample any frames.")
    return sampled


def probe_video_duration_sec(video_path: Path) -> float:
    if _ffmpeg_available():
        return _probe_video_duration_ffprobe(video_path)
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        raise VideoSamplingError(
            "Cannot probe video duration: neither ffprobe nor OpenCV is available."
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoSamplingError(f"OpenCV failed to open video: {video_path}")
    try:
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if frame_count <= 0 or fps <= 0:
            raise VideoSamplingError(f"Invalid OpenCV metadata for `{video_path}`: frames={frame_count}, fps={fps}.")
        return frame_count / fps
    finally:
        cap.release()


def sample_video_frames_as_data_urls(
    video_path: Path,
    *,
    sample_every_sec: float = 0.5,
    max_frames: int = 24,
    max_width: int = 640,
) -> list[SampledFrame]:
    if not video_path.exists():
        raise VideoSamplingError(f"Video file not found: {video_path}")
    if sample_every_sec <= 0:
        raise VideoSamplingError("`sample_every_sec` must be positive.")
    if max_frames <= 0:
        raise VideoSamplingError("`max_frames` must be positive.")
    if max_width <= 0:
        raise VideoSamplingError("`max_width` must be positive.")
    if _ffmpeg_available():
        return _sample_frames_ffmpeg(
            video_path,
            sample_every_sec=sample_every_sec,
            max_frames=max_frames,
            max_width=max_width,
        )
    return _sample_frames_cv2(video_path, sample_every_sec=sample_every_sec, max_frames=max_frames)
