#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import subprocess
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cache
from fractions import Fraction
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pyarrow as pa
import torch
import torchvision
from datasets.features.features import register_feature
from PIL import Image


def _finalize_loaded_video_frames(
    loaded_frames: list[torch.Tensor],
    loaded_ts: list[float],
    timestamps: list[float],
    tolerance_s: float,
    video_path: str,
    backend: str,
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    if len(loaded_frames) == 0 or len(loaded_ts) == 0:
        raise RuntimeError(f"No frames could be decoded from video: {video_path}")

    query_ts = torch.tensor(timestamps)
    loaded_ts_tensor = torch.tensor(loaded_ts)

    dist = torch.cdist(query_ts[:, None], loaded_ts_tensor[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts_tensor}"
        f"\nvideo: {video_path}"
        f"\nbackend: {backend}"
    )

    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts_tensor[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    closest_frames = closest_frames.type(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames


def _decode_video_frames_with_av(
    video_path: str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str,
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    import av

    first_ts = min(timestamps)
    last_ts = max(timestamps)

    container = av.open(video_path)
    stream = container.streams.video[0]

    # Seek slightly earlier than the first requested timestamp to increase the odds of landing
    # on a preceding keyframe while still avoiding a full decode from t=0 for every query.
    seek_ts = max(first_ts - 1.0, 0.0)
    try:
        if stream.time_base is not None:
            seek_pts = int(seek_ts / float(stream.time_base))
            container.seek(seek_pts, stream=stream, any_frame=False, backward=True)
    except av.AVError:
        container.seek(0)

    loaded_frames: list[torch.Tensor] = []
    loaded_ts: list[float] = []

    try:
        for frame in container.decode(video=0):
            if frame.pts is not None and stream.time_base is not None:
                current_ts = float(frame.pts * stream.time_base)
            elif frame.time is not None:
                current_ts = float(frame.time)
            else:
                continue

            if current_ts + tolerance_s < first_ts:
                continue

            if log_loaded_timestamps:
                logging.info(f"frame loaded at timestamp={current_ts:.4f}")

            loaded_frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
            loaded_ts.append(current_ts)

            if current_ts >= last_ts:
                break
    finally:
        container.close()

    return _finalize_loaded_video_frames(
        loaded_frames,
        loaded_ts,
        timestamps,
        tolerance_s,
        video_path,
        backend=backend,
        log_loaded_timestamps=log_loaded_timestamps,
    )


def decode_video_frames_torchvision(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video

    The backend can be either "pyav" (default) or "video_reader".
    "video_reader" requires installing torchvision from source, see:
    https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
    (note that you need to compile against ffmpeg<4.3)

    While both use cpu, "video_reader" is supposedly faster than "pyav" but requires additional setup.
    For more info on video decoding, see `benchmark/video/README.md`

    See torchvision doc for more info on these two backends:
    https://pytorch.org/vision/0.18/index.html?highlight=backend#torchvision.set_video_backend

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    video_path = str(video_path)

    if not hasattr(torchvision.io, "VideoReader"):
        return _decode_video_frames_with_av(
            video_path,
            timestamps,
            tolerance_s,
            backend=f"{backend}+av_fallback",
            log_loaded_timestamps=log_loaded_timestamps,
        )

    # set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesnt support accuracte seek

    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    reader = torchvision.io.VideoReader(video_path, "video")

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    # access closest key frame of the first requested frame
    # Note: closest key frame timestamp is usually smaller than `first_ts` (e.g. key frame can be the first frame of the video)
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    reader = None

    return _finalize_loaded_video_frames(
        loaded_frames,
        loaded_ts,
        timestamps,
        tolerance_s,
        video_path,
        backend=backend,
        log_loaded_timestamps=log_loaded_timestamps,
    )


@cache
def _ffmpeg_encoder_names() -> set[str]:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    encoder_names = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoder_names.add(parts[1])
    return encoder_names


def encode_video_frames(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: str = "libopenh264",
    pix_fmt: str = "yuv420p",
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: str | None = "error",
    overwrite: bool = False,
    frame_timestamps: np.ndarray | list[float] | None = None,
    frame_durations_s: np.ndarray | list[float] | None = None,
) -> None:
    """More info on ffmpeg arguments tuning on `benchmark/video/README.md`"""
    if frame_timestamps is not None:
        _encode_video_frames_real_time(
            imgs_dir=imgs_dir,
            video_path=video_path,
            fps=fps,
            frame_timestamps=frame_timestamps,
            frame_durations_s=frame_durations_s,
            vcodec=vcodec,
            pix_fmt=pix_fmt,
            g=g,
            crf=crf,
            fast_decode=fast_decode,
        )
        return

    _encode_video_frames_fixed_fps(
        imgs_dir=imgs_dir,
        video_path=video_path,
        fps=fps,
        vcodec=vcodec,
        pix_fmt=pix_fmt,
        g=g,
        crf=crf,
        fast_decode=fast_decode,
        log_level=log_level,
        overwrite=overwrite,
    )


def _encode_video_frames_fixed_fps(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: str,
    pix_fmt: str,
    g: int | None,
    crf: int | None,
    fast_decode: int,
    log_level: str | None,
    overwrite: bool,
) -> None:
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    selected_vcodec = vcodec
    if vcodec == "libopenh264" and vcodec not in _ffmpeg_encoder_names():
        logging.warning("Falling back to libx264 because libopenh264 is unavailable in ffmpeg.")
        selected_vcodec = "libx264"

    ffmpeg_args = OrderedDict(
        [
            ("-f", "image2"),
            ("-r", str(fps)),
            ("-i", str(imgs_dir / "frame_%06d.png")),
            ("-vcodec", selected_vcodec),
            ("-pix_fmt", pix_fmt),
        ]
    )

    if g is not None:
        ffmpeg_args["-g"] = str(g)

    if crf is not None:
        ffmpeg_args["-crf"] = str(crf)

    if fast_decode:
        key = "-svtav1-params" if vcodec == "libsvtav1" else "-tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        ffmpeg_args[key] = value

    if log_level is not None:
        ffmpeg_args["-loglevel"] = str(log_level)

    ffmpeg_args = [item for pair in ffmpeg_args.items() for item in pair]
    if overwrite:
        ffmpeg_args.append("-y")

    ffmpeg_cmd = ["ffmpeg"] + ffmpeg_args + [str(video_path)]
    # redirect stdin to subprocess.DEVNULL to prevent reading random keyboard inputs from terminal
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdin=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        if selected_vcodec != "libopenh264":
            raise
        fallback_cmd = list(ffmpeg_cmd)
        vcodec_index = fallback_cmd.index("libopenh264")
        fallback_cmd[vcodec_index] = "libx264"
        logging.warning("Falling back to libx264 because libopenh264 is unavailable in ffmpeg.")
        subprocess.run(fallback_cmd, check=True, stdin=subprocess.DEVNULL)

    if not video_path.exists():
        raise OSError(
            f"Video encoding did not work. File not found: {video_path}. "
            f"Try running the command manually to debug: `{''.join(ffmpeg_cmd)}`"
        )


def _encode_video_frames_real_time(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    frame_timestamps: np.ndarray | list[float],
    frame_durations_s: np.ndarray | list[float] | None,
    vcodec: str,
    pix_fmt: str,
    g: int | None,
    crf: int | None,
    fast_decode: int,
) -> None:
    import av

    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    timestamps_s = np.asarray(frame_timestamps, dtype=np.float64).reshape(-1)
    if timestamps_s.ndim != 1 or len(timestamps_s) == 0:
        raise ValueError("frame_timestamps must contain at least one timestamp.")

    if len(timestamps_s) > 1 and np.any(np.diff(timestamps_s) <= 0.0):
        raise ValueError("frame_timestamps must be strictly increasing for real-time video encoding.")

    if frame_durations_s is not None:
        durations_s = np.asarray(frame_durations_s, dtype=np.float64).reshape(-1)
        if durations_s.shape != timestamps_s.shape:
            raise ValueError("frame_durations_s must have the same length as frame_timestamps.")
        if np.any(durations_s <= 0.0):
            raise ValueError("frame_durations_s must be strictly positive.")
    else:
        durations_s = None

    frame_paths = [Path(imgs_dir) / f"frame_{idx:06d}.png" for idx in range(len(timestamps_s))]
    missing_frames = [str(path) for path in frame_paths if not path.is_file()]
    if missing_frames:
        raise FileNotFoundError(f"Missing image frames for video encoding: {missing_frames[:3]}")

    relative_timestamps_s = timestamps_s - timestamps_s[0]
    pts_us = np.round(relative_timestamps_s * 1_000_000).astype(np.int64)
    time_base = Fraction(1, 1_000_000)
    nominal_rate = max(int(fps), 1000)

    with Image.open(frame_paths[0]) as first_image:
        first_rgb = np.asarray(first_image.convert("RGB"))
    height, width = first_rgb.shape[:2]

    codec_candidates = [vcodec]
    if vcodec == "libopenh264":
        codec_candidates.append("libx264")

    container = None
    last_error = None
    stream = None
    selected_codec = None
    for candidate in codec_candidates:
        try:
            container = av.open(str(video_path), mode="w")
            stream = container.add_stream(candidate, rate=nominal_rate)
            selected_codec = candidate
            break
        except Exception as exc:
            last_error = exc
            if container is not None:
                container.close()
                container = None
    if container is None or stream is None or selected_codec is None:
        raise RuntimeError(f"Unable to initialize video encoder {vcodec!r}: {last_error}") from last_error

    if selected_codec != vcodec:
        logging.warning("Falling back to libx264 because libopenh264 is unavailable in PyAV.")

    stream.width = width
    stream.height = height
    stream.pix_fmt = pix_fmt
    stream.time_base = time_base
    stream.codec_context.time_base = time_base
    if g is not None:
        stream.codec_context.gop_size = g

    stream_options = {"bf": "0", "force-cfr": "0"}
    if crf is not None:
        stream_options["crf"] = str(crf)
    if fast_decode:
        stream_options["tune"] = "fastdecode"
    stream.options = stream_options

    packets = []
    try:
        for frame_index, frame_path in enumerate(frame_paths):
            if frame_index == 0:
                rgb_frame = first_rgb
            else:
                with Image.open(frame_path) as image:
                    rgb_frame = np.asarray(image.convert("RGB"))
            frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")
            if pix_fmt != "rgb24":
                frame = frame.reformat(format=pix_fmt)
            frame.pts = int(pts_us[frame_index])
            frame.time_base = time_base
            for packet in stream.encode(frame):
                packets.append(packet)

        for packet in stream.encode():
            packets.append(packet)

        if durations_s is not None and packets:
            packets[-1].duration = int(round(float(durations_s[-1]) * 1_000_000))

        for packet in packets:
            container.mux(packet)
    finally:
        container.close()

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")


@dataclass
class VideoFrame:
    # TODO(rcadene, lhoestq): move to Hugging Face `datasets` repo
    """
    Provides a type for a dataset containing video frames.

    Example:

    ```python
    data_dict = [{"image": {"path": "videos/episode_0.mp4", "timestamp": 0.3}}]
    features = {"image": VideoFrame()}
    Dataset.from_dict(data_dict, features=Features(features))
    ```
    """

    pa_type: ClassVar[Any] = pa.struct({"path": pa.string(), "timestamp": pa.float32()})
    _type: str = field(default="VideoFrame", init=False, repr=False)

    def __call__(self):
        return self.pa_type


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental and might be subject to breaking changes in the future.",
        category=UserWarning,
    )
    # to make VideoFrame available in HuggingFace `datasets`
    register_feature(VideoFrame, "VideoFrame")


def get_audio_info(video_path: Path | str) -> dict:
    ffprobe_audio_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels,codec_name,bit_rate,sample_rate,bit_depth,channel_layout,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(ffprobe_audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running ffprobe: {result.stderr}")

    info = json.loads(result.stdout)
    audio_stream_info = info["streams"][0] if info.get("streams") else None
    if audio_stream_info is None:
        return {"has_audio": False}

    # Return the information, defaulting to None if no audio stream is present
    return {
        "has_audio": True,
        "audio.channels": audio_stream_info.get("channels", None),
        "audio.codec": audio_stream_info.get("codec_name", None),
        "audio.bit_rate": int(audio_stream_info["bit_rate"]) if audio_stream_info.get("bit_rate") else None,
        "audio.sample_rate": int(audio_stream_info["sample_rate"])
        if audio_stream_info.get("sample_rate")
        else None,
        "audio.bit_depth": audio_stream_info.get("bit_depth", None),
        "audio.channel_layout": audio_stream_info.get("channel_layout", None),
    }


def get_video_info(video_path: Path | str) -> dict:
    ffprobe_video_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,width,height,codec_name,nb_frames,duration,pix_fmt,time_base",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(ffprobe_video_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running ffprobe: {result.stderr}")

    info = json.loads(result.stdout)
    video_stream_info = info["streams"][0]

    def parse_fraction(value: str | None) -> float | None:
        if not value or value == "0/0":
            return None
        num, denom = map(int, value.split("/"))
        if denom == 0:
            return None
        return num / denom

    avg_fps = parse_fraction(video_stream_info.get("avg_frame_rate"))
    nominal_fps = parse_fraction(video_stream_info.get("r_frame_rate"))
    duration_s = float(video_stream_info["duration"]) if video_stream_info.get("duration") else None
    nb_frames = int(video_stream_info["nb_frames"]) if video_stream_info.get("nb_frames") else None
    fps = avg_fps or nominal_fps or (
        (nb_frames / duration_s) if nb_frames is not None and duration_s not in (None, 0.0) else None
    )
    if fps is None:
        raise RuntimeError(f"Unable to infer video fps for {video_path}.")

    pixel_channels = get_video_pixel_channels(video_stream_info["pix_fmt"])

    video_info = {
        "video.fps": fps,
        "video.avg_fps": avg_fps,
        "video.nominal_fps": nominal_fps,
        "video.duration_s": duration_s,
        "video.nb_frames": nb_frames,
        "video.time_base": video_stream_info.get("time_base"),
        "video.height": video_stream_info["height"],
        "video.width": video_stream_info["width"],
        "video.channels": pixel_channels,
        "video.codec": video_stream_info["codec_name"],
        "video.pix_fmt": video_stream_info["pix_fmt"],
        "video.is_depth_map": False,
        **get_audio_info(video_path),
    }

    return video_info


def get_video_pixel_channels(pix_fmt: str) -> int:
    if "gray" in pix_fmt or "depth" in pix_fmt or "monochrome" in pix_fmt:
        return 1
    elif "rgba" in pix_fmt or "yuva" in pix_fmt:
        return 4
    elif "rgb" in pix_fmt or "yuv" in pix_fmt:
        return 3
    else:
        raise ValueError("Unknown format")


def get_image_pixel_channels(image: Image):
    if image.mode == "L":
        return 1  # Grayscale
    elif image.mode == "LA":
        return 2  # Grayscale + Alpha
    elif image.mode == "RGB":
        return 3  # RGB
    elif image.mode == "RGBA":
        return 4  # RGBA
    else:
        raise ValueError("Unknown format")
