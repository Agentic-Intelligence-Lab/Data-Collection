#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import queue
import select
import sys
import termios
import threading
import time
import tty
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_SCRIPT = PROJECT_ROOT / "runtime" / "piper_openpi_runtime.py"
DEFAULT_LEROBOT_RUNTIME_ROOT = PROJECT_ROOT / "vendor" / "lerobot_piper"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "outputs" / "datasets"
COLLECTION_COMMANDS = {"intervention", "next_episode", "next_success", "next_failure", "stop", "confirm"}
EPISODE_COMMIT_COMMANDS = {"next_episode", "next_success", "next_failure"}


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_path(name: str, default: Path | None = None) -> Path | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_rotation(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip().lower()
    if value in {"", "none", "null"}:
        return None
    return int(value)


def build_camera_layout(args: argparse.Namespace) -> str:
    def make_camera(serial: int, rotation: int | None) -> dict[str, Any]:
        return {
            "type": "intelrealsense",
            "serial_number": int(serial),
            "fps": int(args.camera_fps),
            "width": int(args.camera_width),
            "height": int(args.camera_height),
            "force_hardware_reset": bool(args.force_camera_reset),
            "rotation": rotation,
        }

    return json.dumps(
        {
            "head": make_camera(args.head_camera_serial, args.head_camera_rotation),
            "left_wrist": make_camera(args.left_wrist_camera_serial, args.left_wrist_camera_rotation),
            "right_wrist": make_camera(args.right_wrist_camera_serial, args.right_wrist_camera_rotation),
            "front_view": make_camera(args.front_view_camera_serial, args.front_view_camera_rotation),
        },
        separators=(",", ":"),
    )


def camera_keys_from_json(robot_cameras_json: str | None) -> list[str]:
    if not robot_cameras_json:
        return ["head", "left_wrist", "right_wrist", "front_view"]
    try:
        raw = json.loads(robot_cameras_json)
    except json.JSONDecodeError:
        return ["head", "left_wrist", "right_wrist", "front_view"]
    return list(raw.keys()) or ["head", "left_wrist", "right_wrist", "front_view"]


def episode_success_from_command(args: argparse.Namespace, command: str) -> str:
    if command == "next_success":
        return "success"
    if command == "next_failure":
        return "failure"
    return args.episode_success


def save_button_label(command: str) -> str:
    if command == "next_success":
        return "Save Success Episode"
    if command == "next_failure":
        return "Save Failure Episode"
    return "Save Episode"


def default_capture_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_DATA_ROOT / f"piper_correction_{stamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an OpenPI single-arm rollout and record an interactive DAgger-style "
            "dataset. Press i to pause policy rollout, manually connect the hardware master, "
            "press Enter to record human correction frames, press i again to pause, disconnect, "
            "and press Enter to resume policy rollout."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--lerobot-runtime-root", type=Path, default=env_path("ROBOT_RUNTIME_ROOT", env_path("PIPER_RUNTIME_ROOT", DEFAULT_LEROBOT_RUNTIME_ROOT)))
    parser.add_argument("--runtime-script", type=Path, default=env_path("POLICY_RUNTIME_SCRIPT", env_path("PIPER_OPENPI_RUNTIME", DEFAULT_RUNTIME_SCRIPT)))
    parser.add_argument("--openpi-root", type=Path, default=env_path("OPENPI_ROOT", PROJECT_ROOT / "external" / "openpi"))
    parser.add_argument("--checkpoint-dir", type=Path, default=env_path("CHECKPOINT_DIR"))
    parser.add_argument("--norm-stats-path", type=Path, default=env_path("NORM_STATS_PATH"))
    parser.add_argument("--can-name", default=os.environ.get("PIPER_CAN_NAME", "can1"))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"), choices=("cuda", "cpu"))
    parser.add_argument("--prompt", default=os.environ.get("PROMPT", "stack the yellow bowl on the green bowl"))
    parser.add_argument("--fps", type=int, default=int(float(os.environ.get("FPS", "10"))))
    parser.add_argument("--actions-per-inference", type=int, default=int(os.environ.get("ACTIONS_PER_INFERENCE", "8")))
    parser.add_argument("--sync-inference", action="store_true", default=env_bool("SYNC_INFERENCE", True))
    parser.add_argument("--prefetch-threshold", type=int, default=int(os.environ.get("PREFETCH_THRESHOLD", "2")))
    parser.add_argument("--action-scale", type=float, default=float(os.environ.get("ACTION_SCALE", "1.0")))
    parser.add_argument("--max-abs-delta", type=float, default=float(os.environ.get("MAX_ABS_DELTA", "0.04")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", "0")), help="Max recorded control frames per episode. 0 means no frame limit.")
    parser.add_argument("--num-episodes", type=int, default=int(os.environ.get("NUM_EPISODES", "1")), help="Number of episodes to save. 0 means run until q/Ctrl+C.")
    parser.add_argument("--enable-arm", action="store_true", default=env_bool("ENABLE_ARM", False))
    parser.add_argument("--dry-run", action="store_true", default=env_bool("DRY_RUN", False))
    parser.add_argument("--validate-only", action="store_true", default=env_bool("VALIDATE_ONLY", False))
    parser.add_argument("--debug-action-details", action="store_true", default=env_bool("DEBUG_ACTION_DETAILS", False))
    parser.add_argument("--print-command-counts", action="store_true", default=env_bool("PRINT_COMMAND_COUNTS", False))
    parser.add_argument("--disable-arm-on-exit", action="store_true", default=env_bool("DISABLE_ARM_ON_EXIT", False))
    parser.add_argument("--disable-settle-seconds", type=float, default=float(os.environ.get("DISABLE_SETTLE_SECONDS", "5.0")))
    parser.add_argument("--log-every", type=int, default=int(os.environ.get("LOG_EVERY", "1")))
    parser.add_argument("--gui", action="store_true", default=env_bool("GUI", False))
    parser.add_argument("--gui-backend", choices=("web", "opencv", "auto"), default=os.environ.get("GUI_BACKEND", "web"))
    parser.add_argument("--gui-window-name", default=os.environ.get("GUI_WINDOW_NAME", "Interactive Piper DAgger Collection"))
    parser.add_argument("--gui-host", default=os.environ.get("GUI_HOST", "127.0.0.1"))
    parser.add_argument("--gui-port", type=int, default=int(os.environ.get("GUI_PORT", "8765")))

    parser.add_argument("--repo-id", default=os.environ.get("COLLECTION_REPO_ID", os.environ.get("DATASET_REPO_ID", "local/piper_correction_collection")))
    parser.add_argument("--dataset-root", type=Path, default=env_path("COLLECTION_OUTPUT_ROOT", env_path("DATASET_ROOT", default_capture_root())))
    parser.add_argument("--task", default=os.environ.get("TASK", os.environ.get("PROMPT", "stack the yellow bowl on the green bowl")))
    parser.add_argument("--policy-id", default=os.environ.get("POLICY_ID"))
    parser.add_argument("--episode-success", default=os.environ.get("EPISODE_SUCCESS", "unlabeled"))
    parser.add_argument("--no-video", action="store_true", default=env_bool("NO_VIDEO", False))
    parser.add_argument("--image-writer-processes", type=int, default=int(os.environ.get("IMAGE_WRITER_PROCESSES", "0")))
    parser.add_argument("--image-writer-threads", type=int, default=int(os.environ.get("IMAGE_WRITER_THREADS", "4")))

    parser.add_argument("--robot-cameras-json", default=os.environ.get("ROBOT_CAMERAS_JSON"))
    parser.add_argument("--head-camera-serial", type=int, default=int(os.environ.get("HEAD_CAMERA_SERIAL", "254622073267")))
    parser.add_argument("--left-wrist-camera-serial", type=int, default=int(os.environ.get("LEFT_WRIST_CAMERA_SERIAL", "244222071617")))
    parser.add_argument("--right-wrist-camera-serial", type=int, default=int(os.environ.get("RIGHT_WRIST_CAMERA_SERIAL", "317622070857")))
    parser.add_argument("--front-view-camera-serial", type=int, default=int(os.environ.get("FRONT_VIEW_CAMERA_SERIAL", "254622079402")))
    parser.add_argument("--head-camera-key", default=os.environ.get("HEAD_CAMERA_KEY", "head"))
    parser.add_argument("--left-wrist-camera-key", default=os.environ.get("LEFT_WRIST_CAMERA_KEY", "left_wrist"))
    parser.add_argument("--right-wrist-camera-key", default=os.environ.get("RIGHT_WRIST_CAMERA_KEY", "right_wrist"))
    parser.add_argument("--front-view-camera-key", default=os.environ.get("FRONT_VIEW_CAMERA_KEY", "front_view"))
    parser.add_argument("--head-camera-rotation", type=parse_rotation, default=parse_rotation(os.environ.get("HEAD_CAMERA_ROTATION", "none")))
    parser.add_argument("--left-wrist-camera-rotation", type=parse_rotation, default=parse_rotation(os.environ.get("LEFT_WRIST_CAMERA_ROTATION", "none")))
    parser.add_argument("--right-wrist-camera-rotation", type=parse_rotation, default=parse_rotation(os.environ.get("RIGHT_WRIST_CAMERA_ROTATION", "none")))
    parser.add_argument("--front-view-camera-rotation", type=parse_rotation, default=parse_rotation(os.environ.get("FRONT_VIEW_CAMERA_ROTATION", "none")))
    parser.add_argument("--camera-fps", type=int, default=int(os.environ.get("CAMERA_FPS", "15")))
    parser.add_argument("--camera-width", type=int, default=int(os.environ.get("CAMERA_WIDTH", "640")))
    parser.add_argument("--camera-height", type=int, default=int(os.environ.get("CAMERA_HEIGHT", "480")))
    parser.add_argument("--force-camera-reset", action="store_true", default=env_bool("FORCE_CAMERA_RESET", False))

    parser.add_argument("--intervention-key", default=os.environ.get("INTERVENTION_KEY", "i"))
    parser.add_argument("--next-episode-key", default=os.environ.get("NEXT_EPISODE_KEY", "n"))
    parser.add_argument("--stop-key", default=os.environ.get("STOP_KEY", "q"))

    args = parser.parse_args()
    if args.checkpoint_dir is None:
        parser.error("--checkpoint-dir is required, or set CHECKPOINT_DIR.")
    if not 0.0 <= args.action_scale <= 1.0:
        parser.error("--action-scale must be within [0, 1].")
    if args.fps <= 0:
        parser.error("--fps must be positive.")
    if args.steps < 0:
        parser.error("--steps must be >= 0.")
    if args.num_episodes < 0:
        parser.error("--num-episodes must be >= 0.")
    if len(args.intervention_key) != 1:
        parser.error("--intervention-key must be a single character.")
    if len(args.next_episode_key) != 1:
        parser.error("--next-episode-key must be a single character.")
    if len(args.stop_key) != 1:
        parser.error("--stop-key must be a single character.")
    if len({args.intervention_key, args.next_episode_key, args.stop_key}) != 3:
        parser.error("--intervention-key, --next-episode-key, and --stop-key must be different.")
    if args.dataset_root.exists() and not args.validate_only:
        parser.error(f"--dataset-root already exists: {args.dataset_root}")
    return args


def load_runtime_module(runtime_script: Path):
    script_path = runtime_script
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find OpenPI Piper runtime script: {script_path}")
    spec = importlib.util.spec_from_file_location("openpi_piper_adapter", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import runtime script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TerminalKeyPoller:
    def __init__(self) -> None:
        self._fd: int | None = None
        self._old_settings: list[Any] | None = None
        self.enabled = False

    def __enter__(self) -> "TerminalKeyPoller":
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self.enabled = True
        else:
            print("stdin is not a TTY; keyboard intervention is unavailable in this process.")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def poll(self) -> str | None:
        if not self.enabled or self._fd is None:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if not readable:
            return None
        ch = sys.stdin.read(1)
        if ch in {"\r", "\n"}:
            return "ENTER"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch


class OpenCVCollectionUI:
    def __init__(self, args: argparse.Namespace, camera_keys: list[str]) -> None:
        import cv2
        import numpy as np

        self.cv2 = cv2
        self.np = np
        self.window_name = args.gui_window_name
        self.camera_keys = camera_keys[:4]
        self.latest_images: dict[str, Any] = {}
        self.button_rects: dict[str, tuple[int, int, int, int]] = {}
        self.pending_command: str | None = None
        self.window_w = 1320
        self.window_h = 940
        self.cv2.namedWindow(self.window_name, self.cv2.WINDOW_NORMAL)
        self.cv2.resizeWindow(self.window_name, self.window_w, self.window_h)
        self.cv2.setMouseCallback(self.window_name, self._on_mouse)

    def _on_mouse(self, event, x, y, _flags, _param) -> None:
        if event != self.cv2.EVENT_LBUTTONDOWN:
            return
        for command, rect in self.button_rects.items():
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.pending_command = command
                return

    def poll_command(self) -> str | None:
        try:
            if self.cv2.getWindowProperty(self.window_name, self.cv2.WND_PROP_VISIBLE) < 1:
                return "stop"
        except Exception:
            pass

        if self.pending_command is not None:
            command = self.pending_command
            self.pending_command = None
            return command

        key = self.cv2.waitKey(1)
        if key == -1:
            return None
        key = key & 0xFF
        if key in (10, 13):
            return "confirm"
        if key in (ord("i"), ord("I")):
            return "intervention"
        if key in (ord("n"), ord("N")):
            return "next_episode"
        if key in (ord("s"), ord("S")):
            return "next_success"
        if key in (ord("f"), ord("F")):
            return "next_failure"
        if key in (ord("q"), ord("Q"), 27):
            return "stop"
        return None

    def update_from_observation(self, observation: dict[str, Any]) -> None:
        for key in self.camera_keys:
            obs_key = f"observation.images.{key}"
            if obs_key in observation:
                self.latest_images[key] = self._to_hwc_uint8(observation[obs_key])

    def _to_hwc_uint8(self, image: Any):
        array = image
        if hasattr(array, "detach"):
            array = array.detach().cpu().numpy()
        array = self.np.asarray(array)
        if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3, 4):
            array = self.np.transpose(array, (1, 2, 0))
        if array.ndim != 3:
            return self._placeholder_array("BAD IMAGE")
        if array.shape[-1] == 1:
            array = self.np.repeat(array, 3, axis=-1)
        if array.shape[-1] > 3:
            array = array[..., :3]
        if self.np.issubdtype(array.dtype, self.np.floating):
            high = 255.0 if float(self.np.nanmax(array)) > 1.5 else 1.0
            array = self.np.clip(array, 0.0, high)
            if high == 1.0:
                array = array * 255.0
        return array.astype(self.np.uint8, copy=False)

    def _placeholder_array(self, text: str):
        frame = self.np.full((480, 640, 3), 226, dtype=self.np.uint8)
        frame[:90, :, :] = (28, 45, 63)
        self.cv2.putText(frame, text, (36, 250), self.cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 70, 80), 3)
        return frame

    def _fit_panel(self, rgb_image, width: int, height: int):
        canvas = self.np.full((height, width, 3), (38, 44, 52), dtype=self.np.uint8)
        image = self.cv2.cvtColor(rgb_image, self.cv2.COLOR_RGB2BGR)
        ih, iw = image.shape[:2]
        scale = min(width / max(iw, 1), height / max(ih, 1))
        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))
        resized = self.cv2.resize(image, (nw, nh), interpolation=self.cv2.INTER_AREA)
        x0 = (width - nw) // 2
        y0 = (height - nh) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
        return canvas

    def _draw_button(self, canvas, command: str, label: str, rect, active: bool = True) -> None:
        x1, y1, x2, y2 = rect
        color = (58, 119, 92) if active else (94, 104, 116)
        self.cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=-1)
        self.cv2.rectangle(canvas, (x1, y1), (x2, y2), (232, 237, 241), thickness=2)
        self.cv2.putText(canvas, label, (x1 + 18, y1 + 42), self.cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)
        self.button_rects[command] = rect

    def _put_multiline(self, canvas, text: str, x: int, y: int, line_h: int, scale: float, color, thickness: int = 2) -> None:
        max_chars = 98
        lines: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            while len(line) > max_chars:
                lines.append(line[:max_chars])
                line = line[max_chars:]
            lines.append(line)
        for idx, line in enumerate(lines[:5]):
            self.cv2.putText(canvas, line, (x, y + idx * line_h), self.cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    def render(
        self,
        *,
        mode: str,
        episode_index: int,
        num_episodes: int,
        episode_frames: int,
        policy_frames: int,
        intervention_frames: int,
        dataset_root: Path,
        observation: dict[str, Any] | None = None,
        message: str | None = None,
        confirm_label: str = "Enter Confirm",
    ) -> None:
        if observation is not None:
            self.update_from_observation(observation)

        canvas = self.np.full((self.window_h, self.window_w, 3), (244, 241, 235), dtype=self.np.uint8)
        self.button_rects = {}

        display_mode = "confirm" if message else mode
        if display_mode == "human":
            mode_text = "HUMAN DAGGER CORRECTION"
            mode_color = (51, 82, 202)
        elif display_mode == "confirm":
            mode_text = "PAUSED / WAITING FOR CONFIRMATION"
            mode_color = (178, 113, 38)
        else:
            mode_text = "MODEL ROLLOUT"
            mode_color = (58, 119, 92)

        self.cv2.rectangle(canvas, (0, 0), (self.window_w, 132), (31, 47, 65), thickness=-1)
        self.cv2.putText(canvas, mode_text, (28, 46), self.cv2.FONT_HERSHEY_SIMPLEX, 1.22, (255, 255, 255), 3)
        self.cv2.rectangle(canvas, (28, 66), (620, 106), mode_color, thickness=-1)
        total_label = "unlimited" if num_episodes == 0 else str(num_episodes)
        self.cv2.putText(
            canvas,
            f"episode {episode_index + 1}/{total_label}   frames {episode_frames}   policy {policy_frames}   human {intervention_frames}",
            (42, 94),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
        )
        self.cv2.putText(
            canvas,
            f"dataset: {dataset_root}",
            (660, 92),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (230, 235, 240),
            1,
        )

        panel_w = 610
        panel_h = 330
        gap = 22
        x0 = 28
        y0 = 150
        for idx, key in enumerate(self.camera_keys):
            row = idx // 2
            col = idx % 2
            px = x0 + col * (panel_w + gap)
            py = y0 + row * (panel_h + gap)
            self.cv2.rectangle(canvas, (px - 2, py - 30), (px + panel_w + 2, py + panel_h + 2), (214, 207, 195), 2)
            self.cv2.putText(canvas, key, (px + 8, py - 8), self.cv2.FONT_HERSHEY_SIMPLEX, 0.76, (34, 54, 72), 2)
            image = self.latest_images.get(key)
            if image is None:
                image = self._placeholder_array("WAITING FOR FRAME")
            panel = self._fit_panel(image, panel_w, panel_h)
            canvas[py : py + panel_h, px : px + panel_w] = panel

        controls_y = 855
        self.cv2.rectangle(canvas, (0, 830), (self.window_w, self.window_h), (31, 47, 65), thickness=-1)
        if message:
            self._put_multiline(canvas, message, 28, 860, 24, 0.58, (238, 242, 245), 1)
            self._draw_button(canvas, "confirm", confirm_label, (870, controls_y, 1090, controls_y + 62))
            self._draw_button(canvas, "stop", "q  Save + Stop", (1110, controls_y, 1290, controls_y + 62))
        else:
            help_text = "Keys: i intervention | s success+next | f failure+next | n unlabeled+next | q stop."
            self.cv2.putText(canvas, help_text, (28, 868), self.cv2.FONT_HERSHEY_SIMPLEX, 0.58, (238, 242, 245), 1)
            intervention_label = "i  Pause Human" if mode == "human" else "i  Intervention"
            self._draw_button(canvas, "intervention", intervention_label, (480, controls_y, 660, controls_y + 62))
            self._draw_button(canvas, "next_success", "s  Success + Next", (675, controls_y, 875, controls_y + 62))
            self._draw_button(canvas, "next_failure", "f  Failure + Next", (890, controls_y, 1090, controls_y + 62))
            self._draw_button(canvas, "stop", "q  Save + Stop", (1110, controls_y, 1290, controls_y + 62))

        self.cv2.imshow(self.window_name, canvas)

    def close(self) -> None:
        try:
            self.cv2.destroyWindow(self.window_name)
        except Exception:
            pass


class CollectionWebHandler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *_args) -> None:
        return

    @property
    def ui(self):
        return self.server.ui

    def _send_bytes(self, payload: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_bytes(self.ui.index_html().encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/state.json":
            self._send_bytes(self.ui.state_json().encode("utf-8"), "application/json; charset=utf-8")
            return
        if path.startswith("/image/") and path.endswith(".jpg"):
            key = path.removeprefix("/image/").removesuffix(".jpg")
            self._send_bytes(self.ui.image_jpeg(key), "image/jpeg")
            return
        self._send_bytes(b"not found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/command/"):
            command = path.removeprefix("/command/")
            if command in COLLECTION_COMMANDS:
                self.ui.command_queue.put(command)
                self._send_bytes(b'{"ok": true}', "application/json; charset=utf-8")
                return
        self._send_bytes(b'{"ok": false}', "application/json; charset=utf-8", HTTPStatus.BAD_REQUEST)


class CollectionWebServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, ui):
        super().__init__(server_address, RequestHandlerClass)
        self.ui = ui


class WebCollectionUI:
    def __init__(self, args: argparse.Namespace, camera_keys: list[str]) -> None:
        import cv2
        import numpy as np

        self.cv2 = cv2
        self.np = np
        self.camera_keys = camera_keys[:4]
        self.lock = threading.Lock()
        self.command_queue: queue.Queue[str] = queue.Queue()
        self.latest_jpegs: dict[str, bytes] = {}
        self.image_versions: dict[str, int] = {key: 0 for key in self.camera_keys}
        self.state: dict[str, Any] = {
            "mode": "starting",
            "message": "Starting...",
            "episode_index": 0,
            "num_episodes": 0,
            "episode_frames": 0,
            "policy_frames": 0,
            "intervention_frames": 0,
            "dataset_root": "",
            "camera_keys": self.camera_keys,
            "image_versions": self.image_versions,
            "confirm_label": "Enter Confirm",
        }
        self.placeholder_jpeg = self._encode_placeholder("WAITING FOR FRAME")
        self.httpd = self._make_server(args.gui_host, args.gui_port)
        host, port = self.httpd.server_address[:2]
        self.url = f"http://{host}:{port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="collection-web-ui", daemon=True)
        self.thread.start()
        print(f"Web UI is running at {self.url}")

    def _make_server(self, host: str, port: int):
        ports = [port] if port == 0 else list(range(port, port + 20))
        last_error = None
        for candidate_port in ports:
            try:
                return CollectionWebServer((host, candidate_port), CollectionWebHandler, self)
            except OSError as exc:
                last_error = exc
        raise RuntimeError(f"Could not start Web UI on {host}:{port}: {last_error}")

    def _encode_placeholder(self, text: str) -> bytes:
        frame = self.np.full((480, 640, 3), 226, dtype=self.np.uint8)
        frame[:90, :, :] = (63, 45, 28)
        self.cv2.putText(frame, text, (34, 246), self.cv2.FONT_HERSHEY_SIMPLEX, 1.05, (70, 70, 70), 3)
        ok, encoded = self.cv2.imencode(".jpg", frame, [int(self.cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return b""
        return encoded.tobytes()

    def _to_hwc_uint8(self, image: Any):
        array = image
        if hasattr(array, "detach"):
            array = array.detach().cpu().numpy()
        array = self.np.asarray(array)
        if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3, 4):
            array = self.np.transpose(array, (1, 2, 0))
        if array.ndim != 3:
            return None
        if array.shape[-1] == 1:
            array = self.np.repeat(array, 3, axis=-1)
        if array.shape[-1] > 3:
            array = array[..., :3]
        if self.np.issubdtype(array.dtype, self.np.floating):
            high = 255.0 if float(self.np.nanmax(array)) > 1.5 else 1.0
            array = self.np.clip(array, 0.0, high)
            if high == 1.0:
                array = array * 255.0
        return array.astype(self.np.uint8, copy=False)

    def _encode_rgb_jpeg(self, rgb_image) -> bytes | None:
        bgr = self.cv2.cvtColor(rgb_image, self.cv2.COLOR_RGB2BGR)
        ok, encoded = self.cv2.imencode(".jpg", bgr, [int(self.cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return None
        return encoded.tobytes()

    def update_from_observation(self, observation: dict[str, Any]) -> None:
        updates: dict[str, bytes] = {}
        for key in self.camera_keys:
            obs_key = f"observation.images.{key}"
            if obs_key not in observation:
                continue
            image = self._to_hwc_uint8(observation[obs_key])
            if image is None:
                continue
            jpeg = self._encode_rgb_jpeg(image)
            if jpeg is not None:
                updates[key] = jpeg

        if not updates:
            return
        with self.lock:
            for key, jpeg in updates.items():
                self.latest_jpegs[key] = jpeg
                self.image_versions[key] = self.image_versions.get(key, 0) + 1
            self.state["image_versions"] = dict(self.image_versions)

    def poll_command(self) -> str | None:
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    def render(
        self,
        *,
        mode: str,
        episode_index: int,
        num_episodes: int,
        episode_frames: int,
        policy_frames: int,
        intervention_frames: int,
        dataset_root: Path,
        observation: dict[str, Any] | None = None,
        message: str | None = None,
        confirm_label: str = "Enter Confirm",
    ) -> None:
        if observation is not None:
            self.update_from_observation(observation)
        with self.lock:
            self.state.update(
                {
                    "mode": "paused" if message else mode,
                    "message": message or "",
                    "episode_index": episode_index,
                    "num_episodes": num_episodes,
                    "episode_frames": episode_frames,
                    "policy_frames": policy_frames,
                    "intervention_frames": intervention_frames,
                    "dataset_root": str(dataset_root),
                    "camera_keys": list(self.camera_keys),
                    "image_versions": dict(self.image_versions),
                    "confirm_label": confirm_label,
                    "updated_at": time.time(),
                }
            )

    def state_json(self) -> str:
        with self.lock:
            return json.dumps(self.state, separators=(",", ":"))

    def image_jpeg(self, key: str) -> bytes:
        with self.lock:
            return self.latest_jpegs.get(key, self.placeholder_jpeg)

    def index_html(self) -> str:
        camera_items = "\n".join(
            f'<div class="camera"><div class="camera-title">{key}</div><img id="img-{key}" src="/image/{key}.jpg"></div>'
            for key in self.camera_keys
        )
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Interactive Piper DAgger Collection</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; }}
    html, body {{ height: 100%; overflow: hidden; }}
    body {{ margin: 0; background: #f2eee6; color: #1d3144; display: flex; flex-direction: column; }}
    header {{ background: #203044; color: white; padding: 8px 14px 7px; flex: 0 0 auto; }}
    #mode {{ display: inline-block; padding: 5px 10px; border-radius: 4px; background: #3a775c; font-size: 18px; font-weight: 800; }}
    #stats {{ margin-top: 5px; color: #e5edf2; font-size: 13px; }}
    #dataset {{ margin-top: 3px; color: #cbd6df; font-size: 11px; word-break: break-all; }}
    main {{
      flex: 1 1 auto;
      min-height: 0;
      display: grid;
      grid-template-columns: repeat(2, minmax(260px, 1fr));
      grid-template-rows: repeat(2, minmax(0, 1fr));
      gap: 8px;
      padding: 8px;
      overflow: hidden;
    }}
    .camera {{ min-height: 0; background: #fffdf8; border: 1px solid #d8d0c2; padding: 5px; display: flex; flex-direction: column; }}
    .camera-title {{ flex: 0 0 auto; font-weight: 800; font-size: 14px; margin: 0 0 4px; line-height: 1.1; }}
    .camera img {{ flex: 1 1 auto; min-height: 0; width: 100%; height: 100%; object-fit: contain; background: #252b33; display: block; }}
    footer {{ flex: 0 0 auto; background: #203044; color: white; padding: 8px 10px; display: flex; gap: 7px; align-items: center; }}
    button {{ border: 1px solid #edf3f8; color: white; background: #3a775c; padding: 8px 10px; border-radius: 4px; font-size: 13px; font-weight: 800; cursor: pointer; white-space: nowrap; }}
    button.stop {{ background: #9a493d; }}
    button.secondary {{ background: #576779; }}
    button.failure {{ background: #8f4d5b; }}
    #message {{ flex: 1; min-height: 36px; max-height: 78px; overflow: auto; color: #ecf2f6; white-space: pre-line; font-size: 13px; line-height: 1.18; }}
    @media (max-width: 900px) {{
      html, body {{ overflow: auto; }}
      body {{ display: block; }}
      main {{ grid-template-columns: 1fr; grid-template-rows: none; overflow: visible; }}
      .camera img {{ height: auto; aspect-ratio: 4 / 3; }}
      footer {{ position: sticky; bottom: 0; flex-wrap: wrap; }}
    }}
  </style>
</head>
<body>
  <header>
    <div id="mode">STARTING</div>
    <div id="stats"></div>
    <div id="dataset"></div>
  </header>
  <main>{camera_items}</main>
  <footer>
    <div id="message">Loading...</div>
    <button onclick="sendCommand('intervention')">i Intervention</button>
    <button onclick="sendCommand('next_success')">s Save Success + Next</button>
    <button class="failure" onclick="sendCommand('next_failure')">f Save Failure + Next</button>
    <button class="secondary" onclick="sendCommand('next_episode')">n Save Unlabeled + Next</button>
    <button id="confirm-btn" class="secondary" onclick="sendCommand('confirm')">Enter Confirm</button>
    <button class="stop" onclick="sendCommand('stop')">q Save + Stop</button>
  </footer>
  <script>
    const cameraKeys = {json.dumps(self.camera_keys)};
    let versions = {{}};
    async function sendCommand(command) {{
      await fetch('/command/' + command, {{method: 'POST'}});
    }}
    function modeLabel(mode) {{
      if (mode === 'human') return ['HUMAN DAGGER CORRECTION', '#c94f3f'];
      if (mode === 'paused') return ['PAUSED / WAITING FOR CONFIRMATION', '#b27126'];
      if (mode === 'policy') return ['MODEL ROLLOUT', '#3a775c'];
      return [mode.toUpperCase(), '#576779'];
    }}
    async function refresh() {{
      const res = await fetch('/state.json', {{cache: 'no-store'}});
      const state = await res.json();
      const [label, color] = modeLabel(state.mode || 'starting');
      const mode = document.getElementById('mode');
      mode.textContent = label;
      mode.style.background = color;
      const total = state.num_episodes === 0 ? 'unlimited' : state.num_episodes;
      document.getElementById('stats').textContent =
        `episode ${{(state.episode_index || 0) + 1}}/${{total}}   frames ${{state.episode_frames || 0}}   policy ${{state.policy_frames || 0}}   human ${{state.intervention_frames || 0}}`;
      document.getElementById('dataset').textContent = state.dataset_root || '';
      document.getElementById('message').textContent = state.message || 'Keys: i intervention, s success + next, f failure + next, n unlabeled + next, Enter confirm, q save + stop.';
      document.getElementById('confirm-btn').textContent = state.confirm_label || 'Enter Confirm';
      for (const key of cameraKeys) {{
        const version = (state.image_versions || {{}})[key] || 0;
        if (versions[key] !== version) {{
          versions[key] = version;
          const img = document.getElementById('img-' + key);
          img.src = `/image/${{key}}.jpg?v=${{version}}&t=${{Date.now()}}`;
        }}
      }}
    }}
    document.addEventListener('keydown', (event) => {{
      if (event.key === 'i' || event.key === 'I') sendCommand('intervention');
      else if (event.key === 's' || event.key === 'S') sendCommand('next_success');
      else if (event.key === 'f' || event.key === 'F') sendCommand('next_failure');
      else if (event.key === 'n' || event.key === 'N') sendCommand('next_episode');
      else if (event.key === 'q' || event.key === 'Q' || event.key === 'Escape') sendCommand('stop');
      else if (event.key === 'Enter') sendCommand('confirm');
    }});
    refresh();
    setInterval(refresh, 200);
  </script>
</body>
</html>"""

    def close(self) -> None:
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.thread.join(timeout=2.0)
        except Exception:
            pass


def create_collection_ui(args: argparse.Namespace, camera_keys: list[str]):
    if not args.gui:
        return None
    if args.gui_backend in {"web", "auto"}:
        return WebCollectionUI(args, camera_keys)
    try:
        return OpenCVCollectionUI(args, camera_keys)
    except Exception as exc:
        if args.gui_backend == "auto":
            print(f"OpenCV GUI is unavailable ({exc}); falling back to Web UI.")
            return WebCollectionUI(args, camera_keys)
        raise


def poll_collection_command(args: argparse.Namespace, poller: TerminalKeyPoller, ui: Any | None) -> str | None:
    if ui is not None:
        command = ui.poll_command()
        if command is not None:
            return command

    key = poller.poll()
    if key == args.intervention_key:
        return "intervention"
    if key == args.next_episode_key:
        return "next_episode"
    if key in {"s", "S"}:
        return "next_success"
    if key in {"f", "F"}:
        return "next_failure"
    if key == args.stop_key:
        return "stop"
    if key == "ENTER":
        return "confirm"
    return None


def wait_for_enter(
    poller: TerminalKeyPoller,
    prompt: str,
    stop_key: str,
    *,
    ui: Any | None = None,
    ui_context=None,
    confirm_label: str = "Enter Confirm",
) -> bool:
    print(prompt, flush=True)
    while True:
        if ui is not None:
            context = ui_context() if ui_context is not None else {}
            ui.render(message=prompt, confirm_label=confirm_label, **context)
            command = ui.poll_command()
            if command == "confirm":
                return True
            if command == "stop":
                return False

        key = poller.poll()
        if key == "ENTER":
            return True
        if key == stop_key:
            return False
        time.sleep(0.05)


def make_action_prefetcher(runtime, policy, args: argparse.Namespace):
    return runtime.AsyncActionChunkPrefetcher(
        policy,
        actions_per_inference=args.actions_per_inference,
        enabled=not args.sync_inference,
        prefetch_threshold=args.prefetch_threshold,
    )


def as_float32_array(runtime, value: Any):
    import numpy as np

    return runtime.to_numpy(value).astype(np.float32, copy=False)


def build_dataset_schema(robot) -> dict[str, dict]:
    features: dict[str, dict] = {}
    for key in ("action", "observation.state", "observation.ee_pose"):
        features[key] = dict(robot.motor_features[key])
    for key, ft in robot.camera_features.items():
        features[key] = {"dtype": "video", **ft}
    features["complementary_info.policy_action"] = {
        "dtype": "float32",
        "shape": tuple(features["action"]["shape"]),
        "names": list(features["action"].get("names", [])),
    }
    features["complementary_info.is_intervention"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["is_intervention"],
    }
    features["complementary_info.collector_policy_id"] = {
        "dtype": "string",
        "shape": (1,),
        "names": ["collector_policy_id"],
    }
    features["complementary_info.state"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["state"],
    }
    return features


def build_collection_frame(
    *,
    runtime,
    robot_observation: dict[str, Any],
    action,
    policy_action,
    is_intervention: bool,
    policy_id: str,
    task: str,
) -> dict[str, Any]:
    import numpy as np

    frame: dict[str, Any] = {
        "action": np.asarray(action, dtype=np.float32),
        "observation.state": as_float32_array(runtime, robot_observation["observation.state"]).copy(),
        "observation.ee_pose": as_float32_array(runtime, robot_observation["observation.ee_pose"]).copy(),
        "complementary_info.policy_action": np.asarray(policy_action, dtype=np.float32),
        "complementary_info.is_intervention": np.asarray([1.0 if is_intervention else 0.0], dtype=np.float32),
        "complementary_info.collector_policy_id": policy_id,
        "complementary_info.state": np.asarray([1.0 if is_intervention else 0.0], dtype=np.float32),
        "task": task,
    }
    for key, value in robot_observation.items():
        if key.startswith("observation.images."):
            frame[key] = value
    return frame


def build_episode_summary(
    args: argparse.Namespace,
    intervention_flags: list[bool],
    *,
    episode_success: str,
) -> dict[str, Any]:
    intervention_indices = [idx for idx, flag in enumerate(intervention_flags) if flag]
    intervention_frames = len(intervention_indices)
    metadata = {
        "episode_success": episode_success,
        "policy_frames": len(intervention_flags) - intervention_frames,
        "intervention_frames": intervention_frames,
        "first_intervention_frame": intervention_indices[0] if intervention_indices else -1,
        "last_intervention_frame": intervention_indices[-1] if intervention_indices else -1,
    }
    return metadata


def commit_current_episode(
    dataset,
    args: argparse.Namespace,
    *,
    episode_frames: int,
    intervention_flags: list[bool],
    episode_success: str,
) -> bool:
    if dataset is None or dataset.episode_buffer is None or dataset.episode_buffer["size"] == 0:
        print("Current episode has no frames; nothing was saved.")
        return False

    episode_index = int(dataset.meta.total_episodes)
    metadata = build_episode_summary(args, intervention_flags, episode_success=episode_success)
    dataset.save_episode(episode_metadata=metadata)
    print(
        f"Saved episode {episode_index} with {episode_frames} frames to {args.dataset_root} "
        f"(success={metadata['episode_success']}, policy={metadata['policy_frames']}, "
        f"intervention={metadata['intervention_frames']}, "
        f"first_intervention={metadata['first_intervention_frame']})"
    )
    return True


def sample_policy_action(runtime, policy, train_config, norm_stats, args, robot_observation, pending_actions, prefetcher_state):
    import numpy as np

    can_state = as_float32_array(runtime, robot_observation["observation.state"])
    full_state, active_dims = runtime.build_full_policy_state(can_state, train_config, norm_stats)
    raw_observation = runtime.build_raw_openpi_observation(robot_observation, full_state, args.prompt, args)

    ready_chunk = prefetcher_state["prefetcher"].take_ready()
    if ready_chunk is not None:
        prefetcher_state["prefetched_chunk"] = ready_chunk

    if not pending_actions:
        if prefetcher_state["prefetched_chunk"] is not None:
            action_chunk = prefetcher_state["prefetched_chunk"].actions
            prefetcher_state["last_infer_ms"] = prefetcher_state["prefetched_chunk"].infer_ms
            prefetcher_state["prefetched_chunk"] = None
        elif prefetcher_state["prefetcher"].has_in_flight():
            chunk = prefetcher_state["prefetcher"].wait()
            action_chunk = chunk.actions
            prefetcher_state["last_infer_ms"] = chunk.infer_ms
        else:
            action_chunk, prefetcher_state["last_infer_ms"] = runtime.infer_action_chunk(
                policy,
                raw_observation,
                args.actions_per_inference,
            )
        arm_action_chunk = runtime.extract_active_arm_action(action_chunk, active_dims)
        pending_actions.extend(arm_action_chunk)

    model_action = np.asarray(pending_actions.popleft(), dtype=np.float32)
    if prefetcher_state["prefetched_chunk"] is None:
        prefetcher_state["prefetcher"].maybe_submit(raw_observation, len(pending_actions))

    scaled_action = runtime.scale_active_arm_action(model_action, can_state, args.action_scale)
    final_action = runtime.clamp_active_arm_action(scaled_action, can_state, args.max_abs_delta).astype(np.float32, copy=False)
    return can_state, model_action, scaled_action, final_action, prefetcher_state["last_infer_ms"]


def log_policy_step(runtime, args, step: int, robot, can_state, model_action, scaled_action, final_action, infer_ms, queue_len) -> None:
    command_counts = None
    if args.print_command_counts:
        command_counts = runtime.action_to_command_counts(final_action, robot.follower_arms["main"])
    if args.debug_action_details:
        runtime.log_action_details(
            step=step,
            infer_ms=infer_ms,
            queue_len=queue_len,
            can_name=args.can_name,
            current_state_7d=can_state,
            model_action_7d=model_action,
            scaled_action_7d=scaled_action,
            final_action_7d=final_action,
            dry_run=args.dry_run,
            command_counts=command_counts,
        )
    elif step % max(1, args.log_every) == 0:
        preview = ", ".join(f"{value:+.4f}" for value in final_action[:6])
        suffix = ""
        if command_counts is not None:
            suffix = " " + runtime.format_command_counts(f"{args.can_name}.cmd_counts", *command_counts)
        dry = " dry-run" if args.dry_run else ""
        print(
            f"[policy{dry} step={step}] infer={infer_ms:.1f}ms queue={queue_len} "
            f"{args.can_name}[:6]=[{preview}] grip={final_action[6]:+.4f}{suffix}"
        )


def main() -> None:
    args = parse_args()
    args.runtime_script = args.runtime_script.resolve()
    args.lerobot_runtime_root = args.lerobot_runtime_root.resolve()
    args.openpi_root = args.openpi_root.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.dataset_root = args.dataset_root.resolve()
    if args.norm_stats_path is not None:
        args.norm_stats_path = args.norm_stats_path.resolve()
    if args.robot_cameras_json is None:
        args.robot_cameras_json = build_camera_layout(args)

    if not args.runtime_script.exists():
        raise FileNotFoundError(
            f"Policy runtime script does not exist: {args.runtime_script}. "
            "Set POLICY_RUNTIME_SCRIPT if you use a custom runtime adapter."
        )
    if not args.openpi_root.exists():
        raise FileNotFoundError(
            f"OpenPI root does not exist: {args.openpi_root}. Set OPENPI_ROOT or pass --openpi-root."
        )
    if not args.checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {args.checkpoint_dir}. Set CHECKPOINT_DIR or pass --checkpoint-dir."
        )

    runtime = load_runtime_module(args.runtime_script)
    runtime.ensure_lerobot_runtime_on_path(args.lerobot_runtime_root)
    runtime.ensure_openpi_on_path(args.openpi_root)

    runtime.ensure_checkpoint_artifacts(args.checkpoint_dir, args.norm_stats_path)
    runtime.validate_checkpoint_safetensors(args.checkpoint_dir / "model.safetensors")
    metadata = runtime.load_checkpoint_metadata(args.checkpoint_dir)
    policy, train_config, norm_stats, resolved_norm_stats_path = runtime.load_policy(
        args.checkpoint_dir,
        args.openpi_root,
        metadata,
        args.prompt,
        args.norm_stats_path,
        args.device,
    )
    policy_id = args.policy_id or f"{metadata['config']['name']}:{metadata.get('global_step', 'unknown')}"

    print(
        "Loaded OpenPI policy: "
        f"name={metadata['config']['name']} step={metadata.get('global_step')} "
        f"prompt={args.prompt!r} can_name={args.can_name} fps={args.fps} "
        f"action_scale={args.action_scale:.2f} max_abs_delta={args.max_abs_delta:.3f} "
        f"norm_stats={resolved_norm_stats_path}"
    )

    if args.validate_only:
        runtime.run_validation_once(policy, train_config, norm_stats, args)
        return

    import torch

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.robot_devices.robots.piper import PiperRobot
    from lerobot.common.robot_devices.utils import busy_wait

    robot = PiperRobot(runtime.load_single_arm_robot_config(args.robot_cameras_json, args.can_name))
    ui = create_collection_ui(args, camera_keys_from_json(args.robot_cameras_json))
    dataset = None
    pending_actions = deque()
    prefetcher_state: dict[str, Any] = {
        "prefetcher": make_action_prefetcher(runtime, policy, args),
        "prefetched_chunk": None,
        "last_infer_ms": 0.0,
    }
    episode_frames = 0
    total_recorded_frames = 0
    episodes_saved = 0
    intervention_flags: list[bool] = []
    mode = "policy"
    max_episode_frames = args.steps if args.steps > 0 else None

    def current_ui_context() -> dict[str, Any]:
        intervention_frames = sum(1 for flag in intervention_flags if flag)
        return {
            "mode": mode,
            "episode_index": episodes_saved,
            "num_episodes": args.num_episodes,
            "episode_frames": episode_frames,
            "policy_frames": episode_frames - intervention_frames,
            "intervention_frames": intervention_frames,
            "dataset_root": args.dataset_root,
        }

    try:
        robot.connect()
        if not args.dry_run and args.enable_arm:
            runtime.maybe_enable_main_arm(robot)

        dataset = LeRobotDataset.create(
            args.repo_id,
            fps=args.fps,
            root=args.dataset_root,
            robot_type=robot.robot_type,
            features=build_dataset_schema(robot),
            use_videos=not args.no_video,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
        )
        print(f"Recording dataset to: {args.dataset_root}")
        print(
            f"Controls: press '{args.intervention_key}' to enter/leave human correction; "
            f"press '{args.next_episode_key}' to save this episode and start the next; "
            f"press '{args.stop_key}' to save and stop."
        )
        if ui is not None:
            ui.render(**current_ui_context())

        with TerminalKeyPoller() as poller:
            while args.num_episodes == 0 or episodes_saved < args.num_episodes:
                command = poll_collection_command(args, poller, ui)
                if command == "stop":
                    print("Stop requested. Saving current episode.")
                    break

                if command in EPISODE_COMMIT_COMMANDS:
                    episode_success = episode_success_from_command(args, command)
                    if mode == "human":
                        ok = wait_for_enter(
                            poller,
                            "\nEpisode save requested during human correction.\n"
                            f"1) Disconnect the hardware master from {args.can_name}.\n"
                            "2) Leave only the follower on the PC CAN bus.\n"
                            f"3) Press Enter here to save this episode as '{episode_success}' and prepare the next one.\n"
                            f"Press '{args.stop_key}' instead to stop and save.",
                            args.stop_key,
                            ui=ui,
                            ui_context=current_ui_context,
                            confirm_label=save_button_label(command),
                        )
                        if not ok:
                            break
                        if not args.dry_run and args.enable_arm:
                            runtime.maybe_enable_main_arm(robot)
                        mode = "policy"

                    saved = commit_current_episode(
                        dataset,
                        args,
                        episode_frames=episode_frames,
                        intervention_flags=intervention_flags,
                        episode_success=episode_success,
                    )
                    if saved:
                        episodes_saved += 1
                        total_recorded_frames += episode_frames
                        episode_frames = 0
                        intervention_flags = []
                        pending_actions.clear()
                        prefetcher_state["prefetcher"].shutdown()
                        prefetcher_state["prefetcher"] = make_action_prefetcher(runtime, policy, args)
                        prefetcher_state["prefetched_chunk"] = None
                        if args.num_episodes != 0 and episodes_saved >= args.num_episodes:
                            break
                        ok = wait_for_enter(
                            poller,
                            f"\nEpisode {episodes_saved - 1} saved. Reset the scene for episode {episodes_saved}.\n"
                            f"Keep only the follower connected to {args.can_name}, then press Enter to start the next model rollout.\n"
                            f"Press '{args.stop_key}' instead to stop.",
                            args.stop_key,
                            ui=ui,
                            ui_context=current_ui_context,
                            confirm_label="Start Next Rollout",
                        )
                        if not ok:
                            break
                    continue

                if command == "intervention" and mode == "policy":
                    pending_actions.clear()
                    prefetcher_state["prefetcher"].shutdown()
                    prefetcher_state["prefetcher"] = make_action_prefetcher(runtime, policy, args)
                    prefetcher_state["prefetched_chunk"] = None
                    ok = wait_for_enter(
                        poller,
                        "\nPolicy paused. No model commands are being sent.\n"
                        f"1) Connect the hardware master and follower on {args.can_name}.\n"
                        "2) Verify the master can drive the follower.\n"
                        "3) Press Enter here to start recording human correction frames.\n"
                        f"Press '{args.stop_key}' instead to stop and save.",
                        args.stop_key,
                        ui=ui,
                        ui_context=current_ui_context,
                        confirm_label="Start Human Correction",
                    )
                    if not ok:
                        break
                    mode = "human"
                    print("Human correction recording started.")
                    continue

                if command == "intervention" and mode == "human":
                    ok = wait_for_enter(
                        poller,
                        "\nHuman correction paused.\n"
                        f"1) Disconnect the hardware master from {args.can_name}.\n"
                        "2) Leave only the follower on the PC CAN bus.\n"
                        "3) Press Enter here to re-enable policy rollout.\n"
                        f"Press '{args.stop_key}' instead to stop and save.",
                        args.stop_key,
                        ui=ui,
                        ui_context=current_ui_context,
                        confirm_label="Resume Model Rollout",
                    )
                    if not ok:
                        break
                    if not args.dry_run and args.enable_arm:
                        runtime.maybe_enable_main_arm(robot)
                    pending_actions.clear()
                    prefetcher_state["prefetcher"].shutdown()
                    prefetcher_state["prefetcher"] = make_action_prefetcher(runtime, policy, args)
                    prefetcher_state["prefetched_chunk"] = None
                    mode = "policy"
                    print("Policy rollout resumed.")
                    continue

                loop_start = time.perf_counter()
                robot_observation = robot.capture_observation()

                if mode == "policy":
                    can_state, model_action, scaled_action, final_action, infer_ms = sample_policy_action(
                        runtime,
                        policy,
                        train_config,
                        norm_stats,
                        args,
                        robot_observation,
                        pending_actions,
                        prefetcher_state,
                    )
                    log_policy_step(
                        runtime,
                        args,
                        episode_frames,
                        robot,
                        can_state,
                        model_action,
                        scaled_action,
                        final_action,
                        infer_ms,
                        len(pending_actions),
                    )
                    if not args.dry_run:
                        robot.send_action(torch.from_numpy(final_action))
                    action = final_action
                    policy_action = final_action
                    is_intervention = False
                else:
                    observed_state = as_float32_array(runtime, robot_observation["observation.state"]).copy()
                    action = observed_state
                    policy_action = observed_state
                    is_intervention = True
                    if episode_frames % max(1, args.log_every) == 0:
                        preview = ", ".join(f"{value:+.4f}" for value in observed_state[:6])
                        print(
                            f"[human step={episode_frames}] recorded follower state as action "
                            f"{args.can_name}[:6]=[{preview}] grip={observed_state[6]:+.4f}"
                        )

                dataset.add_frame(
                    build_collection_frame(
                        runtime=runtime,
                        robot_observation=robot_observation,
                        action=action,
                        policy_action=policy_action,
                        is_intervention=is_intervention,
                        policy_id=policy_id,
                        task=args.task,
                    )
                )
                intervention_flags.append(is_intervention)
                episode_frames += 1
                if ui is not None:
                    ui.render(observation=robot_observation, **current_ui_context())
                loop_dt = time.perf_counter() - loop_start
                busy_wait(max(0.0, 1.0 / args.fps - loop_dt))

                if max_episode_frames is not None and episode_frames >= max_episode_frames:
                    if mode == "human":
                        ok = wait_for_enter(
                            poller,
                            f"\nReached --steps={args.steps} during human correction.\n"
                            f"1) Disconnect the hardware master from {args.can_name}.\n"
                            "2) Leave only the follower on the PC CAN bus.\n"
                            "3) Press Enter here to save this episode and prepare the next one.\n"
                            f"Press '{args.stop_key}' instead to stop and save.",
                            args.stop_key,
                            ui=ui,
                            ui_context=current_ui_context,
                            confirm_label="Save Episode",
                        )
                        if not ok:
                            break
                        if not args.dry_run and args.enable_arm:
                            runtime.maybe_enable_main_arm(robot)
                        mode = "policy"

                    saved = commit_current_episode(
                        dataset,
                        args,
                        episode_frames=episode_frames,
                        intervention_flags=intervention_flags,
                        episode_success=args.episode_success,
                    )
                    if saved:
                        episodes_saved += 1
                        total_recorded_frames += episode_frames
                        episode_frames = 0
                        intervention_flags = []
                        pending_actions.clear()
                        prefetcher_state["prefetcher"].shutdown()
                        prefetcher_state["prefetcher"] = make_action_prefetcher(runtime, policy, args)
                        prefetcher_state["prefetched_chunk"] = None
                        if args.num_episodes != 0 and episodes_saved >= args.num_episodes:
                            break
                        ok = wait_for_enter(
                            poller,
                            f"\nReached --steps={args.steps}; episode {episodes_saved - 1} saved.\n"
                            f"Reset the scene for episode {episodes_saved}, keep only the follower on {args.can_name}, "
                            "then press Enter to start the next model rollout.\n"
                            f"Press '{args.stop_key}' instead to stop.",
                            args.stop_key,
                            ui=ui,
                            ui_context=current_ui_context,
                            confirm_label="Start Next Rollout",
                        )
                        if not ok:
                            break

    except KeyboardInterrupt:
        print("Interrupted by user. Saving current episode if it contains frames.")
    finally:
        try:
            prefetcher_state["prefetcher"].shutdown()
        finally:
            try:
                if dataset is not None and dataset.episode_buffer is not None and dataset.episode_buffer["size"] > 0:
                    if mode == "human":
                        print(
                            "Warning: stopping while in human correction mode. "
                            f"Disconnect the hardware master from {args.can_name} before running policy again."
                        )
                    if commit_current_episode(
                        dataset,
                        args,
                        episode_frames=episode_frames,
                        intervention_flags=intervention_flags,
                        episode_success=args.episode_success,
                    ):
                        episodes_saved += 1
                        total_recorded_frames += episode_frames
                elif dataset is not None:
                    print("No frames recorded in the current episode; nothing else was saved.")
                if dataset is not None:
                    print(
                        f"Finished with {episodes_saved} saved episode(s), "
                        f"{total_recorded_frames} total frame(s), root={args.dataset_root}"
                    )
            finally:
                try:
                    if args.disable_arm_on_exit and not args.dry_run:
                        runtime.maybe_disable_main_arm(robot, args.disable_settle_seconds)
                finally:
                    if robot.is_connected:
                        robot.disconnect()
                    if ui is not None:
                        ui.close()


if __name__ == "__main__":
    main()
