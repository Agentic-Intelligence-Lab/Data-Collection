#!/usr/bin/env python3

import argparse
import json
import time
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from teleoperation.dataset.episode_cleanup import (
    clear_episode_artifacts,
    delete_last_episode,
    repair_trailing_inconsistent_episodes,
)
from teleoperation.dataset.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from teleoperation.dataset.utils import REAL_TIME_TIMESTAMP_MODE, get_features_from_robot
from teleoperation.robot.config import (
    PiperRobotConfig,
    piper_robot_config_from_dict,
    realsense_camera_from_dict,
)
from teleoperation.robot.piper import PiperRobot


CAMERA_LABELS = {
    "head": "Head View",
    "left_wrist": "Left Wrist View",
    "right_wrist": "Right Wrist View",
    "front_view": "Front View",
}

REQUIRED_RESUME_METADATA_FILES = (
    "meta/tasks.jsonl",
    "meta/episodes.jsonl",
    "meta/episodes_stats.jsonl",
)


def _camera_display_label(key: str) -> str:
    label = CAMERA_LABELS.get(key)
    if label is not None:
        return label

    words = [word for word in key.replace("-", "_").split("_") if word]
    pretty = " ".join(word.capitalize() for word in words) or key
    if pretty.lower().endswith(" view"):
        return pretty
    return f"{pretty} View"


def _load_robot_config(camera_json: str | None, robot_config_json: str | None) -> PiperRobotConfig:
    cameras = None
    if camera_json:
        cameras = {
            key: realsense_camera_from_dict(camera_cfg)
            for key, camera_cfg in json.loads(camera_json).items()
        }
    robot_config = json.loads(robot_config_json) if robot_config_json else None
    return piper_robot_config_from_dict(robot_config, cameras=cameras)


class PiperCollectionGUI:
    def __init__(self, args):
        self.args = args
        self.robot_config = _load_robot_config(args.robot_cameras_json, args.robot_config_json)
        self.camera_keys = list(self.robot_config.cameras)
        self.root = tk.Tk()
        self.root.title("Piper Data Collection")
        self.root.geometry("1840x1120")
        self.root.minsize(1620, 980)
        self.root.configure(bg="#f3efe7")
        self.root.bind("<space>", self._on_space)
        self.root.bind("<BackSpace>", self._on_delete_last_episode)
        self.root.bind("<Escape>", self._on_escape)
        self.root.bind("<Configure>", self._on_resize)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.robot = None
        self.dataset = None

        self.state = "booting"
        self.recorded_episodes = 0
        self.current_frame_count = 0
        self.episode_start_t = None
        self.pending_record_sample = None
        self.next_record_tick_deadline_t = None
        self.episode_first_real_timestamp_s = None
        self.episode_last_real_timestamp_s = None
        self.episode_first_wall_time_ns = None
        self.episode_last_wall_time_ns = None
        self.last_preview_t = 0.0
        self.space_requested = False
        self.stop_requested = False
        self.auto_close_scheduled = False
        self.resize_refresh_pending = False
        self.latest_images = {}
        self.tk_images = {}
        self.dataset_repair_note = ""
        self.display_font = self._pick_font(
            ["Aptos Display", "SF Pro Display", "Segoe UI Variable", "Noto Sans", "DejaVu Sans", "Liberation Sans"],
            "TkDefaultFont",
        )
        self.body_font = self._pick_font(
            ["Aptos", "Segoe UI", "Noto Sans", "DejaVu Sans", "Liberation Sans"],
            "TkDefaultFont",
        )

        self._build_ui()
        self._initialize_backend()
        self._schedule_tick()

    def _pick_font(self, preferred: list[str], fallback: str) -> str:
        available = set(tkfont.families(self.root))
        for family in preferred:
            if family in available:
                return family
        return fallback

    def _camera_grid_columns(self) -> int:
        count = max(len(self.camera_keys), 1)
        if count <= 3:
            return count
        if count <= 4:
            return 2
        return 3

    def _camera_grid_rows(self) -> int:
        return max(1, (len(self.camera_keys) + self._camera_grid_columns() - 1) // self._camera_grid_columns())

    def _fallback_preview_thumbnail_size(self) -> tuple[int, int]:
        cols = self._camera_grid_columns()
        rows = self._camera_grid_rows()
        if cols == 1:
            return (1180, 700)
        if cols == 2:
            return (820, 460) if rows == 1 else (760, 320)
        return (560, 360)

    def _preview_thumbnail_size(self, key: str) -> tuple[int, int]:
        panel = self.camera_panels.get(key)
        if panel is None:
            return self._fallback_preview_thumbnail_size()

        width = max(panel.winfo_width() - 12, 1)
        height = max(panel.winfo_height() - 12, 1)
        if width <= 1 or height <= 1:
            return self._fallback_preview_thumbnail_size()
        return (width, height)

    def _set_panel_image(self, key: str, render: Image.Image) -> None:
        panel = self.camera_panels[key]
        render = render.copy()
        render.thumbnail(self._preview_thumbnail_size(key))
        photo = ImageTk.PhotoImage(render)
        panel.configure(image=photo)
        panel.image = photo
        self.tk_images[key] = photo

    def _make_placeholder(self, key: str, detail: str) -> tuple[np.ndarray, str, str]:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (230, 235, 240)
        frame[0:84, :, :] = (24, 52, 79)
        frame[84:, :, :] = (222, 228, 233)

        label = _camera_display_label(key)
        # Simple block lettering effect without adding new deps.
        frame[140:144, 56:584, :] = (188, 198, 208)
        frame[260:264, 56:584, :] = (188, 198, 208)
        return frame, label, detail

    def _render_placeholder_image(
        self,
        key: str,
        detail: str,
        subdetail: str | None = None,
    ) -> Image.Image:
        render_frame, label, detail = self._make_placeholder(key, detail)
        render = Image.fromarray(render_frame)
        draw = ImageDraw.Draw(render)
        draw.text((28, 22), label, fill=(255, 248, 214))
        draw.text((28, 120), detail, fill=(44, 61, 80))
        if subdetail:
            draw.text((28, 168), subdetail, fill=(96, 108, 122))
        return render

    def _missing_camera_summary(self) -> str:
        if self.robot is None or not getattr(self.robot, "unavailable_cameras", None):
            return ""

        parts = []
        for key, info in self.robot.unavailable_cameras.items():
            serial_number = info.get("serial_number", "unknown")
            parts.append(f"{_camera_display_label(key)} ({serial_number})")
        return ", ".join(parts)

    def _append_missing_camera_note(self, footer: str) -> str:
        summary = self._missing_camera_summary()
        parts = [footer]
        if summary:
            parts.append(f"Missing for this run: {summary}.")
        if self.dataset_repair_note:
            parts.append(self.dataset_repair_note)
        return " ".join(parts)

    def _build_ui(self):
        outer = tk.Frame(self.root, bg="#f3efe7", padx=24, pady=22)
        outer.pack(fill="both", expand=True)

        title = tk.Label(
            outer,
            text="Piper Passive Data Collection",
            bg="#f3efe7",
            fg="#18344f",
            font=(self.display_font, 34, "bold"),
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            outer,
            text="Press Space to start or finish an episode. Press Backspace to delete the last saved episode. Press Esc to close the window.",
            bg="#f3efe7",
            fg="#5a6d81",
            font=(self.body_font, 18),
        )
        subtitle.pack(anchor="w", pady=(6, 18))

        camera_grid = tk.Frame(outer, bg="#f3efe7")
        camera_grid.pack(fill="both", expand=True)

        for column in range(self._camera_grid_columns()):
            camera_grid.grid_columnconfigure(column, weight=1, uniform="camera_column")
        for row in range(self._camera_grid_rows()):
            camera_grid.grid_rowconfigure(row, weight=1, uniform="camera_row")

        self.camera_panels = {}
        for index, key in enumerate(self.camera_keys):
            panel = tk.Frame(
                camera_grid,
                bg="#fffdf9",
                highlightbackground="#d8d0c2",
                highlightthickness=1,
                padx=10,
                pady=10,
            )
            row = index // self._camera_grid_columns()
            column = index % self._camera_grid_columns()
            panel.grid(row=row, column=column, sticky="nsew", padx=8, pady=8)

            title = tk.Label(
                panel,
                text=_camera_display_label(key),
                bg="#fffdf9",
                fg="#18344f",
                font=(self.display_font, 22, "bold"),
                pady=10,
            )
            title.pack(fill="x")

            image_label = tk.Label(panel, bg="#dfe5ea")
            image_label.pack(fill="both", expand=True, padx=4, pady=(2, 4))
            self.camera_panels[key] = image_label

        info_card = tk.Frame(outer, bg="#fffdf9", highlightbackground="#d8d0c2", highlightthickness=1)
        info_card.pack(fill="x", expand=False, pady=(18, 0))

        top_info = tk.Frame(info_card, bg="#fffdf9", padx=22, pady=20)
        top_info.pack(fill="x")

        self.status_var = tk.StringVar(value="Starting up the robot and camera pipelines...")
        self.episode_var = tk.StringVar(value="Episode 1 of 1")
        self.frames_var = tk.StringVar(value="Frames captured: 0 / 0")
        self.remaining_var = tk.StringVar(value="Episodes left: 0")
        self.timer_var = tk.StringVar(value="Time remaining: 0.0s")
        self.dataset_var = tk.StringVar(value=f"Dataset: {self.args.dataset_name}")

        status_label = tk.Label(
            top_info,
            textvariable=self.status_var,
            bg="#fffdf9",
            fg="#17324d",
            anchor="w",
            justify="left",
            font=(self.display_font, 24, "bold"),
        )
        status_label.pack(fill="x", pady=(0, 12))

        for text_var in [self.episode_var, self.frames_var, self.remaining_var, self.timer_var, self.dataset_var]:
            label = tk.Label(
                top_info,
                textvariable=text_var,
                bg="#fffdf9",
                fg="#24384e",
                anchor="w",
                justify="left",
                font=(self.body_font, 18),
            )
            label.pack(fill="x", pady=3)

        self.progress = ttk.Progressbar(info_card, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=22, pady=(0, 18))

        actions_row = tk.Frame(info_card, bg="#fffdf9", padx=22, pady=0)
        actions_row.pack(fill="x", pady=(0, 14))

        self.delete_button = tk.Button(
            actions_row,
            text="Delete Last Episode",
            command=self._on_delete_last_episode,
            bg="#fff3ed",
            fg="#8a2f21",
            activebackground="#f7dfd4",
            activeforeground="#7a261b",
            relief="flat",
            borderwidth=0,
            padx=18,
            pady=10,
            font=(self.body_font, 15, "bold"),
            cursor="hand2",
        )
        self.delete_button.pack(anchor="w")

        self.footer_var = tk.StringVar(
            value="Waiting for initialization..."
        )
        footer = tk.Label(
            info_card,
            textvariable=self.footer_var,
            bg="#fffdf9",
            fg="#6a7b8f",
            anchor="w",
            justify="left",
            font=(self.body_font, 16),
            padx=22,
            pady=0,
        )
        footer.pack(fill="x", pady=(0, 18))

        style = ttk.Style()
        style.theme_use(style.theme_use())
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#e4ddd1",
            background="#2d8f68",
            bordercolor="#e4ddd1",
            lightcolor="#2d8f68",
            darkcolor="#2d8f68",
            thickness=32,
        )

    def _initialize_backend(self):
        self.robot = PiperRobot(self.robot_config)
        self.robot.connect()

        self.dataset = self._load_dataset_for_recording()
        self.recorded_episodes = len(self.dataset.meta.episodes)

        if self.recorded_episodes >= self.args.num_episodes:
            self.state = "done"
            self._set_status(
                "Requested episode count is already available.",
                footer="Press Backspace to delete the last saved episode, or press Esc to close the window.",
            )
        else:
            self.state = "ready"
            self._set_status(
                "Warming up live preview...",
                footer="The window stays responsive while each camera delivers its first frame.",
            )
        self._refresh_counts()

    def _read_dataset_info(self) -> dict:
        info_path = self.args.dataset_root / "meta" / "info.json"
        with info_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _ensure_local_resume_metadata_files(self) -> dict:
        info = self._read_dataset_info()
        missing_paths = [
            self.args.dataset_root / relative_path
            for relative_path in REQUIRED_RESUME_METADATA_FILES
            if not (self.args.dataset_root / relative_path).exists()
        ]
        if not missing_paths:
            return info

        content_counters = (
            "total_episodes",
            "total_frames",
            "total_tasks",
            "total_videos",
            "total_chunks",
        )
        has_recorded_content = any(int(info.get(counter, 0) or 0) > 0 for counter in content_counters)
        if has_recorded_content:
            missing_labels = ", ".join(str(path.relative_to(self.args.dataset_root)) for path in missing_paths)
            raise RuntimeError(
                "RESUME=true found incomplete local dataset metadata. "
                f"Missing files: {missing_labels}. "
                "This dataset already reports recorded content, so the missing metadata should be repaired "
                "instead of recreated."
            )

        for path in missing_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
        return info

    def _camera_feature_keys_from_info(self, info: dict) -> set[str]:
        return {
            key
            for key, feature in info["features"].items()
            if feature["dtype"] in {"video", "image"}
        }

    def _available_camera_feature_keys(self) -> set[str]:
        return {f"observation.images.{name}" for name in self.robot.cameras}

    def _normalize_features_for_comparison(self, features: dict) -> dict:
        normalized = {}
        for key, feature in features.items():
            feature_copy = dict(feature)
            if "shape" in feature_copy:
                feature_copy["shape"] = tuple(feature_copy["shape"])
            if feature_copy.get("dtype") in {"video", "image"}:
                feature_copy["info"] = None
            normalized[key] = feature_copy
        return normalized

    def _validate_resume_camera_compatibility(self) -> None:
        if not self.args.resume or not self.args.dataset_root.exists():
            return

        info = self._ensure_local_resume_metadata_files()
        dataset_camera_keys = self._camera_feature_keys_from_info(info)
        available_camera_keys = self._available_camera_feature_keys()
        if dataset_camera_keys != available_camera_keys:
            expected = sorted(dataset_camera_keys)
            current = sorted(available_camera_keys)
            raise RuntimeError(
                "RESUME=true requires the same camera set as the existing dataset. "
                f"Dataset expects {expected}, but this run has {current}. "
                "Start a new dataset or reconnect the missing cameras."
            )

        existing_meta = LeRobotDatasetMetadata(self.args.dataset_name, self.args.dataset_root)
        if existing_meta.timestamp_mode != REAL_TIME_TIMESTAMP_MODE:
            raise RuntimeError(
                "RESUME=true requires an existing dataset recorded in real_time timestamp mode. "
                "Start a new dataset root for the updated recorder."
            )

        existing_features = self._normalize_features_for_comparison(existing_meta.features)
        current_features = self._normalize_features_for_comparison(
            get_features_from_robot(self.robot, use_videos=True)
        )
        if existing_features != current_features:
            raise RuntimeError(
                "RESUME=true requires the same dataset schema as the existing dataset. "
                "The current recorder writes next-state actions, ee pose, and real-time columns. "
                "Start a new dataset root for this schema, or migrate the existing dataset before resuming."
            )

    def _scalar_feature(self, value, dtype) -> np.ndarray:
        return np.array([value], dtype=dtype)

    def _reset_episode_recording_state(self) -> None:
        self.current_frame_count = 0
        self.episode_start_t = None
        self.pending_record_sample = None
        self.next_record_tick_deadline_t = None
        self.episode_first_real_timestamp_s = None
        self.episode_last_real_timestamp_s = None
        self.episode_first_wall_time_ns = None
        self.episode_last_wall_time_ns = None

    def _capture_record_sample(self) -> dict:
        observation = self.robot.capture_passive_record_observation()
        sample_time_s = time.perf_counter()
        return {
            "observation": observation,
            "elapsed_s": sample_time_s - self.episode_start_t,
            "wall_time_ns": time.time_ns(),
        }

    def _build_aligned_frame(self, current_sample: dict, next_sample: dict) -> dict:
        transition_dt_s = next_sample["elapsed_s"] - current_sample["elapsed_s"]
        transition_fps = 0.0 if transition_dt_s <= 0.0 else 1.0 / transition_dt_s

        if self.episode_first_real_timestamp_s is None:
            self.episode_first_real_timestamp_s = current_sample["elapsed_s"]
            self.episode_first_wall_time_ns = current_sample["wall_time_ns"]
        self.episode_last_real_timestamp_s = next_sample["elapsed_s"]
        self.episode_last_wall_time_ns = next_sample["wall_time_ns"]

        observation_timestamp_s = current_sample["elapsed_s"] - self.episode_first_real_timestamp_s
        action_timestamp_s = next_sample["elapsed_s"] - self.episode_first_real_timestamp_s

        return {
            **current_sample["observation"],
            "timestamp": self._scalar_feature(observation_timestamp_s, np.float32),
            "action": next_sample["observation"]["observation.state"].clone(),
            "action.ee_pose": next_sample["observation"]["observation.ee_pose"].clone(),
            "real_observation_timestamp_s": self._scalar_feature(observation_timestamp_s, np.float64),
            "real_action_timestamp_s": self._scalar_feature(action_timestamp_s, np.float64),
            "real_observation_wall_time_ns": self._scalar_feature(current_sample["wall_time_ns"], np.int64),
            "real_action_wall_time_ns": self._scalar_feature(next_sample["wall_time_ns"], np.int64),
            "real_transition_delta_s": self._scalar_feature(transition_dt_s, np.float64),
            "real_transition_fps": self._scalar_feature(transition_fps, np.float64),
            "task": self.args.single_task,
        }

    def _record_sample(self) -> None:
        current_sample = self._capture_record_sample()
        if self.pending_record_sample is None:
            self.pending_record_sample = current_sample
            return

        frame = self._build_aligned_frame(self.pending_record_sample, current_sample)
        self.dataset.add_frame(frame)
        self.pending_record_sample = current_sample
        self.current_frame_count += 1

    def _current_episode_metadata(self) -> dict | None:
        if (
            self.current_frame_count <= 0
            or self.episode_first_real_timestamp_s is None
            or self.episode_last_real_timestamp_s is None
            or self.episode_first_wall_time_ns is None
            or self.episode_last_wall_time_ns is None
        ):
            return None

        real_duration_s = self.episode_last_real_timestamp_s - self.episode_first_real_timestamp_s
        actual_fps = 0.0 if real_duration_s <= 0.0 else self.current_frame_count / real_duration_s
        return {
            "real_duration_s": float(real_duration_s),
            "actual_fps": float(actual_fps),
            "first_observation_time_s": float(self.episode_first_real_timestamp_s),
            "last_action_time_s": float(self.episode_last_real_timestamp_s),
            "first_observation_wall_time_ns": int(self.episode_first_wall_time_ns),
            "last_action_wall_time_ns": int(self.episode_last_wall_time_ns),
        }

    def _format_episode_summary(self, episode_metadata: dict | None) -> str:
        if not episode_metadata:
            return ""
        return (
            f"Last saved episode real duration: {episode_metadata['real_duration_s']:.3f}s. "
            f"Actual recorded FPS: {episode_metadata['actual_fps']:.3f}."
        )

    def _build_recording_dataset_from_existing_root(self) -> LeRobotDataset:
        self._ensure_local_resume_metadata_files()

        dataset = LeRobotDataset.__new__(LeRobotDataset)
        dataset.meta = LeRobotDatasetMetadata(self.args.dataset_name, self.args.dataset_root)
        dataset.repo_id = dataset.meta.repo_id
        dataset.root = dataset.meta.root
        dataset.revision = None
        dataset.tolerance_s = 1e-4
        dataset.image_writer = None
        next_episode_index = len(dataset.meta.episodes)
        dataset.episode_buffer = dataset.create_episode_buffer(episode_index=next_episode_index)
        dataset.episodes = None
        dataset.hf_dataset = dataset.create_hf_dataset()
        dataset.image_transforms = None
        dataset.delta_timestamps = None
        dataset.delta_indices = None
        dataset.episode_data_index = None
        dataset.video_backend = "pyav"
        return dataset

    def _repair_existing_dataset_if_needed(self) -> None:
        if not self.args.dataset_root.exists():
            self.dataset_repair_note = ""
            return

        result = repair_trailing_inconsistent_episodes(self.args.dataset_root)
        if result.repaired:
            deleted = ", ".join(str(index + 1) for index in result.deleted_episodes)
            self.dataset_repair_note = (
                f"Auto-repaired the dataset by trimming incomplete episode(s): {deleted}."
            )
        else:
            self.dataset_repair_note = ""

    def _load_dataset_from_disk(self) -> LeRobotDataset:
        self._repair_existing_dataset_if_needed()
        dataset = self._build_recording_dataset_from_existing_root()

        dataset.start_image_writer(
            num_processes=0,
            num_threads=4 * len(self.robot.cameras),
        )
        next_episode_index = len(dataset.meta.episodes)
        dataset.episode_buffer = dataset.create_episode_buffer(episode_index=next_episode_index)
        clear_episode_artifacts(dataset.root, dataset.meta.info, next_episode_index)
        return dataset

    def _load_dataset_for_recording(self) -> LeRobotDataset:
        if self.args.resume and self.args.dataset_root.exists():
            self._validate_resume_camera_compatibility()
            return self._load_dataset_from_disk()

        return LeRobotDataset.create(
            self.args.dataset_name,
            self.args.fps,
            root=self.args.dataset_root,
            robot=self.robot,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4 * len(self.robot.cameras),
            timestamp_mode=REAL_TIME_TIMESTAMP_MODE,
        )

    def _reload_existing_dataset(self):
        if self.dataset is not None:
            try:
                self.dataset.stop_image_writer()
            except Exception:
                pass

        self.dataset = self._load_dataset_from_disk()
        self.recorded_episodes = len(self.dataset.meta.episodes)

    def _can_delete_last_episode(self) -> bool:
        return self.state in {"ready", "done"} and self.recorded_episodes > 0

    def _count_ready_cameras(self) -> int:
        return sum(1 for camera in self.robot.cameras.values() if camera.color_image is not None)

    def _all_cameras_ready(self) -> bool:
        return self._count_ready_cameras() == len(self.robot.cameras)

    def _set_status(self, status: str, footer: str | None = None):
        self.status_var.set(status)
        if footer is not None:
            self.footer_var.set(footer)

    def _refresh_counts(self):
        total = self.args.num_episodes
        current = total if self.recorded_episodes >= total else self.recorded_episodes + 1
        self.episode_var.set(f"Episode {current} of {total}")
        target_frames = max(1, round(self.args.episode_time_s * self.args.fps))
        self.frames_var.set(f"Frames captured: {self.current_frame_count} / {target_frames}")
        self.remaining_var.set(f"Episodes left: {max(total - self.recorded_episodes, 0)}")

        if self.state == "recording" and self.episode_start_t is not None:
            elapsed = time.perf_counter() - self.episode_start_t
            time_left = max(self.args.episode_time_s - elapsed, 0.0)
            self.timer_var.set(f"Time remaining: {time_left:0.1f}s")
            progress = min(elapsed / self.args.episode_time_s, 1.0) * 100.0
            self.progress["value"] = progress
        else:
            self.timer_var.set(f"Time remaining: {self.args.episode_time_s:0.1f}s")
            self.progress["value"] = 0

        if self.delete_button is not None:
            self.delete_button.configure(state="normal" if self._can_delete_last_episode() else "disabled")

    def _schedule_tick(self):
        if self.state == "recording" and self.next_record_tick_deadline_t is not None:
            interval_s = max(self.next_record_tick_deadline_t - time.perf_counter(), 0.0)
            interval_ms = max(1, int(interval_s * 1000))
        else:
            interval_ms = max(1, int(1000 / self.args.fps))
        self.root.after(interval_ms, self._tick)

    def _refresh_preview_now(self) -> None:
        if self.robot is None or self.state not in {"ready", "recording", "done"}:
            return

        if self.state == "recording":
            observation = self.pending_record_sample["observation"] if self.pending_record_sample is not None else {}
            self._update_preview_from_observation(observation)
        else:
            self._update_preview_from_cameras()

    def _on_resize(self, _event=None):
        if self.resize_refresh_pending:
            return

        self.resize_refresh_pending = True

        def refresh():
            self.resize_refresh_pending = False
            self._refresh_preview_now()

        self.root.after(20, refresh)

    def _tick(self):
        if self.stop_requested:
            self._shutdown()
            return

        try:
            if self.state == "ready":
                self._update_preview_from_cameras()
                ready_count = self._count_ready_cameras()
                total_cameras = len(self.robot.cameras)
                if ready_count == total_cameras:
                    self._set_status(
                        f"Ready to record Episode {self.recorded_episodes + 1}.",
                        footer=self._append_missing_camera_note(
                            "All available camera feeds are live. Press Space to begin, or press Backspace to delete the last saved episode."
                        ),
                    )
                else:
                    self._set_status(
                        f"Camera preview is still warming up ({ready_count}/{total_cameras} ready).",
                        footer=self._append_missing_camera_note(
                            "Please wait until the available previews are live before you start recording."
                        ),
                    )
                if self.space_requested:
                    self.space_requested = False
                    if self._all_cameras_ready():
                        self._start_episode()
                    else:
                        self._set_status(
                            "The cameras are still warming up.",
                            footer="Press Space again after all available previews are live.",
                        )

            elif self.state == "recording":
                self._record_sample()
                preview_observation = (
                    self.pending_record_sample["observation"] if self.pending_record_sample is not None else {}
                )
                self._update_preview_from_observation(preview_observation)

                elapsed = time.perf_counter() - self.episode_start_t
                if self.space_requested or elapsed >= self.args.episode_time_s:
                    self.space_requested = False
                    self._finish_episode()

            elif self.state == "done":
                pass

        except Exception as exc:
            self.state = "error"
            self._set_status(
                f"Error: {exc}",
                footer="Please check the terminal output for the full traceback.",
            )
            raise

        if self.state == "recording" and self.next_record_tick_deadline_t is not None:
            next_deadline_t = self.next_record_tick_deadline_t + (1.0 / self.args.fps)
            self.next_record_tick_deadline_t = max(next_deadline_t, time.perf_counter())

        self._refresh_counts()
        self._schedule_tick()

    def _start_episode(self):
        if self.recorded_episodes >= self.args.num_episodes:
            self.state = "done"
            self._set_status("All requested episodes are complete.")
            return

        start_t = time.perf_counter()
        self.state = "recording"
        self.pending_record_sample = None
        self.next_record_tick_deadline_t = start_t
        self.current_frame_count = 0
        self.episode_start_t = start_t
        self.episode_first_real_timestamp_s = None
        self.episode_last_real_timestamp_s = None
        self.episode_first_wall_time_ns = None
        self.episode_last_wall_time_ns = None
        self._set_status(
            f"Recording Episode {self.recorded_episodes + 1}.",
            footer="Recording is live. Press Space if you want to finish this episode early.",
        )

    def _finish_episode(self):
        episode_metadata = self._current_episode_metadata()
        summary_note = self._format_episode_summary(episode_metadata)
        if self.current_frame_count > 0:
            self._set_status(
                f"Saving Episode {self.recorded_episodes + 1} with {self.current_frame_count} frames...",
                footer="Please wait while the videos and parquet files are written.",
            )
            self.root.update_idletasks()
            try:
                self.dataset.save_episode(episode_metadata=episode_metadata)
            except Exception as exc:
                if self._recover_from_failed_save(exc):
                    return
                raise
            self.recorded_episodes += 1
        else:
            self.dataset.clear_episode_buffer()

        self._reset_episode_recording_state()

        if self.recorded_episodes >= self.args.num_episodes:
            self.state = "done"
            self._set_status(
                "All requested episodes have been saved.",
                footer=(
                    "Press Backspace to delete the last saved episode, or press Esc to close the window."
                    + (f" {summary_note}" if summary_note else "")
                ),
            )
        else:
            self.state = "ready"
            self._set_status(
                f"Ready for Episode {self.recorded_episodes + 1}.",
                footer=(
                    "Reposition the scene, press Space to continue, or press Backspace to delete the last saved episode."
                    + (f" {summary_note}" if summary_note else "")
                ),
            )

    def _recover_from_failed_save(self, exc: Exception) -> bool:
        if not self.args.dataset_root.exists():
            return False

        failed_episode_number = self.recorded_episodes + 1
        self._set_status(
            f"Recovering from save error for Episode {failed_episode_number}...",
            footer="Discarding the incomplete episode and reloading the dataset so you can continue.",
        )
        self.root.update_idletasks()

        try:
            self._reload_existing_dataset()
        except Exception:
            return False

        self._reset_episode_recording_state()
        self.space_requested = False

        if self.recorded_episodes >= self.args.num_episodes:
            self.state = "done"
            self._set_status(
                f"Recovered from save error ({type(exc).__name__}).",
                footer=self._append_missing_camera_note(
                    "The incomplete episode was discarded and the dataset was reloaded. "
                    "Press Backspace to delete again, or press Esc to close."
                ),
            )
        else:
            self.state = "ready"
            self._set_status(
                f"Recovered from save error ({type(exc).__name__}). Ready for Episode {self.recorded_episodes + 1}.",
                footer=self._append_missing_camera_note(
                    "The incomplete episode was discarded and the dataset was reloaded. "
                    "Press Space to continue recording."
                ),
            )
        return True

    def _update_preview_from_cameras(self):
        for key, panel in self.camera_panels.items():
            camera = self.robot.cameras.get(key)
            if camera is None:
                missing_info = self.robot.unavailable_cameras.get(key, {})
                subdetail = None
                serial_number = missing_info.get("serial_number")
                if serial_number is not None:
                    subdetail = f"Configured serial {serial_number} is not detected."
                render = self._render_placeholder_image(
                    key,
                    "CAMERA NOT AVAILABLE",
                    subdetail=subdetail,
                )
                self._set_panel_image(key, render)
                continue

            frame = camera.color_image
            if frame is not None:
                self.latest_images[key] = frame

            render_frame = self.latest_images.get(key)
            if render_frame is None:
                render = self._render_placeholder_image(
                    key,
                    "WAITING FOR FIRST FRAME",
                    subdetail="Background reader is retrying." if camera.last_read_error is not None else None,
                )
            else:
                render = Image.fromarray(render_frame)

            self._set_panel_image(key, render)

    def _update_preview_from_observation(self, observation: dict):
        for key, panel in self.camera_panels.items():
            if key not in self.robot.cameras:
                missing_info = self.robot.unavailable_cameras.get(key, {})
                subdetail = None
                serial_number = missing_info.get("serial_number")
                if serial_number is not None:
                    subdetail = f"Configured serial {serial_number} is not detected."
                render = self._render_placeholder_image(
                    key,
                    "CAMERA NOT AVAILABLE",
                    subdetail=subdetail,
                )
                self._set_panel_image(key, render)
                continue

            obs_key = f"observation.images.{key}"
            if obs_key in observation:
                image = observation[obs_key]
                if hasattr(image, "cpu"):
                    image = image.cpu().numpy()
                self.latest_images[key] = image

            frame = self.latest_images.get(key)
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

            render = Image.fromarray(frame)
            self._set_panel_image(key, render)

    def _on_space(self, _event=None):
        if self.state in {"ready", "recording"}:
            self.space_requested = True

    def _on_delete_last_episode(self, _event=None):
        if not self._can_delete_last_episode():
            if self.state == "recording":
                self._set_status(
                    "Finish the current episode before deleting the previous one.",
                    footer="Backspace works only while the recorder is idle.",
                )
            else:
                self._set_status(
                    "No saved episode is available to delete yet.",
                    footer="Record at least one episode before using Backspace.",
                )
            return

        episode_number = self.recorded_episodes
        should_delete = messagebox.askyesno(
            "Delete last episode",
            f"Delete Episode {episode_number} from {self.args.dataset_name} and continue recording?",
        )
        if not should_delete:
            return

        self._set_status(
            f"Deleting Episode {episode_number}...",
            footer="Please wait while the dataset metadata and files are updated.",
        )
        self.root.update_idletasks()

        if self.dataset is not None:
            try:
                self.dataset.stop_image_writer()
            except Exception:
                pass

        result = delete_last_episode(self.args.dataset_root)
        self.recorded_episodes = result.remaining_episodes
        self.current_frame_count = 0
        self.episode_start_t = None
        self.space_requested = False
        self._reload_existing_dataset()

        if self.recorded_episodes >= self.args.num_episodes:
            self.state = "done"
            self._set_status(
                f"Episode {episode_number} deleted.",
                footer="Requested episode count is already available. Press Backspace to delete again, or press Esc to close the window.",
            )
        else:
            self.state = "ready"
            self._set_status(
                f"Episode {episode_number} deleted.",
                footer=f"Ready to record Episode {self.recorded_episodes + 1}. Press Space to continue.",
            )

    def _on_escape(self, _event=None):
        self.stop_requested = True

    def _on_close(self):
        self.stop_requested = True

    def _shutdown(self):
        try:
            if self.state == "recording" and self.current_frame_count > 0:
                self.dataset.save_episode()
                self.recorded_episodes += 1
        except Exception:
            pass

        try:
            if self.dataset is not None:
                self.dataset.stop_image_writer()
        except Exception:
            pass

        try:
            if self.robot is not None and self.robot.is_connected:
                self.robot.disconnect()
        except Exception:
            pass

        self.root.destroy()

    def run(self):
        self.root.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(description="GUI data collection for Piper passive recording.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--single-task", default="stack_bowls")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--episode-time-s", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--robot-cameras-json")
    parser.add_argument("--robot-config-json")
    return parser.parse_args()


def main():
    args = parse_args()
    app = PiperCollectionGUI(args)
    app.run()


if __name__ == "__main__":
    main()
