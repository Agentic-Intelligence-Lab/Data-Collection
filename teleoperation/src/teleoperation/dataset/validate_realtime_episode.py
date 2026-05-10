#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


INFO_PATH = Path("meta/info.json")
EPISODES_PATH = Path("meta/episodes.jsonl")

REQUIRED_COLUMNS = [
    "timestamp",
    "observation.state",
    "action",
    "observation.ee_pose",
    "action.ee_pose",
    "real_observation_timestamp_s",
    "real_action_timestamp_s",
    "real_transition_delta_s",
    "real_transition_fps",
]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _episode_chunk(info: dict, episode_index: int) -> int:
    chunk_size = max(int(info.get("chunks_size", 1000)), 1)
    return episode_index // chunk_size


def _episode_parquet_path(dataset_root: Path, info: dict, episode_index: int) -> Path:
    return dataset_root / info["data_path"].format(
        episode_chunk=_episode_chunk(info, episode_index),
        episode_index=episode_index,
    )


def _episode_video_paths(dataset_root: Path, info: dict, episode_index: int) -> dict[str, Path]:
    video_path_template = info.get("video_path")
    if not video_path_template:
        return {}

    chunk = _episode_chunk(info, episode_index)
    return {
        key: dataset_root / video_path_template.format(
            episode_chunk=chunk,
            video_key=key,
            episode_index=episode_index,
        )
        for key, feature in info["features"].items()
        if feature["dtype"] == "video"
    }


def _stack_column(table, column_name: str, dtype=np.float64) -> np.ndarray:
    return np.stack([np.asarray(x, dtype=dtype) for x in table[column_name].to_pylist()], axis=0)


def _load_video_pts(video_path: Path) -> tuple[np.ndarray, np.ndarray]:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "frame=best_effort_timestamp_time,pkt_duration_time",
            "-select_streams",
            "v:0",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    pts: list[float] = []
    durations: list[float] = []
    for line in probe.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        pts.append(float(parts[0]))
        durations.append(float(parts[1]) if len(parts) > 1 and parts[1] else 0.0)

    return np.asarray(pts, dtype=np.float64), np.asarray(durations, dtype=np.float64)


def _load_video_stream_info(video_path: Path) -> dict:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=duration,nb_frames,avg_frame_rate,r_frame_rate,time_base",
            "-select_streams",
            "v:0",
            "-of",
            "json",
            str(video_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return json.loads(probe.stdout)["streams"][0]


def _parse_fraction(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    num, denom = map(int, value.split("/"))
    if denom == 0:
        return None
    return num / denom


def _latest_episode_index(episodes: list[dict]) -> int:
    if not episodes:
        raise ValueError("No episodes found in dataset.")
    return max(int(row["episode_index"]) for row in episodes)


def validate_episode(
    dataset_root: Path,
    episode_index: int,
    *,
    timestamp_tol_s: float,
    duration_tol_s: float,
    alignment_tol: float,
) -> dict:
    info = _load_json(dataset_root / INFO_PATH)
    episodes = _load_jsonl(dataset_root / EPISODES_PATH)
    episodes_by_index = {int(row["episode_index"]): row for row in episodes}
    if episode_index not in episodes_by_index:
        raise ValueError(f"Episode {episode_index} not found in {dataset_root / EPISODES_PATH}")
    episode = episodes_by_index[episode_index]

    parquet_path = _episode_parquet_path(dataset_root, info, episode_index)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing parquet file: {parquet_path}")

    table = pq.read_table(parquet_path)
    missing_cols = [column for column in REQUIRED_COLUMNS if column not in table.column_names]

    state = _stack_column(table, "observation.state")
    action = _stack_column(table, "action")
    state_ee = _stack_column(table, "observation.ee_pose")
    action_ee = _stack_column(table, "action.ee_pose")

    timestamp = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    real_obs_timestamp = np.asarray(table["real_observation_timestamp_s"].to_pylist(), dtype=np.float64)
    real_action_timestamp = np.asarray(table["real_action_timestamp_s"].to_pylist(), dtype=np.float64)
    real_transition_dt = np.asarray(table["real_transition_delta_s"].to_pylist(), dtype=np.float64)
    real_transition_fps = np.asarray(table["real_transition_fps"].to_pylist(), dtype=np.float64)

    row_count = int(table.num_rows)
    episode_length = int(episode["length"])
    timestamp_monotonic = bool(np.all(np.diff(timestamp) > 0.0))
    timestamp_matches_real_obs = bool(np.allclose(timestamp, real_obs_timestamp, atol=timestamp_tol_s))

    next_state_mae = float(np.abs(action[:-1] - state[1:]).mean()) if row_count > 1 else 0.0
    next_ee_mae = float(np.abs(action_ee[:-1] - state_ee[1:]).mean()) if row_count > 1 else 0.0
    same_state_mae = float(np.abs(action - state).mean())
    same_ee_mae = float(np.abs(action_ee - state_ee).mean())

    expected_duration_s = float(
        episode.get(
            "real_duration_s",
            real_action_timestamp[-1] - real_obs_timestamp[0],
        )
    )
    expected_last_frame_duration_s = float(real_transition_dt[-1]) if row_count > 0 else 0.0

    per_video = {}
    video_paths = _episode_video_paths(dataset_root, info, episode_index)
    for key, video_path in sorted(video_paths.items()):
        exists = video_path.exists()
        entry = {
            "path": str(video_path),
            "exists": exists,
        }
        if exists:
            pts, pkt_durations = _load_video_pts(video_path)
            stream_info = _load_video_stream_info(video_path)
            max_abs_pts_error_s = (
                float(np.max(np.abs(pts - timestamp))) if len(pts) == len(timestamp) and len(pts) > 0 else None
            )
            duration_s = float(stream_info["duration"]) if stream_info.get("duration") else None
            avg_fps = _parse_fraction(stream_info.get("avg_frame_rate"))
            nominal_fps = _parse_fraction(stream_info.get("r_frame_rate"))
            nb_frames = int(stream_info["nb_frames"]) if stream_info.get("nb_frames") else None

            entry.update(
                {
                    "nb_frames": nb_frames,
                    "duration_s": duration_s,
                    "avg_fps": avg_fps,
                    "nominal_fps": nominal_fps,
                    "time_base": stream_info.get("time_base"),
                    "first_pts_s": float(pts[0]) if len(pts) else None,
                    "last_pts_s": float(pts[-1]) if len(pts) else None,
                    "last_pkt_duration_s": float(pkt_durations[-1]) if len(pkt_durations) else None,
                    "max_abs_pts_error_s": max_abs_pts_error_s,
                    "frame_count_matches_parquet": nb_frames == row_count,
                    "pts_count_matches_parquet": len(pts) == row_count,
                    "pts_match_parquet": max_abs_pts_error_s is not None and max_abs_pts_error_s <= timestamp_tol_s,
                    "duration_matches_episode": duration_s is not None
                    and abs(duration_s - expected_duration_s) <= duration_tol_s,
                    "last_pkt_duration_matches_transition": len(pkt_durations) > 0
                    and abs(float(pkt_durations[-1]) - expected_last_frame_duration_s) <= duration_tol_s,
                }
            )
        per_video[key] = entry

    checks = {
        "timestamp_mode_real_time": info.get("timestamp_mode") == "real_time",
        "required_columns_present": len(missing_cols) == 0,
        "episode_length_matches_parquet": row_count == episode_length,
        "timestamp_monotonic": timestamp_monotonic,
        "timestamp_matches_real_observation": timestamp_matches_real_obs,
        "action_matches_next_state": next_state_mae <= alignment_tol,
        "action_ee_matches_next_observation_ee_pose": next_ee_mae <= alignment_tol,
        "videos_exist": all(entry["exists"] for entry in per_video.values()),
        "videos_match_frame_count": all(entry.get("frame_count_matches_parquet", False) for entry in per_video.values()),
        "videos_pts_match_parquet": all(entry.get("pts_match_parquet", False) for entry in per_video.values()),
        "videos_duration_match_episode": all(
            entry.get("duration_matches_episode", False) for entry in per_video.values()
        ),
        "videos_last_duration_match_transition": all(
            entry.get("last_pkt_duration_matches_transition", False) for entry in per_video.values()
        ),
    }

    return {
        "dataset_root": str(dataset_root),
        "episode_index": episode_index,
        "overall_pass": all(checks.values()),
        "checks": checks,
        "episode_summary": {
            "row_count": row_count,
            "episode_length": episode_length,
            "real_duration_s": expected_duration_s,
            "actual_fps": episode.get("actual_fps"),
            "timestamp_first_s": float(timestamp[0]) if row_count else None,
            "timestamp_last_s": float(timestamp[-1]) if row_count else None,
            "mean_transition_dt_s": float(real_transition_dt.mean()) if row_count else None,
            "median_transition_dt_s": float(np.median(real_transition_dt)) if row_count else None,
            "mean_transition_fps": float(real_transition_fps.mean()) if row_count else None,
            "median_transition_fps": float(np.median(real_transition_fps)) if row_count else None,
            "same_frame_state_action_mae": same_state_mae,
            "next_frame_state_action_mae": next_state_mae,
            "same_frame_ee_mae": same_ee_mae,
            "next_frame_ee_mae": next_ee_mae,
        },
        "missing_columns": missing_cols,
        "videos": per_video,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a real_time LeRobot episode: parquet schema, next-state action alignment, ee pose, and video PTS."
    )
    parser.add_argument("dataset_root", type=Path, help="Path to dataset root.")
    parser.add_argument("--episode-index", type=int, default=None, help="Episode index to validate. Defaults to latest.")
    parser.add_argument("--timestamp-tol-s", type=float, default=1e-3, help="Tolerance for timestamp/PTS comparisons.")
    parser.add_argument("--duration-tol-s", type=float, default=5e-3, help="Tolerance for duration comparisons.")
    parser.add_argument("--alignment-tol", type=float, default=1e-8, help="Tolerance for next-state alignment checks.")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    info_path = dataset_root / INFO_PATH
    episodes_path = dataset_root / EPISODES_PATH
    if not info_path.exists():
        raise FileNotFoundError(f"Missing dataset info file: {info_path}")
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing episodes file: {episodes_path}")

    episodes = _load_jsonl(episodes_path)
    episode_index = args.episode_index if args.episode_index is not None else _latest_episode_index(episodes)
    report = validate_episode(
        dataset_root,
        episode_index,
        timestamp_tol_s=args.timestamp_tol_s,
        duration_tol_s=args.duration_tol_s,
        alignment_tol=args.alignment_tol,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["overall_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
