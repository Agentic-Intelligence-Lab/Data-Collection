#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


INFO_PATH = Path("meta/info.json")
EPISODES_PATH = Path("meta/episodes.jsonl")
EPISODES_STATS_PATH = Path("meta/episodes_stats.jsonl")
TASKS_PATH = Path("meta/tasks.jsonl")


@dataclass
class DeleteEpisodeResult:
    deleted_episode_index: int
    deleted_episode_length: int
    remaining_episodes: int
    remaining_frames: int


@dataclass
class RepairDatasetResult:
    deleted_episodes: list[int]

    @property
    def repaired(self) -> bool:
        return bool(self.deleted_episodes)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _load_dataset_records(dataset_root: Path) -> tuple[dict, list[dict], list[dict]]:
    info = _load_json(dataset_root / INFO_PATH)
    episodes = _load_jsonl(dataset_root / EPISODES_PATH)
    episodes_stats = _load_jsonl(dataset_root / EPISODES_STATS_PATH)
    return info, episodes, episodes_stats


def _episode_indices(rows: list[dict]) -> list[int]:
    return [int(row["episode_index"]) for row in rows]


def _is_contiguous(indices: list[int]) -> bool:
    return indices == list(range(len(indices)))


def _video_keys(info: dict) -> list[str]:
    return [key for key, feature in info["features"].items() if feature["dtype"] == "video"]


def _camera_like_keys(info: dict) -> list[str]:
    return [
        key
        for key, feature in info["features"].items()
        if feature["dtype"] in {"video", "image"}
    ]


def _episode_chunk(info: dict, episode_index: int) -> int:
    chunk_size = max(int(info.get("chunks_size", 1000)), 1)
    return episode_index // chunk_size


def _episode_parquet_path(dataset_root: Path, info: dict, episode_index: int) -> Path:
    return dataset_root / info["data_path"].format(
        episode_chunk=_episode_chunk(info, episode_index),
        episode_index=episode_index,
    )


def _episode_video_paths(dataset_root: Path, info: dict, episode_index: int) -> list[Path]:
    video_path_template = info.get("video_path")
    if not video_path_template:
        return []

    chunk = _episode_chunk(info, episode_index)
    return [
        dataset_root / video_path_template.format(
            episode_chunk=chunk,
            video_key=video_key,
            episode_index=episode_index,
        )
        for video_key in _video_keys(info)
    ]


def _episode_image_dirs(dataset_root: Path, info: dict, episode_index: int) -> list[Path]:
    return [
        dataset_root / "images" / camera_key / f"episode_{episode_index:06d}"
        for camera_key in _camera_like_keys(info)
    ]


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _prune_empty_parents(path: Path, stop_at: Path) -> None:
    current = path
    stop_at = stop_at.resolve()

    while current.exists() and current.resolve() != stop_at:
        if not current.is_dir():
            current = current.parent
            continue
        try:
            next(current.iterdir())
            break
        except StopIteration:
            parent = current.parent
            current.rmdir()
            current = parent


def clear_episode_artifacts(dataset_root: Path | str, info: dict, episode_index: int) -> None:
    dataset_root = Path(dataset_root)

    parquet_path = _episode_parquet_path(dataset_root, info, episode_index)
    if parquet_path.exists():
        _remove_path(parquet_path)
        _prune_empty_parents(parquet_path.parent, dataset_root)

    for video_path in _episode_video_paths(dataset_root, info, episode_index):
        if video_path.exists():
            _remove_path(video_path)
            _prune_empty_parents(video_path.parent, dataset_root)

    for image_dir in _episode_image_dirs(dataset_root, info, episode_index):
        if image_dir.exists():
            _remove_path(image_dir)
            _prune_empty_parents(image_dir.parent, dataset_root)


def delete_last_episode(dataset_root: Path | str) -> DeleteEpisodeResult:
    dataset_root = Path(dataset_root)
    info_path = dataset_root / INFO_PATH
    episodes_path = dataset_root / EPISODES_PATH
    episodes_stats_path = dataset_root / EPISODES_STATS_PATH
    tasks_path = dataset_root / TASKS_PATH

    if not info_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {info_path}")

    info = _load_json(info_path)
    total_episodes = int(info.get("total_episodes", 0))
    if total_episodes <= 0:
        raise ValueError(f"No saved episode can be deleted in {dataset_root}.")

    episodes = _load_jsonl(episodes_path)
    if not episodes:
        raise ValueError(f"Episode metadata is empty: {episodes_path}")

    last_episode = episodes[-1]
    expected_episode_index = total_episodes - 1
    last_episode_index = int(last_episode["episode_index"])
    if last_episode_index != expected_episode_index:
        raise RuntimeError(
            "Refusing to delete the last episode because dataset metadata is not contiguous. "
            f"Expected episode_index {expected_episode_index}, got {last_episode_index}."
        )

    episode_stats = _load_jsonl(episodes_stats_path)
    if episode_stats:
        last_stats_episode_index = int(episode_stats[-1]["episode_index"])
        if last_stats_episode_index != expected_episode_index:
            raise RuntimeError(
                "Refusing to delete the last episode because episodes_stats metadata is not aligned. "
                f"Expected episode_index {expected_episode_index}, got {last_stats_episode_index}."
            )

    clear_episode_artifacts(dataset_root, info, expected_episode_index)

    remaining_episodes = episodes[:-1]
    remaining_episode_stats = episode_stats[:-1] if episode_stats else []
    remaining_frames = sum(int(episode["length"]) for episode in remaining_episodes)
    remaining_episode_count = len(remaining_episodes)
    remaining_tasks = _load_jsonl(tasks_path)

    info["total_episodes"] = remaining_episode_count
    info["total_frames"] = remaining_frames
    info["total_videos"] = remaining_episode_count * len(_video_keys(info))
    info["total_chunks"] = (
        0 if remaining_episode_count == 0 else _episode_chunk(info, remaining_episode_count - 1) + 1
    )
    info["total_tasks"] = len(remaining_tasks)
    info["splits"] = {} if remaining_episode_count == 0 else {"train": f"0:{remaining_episode_count}"}

    _write_json(info_path, info)
    _write_jsonl(episodes_path, remaining_episodes)
    if episodes_stats_path.exists() or remaining_episode_stats:
        _write_jsonl(episodes_stats_path, remaining_episode_stats)

    return DeleteEpisodeResult(
        deleted_episode_index=expected_episode_index,
        deleted_episode_length=int(last_episode["length"]),
        remaining_episodes=remaining_episode_count,
        remaining_frames=remaining_frames,
    )


def repair_trailing_inconsistent_episodes(dataset_root: Path | str) -> RepairDatasetResult:
    dataset_root = Path(dataset_root)
    deleted_episodes = []

    while True:
        info, episodes, episodes_stats = _load_dataset_records(dataset_root)
        total_episodes = int(info.get("total_episodes", 0))
        episode_indices = _episode_indices(episodes)
        stats_indices = _episode_indices(episodes_stats) if episodes_stats else []

        episodes_contiguous = _is_contiguous(episode_indices)
        stats_contiguous = (not episodes_stats) or _is_contiguous(stats_indices)
        counts_aligned = total_episodes == len(episodes) and (
            not episodes_stats or len(episodes_stats) == len(episodes)
        )

        if episodes_contiguous and stats_contiguous and counts_aligned:
            return RepairDatasetResult(deleted_episodes=deleted_episodes)

        if not episodes:
            raise RuntimeError(
                f"Dataset {dataset_root} is inconsistent and has no episode rows left to trim automatically."
            )

        last_episode_index = int(episodes[-1]["episode_index"])
        if last_episode_index != total_episodes - 1:
            raise RuntimeError(
                "Dataset metadata is inconsistent in a non-trailing way and cannot be auto-repaired safely. "
                f"total_episodes={total_episodes}, episode_indices={episode_indices}."
            )

        result = delete_last_episode(dataset_root)
        deleted_episodes.append(result.deleted_episode_index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete the last saved episode from a local LeRobot dataset.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    args = parser.parse_args()

    result = delete_last_episode(args.dataset_root)
    print(
        "Deleted episode "
        f"{result.deleted_episode_index} "
        f"({result.deleted_episode_length} frames). "
        f"Remaining episodes: {result.remaining_episodes}, "
        f"remaining frames: {result.remaining_frames}."
    )


if __name__ == "__main__":
    main()
