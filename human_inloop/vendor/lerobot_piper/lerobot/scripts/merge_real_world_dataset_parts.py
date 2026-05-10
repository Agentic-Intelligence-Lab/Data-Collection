#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Merge split LeRobot dataset parts into one flat dataset directory.

This is useful when one person/task directory contains several smaller LeRobot
datasets such as:

    pick_up_banana_and_place_it_into_the_basket_100/
      pick_up_banana_and_place_it_into_the_basket1/
      pick_up_banana_and_place_it_into_the_basket2/
      pick_up_banana_and_place_it_into_the_basket3/
      pick_up_banana_and_place_it_into_the_basket4/

After merging, the parent directory can directly contain:

    pick_up_banana_and_place_it_into_the_basket_100/
      data/
      meta/
      videos/

The script keeps the source folders by default. Add `--remove-source-dirs` only
after you are happy with the merged result.

Examples:

Dry run:
    conda run -n lerobot python lerobot/scripts/merge_real_world_dataset_parts.py \
        --parent-dir /path/to/real_world_data/pick_up_banana_and_place_it_into_the_basket_100 \
        --dry-run \
        --expected-total-episodes 100

Write merged data into the parent folder and then remove the source parts:
    conda run -n lerobot python lerobot/scripts/merge_real_world_dataset_parts.py \
        --parent-dir /path/to/real_world_data/pick_up_banana_and_place_it_into_the_basket_100 \
        --expected-total-episodes 100 \
        --remove-source-dirs
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REQUIRED_META_FILES = (
    Path("meta/info.json"),
    Path("meta/episodes.jsonl"),
    Path("meta/episodes_stats.jsonl"),
    Path("meta/tasks.jsonl"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent-dir",
        type=Path,
        required=True,
        help="Directory that contains several split LeRobot datasets to be merged.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Where to write the merged dataset. Defaults to --parent-dir, which will create "
            "`data/meta/videos` directly under the parent directory."
        ),
    )
    parser.add_argument(
        "--expected-total-episodes",
        type=int,
        default=None,
        help="Optional safety check. If set, the merged episode count must match this value.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only inspect and print the merge plan without writing files.",
    )
    parser.add_argument(
        "--remove-source-dirs",
        action="store_true",
        help="Delete the original split subdirectories after a successful merge.",
    )
    return parser.parse_args()


def natural_sort_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False)
        handle.write("\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def is_split_dataset_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "data").is_dir():
        return False
    for rel_path in REQUIRED_META_FILES:
        if not (path / rel_path).is_file():
            return False
    return True


def discover_source_dirs(parent_dir: Path) -> list[Path]:
    children = [path for path in parent_dir.iterdir() if is_split_dataset_dir(path)]
    children.sort(key=lambda path: natural_sort_key(path.name))
    return children


def info_signature(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "codebase_version": info["codebase_version"],
        "robot_type": info["robot_type"],
        "chunks_size": info["chunks_size"],
        "fps": info["fps"],
        "data_path": info["data_path"],
        "video_path": info["video_path"],
        "features": info["features"],
    }


def get_video_keys(info: dict[str, Any]) -> list[str]:
    return [
        name
        for name, feature in info["features"].items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]


def load_child_metadata(child_dir: Path) -> dict[str, Any]:
    info = load_json(child_dir / "meta/info.json")
    episodes = sorted(load_jsonl(child_dir / "meta/episodes.jsonl"), key=lambda item: item["episode_index"])
    episode_stats = sorted(
        load_jsonl(child_dir / "meta/episodes_stats.jsonl"),
        key=lambda item: item["episode_index"],
    )
    tasks = load_jsonl(child_dir / "meta/tasks.jsonl")

    if len(episodes) != len(episode_stats):
        raise ValueError(
            f"{child_dir}: episodes.jsonl has {len(episodes)} rows but episodes_stats.jsonl has "
            f"{len(episode_stats)} rows."
        )

    expected_episode_indices = list(range(len(episodes)))
    actual_episode_indices = [row["episode_index"] for row in episodes]
    if actual_episode_indices != expected_episode_indices:
        raise ValueError(
            f"{child_dir}: expected contiguous episode indices {expected_episode_indices[:5]}..., "
            f"got {actual_episode_indices[:5]}..."
        )

    stats_episode_indices = [row["episode_index"] for row in episode_stats]
    if stats_episode_indices != expected_episode_indices:
        raise ValueError(
            f"{child_dir}: episode_stats indices do not match episodes.jsonl indices."
        )

    return {
        "dir": child_dir,
        "info": info,
        "episodes": episodes,
        "episode_stats": episode_stats,
        "tasks": tasks,
    }


def validate_children(children: list[dict[str, Any]]) -> None:
    if not children:
        raise ValueError("No split dataset directories were found.")

    ref_info = info_signature(children[0]["info"])
    ref_tasks = children[0]["tasks"]

    for child in children[1:]:
        if info_signature(child["info"]) != ref_info:
            raise ValueError(
                f"{child['dir']}: dataset structure differs from {children[0]['dir']}. "
                "Only datasets with the same feature layout can be merged automatically."
            )
        if child["tasks"] != ref_tasks:
            raise ValueError(
                f"{child['dir']}: tasks.jsonl differs from {children[0]['dir']}. "
                "This script currently expects the same task mapping in every split."
            )


def format_episode_path(base_dir: Path, info: dict[str, Any], episode_index: int) -> Path:
    episode_chunk = episode_index // info["chunks_size"]
    return base_dir / info["data_path"].format(episode_chunk=episode_chunk, episode_index=episode_index)


def format_video_path(base_dir: Path, info: dict[str, Any], episode_index: int, video_key: str) -> Path:
    episode_chunk = episode_index // info["chunks_size"]
    return base_dir / info["video_path"].format(
        episode_chunk=episode_chunk,
        video_key=video_key,
        episode_index=episode_index,
    )


def shift_nested_numbers(value: Any, offset: int | float) -> Any:
    if isinstance(value, list):
        return [shift_nested_numbers(item, offset) for item in value]
    return value + offset


def shift_scalar_stats(stats_block: dict[str, Any], offset: int) -> dict[str, Any]:
    shifted = copy.deepcopy(stats_block)
    for field_name in ("min", "max", "mean"):
        if field_name in shifted:
            shifted[field_name] = shift_nested_numbers(shifted[field_name], offset)
    return shifted


def remap_episode_stats(stats: dict[str, Any], episode_offset: int, frame_offset: int) -> dict[str, Any]:
    remapped = copy.deepcopy(stats)
    if "episode_index" in remapped:
        remapped["episode_index"] = shift_scalar_stats(remapped["episode_index"], episode_offset)
    if "index" in remapped:
        remapped["index"] = shift_scalar_stats(remapped["index"], frame_offset)
    return remapped


def set_constant_column(table: pa.Table, column_name: str, value: int) -> pa.Table:
    column_index = table.schema.get_field_index(column_name)
    if column_index < 0:
        raise ValueError(f"Missing required parquet column: {column_name}")
    field = table.schema.field(column_name)
    values = pa.array([value] * table.num_rows, type=field.type)
    return table.set_column(column_index, field, values)


def shift_index_column(table: pa.Table, frame_offset: int) -> pa.Table:
    column_index = table.schema.get_field_index("index")
    if column_index < 0:
        raise ValueError("Missing required parquet column: index")
    field = table.schema.field("index")
    shifted_values = pc.add(table["index"], pa.scalar(frame_offset, type=field.type))
    return table.set_column(column_index, field, shifted_values)


def prepare_stage_dir(parent_dir: Path, output_dir: Path) -> Path:
    if output_dir == parent_dir:
        stage_dir = parent_dir / ".merge_staging"
        final_targets = [parent_dir / name for name in ("data", "meta", "videos")]
        existing_targets = [path for path in final_targets if path.exists()]
        if existing_targets:
            names = ", ".join(str(path) for path in existing_targets)
            raise ValueError(
                f"Refusing to merge into {parent_dir} because these targets already exist: {names}"
            )
    else:
        if output_dir.exists():
            raise ValueError(f"Output directory already exists: {output_dir}")
        stage_dir = output_dir.parent / f".{output_dir.name}.merge_staging"

    if stage_dir.exists():
        raise ValueError(f"Temporary staging directory already exists: {stage_dir}")

    return stage_dir


def finalize_stage_dir(parent_dir: Path, output_dir: Path, stage_dir: Path) -> None:
    if output_dir == parent_dir:
        for name in ("data", "meta", "videos"):
            source = stage_dir / name
            if source.exists():
                shutil.move(str(source), str(parent_dir / name))
        stage_dir.rmdir()
    else:
        shutil.move(str(stage_dir), str(output_dir))


def validate_output_location(parent_dir: Path, output_dir: Path, children: list[dict[str, Any]]) -> None:
    if output_dir == parent_dir:
        return

    for child in children:
        child_dir = child["dir"]
        if output_dir == child_dir or child_dir in output_dir.parents:
            raise ValueError(
                f"Output directory {output_dir} cannot be the same as or inside source directory {child_dir}."
            )


def print_plan(
    parent_dir: Path,
    output_dir: Path,
    children: list[dict[str, Any]],
    expected_total_episodes: int | None,
) -> None:
    total_episodes = sum(len(child["episodes"]) for child in children)
    total_frames = sum(child["info"]["total_frames"] for child in children)
    print(f"Parent directory: {parent_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Detected {len(children)} source folders:")
    for child in children:
        print(
            f"  - {child['dir'].name}: episodes={len(child['episodes'])}, "
            f"frames={child['info']['total_frames']}"
        )
    print(f"Planned merged episodes: {total_episodes}")
    print(f"Planned merged frames: {total_frames}")
    if expected_total_episodes is not None:
        print(f"Expected merged episodes: {expected_total_episodes}")


def merge_datasets(
    parent_dir: Path,
    output_dir: Path,
    children: list[dict[str, Any]],
    remove_source_dirs: bool,
) -> None:
    stage_dir = prepare_stage_dir(parent_dir, output_dir)
    stage_dir.mkdir(parents=True, exist_ok=False)

    template_info = copy.deepcopy(children[0]["info"])
    tasks = copy.deepcopy(children[0]["tasks"])
    video_keys = get_video_keys(template_info)

    merged_episodes: list[dict[str, Any]] = []
    merged_episode_stats: list[dict[str, Any]] = []
    copied_video_files = 0
    total_frames = 0
    total_episodes = 0

    try:
        for child in children:
            child_dir = child["dir"]
            child_info = child["info"]
            child_episode_offset = total_episodes
            child_frame_offset = total_frames
            child_frames_from_parquet = 0

            episode_stats_by_index = {
                row["episode_index"]: row for row in child["episode_stats"]
            }

            for episode in child["episodes"]:
                source_episode_index = episode["episode_index"]
                merged_episode_index = child_episode_offset + source_episode_index

                source_parquet = format_episode_path(child_dir, child_info, source_episode_index)
                target_parquet = format_episode_path(stage_dir, template_info, merged_episode_index)
                target_parquet.parent.mkdir(parents=True, exist_ok=True)

                table = pq.read_table(source_parquet)
                table = set_constant_column(table, "episode_index", merged_episode_index)
                table = shift_index_column(table, child_frame_offset)
                pq.write_table(table, target_parquet)

                if table.num_rows != episode["length"]:
                    raise ValueError(
                        f"{source_parquet}: parquet rows ({table.num_rows}) do not match "
                        f"episodes.jsonl length ({episode['length']})."
                    )
                child_frames_from_parquet += table.num_rows

                merged_episodes.append(
                    {
                        **copy.deepcopy(episode),
                        "episode_index": merged_episode_index,
                    }
                )

                stats_row = copy.deepcopy(episode_stats_by_index[source_episode_index])
                stats_row["episode_index"] = merged_episode_index
                stats_row["stats"] = remap_episode_stats(
                    stats_row["stats"],
                    episode_offset=child_episode_offset,
                    frame_offset=child_frame_offset,
                )
                merged_episode_stats.append(stats_row)

                for video_key in video_keys:
                    source_video = format_video_path(child_dir, child_info, source_episode_index, video_key)
                    target_video = format_video_path(stage_dir, template_info, merged_episode_index, video_key)
                    target_video.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_video, target_video)
                    copied_video_files += 1

            if child_frames_from_parquet != child_info["total_frames"]:
                raise ValueError(
                    f"{child_dir}: parquet frame total ({child_frames_from_parquet}) does not match "
                    f"info.json total_frames ({child_info['total_frames']})."
                )

            total_episodes += len(child["episodes"])
            total_frames += child_frames_from_parquet

        merged_info = copy.deepcopy(template_info)
        merged_info["total_episodes"] = total_episodes
        merged_info["total_frames"] = total_frames
        merged_info["total_tasks"] = len(tasks)
        merged_info["total_videos"] = copied_video_files
        merged_info["total_chunks"] = math.ceil(total_episodes / merged_info["chunks_size"]) if total_episodes else 0
        merged_info["splits"] = {"train": f"0:{total_episodes}"}

        write_json(stage_dir / "meta/info.json", merged_info)
        write_jsonl(stage_dir / "meta/tasks.jsonl", tasks)
        write_jsonl(stage_dir / "meta/episodes.jsonl", merged_episodes)
        write_jsonl(stage_dir / "meta/episodes_stats.jsonl", merged_episode_stats)

        finalize_stage_dir(parent_dir, output_dir, stage_dir)

        if remove_source_dirs:
            for child in children:
                shutil.rmtree(child["dir"])

    except Exception:
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()

    parent_dir = args.parent_dir.expanduser().resolve()
    output_dir = (args.output_dir or args.parent_dir).expanduser().resolve()

    if not parent_dir.is_dir():
        raise ValueError(f"Parent directory does not exist: {parent_dir}")

    children = [load_child_metadata(child_dir) for child_dir in discover_source_dirs(parent_dir)]
    validate_children(children)
    validate_output_location(parent_dir, output_dir, children)

    total_episodes = sum(len(child["episodes"]) for child in children)
    if args.expected_total_episodes is not None and total_episodes != args.expected_total_episodes:
        raise ValueError(
            f"Expected {args.expected_total_episodes} merged episodes, but found {total_episodes} "
            f"across {len(children)} source folders."
        )

    print_plan(
        parent_dir=parent_dir,
        output_dir=output_dir,
        children=children,
        expected_total_episodes=args.expected_total_episodes,
    )

    if args.dry_run:
        print("Dry run only. No files were written.")
        return

    merge_datasets(
        parent_dir=parent_dir,
        output_dir=output_dir,
        children=children,
        remove_source_dirs=args.remove_source_dirs,
    )

    print("Merge finished successfully.")
    if args.remove_source_dirs:
        print("Source split directories were removed.")
    else:
        print("Source split directories were kept.")


if __name__ == "__main__":
    main()
