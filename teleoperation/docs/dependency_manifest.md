# Dependency Manifest

Source root used for migration: the original local `lerobot-piper` workspace.

The old project was treated as read-only. Files below were copied with `cp`; imports and paths were adjusted only in the new project.

| New file | Source file | Why it is needed |
| --- | --- | --- |
| `scripts/collect_data.sh` | `piper_scripts/collect_data_gui_4cam.sh` | Original launch flow: config validation, camera preflight, CAN activation, GUI launch, post-run validation. Renamed and rewritten as the clean public entry point. |
| `scripts/can_activate.sh` | `piper_scripts/can_activate.sh` | Required helper for bringing CAN interfaces up before collection. |
| `src/teleoperation/gui/collect_data_gui.py` | `piper_scripts/collect_data_gui.py` | Main Tk GUI and recording loop. |
| `src/teleoperation/dataset/episode_cleanup.py` | `piper_scripts/delete_last_episode.py` | GUI delete-last-episode and incomplete episode repair support. |
| `src/teleoperation/dataset/validate_realtime_episode.py` | `piper_scripts/validate_realtime_episode.py` | Optional post-run validation of parquet/video/timestamp alignment. |
| `src/teleoperation/dataset/lerobot_dataset.py` | `lerobot/common/datasets/lerobot_dataset.py` | Local LeRobot-compatible dataset writer used by the GUI. |
| `src/teleoperation/dataset/utils.py` | `lerobot/common/datasets/utils.py` | Dataset metadata, feature conversion, timestamp checks, and JSONL utilities. |
| `src/teleoperation/dataset/compute_stats.py` | `lerobot/common/datasets/compute_stats.py` | Episode statistics written into dataset metadata. |
| `src/teleoperation/dataset/image_writer.py` | `lerobot/common/datasets/image_writer.py` | Async image frame writing before mp4 encoding. |
| `src/teleoperation/dataset/video_utils.py` | `lerobot/common/datasets/video_utils.py` | ffmpeg/PyAV video encoding and video metadata probing. |
| `src/teleoperation/dataset/backward_compatibility.py` | `lerobot/common/datasets/backward_compatibility.py` | Dataset version compatibility errors used by dataset loading. |
| `src/teleoperation/dataset/card_template.md` | `lerobot/common/datasets/card_template.md` | Package data for optional dataset card generation. |
| `src/teleoperation/dataset/constants.py` | `lerobot/common/constants.py` | LeRobot cache and dataset constants used by the dataset module. |
| `src/teleoperation/cameras/realsense.py` | `lerobot/common/robot_devices/cameras/intelrealsense.py` | RealSense discovery, connection, async frame capture, and rotation handling. |
| `src/teleoperation/cameras/factory.py` | `lerobot/common/robot_devices/cameras/utils.py` | Camera factory, pruned to RealSense only. |
| `src/teleoperation/cameras/configs.py` | `lerobot/common/robot_devices/cameras/configs.py` | Minimal RealSense camera config dataclass derived from the original. |
| `src/teleoperation/robot/piper.py` | `lerobot/common/robot_devices/robots/piper.py` | Piper robot abstraction used by the GUI recorder. Kept passive recording path. |
| `src/teleoperation/robot/piper_motor.py` | `lerobot/common/robot_devices/motors/piper.py` | Piper SDK wrapper for CAN feedback and joint/EE pose reads. |
| `src/teleoperation/robot/motor_factory.py` | `lerobot/common/robot_devices/motors/utils.py` | Motor bus factory, pruned to Piper only. |
| `src/teleoperation/robot/motor_config.py` | `lerobot/common/robot_devices/motors/configs.py` | Minimal Piper motor config dataclass derived from the original. |
| `src/teleoperation/robot/config.py` | `lerobot/common/robot_devices/robots/configs.py` | Minimal Piper robot config derived from the original Piper section. |
| `src/teleoperation/utils/robot_devices.py` | `lerobot/common/robot_devices/utils.py` | Device connection errors and timing helper used by cameras and robot code. |
| `src/teleoperation/utils/common.py` | `lerobot/common/utils/utils.py` | Timestamp and dtype utilities used by dataset/camera code. |
| `src/teleoperation/utils/types.py` | `lerobot/configs/types.py` | Dataset feature typing helpers used by dataset utilities. |

New files without a direct old source:

| New file | Purpose |
| --- | --- |
| `configs/piper_4cam.yaml` | Clean top-level run config for dataset, runtime, and referenced hardware configs. |
| `configs/camera_mapping.yaml` | 4-camera RealSense mapping and serial numbers. |
| `configs/robot_config.yaml` | Piper passive recording CAN/motor mapping. |
| `src/teleoperation/recorder/config.py` | YAML config loader and shell export generator for `scripts/collect_data.sh`. |
| `src/teleoperation/robot/base.py` | Minimal `Robot` protocol for dataset type hints. |
| `tools/check_env.py` | Non-invasive dependency and system command checker. |
| `tools/dry_run_collect.py` | Non-invasive config/import/camera dry-run. |
| `README.md`, `requirements.txt`, `pyproject.toml`, `.gitignore` | Open-source project metadata and usage documentation. |
