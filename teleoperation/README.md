# Piper Teleoperation Data Collector

Standalone Piper passive teleoperation data collection project with a Tk GUI, 4 Intel RealSense cameras, CAN-based Piper state feedback, and LeRobot-compatible dataset output.

The migrated entry point is:

```bash
cd teleoperation
bash scripts/collect_data.sh
```

## What It Does

- Starts a GUI for passive Piper data collection.
- Records dual-arm Piper joint state and end-effector pose from CAN feedback.
- Records 4 RealSense RGB streams configured in `configs/camera_mapping.yaml`.
- Writes a local LeRobot-style dataset with parquet metadata and mp4 videos.
- Stores real-time timestamps and next-state action alignment columns.
- Supports deleting the last saved episode from the GUI.

## Hardware

- Agilex Piper arms available through CAN interfaces, default `can0` and `can1`.
- 4 Intel RealSense cameras, default keys:
  - `head`
  - `left_wrist`
  - `right_wrist`
  - `front_view`
- Linux tools: `ffmpeg`, `ffprobe`, `ip`, `ethtool`, `can-utils`.

## Environment

Recommended workflow, matching the existing machine setup:

```bash
conda activate lerobot
cd teleoperation
pip install -r requirements.txt
python tools/check_env.py
```

System packages usually needed:

```bash
sudo apt update
sudo apt install ffmpeg ethtool can-utils
```

RealSense also requires Intel librealsense support on the host. Piper control requires `piper-sdk`; this project does not vendor those third-party packages.

## Configuration

Main run config:

```text
configs/piper_4cam.yaml
```

Camera serials and image settings:

```text
configs/camera_mapping.yaml
```

Piper CAN and motor mapping:

```text
configs/robot_config.yaml
```

To run with a different config:

```bash
bash scripts/collect_data.sh --config configs/piper_4cam.yaml
```

## Dry Run

Dry-run does not activate CAN, does not connect the Piper SDK, and does not move the robot:

```bash
bash scripts/collect_data.sh --dry-run
```

It checks config loading, core imports, and RealSense enumeration when `pyrealsense2` is available.

## Data Output

Default dataset path is controlled by `configs/piper_4cam.yaml`:

```text
data/place_plate_20260507
```

The dataset follows the LeRobot v2.1 local layout:

```text
meta/info.json
meta/tasks.jsonl
meta/episodes.jsonl
meta/episodes_stats.jsonl
data/chunk-000/episode_000000.parquet
videos/chunk-000/observation.images.<camera_key>/episode_000000.mp4
```
