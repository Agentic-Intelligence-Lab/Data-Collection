# Migration Report

## Summary

The migration started from the original Piper collection launcher and produced this independent `teleoperation` module.

The old project was not modified. The new public entry point is:

```bash
bash scripts/collect_data.sh
```

## Main Changes

- Renamed the launcher from `collect_data_gui_4cam.sh` to `scripts/collect_data.sh`.
- Moved hardware-specific settings out of the script into YAML configs.
- Replaced old source-tree `PYTHONPATH` behavior with `PYTHONPATH=<new_project>/src`.
- Moved GUI code to `src/teleoperation/gui/collect_data_gui.py`.
- Moved RealSense code to `src/teleoperation/cameras/`.
- Moved Piper robot and motor code to `src/teleoperation/robot/`.
- Moved LeRobot-compatible dataset writing code to `src/teleoperation/dataset/`.
- Added `tools/dry_run_collect.py` so validation can avoid CAN activation and Piper SDK connection.

## Import Adjustments

Old imports such as:

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.robots.piper import PiperRobot
from delete_last_episode import delete_last_episode
```

were changed to:

```python
from teleoperation.dataset.lerobot_dataset import LeRobotDataset
from teleoperation.robot.piper import PiperRobot
from teleoperation.dataset.episode_cleanup import delete_last_episode
```

Dataset resource loading was changed from `lerobot.common.datasets` to `teleoperation.dataset`.

## Pruned Local Code

- Did not copy unrelated `piper_scripts` inference, OpenPI, model serving, debug, or experiment scripts.
- Did not copy LeRobot tests, examples, training code, policies, environments, or unrelated robot adapters.
- Pruned camera factory support to Intel RealSense only.
- Pruned motor factory support to Piper only.
- The standalone Piper robot class now supports the passive recording path used by this collector.

## Third-Party Dependencies

Third-party libraries were not vendored. They are declared in `requirements.txt` and `pyproject.toml`, including:

- `piper-sdk`
- `pyrealsense2`
- `torch`
- `torchvision`
- `datasets`
- `pyarrow`
- `av`
- `Pillow`
- `numpy`
- `PyYAML`
- `jsonlines`
- `huggingface-hub`

System tools such as `ffmpeg`, `ffprobe`, `ethtool`, and `can-utils` are documented in the README.

## Validation Notes

Validation was run in the `lerobot` conda environment because the default shell Python does not contain the robotics/scientific dependencies.

Checks performed:

- `bash -n scripts/collect_data.sh`
- `bash -n scripts/can_activate.sh`
- `python -m compileall src`
- `tools/dry_run_collect.py --json`
- `rg` for stale old-project absolute paths
- `find`/`rg` checks for copied datasets, caches, logs, and weights

Dry-run did not activate CAN, did not connect the Piper SDK, and did not move the robot.
