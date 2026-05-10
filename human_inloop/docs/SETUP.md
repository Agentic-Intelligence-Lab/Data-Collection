# Setup Guide

## Python

Use Python 3.11. The OpenPI runtime checked by this project expects Python 3.11 and will fail early on other Python versions.

Recommended local environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-collection.txt
```

For conda:

```bash
conda create -n piper-correction python=3.11
conda activate piper-correction
python -m pip install -r requirements-collection.txt
```

## CUDA And Torch

Install a `torch` / `torchvision` build that matches your NVIDIA driver and CUDA runtime. The requirements file gives a compatible version range, but GPU wheel selection is platform-specific. If the default pip install pulls a CPU wheel, reinstall torch from the official PyTorch CUDA index for your CUDA version.

Set `DEVICE=cpu` only for validation or debugging. Real collection with OpenPI policy inference is expected to use `DEVICE=cuda`.

## OpenPI

OpenPI is an external source checkout, not vendored in this repository.

Known reference for this workflow:

- Repository: `git@github.com:1939645507/openpi.git`
- Commit: `edfcb37eb2eaa0472627899edefae6af63850fe3`

The checkout assigned to `OPENPI_ROOT` must contain:

- `src/openpi`
- `packages/openpi-client/src`

Example:

```bash
git clone git@github.com:1939645507/openpi.git external/openpi
cd external/openpi
git checkout edfcb37eb2eaa0472627899edefae6af63850fe3
cd ../..
export OPENPI_ROOT=$PWD/external/openpi
```

If you keep OpenPI somewhere else, set `OPENPI_ROOT=/absolute/path/to/openpi`.

## Checkpoints

Set `CHECKPOINT_DIR` to a Pi0.5/OpenPI checkpoint directory. See `docs/EXTERNAL_ARTIFACTS.md` for the expected layout. Do not commit model weights directly to a normal GitHub repository.

For local runs, put machine-specific values in `.env.local`:

```bash
cp .env.example .env.local
# Edit OPENPI_ROOT, CHECKPOINT_DIR, COLLECTION_OUTPUT_ROOT, PIPER_CAN_NAME, and camera serials.
```

The main run script, UI launcher, and validator load `.env.local` automatically if it exists.

## RealSense

The collection path uses Intel RealSense cameras through `pyrealsense2`. You may need system packages, udev rules, firmware compatibility, and USB bandwidth checks depending on the machine. Confirm each camera serial before collection:

```bash
rs-enumerate-devices
```

Then update `.env.local` with the serials used by your rig.

## CAN, python-can, And Piper SDK

The active Piper arm is selected by `PIPER_CAN_NAME`, for example `can1`. Bring up the CAN interface before running collection. The exact bitrate depends on your Piper setup.

Example shape:

```bash
ip link show can1
# Configure and bring up the interface according to your hardware manual.
```

The Python dependencies include `python-can` and `piper_sdk`. `scripts/validate_environment.py` checks imports and whether the CAN interface exists, but it does not move the robot.

## UI Dependencies

The Web UI can be opened with Chrome/Chromium or `xdg-open`:

```bash
bash scripts/open_collection_ui.sh http://127.0.0.1:8765
```

Set `BROWSER_BIN=/path/to/browser` if auto-detection fails. `wmctrl` is optional and only used to keep the window always on top.

## Common Failures

- `OpenPI root does not exist`: set `OPENPI_ROOT`, or place the checkout at `external/openpi`.
- `Missing required OpenPI source path`: the checkout is wrong or incomplete; verify `src/openpi` and `packages/openpi-client/src`.
- `Checkpoint directory does not exist`: set `CHECKPOINT_DIR` to the directory containing `model.safetensors` and `metadata.pt`.
- `Python 3.11 expected`: activate a Python 3.11 environment or set `OPENPI_PYTHON_BIN`.
- `CAN interface not found`: bring up the interface named by `PIPER_CAN_NAME`.
- `pyrealsense2` import fails: install RealSense system support and a compatible wheel for your platform.
