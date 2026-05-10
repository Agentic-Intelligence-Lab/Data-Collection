# Piper Correction Collection

Interactive correction / DAgger-style data collection for AgileX Piper with OpenPI policies.

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)
![Hardware](https://img.shields.io/badge/hardware-AgileX%20Piper-orange)
![Policy](https://img.shields.io/badge/policy-OpenPI%2FPi0.5-purple)

## Overview

`piper-correction-collection` is a robotics data-collection workflow for running an OpenPI/Pi0.5 policy on an AgileX Piper arm, pausing policy rollout when needed, recording human corrective control, and saving LeRobot-style episodes with intervention metadata. It is intended for robotics and embodied AI researchers who need a reproducible interactive correction loop rather than a one-shot teleoperation recorder or a full policy-training framework.

The project packages the Piper collection loop, an OpenPI policy runtime adapter, a lightweight browser UI launcher, fail-fast environment checks, and setup documentation for external artifacts. It references and preserves attribution to the Apache-2.0 Evo-RL project: https://github.com/MINT-SJTU/Evo-RL.

## Key Features

- OpenPI/Pi0.5 policy runtime adapter for a single Piper arm.
- Interactive correction / DAgger-style episode workflow with policy and intervention frame labeling.
- Fail-fast checks for OpenPI source paths, checkpoint files, normalization stats, Python packages, and CAN interface.
- `.env.local` based local configuration for machine-specific paths, camera serials, and robot settings.
- Dry-run and validation modes for setup checks before commanding hardware.
- Browser UI launch helper for the collection control panel.
- External artifact documentation for OpenPI source, checkpoints, and checksum TODOs.
- Explicit Evo-RL, OpenPI, and LeRobot/Piper ecosystem attribution.

## Repository Layout

```text
.
|-- scripts/
|   |-- run_stack_bowls_collection.sh
|   |-- collect_interactive_dagger.py
|   |-- validate_environment.py
|   `-- open_collection_ui.sh
|-- runtime/
|   `-- piper_openpi_runtime.py
|-- docs/
|   |-- SETUP.md
|   |-- HARDWARE.md
|   `-- EXTERNAL_ARTIFACTS.md
|-- vendor/
|   `-- lerobot_piper/
|-- requirements-collection.txt
|-- .env.example
|-- NOTICE
`-- THIRD_PARTY_NOTICES.md
```

## What You Need

This repository intentionally does not include OpenPI source, policy checkpoints, collected datasets, or machine-local secrets.

Required external pieces:

- Linux machine with Python 3.11.
- AgileX Piper arm connected through a CAN interface such as `can1`.
- Intel RealSense cameras if using the default four-camera setup.
- OpenPI source checkout referenced by `OPENPI_ROOT`.
- OpenPI/Pi0.5 inference checkpoint referenced by `CHECKPOINT_DIR`.
- Local output directory referenced by `COLLECTION_OUTPUT_ROOT`.
- Chrome/Chromium or `xdg-open` for the web UI helper.

Known OpenPI reference for this workflow:

- Repository: `git@github.com:1939645507/openpi.git`
- Commit: `edfcb37eb2eaa0472627899edefae6af63850fe3`
- Required paths under `OPENPI_ROOT`: `src/openpi` and `packages/openpi-client/src`

Checkpoint artifacts are documented in [docs/EXTERNAL_ARTIFACTS.md](docs/EXTERNAL_ARTIFACTS.md). The public artifact URL and SHA256 checksums are still TODO and must be filled before a release that points users to a specific checkpoint.

## Quick Start

Clone this repository:

```bash
git clone <your-repo-url> piper-correction-collection
cd piper-correction-collection
```

Create a Python 3.11 environment and install the collection dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-collection.txt
```

Install OpenPI separately. Do not copy OpenPI source into this repository unless you have reviewed the license and have a specific reason to vendor it.

```bash
git clone git@github.com:1939645507/openpi.git external/openpi
cd external/openpi
git checkout edfcb37eb2eaa0472627899edefae6af63850fe3
cd ../..
```

Create local configuration:

```bash
cp .env.example .env.local
```

Edit `.env.local` for your machine. At minimum, set:

```bash
OPENPI_ROOT=/path/to/openpi
CHECKPOINT_DIR=/path/to/pi05_checkpoint_dir
COLLECTION_OUTPUT_ROOT=/path/to/output/piper_correction_YYYYMMDD_HHMMSS
PIPER_CAN_NAME=can1
```

The main run script, UI launcher, and validator load `.env.local` automatically when it exists. Keep `.env.local` out of Git.

## Validate The Environment

Run the validator before collecting:

```bash
python scripts/validate_environment.py
```

For software-only checks that should not fail on missing camera/robot imports or CAN hardware, use:

```bash
python scripts/validate_environment.py --skip-hardware
```

The validator does not command the robot. It reports `PASS`, `WARN`, and `FAIL` for Python version, OpenPI source paths, runtime files, checkpoint files, `norm_stats.json`, imports, and the selected CAN interface.

You can also exercise the policy-loading path without sending robot actions:

```bash
VALIDATE_ONLY=true DRY_RUN=true ENABLE_ARM=false bash scripts/run_stack_bowls_collection.sh
```

This still requires a valid OpenPI checkout and checkpoint.

## Run Collection

After validating software, hardware, cameras, CAN, and checkpoint artifacts:

```bash
OPENPI_ROOT=/path/to/openpi \
CHECKPOINT_DIR=/path/to/pi05_checkpoint_dir \
COLLECTION_OUTPUT_ROOT=/path/to/output/piper_correction_$(date +%Y%m%d_%H%M%S) \
PIPER_CAN_NAME=can1 \
NUM_EPISODES=100 \
GUI=true \
ENABLE_ARM=true \
DEVICE=cuda \
ACTION_SCALE=1.0 \
MAX_ABS_DELTA=0.04 \
bash scripts/run_stack_bowls_collection.sh
```

Open the browser UI from another terminal:

```bash
bash scripts/open_collection_ui.sh http://127.0.0.1:8765
```

Set `BROWSER_BIN=/path/to/browser` or `CHROME_BIN=/path/to/browser` if Chrome/Chromium auto-detection fails.

## Collection Workflow

1. Start in model rollout mode.
2. Trigger intervention with the UI or the configured intervention key.
3. Connect the master arm according to your hardware procedure.
4. Record human correction frames.
5. Save the episode as success or failure.
6. Disconnect the master/follower link before policy control resumes if your hardware setup requires it.
7. Reset the scene and start the next rollout.

Recorded frames include:

- `action`: executed policy action in model mode, observed follower state in human correction mode.
- `complementary_info.policy_action`: policy/source action trace.
- `complementary_info.is_intervention`: `0` for policy frames, `1` for human correction frames.
- Episode metadata including success label, policy frame count, intervention frame count, and intervention index range.

## Safety Notes

This project controls real robot hardware. Start with `ENABLE_ARM=false`, `DRY_RUN=true`, or `VALIDATE_ONLY=true` on a new setup. Confirm emergency stop access, CAN interface mapping, camera serials, workspace clearance, and master/follower handoff procedure before collecting. See [docs/HARDWARE.md](docs/HARDWARE.md) for the hardware checklist.

## Documentation

- [docs/SETUP.md](docs/SETUP.md): Python, CUDA/Torch, OpenPI, RealSense, CAN, Piper SDK, and UI setup.
- [docs/HARDWARE.md](docs/HARDWARE.md): Piper CAN, camera serials, safety checklist, and non-motion modes.
- [docs/EXTERNAL_ARTIFACTS.md](docs/EXTERNAL_ARTIFACTS.md): checkpoint layout, required files, artifact distribution recommendations, and checksum TODOs.
- [.env.example](.env.example): local configuration template.

## Attribution And License

This repository contains an interactive correction / DAgger-style data-collection workflow adapted from local robotics code derived from the open-source Evo-RL project:

- Evo-RL upstream: https://github.com/MINT-SJTU/Evo-RL
- Evo-RL license: Apache License 2.0

The bundled `vendor/lerobot_piper/lerobot` runtime package and related integration code retain upstream license notices. OpenPI is treated as an external dependency and is not vendored here.

Keep [LICENSE](LICENSE), [NOTICE](NOTICE), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) with redistributed versions.
