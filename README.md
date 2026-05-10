# Data Collection

A unified data collection toolkit for robot learning, including teleoperation-based demonstration collection, human-in-the-loop rollout collection, and future VR-based data collection.

## Project Overview

This repository organizes multiple robot data collection workflows under one project root. The goal is to provide clean, reusable collection entry points for imitation learning, robot policy learning, VLA, world model, and real-robot learning experiments.

Each module is self-contained so hardware-specific setup, runtime configuration, and collection tools can evolve independently while sharing a consistent top-level structure.

## Modules

| Module | Status | Description |
| --- | --- | --- |
| [Teleoperation](./teleoperation) | Available | Piper teleoperation data collection with multi-camera support. |
| [Human-in-the-Loop Rollout](./human_inloop) | Available | Rollout data collection with human intervention/takeover, following the `lerobot-human-inloop-record` style workflow. |
| [VR](./VR) | Coming soon | VR-based data collection interface. |

## Quick Start

### Teleoperation

```bash
cd teleoperation
```

See [Teleoperation](./teleoperation) for Piper hardware setup, configuration, dry-run checks, and recording commands.

### Human-in-the-Loop Rollout

```bash
cd human_inloop
```

See [Human-in-the-Loop Rollout](./human_inloop) for rollout collection with policy execution and human takeover/correction segments.

### VR

Coming soon.

## Directory Structure

```bash
data_collection/
├── README.md
├── teleoperation/
├── human_inloop/
└── VR/
    └── README.md
```

## TODO

- Add a top-level license file before the official release.
- Add module-level setup validation commands where missing.
- Add VR-based data collection once the interface is ready.
- Keep collected datasets, model checkpoints, logs, and machine-local configuration out of git.

## Acknowledgements

- LeRobot: https://github.com/huggingface/lerobot
- Evo-RL: https://github.com/MINT-SJTU/Evo-RL

## Contact

Xiaoquan Sun

sunxiaoquan@hust.edu.cn

## License

License information will be added before the official release.

