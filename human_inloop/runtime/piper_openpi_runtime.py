#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import importlib.util
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The bundled OpenPI Python 3.11 runtime on this machine lacks Python.h, which
# breaks torch.compile/Triton launcher builds at runtime. Default to eager mode
# unless the caller explicitly opts back into torch.compile.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEROBOT_RUNTIME_ROOT = Path(os.environ.get("ROBOT_RUNTIME_ROOT") or os.environ["PIPER_RUNTIME_ROOT"]) if (os.environ.get("ROBOT_RUNTIME_ROOT") or os.environ.get("PIPER_RUNTIME_ROOT")) else PROJECT_ROOT / "vendor" / "lerobot_piper"


def pick_first_existing_path(*candidates: Path | None) -> Path:
    existing = [candidate.resolve() for candidate in candidates if candidate is not None and candidate.exists()]
    if existing:
        return existing[0]
    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise ValueError("At least one candidate path is required.")


DEFAULT_OPENPI_ROOT = pick_first_existing_path(
    Path(os.environ["OPENPI_ROOT"]) if os.environ.get("OPENPI_ROOT") else None,
    PROJECT_ROOT / "external" / "openpi",
)
DEFAULT_CHECKPOINT_DIR = pick_first_existing_path(
    Path(os.environ["CHECKPOINT_DIR"]) if os.environ.get("CHECKPOINT_DIR") else None,
    PROJECT_ROOT / "checkpoints" / "pi05_stack_bowls",
)
DEFAULT_PROMPT = "stack the yellow bowl on the green bowl"

DEFAULT_CAMERA_LAYOUT = {
    "head": {
        "type": "intelrealsense",
        "serial_number": 254622073267,
        "fps": 15,
        "width": 640,
        "height": 480,
        "force_hardware_reset": False,
        "rotation": None,
    },
    "left_wrist": {
        "type": "intelrealsense",
        "serial_number": 244222071617,
        "fps": 15,
        "width": 640,
        "height": 480,
        "force_hardware_reset": False,
        "rotation": None,
    },
    "right_wrist": {
        "type": "intelrealsense",
        "serial_number": 317622070857,
        "fps": 15,
        "width": 640,
        "height": 480,
        "force_hardware_reset": False,
        "rotation": None,
    },
    "front_view": {
        "type": "intelrealsense",
        "serial_number": 254622079402,
        "fps": 15,
        "width": 640,
        "height": 480,
        "force_hardware_reset": False,
        "rotation": None,
    },
}


@dataclass
class PrefetchedActionChunk:
    actions: Any
    infer_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 4-camera stack bowls OpenPI checkpoint on a single Piper arm "
            "(right arm on the selected CAN interface, mapped to state dims 0-6)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--openpi-root", type=Path, default=DEFAULT_OPENPI_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--can-name",
        default="can1",
        help="CAN interface for the active right arm used during deployment.",
    )
    parser.add_argument(
        "--norm-stats-path",
        type=Path,
        default=None,
        help="Optional override. If omitted, resolve from metadata assets info under --openpi-root/assets.",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--actions-per-inference", type=int, default=1)
    parser.add_argument(
        "--sync-inference",
        action="store_true",
        help="Disable background chunk prefetch and run policy inference on the control loop.",
    )
    parser.add_argument(
        "--prefetch-threshold",
        type=int,
        default=2,
        help=(
            "When async prefetch is enabled, start computing the next action chunk once the "
            "remaining queued actions fall to this many steps or fewer."
        ),
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=1.0,
        help=(
            "Scale the model-suggested displacement around the current arm state. "
            "0 keeps the current state, 1 uses the full model action."
        ),
    )
    parser.add_argument(
        "--max-abs-delta",
        type=float,
        default=0.10,
        help="Clamp each commanded arm dimension around the latest observed arm state.",
    )
    parser.add_argument("--steps", type=int, default=0, help="0 means run until Ctrl+C.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Load the policy and run one dummy inference, then exit.")
    parser.add_argument("--enable-arm", action="store_true", help="Enable the selected arm after connecting.")
    parser.add_argument(
        "--move-to-start-pose",
        action="store_true",
        help="Interpolate the selected arm to the configured start pose before the first inference step.",
    )
    parser.add_argument(
        "--start-pose",
        default=None,
        help=(
            "Explicit comma-separated 7D start pose in radians/gripper units. "
            "Required when using --move-to-start-pose."
        ),
    )
    parser.add_argument("--start-pose-steps", type=int, default=80, help="Interpolation steps used for the start pose move.")
    parser.add_argument(
        "--start-pose-step-seconds",
        type=float,
        default=0.05,
        help="Sleep duration between each interpolated start pose command.",
    )
    parser.add_argument(
        "--start-pose-settle-seconds",
        type=float,
        default=0.5,
        help="Extra settle time after reaching the start pose before inference begins.",
    )
    parser.add_argument("--disable-arm-on-exit", action="store_true", help="Move to the safe pose then disable the selected arm on exit.")
    parser.add_argument(
        "--debug-action-details",
        action="store_true",
        help="Print current state, raw model absolute action, scaled action, and final commanded action.",
    )
    parser.add_argument(
        "--print-command-counts",
        action="store_true",
        help="Print the integer joint/gripper command counts that are sent to the Piper SDK.",
    )
    parser.add_argument("--disable-settle-seconds", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--robot-cameras-json",
        default=None,
        help="JSON object describing the four RealSense cameras. If omitted, use the local 4-camera defaults.",
    )
    parser.add_argument("--head-camera-key", default="head")
    parser.add_argument("--left-wrist-camera-key", default="left_wrist")
    parser.add_argument("--right-wrist-camera-key", default="right_wrist")
    parser.add_argument("--front-view-camera-key", default="front_view")
    args = parser.parse_args()
    if not 0.0 <= args.action_scale <= 1.0:
        parser.error("--action-scale must be within [0, 1].")
    if args.prefetch_threshold < 0:
        parser.error("--prefetch-threshold must be >= 0.")
    if args.start_pose_steps < 1:
        parser.error("--start-pose-steps must be at least 1.")
    if args.move_to_start_pose:
        start_pose_arg = "" if args.start_pose is None else args.start_pose.strip()
        if not start_pose_arg:
            parser.error(
                "--move-to-start-pose requires an explicit --start-pose. "
                "The Piper built-in init pose is a calibration zero pose and is not treated as deployment-safe."
            )
        if start_pose_arg.lower() == "init":
            parser.error(
                "--start-pose=init is disabled for single-arm deployment because the Piper init pose "
                "is not treated as deployment-safe. Provide an explicit 7D pose or leave --move-to-start-pose off."
            )
    return args


def ensure_lerobot_runtime_on_path(runtime_root: Path | None = None) -> None:
    root = (runtime_root or DEFAULT_LEROBOT_RUNTIME_ROOT).resolve()
    if not (root / "lerobot").exists():
        raise FileNotFoundError(
            f"Missing LeRobot-Piper runtime package under {root}. Expected a `lerobot/` directory."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def ensure_openpi_on_path(openpi_root: Path) -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            "This OpenPI codebase expects Python 3.11. "
            "Set OPENPI_PYTHON_BIN to a Python 3.11 interpreter, or run this script from a Python 3.11 environment."
        )

    openpi_root = openpi_root.resolve()
    source_paths = [
        (openpi_root / "src").resolve(),
        (openpi_root / "packages" / "openpi-client" / "src").resolve(),
    ]
    for path in source_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required OpenPI source path: {path}")

    for module_name in list(sys.modules):
        if module_name == "openpi" or module_name.startswith("openpi."):
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()

    # Keep third-party dependencies from the active runtime. Some local OpenPI
    # checkouts carry a broken `.venv/site-packages` tree whose torch build
    # crashes with SIGBUS on import, so only prepend the OpenPI source roots.
    for path in reversed(source_paths):
        path_str = str(path)
        if not path.exists():
            continue
        while path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)

    spec = importlib.util.find_spec("openpi")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(f"Unable to resolve the `openpi` package after adding paths under {openpi_root}")
    expected_package_dir = (openpi_root / "src" / "openpi").resolve()
    resolved_locations = [Path(location).resolve() for location in spec.submodule_search_locations]
    if expected_package_dir not in resolved_locations:
        formatted_locations = ", ".join(str(location) for location in resolved_locations)
        raise RuntimeError(
            "Resolved `openpi` from an unexpected location. "
            f"Expected {expected_package_dir}, got [{formatted_locations}]"
        )


def validate_checkpoint_safetensors(model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint file: {model_path}. "
            "Set CHECKPOINT_DIR to an inference checkpoint directory containing model.safetensors."
        )
    with model_path.open("rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))

    max_end = 0
    for key, value in header.items():
        if key == "__metadata__":
            continue
        max_end = max(max_end, int(value["data_offsets"][1]))

    required_bytes = 8 + header_len + max_end
    actual_bytes = model_path.stat().st_size
    if actual_bytes < required_bytes:
        raise RuntimeError(
            "Checkpoint is truncated: "
            f"{model_path} has {actual_bytes} bytes but requires {required_bytes} bytes."
        )


def load_checkpoint_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    import torch

    metadata_path = checkpoint_dir / "metadata.pt"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint file: {metadata_path}. "
            "Set CHECKPOINT_DIR to an inference checkpoint directory containing metadata.pt."
        )
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    if not isinstance(metadata, dict) or "config" not in metadata:
        raise RuntimeError(f"Unexpected metadata format in {metadata_path}")
    return metadata


def ensure_checkpoint_artifacts(checkpoint_dir: Path, norm_stats_path: Path | None = None) -> None:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_dir}. Set CHECKPOINT_DIR or pass --checkpoint-dir."
        )
    for name in ("model.safetensors", "metadata.pt"):
        candidate = checkpoint_dir / name
        if not candidate.exists():
            raise FileNotFoundError(
                f"Missing checkpoint file: {candidate}. "
                "Set CHECKPOINT_DIR to an inference checkpoint directory with model.safetensors and metadata.pt."
            )
    if norm_stats_path is not None and not norm_stats_path.exists():
        raise FileNotFoundError(
            f"NORM_STATS_PATH does not exist: {norm_stats_path}. "
            "Set NORM_STATS_PATH to assets/<asset_id>/norm_stats.json or leave it unset when CHECKPOINT_DIR/assets contains it."
        )


def to_numpy(value: Any):
    import numpy as np
    import torch

    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def parse_rotation_value(rotation: Any) -> int | None:
    if rotation in (None, "none"):
        return None
    return int(rotation)


def load_single_arm_robot_config(robot_cameras_json: str | None, can_name: str):
    from lerobot.common.robot_devices.cameras.configs import IntelRealSenseCameraConfig
    from lerobot.common.robot_devices.motors.configs import PiperMotorsBusConfig
    from lerobot.common.robot_devices.robots.configs import PiperRobotConfig

    raw_cameras = DEFAULT_CAMERA_LAYOUT if robot_cameras_json is None else json.loads(robot_cameras_json)
    cameras = {}
    for key, cfg in raw_cameras.items():
        camera_type = cfg.get("type", "intelrealsense")
        if camera_type != "intelrealsense":
            raise ValueError(f"Unsupported camera type `{camera_type}` for key `{key}`.")
        cameras[key] = IntelRealSenseCameraConfig(
            serial_number=int(cfg["serial_number"]),
            fps=int(cfg["fps"]) if cfg.get("fps") is not None else None,
            width=int(cfg["width"]) if cfg.get("width") is not None else None,
            height=int(cfg["height"]) if cfg.get("height") is not None else None,
            force_hardware_reset=bool(cfg.get("force_hardware_reset", False)),
            rotation=parse_rotation_value(cfg.get("rotation")),
        )

    default_main = PiperRobotConfig().follower_arm["main"]
    main_arm = PiperMotorsBusConfig(
        can_name=can_name,
        motors=dict(default_main.motors),
    )
    return PiperRobotConfig(
        inference_time=True,
        passive_recording=True,
        follower_arms={},
        follower_arm={"main": main_arm},
        cameras=cameras,
    )


def prune_training_only_repack_leaves(structure: Any):
    if isinstance(structure, dict):
        pruned = {}
        for key, value in structure.items():
            child = prune_training_only_repack_leaves(value)
            if child is not None:
                pruned[key] = child
        return pruned or None
    if isinstance(structure, (list, tuple)):
        pruned_items = [item for item in (prune_training_only_repack_leaves(value) for value in structure) if item is not None]
        if not pruned_items:
            return None
        return type(structure)(pruned_items)
    if structure == "action":
        return None
    return structure


def reconstruct_repack_group(transforms_module, repack_metadata: dict[str, Any]):
    inputs = []
    outputs = []
    for item in repack_metadata.get("inputs", ()):
        structure = prune_training_only_repack_leaves(item.get("structure"))
        if structure is not None:
            inputs.append(transforms_module.RepackTransform(structure))
    for item in repack_metadata.get("outputs", ()):
        structure = item.get("structure")
        if structure is not None:
            outputs.append(transforms_module.RepackTransform(structure))
    return transforms_module.Group(inputs=tuple(inputs), outputs=tuple(outputs))


def reconstruct_post_delta_clamp(transforms_module, clamp_metadata: dict[str, Any] | None):
    if not clamp_metadata:
        return None
    return transforms_module.ClampToValues(
        state_dims=tuple(clamp_metadata.get("state_dims", ())),
        state_values=tuple(clamp_metadata.get("state_values", ())),
        action_dims=tuple(clamp_metadata.get("action_dims", ())),
        action_values=tuple(clamp_metadata.get("action_values", ())),
    )


def build_openpi_train_config_from_metadata(
    metadata: dict[str, Any],
    prompt_override: str | None,
):
    from openpi import transforms as openpi_transforms
    from openpi.models import pi0_config
    from openpi.training import config as openpi_config

    config = metadata["config"]
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    assets_config = data_config.get("assets", {})
    base_config = data_config.get("base_config", {})

    prompt = prompt_override
    if prompt is None:
        prompt = data_config.get("default_prompt")

    repack_group = reconstruct_repack_group(
        openpi_transforms,
        data_config.get("repack_transforms", {}),
    )
    post_delta_clamp = reconstruct_post_delta_clamp(
        openpi_transforms,
        data_config.get("post_delta_clamp"),
    )

    train_config = openpi_config.TrainConfig(
        name=config["name"],
        project_name=config.get("project_name", "openpi"),
        exp_name=config.get("exp_name", "runtime"),
        model=pi0_config.Pi0Config(
            action_dim=model_config.get("action_dim", 32),
            action_horizon=model_config.get("action_horizon", 10),
            max_token_len=model_config.get("max_token_len", 200),
            dtype=model_config.get("dtype", "bfloat16"),
            paligemma_variant=model_config.get("paligemma_variant", "gemma_2b"),
            action_expert_variant=model_config.get("action_expert_variant", "gemma_300m"),
            pi05=model_config.get("pi05", True),
            discrete_state_input=model_config.get("discrete_state_input", False),
        ),
        data=openpi_config.LeRobotAlohaDataConfig(
            repo_id=data_config.get("repo_id", "Real_world_dataset/lerobot_piper"),
            assets=openpi_config.AssetsConfig(
                assets_dir=assets_config.get("assets_dir"),
                asset_id=assets_config.get("asset_id"),
            ),
            base_config=openpi_config.DataConfig(
                prompt_from_task=base_config.get("prompt_from_task", False),
                local_roots=tuple(base_config.get("local_roots", ())),
                download_videos=base_config.get("download_videos", False),
                video_backend=base_config.get("video_backend"),
            ),
            use_delta_joint_actions=data_config.get("use_delta_joint_actions", False),
            default_prompt=prompt,
            adapt_to_pi=data_config.get("adapt_to_pi", False),
            post_delta_clamp=post_delta_clamp,
            repack_transforms=repack_group,
            action_sequence_keys=tuple(data_config.get("action_sequence_keys", ("action",))),
        ),
        checkpoint_base_dir=config.get("checkpoint_base_dir", "./checkpoints"),
        assets_base_dir=config.get("assets_base_dir", "./assets"),
        policy_metadata=config.get("policy_metadata"),
    )
    return train_config


def resolve_norm_stats_path(
    checkpoint_dir: Path,
    openpi_root: Path,
    metadata: dict[str, Any],
    explicit_path: Path | None,
) -> Path:
    if explicit_path is not None:
        explicit_path = explicit_path.resolve()
        if not explicit_path.exists():
            raise FileNotFoundError(f"Explicit norm_stats.json does not exist: {explicit_path}")
        return explicit_path

    config = metadata["config"]
    data_config = config.get("data", {})
    assets_config = data_config.get("assets", {})
    asset_id = assets_config.get("asset_id")
    metadata_assets_dir = assets_config.get("assets_dir")

    candidates: list[Path] = []
    if asset_id:
        candidates.append(checkpoint_dir / "assets" / asset_id / "norm_stats.json")
    if metadata_assets_dir and asset_id:
        metadata_assets_path = Path(metadata_assets_dir)
        if not metadata_assets_path.is_absolute():
            candidates.append(openpi_root / "assets" / metadata_assets_path / asset_id / "norm_stats.json")
            candidates.append(openpi_root / "assets" / metadata_assets_path / "norm_stats.json")
            candidates.append(openpi_root / "assets" / metadata_assets_path.name / asset_id / "norm_stats.json")
            candidates.append(openpi_root / "assets" / metadata_assets_path.name / "norm_stats.json")
    if asset_id:
        candidates.append(openpi_root / "assets" / config["name"] / asset_id / "norm_stats.json")
    candidates.extend(sorted(checkpoint_dir.glob("assets/**/norm_stats.json")))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not resolve norm_stats.json. "
        "Set NORM_STATS_PATH or include assets/<asset_id>/norm_stats.json in CHECKPOINT_DIR. "
        "Absolute asset paths embedded in metadata are intentionally not used for portable releases. "
        f"Tried:\n{searched}"
    )


def load_policy(
    checkpoint_dir: Path,
    openpi_root: Path,
    metadata: dict[str, Any],
    prompt_override: str | None,
    norm_stats_path: Path | None,
    device: str,
):
    from openpi.policies import policy_config
    from openpi.shared import normalize

    train_config = build_openpi_train_config_from_metadata(metadata, prompt_override)
    resolved_norm_stats_path = resolve_norm_stats_path(checkpoint_dir, openpi_root, metadata, norm_stats_path)
    norm_stats = normalize.deserialize_json(resolved_norm_stats_path.read_text())
    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        repack_transforms=train_config.data.repack_transforms,
        norm_stats=norm_stats,
        pytorch_device=device,
    )
    return policy, train_config, norm_stats, resolved_norm_stats_path


def image_to_chw_uint8(image: Any):
    import numpy as np

    array = to_numpy(image)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D image array, got shape {array.shape}")
    if array.shape[-1] == 3:
        array = np.transpose(array, (2, 0, 1))
    elif array.shape[0] != 3:
        raise ValueError(f"Expected CHW or HWC with 3 channels, got {array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 255.0 if array.max() > 1.5 else 1.0)
        if array.max() <= 1.5:
            array = array * 255.0
    return array.astype(np.uint8, copy=False)


def get_camera_image(robot_observation: dict[str, Any], camera_key: str):
    robot_key = f"observation.images.{camera_key}"
    if robot_key not in robot_observation:
        available = sorted(key for key in robot_observation if key.startswith("observation.images."))
        raise KeyError(f"Missing camera `{robot_key}`. Available image keys: {available}")
    return image_to_chw_uint8(robot_observation[robot_key])


def build_full_policy_state(
    active_state_7d,
    train_config,
    norm_stats: dict[str, Any],
):
    import numpy as np

    clamp = train_config.data.post_delta_clamp
    if clamp is None:
        raise RuntimeError("This single-arm deployment expects metadata post_delta_clamp to be present.")

    full_dim = len(norm_stats["state"].mean)
    full_state = np.zeros(full_dim, dtype=np.float32)

    fixed_dims = tuple(clamp.state_dims)
    fixed_values = np.asarray(clamp.state_values, dtype=np.float32)
    full_state[np.asarray(fixed_dims, dtype=np.int64)] = fixed_values

    active_dims = tuple(index for index in range(full_dim) if index not in set(fixed_dims))
    active_state_7d = np.asarray(active_state_7d, dtype=np.float32)
    if len(active_dims) != active_state_7d.shape[-1]:
        raise ValueError(
            f"Active state dims {active_dims} do not match observed arm state shape {active_state_7d.shape}."
        )
    full_state[np.asarray(active_dims, dtype=np.int64)] = active_state_7d
    return full_state, active_dims


def build_raw_openpi_observation(
    robot_observation: dict[str, Any],
    full_state,
    prompt: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "observation.images.head": get_camera_image(robot_observation, args.head_camera_key),
        "observation.images.left_wrist": get_camera_image(robot_observation, args.left_wrist_camera_key),
        "observation.images.right_wrist": get_camera_image(robot_observation, args.right_wrist_camera_key),
        "observation.images.front_view": get_camera_image(robot_observation, args.front_view_camera_key),
        "observation.state": full_state.astype("float32", copy=False),
        "prompt": prompt,
    }


def clone_raw_openpi_observation(raw_observation: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    snapshot = {}
    for key, value in raw_observation.items():
        if isinstance(value, np.ndarray):
            snapshot[key] = np.array(value, copy=True)
        else:
            snapshot[key] = value
    return snapshot


def extract_active_arm_action(action_chunk, active_dims):
    import numpy as np

    actions = np.asarray(action_chunk, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    return actions[:, np.asarray(active_dims, dtype=np.int64)]


def format_arm_action(label: str, action_7d) -> str:
    import numpy as np

    values = np.asarray(action_7d, dtype=np.float32)
    preview = ", ".join(f"{value:+.4f}" for value in values[:6])
    return f"{label}[:6]=[{preview}] grip={values[6]:+.4f}"


def parse_start_pose_arg(start_pose_arg: str | None, fallback_pose) -> Any:
    import numpy as np

    if start_pose_arg is None or not start_pose_arg.strip():
        return np.asarray(fallback_pose, dtype=np.float32)

    if start_pose_arg.strip().lower() == "init":
        return np.asarray(fallback_pose, dtype=np.float32)

    values = [float(item.strip()) for item in start_pose_arg.split(",") if item.strip()]
    if len(values) != 7:
        raise ValueError(
            "--start-pose must contain exactly 7 comma-separated floats "
            f"(got {len(values)} values from {start_pose_arg!r})."
        )
    return np.asarray(values, dtype=np.float32)


def action_to_command_counts(action_7d, arm) -> tuple[list[int], int]:
    import numpy as np

    values = np.asarray(action_7d, dtype=np.float32)
    joint_counts = [round(float(value) * arm.joint_factor) for value in values[:6]]
    gripper_count = round(float(values[6]) * 1_000_000)
    return joint_counts, gripper_count


def format_command_counts(label: str, joint_counts: list[int], gripper_count: int) -> str:
    preview = ", ".join(f"{value:+d}" for value in joint_counts[:6])
    return f"{label}[:6]=[{preview}] grip={gripper_count:+d}"


def log_action_details(
    *,
    step: int,
    infer_ms: float,
    queue_len: int,
    can_name: str,
    current_state_7d,
    model_action_7d,
    scaled_action_7d,
    final_action_7d,
    dry_run: bool,
    command_counts: tuple[list[int], int] | None,
) -> None:
    import numpy as np

    raw_delta = np.asarray(model_action_7d, dtype=np.float32) - np.asarray(current_state_7d, dtype=np.float32)
    final_delta = np.asarray(final_action_7d, dtype=np.float32) - np.asarray(current_state_7d, dtype=np.float32)
    prefix = "dry-run" if dry_run else "step"
    print(f"[{prefix} step={step}] infer={infer_ms:.1f}ms queue={queue_len}")
    print(f"  {format_arm_action(f'{can_name}.obs', current_state_7d)}")
    print(f"  {format_arm_action(f'{can_name}.model_abs', model_action_7d)}")
    print(f"  {format_arm_action(f'{can_name}.model_delta', raw_delta)}")
    print(f"  {format_arm_action(f'{can_name}.scaled_abs', scaled_action_7d)}")
    print(f"  {format_arm_action(f'{can_name}.final_abs', final_action_7d)}")
    print(f"  {format_arm_action(f'{can_name}.final_delta', final_delta)}")
    if command_counts is not None:
        joint_counts, gripper_count = command_counts
        print(f"  {format_command_counts(f'{can_name}.cmd_counts', joint_counts, gripper_count)}")


def scale_active_arm_action(action_7d, current_state_7d, action_scale: float):
    import numpy as np

    if action_scale >= 1.0:
        return action_7d
    if action_scale <= 0.0:
        return np.asarray(current_state_7d, dtype=np.float32).copy()
    action_7d = np.asarray(action_7d, dtype=np.float32)
    current_state_7d = np.asarray(current_state_7d, dtype=np.float32)
    return current_state_7d + action_scale * (action_7d - current_state_7d)


def clamp_active_arm_action(action_7d, current_state_7d, max_abs_delta: float | None):
    import numpy as np

    if max_abs_delta is None:
        return action_7d
    lower = current_state_7d - max_abs_delta
    upper = current_state_7d + max_abs_delta
    return np.clip(action_7d, lower, upper)


def maybe_enable_main_arm(robot) -> None:
    arm = robot.follower_arms["main"]
    print(f"Enabling single arm on {arm.can_name}...")
    if not arm.connect(enable=True):
        raise RuntimeError(f"Failed to enable arm on {arm.can_name}.")


def maybe_move_main_arm_to_start_pose(robot, args: argparse.Namespace) -> None:
    import numpy as np

    arm = robot.follower_arms["main"]
    state = arm.wait_for_feedback(timeout_s=3.0)
    start = np.asarray([state[f"joint_{i}"] for i in range(1, 7)] + [state["gripper"]], dtype=np.float32)
    target = parse_start_pose_arg(args.start_pose, arm.init_joint_position)

    print(f"Moving {arm.can_name} arm to the configured start pose.")
    print(f"  {format_arm_action(f'{arm.can_name}.start', start)}")
    print(f"  {format_arm_action(f'{arm.can_name}.target', target)}")
    if args.print_command_counts:
        joint_counts, gripper_count = action_to_command_counts(target, arm)
        print(f"  {format_command_counts(f'{arm.can_name}.start_cmd_counts', joint_counts, gripper_count)}")

    for alpha in np.linspace(0.0, 1.0, args.start_pose_steps, dtype=np.float32):
        command = (1.0 - float(alpha)) * start + float(alpha) * target
        arm.write(command.tolist())
        time.sleep(args.start_pose_step_seconds)

    if args.start_pose_settle_seconds > 0:
        time.sleep(args.start_pose_settle_seconds)

    final_state = arm.wait_for_feedback(timeout_s=3.0)
    final = np.asarray([final_state[f"joint_{i}"] for i in range(1, 7)] + [final_state["gripper"]], dtype=np.float32)
    print(f"  {format_arm_action(f'{arm.can_name}.start_done', final)}")


def maybe_disable_main_arm(robot, settle_seconds: float) -> None:
    arm = robot.follower_arms["main"]
    pose_preview = ", ".join(f"{value:+.4f}" for value in arm.safe_disable_position[:6])
    print(
        f"Moving {arm.can_name} arm to its safe disconnect pose: "
        f"[{pose_preview}] grip={arm.safe_disable_position[6]:+.4f}"
    )
    arm.safe_disconnect()
    if settle_seconds > 0:
        time.sleep(settle_seconds)
    print(f"Disabling arm on {arm.can_name}.")
    arm.connect(enable=False)


def infer_action_chunk(policy, raw_observation: dict[str, Any], actions_per_inference: int):
    import numpy as np

    result = policy.infer(raw_observation)
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    infer_ms = float(result.get("policy_timing", {}).get("infer_ms", 0.0))
    return actions[: max(1, actions_per_inference)], infer_ms


class AsyncActionChunkPrefetcher:
    def __init__(
        self,
        policy,
        *,
        actions_per_inference: int,
        enabled: bool,
        prefetch_threshold: int,
    ) -> None:
        self.policy = policy
        self.actions_per_inference = actions_per_inference
        self.enabled = enabled
        self.prefetch_threshold = prefetch_threshold
        self._executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="openpi-prefetch")
            if enabled
            else None
        )
        self._future: concurrent.futures.Future | None = None

    def has_in_flight(self) -> bool:
        return self._future is not None

    def take_ready(self) -> PrefetchedActionChunk | None:
        if self._future is None or not self._future.done():
            return None
        future = self._future
        self._future = None
        actions, infer_ms = future.result()
        return PrefetchedActionChunk(actions=actions, infer_ms=infer_ms)

    def wait(self) -> PrefetchedActionChunk:
        if self._future is None:
            raise RuntimeError("No prefetch request is currently in flight.")
        future = self._future
        self._future = None
        actions, infer_ms = future.result()
        return PrefetchedActionChunk(actions=actions, infer_ms=infer_ms)

    def maybe_submit(self, raw_observation: dict[str, Any], remaining_queue_len: int) -> bool:
        if not self.enabled or self._executor is None:
            return False
        if self._future is not None or remaining_queue_len > self.prefetch_threshold:
            return False
        snapshot = clone_raw_openpi_observation(raw_observation)
        self._future = self._executor.submit(
            infer_action_chunk,
            self.policy,
            snapshot,
            self.actions_per_inference,
        )
        return True

    def shutdown(self) -> None:
        if self._executor is None:
            return
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._executor = None
        self._future = None


def run_validation_once(policy, train_config, norm_stats, args: argparse.Namespace) -> None:
    import numpy as np

    dummy_active_state = np.zeros(7, dtype=np.float32)
    full_state, active_dims = build_full_policy_state(dummy_active_state, train_config, norm_stats)
    dummy_obs = {
        "observation.images.head": np.zeros((3, 480, 640), dtype=np.uint8),
        "observation.images.left_wrist": np.zeros((3, 480, 640), dtype=np.uint8),
        "observation.images.right_wrist": np.zeros((3, 480, 640), dtype=np.uint8),
        "observation.images.front_view": np.zeros((3, 480, 640), dtype=np.uint8),
        "observation.state": full_state,
        "prompt": args.prompt,
    }
    action_chunk, infer_ms = infer_action_chunk(policy, dummy_obs, args.actions_per_inference)
    active_arm_actions = extract_active_arm_action(action_chunk, active_dims)
    print(
        "Validation inference succeeded: "
        f"chunk_shape={action_chunk.shape} arm_shape={active_arm_actions.shape} infer_ms={infer_ms:.1f}"
    )


def main() -> None:
    args = parse_args()
    ensure_lerobot_runtime_on_path()

    checkpoint_dir = args.checkpoint_dir.resolve()
    openpi_root = args.openpi_root.resolve()
    if not openpi_root.exists():
        raise FileNotFoundError(
            f"OpenPI root does not exist: {openpi_root}. Set OPENPI_ROOT or pass --openpi-root."
        )
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_dir}. Set CHECKPOINT_DIR or pass --checkpoint-dir."
        )
    ensure_openpi_on_path(openpi_root)

    ensure_checkpoint_artifacts(checkpoint_dir, args.norm_stats_path)
    validate_checkpoint_safetensors(checkpoint_dir / "model.safetensors")
    metadata = load_checkpoint_metadata(checkpoint_dir)
    policy, train_config, norm_stats, resolved_norm_stats_path = load_policy(
        checkpoint_dir,
        openpi_root,
        metadata,
        args.prompt,
        args.norm_stats_path,
        args.device,
    )

    print(
        "Loaded OpenPI policy: "
        f"name={metadata['config']['name']} "
        f"step={metadata.get('global_step')} "
        f"prompt={args.prompt!r} "
        f"can_name={args.can_name} "
        f"sync_inference={args.sync_inference} "
        f"prefetch_threshold={args.prefetch_threshold} "
        f"action_scale={args.action_scale:.2f} "
        f"move_to_start_pose={args.move_to_start_pose} "
        f"print_command_counts={args.print_command_counts} "
        f"debug_action_details={args.debug_action_details} "
        f"norm_stats={resolved_norm_stats_path}"
    )

    if args.validate_only:
        run_validation_once(policy, train_config, norm_stats, args)
        return

    import numpy as np
    import torch

    from lerobot.common.robot_devices.robots.piper import PiperRobot
    from lerobot.common.robot_devices.utils import busy_wait

    robot = PiperRobot(load_single_arm_robot_config(args.robot_cameras_json, args.can_name))
    pending_actions = deque()
    prefetched_chunk: PrefetchedActionChunk | None = None
    prefetcher = AsyncActionChunkPrefetcher(
        policy,
        actions_per_inference=args.actions_per_inference,
        enabled=not args.sync_inference,
        prefetch_threshold=args.prefetch_threshold,
    )
    control_step = 0
    max_steps = args.steps if args.steps > 0 else None
    last_infer_ms = 0.0
    active_dims = None

    try:
        robot.connect()
        if not args.dry_run and (args.enable_arm or args.move_to_start_pose):
            maybe_enable_main_arm(robot)
        if args.move_to_start_pose and not args.dry_run:
            maybe_move_main_arm_to_start_pose(robot, args)
        elif args.move_to_start_pose and args.dry_run:
            print("Skipping start pose motion because --dry-run was set.")

        while max_steps is None or control_step < max_steps:
            loop_start = time.perf_counter()
            robot_observation = robot.capture_observation()
            active_arm_state = to_numpy(robot_observation["observation.state"]).astype(np.float32, copy=False)

            full_state, active_dims = build_full_policy_state(active_arm_state, train_config, norm_stats)
            raw_observation = build_raw_openpi_observation(robot_observation, full_state, args.prompt, args)
            ready_chunk = prefetcher.take_ready()
            if ready_chunk is not None:
                prefetched_chunk = ready_chunk
            if not pending_actions:
                if prefetched_chunk is not None:
                    action_chunk = prefetched_chunk.actions
                    last_infer_ms = prefetched_chunk.infer_ms
                    prefetched_chunk = None
                elif prefetcher.has_in_flight():
                    prefetched_chunk = prefetcher.wait()
                    action_chunk = prefetched_chunk.actions
                    last_infer_ms = prefetched_chunk.infer_ms
                    prefetched_chunk = None
                else:
                    action_chunk, last_infer_ms = infer_action_chunk(policy, raw_observation, args.actions_per_inference)
                active_arm_action_chunk = extract_active_arm_action(action_chunk, active_dims)
                pending_actions.extend(active_arm_action_chunk)

            model_action = pending_actions.popleft()
            if prefetched_chunk is None:
                prefetcher.maybe_submit(raw_observation, len(pending_actions))
            scaled_action = scale_active_arm_action(model_action, active_arm_state, args.action_scale)
            next_action = clamp_active_arm_action(scaled_action, active_arm_state, args.max_abs_delta)
            command_counts = None
            if args.print_command_counts:
                command_counts = action_to_command_counts(next_action, robot.follower_arms["main"])

            if args.debug_action_details:
                log_action_details(
                    step=control_step,
                    infer_ms=last_infer_ms,
                    queue_len=len(pending_actions),
                    can_name=args.can_name,
                    current_state_7d=active_arm_state,
                    model_action_7d=model_action,
                    scaled_action_7d=scaled_action,
                    final_action_7d=next_action,
                    dry_run=args.dry_run,
                    command_counts=command_counts,
                )

            if args.dry_run:
                preview = ", ".join(f"{value:+.4f}" for value in next_action[:6])
                message = (
                    f"[dry-run step={control_step}] infer={last_infer_ms:.1f}ms "
                    f"queue={len(pending_actions)} {args.can_name}[:6]=[{preview}] grip={next_action[6]:+.4f}"
                )
                if command_counts is not None:
                    joint_counts, gripper_count = command_counts
                    message += " " + format_command_counts(f"{args.can_name}.cmd_counts", joint_counts, gripper_count)
                print(message)
            else:
                robot.send_action(torch.from_numpy(next_action))
                if not args.debug_action_details and control_step % max(1, args.log_every) == 0:
                    preview = ", ".join(f"{value:+.4f}" for value in next_action[:6])
                    message = (
                        f"[step={control_step}] infer={last_infer_ms:.1f}ms "
                        f"queue={len(pending_actions)} {args.can_name}[:6]=[{preview}] grip={next_action[6]:+.4f}"
                    )
                    if command_counts is not None:
                        joint_counts, gripper_count = command_counts
                        message += " " + format_command_counts(f"{args.can_name}.cmd_counts", joint_counts, gripper_count)
                    print(message)

            loop_dt = time.perf_counter() - loop_start
            busy_wait(max(0.0, 1.0 / args.fps - loop_dt))
            control_step += 1

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        try:
            prefetcher.shutdown()
        finally:
            try:
                if args.disable_arm_on_exit and not args.dry_run:
                    maybe_disable_main_arm(robot, args.disable_settle_seconds)
            finally:
                robot.disconnect()


if __name__ == "__main__":
    main()
