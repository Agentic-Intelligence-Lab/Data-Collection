#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/piper_4cam.yaml}"
DRY_RUN=false

usage() {
    cat <<EOF
Usage: bash scripts/collect_data.sh [--config PATH] [--dry-run]

Options:
  --config PATH  YAML run config. Defaults to configs/piper_4cam.yaml.
  --dry-run      Validate config/imports/camera detection only; do not activate CAN or start the GUI.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Config file not found: ${CONFIG_PATH}" >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [ "${DRY_RUN}" = "true" ]; then
    python "${PROJECT_ROOT}/tools/dry_run_collect.py" --config "${CONFIG_PATH}"
    exit 0
fi

eval "$(
    python -m teleoperation.recorder.config \
        --config "${CONFIG_PATH}" \
        --project-root "${PROJECT_ROOT}" \
        --format shell
)"

get_episode_count() {
    local dataset_root="$1"
    python - "$dataset_root" <<'PY'
from pathlib import Path
import sys

episodes_path = Path(sys.argv[1]) / "meta" / "episodes.jsonl"
if not episodes_path.exists():
    print(0)
    raise SystemExit

count = 0
with episodes_path.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            count += 1
print(count)
PY
}

if [ "${RESUME}" = "true" ]; then
    if [ -e "${DATASET_ROOT}" ] && [ ! -f "${DATASET_ROOT}/meta/info.json" ]; then
        echo "Error: RESUME=true but ${DATASET_ROOT} is not a valid existing LeRobot dataset."
        echo "Expected file not found: ${DATASET_ROOT}/meta/info.json"
        echo "Either remove this folder, choose a new dataset name, or set resume=false in the config."
        exit 1
    fi
else
    if [ -e "${DATASET_ROOT}" ]; then
        echo "Error: target dataset directory already exists: ${DATASET_ROOT}"
        echo "Choose a new dataset name, set resume=true, or remove the existing directory."
        exit 1
    fi
fi

INITIAL_EPISODE_COUNT="$(get_episode_count "${DATASET_ROOT}")"

echo "================ Camera Preflight ================"
echo "camera_keys=${CAMERA_KEYS}"

EXPECTED_CAMERA_JSON="${ROBOT_CAMERAS_JSON}" python - <<'PY'
import json
import os
import sys

from teleoperation.cameras.realsense import find_cameras

expected = json.loads(os.environ["EXPECTED_CAMERA_JSON"])
detected_infos = find_cameras()
detected = {int(item["serial_number"]): item["name"] for item in detected_infos}

print("Detected RealSense cameras:")
for serial_number in sorted(detected):
    print(f"  serial={serial_number}  name={detected[serial_number]}")

missing = []
for key, camera_cfg in expected.items():
    serial_number = int(camera_cfg["serial_number"])
    print(f"  configured {key}: serial={serial_number} rotation={camera_cfg.get('rotation')}")
    if serial_number not in detected:
        missing.append((key, serial_number))

if missing:
    print("Error: missing configured RealSense camera(s):")
    for key, serial_number in missing:
        print(f"  {key}: serial={serial_number}")
    print("Reconnect the missing camera(s) or update configs/camera_mapping.yaml before recording.")
    sys.exit(1)

print("All configured RealSense serial numbers are currently detected.")
PY

echo "================ CAN Activation ================"
bash "${SCRIPT_DIR}/can_activate.sh"

echo "================ GUI Collection ================"
echo "config=${CONFIG_PATH}"
echo "dataset_name=${DATASET_NAME}"
echo "dataset_root=${DATASET_ROOT}"
echo "single_task=${SINGLE_TASK}"
echo "fps=${FPS}"
echo "num_episodes=${NUM_EPISODES}"
echo "episode_time_s=${EPISODE_TIME_S}"
echo "can_settle_time_s=${CAN_SETTLE_TIME_S}"
echo "resume=${RESUME}"
echo "camera_keys=${CAMERA_KEYS}"

echo "Waiting ${CAN_SETTLE_TIME_S}s for CAN feedback to stabilize..."
sleep "${CAN_SETTLE_TIME_S}"

missing_can_interfaces=()
for iface in ${EXPECTED_CAN_INTERFACES}; do
    if [ ! -d "/sys/class/net/${iface}" ]; then
        missing_can_interfaces+=("${iface}")
    fi
done

if [ "${#missing_can_interfaces[@]}" -gt 0 ]; then
    echo "Error: missing required CAN interface(s): ${missing_can_interfaces[*]}"
    echo "Detected network interfaces:"
    ls /sys/class/net | sed 's/^/  - /'
    echo "The configured Piper setup requires all of: ${EXPECTED_CAN_INTERFACES}"
    exit 1
fi

PY_ARGS=(
  --dataset-name "${DATASET_NAME}"
  --dataset-root "${DATASET_ROOT}"
  --single-task "${SINGLE_TASK}"
  --fps "${FPS}"
  --num-episodes "${NUM_EPISODES}"
  --episode-time-s "${EPISODE_TIME_S}"
  --robot-cameras-json "${ROBOT_CAMERAS_JSON}"
  --robot-config-json "${ROBOT_CONFIG_JSON}"
)

if [ "${RESUME}" = "true" ]; then
    PY_ARGS+=(--resume)
fi

python -m teleoperation.gui.collect_data_gui "${PY_ARGS[@]}"

if [ "${VALIDATE_AFTER_RECORD}" = "true" ]; then
    FINAL_EPISODE_COUNT="$(get_episode_count "${DATASET_ROOT}")"
    echo "================ Post-Run Validation ================"
    echo "initial_episode_count=${INITIAL_EPISODE_COUNT}"
    echo "final_episode_count=${FINAL_EPISODE_COUNT}"

    if [ "${FINAL_EPISODE_COUNT}" -le "${INITIAL_EPISODE_COUNT}" ]; then
        echo "No new episodes were recorded in this run. Skipping validation."
        exit 0
    fi

    for ((episode_index=INITIAL_EPISODE_COUNT; episode_index<FINAL_EPISODE_COUNT; episode_index++)); do
        echo "Validating episode ${episode_index}..."
        python -m teleoperation.dataset.validate_realtime_episode \
            "${DATASET_ROOT}" \
            --episode-index "${episode_index}"
    done
fi

