#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env.local"
if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
fi

project_path() {
    local value="$1"
    if [[ -z "${value}" ]]; then
        return 0
    fi
    if [[ "${value}" = /* ]]; then
        printf '%s\n' "${value}"
    elif [[ "${value}" == "~"* ]]; then
        printf '%s\n' "${value/#\~/${HOME}}"
    else
        printf '%s/%s\n' "${PROJECT_ROOT}" "${value}"
    fi
}

ROBOT_RUNTIME_ROOT="${ROBOT_RUNTIME_ROOT:-${PIPER_RUNTIME_ROOT:-${PROJECT_ROOT}/vendor/lerobot_piper}}"
POLICY_RUNTIME_SCRIPT="${POLICY_RUNTIME_SCRIPT:-${PIPER_OPENPI_RUNTIME:-${PROJECT_ROOT}/runtime/piper_openpi_runtime.py}}"
OPENPI_ROOT="${OPENPI_ROOT:-${PROJECT_ROOT}/external/openpi}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/pi05_stack_bowls}"
NORM_STATS_PATH="${NORM_STATS_PATH:-}"

LOCAL_OPENPI_PYTHON="${OPENPI_ROOT}/.venv/bin/python"

is_usable_python() {
    local candidate="$1"
    local resolved="${candidate}"
    if [[ ! -x "${resolved}" ]]; then
        resolved="$(command -v "${candidate}" 2>/dev/null || true)"
    fi
    [[ -n "${resolved}" && -x "${resolved}" ]] && "${resolved}" -c "import sys; print(sys.version)" >/dev/null 2>&1
}

if is_usable_python "${LOCAL_OPENPI_PYTHON}"; then
    DEFAULT_PYTHON_BIN="${LOCAL_OPENPI_PYTHON}"
elif command -v python3.11 >/dev/null 2>&1; then
    DEFAULT_PYTHON_BIN="$(command -v python3.11)"
else
    DEFAULT_PYTHON_BIN="python3"
fi

PYTHON_BIN="${OPENPI_PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"

PIPER_CAN_NAME="${PIPER_CAN_NAME:-can1}"
DEVICE="${DEVICE:-cuda}"
PROMPT="${PROMPT:-stack the yellow bowl on the green bowl}"
TASK="${TASK:-${PROMPT}}"
FPS="${FPS:-10}"
ACTIONS_PER_INFERENCE="${ACTIONS_PER_INFERENCE:-8}"
SYNC_INFERENCE="${SYNC_INFERENCE:-true}"
PREFETCH_THRESHOLD="${PREFETCH_THRESHOLD:-2}"
ACTION_SCALE="${ACTION_SCALE:-1.0}"
MAX_ABS_DELTA="${MAX_ABS_DELTA:-0.04}"
NUM_EPISODES="${NUM_EPISODES:-1}"
NEXT_EPISODE_KEY="${NEXT_EPISODE_KEY:-n}"
GUI="${GUI:-false}"
GUI_BACKEND="${GUI_BACKEND:-web}"
GUI_HOST="${GUI_HOST:-127.0.0.1}"
GUI_PORT="${GUI_PORT:-8765}"
GUI_WINDOW_NAME="${GUI_WINDOW_NAME:-Interactive Piper DAgger Collection}"
ENABLE_ARM="${ENABLE_ARM:-false}"
DRY_RUN="${DRY_RUN:-false}"
VALIDATE_ONLY="${VALIDATE_ONLY:-false}"
DEBUG_ACTION_DETAILS="${DEBUG_ACTION_DETAILS:-false}"
PRINT_COMMAND_COUNTS="${PRINT_COMMAND_COUNTS:-false}"
DISABLE_ARM_ON_EXIT="${DISABLE_ARM_ON_EXIT:-false}"
COLLECTION_REPO_ID="${COLLECTION_REPO_ID:-${DATASET_REPO_ID:-local/piper_correction_collection}}"
COLLECTION_OUTPUT_ROOT="${COLLECTION_OUTPUT_ROOT:-${DATASET_ROOT:-${PROJECT_ROOT}/outputs/datasets/piper_correction_$(date +%Y%m%d_%H%M%S)}}"
LOG_EVERY="${LOG_EVERY:-1}"

ROBOT_RUNTIME_ROOT="$(project_path "${ROBOT_RUNTIME_ROOT}")"
POLICY_RUNTIME_SCRIPT="$(project_path "${POLICY_RUNTIME_SCRIPT}")"
OPENPI_ROOT="$(project_path "${OPENPI_ROOT}")"
CHECKPOINT_DIR="$(project_path "${CHECKPOINT_DIR}")"
COLLECTION_OUTPUT_ROOT="$(project_path "${COLLECTION_OUTPUT_ROOT}")"
if [[ -n "${NORM_STATS_PATH}" ]]; then
    NORM_STATS_PATH="$(project_path "${NORM_STATS_PATH}")"
fi

HEAD_CAMERA_SERIAL="${HEAD_CAMERA_SERIAL:-254622073267}"
LEFT_WRIST_CAMERA_SERIAL="${LEFT_WRIST_CAMERA_SERIAL:-244222071617}"
RIGHT_WRIST_CAMERA_SERIAL="${RIGHT_WRIST_CAMERA_SERIAL:-317622070857}"
FRONT_VIEW_CAMERA_SERIAL="${FRONT_VIEW_CAMERA_SERIAL:-254622079402}"
HEAD_CAMERA_ROTATION="${HEAD_CAMERA_ROTATION:-none}"
LEFT_WRIST_CAMERA_ROTATION="${LEFT_WRIST_CAMERA_ROTATION:-none}"
RIGHT_WRIST_CAMERA_ROTATION="${RIGHT_WRIST_CAMERA_ROTATION:-none}"
FRONT_VIEW_CAMERA_ROTATION="${FRONT_VIEW_CAMERA_ROTATION:-none}"
CAMERA_FPS="${CAMERA_FPS:-15}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"

if ! is_usable_python "${PYTHON_BIN}"; then
    echo "Error: Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
    echo "Set OPENPI_PYTHON_BIN to a valid Python 3.11 binary." >&2
    exit 1
fi

if [[ ! -d "${ROBOT_RUNTIME_ROOT}/lerobot" ]]; then
    echo "Error: LeRobot-Piper runtime package does not exist: ${ROBOT_RUNTIME_ROOT}/lerobot" >&2
    echo "Set ROBOT_RUNTIME_ROOT, or place the bundled runtime at vendor/lerobot_piper." >&2
    exit 1
fi

if [[ ! -f "${POLICY_RUNTIME_SCRIPT}" ]]; then
    echo "Error: policy runtime script does not exist: ${POLICY_RUNTIME_SCRIPT}" >&2
    echo "Set POLICY_RUNTIME_SCRIPT if you use a custom runtime adapter." >&2
    exit 1
fi

if [[ ! -d "${OPENPI_ROOT}" ]]; then
    echo "Error: OpenPI root does not exist: ${OPENPI_ROOT}" >&2
    echo "Set OPENPI_ROOT, or clone the reference OpenPI checkout to external/openpi." >&2
    exit 1
fi

for required_openpi_path in "src/openpi" "packages/openpi-client/src"; do
    if [[ ! -d "${OPENPI_ROOT}/${required_openpi_path}" ]]; then
        echo "Error: missing OpenPI source path: ${OPENPI_ROOT}/${required_openpi_path}" >&2
        echo "Set OPENPI_ROOT to a checkout containing src/openpi and packages/openpi-client/src." >&2
        exit 1
    fi
done

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "Error: checkpoint directory does not exist: ${CHECKPOINT_DIR}" >&2
    echo "Set CHECKPOINT_DIR to a Pi0.5/OpenPI checkpoint directory containing model.safetensors and metadata.pt." >&2
    exit 1
fi

for required_checkpoint_file in "model.safetensors" "metadata.pt"; do
    if [[ ! -f "${CHECKPOINT_DIR}/${required_checkpoint_file}" ]]; then
        echo "Error: missing checkpoint file: ${CHECKPOINT_DIR}/${required_checkpoint_file}" >&2
        echo "Set CHECKPOINT_DIR to an inference checkpoint directory with model.safetensors and metadata.pt." >&2
        exit 1
    fi
done

if [[ -z "${NORM_STATS_PATH}" ]]; then
    shopt -s globstar nullglob
    norm_stats_candidates=("${CHECKPOINT_DIR}"/assets/**/norm_stats.json)
    shopt -u globstar nullglob
    if [[ ${#norm_stats_candidates[@]} -eq 0 ]]; then
        echo "Error: could not find norm_stats.json under ${CHECKPOINT_DIR}/assets." >&2
        echo "Set NORM_STATS_PATH, or include assets/<asset_id>/norm_stats.json in CHECKPOINT_DIR." >&2
        exit 1
    fi
fi

if [[ -n "${NORM_STATS_PATH}" && ! -f "${NORM_STATS_PATH}" ]]; then
    echo "Error: NORM_STATS_PATH does not exist: ${NORM_STATS_PATH}" >&2
    exit 1
fi

ARGS=(
    "${PROJECT_ROOT}/scripts/collect_interactive_dagger.py"
    "--runtime-script" "${POLICY_RUNTIME_SCRIPT}"
    "--lerobot-runtime-root" "${ROBOT_RUNTIME_ROOT}"
    "--openpi-root" "${OPENPI_ROOT}"
    "--checkpoint-dir" "${CHECKPOINT_DIR}"
    "--can-name" "${PIPER_CAN_NAME}"
    "--device" "${DEVICE}"
    "--prompt" "${PROMPT}"
    "--task" "${TASK}"
    "--fps" "${FPS}"
    "--actions-per-inference" "${ACTIONS_PER_INFERENCE}"
    "--prefetch-threshold" "${PREFETCH_THRESHOLD}"
    "--action-scale" "${ACTION_SCALE}"
    "--max-abs-delta" "${MAX_ABS_DELTA}"
    "--num-episodes" "${NUM_EPISODES}"
    "--next-episode-key" "${NEXT_EPISODE_KEY}"
    "--gui-backend" "${GUI_BACKEND}"
    "--gui-host" "${GUI_HOST}"
    "--gui-port" "${GUI_PORT}"
    "--gui-window-name" "${GUI_WINDOW_NAME}"
    "--repo-id" "${COLLECTION_REPO_ID}"
    "--dataset-root" "${COLLECTION_OUTPUT_ROOT}"
    "--head-camera-serial" "${HEAD_CAMERA_SERIAL}"
    "--left-wrist-camera-serial" "${LEFT_WRIST_CAMERA_SERIAL}"
    "--right-wrist-camera-serial" "${RIGHT_WRIST_CAMERA_SERIAL}"
    "--front-view-camera-serial" "${FRONT_VIEW_CAMERA_SERIAL}"
    "--head-camera-rotation" "${HEAD_CAMERA_ROTATION}"
    "--left-wrist-camera-rotation" "${LEFT_WRIST_CAMERA_ROTATION}"
    "--right-wrist-camera-rotation" "${RIGHT_WRIST_CAMERA_ROTATION}"
    "--front-view-camera-rotation" "${FRONT_VIEW_CAMERA_ROTATION}"
    "--camera-fps" "${CAMERA_FPS}"
    "--camera-width" "${CAMERA_WIDTH}"
    "--camera-height" "${CAMERA_HEIGHT}"
    "--log-every" "${LOG_EVERY}"
)

if [[ -n "${NORM_STATS_PATH}" ]]; then
    ARGS+=("--norm-stats-path" "${NORM_STATS_PATH}")
fi

if [[ "${ENABLE_ARM}" == "true" ]]; then
    ARGS+=("--enable-arm")
fi

if [[ "${GUI}" == "true" ]]; then
    ARGS+=("--gui")
fi

if [[ "${SYNC_INFERENCE}" == "true" ]]; then
    ARGS+=("--sync-inference")
fi

if [[ "${DRY_RUN}" == "true" ]]; then
    ARGS+=("--dry-run")
fi

if [[ "${VALIDATE_ONLY}" == "true" ]]; then
    ARGS+=("--validate-only")
fi

if [[ "${DEBUG_ACTION_DETAILS}" == "true" ]]; then
    ARGS+=("--debug-action-details")
fi

if [[ "${PRINT_COMMAND_COUNTS}" == "true" ]]; then
    ARGS+=("--print-command-counts")
fi

if [[ "${DISABLE_ARM_ON_EXIT}" == "true" ]]; then
    ARGS+=("--disable-arm-on-exit")
fi

exec "${PYTHON_BIN}" "${ARGS[@]}" "$@"
