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

URL="${1:-http://127.0.0.1:${GUI_PORT:-8765}}"
TITLE="${COLLECTION_UI_TITLE:-Interactive Piper DAgger Collection}"
CHROME_BIN="${BROWSER_BIN:-${CHROME_BIN:-}}"
UI_LOG_DIR="${UI_LOG_DIR:-${TMPDIR:-/tmp}}"

resolve_executable() {
    local candidate="$1"
    if [[ -x "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
    fi
    command -v "${candidate}" 2>/dev/null
}

if [[ -z "${CHROME_BIN}" ]]; then
    if command -v google-chrome >/dev/null 2>&1; then
        CHROME_BIN="$(command -v google-chrome)"
    elif command -v chromium >/dev/null 2>&1; then
        CHROME_BIN="$(command -v chromium)"
    elif command -v chromium-browser >/dev/null 2>&1; then
        CHROME_BIN="$(command -v chromium-browser)"
    else
        CHROME_BIN=""
    fi
elif ! CHROME_BIN="$(resolve_executable "${CHROME_BIN}")"; then
    echo "Error: browser executable not found or not executable." >&2
    echo "Set BROWSER_BIN or CHROME_BIN to Chrome/Chromium, or install xdg-open for fallback." >&2
    exit 1
fi

if [[ -n "${CHROME_BIN}" ]]; then
    "${CHROME_BIN}" --new-window --app="${URL}" >"${UI_LOG_DIR}/correction_collection_ui_chrome.log" 2>&1 &
else
    if ! command -v xdg-open >/dev/null 2>&1; then
        echo "Error: no Chrome/Chromium executable was found and xdg-open is unavailable." >&2
        echo "Set BROWSER_BIN or CHROME_BIN to a browser executable." >&2
        exit 1
    fi
    xdg-open "${URL}" >"${UI_LOG_DIR}/correction_collection_ui_xdg_open.log" 2>&1 &
fi

if ! command -v wmctrl >/dev/null 2>&1; then
    echo "Opened ${URL}."
    echo "wmctrl is not installed, so this script cannot set the window always-on-top automatically."
    echo "Install it once with: sudo apt install wmctrl"
    echo "Manual fallback: focus the UI window, press Alt+Space, then choose 'Always on Top' if your desktop supports it."
    exit 0
fi

for _ in $(seq 1 40); do
    if wmctrl -l | grep -F "${TITLE}" >/dev/null 2>&1; then
        wmctrl -r "${TITLE}" -b add,above
        echo "Opened ${URL} and set '${TITLE}' always-on-top."
        exit 0
    fi
    sleep 0.25
done

echo "Opened ${URL}, but did not find a window title containing: ${TITLE}"
echo "Visible windows:"
wmctrl -l || true
echo "Try manually: focus the UI window, press Alt+Space, then choose 'Always on Top'."
