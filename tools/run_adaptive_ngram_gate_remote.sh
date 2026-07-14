#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/sitian-workspace01/tllm/env/bin/python}"
REMOTE_BASE="${REMOTE_BASE:-/data00/home/sitian/sitian-workspace01/tllm/adaptive-ngram-gates}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-7}"
SSH_CONTROL_PATH="${SSH_CONTROL_PATH:-/tmp/ssh-sitian-10.232.195.203}"
MODE="${1:-smoke}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)-$$}"
REMOTE_DIR="${REMOTE_BASE}/${RUN_TAG}"
LOCAL_OUT="${LOCAL_OUT:-${REPO_ROOT}/experiments/adaptive_ngram/${RUN_TAG}}"
BASE_SEED="${BASE_SEED:-20260714}"

SSH_CMD=(ssh)
SCP_CMD=(scp)
if [[ -S "${SSH_CONTROL_PATH}" ]]; then
  SSH_CMD+=(-S "${SSH_CONTROL_PATH}" -o BatchMode=yes)
  SCP_CMD+=(-o "ControlPath=${SSH_CONTROL_PATH}" -o BatchMode=yes)
fi

case "${MODE}" in
  preflight)
    REPETITIONS=1
    ;;
  smoke)
    REPETITIONS=1
    ;;
  canonical)
    REPETITIONS=7
    ;;
  *)
    echo "usage: $0 {preflight|smoke|canonical}" >&2
    exit 2
    ;;
esac

SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  SOURCE_DIRTY=1
else
  SOURCE_DIRTY=0
fi

discover_model() {
  "${SSH_CMD[@]}" "${REMOTE_HOST}" "${REMOTE_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

roots = [
    Path("/data00/home/sitian/sitian-workspace01/.ms_cache"),
    Path("/data00/home/sitian/sitian-workspace01"),
]
skip_names = {".git", "env", "__pycache__", "node_modules"}
candidates = []
seen = set()
for root in roots:
    if not root.exists():
        continue
    for current, dirs, files in os.walk(root):
        dirs[:] = [
            name for name in dirs
            if name not in skip_names and not name.startswith(".cache")
        ]
        if "config.json" not in files:
            continue
        config_path = Path(current) / "config.json"
        try:
            payload = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        normalized = str(config_path.parent).lower().replace("_", "").replace("-", "").replace(".", "")
        if (
            str(payload.get("model_type", "")).lower() == "qwen3"
            and "qwen3" in normalized
            and ("06b" in normalized or payload.get("hidden_size") == 1024)
        ):
            resolved = str(config_path.parent.resolve())
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(resolved)

candidates.sort()
print(json.dumps(candidates))
PY
}

validate_model() {
  local model_path="$1"
  "${SSH_CMD[@]}" "${REMOTE_HOST}" "${REMOTE_PYTHON}" - "${model_path}" <<'PY'
import json
import sys
from pathlib import Path

model = Path(sys.argv[1])
config_path = model / "config.json"
if not config_path.is_file():
    raise SystemExit(f"missing model config: {config_path}")
config = json.loads(config_path.read_text())
normalized = str(model).lower().replace("_", "").replace("-", "").replace(".", "")
if config.get("model_type") != "qwen3":
    raise SystemExit(f"model_type is not qwen3: {config.get('model_type')}")
if "qwen3" not in normalized or ("06b" not in normalized and config.get("hidden_size") != 1024):
    raise SystemExit(f"model is not identifiable as Qwen3-0.6B: {model}")
print(str(model.resolve()))
PY
}

if [[ -n "${MODEL_PATH:-}" ]]; then
  RESOLVED_MODEL_PATH="$(validate_model "${MODEL_PATH}")"
else
  DISCOVERED_JSON="$(discover_model)"
  RESOLVED_MODEL_PATH="$("${REMOTE_PYTHON_LOCAL:-python3}" - "${DISCOVERED_JSON}" <<'PY'
import json
import sys

candidates = json.loads(sys.argv[1])
if not candidates:
    raise SystemExit("no Qwen3-0.6B model path discovered")
if len(candidates) > 1:
    print("discovered candidates:", file=sys.stderr)
    for candidate in candidates:
        print(candidate, file=sys.stderr)
    raise SystemExit("multiple Qwen3-0.6B model paths discovered; set MODEL_PATH explicitly")
print(candidates[0])
PY
)"
  RESOLVED_MODEL_PATH="$(validate_model "${RESOLVED_MODEL_PATH}")"
fi

echo "[adaptive-ngram] host=${REMOTE_HOST}"
echo "[adaptive-ngram] python=${REMOTE_PYTHON}"
echo "[adaptive-ngram] model=${RESOLVED_MODEL_PATH}"
echo "[adaptive-ngram] source_commit=${SOURCE_COMMIT} dirty=${SOURCE_DIRTY}"
echo "[adaptive-ngram] remote_dir=${REMOTE_DIR}"
echo "[adaptive-ngram] local_out=${LOCAL_OUT}"

"${SSH_CMD[@]}" "${REMOTE_HOST}" "mkdir -p '${REMOTE_DIR}/tools'"
tar -C "${REPO_ROOT}" -cf - \
  tinyvllm \
  tools/draft_model_schema.py \
  tools/profile_ngram_commit.py \
  tools/adaptive_ngram_gate.py |
  "${SSH_CMD[@]}" "${REMOTE_HOST}" "tar -C '${REMOTE_DIR}' -xf -"

"${SSH_CMD[@]}" "${REMOTE_HOST}" \
  "cd '${REMOTE_DIR}' && \
   PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${REMOTE_DIR}' \
   '${REMOTE_PYTHON}' -m py_compile \
     tinyvllm/speculative/ngram.py \
     tools/profile_ngram_commit.py \
     tools/adaptive_ngram_gate.py && \
   PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${REMOTE_DIR}' \
   '${REMOTE_PYTHON}' tools/profile_ngram_commit.py --help >/dev/null && \
   PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${REMOTE_DIR}' \
   '${REMOTE_PYTHON}' tools/adaptive_ngram_gate.py --help >/dev/null"

if [[ "${MODE}" == "preflight" ]]; then
  echo "[adaptive-ngram] remote preflight passed"
  exit 0
fi

REMOTE_OUT="${REMOTE_DIR}/artifacts"
RUN_ARGS=(
  run
  --out-dir "${REMOTE_OUT}"
  --python-bin "${REMOTE_PYTHON}"
  --model-path "${RESOLVED_MODEL_PATH}"
  --repetitions "${REPETITIONS}"
  --base-seed "${BASE_SEED}"
  --source-commit "${SOURCE_COMMIT}"
  --host "${REMOTE_HOST}"
)
if [[ "${SOURCE_DIRTY}" == "1" ]]; then
  RUN_ARGS+=(--source-dirty)
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  RUN_ARGS+=(--resume)
fi

set +e
"${SSH_CMD[@]}" "${REMOTE_HOST}" \
  "cd '${REMOTE_DIR}' && \
   CUDA_VISIBLE_DEVICES='${CUDA_DEVICE}' \
   PYTHONDONTWRITEBYTECODE=1 \
   PYTHONPATH='${REMOTE_DIR}' \
   '${REMOTE_PYTHON}' tools/adaptive_ngram_gate.py $(printf "'%s' " "${RUN_ARGS[@]}")"
REMOTE_RUN_STATUS=$?
set -e

mkdir -p "${LOCAL_OUT}"
"${SCP_CMD[@]}" -r "${REMOTE_HOST}:${REMOTE_OUT}/." "${LOCAL_OUT}/"

if [[ -f "${LOCAL_OUT}/manifest.json" && -f "${LOCAL_OUT}/raw_rows.json" ]]; then
  PYTHONDONTWRITEBYTECODE=1 python3 "${REPO_ROOT}/tools/adaptive_ngram_gate.py" \
    verify \
    --out-dir "${LOCAL_OUT}"
fi

if [[ "${REMOTE_RUN_STATUS}" != "0" ]]; then
  echo "[adaptive-ngram] remote gate exited ${REMOTE_RUN_STATUS}; retained ${REMOTE_DIR}" >&2
  exit "${REMOTE_RUN_STATUS}"
fi

echo "[adaptive-ngram] completed ${MODE}; artifacts=${LOCAL_OUT}"
