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
STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/adaptive-ngram-sam.XXXXXX")"

cleanup() {
  rm -rf "${STAGING_DIR}"
}
trap cleanup EXIT

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

PYTHONDONTWRITEBYTECODE=1 python3 \
  "${REPO_ROOT}/tools/adaptive_ngram_gate.py" \
  snapshot-source \
  --repo-root "${REPO_ROOT}" \
  --out-dir "${STAGING_DIR}" >/dev/null

IFS=$'\t' read -r \
  SOURCE_COMMIT SOURCE_DIRTY SOURCE_TREE_SHA256 SOURCE_PATCH_SHA256 < <(
  python3 - "${STAGING_DIR}/source_evidence.json" <<'PY'
import json
import sys
from pathlib import Path

evidence = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(
    evidence["base_commit"],
    1 if evidence["dirty"] else 0,
    evidence["tree_sha256"],
    evidence["patch_sha256"],
    sep="\t",
)
PY
)

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
echo "[adaptive-ngram] source_tree_sha256=${SOURCE_TREE_SHA256}"
echo "[adaptive-ngram] source_patch_sha256=${SOURCE_PATCH_SHA256}"
echo "[adaptive-ngram] remote_dir=${REMOTE_DIR}"
echo "[adaptive-ngram] local_out=${LOCAL_OUT}"

"${SSH_CMD[@]}" "${REMOTE_HOST}" \
  "mkdir -p '${REMOTE_DIR}/source'"
tar -C "${STAGING_DIR}/source" -cf - . |
  "${SSH_CMD[@]}" "${REMOTE_HOST}" \
    "tar -C '${REMOTE_DIR}/source' -xf -"
"${SCP_CMD[@]}" \
  "${STAGING_DIR}/source_evidence.json" \
  "${STAGING_DIR}/source.patch" \
  "${REMOTE_HOST}:${REMOTE_DIR}/"

"${SSH_CMD[@]}" "${REMOTE_HOST}" bash -s -- \
  "${REMOTE_DIR}" "${REMOTE_PYTHON}" <<'REMOTE_BASH'
set -u

remote_dir="$1"
remote_python="$2"
source_root="${remote_dir}/source"
source_stdout="${remote_dir}/source_verify.stdout.log"
source_stderr="${remote_dir}/source_verify.stderr.log"
k1_stdout="${remote_dir}/k1_test.stdout.log"
k1_stderr="${remote_dir}/k1_test.stderr.log"

set +e
{
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="${remote_dir}/pycache" \
    PYTHONPATH="${source_root}" \
    "${remote_python}" "${source_root}/tools/adaptive_ngram_gate.py" \
      verify-source \
      --source-root "${source_root}" \
      --evidence "${remote_dir}/source_evidence.json" \
      --patch "${remote_dir}/source.patch" &&
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="${remote_dir}/pycache" \
    PYTHONPATH="${source_root}" \
    "${remote_python}" -m py_compile \
      "${source_root}/tinyvllm/speculative/ngram.py" \
      "${source_root}/tools/profile_ngram_commit.py" \
      "${source_root}/tools/adaptive_ngram_gate.py" \
      "${source_root}/tools/test_ngram_speculative.py" &&
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="${remote_dir}/pycache" \
    PYTHONPATH="${source_root}" \
    "${remote_python}" "${source_root}/tools/profile_ngram_commit.py" \
      --help >/dev/null &&
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="${remote_dir}/pycache" \
    PYTHONPATH="${source_root}" \
    "${remote_python}" "${source_root}/tools/adaptive_ngram_gate.py" \
      --help >/dev/null
} >"${source_stdout}" 2>"${source_stderr}"
source_status=$?

if [[ "${source_status}" == "0" ]]; then
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="${remote_dir}/pycache" \
    PYTHONPATH="${source_root}" \
    "${remote_python}" "${source_root}/tools/test_ngram_speculative.py" \
      >"${k1_stdout}" 2>"${k1_stderr}"
  k1_status=$?
else
  : >"${k1_stdout}"
  printf 'skipped because source verification failed\n' >"${k1_stderr}"
  k1_status=125
fi

"${remote_python}" - \
  "${source_status}" "${k1_status}" \
  "${source_stdout}" "${source_stderr}" \
  "${k1_stdout}" "${k1_stderr}" \
  "${remote_dir}/command_record.json" <<'PY'
import json
import sys
from pathlib import Path

(
    source_status,
    k1_status,
    source_stdout,
    source_stderr,
    k1_stdout,
    k1_stderr,
    output_path,
) = sys.argv[1:]
payload = {
    "source_verify": {
        "returncode": int(source_status),
        "stdout": Path(source_stdout).read_text(
            encoding="utf-8",
            errors="replace",
        ),
        "stderr": Path(source_stderr).read_text(
            encoding="utf-8",
            errors="replace",
        ),
    },
    "k1_test": {
        "command": [
            sys.executable,
            "tools/test_ngram_speculative.py",
        ],
        "returncode": int(k1_status),
        "stdout": Path(k1_stdout).read_text(
            encoding="utf-8",
            errors="replace",
        ),
        "stderr": Path(k1_stderr).read_text(
            encoding="utf-8",
            errors="replace",
        ),
    },
}
Path(output_path).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="${remote_dir}/pycache" \
  PYTHONPATH="${source_root}" \
  "${remote_python}" "${source_root}/tools/adaptive_ngram_gate.py" \
    write-source-preflight \
    --source-root "${source_root}" \
    --evidence "${remote_dir}/source_evidence.json" \
    --patch "${remote_dir}/source.patch" \
    --command-record "${remote_dir}/command_record.json" \
    --out "${remote_dir}/source_preflight.json"
REMOTE_BASH

if [[ "${MODE}" == "preflight" ]]; then
  echo "[adaptive-ngram] remote preflight passed source_tree_sha256=${SOURCE_TREE_SHA256}"
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
  --source-root "${REMOTE_DIR}/source"
  --source-evidence "${REMOTE_DIR}/source_evidence.json"
  --source-patch "${REMOTE_DIR}/source.patch"
  --source-preflight "${REMOTE_DIR}/source_preflight.json"
  --host "${REMOTE_HOST}"
)
if [[ "${RESUME:-0}" == "1" ]]; then
  RUN_ARGS+=(--resume)
fi

REMOTE_GATE_PID="${REMOTE_DIR}/gate.pid"
REMOTE_GATE_EXITCODE="${REMOTE_DIR}/gate.exitcode"
REMOTE_GATE_STDOUT="${REMOTE_DIR}/gate.stdout.log"
REMOTE_GATE_STDERR="${REMOTE_DIR}/gate.stderr.log"

"${SSH_CMD[@]}" "${REMOTE_HOST}" bash -s -- \
  "${REMOTE_DIR}" \
  "${REMOTE_PYTHON}" \
  "${CUDA_DEVICE}" \
  "${RUN_ARGS[@]}" <<'REMOTE_LAUNCH'
set -euo pipefail

remote_dir="$1"
remote_python="$2"
cuda_device="$3"
shift 3
pid_path="${remote_dir}/gate.pid"
exitcode_path="${remote_dir}/gate.exitcode"
stdout_path="${remote_dir}/gate.stdout.log"
stderr_path="${remote_dir}/gate.stderr.log"

if [[ -f "${exitcode_path}" ]]; then
  echo "COMPLETE $(cat "${exitcode_path}")"
  exit 0
fi
if [[ -f "${pid_path}" ]] && kill -0 "$(cat "${pid_path}")" 2>/dev/null; then
  echo "RUNNING $(cat "${pid_path}")"
  exit 0
fi

rm -f "${pid_path}" "${exitcode_path}" "${exitcode_path}.tmp"
nohup bash -c '
  set +e
  source_root="$1"
  remote_python="$2"
  cuda_device="$3"
  remote_dir="$4"
  shift 4
  cd "${source_root}" || exit 125
  CUDA_VISIBLE_DEVICES="${cuda_device}" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="${source_root}" \
    "${remote_python}" tools/adaptive_ngram_gate.py "$@"
  status=$?
  printf "%s\n" "${status}" >"${remote_dir}/gate.exitcode.tmp"
  mv "${remote_dir}/gate.exitcode.tmp" "${remote_dir}/gate.exitcode"
  exit "${status}"
' _ \
  "${remote_dir}/source" \
  "${remote_python}" \
  "${cuda_device}" \
  "${remote_dir}" \
  "$@" \
  </dev/null >"${stdout_path}" 2>"${stderr_path}" &
gate_pid=$!
printf "%s\n" "${gate_pid}" >"${pid_path}"
echo "LAUNCHED ${gate_pid}"
REMOTE_LAUNCH

while true; do
  set +e
  REMOTE_GATE_STATE="$(
    "${SSH_CMD[@]}" "${REMOTE_HOST}" bash -s -- "${REMOTE_DIR}" <<'REMOTE_POLL'
set -euo pipefail
remote_dir="$1"
pid_path="${remote_dir}/gate.pid"
exitcode_path="${remote_dir}/gate.exitcode"
raw_path="${remote_dir}/artifacts/raw_rows.json"
if [[ -f "${exitcode_path}" ]]; then
  echo "COMPLETE $(cat "${exitcode_path}")"
  exit 0
fi
rows=0
if [[ -f "${raw_path}" ]]; then
  rows="$(python3 - "${raw_path}" <<'PY'
import json
import sys
from pathlib import Path

print(len(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))))
PY
)"
fi
if [[ -f "${pid_path}" ]] && kill -0 "$(cat "${pid_path}")" 2>/dev/null; then
  echo "RUNNING $(cat "${pid_path}") rows=${rows}"
else
  echo "LOST rows=${rows}"
fi
REMOTE_POLL
  )"
  POLL_STATUS=$?
  set -e
  if [[ "${POLL_STATUS}" != "0" ]]; then
    echo "[adaptive-ngram] poll transport failed; retrying" >&2
    sleep 10
    continue
  fi
  echo "[adaptive-ngram] remote_gate=${REMOTE_GATE_STATE}"
  case "${REMOTE_GATE_STATE}" in
    COMPLETE\ *)
      REMOTE_RUN_STATUS="${REMOTE_GATE_STATE#COMPLETE }"
      break
      ;;
    RUNNING\ *)
      sleep 20
      ;;
    *)
      echo "[adaptive-ngram] remote gate lost without exitcode; retained ${REMOTE_DIR}" >&2
      exit 3
      ;;
  esac
done

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
