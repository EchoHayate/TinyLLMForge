#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_BASE="${REMOTE_BASE:-/data00/home/sitian/sitian-workspace01/tllm}"
REMOTE_PYTHON="${REMOTE_PYTHON:-${REMOTE_BASE}/env/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B}"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)-$$}"
REMOTE_TREE="${REMOTE_BASE}/prefix-cache-gate-${TAG}"
REMOTE_OUT="${REMOTE_TREE}/gate_out"
REMOTE_TMP="${REMOTE_TREE}/tmp"
LOCAL_OUT="${LOCAL_OUT:-${REPO_ROOT}/experiments/prefix_cache/qwen3_0_6b_gate_${TAG}}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-3}"
TINYVLLM_PORT="${TINYVLLM_DIST_PORT:-$((25000 + RANDOM % 12000))}"
MASTER_PORT_VALUE="${MASTER_PORT:-$((37000 + RANDOM % 12000))}"
REPETITIONS="${REPETITIONS:-7}"
WARMUP_REPETITIONS="${WARMUP_REPETITIONS:-2}"
CONTROL_SOCKET="${CONTROL_SOCKET:-/tmp/ssh-sitian-10.232.195.203}"

if [[ "${TINYVLLM_PORT}" == "${MASTER_PORT_VALUE}" ]]; then
  MASTER_PORT_VALUE=$((MASTER_PORT_VALUE + 1))
fi

SSH_ARGS=(-o BatchMode=yes)
RSYNC_SSH="ssh -o BatchMode=yes"
if [[ -S "${CONTROL_SOCKET}" ]]; then
  SSH_ARGS+=(-S "${CONTROL_SOCKET}")
  RSYNC_SSH+=" -S ${CONTROL_SOCKET}"
fi

mkdir -p "${LOCAL_OUT}"

ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" \
  "set -euo pipefail; \
   test -x '${REMOTE_PYTHON}'; \
   test -f '${MODEL_PATH}/config.json'; \
   mkdir -p '${REMOTE_TREE}'; \
   test -z \"\$(find '${REMOTE_TREE}' -mindepth 1 -maxdepth 1 -print -quit)\"; \
   mkdir -p '${REMOTE_TMP}'"

git ls-files -z \
  | rsync -a --from0 --files-from=- -e "${RSYNC_SSH}" \
      ./ "${REMOTE_HOST}:${REMOTE_TREE}/"

ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" \
  "set -euo pipefail
   cd '${REMOTE_TREE}'
   export PYTHONPATH='${REMOTE_TREE}'
   export CUDA_VISIBLE_DEVICES='${CUDA_DEVICE}'
   export TINYVLLM_DIST_PORT='${TINYVLLM_PORT}'
   export MASTER_PORT='${MASTER_PORT_VALUE}'
   export PYTHONDONTWRITEBYTECODE=1
   export TMPDIR='${REMOTE_TMP}'
   '${REMOTE_PYTHON}' -m py_compile \
     tinyvllm/engine/block_manager.py \
     tinyvllm/engine/scheduler.py \
     tools/profile_prefix_cache.py \
     tools/test_profile_prefix_cache.py
   '${REMOTE_PYTHON}' tools/test_chunked_prefill.py
   '${REMOTE_PYTHON}' tools/test_profile_prefix_cache.py
   '${REMOTE_PYTHON}' tools/profile_prefix_cache.py \
     --model '${MODEL_PATH}' \
     --mode full \
     --out-dir '${REMOTE_OUT}' \
     --shared-prefix-tokens 256,1024,2048 \
     --suffix-tokens 64 \
     --repetitions '${REPETITIONS}' \
     --warmup-repetitions '${WARMUP_REPETITIONS}' \
     --enforce-eager"

rsync -a -e "${RSYNC_SSH}" \
  "${REMOTE_HOST}:${REMOTE_OUT}/" "${LOCAL_OUT}/"

python3 - "${LOCAL_OUT}" "${REPETITIONS}" <<'PY'
import json
import sys
from copy import deepcopy
from pathlib import Path

from tools.profile_prefix_cache import decide_gate

root = Path(sys.argv[1])
repetitions = int(sys.argv[2])
required_files = {
    "manifest.json",
    "correctness_rows.json",
    "performance_rows.json",
    "summary.json",
    "report.md",
}
missing_files = sorted(name for name in required_files if not (root / name).is_file())
if missing_files:
    raise SystemExit(f"missing artifact files: {missing_files}")
manifest = json.loads((root / "manifest.json").read_text())
correctness = json.loads((root / "correctness_rows.json").read_text())
summary = json.loads((root / "summary.json").read_text())
required = {
    "repeat_255",
    "repeat_256",
    "repeat_257",
    "repeat_512",
    "repeat_513",
    "same_batch_p_q_p",
    "shared_prefix_different_suffix",
    "cache_cleared",
}
seen = {row["case"] for row in correctness}
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"missing correctness cases: {missing}")
for path in (
    "tinyvllm/engine/block_manager.py",
    "tinyvllm/engine/scheduler.py",
    "tools/profile_prefix_cache.py",
):
    digest = manifest["source_sha256"].get(path, "")
    if len(digest) != 64:
        raise SystemExit(f"invalid source hash for {path}: {digest!r}")
decision = summary.get("decision", {}).get("decision")
if decision not in {"GO", "NO_GO"}:
    raise SystemExit(f"invalid gate decision: {decision!r}")
recomputed = decide_gate(
    deepcopy(summary.get("correctness_rows", [])),
    deepcopy(summary.get("performance_cases", [])),
)
if recomputed != summary.get("decision"):
    raise SystemExit(
        "summary decision does not match recomputed gate: "
        f"stored={summary.get('decision')!r} recomputed={recomputed!r}"
    )
performance_cases = summary.get("performance_cases", [])
prefixes = {case.get("shared_prefix_tokens") for case in performance_cases}
if not {256, 1024, 2048} <= prefixes:
    raise SystemExit(f"missing performance prefixes: {sorted({256, 1024, 2048} - prefixes)}")
for case in performance_cases:
    for state in ("cold", "warm", "cache_cleared"):
        samples = case.get(state, {}).get("samples")
        if samples != repetitions:
            raise SystemExit(
                f"{case.get('shared_prefix_tokens')} {state} samples "
                f"{samples!r} != {repetitions}"
            )
print("PREFIX_CACHE_GATE_ARTIFACTS_OK")
PY

echo "prefix cache remote gate completed"
echo "remote_tree=${REMOTE_TREE}"
echo "local_out=${LOCAL_OUT}"
echo "tinyvllm_dist_port=${TINYVLLM_PORT}"
echo "master_port=${MASTER_PORT_VALUE}"
