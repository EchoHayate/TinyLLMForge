#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
CONTROL_SOCKET="${CONTROL_SOCKET:-/tmp/ssh-sitian-10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/sitian-workspace01/tllm/env/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B}"
REMOTE_REPO="${REMOTE_REPO:-/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge}"
GPU_ID="${GPU_ID:-0}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOCAL_OUT="${LOCAL_OUT:-artifacts/speculative_residency_boundary/${RUN_TAG}}"
REMOTE_OUT="${REMOTE_OUT:-/tmp/tinyllmforge-speculative-residency-boundary-${RUN_TAG}}"

SSH_OPTIONS=(
  -S "${CONTROL_SOCKET}"
  -o BatchMode=yes
)

SYNC_PATHS=(
  tinyvllm/
  tools/speculative_residency_boundary_gate.py
  tools/verify_speculative_residency_boundary_gate.py
  tools/test_speculative_residency_boundary_gate.py
  tools/run_speculative_residency_boundary_gate_remote.sh
)

for sync_path in "${SYNC_PATHS[@]}"; do
  if [[ ! -e "${sync_path}" ]]; then
    printf 'missing sync path: %s\n' "${sync_path}" >&2
    exit 2
  fi
done

mkdir -p "${LOCAL_OUT}"

ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}" true

rsync -av \
  -e "ssh -S ${CONTROL_SOCKET} -o BatchMode=yes" \
  --relative \
  "${SYNC_PATHS[@]}" \
  "${REMOTE_HOST}:${REMOTE_REPO}/"

set +e
ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}" bash -s -- \
  "${REMOTE_REPO}" \
  "${REMOTE_PYTHON}" \
  "${MODEL_PATH}" \
  "${GPU_ID}" \
  "${REMOTE_OUT}" <<'REMOTE_SCRIPT'
set -euo pipefail

remote_repo="$1"
remote_python="$2"
model_path="$3"
gpu_id="$4"
remote_out="$5"

mkdir -p "${remote_out}"
cd "${remote_repo}"

CUDA_VISIBLE_DEVICES="${gpu_id}" \
PYTHONPATH="${remote_repo}" \
"${remote_python}" \
  tools/speculative_residency_boundary_gate.py run \
  --model "${model_path}" \
  --out "${remote_out}/result.json" \
  >"${remote_out}/remote.log" 2>&1

PYTHONPATH="${remote_repo}" \
"${remote_python}" \
  tools/verify_speculative_residency_boundary_gate.py \
  "${remote_out}/result.json" \
  "${remote_repo}" \
  --output "${remote_out}/verify.remote.json"
REMOTE_SCRIPT
remote_status=$?
set -e

rsync -av \
  -e "ssh -S ${CONTROL_SOCKET} -o BatchMode=yes" \
  "${REMOTE_HOST}:${REMOTE_OUT}/" \
  "${LOCAL_OUT}/"

if [[ "${remote_status}" -ne 0 ]]; then
  if [[ -f "${LOCAL_OUT}/remote.log" ]]; then
    cat "${LOCAL_OUT}/remote.log" >&2
  fi
  exit "${remote_status}"
fi

python3 tools/verify_speculative_residency_boundary_gate.py \
  "${LOCAL_OUT}/result.json" \
  . \
  --output "${LOCAL_OUT}/verify.json"

printf 'speculative residency boundary artifacts: %s\n' "${LOCAL_OUT}"
