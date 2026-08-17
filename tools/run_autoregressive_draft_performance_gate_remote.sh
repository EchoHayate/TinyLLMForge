#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/miniconda3/envs/py311/bin/python}"
REMOTE_BASE="${REMOTE_BASE:-/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815}"
REMOTE_PACKAGE_ROOT="${REMOTE_PACKAGE_ROOT:-${REMOTE_BASE}/run_packages}"
TARGET_MODEL="${TARGET_MODEL:-${REMOTE_BASE}/target-qwen3-1.7b}"
DRAFT_MODEL="${DRAFT_MODEL:-${REMOTE_BASE}/draft}"
GPU_INDICES="${GPU_INDICES:-3,4,6,7}"
DIST_PORT="${DIST_PORT:-29631}"
MASTER_PORT="${MASTER_PORT:-29731}"
HARD_TIMEOUT_SECONDS="${HARD_TIMEOUT_SECONDS:-3600}"
RUN_TAG="${RUN_TAG:-tp4-qwen3-controlled-performance-$(date -u +%Y%m%dT%H%M%SZ)}"
LOCAL_RUN="${LOCAL_RUN:-${REPO_ROOT}/experiments/autoregressive_draft/${RUN_TAG}}"
REMOTE_RUN="${REMOTE_RUN:-${REMOTE_BASE}/run/${RUN_TAG}}"
REMOTE_SOURCE="${REMOTE_RUN}/source"
REMOTE_ARTIFACTS="${REMOTE_RUN}/artifacts"

SSH_OPTIONS=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ControlMaster=no
  -o ControlPath=none
  -o GSSAPIAuthentication=yes
)
SSH=(ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}")
RSYNC_SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none -o GSSAPIAuthentication=yes"

if [[ -e "${LOCAL_RUN}" ]]; then
  printf 'refusing to overwrite local run: %s\n' "${LOCAL_RUN}" >&2
  exit 2
fi
mkdir -p "${LOCAL_RUN}"

SOURCE_PATHS=(
  tinyvllm
  tools/autoregressive_draft_tp1_engine_gate.py
  tools/autoregressive_draft_tp4_engine_gate.py
  tools/autoregressive_draft_tp4_local_gate.py
  tools/autoregressive_draft_performance_gate.py
  tools/autoregressive_draft_performance_worker.py
  tools/speculative_runtime_performance_gate.py
  tools/verify_autoregressive_draft_performance_gate.py
  tools/test_autoregressive_draft_performance_gate.py
  tools/run_autoregressive_draft_performance_gate_remote.sh
)

for source_path in "${SOURCE_PATHS[@]}"; do
  if [[ ! -e "${REPO_ROOT}/${source_path}" ]]; then
    printf 'missing source path: %s\n' "${source_path}" >&2
    exit 2
  fi
done

(
  cd "${REPO_ROOT}"
  COPYFILE_DISABLE=1 tar --no-xattrs -cf "${LOCAL_RUN}/source.tar" \
    "${SOURCE_PATHS[@]}"
)

cat > "${LOCAL_RUN}/command.txt" <<EOF
REMOTE_HOST=${REMOTE_HOST}
REMOTE_PYTHON=${REMOTE_PYTHON}
REMOTE_PACKAGE_ROOT=${REMOTE_PACKAGE_ROOT}
TARGET_MODEL=${TARGET_MODEL}
DRAFT_MODEL=${DRAFT_MODEL}
GPU_INDICES=${GPU_INDICES}
DIST_PORT=${DIST_PORT}
MASTER_PORT=${MASTER_PORT}
HARD_TIMEOUT_SECONDS=${HARD_TIMEOUT_SECONDS}
REMOTE_RUN=${REMOTE_RUN}
EOF

"${SSH[@]}" \
  "if [[ -e '${REMOTE_RUN}' ]]; then
     printf 'refusing to overwrite remote run: %s\n' '${REMOTE_RUN}' >&2
     exit 2
   fi
   mkdir -p '${REMOTE_SOURCE}' '${REMOTE_ARTIFACTS}'"

rsync -a -e "${RSYNC_SSH}" \
  "${LOCAL_RUN}/source.tar" \
  "${REMOTE_HOST}:${REMOTE_RUN}/source.tar"

"${SSH[@]}" \
  "tar -xf '${REMOTE_RUN}/source.tar' -C '${REMOTE_SOURCE}'"

set +e
"${SSH[@]}" bash -s -- \
  "${REMOTE_SOURCE}" \
  "${REMOTE_ARTIFACTS}" \
  "${REMOTE_PYTHON}" \
  "${REMOTE_PACKAGE_ROOT}" \
  "${TARGET_MODEL}" \
  "${DRAFT_MODEL}" \
  "${GPU_INDICES}" \
  "${DIST_PORT}" \
  "${MASTER_PORT}" \
  "${HARD_TIMEOUT_SECONDS}" <<'REMOTE_SCRIPT'
set -uo pipefail

remote_source="$1"
remote_artifacts="$2"
remote_python="$3"
remote_package_root="$4"
target_model="$5"
draft_model="$6"
gpu_indices="$7"
dist_port="$8"
master_port="$9"
hard_timeout_seconds="${10}"

cd "${remote_source}" || exit 2
date -u +%Y-%m-%dT%H:%M:%SZ > "${remote_artifacts}/started_at_utc.txt"
nvidia-smi > "${remote_artifacts}/gpu-before.txt" 2>&1

"${remote_python}" - <<PY >"${remote_artifacts}/port-preflight.log" 2>&1
import socket

ports = (${dist_port}, ${master_port})
sockets = []
try:
    for port in ports:
        sock = socket.socket()
        sock.bind(("127.0.0.1", port))
        sockets.append(sock)
finally:
    for sock in sockets:
        sock.close()
PY
preflight_status=$?

if [[ "${preflight_status}" -eq 0 ]] &&
   [[ ! -d "${remote_package_root}" ]]; then
  printf 'remote package root is missing: %s\n' \
    "${remote_package_root}" \
    >"${remote_artifacts}/package-preflight.log"
  preflight_status=2
fi

if [[ "${preflight_status}" -eq 0 ]]; then
  PYTHONPATH="${remote_package_root}:${remote_source}" \
  "${remote_python}" -m py_compile \
    tools/autoregressive_draft_performance_gate.py \
    tools/autoregressive_draft_performance_worker.py \
    tools/verify_autoregressive_draft_performance_gate.py \
    >"${remote_artifacts}/py-compile.log" 2>&1
  preflight_status=$?
fi

gate_status="${preflight_status}"
if [[ "${preflight_status}" -eq 0 ]]; then
  CUDA_VISIBLE_DEVICES="${gpu_indices}" \
  TINYVLLM_DIST_PORT="${dist_port}" \
  MASTER_PORT="${master_port}" \
  PYTHONPATH="${remote_package_root}:${remote_source}" \
  timeout --signal=TERM --kill-after=30s \
    "${hard_timeout_seconds}s" \
    "${remote_python}" \
    tools/autoregressive_draft_performance_gate.py \
      --target-model "${target_model}" \
      --draft-model "${draft_model}" \
      --repo-root "${remote_source}" \
      --out "${remote_artifacts}/result.json" \
      >"${remote_artifacts}/gate.log" 2>&1
  gate_status=$?
fi
printf '%s\n' "${gate_status}" > "${remote_artifacts}/exit-code.txt"

verify_status=1
if [[ "${gate_status}" -eq 0 ]]; then
  PYTHONPATH="${remote_package_root}:${remote_source}" \
  "${remote_python}" \
    tools/verify_autoregressive_draft_performance_gate.py \
      --artifact "${remote_artifacts}/result.json" \
      --repo-root "${remote_source}" \
      --receipt "${remote_artifacts}/verify.remote.json" \
      >"${remote_artifacts}/verify.remote.log" 2>&1
  verify_status=$?
fi
printf '%s\n' "${verify_status}" \
  > "${remote_artifacts}/verify-remote-exit-code.txt"
nvidia-smi > "${remote_artifacts}/gpu-after.txt" 2>&1
date -u +%Y-%m-%dT%H:%M:%SZ > "${remote_artifacts}/finished_at_utc.txt"

if [[ "${gate_status}" -ne 0 ]]; then
  exit "${gate_status}"
fi
exit "${verify_status}"
REMOTE_SCRIPT
remote_status=$?
set -e

rsync -a -e "${RSYNC_SSH}" \
  "${REMOTE_HOST}:${REMOTE_ARTIFACTS}/" \
  "${LOCAL_RUN}/"

printf '%s\n' "${remote_status}" > "${LOCAL_RUN}/remote-status.txt"
if [[ "${remote_status}" -ne 0 ]]; then
  printf 'remote performance campaign failed: %s\n' \
    "${remote_status}" >&2
  if [[ -f "${LOCAL_RUN}/gate.log" ]]; then
    tail -200 "${LOCAL_RUN}/gate.log" >&2
  fi
  exit "${remote_status}"
fi

PYTHONPATH="${REPO_ROOT}" \
python3 \
  "${REPO_ROOT}/tools/verify_autoregressive_draft_performance_gate.py" \
    --artifact "${LOCAL_RUN}/result.json" \
    --repo-root "${REPO_ROOT}" \
    --receipt "${LOCAL_RUN}/verify.local.json"

(
  cd "${LOCAL_RUN}"
  find . -type f ! -name manifest.sha256 -print0 |
    sort -z |
    xargs -0 shasum -a 256 > manifest.sha256
  shasum -a 256 -c manifest.sha256
)

printf 'performance bundle: %s\n' "${LOCAL_RUN}"
