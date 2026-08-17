#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
SSH_CONTROL_PATH="${SSH_CONTROL_PATH:-}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/miniconda3/envs/py311/bin/python}"
REMOTE_BASE="${REMOTE_BASE:-/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815}"
REMOTE_PACKAGE_ROOT="${REMOTE_PACKAGE_ROOT:-${REMOTE_BASE}/run_packages}"
TARGET_MODEL="${TARGET_MODEL:-${REMOTE_BASE}/target-qwen3-1.7b}"
DRAFT_MODEL="${DRAFT_MODEL:-${REMOTE_BASE}/draft}"
GPU_INDICES="${GPU_INDICES:-3,4,6,7}"
DIST_PORT="${DIST_PORT:-29661}"
MASTER_PORT="${MASTER_PORT:-29761}"
HARD_TIMEOUT_SECONDS="${HARD_TIMEOUT_SECONDS:-7200}"
EPOCH_ORDER="learned_a,learned_b"
PRIME_EACH_EPOCH=1
BUNDLE_ROLE="${BUNDLE_ROLE:-discovery}"
RUN_TAG="${RUN_TAG:-tp4-qwen3-b4-learned-aa-$(date -u +%Y%m%dT%H%M%SZ)}"
LOCAL_RUN="${LOCAL_RUN:-}"
REMOTE_RUN="${REMOTE_RUN:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote-host)
      REMOTE_HOST="$2"
      shift 2
      ;;
    --ssh-control-path)
      SSH_CONTROL_PATH="$2"
      shift 2
      ;;
    --remote-python)
      REMOTE_PYTHON="$2"
      shift 2
      ;;
    --remote-base)
      REMOTE_BASE="$2"
      shift 2
      ;;
    --target-model)
      TARGET_MODEL="$2"
      shift 2
      ;;
    --draft-model)
      DRAFT_MODEL="$2"
      shift 2
      ;;
    --gpu-indices)
      GPU_INDICES="$2"
      shift 2
      ;;
    --dist-port)
      DIST_PORT="$2"
      shift 2
      ;;
    --master-port)
      MASTER_PORT="$2"
      shift 2
      ;;
    --hard-timeout-seconds)
      HARD_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --bundle-role)
      BUNDLE_ROLE="$2"
      shift 2
      ;;
    --local-run)
      LOCAL_RUN="$2"
      shift 2
      ;;
    --remote-run)
      REMOTE_RUN="$2"
      shift 2
      ;;
    --run-tag)
      RUN_TAG="$2"
      shift 2
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

if [[ "${BUNDLE_ROLE}" != "discovery" ]]; then
  printf 'bundle role must be discovery: %s\n' "${BUNDLE_ROLE}" >&2
  exit 2
fi

LOCAL_RUN="${LOCAL_RUN:-${REPO_ROOT}/experiments/autoregressive_draft/${RUN_TAG}}"
REMOTE_RUN="${REMOTE_RUN:-${REMOTE_BASE}/run/${RUN_TAG}}"
REMOTE_SOURCE="${REMOTE_RUN}/source"
REMOTE_ARTIFACTS="${REMOTE_RUN}/artifacts"

SSH_OPTIONS=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ControlMaster=no
  -o GSSAPIAuthentication=yes
)
RSYNC_SSH_OPTIONS=(
  ssh
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ControlMaster=no
  -o GSSAPIAuthentication=yes
)
if [[ -n "${SSH_CONTROL_PATH}" ]]; then
  SSH_OPTIONS+=(-o "ControlPath=${SSH_CONTROL_PATH}")
  RSYNC_SSH_OPTIONS+=(-o "ControlPath=${SSH_CONTROL_PATH}")
else
  SSH_OPTIONS+=(-o ControlPath=none)
  RSYNC_SSH_OPTIONS+=(-o ControlPath=none)
fi
SSH=(ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}")
printf -v RSYNC_SSH '%q ' "${RSYNC_SSH_OPTIONS[@]}"

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
  tools/speculative_runtime_performance_gate.py
  tools/verify_autoregressive_draft_performance_gate.py
  tools/autoregressive_draft_b4_timing_diagnostic.py
  tools/verify_autoregressive_draft_b4_timing_diagnostic.py
  tools/run_autoregressive_draft_b4_timing_diagnostic_remote.sh
  tools/run_autoregressive_draft_performance_gate_remote.sh
  tools/autoregressive_draft_performance_worker.py
  tools/autoregressive_draft_instability_telemetry.py
  tools/verify_autoregressive_draft_instability_telemetry.py
  tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh
  tools/autoregressive_draft_host_sampler.py
  tools/autoregressive_draft_host_semantic_diagnostic.py
  tools/verify_autoregressive_draft_host_semantic_diagnostic.py
  tools/autoregressive_draft_learned_aa_diagnostic.py
  tools/verify_autoregressive_draft_learned_aa_diagnostic.py
  tools/test_autoregressive_draft_executor.py
  tools/test_autoregressive_draft_performance_gate.py
  tools/test_autoregressive_draft_instability_telemetry.py
  tools/test_autoregressive_draft_host_sampler.py
  tools/test_autoregressive_draft_host_semantic_diagnostic.py
  tools/test_autoregressive_draft_learned_aa_diagnostic.py
  tools/run_autoregressive_draft_learned_aa_remote.sh
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

cat >"${LOCAL_RUN}/command.txt" <<EOF
REMOTE_HOST=${REMOTE_HOST}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH}
REMOTE_PYTHON=${REMOTE_PYTHON}
REMOTE_PACKAGE_ROOT=${REMOTE_PACKAGE_ROOT}
TARGET_MODEL=${TARGET_MODEL}
DRAFT_MODEL=${DRAFT_MODEL}
GPU_INDICES=${GPU_INDICES}
DIST_PORT=${DIST_PORT}
MASTER_PORT=${MASTER_PORT}
HARD_TIMEOUT_SECONDS=${HARD_TIMEOUT_SECONDS}
EPOCH_ORDER=${EPOCH_ORDER}
PRIME_EACH_EPOCH=${PRIME_EACH_EPOCH}
BUNDLE_ROLE=${BUNDLE_ROLE}
REMOTE_RUN=${REMOTE_RUN}
EOF

"${SSH[@]}" \
  "if [[ -e '${REMOTE_RUN}' ]]; then
     printf 'refusing to overwrite remote run: %s\n' '${REMOTE_RUN}' >&2
     exit 2
   fi
   mkdir -p \
     '${REMOTE_SOURCE}' \
     '${REMOTE_ARTIFACTS}/workers' \
     '${REMOTE_ARTIFACTS}/logs' \
     '${REMOTE_ARTIFACTS}/prime-workers' \
     '${REMOTE_ARTIFACTS}/prime-logs' \
     '${REMOTE_ARTIFACTS}/telemetry' \
     '${REMOTE_ARTIFACTS}/host-semantic' \
     '${REMOTE_ARTIFACTS}/host'"

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
  "${HARD_TIMEOUT_SECONDS}" \
  "${BUNDLE_ROLE}" <<'REMOTE_SCRIPT'
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
bundle_role="${11}"

cd "${remote_source}" || exit 2
date -u +%Y-%m-%dT%H:%M:%SZ >"${remote_artifacts}/started_at_utc.txt"
printf '%s\n' "learned_a,learned_b" \
  >"${remote_artifacts}/epoch-order.txt"
printf '%s\n' "1" >"${remote_artifacts}/prime-each-epoch.txt"
printf '%s\n' "${bundle_role}" >"${remote_artifacts}/bundle-role.txt"
nvidia-smi >"${remote_artifacts}/gpu-before.txt" 2>&1
(
  find . -type f -print0 |
    sort -z |
    xargs -0 shasum -a 256
) >"${remote_artifacts}/source-manifest.sha256"

for receipt in \
  learned-a-prime-exit-code.txt \
  learned-a-worker-exit-code.txt \
  learned-b-prime-exit-code.txt \
  learned-b-worker-exit-code.txt \
  diagnostic-exit-code.txt \
  verify-learned-aa-remote-exit-code.txt; do
  printf '%s\n' "125" >"${remote_artifacts}/${receipt}"
done

"${remote_python}" - <<PY >"${remote_artifacts}/port-preflight.log" 2>&1
import socket

sockets = []
try:
    for port in (${dist_port}, ${master_port}):
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
    tools/autoregressive_draft_performance_worker.py \
    tools/autoregressive_draft_host_sampler.py \
    tools/autoregressive_draft_learned_aa_diagnostic.py \
    tools/verify_autoregressive_draft_learned_aa_diagnostic.py \
    >"${remote_artifacts}/py-compile.log" 2>&1
  preflight_status=$?
fi

if [[ "${preflight_status}" -eq 0 ]]; then
  bash -n tools/run_autoregressive_draft_learned_aa_remote.sh \
    >"${remote_artifacts}/bash-syntax.log" 2>&1
  preflight_status=$?
fi

if [[ "${preflight_status}" -eq 0 ]]; then
  PYTHONPATH="${remote_package_root}:${remote_source}" \
  "${remote_python}" -m pytest -q \
    tools/test_autoregressive_draft_executor.py \
    tools/test_autoregressive_draft_performance_gate.py \
    tools/test_autoregressive_draft_instability_telemetry.py \
    tools/test_autoregressive_draft_host_sampler.py \
    tools/test_autoregressive_draft_host_semantic_diagnostic.py \
    tools/test_autoregressive_draft_learned_aa_diagnostic.py \
    >"${remote_artifacts}/tests.log" 2>&1
  preflight_status=$?
fi
printf '%s\n' "${preflight_status}" \
  >"${remote_artifacts}/preflight-exit-code.txt"

campaign_status="${preflight_status}"
if [[ "${preflight_status}" -eq 0 ]]; then
  CUDA_VISIBLE_DEVICES="${gpu_indices}" \
  TINYVLLM_DIST_PORT="${dist_port}" \
  MASTER_PORT="${master_port}" \
  PYTHONPATH="${remote_package_root}:${remote_source}" \
  timeout --signal=TERM --kill-after=30s \
    "${hard_timeout_seconds}s" \
    bash -s -- \
      "${remote_python}" \
      "${target_model}" \
      "${draft_model}" \
      "${remote_source}" \
      "${remote_artifacts}" \
      "${gpu_indices}" \
      "${bundle_role}" <<'CAMPAIGN'
set -euo pipefail

python_executable="$1"
target_model="$2"
draft_model="$3"
repo_root="$4"
artifacts="$5"
gpu_indices="$6"
bundle_role="$7"
sampler_pids=()

epoch_slug() {
  printf '%s\n' "${1//_/-}"
}

stop_samplers() {
  local pid
  for pid in "${sampler_pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${sampler_pids[@]:-}"; do
    wait "${pid}" 2>/dev/null || true
  done
  sampler_pids=()
}

trap stop_samplers EXIT TERM INT

start_samplers() {
  local epoch="$1"
  local slug
  slug="$(epoch_slug "${epoch}")"
  sampler_pids=()
  (
    while true; do
      sampled_at_unix_ns="$(date +%s%N)"
      while IFS= read -r row; do
        printf '%s, %s\n' "${sampled_at_unix_ns}" "${row}"
      done < <(
        nvidia-smi \
          --id="${gpu_indices}" \
          --query-gpu=timestamp,index,uuid,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.used,clocks_throttle_reasons.active \
          --format=csv,noheader,nounits
      )
      sleep 0.2
    done
  ) >"${artifacts}/telemetry/${slug}-gpu.csv" \
    2>"${artifacts}/telemetry/${slug}-gpu.stderr.log" &
  sampler_pids+=("$!")

  "${python_executable}" \
    tools/autoregressive_draft_host_sampler.py \
      --interval-seconds 0.2 \
    >"${artifacts}/host-semantic/${slug}-host.jsonl" \
    2>"${artifacts}/host-semantic/${slug}-host.stderr.log" &
  sampler_pids+=("$!")

  vmstat -t 1 >"${artifacts}/host/${slug}-vmstat.log" 2>&1 &
  sampler_pids+=("$!")
  mpstat -P ALL 1 >"${artifacts}/host/${slug}-mpstat.log" 2>&1 &
  sampler_pids+=("$!")
  pidstat -u -r -d -h 1 \
    >"${artifacts}/host/${slug}-pidstat.log" 2>&1 &
  sampler_pids+=("$!")
}

prime_epoch() {
  local epoch="$1"
  local slug
  local status
  slug="$(epoch_slug "${epoch}")"
  set +e
  "${python_executable}" \
    tools/autoregressive_draft_performance_worker.py \
      --target-model "${target_model}" \
      --draft-model "${draft_model}" \
      --policy learned \
      --batch-size 4 \
      --warmup-runs 2 \
      --measured-runs 1 \
      --out "${artifacts}/prime-workers/${slug}-prime-b4.json" \
      >"${artifacts}/prime-logs/${slug}-prime-b4.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "${status}" \
    >"${artifacts}/${slug}-prime-exit-code.txt"
  return "${status}"
}

run_epoch() {
  local epoch="$1"
  local slug
  local status
  slug="$(epoch_slug "${epoch}")"
  start_samplers "${epoch}"
  set +e
  "${python_executable}" \
    tools/autoregressive_draft_performance_worker.py \
      --target-model "${target_model}" \
      --draft-model "${draft_model}" \
      --policy learned \
      --batch-size 4 \
      --warmup-runs 2 \
      --measured-runs 8 \
      --out "${artifacts}/workers/${slug}-b4.json" \
      >"${artifacts}/logs/${slug}-b4.log" 2>&1
  status=$?
  set -e
  stop_samplers
  printf '%s\n' "${status}" \
    >"${artifacts}/${slug}-worker-exit-code.txt"
  return "${status}"
}

run_epochs() {
for epoch in learned_a learned_b; do
  prime_epoch "${epoch}"
  run_epoch "${epoch}"
done
}

set +e
run_epochs
epoch_status=$?
set -e

diagnostic_status=125
if [[ "${epoch_status}" -eq 0 ]]; then
  set +e
  "${python_executable}" \
    tools/autoregressive_draft_learned_aa_diagnostic.py \
      --learned-a-prime-worker \
        "${artifacts}/prime-workers/learned-a-prime-b4.json" \
      --learned-b-prime-worker \
        "${artifacts}/prime-workers/learned-b-prime-b4.json" \
      --learned-a-worker \
        "${artifacts}/workers/learned-a-b4.json" \
      --learned-b-worker \
        "${artifacts}/workers/learned-b-b4.json" \
      --learned-a-gpu-csv \
        "${artifacts}/telemetry/learned-a-gpu.csv" \
      --learned-b-gpu-csv \
        "${artifacts}/telemetry/learned-b-gpu.csv" \
      --learned-a-host-jsonl \
        "${artifacts}/host-semantic/learned-a-host.jsonl" \
      --learned-b-host-jsonl \
        "${artifacts}/host-semantic/learned-b-host.jsonl" \
      --epoch-order-file "${artifacts}/epoch-order.txt" \
      --prime-each-epoch-file \
        "${artifacts}/prime-each-epoch.txt" \
      --bundle-role "${bundle_role}" \
      --repo-root "${repo_root}" \
      --out "${artifacts}/learned-aa.json" \
      >"${artifacts}/learned-aa-assemble.log" 2>&1
  diagnostic_status=$?
  set -e
fi
printf '%s\n' "${diagnostic_status}" \
  >"${artifacts}/diagnostic-exit-code.txt"

if [[ "${epoch_status}" -ne 0 ]]; then
  exit "${epoch_status}"
fi
exit "${diagnostic_status}"
CAMPAIGN
  campaign_status=$?
fi

remote_verify_status=125
if [[ "${campaign_status}" -eq 0 ]]; then
  PYTHONPATH="${remote_package_root}:${remote_source}" \
  "${remote_python}" \
    tools/verify_autoregressive_draft_learned_aa_diagnostic.py \
      --artifact "${remote_artifacts}/learned-aa.json" \
      --repo-root "${remote_source}" \
      --receipt \
        "${remote_artifacts}/verify.learned-aa.remote.json" \
      >"${remote_artifacts}/verify.learned-aa.remote.log" 2>&1
  remote_verify_status=$?
fi
printf '%s\n' "${remote_verify_status}" \
  >"${remote_artifacts}/verify-learned-aa-remote-exit-code.txt"
printf '%s\n' "${campaign_status}" \
  >"${remote_artifacts}/campaign-exit-code.txt"
nvidia-smi >"${remote_artifacts}/gpu-after.txt" 2>&1
date -u +%Y-%m-%dT%H:%M:%SZ >"${remote_artifacts}/finished_at_utc.txt"

if [[ "${campaign_status}" -ne 0 ]]; then
  exit "${campaign_status}"
fi
exit "${remote_verify_status}"
REMOTE_SCRIPT
remote_status=$?
set -e

set +e
rsync -a -e "${RSYNC_SSH}" \
  "${REMOTE_HOST}:${REMOTE_ARTIFACTS}/" \
  "${LOCAL_RUN}/"
transfer_status=$?
set -e
printf '%s\n' "${remote_status}" >"${LOCAL_RUN}/remote-status.txt"
printf '%s\n' "${transfer_status}" >"${LOCAL_RUN}/transfer-status.txt"

if [[ "${transfer_status}" -ne 0 ]]; then
  printf 'remote artifact transfer failed: %s\n' \
    "${transfer_status}" >&2
  exit "${transfer_status}"
fi

local_verify_status=125
if [[ "${remote_status}" -eq 0 ]] &&
   [[ -f "${LOCAL_RUN}/learned-aa.json" ]]; then
  set +e
  PYTHONPATH="${REPO_ROOT}" \
  python3 \
    "${REPO_ROOT}/tools/verify_autoregressive_draft_learned_aa_diagnostic.py" \
      --artifact "${LOCAL_RUN}/learned-aa.json" \
      --repo-root "${REPO_ROOT}" \
      --receipt "${LOCAL_RUN}/verify.learned-aa.local.json"
  local_verify_status=$?
  set -e
fi
printf '%s\n' "${local_verify_status}" \
  >"${LOCAL_RUN}/verify-learned-aa-local-exit-code.txt"

(
  cd "${LOCAL_RUN}"
  find . -type f ! -name manifest.sha256 -print0 |
    sort -z |
    xargs -0 shasum -a 256 >manifest.sha256
  shasum -a 256 -c manifest.sha256
)

if [[ "${remote_status}" -ne 0 ]]; then
  printf 'remote learned A/A campaign failed: %s\n' \
    "${remote_status}" >&2
  exit "${remote_status}"
fi
exit "${local_verify_status}"
