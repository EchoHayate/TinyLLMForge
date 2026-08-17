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
DIST_PORT="${DIST_PORT:-29651}"
MASTER_PORT="${MASTER_PORT:-29751}"
HARD_TIMEOUT_SECONDS="${HARD_TIMEOUT_SECONDS:-5400}"
POLICY_ORDER="${POLICY_ORDER:-target,learned}"
PRIME_EACH_POLICY="${PRIME_EACH_POLICY:-0}"
RUN_TAG="${RUN_TAG:-tp4-qwen3-b4-instability-telemetry-$(date -u +%Y%m%dT%H%M%SZ)}"
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
    --policy-order)
      POLICY_ORDER="$2"
      shift 2
      ;;
    --prime-each-policy)
      PRIME_EACH_POLICY=1
      shift
      ;;
    --local-run)
      LOCAL_RUN="$2"
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

case "${POLICY_ORDER}" in
  "target,learned"|"learned,target")
    ;;
  *)
    printf 'invalid policy order: %s\n' "${POLICY_ORDER}" >&2
    exit 2
    ;;
esac

case "${PRIME_EACH_POLICY}" in
  0|1)
    ;;
  *)
    printf 'invalid prime-each-policy value: %s\n' \
      "${PRIME_EACH_POLICY}" >&2
    exit 2
    ;;
esac

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
  tools/autoregressive_draft_performance_worker.py
  tools/autoregressive_draft_b4_timing_diagnostic.py
  tools/autoregressive_draft_instability_telemetry.py
  tools/autoregressive_draft_host_sampler.py
  tools/autoregressive_draft_host_semantic_diagnostic.py
  tools/speculative_runtime_performance_gate.py
  tools/verify_autoregressive_draft_performance_gate.py
  tools/verify_autoregressive_draft_b4_timing_diagnostic.py
  tools/verify_autoregressive_draft_instability_telemetry.py
  tools/verify_autoregressive_draft_host_semantic_diagnostic.py
  tools/test_autoregressive_draft_executor.py
  tools/test_autoregressive_draft_performance_gate.py
  tools/test_autoregressive_draft_instability_telemetry.py
  tools/test_autoregressive_draft_host_sampler.py
  tools/test_autoregressive_draft_host_semantic_diagnostic.py
  tools/run_autoregressive_draft_performance_gate_remote.sh
  tools/run_autoregressive_draft_b4_timing_diagnostic_remote.sh
  tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh
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
SSH_CONTROL_PATH=${SSH_CONTROL_PATH}
REMOTE_PYTHON=${REMOTE_PYTHON}
REMOTE_PACKAGE_ROOT=${REMOTE_PACKAGE_ROOT}
TARGET_MODEL=${TARGET_MODEL}
DRAFT_MODEL=${DRAFT_MODEL}
GPU_INDICES=${GPU_INDICES}
DIST_PORT=${DIST_PORT}
MASTER_PORT=${MASTER_PORT}
HARD_TIMEOUT_SECONDS=${HARD_TIMEOUT_SECONDS}
POLICY_ORDER=${POLICY_ORDER}
PRIME_EACH_POLICY=${PRIME_EACH_POLICY}
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
  "${POLICY_ORDER}" \
  "${PRIME_EACH_POLICY}" <<'REMOTE_SCRIPT'
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
policy_order_csv="${11}"
prime_each_policy="${12}"

cd "${remote_source}" || exit 2
date -u +%Y-%m-%dT%H:%M:%SZ > "${remote_artifacts}/started_at_utc.txt"
printf '%s\n' "${policy_order_csv}" > "${remote_artifacts}/policy-order.txt"
printf '%s\n' "${prime_each_policy}" \
  > "${remote_artifacts}/prime-each-policy.txt"
nvidia-smi > "${remote_artifacts}/gpu-before.txt" 2>&1

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
    tools/autoregressive_draft_performance_gate.py \
    tools/autoregressive_draft_performance_worker.py \
    tools/autoregressive_draft_b4_timing_diagnostic.py \
    tools/autoregressive_draft_instability_telemetry.py \
    tools/autoregressive_draft_host_sampler.py \
    tools/autoregressive_draft_host_semantic_diagnostic.py \
    tools/verify_autoregressive_draft_b4_timing_diagnostic.py \
    tools/verify_autoregressive_draft_instability_telemetry.py \
    tools/verify_autoregressive_draft_host_semantic_diagnostic.py \
    >"${remote_artifacts}/py-compile.log" 2>&1
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
    >"${remote_artifacts}/tests.log" 2>&1
  preflight_status=$?
fi

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
      "${policy_order_csv}" \
      "${prime_each_policy}" <<'CAMPAIGN'
set -euo pipefail

python_executable="$1"
target_model="$2"
draft_model="$3"
repo_root="$4"
artifacts="$5"
gpu_indices="$6"
policy_order_csv="$7"
prime_each_policy="$8"
sampler_pids=()

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
  local policy="$1"
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
  ) >"${artifacts}/telemetry/${policy}-gpu.csv" \
    2>"${artifacts}/telemetry/${policy}-gpu.stderr.log" &
  sampler_pids+=("$!")

  "${python_executable}" \
    tools/autoregressive_draft_host_sampler.py \
      --interval-seconds 0.2 \
    >"${artifacts}/host-semantic/${policy}-host.jsonl" \
    2>"${artifacts}/host-semantic/${policy}-host.stderr.log" &
  sampler_pids+=("$!")

  vmstat -t 1 \
    >"${artifacts}/host/${policy}-vmstat.log" 2>&1 &
  sampler_pids+=("$!")
  mpstat -P ALL 1 \
    >"${artifacts}/host/${policy}-mpstat.log" 2>&1 &
  sampler_pids+=("$!")
  pidstat -u -r -d -h 1 \
    >"${artifacts}/host/${policy}-pidstat.log" 2>&1 &
  sampler_pids+=("$!")
}

prime_policy() {
  local policy="$1"
  "${python_executable}" \
    tools/autoregressive_draft_performance_worker.py \
      --target-model "${target_model}" \
      --draft-model "${draft_model}" \
      --policy "${policy}" \
      --batch-size 4 \
      --warmup-runs 2 \
      --measured-runs 1 \
      --out "${artifacts}/prime-workers/${policy}-prime-b4.json" \
      >"${artifacts}/prime-logs/${policy}-prime-b4.log" 2>&1
}

run_policy() {
  local policy="$1"
  start_samplers "${policy}"
  "${python_executable}" \
    tools/autoregressive_draft_performance_worker.py \
      --target-model "${target_model}" \
      --draft-model "${draft_model}" \
      --policy "${policy}" \
      --batch-size 4 \
      --warmup-runs 2 \
      --measured-runs 8 \
      --out "${artifacts}/workers/${policy}-b4.json" \
      >"${artifacts}/logs/${policy}-b4.log" 2>&1
  stop_samplers
}

IFS="," read -r -a policy_order <<< "${policy_order_csv}"
for policy in "${policy_order[@]}"; do
  if [[ "${prime_each_policy}" -eq 1 ]]; then
    prime_policy "${policy}"
  fi
  run_policy "${policy}"
done

"${python_executable}" \
  tools/autoregressive_draft_b4_timing_diagnostic.py \
    --target-worker "${artifacts}/workers/target-b4.json" \
    --learned-worker "${artifacts}/workers/learned-b4.json" \
    --repo-root "${repo_root}" \
    --out "${artifacts}/result.json" \
    >"${artifacts}/diagnostic.log" 2>&1

"${python_executable}" \
  tools/autoregressive_draft_instability_telemetry.py \
    --timing-artifact "${artifacts}/result.json" \
    --target-worker "${artifacts}/workers/target-b4.json" \
    --learned-worker "${artifacts}/workers/learned-b4.json" \
    --target-gpu-csv "${artifacts}/telemetry/target-gpu.csv" \
    --learned-gpu-csv "${artifacts}/telemetry/learned-gpu.csv" \
    --repo-root "${repo_root}" \
    --host-file target_vmstat=host/target-vmstat.log \
    --host-file target_mpstat=host/target-mpstat.log \
    --host-file target_pidstat=host/target-pidstat.log \
    --host-file learned_vmstat=host/learned-vmstat.log \
    --host-file learned_mpstat=host/learned-mpstat.log \
    --host-file learned_pidstat=host/learned-pidstat.log \
    --out "${artifacts}/telemetry.json" \
    >"${artifacts}/telemetry-assemble.log" 2>&1

if [[ "${prime_each_policy}" -ne 1 ]]; then
  printf 'host semantic diagnostic requires same-policy priming\n' >&2
  exit 2
fi

"${python_executable}" \
  tools/autoregressive_draft_host_semantic_diagnostic.py \
    --timing-artifact "${artifacts}/result.json" \
    --gpu-telemetry-artifact "${artifacts}/telemetry.json" \
    --target-worker "${artifacts}/workers/target-b4.json" \
    --learned-worker "${artifacts}/workers/learned-b4.json" \
    --target-host-jsonl "${artifacts}/host-semantic/target-host.jsonl" \
    --learned-host-jsonl "${artifacts}/host-semantic/learned-host.jsonl" \
    --policy-order "${policy_order_csv}" \
    --prime-each-policy \
    --repo-root "${repo_root}" \
    --out "${artifacts}/host-semantic.json" \
    >"${artifacts}/host-semantic-assemble.log" 2>&1
CAMPAIGN
  campaign_status=$?
fi
printf '%s\n' "${campaign_status}" > "${remote_artifacts}/exit-code.txt"

timing_verify_status=1
telemetry_verify_status=1
host_verify_status=1
if [[ "${campaign_status}" -eq 0 ]]; then
  PYTHONPATH="${remote_package_root}:${remote_source}" \
  "${remote_python}" \
    tools/verify_autoregressive_draft_b4_timing_diagnostic.py \
      --artifact "${remote_artifacts}/result.json" \
      --repo-root "${remote_source}" \
      --receipt "${remote_artifacts}/verify.timing.remote.json" \
      >"${remote_artifacts}/verify.timing.remote.log" 2>&1
  timing_verify_status=$?
fi
if [[ "${timing_verify_status}" -eq 0 ]]; then
  PYTHONPATH="${remote_package_root}:${remote_source}" \
  "${remote_python}" \
    tools/verify_autoregressive_draft_instability_telemetry.py \
      --artifact "${remote_artifacts}/telemetry.json" \
      --repo-root "${remote_source}" \
      --receipt "${remote_artifacts}/verify.remote.json" \
      >"${remote_artifacts}/verify.remote.log" 2>&1
  telemetry_verify_status=$?
fi
if [[ "${telemetry_verify_status}" -eq 0 ]]; then
  PYTHONPATH="${remote_package_root}:${remote_source}" \
  "${remote_python}" \
    tools/verify_autoregressive_draft_host_semantic_diagnostic.py \
      --artifact "${remote_artifacts}/host-semantic.json" \
      --repo-root "${remote_source}" \
      --receipt "${remote_artifacts}/verify.host.remote.json" \
      >"${remote_artifacts}/verify.host.remote.log" 2>&1
  host_verify_status=$?
fi
printf '%s\n' "${timing_verify_status}" \
  >"${remote_artifacts}/verify-timing-remote-exit-code.txt"
printf '%s\n' "${telemetry_verify_status}" \
  >"${remote_artifacts}/verify-remote-exit-code.txt"
printf '%s\n' "${host_verify_status}" \
  >"${remote_artifacts}/verify-host-remote-exit-code.txt"
nvidia-smi > "${remote_artifacts}/gpu-after.txt" 2>&1
date -u +%Y-%m-%dT%H:%M:%SZ > "${remote_artifacts}/finished_at_utc.txt"

if [[ "${campaign_status}" -ne 0 ]]; then
  exit "${campaign_status}"
fi
if [[ "${timing_verify_status}" -ne 0 ]]; then
  exit "${timing_verify_status}"
fi
if [[ "${telemetry_verify_status}" -ne 0 ]]; then
  exit "${telemetry_verify_status}"
fi
exit "${host_verify_status}"
REMOTE_SCRIPT
remote_status=$?
set -e

rsync -a -e "${RSYNC_SSH}" \
  "${REMOTE_HOST}:${REMOTE_ARTIFACTS}/" \
  "${LOCAL_RUN}/"

printf '%s\n' "${remote_status}" > "${LOCAL_RUN}/remote-status.txt"
if [[ "${remote_status}" -ne 0 ]]; then
  printf 'remote telemetry diagnostic failed: %s\n' \
    "${remote_status}" >&2
  for log_path in \
    "${LOCAL_RUN}/prime-logs/target-prime-b4.log" \
    "${LOCAL_RUN}/prime-logs/learned-prime-b4.log" \
    "${LOCAL_RUN}/logs/target-b4.log" \
    "${LOCAL_RUN}/logs/learned-b4.log" \
    "${LOCAL_RUN}/diagnostic.log" \
    "${LOCAL_RUN}/telemetry-assemble.log" \
    "${LOCAL_RUN}/host-semantic-assemble.log" \
    "${LOCAL_RUN}/verify.remote.log" \
    "${LOCAL_RUN}/verify.host.remote.log" \
    "${LOCAL_RUN}/host-semantic/target-host.stderr.log" \
    "${LOCAL_RUN}/host-semantic/learned-host.stderr.log"; do
    if [[ -f "${log_path}" ]]; then
      printf '\n==> %s <==\n' "${log_path}" >&2
      tail -200 "${log_path}" >&2
    fi
  done
  exit "${remote_status}"
fi

PYTHONPATH="${REPO_ROOT}" \
python3 \
  "${REPO_ROOT}/tools/verify_autoregressive_draft_b4_timing_diagnostic.py" \
    --artifact "${LOCAL_RUN}/result.json" \
    --repo-root "${REPO_ROOT}" \
    --receipt "${LOCAL_RUN}/verify.timing.local.json"

PYTHONPATH="${REPO_ROOT}" \
python3 \
  "${REPO_ROOT}/tools/verify_autoregressive_draft_instability_telemetry.py" \
    --artifact "${LOCAL_RUN}/telemetry.json" \
    --repo-root "${REPO_ROOT}" \
    --receipt "${LOCAL_RUN}/verify.local.json"

PYTHONPATH="${REPO_ROOT}" \
python3 \
  "${REPO_ROOT}/tools/verify_autoregressive_draft_host_semantic_diagnostic.py" \
    --artifact "${LOCAL_RUN}/host-semantic.json" \
    --repo-root "${REPO_ROOT}" \
    --receipt "${LOCAL_RUN}/verify.host.local.json"

(
  cd "${LOCAL_RUN}"
  find . -type f ! -name manifest.sha256 -print0 |
    sort -z |
    xargs -0 shasum -a 256 > manifest.sha256
  shasum -a 256 -c manifest.sha256
)

printf 'telemetry diagnostic bundle: %s\n' "${LOCAL_RUN}"
