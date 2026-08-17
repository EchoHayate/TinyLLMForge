#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian
REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/sitian-workspace01/tllm/env/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model}"
REMOTE_PARENT="${REMOTE_PARENT:-/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-engine-runs}"
LOCAL_PARENT="${REPO_ROOT}/artifacts/qwen35_native_mtp_tp1_4k_engine"
RUN_ID="${RUN_ID:-opaque-$(python3 -c 'import secrets; print(secrets.token_hex(12))')}"
LOCAL_RUN="${LOCAL_PARENT}/${RUN_ID}"
LOCAL_AUTHORITY="${LOCAL_RUN}/local-authority"
MIN_FREE_MEMORY_MIB="${MIN_FREE_MEMORY_MIB:-18000}"
REMOTE_COMMAND_RETRY_ATTEMPTS="${REMOTE_COMMAND_RETRY_ATTEMPTS:-3}"
REMOTE_RSYNC_RETRY_ATTEMPTS="${REMOTE_RSYNC_RETRY_ATTEMPTS:-3}"
RETRY_INTERVAL_SECONDS="${RETRY_INTERVAL_SECONDS:-3}"
CELL_ORDER=(
  "baseline 1"
  "native_mtp 1"
  "baseline 4"
  "native_mtp 4"
)
SSH_OPTIONS=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ControlMaster=no
  -o ControlPath=none
  -o GSSAPIAuthentication=yes
)
SSH=(ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}")
RSYNC_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none -o GSSAPIAuthentication=yes"

retry_remote_command() {
  local attempt
  local status=1
  for ((attempt = 1; attempt <= REMOTE_COMMAND_RETRY_ATTEMPTS; attempt++)); do
    if "${SSH[@]}" "$@"; then
      return 0
    else
      status=$?
    fi
    if (( attempt < REMOTE_COMMAND_RETRY_ATTEMPTS )); then
      sleep "${RETRY_INTERVAL_SECONDS}"
    fi
  done
  return "${status}"
}

retry_remote_rsync() {
  local attempt
  local status=1
  for ((attempt = 1; attempt <= REMOTE_RSYNC_RETRY_ATTEMPTS; attempt++)); do
    if rsync -a -e "${RSYNC_SSH_COMMAND}" "$@"; then
      return 0
    else
      status=$?
    fi
    if (( attempt < REMOTE_RSYNC_RETRY_ATTEMPTS )); then
      sleep "${RETRY_INTERVAL_SECONDS}"
    fi
  done
  return "${status}"
}

if ! klist -t -c "${KRB5CCNAME}" >/dev/null 2>&1; then
  printf \
    'Kerberos credentials are missing or expired in %s; refresh with: KRB5CCNAME=%s kinit sitian@BYTEDANCE.COM\n' \
    "${KRB5CCNAME}" \
    "${KRB5CCNAME}" >&2
  exit 2
fi

retry_remote_command "true"

mkdir -p "${LOCAL_RUN}"
printf '%s\n' "${CELL_ORDER[@]}" > "${LOCAL_RUN}/cell-order.txt"

retry_remote_command "mkdir -p '${REMOTE_PARENT}'"
REMOTE_RUN="$(
  retry_remote_command \
    "mktemp -d '${REMOTE_PARENT}/opaque-XXXXXXXXXXXX'"
)"
REMOTE_SOURCE="${REMOTE_RUN}/source"
REMOTE_AUTHORITY="${REMOTE_RUN}/authority"
printf '%s\n' "${REMOTE_RUN}" > "${LOCAL_RUN}/remote-run.txt"

gpu_inventory="$(
  retry_remote_command \
    "nvidia-smi --query-gpu=index,uuid,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits"
)"
printf '%s\n' "${gpu_inventory}" > "${LOCAL_RUN}/gpu-before.csv"
selected_gpu="$(
  printf '%s\n' "${gpu_inventory}" |
    awk -F',' -v minimum="${MIN_FREE_MEMORY_MIB}" '
      {
        for (field = 1; field <= NF; field++) {
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", $field)
        }
        if (($3 + 0) >= minimum) {
          print $1
          exit
        }
      }
    '
)"
if [[ -z "${selected_gpu}" ]]; then
  printf \
    'GPU preflight failed: memory.free must be >= %s MiB\n' \
    "${MIN_FREE_MEMORY_MIB}" >&2
  exit 2
fi
printf '%s\n' "${selected_gpu}" > "${LOCAL_RUN}/selected-gpu.txt"

retry_remote_command \
  "mkdir -p '${REMOTE_SOURCE}' && \
   nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits \
     > '${REMOTE_RUN}/gpu-processes-before.csv' || true"

source_tar="${LOCAL_RUN}/source.tar"
(
  cd "${REPO_ROOT}"
  COPYFILE_DISABLE=1 tar --no-xattrs -cf "${source_tar}" \
    tinyvllm \
    tools/qwen35_native_mtp_tp1_4k_engine_gate.py \
    tools/qwen35_native_mtp_tp1_4k_engine_worker.py \
    tools/verify_qwen35_native_mtp_tp1_4k_engine_gate.py
)
retry_remote_rsync \
  "${source_tar}" \
  "${REMOTE_HOST}:${REMOTE_RUN}/source.tar"
retry_remote_command \
  "tar -xf '${REMOTE_RUN}/source.tar' -C '${REMOTE_SOURCE}'"

set +e
"${SSH[@]}" \
  "cd '${REMOTE_SOURCE}' && \
   export PYTHONPATH='${REMOTE_SOURCE}' && \
   export CUDA_VISIBLE_DEVICES='${selected_gpu}' && \
   '${REMOTE_PYTHON}' \
     tools/qwen35_native_mtp_tp1_4k_engine_gate.py \
     --model '${MODEL_PATH}' \
     --gpu-index '${selected_gpu}' \
     --output-dir '${REMOTE_AUTHORITY}' \
     > '${REMOTE_RUN}/campaign.log' 2>&1"
campaign_status=$?
set -e

retry_remote_command \
  "nvidia-smi --query-gpu=index,uuid,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits \
     > '${REMOTE_RUN}/gpu-after.csv'; \
   nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits \
     > '${REMOTE_RUN}/gpu-processes-after.csv' || true"

set +e
retry_remote_command \
  "cmp -s '${REMOTE_RUN}/gpu-processes-before.csv' '${REMOTE_RUN}/gpu-processes-after.csv'"
gpu_process_status=$?
set -e
if (( gpu_process_status != 0 )); then
  campaign_status=3
  printf '%s\n' \
    "GPU process inventory changed; remote run retained at ${REMOTE_RUN}" \
    >&2
fi

if (( campaign_status == 0 )); then
  set +e
  "${SSH[@]}" \
    "cd '${REMOTE_SOURCE}' && \
     export PYTHONPATH='${REMOTE_SOURCE}' && \
     '${REMOTE_PYTHON}' \
       tools/verify_qwen35_native_mtp_tp1_4k_engine_gate.py \
       '${REMOTE_AUTHORITY}' \
       --source-root '${REMOTE_SOURCE}' \
       > '${REMOTE_RUN}/verify.remote.json'"
  verifier_status=$?
  set -e
  if (( verifier_status != 0 )); then
    campaign_status="${verifier_status}"
  fi
fi

retry_remote_rsync \
  "${REMOTE_HOST}:${REMOTE_RUN}/gpu-after.csv" \
  "${LOCAL_RUN}/gpu-after.csv"
retry_remote_rsync \
  "${REMOTE_HOST}:${REMOTE_RUN}/gpu-processes-before.csv" \
  "${LOCAL_RUN}/gpu-processes-before.csv"
retry_remote_rsync \
  "${REMOTE_HOST}:${REMOTE_RUN}/gpu-processes-after.csv" \
  "${LOCAL_RUN}/gpu-processes-after.csv"
retry_remote_rsync \
  "${REMOTE_HOST}:${REMOTE_RUN}/campaign.log" \
  "${LOCAL_RUN}/campaign.log"

if (( campaign_status != 0 )); then
  printf \
    'remote campaign failed with status %s; artifacts retained at %s\n' \
    "${campaign_status}" \
    "${REMOTE_RUN}" >&2
  exit "${campaign_status}"
fi

mkdir -p "${LOCAL_AUTHORITY}"
retry_remote_rsync \
  "${REMOTE_HOST}:${REMOTE_AUTHORITY}/" \
  "${LOCAL_AUTHORITY}/"
retry_remote_rsync \
  "${REMOTE_HOST}:${REMOTE_RUN}/verify.remote.json" \
  "${LOCAL_RUN}/verify.remote.json"

python3 \
  "${REPO_ROOT}/tools/verify_qwen35_native_mtp_tp1_4k_engine_gate.py" \
  "${LOCAL_AUTHORITY}" \
  --source-root "${REPO_ROOT}" \
  > "${LOCAL_RUN}/verify.local.json"

last_path_tmp="${LOCAL_PARENT}/.last_completed_run_path.txt.tmp"
printf '%s\n' "${LOCAL_AUTHORITY}" > "${last_path_tmp}"
mv \
  "${last_path_tmp}" \
  "${LOCAL_PARENT}/last_completed_run_path.txt"
printf 'authority=%s\n' "${LOCAL_AUTHORITY}"
