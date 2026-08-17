#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KRB5CCNAME="${KRB5CCNAME:-FILE:/Users/bytedance/krb5cc_sitian}"
export KRB5CCNAME
REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/sitian-workspace01/tllm/env/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B}"
REMOTE_REPO="${REMOTE_REPO:-/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge}"
MIN_FREE_MEMORY_MIB="${MIN_FREE_MEMORY_MIB:-12000}"
RUN_ID="${RUN_ID:-tp4-opaque-$(python3 -c 'import secrets; print(secrets.token_hex(8))')}"
LOCAL_PARENT="${REPO_ROOT}/artifacts/generic_speculative_tp4"
LOCAL_OUT="${LOCAL_OUT:-${LOCAL_PARENT}/${RUN_ID}}"
REMOTE_OUT="${REMOTE_REPO}/artifacts/generic_speculative_tp4/${RUN_ID}"
REMOTE_AUTHORITY="${REMOTE_OUT}/authority"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-30}"
STALE_RECHECK_ATTEMPTS="${STALE_RECHECK_ATTEMPTS:-10}"
STALE_RECHECK_INTERVAL_SECONDS="${STALE_RECHECK_INTERVAL_SECONDS:-0.1}"
REMOTE_COMMAND_RETRY_ATTEMPTS="${REMOTE_COMMAND_RETRY_ATTEMPTS:-5}"
REMOTE_COMMAND_RETRY_INTERVAL_SECONDS="${REMOTE_COMMAND_RETRY_INTERVAL_SECONDS:-3}"
SSH_OPTIONS=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ControlMaster=no
  -o ControlPath=none
  -o GSSAPIAuthentication=yes
)
SSH=(ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}")
RSYNC_SSH=(
  ssh
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ControlMaster=no
  -o ControlPath=none
  -o GSSAPIAuthentication=yes
)
RSYNC_SSH_COMMAND="${RSYNC_SSH[*]}"

retry_remote_command() {
  local attempt
  local status=1
  for ((
    attempt = 1;
    attempt <= REMOTE_COMMAND_RETRY_ATTEMPTS;
    attempt++
  )); do
    if "${SSH[@]}" "$@"; then
      return 0
    else
      status=$?
    fi
    if (( attempt < REMOTE_COMMAND_RETRY_ATTEMPTS )); then
      printf \
        'remote command failed (attempt %s/%s); retrying\n' \
        "${attempt}" \
        "${REMOTE_COMMAND_RETRY_ATTEMPTS}" >&2
      sleep "${REMOTE_COMMAND_RETRY_INTERVAL_SECONDS}"
    fi
  done
  return "${status}"
}

retry_remote_rsync() {
  local attempt
  local status=1
  for ((
    attempt = 1;
    attempt <= REMOTE_COMMAND_RETRY_ATTEMPTS;
    attempt++
  )); do
    if rsync -a -e "${RSYNC_SSH_COMMAND}" "$@"; then
      return 0
    else
      status=$?
    fi
    if (( attempt < REMOTE_COMMAND_RETRY_ATTEMPTS )); then
      printf \
        'remote rsync failed (attempt %s/%s); retrying\n' \
        "${attempt}" \
        "${REMOTE_COMMAND_RETRY_ATTEMPTS}" >&2
      sleep "${REMOTE_COMMAND_RETRY_INTERVAL_SECONDS}"
    fi
  done
  return "${status}"
}

mkdir -p "${LOCAL_OUT}"

gpu_inventory="$(retry_remote_command \
  "nvidia-smi \
    --query-gpu=index,memory.free,memory.total,utilization.gpu \
    --format=csv,noheader,nounits")"
printf '%s\n' "${gpu_inventory}" \
  > "${LOCAL_OUT}/gpu_inventory.csv"

gpu_csv="$(
  printf '%s\n' "${gpu_inventory}" |
    awk -F',' -v minimum="${MIN_FREE_MEMORY_MIB}" '
      {
        for (field = 1; field <= NF; field++) {
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", $field)
        }
        if (($2 + 0) >= minimum) {
          printf "%d %d\n", $2, $1
        }
      }
    ' |
    sort -nr |
    head -n 4 |
    awk '{print $2}' |
    paste -sd, -
)"
if [[ -z "${gpu_csv}" ]] ||
   [[ "$(awk -F',' '{print NF}' <<<"${gpu_csv}")" -ne 4 ]] ||
   [[ "$(tr ',' '\n' <<<"${gpu_csv}" | sort -u | wc -l | tr -d ' ')" -ne 4 ]]; then
  printf \
    'four-GPU preflight failed: need four distinct GPUs with memory.free >= %s MiB\n' \
    "${MIN_FREE_MEMORY_MIB}" >&2
  exit 2
fi
printf '%s\n' "${gpu_csv}" \
  > "${LOCAL_OUT}/selected_gpu_indices.txt"

read -r dist_port_base master_port_base < <(
  retry_remote_command "${REMOTE_PYTHON} - <<'PY'
import random
import socket


def available(base):
    sockets = []
    try:
        for port in tuple(range(base, base + 4)) + tuple(
            range(base + 100, base + 104)
        ):
            sock = socket.socket()
            sock.bind(('127.0.0.1', port))
            sockets.append(sock)
        return True
    except OSError:
        return False
    finally:
        for sock in sockets:
            sock.close()


with open(
    '/proc/sys/net/ipv4/ip_local_port_range',
    encoding='utf-8',
) as port_range_file:
    ephemeral_start = int(
        port_range_file.read().split()[0]
    )
minimum_candidate = 2000
maximum_candidate = ephemeral_start - 104
if maximum_candidate < minimum_candidate:
    raise SystemExit(
        'no non-ephemeral TP4 port range is available'
    )
for _ in range(1000):
    candidate = random.randint(
        minimum_candidate,
        maximum_candidate,
    )
    if available(candidate):
        print(candidate, candidate + 100)
        raise SystemExit(0)
raise SystemExit('no fresh contiguous TP4 port ranges available')
PY"
)
printf '%s %s\n' \
  "${dist_port_base}" \
  "${master_port_base}" \
  > "${LOCAL_OUT}/selected_ports.txt"

retry_remote_command \
  "mkdir -p '${REMOTE_OUT}' && \
   printf '%s\n' '${gpu_csv}' \
     > '${REMOTE_OUT}/selected_gpu_indices.txt' && \
   printf '%s %s\n' \
     '${dist_port_base}' \
     '${master_port_base}' \
     > '${REMOTE_OUT}/selected_ports.txt' && \
   nvidia-smi \
     --query-gpu=index,memory.free,memory.total,utilization.gpu \
     --format=csv,noheader,nounits \
     > '${REMOTE_OUT}/gpu_inventory.csv' && \
   nvidia-smi \
     --query-compute-apps=gpu_uuid,pid,used_memory \
     --format=csv,noheader,nounits \
     > '${REMOTE_OUT}/compute_apps.csv' 2>/dev/null || true"

retry_remote_rsync \
  "${REPO_ROOT}/tinyvllm/" \
  "${REMOTE_HOST}:${REMOTE_REPO}/tinyvllm/"

(cd "${REPO_ROOT}" && retry_remote_rsync --relative \
  ./tools/generic_speculative_tp4_gate.py \
  ./tools/generic_speculative_tp4_worker.py \
  ./tools/verify_generic_speculative_tp4_gate.py \
  ./tools/test_generic_speculative_tp4_gate.py \
  "${REMOTE_HOST}:${REMOTE_REPO}/")

retry_remote_command \
  "cd '${REMOTE_REPO}' && \
   '${REMOTE_PYTHON}' -m py_compile \
     tinyvllm/engine/decode_internal_profiler.py \
     tinyvllm/engine/llm_engine.py \
     tinyvllm/engine/model_runner.py \
     tinyvllm/engine/speculative_execution.py \
     tinyvllm/engine/speculative_model_runner.py \
     tinyvllm/engine/speculative_residency.py \
     tinyvllm/engine/speculative_runtime.py \
     tools/generic_speculative_tp4_gate.py \
     tools/generic_speculative_tp4_worker.py \
     tools/verify_generic_speculative_tp4_gate.py"

cat > "${LOCAL_OUT}/campaign.sh" <<EOF
#!/usr/bin/env bash
set +e
cd '${REMOTE_REPO}'
export CUDA_VISIBLE_DEVICES='${gpu_csv}'
export PYTHONPATH='${REMOTE_REPO}'
'${REMOTE_PYTHON}' \
  tools/generic_speculative_tp4_gate.py \
  --model '${MODEL_PATH}' \
  --gpu-indices '${gpu_csv}' \
  --dist-port-base '${dist_port_base}' \
  --master-port-base '${master_port_base}' \
  --output-dir '${REMOTE_AUTHORITY}' \
  > '${REMOTE_OUT}/remote.log' 2>&1
campaign_status=\$?
if (( campaign_status == 0 )); then
  '${REMOTE_PYTHON}' \
    tools/verify_generic_speculative_tp4_gate.py \
    '${REMOTE_AUTHORITY}' \
    --source-root '${REMOTE_REPO}' \
    --out '${REMOTE_OUT}/verify.remote.json' \
    >> '${REMOTE_OUT}/remote.log' 2>&1
  campaign_status=\$?
fi
printf '%s\n' "\${campaign_status}" \
  > '${REMOTE_OUT}/campaign.exit_code.tmp'
mv \
  '${REMOTE_OUT}/campaign.exit_code.tmp' \
  '${REMOTE_OUT}/campaign.exit_code'
if (( campaign_status == 0 )); then
  printf '%s\n' COMPLETE > '${REMOTE_OUT}/campaign.status'
else
  printf '%s\n' FAILED > '${REMOTE_OUT}/campaign.status'
fi
exit "\${campaign_status}"
EOF
chmod 700 "${LOCAL_OUT}/campaign.sh"
retry_remote_rsync \
  "${LOCAL_OUT}/campaign.sh" \
  "${REMOTE_HOST}:${REMOTE_OUT}/campaign.sh.tmp"
retry_remote_command \
  "chmod 700 '${REMOTE_OUT}/campaign.sh.tmp' && \
   mv \
     '${REMOTE_OUT}/campaign.sh.tmp' \
     '${REMOTE_OUT}/campaign.sh'"

set +e
retry_remote_command "
  status=\$(cat '${REMOTE_OUT}/campaign.status' 2>/dev/null || true)
  pid=\$(cat '${REMOTE_OUT}/campaign.pid' 2>/dev/null || true)
  if [[ \"\${status}\" == RUNNING ]] &&
     [[ \"\${pid}\" =~ ^[0-9]+\$ ]] &&
     ! kill -0 \"\${pid}\" 2>/dev/null; then
    for ((attempt = 0; attempt < ${STALE_RECHECK_ATTEMPTS}; attempt++)); do
      sleep '${STALE_RECHECK_INTERVAL_SECONDS}'
      status=\$(cat '${REMOTE_OUT}/campaign.status' 2>/dev/null || true)
      pid=\$(cat '${REMOTE_OUT}/campaign.pid' 2>/dev/null || true)
      if [[ \"\${status}\" != RUNNING ]] ||
         { [[ \"\${pid}\" =~ ^[0-9]+\$ ]] &&
           kill -0 \"\${pid}\" 2>/dev/null; }; then
        break
      fi
    done
  fi
  if [[ \"\${status}\" == RUNNING ]] &&
     [[ \"\${pid}\" =~ ^[0-9]+\$ ]] &&
     kill -0 \"\${pid}\" 2>/dev/null; then
    printf 'campaign already running: pid=%s\n' \"\${pid}\"
  elif [[ \"\${status}\" == RUNNING ]]; then
    printf '%s\n' 125 > '${REMOTE_OUT}/campaign.exit_code'
    printf '%s\n' FAILED > '${REMOTE_OUT}/campaign.status'
    printf 'stale RUNNING campaign marked FAILED: pid=%s\n' \"\${pid}\"
  elif [[ \"\${status}\" == COMPLETE || \"\${status}\" == FAILED ]] &&
       [[ -f '${REMOTE_OUT}/campaign.exit_code' ]]; then
    printf 'campaign already terminal: status=%s\n' \"\${status}\"
  else
    rm -f \
      '${REMOTE_OUT}/campaign.pid' \
      '${REMOTE_OUT}/campaign.exit_code'
    printf '%s\n' RUNNING > '${REMOTE_OUT}/campaign.status'
    nohup bash '${REMOTE_OUT}/campaign.sh' </dev/null >/dev/null 2>&1 &
    pid=\$!
    printf '%s\n' \"\${pid}\" > '${REMOTE_OUT}/campaign.pid'
    printf 'campaign started: pid=%s\n' \"\${pid}\"
  fi"
launch_status=$?
remote_status="${launch_status}"

while (( launch_status == 0 )); do
  poll_output=$(retry_remote_command "
    status=\$(cat '${REMOTE_OUT}/campaign.status' 2>/dev/null || true)
    pid=\$(cat '${REMOTE_OUT}/campaign.pid' 2>/dev/null || true)
    if [[ \"\${status}\" == RUNNING ]] &&
       [[ \"\${pid}\" =~ ^[0-9]+\$ ]] &&
       ! kill -0 \"\${pid}\" 2>/dev/null; then
      for ((attempt = 0; attempt < ${STALE_RECHECK_ATTEMPTS}; attempt++)); do
        sleep '${STALE_RECHECK_INTERVAL_SECONDS}'
        status=\$(cat '${REMOTE_OUT}/campaign.status' 2>/dev/null || true)
        pid=\$(cat '${REMOTE_OUT}/campaign.pid' 2>/dev/null || true)
        if [[ \"\${status}\" != RUNNING ]] ||
           { [[ \"\${pid}\" =~ ^[0-9]+\$ ]] &&
             kill -0 \"\${pid}\" 2>/dev/null; }; then
          break
        fi
      done
      if [[ \"\${status}\" == RUNNING ]] &&
         { [[ ! \"\${pid}\" =~ ^[0-9]+\$ ]] ||
           ! kill -0 \"\${pid}\" 2>/dev/null; }; then
        printf '%s\n' 125 > '${REMOTE_OUT}/campaign.exit_code'
        printf '%s\n' FAILED > '${REMOTE_OUT}/campaign.status'
        status=FAILED
      fi
    fi
    exit_code=\$(cat '${REMOTE_OUT}/campaign.exit_code' 2>/dev/null || true)
    printf '%s:%s\n' \"\${status:-UNKNOWN}\" \"\${exit_code}\"")
  poll_status=$?
  if (( poll_status != 0 )); then
    printf '%s\n' "${poll_output}" >&2
    printf \
      'remote poll unavailable; campaign %s may still run. Resume with RUN_ID=%s\n' \
      "${RUN_ID}" \
      "${RUN_ID}" >&2
    remote_status="${poll_status}"
    break
  fi
  campaign_status="${poll_output%%:*}"
  campaign_exit_code="${poll_output#*:}"
  printf 'campaign %s: %s\n' \
    "${RUN_ID}" \
    "${campaign_status}"
  if [[ "${campaign_status}" == COMPLETE ]]; then
    remote_status="${campaign_exit_code:-0}"
    break
  fi
  if [[ "${campaign_status}" == FAILED ]]; then
    remote_status="${campaign_exit_code:-1}"
    break
  fi
  sleep "${POLL_INTERVAL_SECONDS}"
done
set -e

download_status=0
retry_remote_rsync \
  "${REMOTE_HOST}:${REMOTE_OUT}/" \
  "${LOCAL_OUT}/" || download_status=$?

if [[ -f "${LOCAL_OUT}/remote.log" ]]; then
  cat "${LOCAL_OUT}/remote.log"
fi

local_verify_status=0
if [[ -d "${LOCAL_OUT}/authority" ]]; then
  python3 \
    "${REPO_ROOT}/tools/verify_generic_speculative_tp4_gate.py" \
    "${LOCAL_OUT}/authority" \
    --source-root "${REPO_ROOT}" \
    --out "${LOCAL_OUT}/authority/verify.local.json" \
    || local_verify_status=$?
else
  local_verify_status=1
fi

if (( download_status == 0 &&
      remote_status == 0 &&
      local_verify_status == 0 )); then
  mkdir -p "${LOCAL_PARENT}"
  printf '%s\n' "${LOCAL_OUT}/authority" \
    > "${LOCAL_PARENT}/last_completed_run_path.txt.tmp"
  mv \
    "${LOCAL_PARENT}/last_completed_run_path.txt.tmp" \
    "${LOCAL_PARENT}/last_completed_run_path.txt"
fi

if (( download_status != 0 )); then
  exit "${download_status}"
fi
if (( remote_status != 0 )); then
  exit "${remote_status}"
fi
exit "${local_verify_status}"
