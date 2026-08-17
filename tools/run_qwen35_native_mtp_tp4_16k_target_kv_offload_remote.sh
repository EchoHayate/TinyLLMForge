#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_REPO_ROOT="${EXPECTED_REPO_ROOT:-/Users/bytedance/dev/TinyLLMForge-adaptive-ngram}"
export KRB5CCNAME="${KRB5CCNAME:-FILE:/Users/bytedance/krb5cc_sitian}"
REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/sitian-workspace01/tllm/env/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model}"
TARGET_MODEL_MANIFEST_SHA256="3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0"
MTP_CHECKPOINT_MANIFEST_SHA256="9a975bdcf0383774183cae560594dd60b522b83fe9c4cd595c47c12e2403702b"
REMOTE_PARENT="${REMOTE_PARENT:-/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-16k-target-kv-offload-runs}"
LOCAL_PARENT="${REPO_ROOT}/artifacts/qwen35_native_mtp_tp4_16k_target_kv_offload"
RUN_ID="${RUN_ID:-opaque-$(python3 -c 'import secrets; print(secrets.token_hex(12))')}"
LOCAL_RUN="${LOCAL_PARENT}/${RUN_ID}"
REMOTE_RUN="${REMOTE_PARENT}/${RUN_ID}"
REMOTE_SOURCE="${REMOTE_RUN}/source"
REMOTE_ARTIFACTS="${REMOTE_RUN}/artifacts"
MIN_FREE_MEMORY_MIB="${MIN_FREE_MEMORY_MIB:-18000}"
REMOTE_COMMAND_RETRY_ATTEMPTS="${REMOTE_COMMAND_RETRY_ATTEMPTS:-3}"
REMOTE_RSYNC_RETRY_ATTEMPTS="${REMOTE_RSYNC_RETRY_ATTEMPTS:-3}"
RETRY_INTERVAL_SECONDS="${RETRY_INTERVAL_SECONDS:-3}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-15}"
MAX_POLL_ATTEMPTS="${MAX_POLL_ATTEMPTS:-480}"
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
      sleep "${RETRY_INTERVAL_SECONDS}"
    fi
  done
  return "${status}"
}

retry_remote_rsync() {
  local attempt
  local status=1
  for ((
    attempt = 1;
    attempt <= REMOTE_RSYNC_RETRY_ATTEMPTS;
    attempt++
  )); do
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

if [[ "${REPO_ROOT}" != "${EXPECTED_REPO_ROOT}" ]]; then
  printf \
    'runner must execute from the approved checkout: expected=%s actual=%s\n' \
    "${EXPECTED_REPO_ROOT}" \
    "${REPO_ROOT}" >&2
  exit 2
fi

if ! klist -t -c "${KRB5CCNAME}" >/dev/null 2>&1; then
  if ! kinit -R -c "${KRB5CCNAME}" >/dev/null 2>&1; then
    printf \
      'Kerberos credentials are missing or cannot be renewed in %s\n' \
      "${KRB5CCNAME}" >&2
    exit 2
  fi
fi
if ! klist -t -c "${KRB5CCNAME}" >/dev/null 2>&1; then
  printf \
    'Kerberos credential validation failed in %s\n' \
    "${KRB5CCNAME}" >&2
  exit 2
fi

if [[ -e "${LOCAL_RUN}" ]]; then
  printf \
    'refusing to replay existing local campaign: %s\n' \
    "${LOCAL_RUN}" >&2
  exit 2
fi
mkdir -p "${LOCAL_PARENT}"

retry_remote_command \
  "if [[ -e '${REMOTE_RUN}' ]]; then \
     printf 'refusing to replay existing remote campaign: %s\n' \
       '${REMOTE_RUN}' >&2; \
     exit 2; \
   fi; \
   mkdir -p '${REMOTE_SOURCE}' '${REMOTE_ARTIFACTS}'"

mkdir -p "${LOCAL_RUN}"

read -r dist_port_base master_port_base < <(
  retry_remote_command "${REMOTE_PYTHON} - <<'PY'
import random
import socket


def available(dist_base, master_base):
    sockets = []
    try:
        ports = tuple(range(dist_base, dist_base + 4))
        ports += tuple(range(master_base, master_base + 4))
        for port in ports:
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
maximum_candidate = ephemeral_start - 204
if maximum_candidate < minimum_candidate:
    raise SystemExit(
        'no non-ephemeral TP4 port range is available'
    )
for _ in range(1000):
    dist_base = random.randint(
        minimum_candidate,
        maximum_candidate,
    )
    master_base = dist_base + 100
    if available(dist_base, master_base):
        print(dist_base, master_base)
        raise SystemExit(0)
raise SystemExit(
    'no fresh contiguous TP4 port ranges available'
)
PY"
)
if [[ ! "${dist_port_base}" =~ ^[1-9][0-9]*$ ]] ||
   [[ ! "${master_port_base}" =~ ^[1-9][0-9]*$ ]]; then
  printf \
    'remote port selection returned invalid bases\n' >&2
  exit 2
fi
printf '%s %s\n' \
  "${dist_port_base}" \
  "${master_port_base}" \
  > "${LOCAL_RUN}/selected_ports.txt"

source_tar="${LOCAL_RUN}/source.tar"
(
  cd "${REPO_ROOT}"
  COPYFILE_DISABLE=1 tar --no-xattrs -cf "${source_tar}" \
    tinyvllm \
    tools/qwen35_generic_speculative_tp4_gate.py \
    tools/qwen35_native_mtp_tp1_4k_engine_gate.py \
    tools/qwen35_native_mtp_tp1_4k_engine_worker.py \
    tools/qwen35_native_mtp_tp4_4k_engine_gate.py \
    tools/qwen35_native_mtp_tp4_4k_engine_worker.py \
    tools/qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
    tools/qwen35_native_mtp_tp4_16k_target_kv_offload_worker.py \
    tools/verify_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py
)
retry_remote_rsync \
  "${source_tar}" \
  "${REMOTE_HOST}:${REMOTE_RUN}/source.tar"
retry_remote_command \
  "tar -xf '${REMOTE_RUN}/source.tar' -C '${REMOTE_SOURCE}'"

retry_remote_command \
  "cd '${REMOTE_SOURCE}' && '${REMOTE_PYTHON}' - <<'PY'
import importlib.util
from pathlib import Path


worker_path = Path(
    'tools/qwen35_native_mtp_tp4_16k_'
    'target_kv_offload_worker.py'
)
spec = importlib.util.spec_from_file_location(
    'qwen35_native_mtp_tp4_16k_preflight',
    worker_path,
)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
model_path = '${MODEL_PATH}'
target_actual = worker.target_model_manifest_sha256(
    model_path
)
target_expected = '${TARGET_MODEL_MANIFEST_SHA256}'
if target_actual != target_expected:
    raise SystemExit(
        f'target model manifest mismatch: {target_actual}'
    )
mtp_actual = worker.mtp_checkpoint_manifest_sha256(
    model_path
)
mtp_expected = '${MTP_CHECKPOINT_MANIFEST_SHA256}'
if mtp_actual != mtp_expected:
    raise SystemExit(
        f'MTP checkpoint manifest mismatch: {mtp_actual}'
    )
print(target_actual)
print(mtp_actual)
PY"

gpu_inventory="$(
  retry_remote_command \
    "nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits"
)"
printf '%s\n' "${gpu_inventory}" \
  > "${LOCAL_RUN}/gpu_inventory.csv"
gpu_csv="$(
  printf '%s\n' "${gpu_inventory}" |
    awk -F',' -v minimum="${MIN_FREE_MEMORY_MIB}" '
      {
        for (field = 1; field <= NF; field++) {
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", $field)
        }
        if (($2 + 0) >= minimum && ($4 + 0) == 0) {
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
    'four-GPU preflight failed: need four distinct idle GPUs with memory.free >= %s MiB\n' \
    "${MIN_FREE_MEMORY_MIB}" >&2
  exit 2
fi
printf '%s\n' "${gpu_csv}" \
  > "${LOCAL_RUN}/selected_gpu_indices.txt"

cat > "${LOCAL_RUN}/campaign.sh" <<EOF
#!/usr/bin/env bash
set +e
cd '${REMOTE_SOURCE}'
export CUDA_VISIBLE_DEVICES='${gpu_csv}'
export PYTHONPATH='${REMOTE_SOURCE}'
'${REMOTE_PYTHON}' \
  tools/qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
  --model '${MODEL_PATH}' \
  --gpu-indices '${gpu_csv}' \
  --dist-port-base '${dist_port_base}' \
  --master-port-base '${master_port_base}' \
  --output-dir '${REMOTE_ARTIFACTS}/authority' \
  > '${REMOTE_RUN}/campaign.log' 2>&1
campaign_status=\$?
if (( campaign_status == 0 )); then
  '${REMOTE_PYTHON}' \
    tools/verify_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
    '${REMOTE_ARTIFACTS}/authority' \
    --source-root '${REMOTE_SOURCE}' \
    > '${REMOTE_RUN}/verify.remote.json' \
    2>> '${REMOTE_RUN}/campaign.log'
  campaign_status=\$?
fi
printf '%s\n' "\${campaign_status}" \
  > '${REMOTE_RUN}/campaign.exit_code.tmp'
mv \
  '${REMOTE_RUN}/campaign.exit_code.tmp' \
  '${REMOTE_RUN}/campaign.exit_code'
if (( campaign_status == 0 )); then
  printf '%s\n' COMPLETE \
    > '${REMOTE_RUN}/campaign.status.tmp'
else
  printf '%s\n' FAILED \
    > '${REMOTE_RUN}/campaign.status.tmp'
fi
mv \
  '${REMOTE_RUN}/campaign.status.tmp' \
  '${REMOTE_RUN}/campaign.status'
exit "\${campaign_status}"
EOF
chmod 700 "${LOCAL_RUN}/campaign.sh"
retry_remote_rsync \
  "${LOCAL_RUN}/campaign.sh" \
  "${REMOTE_HOST}:${REMOTE_RUN}/campaign.sh"

retry_remote_command \
  "status=\$(cat '${REMOTE_RUN}/campaign.status' 2>/dev/null || true); \
   pid=\$(cat '${REMOTE_RUN}/campaign.pid' 2>/dev/null || true); \
   if [[ \"\${status}\" == RUNNING ]]; then \
     printf 'campaign already running: %s\n' \"\${pid}\" >&2; \
     exit 2; \
   elif [[ \"\${status}\" == COMPLETE || \"\${status}\" == FAILED ]]; then \
     printf 'campaign already terminal: %s\n' \"\${status}\" >&2; \
     exit 2; \
   else \
     printf '%s\n' RUNNING > '${REMOTE_RUN}/campaign.status'; \
     nohup bash '${REMOTE_RUN}/campaign.sh' \
       </dev/null >/dev/null 2>&1 & \
     pid=\$!; \
     printf '%s\n' \"\${pid}\" > '${REMOTE_RUN}/campaign.pid'; \
     printf 'campaign started: %s\n' \"\${pid}\"; \
   fi"

campaign_status=124
for ((
  poll_attempt = 1;
  poll_attempt <= MAX_POLL_ATTEMPTS;
  poll_attempt++
)); do
  poll="$(
    retry_remote_command \
      "status=\$(cat '${REMOTE_RUN}/campaign.status' 2>/dev/null || true); \
       exit_code=\$(cat '${REMOTE_RUN}/campaign.exit_code' 2>/dev/null || true); \
       pid=\$(cat '${REMOTE_RUN}/campaign.pid' 2>/dev/null || true); \
       if [[ \"\${status}\" == RUNNING ]] && \
          [[ -n \"\${pid}\" ]] && \
          ! kill -0 \"\${pid}\" 2>/dev/null; then \
         status=FAILED; \
         exit_code=1; \
       fi; \
       printf '%s:%s\n' \
         \"\${status:-UNKNOWN}\" \
         \"\${exit_code}\""
  )"
  status="${poll%%:*}"
  exit_code="${poll#*:}"
  if [[ "${status}" == COMPLETE ]]; then
    campaign_status="${exit_code:-0}"
    break
  fi
  if [[ "${status}" == FAILED ]]; then
    campaign_status="${exit_code:-1}"
    break
  fi
  sleep "${POLL_INTERVAL_SECONDS}"
done

copy_status=0
retry_remote_rsync \
  "${REMOTE_HOST}:${REMOTE_RUN}/" \
  "${LOCAL_RUN}/" || copy_status=$?
if (( copy_status != 0 )); then
  exit "${copy_status}"
fi

if (( campaign_status != 0 )); then
  if [[ -f "${LOCAL_RUN}/campaign.log" ]]; then
    cat "${LOCAL_RUN}/campaign.log" >&2
  fi
  if [[ -d "${LOCAL_RUN}/artifacts/authority.failed" ]]; then
    printf \
      'failed authority retained at %s\n' \
      "${LOCAL_RUN}/artifacts/authority.failed" >&2
  fi
  if (( campaign_status == 124 )); then
    printf \
      'campaign polling exceeded %s attempts\n' \
      "${MAX_POLL_ATTEMPTS}" >&2
  fi
  exit "${campaign_status}"
fi

authority_path="${LOCAL_RUN}/artifacts/authority"
if [[ ! -d "${authority_path}" ]]; then
  printf \
    'remote campaign completed without authority: %s\n' \
    "${authority_path}" >&2
  exit 1
fi
python3 \
  "${REPO_ROOT}/tools/verify_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py" \
  "${authority_path}" \
  --source-root "${LOCAL_RUN}/source" \
  > "${LOCAL_RUN}/verify.local.json"

last_path_tmp="${LOCAL_PARENT}/.last_completed_run_path.txt.tmp"
printf '%s\n' \
  "${authority_path}" \
  > "${last_path_tmp}"
mv \
  "${last_path_tmp}" \
  "${LOCAL_PARENT}/last_completed_run_path.txt"
printf 'authority=%s\n' "${authority_path}"
