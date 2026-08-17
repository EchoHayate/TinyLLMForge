#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KRB5CCNAME="${KRB5CCNAME:-FILE:/Users/bytedance/krb5cc_sitian}"
REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/sitian-workspace01/tllm/env/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model}"
MODEL_MANIFEST_SHA256="3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0"
REMOTE_PARENT="${REMOTE_PARENT:-/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp1-runs}"
LOCAL_PARENT="${REPO_ROOT}/artifacts/qwen35_generic_speculative_tp1"
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

mkdir -p "${LOCAL_RUN}"

gpu_inventory="$(
  retry_remote_command \
    "nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits"
)"
printf '%s\n' "${gpu_inventory}" > "${LOCAL_RUN}/gpu_inventory.csv"
selected_gpu="$(
  printf '%s\n' "${gpu_inventory}" |
    awk -F',' -v minimum="${MIN_FREE_MEMORY_MIB}" '
      {
        for (field = 1; field <= NF; field++) {
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", $field)
        }
        if (($2 + 0) >= minimum) {
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
printf '%s\n' "${selected_gpu}" > "${LOCAL_RUN}/selected_gpu.txt"

# DEFAULT_SOURCE_FILES is the independently hashed authority inventory.
source_tar="${LOCAL_RUN}/source.tar"
(
  cd "${REPO_ROOT}"
  COPYFILE_DISABLE=1 tar --no-xattrs -cf "${source_tar}" \
    tinyvllm \
    tools/qwen35_generic_speculative_tp1_gate.py \
    tools/qwen35_generic_speculative_tp1_worker.py \
    tools/verify_qwen35_generic_speculative_tp1_gate.py
)

retry_remote_command \
  "rm -rf '${REMOTE_RUN}' && mkdir -p '${REMOTE_SOURCE}' '${REMOTE_ARTIFACTS}'"
retry_remote_rsync \
  "${source_tar}" \
  "${REMOTE_HOST}:${REMOTE_RUN}/source.tar"
retry_remote_command \
  "tar -xf '${REMOTE_RUN}/source.tar' -C '${REMOTE_SOURCE}'"

retry_remote_command \
  "cd '${REMOTE_SOURCE}' && '${REMOTE_PYTHON}' - <<'PY'
import importlib.util
import json
from pathlib import Path

gate_path = Path('tools/qwen35_generic_speculative_tp1_gate.py')
spec = importlib.util.spec_from_file_location('qwen35_gate_preflight', gate_path)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
model_path = '${MODEL_PATH}'
actual = gate.model_manifest_sha256(model_path)
expected = '${MODEL_MANIFEST_SHA256}'
if actual != expected:
    raise SystemExit(f'model manifest mismatch: {actual}')
config = json.loads((Path(model_path) / 'config.json').read_text())
if config.get('model_type') != 'qwen3_5':
    raise SystemExit('checkpoint is not qwen3_5')
text_config = config.get('text_config')
if not isinstance(text_config, dict):
    raise SystemExit('checkpoint text_config is missing')
layer_types = text_config.get('layer_types')
if not isinstance(layer_types, list):
    raise SystemExit('checkpoint layer_types are missing')
if 'linear_attention' not in layer_types or 'full_attention' not in layer_types:
    raise SystemExit('checkpoint hybrid layer inventory is incomplete')
print(actual)
PY"

cat > "${LOCAL_RUN}/campaign.sh" <<EOF
#!/usr/bin/env bash
set +e
cd '${REMOTE_SOURCE}'
export CUDA_VISIBLE_DEVICES='${selected_gpu}'
export PYTHONPATH='${REMOTE_SOURCE}'
'${REMOTE_PYTHON}' tools/qwen35_generic_speculative_tp1_gate.py \
  --model '${MODEL_PATH}' \
  --gpu-index '${selected_gpu}' \
  --output-dir '${REMOTE_ARTIFACTS}/authority' \
  > '${REMOTE_RUN}/campaign.log' 2>&1
campaign_status=\$?
if (( campaign_status == 0 )); then
  '${REMOTE_PYTHON}' \
    tools/verify_qwen35_generic_speculative_tp1_gate.py \
    '${REMOTE_ARTIFACTS}/authority' \
    --source-root '${REMOTE_SOURCE}' \
    > '${REMOTE_RUN}/verify.remote.txt' 2>&1
  campaign_status=\$?
fi
printf '%s\n' "\${campaign_status}" \
  > '${REMOTE_RUN}/campaign.exit_code.tmp'
mv \
  '${REMOTE_RUN}/campaign.exit_code.tmp' \
  '${REMOTE_RUN}/campaign.exit_code'
if (( campaign_status == 0 )); then
  printf '%s\n' COMPLETE > '${REMOTE_RUN}/campaign.status.tmp'
else
  printf '%s\n' FAILED > '${REMOTE_RUN}/campaign.status.tmp'
fi
mv \
  '${REMOTE_RUN}/campaign.status.tmp' \
  '${REMOTE_RUN}/campaign.status'
exit "\${campaign_status}"
EOF
retry_remote_rsync \
  "${LOCAL_RUN}/campaign.sh" \
  "${REMOTE_HOST}:${REMOTE_RUN}/campaign.sh"
retry_remote_command \
  "status=\$(cat '${REMOTE_RUN}/campaign.status' 2>/dev/null || true); \
   pid=\$(cat '${REMOTE_RUN}/campaign.pid' 2>/dev/null || true); \
   if [[ \"\${status}\" == RUNNING ]] && \
      [[ \"\${pid}\" =~ ^[0-9]+\$ ]] && \
      kill -0 \"\${pid}\" 2>/dev/null; then \
     printf 'campaign already running: %s\n' \"\${pid}\"; \
   elif [[ \"\${status}\" == COMPLETE || \"\${status}\" == FAILED ]]; then \
     printf 'campaign already terminal: %s\n' \"\${status}\"; \
   else \
     rm -f '${REMOTE_RUN}/campaign.exit_code'; \
     printf '%s\n' RUNNING > '${REMOTE_RUN}/campaign.status'; \
     nohup bash '${REMOTE_RUN}/campaign.sh' </dev/null >/dev/null 2>&1 & \
     pid=\$!; \
     printf '%s\n' \"\${pid}\" > '${REMOTE_RUN}/campaign.pid'; \
     printf 'campaign started: %s\n' \"\${pid}\"; \
   fi"

campaign_status=1
while true; do
  poll="$(
    retry_remote_command \
      "status=\$(cat '${REMOTE_RUN}/campaign.status' 2>/dev/null || true); \
       exit_code=\$(cat '${REMOTE_RUN}/campaign.exit_code' 2>/dev/null || true); \
       printf '%s:%s\n' \"\${status:-UNKNOWN}\" \"\${exit_code}\""
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

set +e
retry_remote_rsync \
  "${REMOTE_HOST}:${REMOTE_RUN}/" \
  "${LOCAL_RUN}/"
copy_status=$?
set -e
if (( copy_status != 0 )); then
  printf 'failed to copy remote artifacts for %s\n' "${RUN_ID}" >&2
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
  exit "${campaign_status}"
fi

authority_path="${LOCAL_RUN}/artifacts/authority"
if [[ ! -f "${authority_path}/source_manifest.json" ]]; then
  printf 'copied authority source_manifest is missing\n' >&2
  exit 3
fi
PYTHONPATH="${REPO_ROOT}" \
  python3 \
    "${REPO_ROOT}/tools/verify_qwen35_generic_speculative_tp1_gate.py" \
    "${authority_path}" \
    --source-root "${REPO_ROOT}" \
    > "${LOCAL_RUN}/verify.local.txt"

mkdir -p "${LOCAL_PARENT}"
last_path_tmp="${LOCAL_PARENT}/.last_completed_run_path.txt.tmp"
printf '%s\n' "${authority_path}" > "${last_path_tmp}"
mv \
  "${last_path_tmp}" \
  "${LOCAL_PARENT}/last_completed_run_path.txt"
printf 'authority=%s\n' "${authority_path}"
