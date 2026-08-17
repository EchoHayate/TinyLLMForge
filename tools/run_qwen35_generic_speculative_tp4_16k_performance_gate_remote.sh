#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KRB5CCNAME="${KRB5CCNAME:-FILE:/Users/bytedance/krb5cc_sitian}"
REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/sitian-workspace01/tllm/env/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model}"
MODEL_MANIFEST_SHA256="3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0"
REMOTE_PARENT="${REMOTE_PARENT:-/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-performance-runs}"
LOCAL_PARENT="${REPO_ROOT}/artifacts/qwen35_generic_speculative_tp4_16k_performance"
RUN_ID="${RUN_ID:-opaque-$(python3 -c 'import secrets; print(secrets.token_hex(12))')}"
LOCAL_RUN="${LOCAL_PARENT}/${RUN_ID}"
REMOTE_RUN="${REMOTE_PARENT}/${RUN_ID}"
REMOTE_SOURCE="${REMOTE_RUN}/source"
REMOTE_ARTIFACTS="${REMOTE_RUN}/artifacts"
MIN_FREE_MEMORY_MIB=49152
MAX_GPU_UTILIZATION=10
MAX_POST_CELL_DRIFT_MIB=4096
POST_SETTLE_ATTEMPTS="${POST_SETTLE_ATTEMPTS:-12}"
POST_SETTLE_INTERVAL_SECONDS="${POST_SETTLE_INTERVAL_SECONDS:-5}"
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

if [[ -e "${LOCAL_RUN}" ]]; then
  printf 'refusing to replay existing local campaign: %s\n' \
    "${LOCAL_RUN}" >&2
  exit 2
fi
mkdir -p "${LOCAL_PARENT}" "${LOCAL_RUN}"

retry_remote_command \
  "if [[ -e '${REMOTE_RUN}' ]]; then \
     printf 'refusing to replay existing remote campaign: %s\n' \
       '${REMOTE_RUN}' >&2; \
     exit 2; \
   fi; \
   mkdir -p '${REMOTE_SOURCE}' '${REMOTE_ARTIFACTS}'"

gpu_inventory="$(
  retry_remote_command \
    "nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits"
)"
printf '%s\n' "${gpu_inventory}" > "${LOCAL_RUN}/gpu_inventory.csv"
gpu_csv="$(
  printf '%s\n' "${gpu_inventory}" |
    awk -F',' \
      -v minimum="${MIN_FREE_MEMORY_MIB}" \
      -v maximum_util="${MAX_GPU_UTILIZATION}" '
        {
          for (field = 1; field <= NF; field++) {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $field)
          }
          if (($2 + 0) >= minimum && ($4 + 0) <= maximum_util) {
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
    'four-GPU preflight failed: need 48 GiB free and <=10%% utilization\n' \
    >&2
  exit 2
fi
printf '%s\n' "${gpu_csv}" > "${LOCAL_RUN}/selected_gpu_indices.txt"

read -r dist_port_base master_port_base < <(
  retry_remote_command "${REMOTE_PYTHON} - <<'PY'
import random
import socket


def available(dist_base, master_base):
    sockets = []
    try:
        for port in tuple(range(dist_base, dist_base + 4)) + tuple(
            range(master_base, master_base + 4)
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
) as source:
    ephemeral_start = int(source.read().split()[0])
for _ in range(1000):
    dist_base = random.randint(2000, ephemeral_start - 204)
    master_base = dist_base + 100
    if available(dist_base, master_base):
        print(dist_base, master_base)
        raise SystemExit(0)
raise SystemExit('no fresh non-ephemeral TP4 port ranges available')
PY"
)
printf '%s %s\n' \
  "${dist_port_base}" \
  "${master_port_base}" \
  > "${LOCAL_RUN}/selected_ports.txt"

source_tar="${LOCAL_RUN}/source.tar"
(
  cd "${REPO_ROOT}"
  COPYFILE_DISABLE=1 tar --no-xattrs -cf "${source_tar}" \
    tinyvllm \
    tools/speculative_runtime_performance_gate.py \
    tools/qwen35_generic_speculative_tp4_gate.py \
    tools/qwen35_generic_speculative_tp4_worker.py \
    tools/qwen35_generic_speculative_tp4_16k_gate.py \
    tools/qwen35_generic_speculative_tp4_16k_worker.py \
    tools/qwen35_generic_speculative_tp4_16k_performance_gate.py \
    tools/qwen35_generic_speculative_tp4_16k_performance_worker.py \
    tools/verify_qwen35_generic_speculative_tp4_16k_performance_gate.py
)
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

path = Path('tools/qwen35_generic_speculative_tp4_16k_performance_gate.py')
spec = importlib.util.spec_from_file_location('performance_preflight', path)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
model_path = '${MODEL_PATH}'
actual = gate.model_manifest_sha256(model_path)
if actual != '${MODEL_MANIFEST_SHA256}':
    raise SystemExit(f'model manifest mismatch: {actual}')
config = json.loads((Path(model_path) / 'config.json').read_text())
if config.get('model_type') != 'qwen3_5':
    raise SystemExit('checkpoint is not qwen3_5')
text_config = config.get('text_config')
layer_types = (
    text_config.get('layer_types')
    if isinstance(text_config, dict)
    else None
)
if (
    not isinstance(layer_types, list)
    or 'linear_attention' not in layer_types
    or 'full_attention' not in layer_types
):
    raise SystemExit('checkpoint hybrid layer inventory is incomplete')
print(actual)
PY"

cat > "${LOCAL_RUN}/remote_campaign.py" <<'PY'
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


source_root = Path(os.environ["REMOTE_SOURCE"])
gate_path = (
    source_root
    / "tools"
    / "qwen35_generic_speculative_tp4_16k_performance_gate.py"
)
spec = importlib.util.spec_from_file_location(
    "qwen35_tp4_16k_performance_remote_gate",
    gate_path,
)
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)

gpu_indices = tuple(
    int(item)
    for item in os.environ["SELECTED_GPU_CSV"].split(",")
)
minimum_free = int(os.environ["MIN_FREE_MEMORY_MIB"])
maximum_utilization = int(os.environ["MAX_GPU_UTILIZATION"])
maximum_drift = int(os.environ["MAX_POST_CELL_DRIFT_MIB"])
settle_attempts = int(os.environ["POST_SETTLE_ATTEMPTS"])
settle_interval = int(os.environ["POST_SETTLE_INTERVAL_SECONDS"])


def inventory():
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        rows.append({
            "index": int(fields[0]),
            "memory_free_mib": int(fields[1]),
            "memory_total_mib": int(fields[2]),
            "utilization_gpu_percent": int(fields[3]),
        })
    return rows


def selected(rows):
    by_index = {row["index"]: row for row in rows}
    if set(gpu_indices) - set(by_index):
        raise RuntimeError("selected GPU inventory changed")
    return [by_index[index] for index in gpu_indices]


def require_pre_cell(rows, key):
    for row in rows:
        if row["memory_free_mib"] < minimum_free:
            raise RuntimeError(f"{key} free-memory preflight failed")
        if row["utilization_gpu_percent"] > maximum_utilization:
            raise RuntimeError(f"{key} utilization preflight failed")


environment = {
    "python_version": sys.version.split()[0],
    "torch_version": "loaded-worker-recorded",
    "device_name": "nvidia-smi-inventory",
    "gpu_inventory": {
        "selected_physical_indices": list(gpu_indices),
        "campaign_start": inventory(),
        "pre_cells": {},
        "post_cells": {},
    },
}


def worker_runner(command, *, log_path, cwd):
    policy = command[command.index("--policy") + 1]
    batch_size = command[command.index("--batch-size") + 1]
    key = f"{policy}:b{batch_size}"
    pre_rows = selected(inventory())
    require_pre_cell(pre_rows, key)
    environment["gpu_inventory"]["pre_cells"][key] = pre_rows
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(log_path).open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    post_rows = None
    stable = False
    for attempt in range(settle_attempts):
        if attempt:
            time.sleep(settle_interval)
        candidate = selected(inventory())
        stable = all(
            abs(
                after["memory_free_mib"]
                - before["memory_free_mib"]
            )
            <= maximum_drift
            and after["utilization_gpu_percent"]
            <= maximum_utilization
            for before, after in zip(pre_rows, candidate)
        )
        post_rows = candidate
        if stable:
            break
    environment["gpu_inventory"]["post_cells"][key] = post_rows
    if completed.returncode != 0:
        return int(completed.returncode)
    if not stable:
        return 91
    return 0


gate.run_campaign(
    model_path=os.environ["MODEL_PATH"],
    gpu_indices=gpu_indices,
    output_dir=Path(os.environ["REMOTE_ARTIFACTS"]) / "authority",
    dist_port_base=int(os.environ["DIST_PORT_BASE"]),
    master_port_base=int(os.environ["MASTER_PORT_BASE"]),
    repo_root=source_root,
    worker_script=(
        source_root
        / "tools"
        / "qwen35_generic_speculative_tp4_16k_performance_worker.py"
    ),
    worker_runner=worker_runner,
    python_executable=os.environ["REMOTE_PYTHON"],
    environment=environment,
)
PY

cat > "${LOCAL_RUN}/campaign.sh" <<EOF
#!/usr/bin/env bash
set +e
export REMOTE_SOURCE='${REMOTE_SOURCE}'
export REMOTE_ARTIFACTS='${REMOTE_ARTIFACTS}'
export PYTHONPATH='${REMOTE_SOURCE}'
export MODEL_PATH='${MODEL_PATH}'
export SELECTED_GPU_CSV='${gpu_csv}'
export DIST_PORT_BASE='${dist_port_base}'
export MASTER_PORT_BASE='${master_port_base}'
export REMOTE_PYTHON='${REMOTE_PYTHON}'
export MIN_FREE_MEMORY_MIB='${MIN_FREE_MEMORY_MIB}'
export MAX_GPU_UTILIZATION='${MAX_GPU_UTILIZATION}'
export MAX_POST_CELL_DRIFT_MIB='${MAX_POST_CELL_DRIFT_MIB}'
export POST_SETTLE_ATTEMPTS='${POST_SETTLE_ATTEMPTS}'
export POST_SETTLE_INTERVAL_SECONDS='${POST_SETTLE_INTERVAL_SECONDS}'
cd '${REMOTE_SOURCE}'
'${REMOTE_PYTHON}' \
  '${REMOTE_RUN}/remote_campaign.py' \
  > '${REMOTE_RUN}/campaign.log' 2>&1
campaign_status=\$?
if (( campaign_status == 0 )); then
  '${REMOTE_PYTHON}' \
    tools/verify_qwen35_generic_speculative_tp4_16k_performance_gate.py \
    --authority '${REMOTE_ARTIFACTS}/authority' \
    --source-root '${REMOTE_SOURCE}' \
    > '${REMOTE_RUN}/verify.remote.json' 2>> '${REMOTE_RUN}/campaign.log'
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
chmod 700 "${LOCAL_RUN}/campaign.sh"
retry_remote_rsync \
  "${LOCAL_RUN}/campaign.sh" \
  "${LOCAL_RUN}/remote_campaign.py" \
  "${REMOTE_HOST}:${REMOTE_RUN}/"

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
     nohup bash '${REMOTE_RUN}/campaign.sh' </dev/null >/dev/null 2>&1 & \
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
    printf 'failed authority retained at %s\n' \
      "${LOCAL_RUN}/artifacts/authority.failed" >&2
  fi
  exit "${campaign_status}"
fi

authority_path="${LOCAL_RUN}/artifacts/authority"
python3 \
  "${REPO_ROOT}/tools/verify_qwen35_generic_speculative_tp4_16k_performance_gate.py" \
  --authority "${authority_path}" \
  --source-root "${LOCAL_RUN}/source" \
  > "${LOCAL_RUN}/verify.local.json"

last_path_tmp="${LOCAL_PARENT}/.last_completed_run_path.txt.tmp"
printf '%s\n' "${authority_path}" > "${last_path_tmp}"
mv \
  "${last_path_tmp}" \
  "${LOCAL_PARENT}/last_completed_run_path.txt"
printf 'authority=%s\n' "${authority_path}"
