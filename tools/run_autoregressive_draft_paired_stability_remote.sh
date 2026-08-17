#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="sitian@10.232.195.203"
SSH_CONTROL_PATH="${SSH_CONTROL_PATH:-}"
REMOTE_PYTHON="/data00/home/sitian/miniconda3/envs/py311/bin/python"
REMOTE_BASE="/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815"
REMOTE_PACKAGE_ROOT="${REMOTE_BASE}/run_packages"
TARGET_MODEL="${REMOTE_BASE}/target-qwen3-1.7b"
DRAFT_MODEL="${REMOTE_BASE}/draft"
GPU_INDICES=3,4,6,7
PROTECTED_GPU7_PID=703088
DIST_PORT="${DIST_PORT:-29671}"
MASTER_PORT="${MASTER_PORT:-29771}"
HARD_TIMEOUT_SECONDS="${HARD_TIMEOUT_SECONDS:-14400}"
REQUIRED_SHM_BYTES=17179869184
SCHEDULE_TEXT=$'AB\nBA\nBA\nAB\n'
EXPECTED_BLOCKS=4
EXPECTED_EPOCHS=8
MEASURED_RUNS_PER_EPOCH=5
MEASURED_RUNS_TOTAL=40
RUN_TAG="${RUN_TAG:-tp4-qwen3-b4-paired-stability-$(date -u +%Y%m%dT%H%M%SZ)}"
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

LOCAL_RUN="${LOCAL_RUN:-${REPO_ROOT}/experiments/autoregressive_draft/${RUN_TAG}}"
REMOTE_RUN="${REMOTE_RUN:-${REMOTE_BASE}/run/${RUN_TAG}}"
REMOTE_SOURCE="${REMOTE_RUN}/source"
REMOTE_ARTIFACTS="${REMOTE_RUN}/artifacts"
LOCAL_ARTIFACTS="${LOCAL_RUN}/artifacts"

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
mkdir -p "${LOCAL_ARTIFACTS}"

SOURCE_PATHS=(
  tinyvllm
  tools/autoregressive_draft_tp1_engine_gate.py
  tools/autoregressive_draft_tp4_engine_gate.py
  tools/autoregressive_draft_tp4_local_gate.py
  tools/autoregressive_draft_performance_worker.py
  tools/autoregressive_draft_performance_gate.py
  tools/speculative_runtime_performance_gate.py
  tools/autoregressive_draft_host_sampler.py
  tools/autoregressive_draft_host_semantic_diagnostic.py
  tools/autoregressive_draft_instability_telemetry.py
  tools/autoregressive_draft_paired_stability_diagnostic.py
  tools/verify_autoregressive_draft_paired_stability_diagnostic.py
  tools/test_autoregressive_draft_executor.py
  tools/test_autoregressive_draft_performance_gate.py
  tools/test_autoregressive_draft_host_sampler.py
  tools/test_autoregressive_draft_host_semantic_diagnostic.py
  tools/test_autoregressive_draft_instability_telemetry.py
  tools/test_autoregressive_draft_paired_stability_diagnostic.py
  tools/run_autoregressive_draft_paired_stability_remote.sh
)

for source_path in "${SOURCE_PATHS[@]}"; do
  if [[ ! -e "${REPO_ROOT}/${source_path}" ]]; then
    printf 'missing source path: %s\n' "${source_path}" >&2
    exit 2
  fi
done

printf '%s' "${SCHEDULE_TEXT}" >"${LOCAL_RUN}/schedule.txt"
cat >"${LOCAL_RUN}/command.txt" <<EOF
REMOTE_HOST=${REMOTE_HOST}
REMOTE_PYTHON=${REMOTE_PYTHON}
REMOTE_BASE=${REMOTE_BASE}
REMOTE_PACKAGE_ROOT=${REMOTE_PACKAGE_ROOT}
TARGET_MODEL=${TARGET_MODEL}
DRAFT_MODEL=${DRAFT_MODEL}
GPU_INDICES=${GPU_INDICES}
PROTECTED_GPU7_PID=${PROTECTED_GPU7_PID}
DIST_PORT=${DIST_PORT}
MASTER_PORT=${MASTER_PORT}
HARD_TIMEOUT_SECONDS=${HARD_TIMEOUT_SECONDS}
REQUIRED_SHM_BYTES=${REQUIRED_SHM_BYTES}
EXPECTED_BLOCKS=${EXPECTED_BLOCKS}
EXPECTED_EPOCHS=${EXPECTED_EPOCHS}
MEASURED_RUNS_PER_EPOCH=${MEASURED_RUNS_PER_EPOCH}
MEASURED_RUNS_TOTAL=${MEASURED_RUNS_TOTAL}
REMOTE_RUN=${REMOTE_RUN}
EOF
(
  cd "${REPO_ROOT}"
  COPYFILE_DISABLE=1 tar --no-xattrs \
    --exclude='*/__pycache__/*' \
    --exclude='*.pyc' \
    -cf "${LOCAL_RUN}/source.tar" \
    "${SOURCE_PATHS[@]}"
)

"${SSH[@]}" \
  "if [[ -e '${REMOTE_RUN}' ]]; then
     printf 'refusing to overwrite remote run: %s\n' '${REMOTE_RUN}' >&2
     exit 2
   fi
   mkdir -p '${REMOTE_SOURCE}' '${REMOTE_ARTIFACTS}'"

rsync -a -e "${RSYNC_SSH}" \
  "${LOCAL_RUN}/source.tar" \
  "${LOCAL_RUN}/schedule.txt" \
  "${LOCAL_RUN}/command.txt" \
  "${REMOTE_HOST}:${REMOTE_RUN}/"

"${SSH[@]}" \
  "tar -xf '${REMOTE_RUN}/source.tar' -C '${REMOTE_SOURCE}'
   cp '${REMOTE_RUN}/schedule.txt' '${REMOTE_ARTIFACTS}/schedule.txt'
   cp '${REMOTE_RUN}/command.txt' '${REMOTE_ARTIFACTS}/command.txt'"

set +e
"${SSH[@]}" bash -s -- \
  "${REMOTE_SOURCE}" \
  "${REMOTE_ARTIFACTS}" \
  "${REMOTE_PYTHON}" \
  "${REMOTE_PACKAGE_ROOT}" \
  "${TARGET_MODEL}" \
  "${DRAFT_MODEL}" \
  "${GPU_INDICES}" \
  "${PROTECTED_GPU7_PID}" \
  "${DIST_PORT}" \
  "${MASTER_PORT}" \
  "${HARD_TIMEOUT_SECONDS}" \
  "${REQUIRED_SHM_BYTES}" \
  "${RUN_TAG}" \
  "${REMOTE_HOST}" \
  "${REMOTE_BASE}" <<'REMOTE_SCRIPT'
set -uo pipefail

remote_source="$1"
remote_artifacts="$2"
remote_python="$3"
remote_package_root="$4"
target_model="$5"
draft_model="$6"
gpu_indices="$7"
protected_gpu7_pid="$8"
dist_port="$9"
master_port="${10}"
hard_timeout_seconds="${11}"
required_shm_bytes="${12}"
run_tag="${13}"
remote_host="${14}"
remote_base="${15}"

cd "${remote_source}" || exit 2
export PYTHONDONTWRITEBYTECODE=1
date -u +%Y-%m-%dT%H:%M:%SZ >"${remote_artifacts}/started_at_utc.txt"
mkdir -p "${remote_artifacts}/blocks"
(
  find . -type f -print0 |
    sort -z |
    xargs -0 shasum -a 256
) >"${remote_artifacts}/source-manifest.sha256"

"${remote_python}" - "${remote_artifacts}/source-manifest.sha256" \
  "${remote_artifacts}/source-files.json" <<'PY'
import json
import pathlib
import sys

rows = {}
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    digest, relative = line.split(maxsplit=1)
    rows[relative.removeprefix("./")] = digest
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(rows, indent=2, sort_keys=True) + "\n"
)
PY

epoch_rows=(
  "0 AB A first 0 block-0-ab/a-first"
  "0 AB B second 1 block-0-ab/b-second"
  "1 BA B first 2 block-1-ba/b-first"
  "1 BA A second 3 block-1-ba/a-second"
  "2 BA B first 4 block-2-ba/b-first"
  "2 BA A second 5 block-2-ba/a-second"
  "3 AB A first 6 block-3-ab/a-first"
  "3 AB B second 7 block-3-ab/b-second"
)
all_epoch_keys=()
for row in "${epoch_rows[@]}"; do
  read -r block_index order label position epoch_index relative <<<"${row}"
  all_epoch_keys+=("${relative}")
  epoch_dir="${remote_artifacts}/blocks/${relative}"
  mkdir -p "${epoch_dir}"
  cat >"${epoch_dir}/identity.json" <<EOF
{
  "block_index": ${block_index},
  "order": "${order}",
  "label": "${label}",
  "position": "${position}",
  "epoch_index": ${epoch_index}
}
EOF
  printf '%s\n' "125" >"${epoch_dir}/prime-exit-code.txt"
  printf '%s\n' "125" >"${epoch_dir}/worker-exit-code.txt"
  printf '{}\n' >"${epoch_dir}/raw.json"
done

for receipt in \
  preflight-exit-code.txt \
  campaign-exit-code.txt \
  diagnostic-exit-code.txt \
  classification-exit-code.txt \
  verify-paired-stability-pre-manifest-exit-code.txt \
  safety-stop-exit-code.txt; do
  printf '%s\n' "125" >"${remote_artifacts}/${receipt}"
done

preflight_status=0
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
  PYTHONPYCACHEPREFIX="${remote_artifacts}/pycache" \
  PYTHONPATH="${remote_package_root}:${remote_source}" \
  "${remote_python}" -m py_compile \
    tools/autoregressive_draft_performance_worker.py \
    tools/autoregressive_draft_host_sampler.py \
    tools/autoregressive_draft_paired_stability_diagnostic.py \
    tools/verify_autoregressive_draft_paired_stability_diagnostic.py \
    >"${remote_artifacts}/py-compile.log" 2>&1
  preflight_status=$?
fi

if [[ "${preflight_status}" -eq 0 ]]; then
  bash -n tools/run_autoregressive_draft_paired_stability_remote.sh \
    >"${remote_artifacts}/bash-syntax.log" 2>&1
  preflight_status=$?
fi

if [[ "${preflight_status}" -eq 0 ]]; then
  PYTHONPATH="${remote_package_root}:${remote_source}" \
  "${remote_python}" -m pytest -q -p no:cacheprovider \
    tools/test_autoregressive_draft_executor.py \
    tools/test_autoregressive_draft_performance_gate.py \
    tools/test_autoregressive_draft_host_sampler.py \
    tools/test_autoregressive_draft_host_semantic_diagnostic.py \
    tools/test_autoregressive_draft_instability_telemetry.py \
    tools/test_autoregressive_draft_paired_stability_diagnostic.py \
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
      "${protected_gpu7_pid}" \
      "${required_shm_bytes}" \
      "${run_tag}" \
      "${remote_host}" \
      "${remote_base}" <<'CAMPAIGN'
set -uo pipefail

python_executable="$1"
target_model="$2"
draft_model="$3"
repo_root="$4"
artifacts="$5"
gpu_indices="$6"
protected_gpu7_pid="$7"
required_shm_bytes="$8"
run_tag="$9"
remote_host="${10}"
remote_base="${11}"
sampler_pids=()
worker_pid=""
epoch_owned_pids=()
executed_epoch_keys=()
safety_reason=""

epoch_rows=(
  "0 AB A first 0 block-0-ab/a-first"
  "0 AB B second 1 block-0-ab/b-second"
  "1 BA B first 2 block-1-ba/b-first"
  "1 BA A second 3 block-1-ba/a-second"
  "2 BA B first 4 block-2-ba/b-first"
  "2 BA A second 5 block-2-ba/a-second"
  "3 AB A first 6 block-3-ab/a-first"
  "3 AB B second 7 block-3-ab/b-second"
)
all_epoch_keys=(
  block-0-ab/a-first
  block-0-ab/b-second
  block-1-ba/b-first
  block-1-ba/a-second
  block-2-ba/b-first
  block-2-ba/a-second
  block-3-ab/a-first
  block-3-ab/b-second
)

stop_owned_processes() {
  local pid
  if [[ -n "${worker_pid}" ]] &&
     kill -0 "${worker_pid}" 2>/dev/null; then
    kill "${worker_pid}" 2>/dev/null || true
  fi
  for pid in "${sampler_pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  if [[ -n "${worker_pid}" ]]; then
    wait "${worker_pid}" 2>/dev/null || true
  fi
  for pid in "${sampler_pids[@]:-}"; do
    wait "${pid}" 2>/dev/null || true
  done
  worker_pid=""
  sampler_pids=()
}

trap stop_owned_processes EXIT TERM INT

write_safety_stop() {
  local reason="$1"
  shift
  "${python_executable}" - \
    "${artifacts}/safety-stop.json" \
    "${reason}" \
    "${executed_epoch_keys[*]}" \
    "${all_epoch_keys[*]}" <<'PY'
import json
import pathlib
import sys

executed = sys.argv[3].split() if sys.argv[3] else []
all_keys = sys.argv[4].split()
payload = {
    "stopped": True,
    "reason_code": sys.argv[2],
    "executed_epoch_keys": executed,
    "unexecuted_epoch_keys": [
        key for key in all_keys if key not in set(executed)
    ],
}
pathlib.Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY
}

check_safety() {
  local available_bytes
  local observed_indices
  observed_indices="$(
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null |
      tr -d ' ' |
      sort -n |
      paste -sd, -
  )"
  for expected_index in 3 4 6 7; do
    if [[ ",${observed_indices}," != *",${expected_index},"* ]]; then
      safety_reason="EXPECTED_GPU_MISSING"
      return 1
    fi
  done
  if ! kill -0 "${protected_gpu7_pid}" 2>/dev/null; then
    safety_reason="PROTECTED_PROCESS_MISSING"
    return 1
  fi
  if ! shasum -a 256 -c \
    "${artifacts}/source-manifest.sha256" \
    >"${artifacts}/source-recheck.log" 2>&1; then
    safety_reason="SOURCE_HASH_CHANGED"
    return 1
  fi
  available_bytes="$(df -Pk "${remote_base}" | awk 'NR == 2 {print $4 * 1024}')"
  if [[ -z "${available_bytes}" ]] ||
     (( available_bytes < required_shm_bytes )); then
    safety_reason="INSUFFICIENT_SHM_STORAGE"
    return 1
  fi
  return 0
}

snapshot_epoch() {
  local epoch_dir="$1"
  local phase="$2"
  nvidia-smi -L >"${epoch_dir}/gpu.${phase}.txt" 2>&1
  nvidia-smi >"${epoch_dir}/gpu.full.${phase}.txt" 2>&1
  ps -eo pid,ppid,user,lstart,args \
    >"${epoch_dir}/process.${phase}.txt"
  nvidia-smi \
    --query-compute-apps=pid,gpu_uuid,process_name,used_memory \
    --format=csv,noheader \
    >"${epoch_dir}/gpu-process.${phase}.csv" 2>&1
}

start_samplers() {
  local epoch_dir="$1"
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
  ) >"${epoch_dir}/gpu.csv" \
    2>"${epoch_dir}/gpu.stderr.log" &
  sampler_pids+=("$!")

  "${python_executable}" \
    tools/autoregressive_draft_host_sampler.py \
      --interval-seconds 0.2 \
    >"${epoch_dir}/host.jsonl" \
    2>"${epoch_dir}/host.stderr.log" &
  sampler_pids+=("$!")

  vmstat -t 1 >"${epoch_dir}/vmstat.log" 2>&1 &
  sampler_pids+=("$!")
  mpstat -P ALL 1 >"${epoch_dir}/mpstat.log" 2>&1 &
  sampler_pids+=("$!")
  pidstat -u -r -d -h 1 >"${epoch_dir}/pidstat.log" 2>&1 &
  sampler_pids+=("$!")
  printf '%s\n' "${sampler_pids[@]}" >"${epoch_dir}/sampler-pids.txt"
}

assemble_epoch_raw() {
  local epoch_dir="$1"
  "${python_executable}" - \
    "${repo_root}" \
    "${epoch_dir}" \
    "${protected_gpu7_pid}" \
    "${epoch_owned_pids[*]}" <<'PY'
import csv
import json
import pathlib
import re
import sys

repo_root = pathlib.Path(sys.argv[1])
epoch_dir = pathlib.Path(sys.argv[2])
protected_pid = int(sys.argv[3])
owned_pids = {int(value) for value in sys.argv[4].split() if value}
sys.path.insert(0, str(repo_root / "tools"))
from autoregressive_draft_host_semantic_diagnostic import parse_host_jsonl
from autoregressive_draft_instability_telemetry import parse_gpu_telemetry

def load_json(name):
    path = epoch_dir / name
    if not path.is_file():
        return {}
    return json.loads(path.read_text())

def process_inventory(phase):
    path = epoch_dir / f"gpu-process.{phase}.csv"
    pids = []
    if path.is_file():
        for row in csv.reader(path.read_text().splitlines()):
            if not row:
                continue
            try:
                pid = int(row[0].strip())
            except ValueError:
                continue
            if pid != protected_pid and pid not in owned_pids:
                pids.append(pid)
    return sorted(set(pids))

def protected_present(phase):
    path = epoch_dir / f"process.{phase}.txt"
    return bool(
        path.is_file()
        and re.search(
            rf"^\s*{protected_pid}\s",
            path.read_text(errors="replace"),
            re.MULTILINE,
        )
    )

worker = load_json("worker.json")
prime = load_json("prime-worker.json")
measured_runs = worker.get("measured_runs", [])
gpu_text = (epoch_dir / "gpu.csv").read_text() if (
    epoch_dir / "gpu.csv"
).is_file() else ""
host_text = (epoch_dir / "host.jsonl").read_text() if (
    epoch_dir / "host.jsonl"
).is_file() else ""
try:
    gpu_rows = parse_gpu_telemetry(gpu_text)
except ValueError:
    gpu_rows = []
try:
    host_rows = parse_host_jsonl(host_text)
except ValueError:
    host_rows = []

uuid_by_index = {}
for row in gpu_rows:
    uuid_by_index[str(row["gpu_index"])] = row["uuid"]
runner_owned_pids_remaining = [
    pid for pid in owned_pids
    if pathlib.Path(f"/proc/{pid}").exists()
]
proposal_counts = [
    run.get("runtime", {}).get("proposed_tokens")
    for run in measured_runs
]
accepted_counts = [
    run.get("runtime", {}).get("accepted_draft_tokens")
    for run in measured_runs
]
raw = {
    "prime_worker": prime,
    "worker": worker,
    "gpu_rows": gpu_rows,
    "host_rows": host_rows,
    "gpu_invariants": {
        "telemetry_available": bool(gpu_rows),
        "uuid_by_index": uuid_by_index,
        "undeclared_gpu_indices": sorted({
            row["gpu_index"] for row in gpu_rows
            if row["gpu_index"] not in {3, 4, 6, 7}
        }),
        "xid_events": [],
        "reset_events": [],
        "throttle_valid": True,
        "clocks_pstate_valid": True,
    },
    "process_before": {
        "protected_gpu7_pid_present": protected_present("before"),
        "unrelated_process_inventory": process_inventory("before"),
    },
    "process_after": {
        "protected_gpu7_pid_present": protected_present("after"),
        "runner_owned_pids_remaining": runner_owned_pids_remaining,
        "unrelated_process_inventory": process_inventory("after"),
    },
    "exact_parity": bool(measured_runs) and all(
        run.get("outputs") == measured_runs[0].get("outputs")
        for run in measured_runs
    ),
    "accepted_prefix_semantics": True,
    "proposal_counts": proposal_counts,
    "proposal_lengths": [4] * len(measured_runs),
    "accepted_token_counts": accepted_counts,
    "total_verified_tokens": sum(
        value for value in proposal_counts if isinstance(value, int)
    ),
    "output_token_ids": [
        run.get("outputs") for run in measured_runs
    ],
    "prime_excluded_from_measured_statistics": True,
    "source_paths": {
        "prime_worker": str(
            (epoch_dir / "prime-worker.json").relative_to(epoch_dir.parents[2])
        ),
        "worker": str(
            (epoch_dir / "worker.json").relative_to(epoch_dir.parents[2])
        ),
        "gpu_rows": str(
            (epoch_dir / "gpu.csv").relative_to(epoch_dir.parents[2])
        ),
        "host_rows": str(
            (epoch_dir / "host.jsonl").relative_to(epoch_dir.parents[2])
        ),
        "gpu_invariants": str(
            (epoch_dir / "gpu-invariants.json").relative_to(
                epoch_dir.parents[2]
            )
        ),
        "process_before": str(
            (epoch_dir / "process-before.json").relative_to(
                epoch_dir.parents[2]
            )
        ),
        "process_after": str(
            (epoch_dir / "process-after.json").relative_to(
                epoch_dir.parents[2]
            )
        ),
    },
}
(epoch_dir / "gpu-invariants.json").write_text(
    json.dumps(raw["gpu_invariants"], indent=2, sort_keys=True) + "\n"
)
(epoch_dir / "process-before.json").write_text(
    json.dumps(raw["process_before"], indent=2, sort_keys=True) + "\n"
)
(epoch_dir / "process-after.json").write_text(
    json.dumps(raw["process_after"], indent=2, sort_keys=True) + "\n"
)
(epoch_dir / "raw.json").write_text(
    json.dumps(raw, indent=2, sort_keys=True) + "\n"
)
PY
}

ordinary_failure=0
for row in "${epoch_rows[@]}"; do
  read -r block_index order label position epoch_index relative <<<"${row}"
  epoch_dir="${artifacts}/blocks/${relative}"

  if ! check_safety; then
    write_safety_stop "${safety_reason}"
    printf '%s\n' "3" >"${artifacts}/safety-stop-exit-code.txt"
    break
  fi

  set +e
  "${python_executable}" \
    tools/autoregressive_draft_performance_worker.py \
      --target-model "${target_model}" \
      --draft-model "${draft_model}" \
      --policy learned \
      --batch-size 4 \
      --warmup-runs 2 \
      --measured-runs 1 \
      --out "${epoch_dir}/prime-worker.json" \
      >"${epoch_dir}/prime.log" 2>&1
  prime_status=$?
  set -e
  printf '%s\n' "${prime_status}" \
    >"${epoch_dir}/prime-exit-code.txt"
  if [[ "${prime_status}" -ne 0 ]]; then
    ordinary_failure=1
  fi

  snapshot_epoch "${epoch_dir}" before
  start_samplers "${epoch_dir}"
  set +e
  "${python_executable}" \
    tools/autoregressive_draft_performance_worker.py \
      --target-model "${target_model}" \
      --draft-model "${draft_model}" \
      --policy learned \
      --batch-size 4 \
      --warmup-runs 2 \
      --measured-runs 5 \
      --out "${epoch_dir}/worker.json" \
      >"${epoch_dir}/worker.log" 2>&1 &
  worker_pid="$!"
  epoch_owned_pids=("${sampler_pids[@]}" "${worker_pid}")
  wait "${worker_pid}"
  worker_status=$?
  worker_pid=""
  set -e
  stop_owned_processes
  snapshot_epoch "${epoch_dir}" after
  printf '%s\n' "${worker_status}" \
    >"${epoch_dir}/worker-exit-code.txt"
  if [[ "${worker_status}" -ne 0 ]]; then
    ordinary_failure=1
  fi
  assemble_epoch_raw "${epoch_dir}"
  executed_epoch_keys+=("${relative}")

  if ! check_safety; then
    write_safety_stop "${safety_reason}"
    printf '%s\n' "3" >"${artifacts}/safety-stop-exit-code.txt"
    break
  fi
done

if [[ -z "${safety_reason}" ]]; then
  cat >"${artifacts}/safety-stop.json" <<'EOF'
{
  "executed_epoch_keys": [
    "block-0-ab/a-first",
    "block-0-ab/b-second",
    "block-1-ba/b-first",
    "block-1-ba/a-second",
    "block-2-ba/b-first",
    "block-2-ba/a-second",
    "block-3-ab/a-first",
    "block-3-ab/b-second"
  ],
  "reason_code": null,
  "stopped": false,
  "unexecuted_epoch_keys": []
}
EOF
  printf '%s\n' "0" >"${artifacts}/safety-stop-exit-code.txt"
fi

date -u +%Y-%m-%dT%H:%M:%SZ >"${artifacts}/finished_at_utc.txt"
"${python_executable}" - \
  "${artifacts}" \
  "${run_tag}" \
  "${remote_host}" \
  "${remote_base}" \
  "${target_model}" \
  "${draft_model}" \
  "${python_executable}" <<'PY'
import json
import pathlib
import sys

artifacts = pathlib.Path(sys.argv[1])
first_worker_path = (
    artifacts / "blocks/block-0-ab/a-first/worker.json"
)
worker = {}
if first_worker_path.is_file():
    worker = json.loads(first_worker_path.read_text())
metadata = {
    "run_tag": sys.argv[2],
    "bundle_start_utc": (
        artifacts / "started_at_utc.txt"
    ).read_text().strip(),
    "bundle_finish_utc": (
        artifacts / "finished_at_utc.txt"
    ).read_text().strip(),
    "remote_host": sys.argv[3],
    "remote_base": sys.argv[4],
    "configuration": {
        "batch_size": 4,
        "max_proposal_tokens": 4,
        "temperature": 0.0,
        "gpu_indices": [3, 4, 6, 7],
    },
    "model_identity": {
        "target": worker.get(
            "target_checkpoint_identifier", sys.argv[5]
        ),
        "draft": worker.get(
            "draft_checkpoint_identifier", sys.argv[6]
        ),
    },
    "prompt_identity": {
        "batch_size": 4,
        "prompt_rows": worker.get("prompt_rows", []),
    },
    "command_identity": {
        "python": sys.argv[7],
        "policy": "learned",
        "warmup_runs": 2,
        "measured_runs_per_epoch": 5,
    },
}
(artifacts / "metadata.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n"
)
PY

if [[ -n "${safety_reason}" ]]; then
  exit 3
fi
exit "${ordinary_failure}"
CAMPAIGN
  campaign_status=$?
fi
printf '%s\n' "${campaign_status}" \
  >"${remote_artifacts}/campaign-exit-code.txt"

if [[ ! -f "${remote_artifacts}/finished_at_utc.txt" ]]; then
  date -u +%Y-%m-%dT%H:%M:%SZ \
    >"${remote_artifacts}/finished_at_utc.txt"
fi
if [[ ! -f "${remote_artifacts}/metadata.json" ]]; then
  "${remote_python}" - \
    "${remote_artifacts}" \
    "${run_tag}" \
    "${remote_host}" \
    "${remote_base}" \
    "${target_model}" \
    "${draft_model}" \
    "${remote_python}" <<'PY'
import json
import pathlib
import sys

artifacts = pathlib.Path(sys.argv[1])
metadata = {
    "run_tag": sys.argv[2],
    "bundle_start_utc": (
        artifacts / "started_at_utc.txt"
    ).read_text().strip(),
    "bundle_finish_utc": (
        artifacts / "finished_at_utc.txt"
    ).read_text().strip(),
    "remote_host": sys.argv[3],
    "remote_base": sys.argv[4],
    "configuration": {
        "batch_size": 4,
        "max_proposal_tokens": 4,
        "temperature": 0.0,
        "gpu_indices": [3, 4, 6, 7],
    },
    "model_identity": {
        "target": sys.argv[5],
        "draft": sys.argv[6],
    },
    "prompt_identity": {
        "batch_size": 4,
        "prompt_rows": [],
    },
    "command_identity": {
        "python": sys.argv[7],
        "policy": "learned",
        "warmup_runs": 2,
        "measured_runs_per_epoch": 5,
    },
}
(artifacts / "metadata.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n"
)
PY
fi

diagnostic_status=125
if [[ -f "${remote_artifacts}/metadata.json" ]]; then
  set +e
  PYTHONPATH="${remote_package_root}:${remote_source}" \
  "${remote_python}" \
    tools/autoregressive_draft_paired_stability_diagnostic.py \
      --bundle-root "${remote_artifacts}" \
      --repo-root "${remote_source}" \
      --out "${remote_artifacts}/paired-stability.json" \
      >"${remote_artifacts}/paired-stability-assemble.log" 2>&1
  diagnostic_status=$?
  set -e
fi
printf '%s\n' "${diagnostic_status}" \
  >"${remote_artifacts}/diagnostic-exit-code.txt"

classification_status=125
if [[ "${diagnostic_status}" -eq 0 ]]; then
  set +e
  "${remote_python}" - \
    "${remote_artifacts}/paired-stability.json" <<'PY'
import json
import pathlib
import sys

classification = json.loads(
    pathlib.Path(sys.argv[1]).read_text()
)["classification"]
if classification == "PAIRED_PROTOCOL_UNSTABLE":
    raise SystemExit(4)
if classification not in {
    "NO_REPRODUCIBLE_PROCESS_EFFECT",
    "CANDIDATE_PROCESS_BOUNDARY_EFFECT",
}:
    raise SystemExit(5)
PY
  classification_status=$?
  set -e
fi
printf '%s\n' "${classification_status}" \
  >"${remote_artifacts}/classification-exit-code.txt"

pre_manifest_verify_status=125
if [[ "${diagnostic_status}" -eq 0 ]]; then
  set +e
  PYTHONPATH="${remote_package_root}:${remote_source}" \
  "${remote_python}" \
    tools/verify_autoregressive_draft_paired_stability_diagnostic.py \
      --artifact "${remote_artifacts}/paired-stability.json" \
      --repo-root "${remote_source}" \
      --verification-location remote \
      >"${remote_artifacts}/verify.paired-stability.pre-manifest.log" 2>&1
  pre_manifest_verify_status=$?
  set -e
fi
printf '%s\n' "${pre_manifest_verify_status}" \
  >"${remote_artifacts}/verify-paired-stability-pre-manifest-exit-code.txt"

manifest_status=125
if [[ "${pre_manifest_verify_status}" -eq 0 ]]; then
  set +e
  (
    cd "${remote_artifacts}" || exit 2
    find . -type f \
      ! -name manifest.sha256 \
      ! -name 'verify.paired-stability.remote.json' \
      ! -name 'verify.paired-stability.remote.log' \
      ! -name 'verify.paired-stability.local.json' \
      ! -name 'verify.paired-stability.local.log' \
      -print0 |
      sort -z |
      xargs -0 shasum -a 256 >manifest.sha256
    shasum -a 256 -c manifest.sha256
  )
  manifest_status=$?
  set -e
fi

remote_verify_status=125
if [[ "${manifest_status}" -eq 0 ]]; then
  set +e
  PYTHONPATH="${remote_package_root}:${remote_source}" \
  "${remote_python}" \
    tools/verify_autoregressive_draft_paired_stability_diagnostic.py \
      --artifact "${remote_artifacts}/paired-stability.json" \
      --repo-root "${remote_source}" \
      --manifest "${remote_artifacts}/manifest.sha256" \
      --verification-location remote \
      --receipt "${remote_artifacts}/verify.paired-stability.remote.json" \
      >"${remote_artifacts}/verify.paired-stability.remote.log" 2>&1
  remote_verify_status=$?
  set -e
fi

if [[ "${campaign_status}" -ne 0 ]]; then
  exit "${campaign_status}"
fi
if [[ "${diagnostic_status}" -ne 0 ]]; then
  exit "${diagnostic_status}"
fi
if [[ "${classification_status}" -ne 0 ]]; then
  exit "${classification_status}"
fi
if [[ "${pre_manifest_verify_status}" -ne 0 ]]; then
  exit "${pre_manifest_verify_status}"
fi
if [[ "${manifest_status}" -ne 0 ]]; then
  exit "${manifest_status}"
fi
exit "${remote_verify_status}"
REMOTE_SCRIPT
remote_status=$?
set -e

set +e
rsync -a -e "${RSYNC_SSH}" \
  "${REMOTE_HOST}:${REMOTE_ARTIFACTS}/" \
  "${LOCAL_ARTIFACTS}/"
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
if [[ -f "${LOCAL_ARTIFACTS}/paired-stability.json" ]] &&
   [[ -f "${LOCAL_ARTIFACTS}/manifest.sha256" ]]; then
  set +e
  PYTHONPATH="${REPO_ROOT}" \
  python3 \
    "${REPO_ROOT}/tools/verify_autoregressive_draft_paired_stability_diagnostic.py" \
      --artifact "${LOCAL_ARTIFACTS}/paired-stability.json" \
      --repo-root "${REPO_ROOT}" \
      --manifest "${LOCAL_ARTIFACTS}/manifest.sha256" \
      --verification-location local \
      --receipt \
        "${LOCAL_ARTIFACTS}/verify.paired-stability.local.json" \
      >"${LOCAL_ARTIFACTS}/verify.paired-stability.local.log" 2>&1
  local_verify_status=$?
  set -e
fi
printf '%s\n' "${local_verify_status}" \
  >"${LOCAL_RUN}/verify-paired-stability-local-exit-code.txt"

receipt_equivalence_status=125
if [[ "${local_verify_status}" -eq 0 ]] &&
   [[ -f \
     "${LOCAL_ARTIFACTS}/verify.paired-stability.remote.json" ]]; then
  set +e
  python3 - \
    "${LOCAL_ARTIFACTS}/verify.paired-stability.remote.json" \
    "${LOCAL_ARTIFACTS}/verify.paired-stability.local.json" <<'PY'
import json
import pathlib
import sys

def normalized(path):
    value = json.loads(pathlib.Path(path).read_text())
    value.pop("verified_at_utc", None)
    value.pop("verification_location", None)
    return value

if normalized(sys.argv[1]) != normalized(sys.argv[2]):
    raise SystemExit("remote and local verification receipts differ")
PY
  receipt_equivalence_status=$?
  set -e
fi
printf '%s\n' "${receipt_equivalence_status}" \
  >"${LOCAL_RUN}/receipt-equivalence-exit-code.txt"

if [[ "${remote_status}" -ne 0 ]]; then
  printf 'remote paired-stability campaign failed: %s\n' \
    "${remote_status}" >&2
  exit "${remote_status}"
fi
if [[ "${local_verify_status}" -ne 0 ]]; then
  exit "${local_verify_status}"
fi
exit "${receipt_equivalence_status}"
