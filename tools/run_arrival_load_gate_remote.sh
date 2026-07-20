#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "usage: $0 preflight|smoke|cost-calibration|workload-calibration|canonical|download-only|verify-only" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="sitian@10.232.195.203"
REMOTE_PYTHON="/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
MODEL_PATH="/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B"
SSH_SOCKET="${SSH_SOCKET:-/tmp/ssh-sitian-10.232.195.203}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RUN_TAG_WAS_SET="${RUN_TAG+x}"
RUN_TAG="${RUN_TAG:-qwen3-06b-arrival-${MODE}-$(date +%Y%m%d-%H%M%S)}"
LOCAL_OUT="${REPO_ROOT}/experiments/arrival_load/${RUN_TAG}"
STAGING_DIR="${LOCAL_OUT}.staging"
REMOTE_DIR="/data00/home/sitian/sitian-workspace01/tllm/arrival-load-runs/${RUN_TAG}"
POLL_INTERVAL="${POLL_INTERVAL:-20}"
RESUME="${RESUME:-0}"
DOWNLOAD_BLOCK_BYTES="${DOWNLOAD_BLOCK_BYTES:-8388608}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-8}"
DOWNLOAD_RETRY_DELAY="${DOWNLOAD_RETRY_DELAY:-3}"

SSH=(
  ssh
  -n
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ControlMaster=auto
  -o ControlPersist=600
  -S "${SSH_SOCKET}"
  "${REMOTE_HOST}"
)
SSH_STREAM=(
  ssh
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ControlMaster=auto
  -o ControlPersist=600
  -S "${SSH_SOCKET}"
  "${REMOTE_HOST}"
)

download_remote_path() {
  local remote_path="$1"
  local local_path="$2"
  local artifact_name="$3"
  local partial_path="${local_path}.partial"
  local block_path="${partial_path}.block"
  local remote_size
  local local_size
  local aligned_size
  local offset
  local block_index
  local expected_block_bytes
  local actual_block_bytes
  local attempt

  remote_size="$("${SSH[@]}" "stat -c %s '${remote_path}'")"
  mkdir -p "$(dirname "${local_path}")"
  if [[ -f "${partial_path}" ]]; then
    local_size="$(stat -f %z "${partial_path}")"
    aligned_size=$((
      (local_size / DOWNLOAD_BLOCK_BYTES)
      * DOWNLOAD_BLOCK_BYTES
    ))
    if (( aligned_size != local_size )); then
      truncate -s "${aligned_size}" "${partial_path}"
    fi
    if (( aligned_size > remote_size )); then
      rm -f "${partial_path}"
      aligned_size=0
    fi
    offset="${aligned_size}"
  else
    offset=0
  fi

  if [[ ! -f "${partial_path}" ]]; then
    : > "${partial_path}"
  fi
  while (( offset < remote_size )); do
    block_index=$((offset / DOWNLOAD_BLOCK_BYTES))
    expected_block_bytes=$((remote_size - offset))
    if (( expected_block_bytes > DOWNLOAD_BLOCK_BYTES )); then
      expected_block_bytes="${DOWNLOAD_BLOCK_BYTES}"
    fi
    for attempt in $(seq 1 "${DOWNLOAD_RETRIES}"); do
      rm -f "${block_path}"
      if "${SSH[@]}" \
        "dd if='${remote_path}' bs=${DOWNLOAD_BLOCK_BYTES} skip=${block_index} count=1 iflag=fullblock status=none" \
        >"${block_path}"
      then
        actual_block_bytes="$(stat -f %z "${block_path}")"
        if [[ "${actual_block_bytes}" == "${expected_block_bytes}" ]]; then
          cat "${block_path}" >> "${partial_path}"
          offset=$((offset + actual_block_bytes))
          break
        fi
      else
        actual_block_bytes="$(
          if [[ -f "${block_path}" ]]; then
            stat -f %z "${block_path}"
          else
            printf '0'
          fi
        )"
      fi
      if [[ "${attempt}" -lt "${DOWNLOAD_RETRIES}" ]]; then
        sleep "${DOWNLOAD_RETRY_DELAY}"
      else
        rm -f "${block_path}" "${local_path}"
        echo "artifact block download retries exhausted: ${artifact_name} block=${block_index} bytes=${actual_block_bytes}/${expected_block_bytes}" >&2
        return 1
      fi
    done
    rm -f "${block_path}"
  done

  local_size="$(stat -f %z "${partial_path}")"
  if [[ "${local_size}" == "${remote_size}" ]]; then
    mv "${partial_path}" "${local_path}"
    return 0
  fi
  rm -f "${local_path}"
  echo "artifact size mismatch: ${artifact_name}" >&2
  return 1
}

download_available_artifacts() {
  local relative_path
  local remote_size

  while IFS=$'\t' read -r relative_path remote_size; do
    if [[ -z "${relative_path}" ]]; then
      continue
    fi
    if [[
      ! "${relative_path}" =~ ^[A-Za-z0-9_./-]+$
      || "/${relative_path}/" == *"/../"*
    ]]; then
      echo "unsafe remote artifact path: ${relative_path}" >&2
      return 1
    fi
    download_remote_path \
      "${REMOTE_DIR}/artifacts/${relative_path}" \
      "${LOCAL_OUT}/${relative_path}" \
      "${relative_path}"
  done < <(
    "${SSH[@]}" \
      "find '${REMOTE_DIR}/artifacts' -type f -printf '%P\\t%s\\n' | sort"
  )
}

verify_local_artifacts() {
  python3 "${REPO_ROOT}/tools/arrival_load_verify.py" \
    --run-dir "${LOCAL_OUT}"
  if [[ "$(
    cat "${LOCAL_OUT}/independent-verify/verify.exitcode"
  )" != 0 ]]; then
    echo "independent verifier did not publish exit code 0" >&2
    return 1
  fi
  test -f "${LOCAL_OUT}/artifact_hashes.json"
}

cleanup_staging() {
  if [[ -d "${STAGING_DIR}" ]]; then
    rm -rf "${STAGING_DIR}"
  fi
}
trap cleanup_staging EXIT

case "${MODE}" in
  preflight|smoke|cost-calibration|workload-calibration|canonical|download-only|verify-only) ;;
  *)
    echo "unsupported mode: ${MODE}" >&2
    exit 2
    ;;
esac

if [[ "${MODE}" == download-only ]]; then
  if [[ -z "${RUN_TAG_WAS_SET}" ]]; then
    echo "download-only requires RUN_TAG" >&2
    exit 2
  fi
  rm -rf "${LOCAL_OUT}"
  mkdir -p "${LOCAL_OUT}"
  download_available_artifacts
  exit 0
fi

if [[ "${MODE}" == verify-only ]]; then
  if [[ -z "${RUN_TAG_WAS_SET}" ]]; then
    echo "verify-only requires RUN_TAG" >&2
    exit 2
  fi
  python3 "${REPO_ROOT}/tools/arrival_load_verify.py" \
    --run-dir "${LOCAL_OUT}"
  exit 0
fi

if [[ -e "${STAGING_DIR}" ]]; then
  echo "staging directory already exists: ${STAGING_DIR}" >&2
  exit 2
fi
mkdir -p "$(dirname "${LOCAL_OUT}")"

python3 "${REPO_ROOT}/tools/arrival_load_gate.py" snapshot-source \
  --repo-root "${REPO_ROOT}" \
  --out-dir "${STAGING_DIR}"

"${SSH[@]}" \
  "mkdir -p '${REMOTE_DIR}' '${REMOTE_DIR}/tmp'; rm -rf '${REMOTE_DIR}/staging.upload'; mkdir -p '${REMOTE_DIR}/staging.upload'"
tar -C "${STAGING_DIR}" -cf - . | "${SSH_STREAM[@]}" \
  "set -e; tar -C '${REMOTE_DIR}/staging.upload' -xf -; rm -rf '${REMOTE_DIR}/staging'; mv '${REMOTE_DIR}/staging.upload' '${REMOTE_DIR}/staging'"

"${SSH_STREAM[@]}" "REMOTE_DIR='${REMOTE_DIR}' REMOTE_PYTHON='${REMOTE_PYTHON}' MODEL_PATH='${MODEL_PATH}' CUDA_DEVICE='${CUDA_DEVICE}' RUN_TAG='${RUN_TAG}' bash -s" <<'REMOTE_PREFLIGHT'
set -euo pipefail
export TMPDIR="${REMOTE_DIR}/tmp"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REMOTE_DIR}/staging/source"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
cd "${REMOTE_DIR}/staging/source"

"${REMOTE_PYTHON}" - "${REMOTE_DIR}" "${MODEL_PATH}" "${RUN_TAG}" <<'PY'
import importlib.metadata
import json
import pathlib
import socket
import sys

remote_dir = pathlib.Path(sys.argv[1])
model_path = pathlib.Path(sys.argv[2])
run_tag = sys.argv[3]
evidence = json.loads(
    (remote_dir / "staging/source_evidence.json").read_text()
)
sys.path.insert(0, str(remote_dir / "staging/source/tools"))
import source_audit

validated = source_audit.validate_source_snapshot(
    remote_dir / "staging/source",
    evidence,
    remote_dir / "staging/source.patch",
    expected_owned_roots=tuple(evidence["owned_roots"]),
)
config = json.loads((model_path / "config.json").read_text())
model_type = str(config.get("model_type", "")).lower()
architectures = " ".join(config.get("architectures", [])).lower()
if "qwen3" not in model_type and "qwen3" not in architectures:
    raise SystemExit("model config is not Qwen3")

import torch
import tinyvllm

try:
    flash_attn = importlib.metadata.version("flash-attn")
except importlib.metadata.PackageNotFoundError:
    flash_attn = "unavailable"

capability = {
    "schema_version": 1,
    "run_tag": run_tag,
    "source_tree_sha256": validated["source_tree_sha256"],
    "patch_sha256": evidence["patch_sha256"],
    "python": sys.version,
    "torch": torch.__version__,
    "cuda": str(torch.version.cuda),
    "flash_attn": flash_attn,
    "gpu": torch.cuda.get_device_name(0),
    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    "host": socket.gethostname(),
    "model_path": str(model_path),
    "model_identifier": "Qwen3-0.6B",
    "model_type": config.get("model_type"),
    "tinyvllm_file": str(pathlib.Path(tinyvllm.__file__).resolve()),
}
(remote_dir / "staging/source_preflight.json").write_text(
    json.dumps(capability, indent=2, sort_keys=True) + "\n"
)
(remote_dir / "staging/capability.json").write_text(
    json.dumps(capability, indent=2, sort_keys=True) + "\n"
)
PY

"${REMOTE_PYTHON}" tools/test_arrival_load_cost_calibration.py
"${REMOTE_PYTHON}" tools/test_arrival_load_gate.py
"${REMOTE_PYTHON}" tools/test_arrival_load_driver.py
"${REMOTE_PYTHON}" tools/test_arrival_load_verify.py
"${REMOTE_PYTHON}" tools/test_chunked_prefill.py

mkdir -p "${REMOTE_DIR}/preflight-artifacts"
cp "${REMOTE_DIR}/staging/source_preflight.json" \
  "${REMOTE_DIR}/preflight-artifacts/source_preflight.json"
cp "${REMOTE_DIR}/staging/capability.json" \
  "${REMOTE_DIR}/preflight-artifacts/capability.json"
REMOTE_PREFLIGHT

mkdir -p "${LOCAL_OUT}"
download_remote_path \
  "${REMOTE_DIR}/preflight-artifacts/source_preflight.json" \
  "${LOCAL_OUT}/source_preflight.json" \
  "source_preflight.json"
download_remote_path \
  "${REMOTE_DIR}/preflight-artifacts/capability.json" \
  "${LOCAL_OUT}/capability.json" \
  "capability.json"
cp "${STAGING_DIR}/source_evidence.json" "${LOCAL_OUT}/"
cp "${STAGING_DIR}/source.patch" "${LOCAL_OUT}/"
cp "${STAGING_DIR}/source_snapshot.tar.gz" "${LOCAL_OUT}/"

if [[ "${MODE}" == preflight ]]; then
  echo "preflight artifacts: ${LOCAL_OUT}"
  exit 0
fi

REMOTE_COMMAND=(
  "${REMOTE_PYTHON}"
  tools/arrival_load_gate.py
)
case "${MODE}" in
  smoke)
    REMOTE_COMMAND+=(
      run-smoke
      --run-dir "${REMOTE_DIR}/artifacts.work"
      --python-bin "${REMOTE_PYTHON}"
      --model-path "${MODEL_PATH}"
      --run-tag "${RUN_TAG}"
      --source-evidence "${REMOTE_DIR}/staging/source_evidence.json"
      --environment-evidence "${REMOTE_DIR}/staging/capability.json"
    )
    ;;
  cost-calibration)
    REMOTE_COMMAND+=(
      run-cost-calibration-remote
      --run-dir "${REMOTE_DIR}/artifacts.work"
      --python-bin "${REMOTE_PYTHON}"
      --model-path "${MODEL_PATH}"
      --run-tag "${RUN_TAG}"
      --source-evidence "${REMOTE_DIR}/staging/source_evidence.json"
      --environment-evidence "${REMOTE_DIR}/staging/capability.json"
      --smoke-run-dir "/data00/home/sitian/sitian-workspace01/tllm/arrival-load-runs/${SMOKE_RUN_TAG:?cost-calibration requires SMOKE_RUN_TAG}/artifacts"
    )
    ;;
  workload-calibration)
    REMOTE_COMMAND+=(
      run-workload-calibration-remote
      --run-dir "${REMOTE_DIR}/artifacts.work"
      --python-bin "${REMOTE_PYTHON}"
      --model-path "${MODEL_PATH}"
      --run-tag "${RUN_TAG}"
      --source-evidence "${REMOTE_DIR}/staging/source_evidence.json"
      --environment-evidence "${REMOTE_DIR}/staging/capability.json"
      --smoke-run-dir "/data00/home/sitian/sitian-workspace01/tllm/arrival-load-runs/${SMOKE_RUN_TAG:?workload-calibration requires SMOKE_RUN_TAG}/artifacts"
      --cost-calibration-run-dir "/data00/home/sitian/sitian-workspace01/tllm/arrival-load-runs/${COST_CALIBRATION_RUN_TAG:?workload-calibration requires COST_CALIBRATION_RUN_TAG}/artifacts"
    )
    ;;
  canonical)
    REMOTE_COMMAND+=(
      run-canonical
      --run-dir "${REMOTE_DIR}/artifacts.work"
      --python-bin "${REMOTE_PYTHON}"
      --model-path "${MODEL_PATH}"
      --run-tag "${RUN_TAG}"
      --source-evidence "${REMOTE_DIR}/staging/source_evidence.json"
      --environment-evidence "${REMOTE_DIR}/staging/capability.json"
      --smoke-run-dir "/data00/home/sitian/sitian-workspace01/tllm/arrival-load-runs/${SMOKE_RUN_TAG:?canonical requires SMOKE_RUN_TAG}/artifacts"
      --cost-calibration-run-dir "/data00/home/sitian/sitian-workspace01/tllm/arrival-load-runs/${COST_CALIBRATION_RUN_TAG:?canonical requires COST_CALIBRATION_RUN_TAG}/artifacts"
      --workload-calibration-run-dir "/data00/home/sitian/sitian-workspace01/tllm/arrival-load-runs/${WORKLOAD_CALIBRATION_RUN_TAG:?canonical requires WORKLOAD_CALIBRATION_RUN_TAG}/artifacts"
    )
    if [[ "${RESUME}" == 1 ]]; then
      REMOTE_COMMAND+=(--resume)
    fi
    ;;
esac

printf -v REMOTE_COMMAND_Q '%q ' "${REMOTE_COMMAND[@]}"
"${SSH_STREAM[@]}" "REMOTE_DIR='${REMOTE_DIR}' REMOTE_PYTHON='${REMOTE_PYTHON}' CUDA_DEVICE='${CUDA_DEVICE}' REMOTE_COMMAND_Q='${REMOTE_COMMAND_Q}' MODE='${MODE}' bash -s" <<'REMOTE_RUN'
set -euo pipefail
cd "${REMOTE_DIR}/staging/source"
rm -rf "${REMOTE_DIR}/artifacts.work"
if [[ -d "${REMOTE_DIR}/artifacts" ]]; then
  rm -rf "${REMOTE_DIR}/artifacts.previous"
  mv "${REMOTE_DIR}/artifacts" "${REMOTE_DIR}/artifacts.previous"
  cp -a "${REMOTE_DIR}/artifacts.previous" "${REMOTE_DIR}/artifacts.work"
else
  mkdir -p "${REMOTE_DIR}/artifacts.work"
fi
cp "${REMOTE_DIR}/staging/source_evidence.json" \
  "${REMOTE_DIR}/artifacts.work/source_evidence.json"
cp "${REMOTE_DIR}/staging/source.patch" \
  "${REMOTE_DIR}/artifacts.work/source.patch"
cp "${REMOTE_DIR}/staging/source_snapshot.tar.gz" \
  "${REMOTE_DIR}/artifacts.work/source_snapshot.tar.gz"
cp "${REMOTE_DIR}/staging/source_preflight.json" \
  "${REMOTE_DIR}/artifacts.work/source_preflight.json"
cp "${REMOTE_DIR}/staging/capability.json" \
  "${REMOTE_DIR}/artifacts.work/capability.json"

nohup bash -c "
  set +e
  export TMPDIR='${REMOTE_DIR}/tmp'
  export CUDA_VISIBLE_DEVICES='${CUDA_DEVICE}'
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONPATH='${REMOTE_DIR}/staging/source'
  cd '${REMOTE_DIR}/staging/source'
  eval ${REMOTE_COMMAND_Q} >'${REMOTE_DIR}/artifacts.work/runner.log' 2>&1
  run_rc=\$?
  if [[ \"\${run_rc}\" -eq 0 && '${MODE}' == canonical ]]; then
    '${REMOTE_PYTHON}' tools/arrival_load_gate.py finalize-artifacts \
      --run-dir '${REMOTE_DIR}/artifacts.work' \
      >>'${REMOTE_DIR}/artifacts.work/runner.log' 2>&1
    run_rc=\$?
  fi
  printf '%s\n' \"\${run_rc}\" >'${REMOTE_DIR}/artifacts.work/remote_exitcode.tmp'
  mv '${REMOTE_DIR}/artifacts.work/remote_exitcode.tmp' '${REMOTE_DIR}/artifacts.work/remote_exitcode'
  mv '${REMOTE_DIR}/artifacts.work' '${REMOTE_DIR}/artifacts'
" >/dev/null 2>&1 &
REMOTE_RUN

while true; do
  if "${SSH[@]}" "test -f '${REMOTE_DIR}/artifacts/remote_exitcode'"; then
    break
  fi
  sleep "${POLL_INTERVAL}"
done

rm -rf "${LOCAL_OUT}"
mkdir -p "${LOCAL_OUT}"
download_remote_path \
  "${REMOTE_DIR}/artifacts/remote_exitcode" \
  "${LOCAL_OUT}/remote_exitcode" \
  "remote_exitcode"
REMOTE_RC="$(cat "${LOCAL_OUT}/remote_exitcode")"
if [[ ! "${REMOTE_RC}" =~ ^[0-9]+$ ]]; then
  echo "invalid remote exit code: ${REMOTE_RC}" >&2
  download_available_artifacts
  exit 1
fi
if [[ "${REMOTE_RC}" -ne 0 ]]; then
  download_available_artifacts
  echo "remote gate failed with ${REMOTE_RC}; preserved at ${LOCAL_OUT}" >&2
  exit "${REMOTE_RC}"
fi

download_available_artifacts
if [[ "${MODE}" == canonical || "${MODE}" == smoke ]]; then
  verify_local_artifacts
fi
echo "verified artifacts: ${LOCAL_OUT}"
