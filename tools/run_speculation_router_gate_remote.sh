#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "usage: $0 preflight|controlled-smoke|controlled|real-smoke|real [DRAFT_SOURCE_JSON PROMPT_BANK_JSON]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="sitian@10.232.195.203"
REMOTE_PYTHON="/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
MODEL_PATH="/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B"
SSH_SOCKET="${SSH_SOCKET:-/tmp/ssh-sitian-10.232.195.203}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RUN_TAG="${RUN_TAG:-qwen3-06b-router-${MODE}-$(date +%Y%m%d-%H%M%S)}"
LOCAL_OUT="${REPO_ROOT}/experiments/speculation_router/${RUN_TAG}"
STAGING_DIR="${LOCAL_OUT}.staging"
REMOTE_DIR="/data00/home/sitian/sitian-workspace01/tllm/speculation-router-runs/${RUN_TAG}"
POLL_INTERVAL="${POLL_INTERVAL:-20}"
CASE_LIMIT="${CASE_LIMIT:-}"
PROMPT_LIMIT="${PROMPT_LIMIT:-}"
REPETITIONS="${REPETITIONS:-3}"
WARMUP_REPETITIONS="${WARMUP_REPETITIONS:-1}"
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
SUCCESS_ARTIFACTS=(
  source_evidence.json
  source.patch
  source_snapshot.tar.gz
  source_preflight.json
  manifest.json
  capability.json
  case_rows.json
  event_rows.json
  router_rows.json
  summary.json
  report.md
  artifact_hashes.json
  remote_exitcode
  runner.log
)

download_remote_file() {
  local artifact_name="$1"
  download_remote_path \
    "${REMOTE_DIR}/artifacts/${artifact_name}" \
    "${LOCAL_OUT}/${artifact_name}" \
    "${artifact_name}"
}

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
        echo \
          "artifact block download retries exhausted: ${artifact_name} block=${block_index} bytes=${actual_block_bytes}/${expected_block_bytes}" \
          >&2
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
    download_remote_file "${relative_path}"
  done < <(
    "${SSH[@]}" \
      "find '${REMOTE_DIR}/artifacts' -type f -printf '%P\\t%s\\n' | sort"
  )
}

canonical_raw_sha256() {
  python3 - "$1" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
canonical = json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
).encode("utf-8")
print(hashlib.sha256(canonical).hexdigest())
PY
}

cleanup_staging() {
  if [[ -d "${STAGING_DIR}" ]]; then
    rm -rf "${STAGING_DIR}"
  fi
}
trap cleanup_staging EXIT

case "${MODE}" in
  preflight|controlled-smoke|controlled) ;;
  real-smoke|real)
    if [[ $# -ne 3 ]]; then
      echo "${MODE} requires DRAFT_SOURCE_JSON PROMPT_BANK_JSON" >&2
      exit 2
    fi
    DRAFT_SOURCE_JSON="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
    PROMPT_BANK_JSON="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
    ;;
  *)
    echo "unsupported mode: ${MODE}" >&2
    exit 2
    ;;
esac

if [[ -e "${STAGING_DIR}" ]]; then
  echo "staging directory already exists: ${STAGING_DIR}" >&2
  exit 2
fi
mkdir -p "$(dirname "${LOCAL_OUT}")"

python3 "${REPO_ROOT}/tools/speculation_router_gate.py" snapshot-source \
  --repo-root "${REPO_ROOT}" \
  --out-dir "${STAGING_DIR}"

if [[ "${MODE}" == real-smoke || "${MODE}" == real ]]; then
  python3 "${REPO_ROOT}/tools/speculation_router_gate.py" validate-real-input \
    --draft-source "${DRAFT_SOURCE_JSON}" \
    --prompt-bank "${PROMPT_BANK_JSON}"
  cp "${DRAFT_SOURCE_JSON}" "${STAGING_DIR}/draft_source.json"
  cp "${PROMPT_BANK_JSON}" "${STAGING_DIR}/prompt_bank.json"
fi

"${SSH[@]}" \
  "mkdir -p '${REMOTE_DIR}' '${REMOTE_DIR}/tmp'; rm -rf '${REMOTE_DIR}/staging.upload'; mkdir -p '${REMOTE_DIR}/staging.upload'"
tar -C "${STAGING_DIR}" -cf - . | "${SSH_STREAM[@]}" \
  "set -e; tar -C '${REMOTE_DIR}/staging.upload' -xf -; rm -rf '${REMOTE_DIR}/staging'; mv '${REMOTE_DIR}/staging.upload' '${REMOTE_DIR}/staging'"

"${SSH[@]}" "REMOTE_DIR='${REMOTE_DIR}' REMOTE_PYTHON='${REMOTE_PYTHON}' MODEL_PATH='${MODEL_PATH}' CUDA_DEVICE='${CUDA_DEVICE}' bash -s" <<'REMOTE_PREFLIGHT'
set -euo pipefail
export TMPDIR="${REMOTE_DIR}/tmp"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REMOTE_DIR}/staging/source"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
cd "${REMOTE_DIR}/staging/source"

"${REMOTE_PYTHON}" - "${REMOTE_DIR}" "${MODEL_PATH}" <<'PY'
import importlib.metadata
import json
import pathlib
import socket
import sys

remote_dir = pathlib.Path(sys.argv[1])
model_path = pathlib.Path(sys.argv[2])
evidence = json.loads((remote_dir / "staging/source_evidence.json").read_text())

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
hidden_size = int(config.get("hidden_size", 0))
if hidden_size <= 0:
    raise SystemExit("model config hidden_size is invalid")

import torch
import tinyvllm

try:
    flash_attn = importlib.metadata.version("flash-attn")
except importlib.metadata.PackageNotFoundError:
    flash_attn = "unavailable"

preflight = {
    "schema_version": 1,
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
    "hidden_size": hidden_size,
    "tinyvllm_file": str(pathlib.Path(tinyvllm.__file__).resolve()),
}
(remote_dir / "staging/source_preflight.json").write_text(
    json.dumps(preflight, indent=2, sort_keys=True) + "\n"
)
PY

"${REMOTE_PYTHON}" -m py_compile \
  tinyvllm/speculative/router.py \
  tools/profile_ngram_commit.py \
  tools/source_audit.py \
  tools/speculation_router_gate.py \
  tools/native_verifier_oracle.py

for test_file in \
  tools/test_speculation_router.py \
  tools/test_speculation_router_gate.py \
  tools/test_ngram_speculative.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_native_verifier_contract.py \
  tools/test_native_verifier_attention.py \
  tools/test_native_verifier_oracle.py \
  tools/test_native_verifier_gate.py \
  tools/test_run_speculation_router_gate_remote.py
do
  "${REMOTE_PYTHON}" "${test_file}"
done

mkdir -p "${REMOTE_DIR}/preflight-artifacts"
"${REMOTE_PYTHON}" tools/native_verifier_gate.py capability \
  --out "${REMOTE_DIR}/preflight-artifacts/capability.json"
cp "${REMOTE_DIR}/staging/source_preflight.json" \
  "${REMOTE_DIR}/preflight-artifacts/source_preflight.json"
REMOTE_PREFLIGHT

mkdir -p "${LOCAL_OUT}"
download_remote_path \
  "${REMOTE_DIR}/preflight-artifacts/capability.json" \
  "${LOCAL_OUT}/capability.json" \
  "capability.json"
download_remote_path \
  "${REMOTE_DIR}/preflight-artifacts/source_preflight.json" \
  "${LOCAL_OUT}/source_preflight.json" \
  "source_preflight.json"
cp "${STAGING_DIR}/source_evidence.json" "${LOCAL_OUT}/"
cp "${STAGING_DIR}/source.patch" "${LOCAL_OUT}/"
cp "${STAGING_DIR}/source_snapshot.tar.gz" "${LOCAL_OUT}/"

if [[ "${MODE}" == preflight ]]; then
  echo "preflight artifacts: ${LOCAL_OUT}"
  exit 0
fi

REMOTE_COMMAND=(
  "${REMOTE_PYTHON}"
  tools/speculation_router_gate.py
)
if [[ "${MODE}" == controlled-smoke || "${MODE}" == controlled ]]; then
  REMOTE_COMMAND+=(
    run-controlled
    --out-dir "${REMOTE_DIR}/artifacts.work"
    --python-bin "${REMOTE_PYTHON}"
    --model-path "${MODEL_PATH}"
    --source-evidence "${REMOTE_DIR}/staging/source_evidence.json"
    --source-patch "${REMOTE_DIR}/staging/source.patch"
    --source-preflight "${REMOTE_DIR}/staging/source_preflight.json"
    --host "${REMOTE_HOST}"
    --run-tag "${RUN_TAG}"
  )
  if [[ "${MODE}" == controlled-smoke ]]; then
    REMOTE_COMMAND+=(--case-limit "${CASE_LIMIT:-6}")
  elif [[ -n "${CASE_LIMIT}" ]]; then
    REMOTE_COMMAND+=(--case-limit "${CASE_LIMIT}")
  fi
else
  REMOTE_COMMAND+=(
    run-real
    --out-dir "${REMOTE_DIR}/artifacts.work"
    --python-bin "${REMOTE_PYTHON}"
    --model-path "${MODEL_PATH}"
    --source-evidence "${REMOTE_DIR}/staging/source_evidence.json"
    --source-patch "${REMOTE_DIR}/staging/source.patch"
    --source-preflight "${REMOTE_DIR}/staging/source_preflight.json"
    --draft-source "${REMOTE_DIR}/staging/draft_source.json"
    --prompt-bank "${REMOTE_DIR}/staging/prompt_bank.json"
    --host "${REMOTE_HOST}"
    --run-tag "${RUN_TAG}"
    --repetitions "${REPETITIONS}"
    --warmup-repetitions "${WARMUP_REPETITIONS}"
  )
  if [[ "${MODE}" == real-smoke ]]; then
    REMOTE_COMMAND+=(--prompt-limit "${PROMPT_LIMIT:-2}")
  elif [[ -n "${PROMPT_LIMIT}" ]]; then
    REMOTE_COMMAND+=(--prompt-limit "${PROMPT_LIMIT}")
  fi
fi
if [[ "${RESUME}" == 1 ]]; then
  REMOTE_COMMAND+=(--resume)
fi

printf -v REMOTE_COMMAND_Q '%q ' "${REMOTE_COMMAND[@]}"
"${SSH[@]}" "REMOTE_DIR='${REMOTE_DIR}' REMOTE_PYTHON='${REMOTE_PYTHON}' CUDA_DEVICE='${CUDA_DEVICE}' REMOTE_COMMAND_Q='${REMOTE_COMMAND_Q}' bash -s" <<'REMOTE_RUN'
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
cp "${REMOTE_DIR}/preflight-artifacts/capability.json" \
  "${REMOTE_DIR}/artifacts.work/capability.json"
cp "${REMOTE_DIR}/staging/source_snapshot.tar.gz" \
  "${REMOTE_DIR}/artifacts.work/source_snapshot.tar.gz"

nohup bash -c "
  set +e
  export TMPDIR='${REMOTE_DIR}/tmp'
  export CUDA_VISIBLE_DEVICES='${CUDA_DEVICE}'
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONPATH='${REMOTE_DIR}/staging/source'
  cd '${REMOTE_DIR}/staging/source'
  eval ${REMOTE_COMMAND_Q} >'${REMOTE_DIR}/artifacts.work/runner.log' 2>&1
  run_rc=\$?
  printf '%s\n' \"\${run_rc}\" >'${REMOTE_DIR}/artifacts.work/remote_exitcode.tmp'
  mv '${REMOTE_DIR}/artifacts.work/remote_exitcode.tmp' '${REMOTE_DIR}/artifacts.work/remote_exitcode'
  if [[ \"\${run_rc}\" -eq 0 ]]; then
    '${REMOTE_PYTHON}' tools/speculation_router_gate.py finalize-artifacts \
      --out-dir '${REMOTE_DIR}/artifacts.work' \
      >>'${REMOTE_DIR}/finalize.log' 2>&1
    finalize_rc=\$?
  else
    finalize_rc=\${run_rc}
  fi
  if [[ \"\${finalize_rc}\" -ne 0 ]]; then
    printf '%s\n' \"\${finalize_rc}\" >'${REMOTE_DIR}/artifacts.work/remote_exitcode.tmp'
    mv '${REMOTE_DIR}/artifacts.work/remote_exitcode.tmp' '${REMOTE_DIR}/artifacts.work/remote_exitcode'
  fi
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
download_remote_file remote_exitcode
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

for artifact_name in "${SUCCESS_ARTIFACTS[@]}"; do
  download_remote_file "${artifact_name}"
done
if [[ "${MODE}" == real-smoke || "${MODE}" == real ]]; then
  for artifact_name in \
    draft_source.json \
    prompt_bank.json \
    prompt_bank.sha256
  do
    download_remote_file "${artifact_name}"
  done
fi

while IFS=$'\t' read -r raw_name raw_payload_sha256; do
  download_remote_file "${raw_name}"
  actual_sha256="$(canonical_raw_sha256 "${LOCAL_OUT}/${raw_name}")"
  if [[ "${actual_sha256}" != "${raw_payload_sha256}" ]]; then
    echo "raw payload hash mismatch: ${raw_name}" >&2
    exit 1
  fi
done < <(
  python3 - "${LOCAL_OUT}/case_rows.json" "${MODE}" <<'PY'
import json
import re
import sys
from pathlib import Path

rows = json.loads(Path(sys.argv[1]).read_text())
stage = "real" if sys.argv[2].startswith("real") else "controlled"
for row in rows:
    expected = row.get("raw_payload_sha256")
    if not expected:
        if row.get("status") == "INCOMPLETE":
            continue
        raise SystemExit("complete row is missing raw_payload_sha256")
    identity = row["prompt_id"] if stage == "real" else row["case_id"]
    policy = row["policy"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", identity):
        raise SystemExit(f"unsafe raw payload identity: {identity!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", policy):
        raise SystemExit(f"unsafe raw payload policy: {policy!r}")
    print(f"raw/{identity}.{policy}.json\t{expected}")
PY
)

set +e
python3 "${REPO_ROOT}/tools/speculation_router_gate.py" verify --out-dir "${LOCAL_OUT}"
VERIFY_RC=$?
set -e
if [[ "${VERIFY_RC}" -ne 0 ]]; then
  echo "artifact verification failed; preserved at ${LOCAL_OUT}" >&2
  exit "${VERIFY_RC}"
fi

echo "verified artifacts: ${LOCAL_OUT}"
