#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/sitian-workspace01/tllm/env/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B}"
REMOTE_BASE="${REMOTE_BASE:-/data00/home/sitian/sitian-workspace01/tllm/native-verifier-gates}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-7}"
MODE="${1:-smoke}"
RUN_TAG="${RUN_TAG:-qwen3-06b-$(date +%Y%m%d-%H%M%S)-$$}"
REMOTE_DIR="${REMOTE_BASE}/${RUN_TAG}"
LOCAL_OUT="${LOCAL_OUT:-${REPO_ROOT}/experiments/native_verifier/${RUN_TAG}}"
CONTROL_SOCKET="${CONTROL_SOCKET:-/tmp/ssh-sitian-10.232.195.203}"

case "${MODE}" in
  preflight|smoke) ;;
  *)
    echo "usage: $0 {preflight|smoke}" >&2
    exit 2
    ;;
esac

SSH_ARGS=(-o BatchMode=yes)
SCP_ARGS=(-o BatchMode=yes)
if [[ -S "${CONTROL_SOCKET}" ]]; then
  SSH_ARGS+=(-S "${CONTROL_SOCKET}")
  SCP_ARGS+=(-o "ControlPath=${CONTROL_SOCKET}")
fi

SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
SOURCE_DIRTY=0
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  SOURCE_DIRTY=1
fi
if [[ "${MODE}" == "smoke" && "${SOURCE_DIRTY}" != "0" ]]; then
  echo "native verifier smoke requires a clean local source" >&2
  exit 2
fi

ARCHIVE="$(mktemp "${TMPDIR:-/tmp}/native-verifier-source.XXXXXX.tar.gz")"
trap 'rm -f "${ARCHIVE}"' EXIT

tar -C "${REPO_ROOT}" -czf "${ARCHIVE}" \
  tinyvllm \
  tools/draft_model_schema.py \
  tools/profile_ngram_commit.py \
  tools/native_verifier_oracle.py \
  tools/native_verifier_gate.py \
  tools/test_native_verifier_attention.py

ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" \
  "set -euo pipefail; \
   test -d '${MODEL_PATH}'; \
   test -f '${MODEL_PATH}/config.json'; \
   mkdir -p '${REMOTE_DIR}'; \
   test -z \"\$(find '${REMOTE_DIR}' -mindepth 1 -maxdepth 1 -print -quit)\""
scp "${SCP_ARGS[@]}" \
  "${ARCHIVE}" \
  "${REMOTE_HOST}:${REMOTE_DIR}/source.tar.gz"

ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" \
  "set -euo pipefail; \
   cd '${REMOTE_DIR}'; \
   tar -xzf source.tar.gz; \
   rm source.tar.gz; \
   mkdir -p artifacts logs '${REMOTE_DIR}/tmp'; \
   TMPDIR='${REMOTE_DIR}/tmp' \
   CUDA_VISIBLE_DEVICES='${CUDA_DEVICE}' \
   PYTHONDONTWRITEBYTECODE=1 \
   PYTHONPATH='${REMOTE_DIR}' \
   '${REMOTE_PYTHON}' - '${MODEL_PATH}' '${SOURCE_COMMIT}' '${SOURCE_DIRTY}' > artifacts/preflight.json <<'PY'
import flash_attn
import json
import socket
import sys
from pathlib import Path

import torch
import tinyvllm

model = Path(sys.argv[1])
config = json.loads((model / 'config.json').read_text())
normalized = str(model).lower().replace('_', '').replace('-', '').replace('.', '')
if config.get('model_type') != 'qwen3':
    raise SystemExit(f\"unexpected model_type={config.get('model_type')}\")
if 'qwen3' not in normalized or (
    '06b' not in normalized and config.get('hidden_size') != 1024
):
    raise SystemExit(f\"model is not identifiable as Qwen3-0.6B: {model}\")
print(json.dumps({
    'source_commit': sys.argv[2],
    'source_dirty': bool(int(sys.argv[3])),
    'model_path': str(model.resolve()),
    'model_identifier': 'Qwen3-0.6B',
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'flash_attn': getattr(flash_attn, '__version__', 'unknown'),
    'bf16_supported': torch.cuda.is_bf16_supported(),
    'gpu': torch.cuda.get_device_name(0),
    'hostname': socket.gethostname(),
    'tinyvllm_path': str(Path(tinyvllm.__file__).resolve()),
}, indent=2, sort_keys=True))
PY
   TMPDIR='${REMOTE_DIR}/tmp' \
   CUDA_VISIBLE_DEVICES='${CUDA_DEVICE}' \
   PYTHONDONTWRITEBYTECODE=1 \
   PYTHONPATH='${REMOTE_DIR}' \
   '${REMOTE_PYTHON}' -m py_compile \
     tinyvllm/speculative/verifier.py \
     tinyvllm/utils/context.py \
     tinyvllm/engine/model_runner.py \
     tinyvllm/layers/attention.py \
     tools/profile_ngram_commit.py \
     tools/native_verifier_oracle.py \
     tools/native_verifier_gate.py \
     tools/test_native_verifier_attention.py; \
   TMPDIR='${REMOTE_DIR}/tmp' \
   CUDA_VISIBLE_DEVICES='${CUDA_DEVICE}' \
   PYTHONDONTWRITEBYTECODE=1 \
   PYTHONPATH='${REMOTE_DIR}' \
   '${REMOTE_PYTHON}' tools/test_native_verifier_attention.py; \
   CUDA_VISIBLE_DEVICES='${CUDA_DEVICE}' \
   PYTHONDONTWRITEBYTECODE=1 \
   PYTHONPATH='${REMOTE_DIR}' \
   '${REMOTE_PYTHON}' tools/native_verifier_gate.py capability \
     --out artifacts/capability.json"

mkdir -p "${LOCAL_OUT}"
for file in preflight.json capability.json; do
  scp "${SCP_ARGS[@]}" \
    "${REMOTE_HOST}:${REMOTE_DIR}/artifacts/${file}" \
    "${LOCAL_OUT}/${file}"
done

echo "native verifier remote preflight passed"
echo "remote_dir=${REMOTE_DIR}"
echo "local_out=${LOCAL_OUT}"

if [[ "${MODE}" == "preflight" ]]; then
  exit 0
fi

ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" \
  "set -euo pipefail; cd '${REMOTE_DIR}'; \
   TMPDIR='${REMOTE_DIR}/tmp' \
   CUDA_VISIBLE_DEVICES='${CUDA_DEVICE}' \
   PYTHONDONTWRITEBYTECODE=1 \
   PYTHONPATH='${REMOTE_DIR}' \
   '${REMOTE_PYTHON}' tools/native_verifier_gate.py run \
     --out-dir '${REMOTE_DIR}/artifacts' \
     --python-bin '${REMOTE_PYTHON}' \
     --model-path '${MODEL_PATH}' \
     --source-commit '${SOURCE_COMMIT}' \
     $([[ '${SOURCE_DIRTY}' == '1' ]] && printf '%s' '--source-dirty') \
     --host '${REMOTE_HOST}' \
     --run-tag '${RUN_TAG}' \
     --preflight '${REMOTE_DIR}/artifacts/preflight.json'"

CANONICAL_FILES=(
  manifest.json
  capability.json
  case_rows.json
  event_rows.json
  summary.json
  report.md
)
for file in "${CANONICAL_FILES[@]}"; do
  scp "${SCP_ARGS[@]}" \
    "${REMOTE_HOST}:${REMOTE_DIR}/artifacts/${file}" \
    "${LOCAL_OUT}/${file}"
done

PYTHONDONTWRITEBYTECODE=1 python3 \
  "${REPO_ROOT}/tools/native_verifier_gate.py" verify \
  --out-dir "${LOCAL_OUT}"
