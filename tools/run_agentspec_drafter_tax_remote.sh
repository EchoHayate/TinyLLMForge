#!/usr/bin/env bash
# Stage 1a remote runner: measure the action drafter GPU tax on the
# lab GPU box.
#
# Stage 0 priced action-level speculation over a declared drafter tax
# tau. This runner replaces that declaration with a measurement. It
# uploads a single self-contained worker, runs it under the remote
# CUDA python, and pulls back one deterministic JSON payload.
#
# usage:
#   tools/run_agentspec_drafter_tax_remote.sh preflight
#   tools/run_agentspec_drafter_tax_remote.sh measure
#   tools/run_agentspec_drafter_tax_remote.sh smoke
#
# The remote host is reached through the corporate jump proxy, which
# uses GSSAPI. Run `kinit` first or every mode fails at connect.
set -euo pipefail

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "usage: $0 preflight|smoke|measure" >&2
  exit 2
fi
case "${MODE}" in
  preflight|smoke|measure) ;;
  *)
    echo "unsupported mode: ${MODE}" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/sitian-workspace01/tllm/env/bin/python}"
MODEL_CACHE="${MODEL_CACHE:-/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen}"
ACTOR_MODEL="${ACTOR_MODEL:-${MODEL_CACHE}/Qwen3-8B}"
DRAFTER_MODEL="${DRAFTER_MODEL:-${MODEL_CACHE}/Qwen3-0___6B}"
SSH_SOCKET="${SSH_SOCKET:-/tmp/ssh-agentspec-drafter-tax}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RUN_TAG="${RUN_TAG:-drafter-tax-${MODE}-$(date +%Y%m%d-%H%M%S)}"
REMOTE_DIR="${REMOTE_DIR:-/data00/home/sitian/sitian-workspace01/tllm/agentspec-runs/${RUN_TAG}}"
LOCAL_OUT="${LOCAL_OUT:-${REPO_ROOT}/experiments/agentspec_drafter_tax/${RUN_TAG}}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-1024 4096 16384}"
ACTION_TOKENS="${ACTION_TOKENS:-32}"
COMPRESSED_BUDGET="${COMPRESSED_BUDGET:-512}"
CODE_VOCABULARY="${CODE_VOCABULARY:-4096}"
REPETITIONS="${REPETITIONS:-5}"
WARMUP="${WARMUP:-2}"

WORKER_LOCAL="${REPO_ROOT}/tools/agentspec_drafter_tax_worker.py"
if [[ ! -f "${WORKER_LOCAL}" ]]; then
  echo "missing worker: ${WORKER_LOCAL}" >&2
  exit 2
fi

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

if ! "${SSH[@]}" true 2>/tmp/agentspec-ssh-error; then
  echo "cannot reach ${REMOTE_HOST}" >&2
  sed 's/^/  /' /tmp/agentspec-ssh-error >&2
  echo "  hint: the jump proxy needs a Kerberos ticket; run kinit" >&2
  exit 3
fi

mkdir -p "${LOCAL_OUT}"
"${SSH[@]}" "mkdir -p '${REMOTE_DIR}'"
"${SSH_STREAM[@]}" "cat > '${REMOTE_DIR}/worker.py'" < "${WORKER_LOCAL}"

WORKER_SHA_LOCAL="$(shasum -a 256 "${WORKER_LOCAL}" | awk '{print $1}')"
WORKER_SHA_REMOTE="$(
  "${SSH[@]}" "sha256sum '${REMOTE_DIR}/worker.py' | cut -d' ' -f1"
)"
if [[ "${WORKER_SHA_LOCAL}" != "${WORKER_SHA_REMOTE}" ]]; then
  echo "worker upload hash mismatch" >&2
  exit 1
fi

if [[ "${MODE}" == preflight ]]; then
  "${SSH_STREAM[@]}" \
    "REMOTE_PYTHON='${REMOTE_PYTHON}' CUDA_DEVICE='${CUDA_DEVICE}' ACTOR_MODEL='${ACTOR_MODEL}' DRAFTER_MODEL='${DRAFTER_MODEL}' bash -s" \
    <<'REMOTE_PREFLIGHT' | tee "${LOCAL_OUT}/preflight.txt"
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
echo "host        $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used \
  --format=csv,noheader | sed 's/^/gpu         /'
for path in "${ACTOR_MODEL}" "${DRAFTER_MODEL}"; do
  if [[ -f "${path}/config.json" ]]; then
    echo "model ok    ${path}"
  else
    echo "model MISS  ${path}"
  fi
done
"${REMOTE_PYTHON}" - <<'PY'
import torch
print("torch       %s" % torch.__version__)
print("cuda        %s" % torch.version.cuda)
print("device      %s" % torch.cuda.get_device_name(0))
print("bf16        %s" % torch.cuda.is_bf16_supported())
try:
    import transformers
    print("transformers %s" % transformers.__version__)
except ImportError:
    print("transformers MISSING")
PY
REMOTE_PREFLIGHT
  echo "preflight written to ${LOCAL_OUT}/preflight.txt"
  exit 0
fi

REMOTE_ARGS=(
  --device "cuda:0"
  --output "${REMOTE_DIR}/drafter_tax.json"
  --repetitions "${REPETITIONS}"
  --warmup "${WARMUP}"
  --code-vocabulary "${CODE_VOCABULARY}"
  --compressed-budget "${COMPRESSED_BUDGET}"
  --action-tokens "${ACTION_TOKENS}"
)
if [[ "${MODE}" == smoke ]]; then
  REMOTE_ARGS+=(
    --actor-model "${DRAFTER_MODEL}"
    --drafter-model "${DRAFTER_MODEL}"
    --context-lengths 512
    --repetitions 2
    --warmup 1
  )
else
  # shellcheck disable=SC2206
  CONTEXT_ARRAY=(${CONTEXT_LENGTHS})
  REMOTE_ARGS+=(
    --actor-model "${ACTOR_MODEL}"
    --drafter-model "${DRAFTER_MODEL}"
    --context-lengths "${CONTEXT_ARRAY[@]}"
  )
fi

printf -v REMOTE_ARGS_Q '%q ' "${REMOTE_ARGS[@]}"
"${SSH_STREAM[@]}" \
  "REMOTE_DIR='${REMOTE_DIR}' REMOTE_PYTHON='${REMOTE_PYTHON}' CUDA_DEVICE='${CUDA_DEVICE}' REMOTE_ARGS_Q='${REMOTE_ARGS_Q}' bash -s" \
  <<'REMOTE_RUN' | tee "${LOCAL_OUT}/runner.log"
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
cd "${REMOTE_DIR}"
eval "${REMOTE_PYTHON}" worker.py "${REMOTE_ARGS_Q}"
REMOTE_RUN

"${SSH[@]}" "cat '${REMOTE_DIR}/drafter_tax.json'" \
  > "${LOCAL_OUT}/drafter_tax.json"
python3 - "${LOCAL_OUT}/drafter_tax.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
if payload["synthetic"]:
    raise SystemExit("remote payload is synthetic; not gate evidence")
print("payload sha256 %s" % payload["payload_sha256"])
PY

echo "artifacts: ${LOCAL_OUT}"
