#!/usr/bin/env bash
# Stage 1a-bis remote runner: measure actor demand D on the serving
# path instead of an eager Hugging Face decode loop.
#
# usage:
#   tools/run_agentspec_engine_demand_remote.sh preflight
#   tools/run_agentspec_engine_demand_remote.sh smoke
#   tools/run_agentspec_engine_demand_remote.sh measure
#
# The remote source tree is taken from the local git HEAD so the
# measurement is pinned to a revision rather than to whatever happens
# to be lying around on the box.
#
# Environment notes, all learned the hard way in Stage 1a:
#   - the GSSAPI ticket lives in a shared FILE credential cache, and
#     the macOS default API cache reads as empty from a
#     non-interactive session even when a valid ticket exists;
#   - the venv carries transformers 5.8.1, which is newer than its own
#     torch 2.4.1 and dies at import, but it also carries the only
#     flash-attn build that matches that torch;
#   - the user site carries transformers 4.51.3, which works, next to
#     a flash-attn wheel that does not.
# So the runner builds a symlink farm over the user site with
# flash_attn and torchvision filtered out, puts it ahead of the venv,
# and lets flash-attn resolve to the venv copy. tinyvllm needs
# flash_attn at import, so this combination is required, not cosmetic.
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
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/tllm/env/bin/python}"
MODEL_CACHE="${MODEL_CACHE:-/data00/home/sitian/.ms_cache/Qwen}"
ACTOR_MODEL="${ACTOR_MODEL:-${MODEL_CACHE}/Qwen3-8B}"
DRAFTER_MODEL="${DRAFTER_MODEL:-${MODEL_CACHE}/Qwen3-0___6B}"
SSH_SOCKET="${SSH_SOCKET:-/tmp/ssh-agentspec-engine-demand}"
CUDA_DEVICE="${CUDA_DEVICE:-2}"
RUN_TAG="${RUN_TAG:-engine-demand-${MODE}-$(date +%Y%m%d-%H%M%S)}"
REMOTE_DIR="${REMOTE_DIR:-/data00/home/sitian/tllm/agentspec-runs/${RUN_TAG}}"
LOCAL_OUT="${LOCAL_OUT:-${REPO_ROOT}/experiments/agentspec_engine_demand/${RUN_TAG}}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-1024 4096 16384}"
ACTION_TOKENS="${ACTION_TOKENS:-32}"
COMPRESSED_BUDGET="${COMPRESSED_BUDGET:-512}"
CODE_VOCABULARY="${CODE_VOCABULARY:-4096}"
REPETITIONS="${REPETITIONS:-5}"
WARMUP="${WARMUP:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
REMOTE_USER_SITE="${REMOTE_USER_SITE:-/data00/home/sitian/.local/lib/python3.11/site-packages}"
REMOTE_SITE_EXCLUDE="${REMOTE_SITE_EXCLUDE:-flash_attn torchvision}"
REMOTE_LD_LIBRARY_PATH="${REMOTE_LD_LIBRARY_PATH:-/data00/home/sitian/tllm/miniforge/lib}"

WORKER_LOCAL="${REPO_ROOT}/tools/agentspec_engine_demand_worker.py"
if [[ ! -f "${WORKER_LOCAL}" ]]; then
  echo "missing worker: ${WORKER_LOCAL}" >&2
  exit 2
fi

if [[ -z "${KRB5CCNAME:-}" ]]; then
  for candidate in \
    "${HOME}/krb5cc_sitian" \
    "${HOME}/krb5cc_${USER}" \
    "/tmp/krb5cc_$(id -u)"
  do
    if [[ -f "${candidate}" ]]; then
      export KRB5CCNAME="FILE:${candidate}"
      break
    fi
  done
fi

SSH=(
  ssh -n -o BatchMode=yes -o ConnectTimeout=20
  -o ControlMaster=auto -o ControlPersist=900
  -S "${SSH_SOCKET}" "${REMOTE_HOST}"
)
SSH_STREAM=(
  ssh -o BatchMode=yes -o ConnectTimeout=20
  -o ControlMaster=auto -o ControlPersist=900
  -S "${SSH_SOCKET}" "${REMOTE_HOST}"
)

if ! "${SSH[@]}" true 2>/tmp/agentspec-engine-ssh-error; then
  echo "cannot reach ${REMOTE_HOST}" >&2
  sed 's/^/  /' /tmp/agentspec-engine-ssh-error >&2
  echo "  hint: no valid Kerberos ticket; check klist, then kinit" >&2
  exit 3
fi

SOURCE_REVISION="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
SOURCE_DIRTY="$(
  git -C "${REPO_ROOT}" status --porcelain -- tinyvllm | wc -l | tr -d ' '
)"
mkdir -p "${LOCAL_OUT}"
"${SSH[@]}" "mkdir -p '${REMOTE_DIR}/source'"

# tinyvllm is uploaded from the working tree rather than from git
# archive, because a dirty tinyvllm is exactly the case where the
# measured revision must match what is on disk. The dirty count is
# recorded next to the payload.
tar -C "${REPO_ROOT}" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  -cf - tinyvllm \
  | "${SSH_STREAM[@]}" "tar -C '${REMOTE_DIR}/source' -xf -"
"${SSH_STREAM[@]}" "cat > '${REMOTE_DIR}/worker.py'" < "${WORKER_LOCAL}"

WORKER_SHA_LOCAL="$(shasum -a 256 "${WORKER_LOCAL}" | awk '{print $1}')"
WORKER_SHA_REMOTE="$(
  "${SSH[@]}" "sha256sum '${REMOTE_DIR}/worker.py' | cut -d' ' -f1"
)"
if [[ "${WORKER_SHA_LOCAL}" != "${WORKER_SHA_REMOTE}" ]]; then
  echo "worker upload hash mismatch" >&2
  exit 1
fi
cat > "${LOCAL_OUT}/source_provenance.json" <<PROVENANCE
{
  "source_revision": "${SOURCE_REVISION}",
  "tinyvllm_dirty_paths": ${SOURCE_DIRTY},
  "worker_sha256": "${WORKER_SHA_LOCAL}",
  "remote_dir": "${REMOTE_DIR}",
  "cuda_device": "${CUDA_DEVICE}"
}
PROVENANCE

SITEPATCH="${REMOTE_DIR}/sitepatch"
"${SSH_STREAM[@]}" \
  "REMOTE_USER_SITE='${REMOTE_USER_SITE}' SITEPATCH='${SITEPATCH}' REMOTE_SITE_EXCLUDE='${REMOTE_SITE_EXCLUDE}' bash -s" \
  <<'REMOTE_SITEPATCH'
set -euo pipefail
rm -rf "${SITEPATCH}"
mkdir -p "${SITEPATCH}"
linked=0
skipped=0
for entry in "${REMOTE_USER_SITE}"/*; do
  base="$(basename "${entry}")"
  drop=0
  for prefix in ${REMOTE_SITE_EXCLUDE}; do
    case "${base}" in
      "${prefix}"*) drop=1 ;;
    esac
  done
  if [[ "${drop}" == 1 ]]; then
    skipped=$((skipped + 1))
    continue
  fi
  ln -sfn "${entry}" "${SITEPATCH}/${base}"
  linked=$((linked + 1))
done
echo "sitepatch   linked=${linked} skipped=${skipped}" >&2
REMOTE_SITEPATCH

REMOTE_ENV="CUDA_VISIBLE_DEVICES='${CUDA_DEVICE}'"
REMOTE_ENV+=" PYTHONNOUSERSITE=1"
REMOTE_ENV+=" PYTHONDONTWRITEBYTECODE=1"
REMOTE_ENV+=" TOKENIZERS_PARALLELISM=false"
REMOTE_ENV+=" PYTHONPATH='${SITEPATCH}:${REMOTE_DIR}/source'"
REMOTE_ENV+=" LD_LIBRARY_PATH='${REMOTE_LD_LIBRARY_PATH}'"

if [[ "${MODE}" == preflight ]]; then
  "${SSH_STREAM[@]}" \
    "${REMOTE_ENV} REMOTE_PYTHON='${REMOTE_PYTHON}' ACTOR_MODEL='${ACTOR_MODEL}' DRAFTER_MODEL='${DRAFTER_MODEL}' bash -s" \
    <<'REMOTE_PREFLIGHT' | tee "${LOCAL_OUT}/preflight.txt"
set -euo pipefail
echo "host        $(hostname)"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
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
import transformers
import flash_attn
print("torch        %s" % torch.__version__)
print("transformers %s" % transformers.__version__)
print("flash_attn   %s" % flash_attn.__version__)
print("device       %s" % torch.cuda.get_device_name(0))
import tinyvllm
from tinyvllm import LLM, SamplingParams
from tinyvllm.config import Config
print("tinyvllm     %s" % tinyvllm.__file__)
print("enforce_eager default %s" % Config.enforce_eager)
PY
REMOTE_PREFLIGHT
  echo "preflight written to ${LOCAL_OUT}/preflight.txt"
  exit 0
fi

REMOTE_ARGS=(
  --output "${REMOTE_DIR}/engine_demand.json"
  --code-vocabulary "${CODE_VOCABULARY}"
  --compressed-budget "${COMPRESSED_BUDGET}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
)
if [[ "${MODE}" == smoke ]]; then
  REMOTE_ARGS+=(
    --actor-model "${DRAFTER_MODEL}"
    --drafter-model "${DRAFTER_MODEL}"
    --context-lengths 1024
    --action-tokens 4
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
    --action-tokens "${ACTION_TOKENS}"
    --repetitions "${REPETITIONS}"
    --warmup "${WARMUP}"
  )
fi

printf -v REMOTE_ARGS_Q '%q ' "${REMOTE_ARGS[@]}"
"${SSH_STREAM[@]}" \
  "${REMOTE_ENV} REMOTE_DIR='${REMOTE_DIR}' REMOTE_PYTHON='${REMOTE_PYTHON}' REMOTE_ARGS_Q='${REMOTE_ARGS_Q}' bash -s" \
  <<'REMOTE_RUN' 2>&1 | tee "${LOCAL_OUT}/runner.log"
set -euo pipefail
cd "${REMOTE_DIR}"
eval "${REMOTE_PYTHON}" worker.py "${REMOTE_ARGS_Q}"
REMOTE_RUN

"${SSH[@]}" "cat '${REMOTE_DIR}/engine_demand.json'" \
  > "${LOCAL_OUT}/engine_demand.json"
python3 - "${LOCAL_OUT}/engine_demand.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
print("payload sha256 %s" % payload["payload_sha256"])
print("modes          %s" % ", ".join(sorted(payload["modes"])))
PY

echo "artifacts: ${LOCAL_OUT}"
