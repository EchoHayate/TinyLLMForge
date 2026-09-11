#!/usr/bin/env bash
# Stage 1b step 1 remote runner: measure the prompted drafter's action
# match rate on the A100 box.
#
# usage:
#   tools/run_agentspec_prompted_drafter_remote.sh preflight
#   tools/run_agentspec_prompted_drafter_remote.sh smoke
#   tools/run_agentspec_prompted_drafter_remote.sh measure
#
# This job trains nothing and times nothing. Cost was settled on the
# serving path in Stage 1a-bis; the only question here is whether an
# untrained prompted 0.6B writes the exact next action.
#
# The evaluation sets are built on CPU by
# tools/agentspec_prompted_drafter_evalset.py and are *not* committed:
# they carry corpus text, one corpus is CC BY-NC, and the derived
# statistics are the only thing that belongs in the repository.
#
# Environment notes carried over from Stage 1a/1a-bis, all learned the
# hard way:
#   - the GSSAPI ticket lives in a shared FILE credential cache, and
#     the macOS default API cache reads as empty from a
#     non-interactive session even when a valid ticket exists;
#   - the venv transformers is newer than its own torch and dies at
#     import, while the user site transformers works but ships a
#     flash-attn wheel that does not match.
# So the runner builds a symlink farm over the user site with
# flash_attn and torchvision filtered out and puts it ahead of the
# venv. This worker does not import tinyvllm, but it does import
# transformers, so the same combination is required.
set -euo pipefail

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "usage: $0 preflight|smoke|select|measure|diagnose" >&2
  exit 2
fi
case "${MODE}" in
  preflight|smoke|select|measure|diagnose) ;;
  *)
    echo "unsupported mode: ${MODE}" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/tllm/env/bin/python}"
MODEL_CACHE="${MODEL_CACHE:-/data00/home/sitian/.ms_cache/Qwen}"
DRAFTER_MODEL="${DRAFTER_MODEL:-${MODEL_CACHE}/Qwen3-0___6B}"
SSH_SOCKET="${SSH_SOCKET:-/tmp/ssh-agentspec-prompted-drafter}"
CUDA_DEVICE="${CUDA_DEVICE:-2}"
RUN_TAG="${RUN_TAG:-prompted-drafter-${MODE}-$(date +%Y%m%d-%H%M%S)}"
REMOTE_DIR="${REMOTE_DIR:-/data00/home/sitian/tllm/agentspec-runs/${RUN_TAG}}"
LOCAL_OUT="${LOCAL_OUT:-${REPO_ROOT}/experiments/agentspec_prompted_drafter/${RUN_TAG}}"
EVALSET_DIR="${EVALSET_DIR:-${REPO_ROOT}/.agent_runtime/agentspec_evalsets}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SWE_TOKEN_CAP="${SWE_TOKEN_CAP:-11}"
APIGEN_TOKEN_CAP="${APIGEN_TOKEN_CAP:-27}"
PROMPT_BUDGET="${PROMPT_BUDGET:-512}"
PROMPT_STYLE="${PROMPT_STYLE:-v2}"
SWE_STYLE="${SWE_STYLE:-v2}"
APIGEN_STYLE="${APIGEN_STYLE:-v3}"
SELECT_LIMIT="${SELECT_LIMIT:-256}"
MEASURE_LIMIT="${MEASURE_LIMIT:-1000}"
# Set to e.g. _ctx3584 to point at an evaluation set built with a
# larger context window, which is how the "is the answer even in the
# 512 tokens" question gets asked.
EVALSET_SUFFIX="${EVALSET_SUFFIX:-}"
ACTOR_MODEL="${ACTOR_MODEL:-${MODEL_CACHE}/Qwen3-8B}"
DIAGNOSE_MODEL="${DIAGNOSE_MODEL:-${ACTOR_MODEL}}"
REMOTE_USER_SITE="${REMOTE_USER_SITE:-/data00/home/sitian/.local/lib/python3.11/site-packages}"
REMOTE_SITE_EXCLUDE="${REMOTE_SITE_EXCLUDE:-flash_attn torchvision}"
REMOTE_LD_LIBRARY_PATH="${REMOTE_LD_LIBRARY_PATH:-/data00/home/sitian/tllm/miniforge/lib}"

WORKER_LOCAL="${REPO_ROOT}/tools/agentspec_prompted_drafter_match_worker.py"
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
  -o ControlMaster=auto -o ControlPersist=1800
  -S "${SSH_SOCKET}" "${REMOTE_HOST}"
)
SSH_STREAM=(
  ssh -o BatchMode=yes -o ConnectTimeout=20
  -o ControlMaster=auto -o ControlPersist=1800
  -S "${SSH_SOCKET}" "${REMOTE_HOST}"
)

if ! "${SSH[@]}" true 2>/tmp/agentspec-prompted-ssh-error; then
  echo "cannot reach ${REMOTE_HOST}" >&2
  sed 's/^/  /' /tmp/agentspec-prompted-ssh-error >&2
  echo "  hint: no valid Kerberos ticket; check klist, then kinit" >&2
  exit 3
fi

mkdir -p "${LOCAL_OUT}"
"${SSH[@]}" "mkdir -p '${REMOTE_DIR}/source/tools' '${REMOTE_DIR}/evalsets'"

# The worker resolves tinyvllm.agentspec.action relative to its own
# parent directory, so it has to sit inside a source tree rather than
# next to it.
tar -C "${REPO_ROOT}" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  -cf - tinyvllm/__init__.py tinyvllm/agentspec \
  | "${SSH_STREAM[@]}" "tar -C '${REMOTE_DIR}/source' -xf -"
"${SSH_STREAM[@]}" "cat > '${REMOTE_DIR}/source/tools/worker.py'" \
  < "${WORKER_LOCAL}"

WORKER_SHA_LOCAL="$(shasum -a 256 "${WORKER_LOCAL}" | awk '{print $1}')"
WORKER_SHA_REMOTE="$(
  "${SSH[@]}" "sha256sum '${REMOTE_DIR}/source/tools/worker.py' | cut -d' ' -f1"
)"
if [[ "${WORKER_SHA_LOCAL}" != "${WORKER_SHA_REMOTE}" ]]; then
  echo "worker upload hash mismatch" >&2
  exit 1
fi

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
REMOTE_ENV+=" PYTHONPATH='${SITEPATCH}'"
REMOTE_ENV+=" LD_LIBRARY_PATH='${REMOTE_LD_LIBRARY_PATH}'"

if [[ "${MODE}" == preflight ]]; then
  "${SSH_STREAM[@]}" \
    "${REMOTE_ENV} REMOTE_PYTHON='${REMOTE_PYTHON}' DRAFTER_MODEL='${DRAFTER_MODEL}' bash -s" \
    <<'REMOTE_PREFLIGHT' | tee "${LOCAL_OUT}/preflight.txt"
set -euo pipefail
echo "host        $(hostname)"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
  --format=csv,noheader | sed 's/^/gpu         /'
if [[ -f "${DRAFTER_MODEL}/config.json" ]]; then
  echo "model ok    ${DRAFTER_MODEL}"
else
  echo "model MISS  ${DRAFTER_MODEL}"
fi
"${REMOTE_PYTHON}" - <<'PY'
import torch
import transformers
print("torch        %s" % torch.__version__)
print("transformers %s" % transformers.__version__)
print("device       %s" % torch.cuda.get_device_name(0))
PY
REMOTE_PREFLIGHT
  echo "preflight written to ${LOCAL_OUT}/preflight.txt"
  exit 0
fi

if [[ "${MODE}" == smoke ]]; then
  # corpus:variant:cap:style:offset:limit
  JOBS=("swe_agent:tail:${SWE_TOKEN_CAP}:v2:0:64")
elif [[ "${MODE}" == select ]]; then
  # Prompt selection runs on a held-out head of the evaluation set.
  # The measurement below then starts past it, so the prompt is never
  # chosen on the rows it is scored on.
  JOBS=(
    "swe_agent:tail:${SWE_TOKEN_CAP}:v1:0:${SELECT_LIMIT}"
    "swe_agent:tail:${SWE_TOKEN_CAP}:v2:0:${SELECT_LIMIT}"
    "swe_agent:tail:${SWE_TOKEN_CAP}:v3:0:${SELECT_LIMIT}"
    "swe_agent:tail_tools:${SWE_TOKEN_CAP}:v3:0:${SELECT_LIMIT}"
    "apigen:tail:${APIGEN_TOKEN_CAP}:v2:0:${SELECT_LIMIT}"
    "apigen:tail:${APIGEN_TOKEN_CAP}:v3:0:${SELECT_LIMIT}"
    "apigen:tail_tools:${APIGEN_TOKEN_CAP}:v3:0:${SELECT_LIMIT}"
  )
elif [[ "${MODE}" == diagnose ]]; then
  # Same task, same 512-token compressed context, but the *actor*
  # model does the drafting. This cannot ship: an 8B drafter costs
  # what the actor costs. It separates two very different failure
  # explanations, namely a drafter that is too small and a compressed
  # context that does not contain the answer.
  JOBS=(
    "swe_agent:tail:${SWE_TOKEN_CAP}:${SWE_STYLE}:0:${SELECT_LIMIT}"
    "apigen:tail:${APIGEN_TOKEN_CAP}:${APIGEN_STYLE}:0:${SELECT_LIMIT}"
  )
  DRAFTER_MODEL="${DIAGNOSE_MODEL}"
  BATCH_SIZE="${DIAGNOSE_BATCH_SIZE:-16}"
else
  JOBS=(
    "swe_agent:tail:${SWE_TOKEN_CAP}:${SWE_STYLE}:${SELECT_LIMIT}:${MEASURE_LIMIT}"
    "swe_agent:tail_tools:${SWE_TOKEN_CAP}:${SWE_STYLE}:${SELECT_LIMIT}:${MEASURE_LIMIT}"
    "apigen:tail:${APIGEN_TOKEN_CAP}:${APIGEN_STYLE}:${SELECT_LIMIT}:${MEASURE_LIMIT}"
    "apigen:tail_tools:${APIGEN_TOKEN_CAP}:${APIGEN_STYLE}:${SELECT_LIMIT}:${MEASURE_LIMIT}"
  )
fi

UPLOADED=""
for job in "${JOBS[@]}"; do
  IFS=':' read -r corpus variant cap style offset limit <<<"${job}"
  local_evalset="${EVALSET_DIR}/evalset_${corpus}${EVALSET_SUFFIX}.jsonl"
  if [[ ! -f "${local_evalset}" ]]; then
    echo "missing evalset: ${local_evalset}" >&2
    echo "  build it with tools/agentspec_prompted_drafter_evalset.py" >&2
    exit 2
  fi
  case " ${UPLOADED} " in
    *" ${corpus} "*) ;;
    *)
      "${SSH_STREAM[@]}" \
        "cat > '${REMOTE_DIR}/evalsets/evalset_${corpus}${EVALSET_SUFFIX}.jsonl'" \
        < "${local_evalset}"
      sha_local="$(shasum -a 256 "${local_evalset}" | awk '{print $1}')"
      sha_remote="$(
        "${SSH[@]}" "sha256sum '${REMOTE_DIR}/evalsets/evalset_${corpus}${EVALSET_SUFFIX}.jsonl' | cut -d' ' -f1"
      )"
      if [[ "${sha_local}" != "${sha_remote}" ]]; then
        echo "evalset upload hash mismatch for ${corpus}" >&2
        exit 1
      fi
      echo "uploaded    ${corpus} ${sha_local}"
      UPLOADED="${UPLOADED} ${corpus}"
      ;;
  esac
done

for job in "${JOBS[@]}"; do
  IFS=':' read -r corpus variant cap style offset limit <<<"${job}"
  tag="${corpus}_${variant}_${style}${EVALSET_SUFFIX}"
  REMOTE_ARGS=(
    --evalset "${REMOTE_DIR}/evalsets/evalset_${corpus}${EVALSET_SUFFIX}.jsonl"
    --model "${DRAFTER_MODEL}"
    --output "${REMOTE_DIR}/match_${tag}.json"
    --variant "${variant}"
    --prompt-style "${style}"
    --prompt-budget "${PROMPT_BUDGET}"
    --token-cap "${cap}"
    --batch-size "${BATCH_SIZE}"
    --offset "${offset}"
  )
  if [[ "${limit}" != "0" ]]; then
    REMOTE_ARGS+=(--limit "${limit}")
  fi
  printf -v REMOTE_ARGS_Q '%q ' "${REMOTE_ARGS[@]}"
  echo ""
  echo "=== ${tag} cap=${cap} offset=${offset} limit=${limit} ==="
  "${SSH_STREAM[@]}" \
    "${REMOTE_ENV} REMOTE_DIR='${REMOTE_DIR}' REMOTE_PYTHON='${REMOTE_PYTHON}' REMOTE_ARGS_Q='${REMOTE_ARGS_Q}' bash -s" \
    <<'REMOTE_RUN' 2>&1 | tee "${LOCAL_OUT}/runner_${tag}.log"
set -euo pipefail
cd "${REMOTE_DIR}/source"
eval "${REMOTE_PYTHON}" tools/worker.py "${REMOTE_ARGS_Q}"
REMOTE_RUN
  "${SSH[@]}" "cat '${REMOTE_DIR}/match_${tag}.json'" \
    > "${LOCAL_OUT}/match_${tag}.json"
done

echo ""
echo "artifacts: ${LOCAL_OUT}"
