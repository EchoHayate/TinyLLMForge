#!/usr/bin/env bash
# GATE A remote runner: measure step_ms(L, B) on the tinyvllm serving path.
#
# usage:
#   tools/run_kvcapacity_step_scaling_remote.sh preflight
#   tools/run_kvcapacity_step_scaling_remote.sh smoke
#   tools/run_kvcapacity_step_scaling_remote.sh measure
#
# Stage 0 of the latent KV capacity line assumes step_ms(L, B) = c0 + c1 * L * B.
# Both constants were fit from batch-1 data, so the batch term is an
# extrapolation that every Stage 0 capacity number depends on. This runner
# measures it. The verdict is computed locally afterwards and is allowed to come
# back FAIL, which sends the line back to refit Stage 0 rather than on to GATE B.
#
# The remote source tree is taken from the local git HEAD so the measurement is
# pinned to a revision rather than to whatever happens to be lying around on the
# box.
#
# Environment notes, inherited from run_agentspec_engine_demand_remote.sh and
# learned the hard way in the previous research line:
#   - the GSSAPI ticket lives in a shared FILE credential cache, and the macOS
#     default API cache reads as empty from a non-interactive session even when a
#     valid ticket exists;
#   - the venv carries transformers 5.8.1, which is newer than its own torch
#     2.4.1 and dies at import, but it also carries the only flash-attn build
#     that matches that torch;
#   - the user site carries transformers 4.51.3, which works, next to a
#     flash-attn wheel that does not.
# So the runner builds a symlink farm over the user site with flash_attn and
# torchvision filtered out, puts it ahead of the venv, and lets flash-attn
# resolve to the venv copy. tinyvllm needs flash_attn at import, so this
# combination is required, not cosmetic.
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
TARGET_MODEL="${TARGET_MODEL:-${MODEL_CACHE}/Qwen3-8B}"
SMOKE_MODEL="${SMOKE_MODEL:-${MODEL_CACHE}/Qwen3-0___6B}"
SSH_SOCKET="${SSH_SOCKET:-/tmp/ssh-kvcapacity-step-scaling}"
CUDA_DEVICE="${CUDA_DEVICE:-2}"
RUN_TAG="${RUN_TAG:-step-scaling-${MODE}-$(date +%Y%m%d-%H%M%S)}"
REMOTE_DIR="${REMOTE_DIR:-/data00/home/sitian/tllm/kvcapacity-runs/${RUN_TAG}}"
LOCAL_OUT="${LOCAL_OUT:-${REPO_ROOT}/experiments/kvcapacity_step_scaling/${RUN_TAG}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
WARMUP_STEPS="${WARMUP_STEPS:-8}"
MEASURED_STEPS="${MEASURED_STEPS:-24}"
SEED="${SEED:-20260913}"
# Deliberately small and deliberately not the pre-registered grid. The worker
# records that fact and the verdict refuses to return PASS for it.
SMOKE_GRID="${SMOKE_GRID:-1024:1,2,4;2048:1,2;4096:1}"
REMOTE_USER_SITE="${REMOTE_USER_SITE:-/data00/home/sitian/.local/lib/python3.11/site-packages}"
REMOTE_SITE_EXCLUDE="${REMOTE_SITE_EXCLUDE:-flash_attn torchvision}"
REMOTE_LD_LIBRARY_PATH="${REMOTE_LD_LIBRARY_PATH:-/data00/home/sitian/tllm/miniforge/lib}"

WORKER_LOCAL="${REPO_ROOT}/tools/kvcapacity_step_scaling_worker.py"
VERDICT_LOCAL="${REPO_ROOT}/tools/kvcapacity_step_scaling_verdict.py"
for required in "${WORKER_LOCAL}" "${VERDICT_LOCAL}"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing tool: ${required}" >&2
    exit 2
  fi
done

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

if ! "${SSH[@]}" true 2>/tmp/kvcapacity-step-scaling-ssh-error; then
  echo "cannot reach ${REMOTE_HOST}" >&2
  sed 's/^/  /' /tmp/kvcapacity-step-scaling-ssh-error >&2
  echo "  hint: no valid Kerberos ticket; check klist, then kinit" >&2
  exit 3
fi

SOURCE_REVISION="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
SOURCE_DIRTY="$(
  git -C "${REPO_ROOT}" status --porcelain -- tinyvllm | wc -l | tr -d ' '
)"
mkdir -p "${LOCAL_OUT}"
"${SSH[@]}" "mkdir -p '${REMOTE_DIR}/source'"

# tinyvllm is uploaded from the working tree rather than from git archive,
# because a dirty tinyvllm is exactly the case where the measured revision must
# match what is on disk. The dirty count is recorded next to the payload.
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
  "cuda_device": "${CUDA_DEVICE}",
  "mode": "${MODE}"
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
  # The KV budget is checked explicitly because the largest pre-registered cell
  # needs 262144 resident tokens, and at the Qwen3-8B GQA footprint of 147456
  # bytes per token that is 38.7 GiB of KV on top of the weights.
  "${SSH_STREAM[@]}" \
    "${REMOTE_ENV} REMOTE_PYTHON='${REMOTE_PYTHON}' TARGET_MODEL='${TARGET_MODEL}' SMOKE_MODEL='${SMOKE_MODEL}' bash -s" \
    <<'REMOTE_PREFLIGHT' | tee "${LOCAL_OUT}/preflight.txt"
set -euo pipefail
echo "host        $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader | sed 's/^/gpu         /'
for path in "${TARGET_MODEL}" "${SMOKE_MODEL}"; do
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
free, total = torch.cuda.mem_get_info()
print("gpu free     %.1f GiB of %.1f GiB" % (free / 2**30, total / 2**30))
import tinyvllm
from tinyvllm.config import Config
print("tinyvllm     %s" % tinyvllm.__file__)
print("block size   %s" % Config.kvcache_block_size)
print("eager default %s" % Config.enforce_eager)

# What the largest pre-registered cell will ask for.
bytes_per_token = 147456
largest = 262144
need = largest * bytes_per_token
print("largest cell L*B=%d needs %.1f GiB of KV" % (largest, need / 2**30))
weights = 8.2e9 * 2
budget = total * 0.85 - weights
print("KV budget at util 0.85 is roughly %.1f GiB" % (budget / 2**30))
print("verdict      %s" % ("fits" if budget > need else "DOES NOT FIT"))
PY
REMOTE_PREFLIGHT
  echo "preflight written to ${LOCAL_OUT}/preflight.txt"
  exit 0
fi

# One process per context length, because the engine initialises a
# torch.distributed process group on construction and refuses to do it twice in
# the same process. The smoke run found this by losing three of six cells to
# "trying to initialize the default process group twice". The per-context
# artifacts are merged by the verdict tool, which re-derives whether the union of
# cells covers the pre-registered grid.
if [[ "${MODE}" == smoke ]]; then
  MODEL_FOR_RUN="${SMOKE_MODEL}"
  GRID_GROUPS=()
  IFS=';' read -r -a SMOKE_GROUPS <<< "${SMOKE_GRID}"
  for group in "${SMOKE_GROUPS[@]}"; do
    [[ -n "${group}" ]] && GRID_GROUPS+=("${group}")
  done
  RUN_MEASURED_STEPS=12
else
  MODEL_FOR_RUN="${TARGET_MODEL}"
  # Amended after the first full run: Qwen3-8B clamps max_model_len to its
  # max_position_embeddings of 40960, so 65536 and 131072 were unreachable.
  # Resident tokens are extended through batch instead, preserving the L * B range.
  GRID_GROUPS=(
    "8192:1,2,4,8,16,32"
    "16384:1,2,4,8,16"
    "32768:1,2,4,8"
    "40448:1,2,4"
  )
  RUN_MEASURED_STEPS="${MEASURED_STEPS}"
fi

# Both execution paths are measured. Decode graphs are captured for batches
# [1,2,4,8,...] but not for every batch, and torch.compile recompiles per shape,
# so the default path can be discontinuous in batch for reasons unrelated to KV
# bytes. The eager path is uniform across batches and is the one the affine model
# should be fitted against; the default path is what serving actually runs. A
# disagreement between them is a finding, not noise.
if [[ "${MODE}" == smoke ]]; then
  EXECUTION_PATHS=("eager")
else
  EXECUTION_PATHS=("eager" "graph")
fi

VERDICT_STATUS=0
for path_mode in "${EXECUTION_PATHS[@]}"; do
  PAYLOAD_ARGS=()
  for group in "${GRID_GROUPS[@]}"; do
    context="${group%%:*}"
    tag="${path_mode}-ctx${context}"
    REMOTE_OUT="${REMOTE_DIR}/step_scaling-${tag}.json"
    REMOTE_ARGS=(
      --model-path "${MODEL_FOR_RUN}"
      --out "${REMOTE_OUT}"
      --grid-spec "${group}"
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
      --seed "${SEED}"
      --warmup-steps "${WARMUP_STEPS}"
      --measured-steps "${RUN_MEASURED_STEPS}"
    )
    if [[ "${path_mode}" == eager ]]; then
      REMOTE_ARGS+=(--enforce-eager)
    fi
    printf -v REMOTE_ARGS_Q '%q ' "${REMOTE_ARGS[@]}"
    echo "=== ${tag} ===" | tee -a "${LOCAL_OUT}/runner.log"
    set +e
    "${SSH_STREAM[@]}" \
      "${REMOTE_ENV} REMOTE_DIR='${REMOTE_DIR}' REMOTE_PYTHON='${REMOTE_PYTHON}' REMOTE_ARGS_Q='${REMOTE_ARGS_Q}' bash -s" \
      <<'REMOTE_RUN' 2>&1 | tee -a "${LOCAL_OUT}/runner.log"
set -euo pipefail
cd "${REMOTE_DIR}"
eval "${REMOTE_PYTHON}" worker.py "${REMOTE_ARGS_Q}"
REMOTE_RUN
    RUN_STATUS="${PIPESTATUS[0]}"
    set -e
    if [[ "${RUN_STATUS}" != 0 ]]; then
      echo "context group ${group} failed under ${path_mode}; continuing" >&2
      continue
    fi
    LOCAL_PART="${LOCAL_OUT}/step_scaling-${tag}.json"
    if "${SSH[@]}" "cat '${REMOTE_OUT}'" > "${LOCAL_PART}" 2>/dev/null; then
      PAYLOAD_ARGS+=(--payload "${LOCAL_PART}")
    else
      rm -f "${LOCAL_PART}"
      echo "no artifact produced for ${tag}" >&2
    fi
  done

  if [[ "${#PAYLOAD_ARGS[@]}" == 0 ]]; then
    echo "no artifact was produced under ${path_mode}" >&2
    VERDICT_STATUS=1
    continue
  fi

  # The verdict runs locally and deliberately decides the gate, so its exit
  # status is preserved rather than swallowed.
  echo
  echo "########## verdict: ${path_mode} path ##########"
  set +e
  python3 "${VERDICT_LOCAL}" \
    "${PAYLOAD_ARGS[@]}" \
    --out "${LOCAL_OUT}/verdict-${path_mode}.json" \
    | tee "${LOCAL_OUT}/verdict-${path_mode}.txt"
  STATUS="${PIPESTATUS[0]}"
  set -e
  # The eager path is the one that decides the gate, since it is the only path
  # that is uniform across batches.
  if [[ "${path_mode}" == eager ]]; then
    VERDICT_STATUS="${STATUS}"
  fi
done

echo
echo "artifacts: ${LOCAL_OUT}"
if [[ "${VERDICT_STATUS}" != 0 ]]; then
  echo "GATE A did not pass; see ${LOCAL_OUT}/verdict-eager.txt" >&2
fi
exit "${VERDICT_STATUS}"
