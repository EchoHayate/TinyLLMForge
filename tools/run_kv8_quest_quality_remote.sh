#!/usr/bin/env bash
# Source-bound Qwen3-8B quality diagnostic for KV8 + Quest selective dequant.
#
# Runs two fresh engine processes on one remote GPU:
#   1. bf16 KV, full attention;
#   2. KV8, paired full attention and Quest top-16 on identical prompts.
#
# Remote data is confined to /data00/home/sitian/tllm/kvcapacity-runs/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/tllm/env/bin/python}"
TARGET_MODEL="${TARGET_MODEL:-/data00/home/sitian/.ms_cache/Qwen/Qwen3-8B}"
CUDA_DEVICE="${CUDA_DEVICE:-2}"
RUN_TAG="${RUN_TAG:-kv8-quest-quality-$(date +%Y%m%d-%H%M%S)}"
REMOTE_ROOT="/data00/home/sitian/tllm/kvcapacity-runs/"
REMOTE_DIR="${REMOTE_ROOT}${RUN_TAG}"
LOCAL_OUT="${LOCAL_OUT:-${REPO_ROOT}/experiments/kvcapacity_step_scaling/${RUN_TAG}}"
SSH_SOCKET="${SSH_SOCKET:-/tmp/ssh-kv8-quest-quality}"
REMOTE_USER_SITE="${REMOTE_USER_SITE:-/data00/home/sitian/.local/lib/python3.11/site-packages}"
REMOTE_SITE_EXCLUDE="${REMOTE_SITE_EXCLUDE:-flash_attn torchvision}"
REMOTE_LD_LIBRARY_PATH="${REMOTE_LD_LIBRARY_PATH:-/data00/home/sitian/tllm/miniforge/lib}"

if [[ "${REMOTE_DIR}" != "${REMOTE_ROOT}"* || "${REMOTE_DIR}" == "${REMOTE_ROOT}" ]]; then
  echo "unsafe REMOTE_DIR: ${REMOTE_DIR}" >&2
  exit 2
fi
if [[ -e "${LOCAL_OUT}" ]]; then
  echo "immutable local tag already exists: ${LOCAL_OUT}" >&2
  exit 2
fi

export KRB5CCNAME="${KRB5CCNAME:-FILE:/Users/bytedance/krb5cc_sitian}"
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

"${SSH[@]}" true
if "${SSH[@]}" "test -e '${REMOTE_DIR}'"; then
  echo "immutable remote tag already exists: ${REMOTE_DIR}" >&2
  exit 2
fi

SOURCE_REVISION="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
SOURCE_DIRTY="$(
  git -C "${REPO_ROOT}" status --porcelain -- tinyvllm tools/eval_needle.py |
    wc -l |
    tr -d ' '
)"
if [[ "${SOURCE_DIRTY}" != 0 ]]; then
  echo "source-bound paths are dirty; commit them before running" >&2
  exit 2
fi

mkdir -p "${LOCAL_OUT}"
"${SSH[@]}" "mkdir -p '${REMOTE_DIR}/source'"
(
  cd "${REPO_ROOT}"
  git archive "${SOURCE_REVISION}" -- tinyvllm tools/eval_needle.py
) | "${SSH_STREAM[@]}" "tar -C '${REMOTE_DIR}/source' -xf -"

SITEPATCH="${REMOTE_DIR}/sitepatch"
"${SSH_STREAM[@]}" \
  "REMOTE_USER_SITE='${REMOTE_USER_SITE}' SITEPATCH='${SITEPATCH}' REMOTE_SITE_EXCLUDE='${REMOTE_SITE_EXCLUDE}' bash -s" \
  <<'REMOTE_SITEPATCH'
set -euo pipefail
mkdir -p "${SITEPATCH}"
for entry in "${REMOTE_USER_SITE}"/*; do
  base="$(basename "${entry}")"
  drop=0
  for prefix in ${REMOTE_SITE_EXCLUDE}; do
    case "${base}" in
      "${prefix}"*) drop=1 ;;
    esac
  done
  if [[ "${drop}" == 0 ]]; then
    ln -s "${entry}" "${SITEPATCH}/${base}"
  fi
done
REMOTE_SITEPATCH

cat > "${LOCAL_OUT}/source_provenance.json" <<PROVENANCE
{
  "source_revision": "${SOURCE_REVISION}",
  "source_dirty_paths": ${SOURCE_DIRTY},
  "remote_dir": "${REMOTE_DIR}",
  "cuda_device": "${CUDA_DEVICE}",
  "model": "${TARGET_MODEL}",
  "fixed_prompts": true,
  "needle_style": "newline",
  "context_lens": [8192],
  "depths": [0.0, 0.25, 0.5, 0.75, 1.0],
  "num_trials": 5,
  "quest_top_k_blocks": 16,
  "quest_min_seq_len": 512
}
PROVENANCE

REMOTE_ENV="CUDA_VISIBLE_DEVICES='${CUDA_DEVICE}'"
REMOTE_ENV+=" PYTHONNOUSERSITE=1"
REMOTE_ENV+=" PYTHONDONTWRITEBYTECODE=1"
REMOTE_ENV+=" TOKENIZERS_PARALLELISM=false"
REMOTE_ENV+=" PYTHONPATH='${SITEPATCH}:${REMOTE_DIR}/source'"
REMOTE_ENV+=" LD_LIBRARY_PATH='${REMOTE_LD_LIBRARY_PATH}'"

COMMON_ARGS=(
  --model "${TARGET_MODEL}"
  --fixed-prompts
  --needle-style newline
  --context-lens 8192
  --depths 0.0 0.25 0.5 0.75 1.0
  --num-trials 5
  --max-output-len 16
  --max-model-len 16384
  --gpu-memory-utilization 0.85
  --max-num-seqs 32
  --quest-min-seq-len 512
)
printf -v COMMON_ARGS_Q ' %q' "${COMMON_ARGS[@]}"

"${SSH_STREAM[@]}" \
  "set -o pipefail; ${REMOTE_ENV} '${REMOTE_PYTHON}' '${REMOTE_DIR}/source/tools/eval_needle.py'${COMMON_ARGS_Q} --kv-quant-bits 0 --top-k-blocks-list -1 --out-json '${REMOTE_DIR}/bf16.json' 2>&1 | tee '${REMOTE_DIR}/bf16.log'"

"${SSH_STREAM[@]}" \
  "set -o pipefail; ${REMOTE_ENV} '${REMOTE_PYTHON}' '${REMOTE_DIR}/source/tools/eval_needle.py'${COMMON_ARGS_Q} --kv-quant-bits 8 --top-k-blocks-list -1 16 --out-json '${REMOTE_DIR}/kv8_paired.json' 2>&1 | tee '${REMOTE_DIR}/kv8_paired.log'"

for artifact in bf16.json bf16.log kv8_paired.json kv8_paired.log; do
  "${SSH_STREAM[@]}" "cat '${REMOTE_DIR}/${artifact}'" > "${LOCAL_OUT}/${artifact}"
done
"${SSH[@]}" "sha256sum '${REMOTE_DIR}/bf16.json' '${REMOTE_DIR}/kv8_paired.json'" \
  > "${LOCAL_OUT}/remote_sha256.txt"

echo "quality artifacts written to ${LOCAL_OUT}"
