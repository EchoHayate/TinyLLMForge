#!/usr/bin/env bash
# Does the multi-sequence decode CUDA Graph produce the same tokens as eager?
#
# usage:
#   tools/run_multi_sequence_graph_equivalence_remote.sh
#
# GATE A was rerun with `multi_sequence_cuda_graphs` enabled after finding that
# every decode batch above one had been falling back to eager. The constant fell
# from 40 ms to 12 ms. That number is only worth having if the graph path computes
# the same thing the eager path does, and the comment that installed the fallback
# in the first place says it might not:
#
#   Multi-sequence captured graphs can corrupt rows after the first one.
#
# So the same prompts are decoded greedily on both paths, in two processes,
# because the engine will not build its process group twice in one process. The
# token ids are then diffed. A divergence invalidates every measurement taken on
# the graph path; agreement is the licence to use them.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/tllm/env/bin/python}"
MODEL_CACHE="${MODEL_CACHE:-/data00/home/sitian/.ms_cache/Qwen}"
MODEL="${MODEL:-${MODEL_CACHE}/Qwen3-0___6B}"
SSH_SOCKET="${SSH_SOCKET:-/tmp/ssh-msgraph-equivalence}"
CUDA_DEVICE="${CUDA_DEVICE:-2}"
RUN_TAG="${RUN_TAG:-msgraph-equivalence-$(date +%Y%m%d-%H%M%S)}"
REMOTE_DIR="${REMOTE_DIR:-/data00/home/sitian/tllm/kvcapacity-runs/${RUN_TAG}}"
LOCAL_OUT="${LOCAL_OUT:-${REPO_ROOT}/experiments/kvcapacity_step_scaling/${RUN_TAG}}"
PROMPT_LENGTH="${PROMPT_LENGTH:-1024}"
BATCH="${BATCH:-4}"
MAX_TOKENS="${MAX_TOKENS:-32}"
SEED="${SEED:-20260914}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
REMOTE_USER_SITE="${REMOTE_USER_SITE:-/data00/home/sitian/.local/lib/python3.11/site-packages}"
REMOTE_SITE_EXCLUDE="${REMOTE_SITE_EXCLUDE:-flash_attn torchvision}"
REMOTE_LD_LIBRARY_PATH="${REMOTE_LD_LIBRARY_PATH:-/data00/home/sitian/tllm/miniforge/lib}"

WORKER_LOCAL="${REPO_ROOT}/tools/multi_sequence_graph_equivalence_worker.py"
[[ -f "${WORKER_LOCAL}" ]] || { echo "missing ${WORKER_LOCAL}" >&2; exit 2; }

if [[ -z "${KRB5CCNAME:-}" ]]; then
  for candidate in "${HOME}/krb5cc_sitian" "${HOME}/krb5cc_${USER}" "/tmp/krb5cc_$(id -u)"; do
    [[ -f "${candidate}" ]] && { export KRB5CCNAME="FILE:${candidate}"; break; }
  done
fi

SSH=(ssh -n -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=auto -o ControlPersist=900 -S "${SSH_SOCKET}" "${REMOTE_HOST}")
SSH_STREAM=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=auto -o ControlPersist=900 -S "${SSH_SOCKET}" "${REMOTE_HOST}")

"${SSH[@]}" true || { echo "cannot reach ${REMOTE_HOST}" >&2; exit 3; }

mkdir -p "${LOCAL_OUT}"
"${SSH[@]}" "mkdir -p '${REMOTE_DIR}/source'"
tar -C "${REPO_ROOT}" --exclude='__pycache__' --exclude='*.pyc' -cf - tinyvllm \
  | "${SSH_STREAM[@]}" "tar -C '${REMOTE_DIR}/source' -xf -"
"${SSH_STREAM[@]}" "cat > '${REMOTE_DIR}/worker.py'" < "${WORKER_LOCAL}"

cat > "${LOCAL_OUT}/source_provenance.json" <<PROVENANCE
{
  "source_revision": "$(git -C "${REPO_ROOT}" rev-parse HEAD)",
  "tinyvllm_dirty_paths": $(git -C "${REPO_ROOT}" status --porcelain -- tinyvllm | wc -l | tr -d ' '),
  "worker_sha256": "$(shasum -a 256 "${WORKER_LOCAL}" | awk '{print $1}')",
  "remote_dir": "${REMOTE_DIR}",
  "cuda_device": "${CUDA_DEVICE}"
}
PROVENANCE

SITEPATCH="${REMOTE_DIR}/sitepatch"
"${SSH_STREAM[@]}" "REMOTE_USER_SITE='${REMOTE_USER_SITE}' SITEPATCH='${SITEPATCH}' REMOTE_SITE_EXCLUDE='${REMOTE_SITE_EXCLUDE}' bash -s" <<'REMOTE_SITEPATCH'
set -euo pipefail
rm -rf "${SITEPATCH}"; mkdir -p "${SITEPATCH}"
for entry in "${REMOTE_USER_SITE}"/*; do
  base="$(basename "${entry}")"; drop=0
  for prefix in ${REMOTE_SITE_EXCLUDE}; do
    case "${base}" in "${prefix}"*) drop=1 ;; esac
  done
  [[ "${drop}" == 1 ]] || ln -sfn "${entry}" "${SITEPATCH}/${base}"
done
REMOTE_SITEPATCH

REMOTE_ENV="CUDA_VISIBLE_DEVICES='${CUDA_DEVICE}' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1"
REMOTE_ENV+=" TOKENIZERS_PARALLELISM=false PYTHONPATH='${SITEPATCH}:${REMOTE_DIR}/source'"
REMOTE_ENV+=" LD_LIBRARY_PATH='${REMOTE_LD_LIBRARY_PATH}'"

for path_mode in eager msgraph; do
  echo "########## ${path_mode} ##########"
  "${SSH_STREAM[@]}" "${REMOTE_ENV} '${REMOTE_PYTHON}' '${REMOTE_DIR}/worker.py' \
    --model-path '${MODEL}' \
    --out '${REMOTE_DIR}/${path_mode}.json' \
    --path-mode '${path_mode}' \
    --prompt-length '${PROMPT_LENGTH}' \
    --batch '${BATCH}' \
    --max-tokens '${MAX_TOKENS}' \
    --seed '${SEED}' \
    --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}'" 2>&1 | tee -a "${LOCAL_OUT}/runner.log"
  "${SSH_STREAM[@]}" "cat '${REMOTE_DIR}/${path_mode}.json'" > "${LOCAL_OUT}/${path_mode}.json"
done

python3 - "${LOCAL_OUT}" <<'PY' | tee "${LOCAL_OUT}/equivalence.txt"
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
eager = json.loads((out / "eager.json").read_text())
graph = json.loads((out / "msgraph.json").read_text())

print("multi-sequence decode graph against eager, greedy tokens")
print("=" * 64)
print("prompt length %d, batch %d, max tokens %d, seed %d"
      % (eager["prompt_length"], eager["batch"], eager["max_tokens"], eager["seed"]))
print("eager   dispatch %s" % eager["decode_dispatch_counts"])
print("msgraph dispatch %s" % graph["decode_dispatch_counts"])

graph_steps = graph["decode_dispatch_counts"].get("graph", 0)
if graph_steps == 0:
    print()
    print("VERDICT INCONCLUSIVE: the msgraph run never replayed a captured graph,")
    print("so this compares eager against eager and says nothing about the graph.")
    raise SystemExit(2)

mismatches = []
for key, expected in eager["completions"].items():
    actual = graph["completions"].get(key)
    if actual != expected:
        first = next(
            (i for i, (a, b) in enumerate(zip(expected, actual or [])) if a != b),
            min(len(expected), len(actual or [])),
        )
        mismatches.append((key, first, expected[first:first + 4], (actual or [])[first:first + 4]))

print()
print("sequences compared: %d, graph decode steps: %d" % (len(eager["completions"]), graph_steps))
if not mismatches:
    print("VERDICT MATCH: every sequence produced identical token ids on both paths.")
    print("The graph path is a legitimate execution path, so measurements taken on")
    print("it describe the engine and not a corrupted forward.")
    raise SystemExit(0)

print("VERDICT DIVERGENCE: %d of %d sequences differ." % (len(mismatches), len(eager["completions"])))
for key, index, expected, actual in mismatches:
    print("  seq %s first differs at token %d: eager %s against graph %s"
          % (key, index, expected, actual))
print()
print("Row corruption after the first sequence is the failure the eager fallback")
print("was installed to prevent. Every GATE A number measured on this path is void.")
raise SystemExit(1)
PY

echo "artifacts: ${LOCAL_OUT}"
