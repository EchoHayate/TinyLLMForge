#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/sitian-workspace01/tllm/env/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B}"
REMOTE_BASE="${REMOTE_BASE:-/data00/home/sitian/sitian-workspace01/tllm/sam-drafter-gates}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-7}"
MODE="${1:-smoke}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)-$$}"
REMOTE_DIR="${REMOTE_BASE}/${RUN_TAG}"
LOCAL_OUT="${LOCAL_OUT:-${REPO_ROOT}/experiments/sam_drafter/${RUN_TAG}}"
BASE_SEED="${BASE_SEED:-20260715}"
CONTROL_SOCKET="${CONTROL_SOCKET:-/tmp/ssh-sitian-10.232.195.203}"
RESUME="${RESUME:-0}"

case "${MODE}" in
  preflight) REPETITIONS=0 ;;
  smoke) REPETITIONS=1 ;;
  canonical) REPETITIONS=7 ;;
  *) echo "usage: $0 {preflight|smoke|canonical}" >&2; exit 2 ;;
esac

SSH_ARGS=(-o BatchMode=yes)
if [[ -S "${CONTROL_SOCKET}" ]]; then
  SSH_ARGS+=(-S "${CONTROL_SOCKET}")
fi

SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
SOURCE_DIRTY=0
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  SOURCE_DIRTY=1
fi

ARCHIVE="$(mktemp "${TMPDIR:-/tmp}/sam-drafter-source.XXXXXX.tar.gz")"
REMOTE_HASHES="$(mktemp "${TMPDIR:-/tmp}/sam-drafter-remote-hashes.XXXXXX")"
LOCAL_HASHES="$(mktemp "${TMPDIR:-/tmp}/sam-drafter-local-hashes.XXXXXX")"
trap 'rm -f "${ARCHIVE}" "${REMOTE_HASHES}" "${LOCAL_HASHES}"' EXIT

tar -C "${REPO_ROOT}" -czf "${ARCHIVE}" \
  tinyvllm \
  tools/draft_model_schema.py \
  tools/profile_ngram_commit.py \
  tools/sam_drafter_gate.py

ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" \
  "set -euo pipefail; test -d '${MODEL_PATH}'; test -f '${MODEL_PATH}/config.json'; mkdir -p '${REMOTE_DIR}'; test -z \"\$(find '${REMOTE_DIR}' -mindepth 1 -maxdepth 1 -print -quit)\""
scp "${SSH_ARGS[@]}" "${ARCHIVE}" "${REMOTE_HOST}:${REMOTE_DIR}/source.tar.gz"

ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" \
  "set -euo pipefail; cd '${REMOTE_DIR}'; tar -xzf source.tar.gz; rm source.tar.gz; \
   '${REMOTE_PYTHON}' -m py_compile tinyvllm/speculative/sam.py tools/profile_ngram_commit.py tools/sam_drafter_gate.py; \
   '${REMOTE_PYTHON}' tools/profile_ngram_commit.py --help >/dev/null; \
   '${REMOTE_PYTHON}' tools/sam_drafter_gate.py --help >/dev/null"

echo "remote preflight passed: ${REMOTE_HOST}:${REMOTE_DIR}"
if [[ "${MODE}" == "preflight" ]]; then
  exit 0
fi

DIRTY_ARG=""
if [[ "${SOURCE_DIRTY}" == "1" ]]; then
  DIRTY_ARG="--source-dirty"
fi
RESUME_ARG=""
if [[ "${RESUME}" == "1" ]]; then
  RESUME_ARG="--resume"
fi

ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" \
  "set -euo pipefail; cd '${REMOTE_DIR}'; \
   CUDA_VISIBLE_DEVICES='${CUDA_DEVICE}' PYTHONDONTWRITEBYTECODE=1 \
   '${REMOTE_PYTHON}' tools/sam_drafter_gate.py run \
     --out-dir '${REMOTE_DIR}/artifacts' \
     --python-bin '${REMOTE_PYTHON}' \
     --model-path '${MODEL_PATH}' \
     --repetitions '${REPETITIONS}' \
     --base-seed '${BASE_SEED}' \
     --source-commit '${SOURCE_COMMIT}' \
     ${DIRTY_ARG} \
     ${RESUME_ARG} \
     --host '${REMOTE_HOST}'"

mkdir -p "${LOCAL_OUT}"
CANONICAL_FILES=(
  manifest.json
  raw_rows.json
  event_rows.json
  summary.json
  report.md
)
for file in "${CANONICAL_FILES[@]}"; do
  scp "${SSH_ARGS[@]}" \
    "${REMOTE_HOST}:${REMOTE_DIR}/artifacts/${file}" \
    "${LOCAL_OUT}/${file}"
done

ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" \
  "cd '${REMOTE_DIR}/artifacts' && sha256sum manifest.json raw_rows.json event_rows.json summary.json report.md" \
  | sed 's#  #  #' >"${REMOTE_HASHES}"
(
  cd "${LOCAL_OUT}"
  sha256sum manifest.json raw_rows.json event_rows.json summary.json report.md
) >"${LOCAL_HASHES}"
diff -u "${REMOTE_HASHES}" "${LOCAL_HASHES}"

PYTHONDONTWRITEBYTECODE=1 python3 \
  "${REPO_ROOT}/tools/sam_drafter_gate.py" verify \
  --out-dir "${LOCAL_OUT}" >/dev/null

echo "verified canonical artifacts: ${LOCAL_OUT}"
