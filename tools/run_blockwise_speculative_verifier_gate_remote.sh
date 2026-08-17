#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KRB5CCNAME="${KRB5CCNAME:-FILE:/Users/bytedance/krb5cc_sitian}"
export KRB5CCNAME
REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/sitian-workspace01/tllm/env/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B}"
REMOTE_REPO="${REMOTE_REPO:-/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOCAL_OUT="${LOCAL_OUT:-${REPO_ROOT}/artifacts/blockwise_speculative_verifier/${RUN_TAG}}"
REMOTE_OUT="${REMOTE_REPO}/artifacts/blockwise_speculative_verifier/${RUN_TAG}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-30}"
STALE_RECHECK_ATTEMPTS="${STALE_RECHECK_ATTEMPTS:-10}"
STALE_RECHECK_INTERVAL_SECONDS="${STALE_RECHECK_INTERVAL_SECONDS:-0.1}"
SSH_OPTIONS=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ControlMaster=no
  -o ControlPath=none
)
SSH=(ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}")
RSYNC_SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none"

mkdir -p "${LOCAL_OUT}"

rsync -a \
  -e "${RSYNC_SSH}" \
  "${REPO_ROOT}/tinyvllm/" \
  "${REMOTE_HOST}:${REMOTE_REPO}/tinyvllm/"

(cd "${REPO_ROOT}" && rsync -a --relative \
  -e "${RSYNC_SSH}" \
  ./tools/blockwise_speculative_verifier_gate.py \
  ./tools/blockwise_speculative_verifier_worker.py \
  ./tools/verify_blockwise_speculative_verifier_gate.py \
  ./tools/test_blockwise_speculative_verifier_gate.py \
  "${REMOTE_HOST}:${REMOTE_REPO}/")

"${SSH[@]}" \
  "mkdir -p '${REMOTE_OUT}' && \
   cd '${REMOTE_REPO}' && \
   '${REMOTE_PYTHON}' -m py_compile \
     tinyvllm/engine/speculative_residency.py \
     tinyvllm/engine/model_runner.py \
     tinyvllm/layers/attention.py \
     tinyvllm/utils/context.py \
     tools/blockwise_speculative_verifier_gate.py \
     tools/blockwise_speculative_verifier_worker.py \
     tools/verify_blockwise_speculative_verifier_gate.py"

"${SSH[@]}" \
  "cat > '${REMOTE_OUT}/campaign.sh.tmp' && \
   chmod 700 '${REMOTE_OUT}/campaign.sh.tmp' && \
   mv \
     '${REMOTE_OUT}/campaign.sh.tmp' \
     '${REMOTE_OUT}/campaign.sh'" <<EOF
#!/usr/bin/env bash
set +e
cd '${REMOTE_REPO}'
export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'
export PYTHONPATH='${REMOTE_REPO}'
'${REMOTE_PYTHON}' \
  tools/blockwise_speculative_verifier_gate.py run \
  --model '${MODEL_PATH}' \
  --out '${REMOTE_OUT}/result.json' \
  > '${REMOTE_OUT}/remote.log' 2>&1
campaign_status=\$?
if (( campaign_status == 0 )); then
  '${REMOTE_PYTHON}' \
    tools/verify_blockwise_speculative_verifier_gate.py \
    '${REMOTE_OUT}/result.json' \
    '${REMOTE_REPO}' \
    --output '${REMOTE_OUT}/verify.remote.json' \
    >> '${REMOTE_OUT}/remote.log' 2>&1
  campaign_status=\$?
fi
printf '%s\n' "\${campaign_status}" \
  > '${REMOTE_OUT}/campaign.exit_code.tmp'
mv \
  '${REMOTE_OUT}/campaign.exit_code.tmp' \
  '${REMOTE_OUT}/campaign.exit_code'
if (( campaign_status == 0 )); then
  printf '%s\n' COMPLETE > '${REMOTE_OUT}/campaign.status'
else
  printf '%s\n' FAILED > '${REMOTE_OUT}/campaign.status'
fi
exit "\${campaign_status}"
EOF

set +e
"${SSH[@]}" "
  status=\$(cat '${REMOTE_OUT}/campaign.status' 2>/dev/null || true)
  pid=\$(cat '${REMOTE_OUT}/campaign.pid' 2>/dev/null || true)
  if [[ \"\${status}\" == RUNNING ]] &&
     [[ \"\${pid}\" =~ ^[0-9]+\$ ]] &&
     ! kill -0 \"\${pid}\" 2>/dev/null; then
    for ((attempt = 0; attempt < ${STALE_RECHECK_ATTEMPTS}; attempt++)); do
      sleep '${STALE_RECHECK_INTERVAL_SECONDS}'
      status=\$(cat '${REMOTE_OUT}/campaign.status' 2>/dev/null || true)
      pid=\$(cat '${REMOTE_OUT}/campaign.pid' 2>/dev/null || true)
      if [[ \"\${status}\" != RUNNING ]] ||
         { [[ \"\${pid}\" =~ ^[0-9]+\$ ]] &&
           kill -0 \"\${pid}\" 2>/dev/null; }; then
        break
      fi
    done
  fi
  if [[ \"\${status}\" == RUNNING ]] &&
     [[ \"\${pid}\" =~ ^[0-9]+\$ ]] &&
     kill -0 \"\${pid}\" 2>/dev/null; then
    printf 'campaign already running: pid=%s\n' \"\${pid}\"
  elif [[ \"\${status}\" == RUNNING ]]; then
    printf '%s\n' 125 > '${REMOTE_OUT}/campaign.exit_code'
    printf '%s\n' FAILED > '${REMOTE_OUT}/campaign.status'
    printf 'stale RUNNING campaign marked FAILED: pid=%s\n' \"\${pid}\"
  elif [[ \"\${status}\" == COMPLETE || \"\${status}\" == FAILED ]] &&
       [[ -f '${REMOTE_OUT}/campaign.exit_code' ]]; then
    printf 'campaign already terminal: status=%s\n' \"\${status}\"
  else
    rm -f \
      '${REMOTE_OUT}/campaign.pid' \
      '${REMOTE_OUT}/campaign.exit_code'
    printf '%s\n' RUNNING > '${REMOTE_OUT}/campaign.status'
    nohup bash '${REMOTE_OUT}/campaign.sh' </dev/null >/dev/null 2>&1 &
    pid=\$!
    printf '%s\n' \"\${pid}\" > '${REMOTE_OUT}/campaign.pid'
    printf 'campaign started: pid=%s\n' \"\${pid}\"
  fi"
launch_status=$?
remote_status="${launch_status}"

while (( launch_status == 0 )); do
  poll_output=$("${SSH[@]}" "
    status=\$(cat '${REMOTE_OUT}/campaign.status' 2>/dev/null || true)
    pid=\$(cat '${REMOTE_OUT}/campaign.pid' 2>/dev/null || true)
    if [[ \"\${status}\" == RUNNING ]] &&
       [[ \"\${pid}\" =~ ^[0-9]+\$ ]] &&
       ! kill -0 \"\${pid}\" 2>/dev/null; then
      for ((attempt = 0; attempt < ${STALE_RECHECK_ATTEMPTS}; attempt++)); do
        sleep '${STALE_RECHECK_INTERVAL_SECONDS}'
        status=\$(cat '${REMOTE_OUT}/campaign.status' 2>/dev/null || true)
        pid=\$(cat '${REMOTE_OUT}/campaign.pid' 2>/dev/null || true)
        if [[ \"\${status}\" != RUNNING ]] ||
           { [[ \"\${pid}\" =~ ^[0-9]+\$ ]] &&
             kill -0 \"\${pid}\" 2>/dev/null; }; then
          break
        fi
      done
      if [[ \"\${status}\" == RUNNING ]] &&
         { [[ ! \"\${pid}\" =~ ^[0-9]+\$ ]] ||
           ! kill -0 \"\${pid}\" 2>/dev/null; }; then
        printf '%s\n' 125 > '${REMOTE_OUT}/campaign.exit_code'
        printf '%s\n' FAILED > '${REMOTE_OUT}/campaign.status'
        status=FAILED
      fi
    fi
    exit_code=\$(cat '${REMOTE_OUT}/campaign.exit_code' 2>/dev/null || true)
    printf '%s:%s\n' \"\${status:-UNKNOWN}\" \"\${exit_code}\"" 2>&1)
  poll_status=$?
  if (( poll_status != 0 )); then
    printf '%s\n' "${poll_output}" >&2
    printf \
      'remote poll unavailable; campaign %s may still be running. Resume with RUN_TAG=%s\n' \
      "${RUN_TAG}" \
      "${RUN_TAG}" >&2
    remote_status="${poll_status}"
    break
  fi

  campaign_status="${poll_output%%:*}"
  campaign_exit_code="${poll_output#*:}"
  printf 'campaign %s: %s\n' "${RUN_TAG}" "${campaign_status}"
  if [[ "${campaign_status}" == COMPLETE ]]; then
    remote_status="${campaign_exit_code:-0}"
    break
  fi
  if [[ "${campaign_status}" == FAILED ]]; then
    remote_status="${campaign_exit_code:-1}"
    break
  fi
  sleep "${POLL_INTERVAL_SECONDS}"
done
set -e

download_status=0
rsync -a \
  -e "${RSYNC_SSH}" \
  "${REMOTE_HOST}:${REMOTE_OUT}/" \
  "${LOCAL_OUT}/" || download_status=$?

if [[ -f "${LOCAL_OUT}/remote.log" ]]; then
  cat "${LOCAL_OUT}/remote.log"
fi

local_verify_status=0
if [[ -f "${LOCAL_OUT}/result.json" ]]; then
  python3 \
    "${REPO_ROOT}/tools/verify_blockwise_speculative_verifier_gate.py" \
    "${LOCAL_OUT}/result.json" \
    "${REPO_ROOT}" \
    --output "${LOCAL_OUT}/verify.json" \
    || local_verify_status=$?
else
  local_verify_status=1
fi

if (( download_status != 0 )); then
  exit "${download_status}"
fi
if (( remote_status != 0 )); then
  exit "${remote_status}"
fi
exit "${local_verify_status}"
