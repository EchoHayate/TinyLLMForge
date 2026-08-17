#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REMOTE="${REMOTE:-sitian@10.232.195.203}"
CONTROL_PATH="${CONTROL_PATH:-}"
REMOTE_DIR="${REMOTE_DIR:-/data00/home/sitian/light-doc-cache-work/probe}"
REMOTE_PY="${REMOTE_PY:-/data00/home/sitian/miniconda3/envs/py311/bin/python}"
MODEL="${MODEL:-/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B}"
TEXT_FILE="${TEXT_FILE:-/data00/home/sitian/light-doc-cache-work/TinyLLMForge/docs/kv-sparse-attention.md}"
LOCAL_TEXT_FILES="${LOCAL_TEXT_FILES:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/data00/home/sitian/light-doc-cache-work/probe/runs/recovery_probe_qwen3_0_6b_s1536_b64}"
GPU="${GPU:-3}"
MAX_TOKENS="${MAX_TOKENS:-1536}"
MAX_SAMPLE_TOKENS="${MAX_SAMPLE_TOKENS:-512}"
SKIP_PREFIX_TOKENS="${SKIP_PREFIX_TOKENS:-128}"
BUDGETS="${BUDGETS:-64}"
SELECTOR="${SELECTOR:-highest}"
TRAIN_FRAC="${TRAIN_FRAC:-0.6}"
RIDGE="${RIDGE:-1.0}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
MAX_HEADS="${MAX_HEADS:-0}"
START_LAYER="${START_LAYER:-0}"
END_LAYER="${END_LAYER:-0}"
ACCEPT_R2="${ACCEPT_R2:-0.50}"
RECOVERY_ARGS="${RECOVERY_ARGS:-}"
SRC="${SRC:-$SCRIPT_DIR/train_recovery_probe.py}"
DEP_SRC="${DEP_SRC:-$SCRIPT_DIR/probe_am_compact_cache.py}"
LOCAL_OUT_DIR="${LOCAL_OUT_DIR:-$SCRIPT_DIR/recovery_probe_remote_latest}"
SSH_OPTS="${SSH_OPTS:-}"
SSH_RETRIES="${SSH_RETRIES:-3}"

if [[ ! -f "$SRC" ]]; then
  echo "missing local recovery script: $SRC" >&2
  exit 1
fi
if [[ ! -f "$DEP_SRC" ]]; then
  echo "missing local dependency script: $DEP_SRC" >&2
  exit 1
fi
mkdir -p "$LOCAL_OUT_DIR"

SSH_CMD=(ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o UpdateHostKeys=no)
if [[ -n "$SSH_OPTS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_SSH_OPTS=($SSH_OPTS)
  SSH_CMD+=("${EXTRA_SSH_OPTS[@]}")
fi
if [[ -n "$CONTROL_PATH" ]]; then
  if [[ ! -S "$CONTROL_PATH" ]]; then
    cat >&2 <<EOF
missing SSH ControlMaster socket: $CONTROL_PATH

Either omit CONTROL_PATH and run this script from a normal Terminal, or create it first:

KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \\
ssh -MNf \\
  -o ControlMaster=yes \\
  -o ControlPath=$CONTROL_PATH \\
  -o ControlPersist=2h \\
  $REMOTE
EOF
    exit 2
  fi
  SSH_CMD=(ssh -n -S "$CONTROL_PATH" -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o UpdateHostKeys=no)
  if [[ -n "$SSH_OPTS" ]]; then
    SSH_CMD+=("${EXTRA_SSH_OPTS[@]}")
  fi
fi

run_ssh_with_retry() {
  local attempt=1
  while true; do
    if "$@"; then
      return 0
    fi
    if (( attempt < SSH_RETRIES )); then
      echo "ssh attempt $attempt/$SSH_RETRIES failed; retrying..." >&2
      sleep "$attempt"
      attempt=$((attempt + 1))
      continue
    fi
    return 1
  done
}

remote_sh() {
  run_ssh_with_retry "${SSH_CMD[@]}" "$REMOTE" "$1"
}

remote_sh_stdin() {
  local cmd="$1"
  local ssh_cmd_stdin=()
  local opt
  for opt in "${SSH_CMD[@]}"; do
    if [[ "$opt" != "-n" ]]; then
      ssh_cmd_stdin+=("$opt")
    fi
  done
  run_ssh_with_retry "${ssh_cmd_stdin[@]}" "$REMOTE" "$cmd"
}

local_size="$(wc -c < "$SRC" | tr -d ' ')"
local_sha="$(shasum -a 256 "$SRC" | awk '{print $1}')"
dep_size="$(wc -c < "$DEP_SRC" | tr -d ' ')"
dep_sha="$(shasum -a 256 "$DEP_SRC" | awk '{print $1}')"

remote_text_args=()
remote_verify_text=""
if [[ -n "$LOCAL_TEXT_FILES" ]]; then
  index=0
  for local_text_file in $LOCAL_TEXT_FILES; do
    if [[ ! -f "$local_text_file" ]]; then
      echo "missing local text file: $local_text_file" >&2
      exit 1
    fi
    remote_name="recovery_text_${index}.md"
    text_size="$(wc -c < "$local_text_file" | tr -d ' ')"
    text_sha="$(shasum -a 256 "$local_text_file" | awk '{print $1}')"
    echo "local text[$index]: $local_text_file"
    echo "local text[$index] size: $text_size"
    echo "local text[$index] sha256: $text_sha"
    base64 < "$local_text_file" | remote_sh_stdin "mkdir -p '$REMOTE_DIR' && cd '$REMOTE_DIR' && \
      cat > '$remote_name.b64.incoming' && \
      base64 -d '$remote_name.b64.incoming' > '$remote_name'"
    remote_text_args+=("--text-file '$REMOTE_DIR/$remote_name'")
    remote_verify_text+=" && chmod 0644 '$remote_name' && test \"\$(wc -c < '$remote_name' | tr -d ' ')\" = '$text_size' && test \"\$(sha256sum '$remote_name' | awk '{print \$1}')\" = '$text_sha'"
    index=$((index + 1))
  done
else
  remote_text_args=("--text-file '$TEXT_FILE'")
fi

echo "remote: $REMOTE"
if [[ -n "$CONTROL_PATH" ]]; then
  echo "control path: $CONTROL_PATH"
else
  echo "control path: disabled; using direct ssh config"
fi
echo "local script: $SRC"
echo "local size: $local_size"
echo "local sha256: $local_sha"
echo "dependency script: $DEP_SRC"
echo "dependency size: $dep_size"
echo "dependency sha256: $dep_sha"
echo "local output mirror: $LOCAL_OUT_DIR"

remote_sh "mkdir -p '$REMOTE_DIR' '$OUTPUT_DIR'"
base64 < "$SRC" | remote_sh_stdin "cd '$REMOTE_DIR' && \
  cat > train_recovery_probe.py.b64.incoming && \
  base64 -d train_recovery_probe.py.b64.incoming > train_recovery_probe.py"
base64 < "$DEP_SRC" | remote_sh_stdin "cd '$REMOTE_DIR' && \
  cat > probe_am_compact_cache.py.b64.incoming && \
  base64 -d probe_am_compact_cache.py.b64.incoming > probe_am_compact_cache.py"

remote_sh "cd '$REMOTE_DIR' && \
  rm train_recovery_probe.py.b64.incoming probe_am_compact_cache.py.b64.incoming recovery_text_*.md.b64.incoming 2>/dev/null || true && \
  chmod 0644 train_recovery_probe.py && \
  chmod 0644 probe_am_compact_cache.py && \
  test \"\$(wc -c < train_recovery_probe.py | tr -d ' ')\" = '$local_size' && \
  test \"\$(sha256sum train_recovery_probe.py | awk '{print \$1}')\" = '$local_sha' && \
  test \"\$(wc -c < probe_am_compact_cache.py | tr -d ' ')\" = '$dep_size' && \
  test \"\$(sha256sum probe_am_compact_cache.py | awk '{print \$1}')\" = '$dep_sha' \
  $remote_verify_text && \
  '$REMOTE_PY' -m py_compile train_recovery_probe.py probe_am_compact_cache.py"

echo "remote transfer verified"

text_args_joined="${remote_text_args[*]}"
remote_cmd="cd '$REMOTE_DIR' && \
  mkdir -p '$OUTPUT_DIR' && \
  { \
    date; \
    echo 'running trainable recovery probe'; \
    CUDA_VISIBLE_DEVICES='$GPU' '$REMOTE_PY' train_recovery_probe.py \
      --model '$MODEL' \
      $text_args_joined \
      --output-dir '$OUTPUT_DIR' \
      --max-tokens '$MAX_TOKENS' \
      --max-sample-tokens '$MAX_SAMPLE_TOKENS' \
      --skip-prefix-tokens '$SKIP_PREFIX_TOKENS' \
      --budgets '$BUDGETS' \
      --selector '$SELECTOR' \
      --train-frac '$TRAIN_FRAC' \
      --ridge '$RIDGE' \
      --hidden-dim '$HIDDEN_DIM' \
      --epochs '$EPOCHS' \
      --lr '$LR' \
      --weight-decay '$WEIGHT_DECAY' \
      --max-heads '$MAX_HEADS' \
      --start-layer '$START_LAYER' \
      --end-layer '$END_LAYER' \
      --accept-r2 '$ACCEPT_R2' \
      $RECOVERY_ARGS; \
    date; \
  } 2>&1 | tee '$OUTPUT_DIR/run.log'"

remote_sh "$remote_cmd"

echo
echo "remote report:"
remote_sh "cat '$OUTPUT_DIR/report.md'" | tee "$LOCAL_OUT_DIR/report.md"
echo
echo "remote summary json:"
remote_sh "cat '$OUTPUT_DIR/summary.json'" | tee "$LOCAL_OUT_DIR/summary.json"
remote_sh "cat '$OUTPUT_DIR/recovery_head_rows.csv'" > "$LOCAL_OUT_DIR/recovery_head_rows.csv"
remote_sh "cat '$OUTPUT_DIR/run.log'" > "$LOCAL_OUT_DIR/run.log"
echo
echo "local mirrored report: $LOCAL_OUT_DIR/report.md"
echo "local mirrored summary: $LOCAL_OUT_DIR/summary.json"
echo "local mirrored rows: $LOCAL_OUT_DIR/recovery_head_rows.csv"
echo "local mirrored run log: $LOCAL_OUT_DIR/run.log"
