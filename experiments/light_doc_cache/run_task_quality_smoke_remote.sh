#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REMOTE="${REMOTE:-sitian@10.232.195.203}"
CONTROL_PATH="${CONTROL_PATH:-}"
REMOTE_DIR="${REMOTE_DIR:-/data00/home/sitian/light-doc-cache-work/probe}"
REMOTE_PY="${REMOTE_PY:-/data00/home/sitian/miniconda3/envs/py311/bin/python}"
MODEL="${MODEL:-/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B}"
TEXT_FILE="${TEXT_FILE:-/data00/home/sitian/light-doc-cache-work/TinyLLMForge/docs/kv-sparse-attention.md}"
LOCAL_TEXT_FILE="${LOCAL_TEXT_FILE:-}"
TASK_FILE="${TASK_FILE:-}"
POLICY_DIR="${POLICY_DIR:-/data00/home/sitian/light-doc-cache-work/probe/runs/policy_am_qwen3_0_6b_s1536_holdout_all_r1.0}"
LOCAL_POLICY_DIR="${LOCAL_POLICY_DIR:-}"
LOCAL_ADAPTIVE_POLICY_FILE="${LOCAL_ADAPTIVE_POLICY_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/data00/home/sitian/light-doc-cache-work/probe/runs/task_quality_smoke_qwen3_0_6b_s1536_policy035_050}"
GPU="${GPU:-3}"
THRESHOLDS="${THRESHOLDS:-0.35,0.50}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1536}"
MAX_DOC_TOKENS="${MAX_DOC_TOKENS:-1200}"
BANK_TRAIN_TOKENS="${BANK_TRAIN_TOKENS:-512}"
RIDGE="${RIDGE:-1.0}"
TASK_QUALITY_ARGS="${TASK_QUALITY_ARGS:-}"
CHUNK_SIZE="${CHUNK_SIZE:-600}"
SRC="${SRC:-$SCRIPT_DIR/task_quality_smoke.py}"
DEP_SRC="${DEP_SRC:-$SCRIPT_DIR/probe_am_compact_cache.py}"
RECOVERY_DEP_SRC="${RECOVERY_DEP_SRC:-$SCRIPT_DIR/train_recovery_probe.py}"
LOCAL_OUT_DIR="${LOCAL_OUT_DIR:-$SCRIPT_DIR/task_quality_smoke_remote_latest}"
SSH_OPTS="${SSH_OPTS:-}"
SSH_RETRIES="${SSH_RETRIES:-3}"

if [[ ! -f "$SRC" ]]; then
  echo "missing local smoke script: $SRC" >&2
  exit 1
fi
if [[ ! -f "$DEP_SRC" ]]; then
  echo "missing local dependency script: $DEP_SRC" >&2
  exit 1
fi
if [[ ! -f "$RECOVERY_DEP_SRC" ]]; then
  echo "missing local recovery dependency script: $RECOVERY_DEP_SRC" >&2
  exit 1
fi
if [[ -n "$TASK_FILE" && ! -f "$TASK_FILE" ]]; then
  echo "missing local task file: $TASK_FILE" >&2
  exit 1
fi
if [[ -n "$LOCAL_POLICY_DIR" && ! -f "$LOCAL_POLICY_DIR/policy_rows.csv" ]]; then
  echo "missing local policy rows: $LOCAL_POLICY_DIR/policy_rows.csv" >&2
  exit 1
fi
if [[ -n "$LOCAL_ADAPTIVE_POLICY_FILE" && ! -f "$LOCAL_ADAPTIVE_POLICY_FILE" ]]; then
  echo "missing local adaptive policy file: $LOCAL_ADAPTIVE_POLICY_FILE" >&2
  exit 1
fi
if [[ -n "$LOCAL_TEXT_FILE" && ! -f "$LOCAL_TEXT_FILE" ]]; then
  echo "missing local text file: $LOCAL_TEXT_FILE" >&2
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
recovery_dep_size="$(wc -c < "$RECOVERY_DEP_SRC" | tr -d ' ')"
recovery_dep_sha="$(shasum -a 256 "$RECOVERY_DEP_SRC" | awk '{print $1}')"
task_file_size=""
task_file_sha=""
remote_task_arg=""
remote_policy_dir="$POLICY_DIR"
policy_rows_size=""
policy_rows_sha=""
adaptive_policy_size=""
adaptive_policy_sha=""
text_file_size=""
text_file_sha=""
remote_text_arg="$TEXT_FILE"
if [[ -n "$TASK_FILE" ]]; then
  task_file_size="$(wc -c < "$TASK_FILE" | tr -d ' ')"
  task_file_sha="$(shasum -a 256 "$TASK_FILE" | awk '{print $1}')"
  remote_task_arg="--task-file '$REMOTE_DIR/task_quality_tasks.json'"
fi
if [[ -n "$LOCAL_POLICY_DIR" ]]; then
  policy_rows_size="$(wc -c < "$LOCAL_POLICY_DIR/policy_rows.csv" | tr -d ' ')"
  policy_rows_sha="$(shasum -a 256 "$LOCAL_POLICY_DIR/policy_rows.csv" | awk '{print $1}')"
  remote_policy_dir="$REMOTE_DIR/task_quality_policy"
fi
if [[ -n "$LOCAL_ADAPTIVE_POLICY_FILE" ]]; then
  adaptive_policy_size="$(wc -c < "$LOCAL_ADAPTIVE_POLICY_FILE" | tr -d ' ')"
  adaptive_policy_sha="$(shasum -a 256 "$LOCAL_ADAPTIVE_POLICY_FILE" | awk '{print $1}')"
  TASK_QUALITY_ARGS="$TASK_QUALITY_ARGS --adaptive-policy-file '$REMOTE_DIR/adaptive_policy.json'"
fi
if [[ -n "$LOCAL_TEXT_FILE" ]]; then
  text_file_size="$(wc -c < "$LOCAL_TEXT_FILE" | tr -d ' ')"
  text_file_sha="$(shasum -a 256 "$LOCAL_TEXT_FILE" | awk '{print $1}')"
  remote_text_arg="$REMOTE_DIR/task_quality_text.md"
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
echo "recovery dependency script: $RECOVERY_DEP_SRC"
echo "recovery dependency size: $recovery_dep_size"
echo "recovery dependency sha256: $recovery_dep_sha"
if [[ -n "$TASK_FILE" ]]; then
  echo "task file: $TASK_FILE"
  echo "task file size: $task_file_size"
  echo "task file sha256: $task_file_sha"
fi
if [[ -n "$LOCAL_POLICY_DIR" ]]; then
  echo "local policy dir: $LOCAL_POLICY_DIR"
  echo "local policy_rows size: $policy_rows_size"
  echo "local policy_rows sha256: $policy_rows_sha"
fi
if [[ -n "$LOCAL_ADAPTIVE_POLICY_FILE" ]]; then
  echo "local adaptive policy file: $LOCAL_ADAPTIVE_POLICY_FILE"
  echo "local adaptive_policy size: $adaptive_policy_size"
  echo "local adaptive_policy sha256: $adaptive_policy_sha"
fi
if [[ -n "$LOCAL_TEXT_FILE" ]]; then
  echo "local text file: $LOCAL_TEXT_FILE"
  echo "local text file size: $text_file_size"
  echo "local text file sha256: $text_file_sha"
fi
if [[ -n "$TASK_QUALITY_ARGS" ]]; then
  echo "extra task args: $TASK_QUALITY_ARGS"
fi
echo "transfer: single ssh stdin stream"
echo "local output mirror: $LOCAL_OUT_DIR"

remote_sh "mkdir -p '$REMOTE_DIR' '$OUTPUT_DIR'"
base64 < "$SRC" | remote_sh_stdin "cd '$REMOTE_DIR' && \
  cat > task_quality_smoke.py.b64.incoming && \
  base64 -d task_quality_smoke.py.b64.incoming > task_quality_smoke.py"
base64 < "$DEP_SRC" | remote_sh_stdin "cd '$REMOTE_DIR' && \
  cat > probe_am_compact_cache.py.b64.incoming && \
  base64 -d probe_am_compact_cache.py.b64.incoming > probe_am_compact_cache.py"
base64 < "$RECOVERY_DEP_SRC" | remote_sh_stdin "cd '$REMOTE_DIR' && \
  cat > train_recovery_probe.py.b64.incoming && \
  base64 -d train_recovery_probe.py.b64.incoming > train_recovery_probe.py"
if [[ -n "$TASK_FILE" ]]; then
  base64 < "$TASK_FILE" | remote_sh_stdin "cd '$REMOTE_DIR' && \
    cat > task_quality_tasks.json.b64.incoming && \
    base64 -d task_quality_tasks.json.b64.incoming > task_quality_tasks.json"
fi
if [[ -n "$LOCAL_POLICY_DIR" ]]; then
  base64 < "$LOCAL_POLICY_DIR/policy_rows.csv" | remote_sh_stdin "cd '$REMOTE_DIR' && \
    mkdir -p task_quality_policy && \
    cat > task_quality_policy/policy_rows.csv.b64.incoming && \
    base64 -d task_quality_policy/policy_rows.csv.b64.incoming > task_quality_policy/policy_rows.csv"
fi
if [[ -n "$LOCAL_ADAPTIVE_POLICY_FILE" ]]; then
  base64 < "$LOCAL_ADAPTIVE_POLICY_FILE" | remote_sh_stdin "cd '$REMOTE_DIR' && \
    cat > adaptive_policy.json.b64.incoming && \
    base64 -d adaptive_policy.json.b64.incoming > adaptive_policy.json"
fi
if [[ -n "$LOCAL_TEXT_FILE" ]]; then
  base64 < "$LOCAL_TEXT_FILE" | remote_sh_stdin "cd '$REMOTE_DIR' && \
    cat > task_quality_text.md.b64.incoming && \
    base64 -d task_quality_text.md.b64.incoming > task_quality_text.md"
fi

remote_sh "cd '$REMOTE_DIR' && \
  rm task_quality_smoke.py.b64.incoming probe_am_compact_cache.py.b64.incoming task_quality_tasks.json.b64.incoming task_quality_text.md.b64.incoming 2>/dev/null || true && \
  rm train_recovery_probe.py.b64.incoming 2>/dev/null || true && \
  rm adaptive_policy.json.b64.incoming 2>/dev/null || true && \
  chmod 0644 task_quality_smoke.py && \
  chmod 0644 probe_am_compact_cache.py && \
  chmod 0644 train_recovery_probe.py && \
  test \"\$(wc -c < task_quality_smoke.py | tr -d ' ')\" = '$local_size' && \
  test \"\$(sha256sum task_quality_smoke.py | awk '{print \$1}')\" = '$local_sha' && \
  test \"\$(wc -c < probe_am_compact_cache.py | tr -d ' ')\" = '$dep_size' && \
  test \"\$(sha256sum probe_am_compact_cache.py | awk '{print \$1}')\" = '$dep_sha' && \
  test \"\$(wc -c < train_recovery_probe.py | tr -d ' ')\" = '$recovery_dep_size' && \
  test \"\$(sha256sum train_recovery_probe.py | awk '{print \$1}')\" = '$recovery_dep_sha' && \
  if [ -n '$TASK_FILE' ]; then \
    chmod 0644 task_quality_tasks.json && \
    test \"\$(wc -c < task_quality_tasks.json | tr -d ' ')\" = '$task_file_size' && \
    test \"\$(sha256sum task_quality_tasks.json | awk '{print \$1}')\" = '$task_file_sha'; \
  fi && \
  if [ -n '$LOCAL_POLICY_DIR' ]; then \
    chmod 0644 task_quality_policy/policy_rows.csv && \
    test \"\$(wc -c < task_quality_policy/policy_rows.csv | tr -d ' ')\" = '$policy_rows_size' && \
    test \"\$(sha256sum task_quality_policy/policy_rows.csv | awk '{print \$1}')\" = '$policy_rows_sha'; \
  fi && \
  if [ -n '$LOCAL_ADAPTIVE_POLICY_FILE' ]; then \
    chmod 0644 adaptive_policy.json && \
    test \"\$(wc -c < adaptive_policy.json | tr -d ' ')\" = '$adaptive_policy_size' && \
    test \"\$(sha256sum adaptive_policy.json | awk '{print \$1}')\" = '$adaptive_policy_sha'; \
  fi && \
  if [ -n '$LOCAL_TEXT_FILE' ]; then \
    chmod 0644 task_quality_text.md && \
    test \"\$(wc -c < task_quality_text.md | tr -d ' ')\" = '$text_file_size' && \
    test \"\$(sha256sum task_quality_text.md | awk '{print \$1}')\" = '$text_file_sha'; \
  fi && \
  '$REMOTE_PY' -m py_compile task_quality_smoke.py probe_am_compact_cache.py train_recovery_probe.py"

echo "remote transfer verified"

remote_cmd="cd '$REMOTE_DIR' && \
  mkdir -p '$OUTPUT_DIR' && \
  { \
    date; \
    echo 'running task quality smoke'; \
    CUDA_VISIBLE_DEVICES='$GPU' '$REMOTE_PY' task_quality_smoke.py \
      --model '$MODEL' \
      --text-file '$remote_text_arg' \
      $remote_task_arg \
      --policy-dir '$remote_policy_dir' \
      --output-dir '$OUTPUT_DIR' \
      --thresholds '$THRESHOLDS' \
      --max-prompt-tokens '$MAX_PROMPT_TOKENS' \
      --max-doc-tokens '$MAX_DOC_TOKENS' \
      --bank-train-tokens '$BANK_TRAIN_TOKENS' \
      --ridge '$RIDGE' \
      $TASK_QUALITY_ARGS; \
    date; \
  } 2>&1 | tee '$OUTPUT_DIR/run.log'"

remote_sh "$remote_cmd"

echo
echo "remote report:"
remote_sh "cat '$OUTPUT_DIR/report.md'" | tee "$LOCAL_OUT_DIR/report.md"
echo
echo "remote summary json:"
remote_sh "cat '$OUTPUT_DIR/summary.json'" | tee "$LOCAL_OUT_DIR/summary.json"
remote_sh "cat '$OUTPUT_DIR/task_rows.csv'" > "$LOCAL_OUT_DIR/task_rows.csv"
remote_sh "cat '$OUTPUT_DIR/tasks.json'" > "$LOCAL_OUT_DIR/tasks.json"
echo
echo "local mirrored report: $LOCAL_OUT_DIR/report.md"
echo "local mirrored summary: $LOCAL_OUT_DIR/summary.json"
echo "local mirrored task rows: $LOCAL_OUT_DIR/task_rows.csv"
echo "local mirrored tasks: $LOCAL_OUT_DIR/tasks.json"
