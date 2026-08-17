#!/usr/bin/env bash

if [ -n "$ZSH_VERSION" ]; then
    exec bash "$0" "$@"
fi

set -e

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --repo-path)
                if [[ -z "${2:-}" ]]; then
                    REPO_PATH=""
                    shift
                    continue
                fi
                REPO_PATH="$2"
                shift 2
                ;;
            *)
                echo "Error: unknown argument: $1"
                exit 1
                ;;
        esac
    done

    REPO_PATH="${REPO_PATH:-}"
}

report_bootstrap_event() {
    local endpoint="https://unittest.byted.org/api/agent/utree-event"

    local user=""
    if [[ -n "${USER_CLOUD_JWT:-}" ]]; then
        local payload encoded_payload payload_len padding decoded_payload
        encoded_payload="${USER_CLOUD_JWT#*.}"
        encoded_payload="${encoded_payload%%.*}"
        payload="$(printf '%s' "$encoded_payload" | tr '_-' '/+')"
        payload_len=${#payload}
        padding=$(( (4 - payload_len % 4) % 4 ))
        if (( padding > 0 )); then
            payload+="$(printf '%*s' "$padding" '' | tr ' ' '=')"
        fi
        decoded_payload="$(printf '%s' "$payload" | base64 -d 2>/dev/null || printf '%s' "$payload" | base64 -D 2>/dev/null || true)"
        user="$(printf '%s' "$decoded_payload" | sed -n 's/.*"username"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)"
    fi
    if [[ -z "$user" ]]; then
        local git_email
        git_email="$(git config user.email 2>/dev/null || true)"
        user="${git_email%@*}"
        user="${user%__dcar}"
    fi
    user="${user:-${USER:-}}"

    local agent="${AGENT_SOURCE:-}"
    if [[ -z "$agent" || "$agent" == "unknown" ]]; then
        agent="${AI_AGENT:-unknown}"
    fi
    agent="$(printf '%s' "$agent" | tr '[:upper:]' '[:lower:]')"

    local model="${MODEL_SOURCE:-unknown}"
    model="$(printf '%s' "$model" | tr '[:upper:]' '[:lower:]')"

    local escaped_user="$user"
    escaped_user="${escaped_user//\\/\\\\}"
    escaped_user="${escaped_user//\"/\\\"}"
    escaped_user="${escaped_user//$'\n'/\\n}"
    escaped_user="${escaped_user//$'\r'/\\r}"
    escaped_user="${escaped_user//$'\t'/\\t}"

    local escaped_agent="$agent"
    escaped_agent="${escaped_agent//\\/\\\\}"
    escaped_agent="${escaped_agent//\"/\\\"}"
    escaped_agent="${escaped_agent//$'\n'/\\n}"
    escaped_agent="${escaped_agent//$'\r'/\\r}"
    escaped_agent="${escaped_agent//$'\t'/\\t}"

    local escaped_model="$model"
    escaped_model="${escaped_model//\\/\\\\}"
    escaped_model="${escaped_model//\"/\\\"}"
    escaped_model="${escaped_model//$'\n'/\\n}"
    escaped_model="${escaped_model//$'\r'/\\r}"
    escaped_model="${escaped_model//$'\t'/\\t}"

    local escaped_exec_source="${EXEC_SOURCE:-}"
    escaped_exec_source="${escaped_exec_source//\\/\\\\}"
    escaped_exec_source="${escaped_exec_source//\"/\\\"}"
    escaped_exec_source="${escaped_exec_source//$'\n'/\\n}"
    escaped_exec_source="${escaped_exec_source//$'\r'/\\r}"
    escaped_exec_source="${escaped_exec_source//$'\t'/\\t}"

    local escaped_exec_session_id="${EXEC_SESSION_ID:-}"
    escaped_exec_session_id="${escaped_exec_session_id//\\/\\\\}"
    escaped_exec_session_id="${escaped_exec_session_id//\"/\\\"}"
    escaped_exec_session_id="${escaped_exec_session_id//$'\n'/\\n}"
    escaped_exec_session_id="${escaped_exec_session_id//$'\r'/\\r}"
    escaped_exec_session_id="${escaped_exec_session_id//$'\t'/\\t}"

    local escaped_ci_task_source="${UT_AGENT_TRIGGER_SOURCE:-}"
    escaped_ci_task_source="${escaped_ci_task_source//\\/\\\\}"
    escaped_ci_task_source="${escaped_ci_task_source//\"/\\\"}"
    escaped_ci_task_source="${escaped_ci_task_source//$'\n'/\\n}"
    escaped_ci_task_source="${escaped_ci_task_source//$'\r'/\\r}"
    escaped_ci_task_source="${escaped_ci_task_source//$'\t'/\\t}"

    local escaped_tmp_root="${TMP_ROOT:-}"
    escaped_tmp_root="${escaped_tmp_root//\\/\\\\}"
    escaped_tmp_root="${escaped_tmp_root//\"/\\\"}"
    escaped_tmp_root="${escaped_tmp_root//$'\n'/\\n}"
    escaped_tmp_root="${escaped_tmp_root//$'\r'/\\r}"
    escaped_tmp_root="${escaped_tmp_root//$'\t'/\\t}"

    local escaped_bits_ut_sid="${BITS_UT_SID:-}"
    escaped_bits_ut_sid="${escaped_bits_ut_sid//\\/\\\\}"
    escaped_bits_ut_sid="${escaped_bits_ut_sid//\"/\\\"}"
    escaped_bits_ut_sid="${escaped_bits_ut_sid//$'\n'/\\n}"
    escaped_bits_ut_sid="${escaped_bits_ut_sid//$'\r'/\\r}"
    escaped_bits_ut_sid="${escaped_bits_ut_sid//$'\t'/\\t}"

    local escaped_pwd="$(pwd)"
    escaped_pwd="${escaped_pwd//\\/\\\\}"
    escaped_pwd="${escaped_pwd//\"/\\\"}"
    escaped_pwd="${escaped_pwd//$'\n'/\\n}"
    escaped_pwd="${escaped_pwd//$'\r'/\\r}"
    escaped_pwd="${escaped_pwd//$'\t'/\\t}"

    local payload_json
    payload_json=$(cat <<EOF
{"event":{"agent_name":"gen_ut_skill","event_name":"gen_ut_bootstrap","code":0,"user":"$escaped_user","attrs":{"agent":"$escaped_agent","model":"$escaped_model","exec_source":"$escaped_exec_source","exec_session_id":"$escaped_exec_session_id","ci_task_source":"$escaped_ci_task_source","tmp_root":"$escaped_tmp_root","bits_ut_sid":"$escaped_bits_ut_sid","pwd":"$escaped_pwd"}}}
EOF
)

    if ! curl -sS --max-time 2 -H 'Content-Type: application/json' -d "$payload_json" "$endpoint" >/dev/null 2>&1; then
        echo "Warning: failed to report gen_ut_bootstrap event; continuing with bootstrap"
    fi
}

ensure_tmp_root() {
    if [[ -z "${TMP_ROOT:-}" ]]; then
        TMP_ROOT=$(mktemp -d)
    fi
    export TMP_ROOT
}

ensure_bits_ut_sid() {
    if [[ -n "${BITS_UT_SID:-}" ]]; then
        export BITS_UT_SID
        return 0
    fi

    if command -v uuidgen >/dev/null 2>&1; then
        BITS_UT_SID="$(uuidgen | tr '[:upper:]' '[:lower:]')"
    elif [[ -r /proc/sys/kernel/random/uuid ]]; then
        BITS_UT_SID="$(cat /proc/sys/kernel/random/uuid)"
    fi

    export BITS_UT_SID
}

run_begin() {
    local target="$1"
    local -a begin_args
    begin_args=(begin)

    if [[ -n "$REPO_PATH" ]]; then
        begin_args+=(--repo-path "$REPO_PATH")
    fi

    if ! AGENT_SOURCE="${AGENT_SOURCE:-unknown}" MODEL_SOURCE="${MODEL_SOURCE:-unknown}" \
        TMP_ROOT="${TMP_ROOT:-}" \
        BITS_UT_SID="${BITS_UT_SID:-}" \
        "$target" "${begin_args[@]}"; then
        echo "Warning: utree begin failed; continuing with bootstrap"
    fi
}

ensure_utree_installed() {
    local target="$HOME/.local/bin/utree"
    local cache_dir

    # 检查目标二进制是否存在且有效
    if [[ -f "$target" && -x "$target" ]]; then
        echo "Info: utree already exists at $target"
        run_begin "$target"
        return 0
    fi

    # 根据系统和架构确定二进制名称
    local binary_name
    if [[ "$OSTYPE" == "darwin"* ]]; then
        binary_name="utree_darwin_arm64"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        binary_name="utree_linux_amd64"
    else
        echo "Error: unsupported OS ($OSTYPE)"
        exit 1
    fi

    echo "Info: fetching latest utree version..."
    local api_url="https://scm.byted.org/api/v2/versions/latest_version/?repo_id=544895&status=build_ok&type=online"
    local tar_url
    local api_response
    api_response=$(curl -s "$api_url")
    if [[ "$OSTYPE" == "darwin"* ]]; then
        tar_url=$(echo "$api_response" | grep -o '"tar_url_aarch64":"[^"]*"' | head -1 | sed 's/"tar_url_aarch64":"//;s/"//')
    else
        tar_url=$(echo "$api_response" | grep -o '"tar_url":"[^"]*"' | head -1 | sed 's/"tar_url":"//;s/"//')
    fi

    if [[ -z "$tar_url" ]]; then
        echo "Error: failed to get tar_url from API response"
        exit 1
    fi

    cache_dir=$(mktemp -d)

    echo "Info: downloading utree from $tar_url..."
    LC_ALL=C curl -sL "$tar_url" | LC_ALL=C tar -xz -C "$cache_dir"

    # 找到下载的二进制并复制到脚本目录，重命名为 utree
    local downloaded_binary="$cache_dir/$binary_name"
    if [[ ! -f "$downloaded_binary" ]]; then
        # 尝试在 cache_dir 的子目录中查找
        downloaded_binary=$(find "$cache_dir" -name "$binary_name" -type f 2>/dev/null | head -1)
    fi

    if [[ -z "$downloaded_binary" || ! -f "$downloaded_binary" ]]; then
        echo "Error: $binary_name not found in downloaded package"
        rm -rf "$cache_dir"
        exit 1
    fi

    mkdir -p "$(dirname "$target")"
    if command -v install >/dev/null 2>&1; then
        install -m 0755 "$downloaded_binary" "$target"
    else
        chmod +x "$downloaded_binary"
        cp -f "$downloaded_binary" "$target"
        chmod 0755 "$target"
    fi
    rm -rf "$cache_dir"

    run_begin "$target"

    echo "Info: utree installed to $target"
}

parse_args "$@"
ensure_tmp_root
ensure_bits_ut_sid
report_bootstrap_event &
ensure_utree_installed &
echo "BITS_TMP_ROOT=${TMP_ROOT}"
