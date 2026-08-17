#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FROZEN_RUNNER="${REPO_ROOT}/tools/run_qwen35_generic_speculative_tp4_gate_remote.sh"
export QWEN35_TP4_16K_REPO_ROOT="${REPO_ROOT}"
export KRB5CCNAME="${KRB5CCNAME:-FILE:/Users/bytedance/krb5cc_sitian}"
export REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
export REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/sitian-workspace01/tllm/env/bin/python}"
export MODEL_PATH="${MODEL_PATH:-/data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model}"

generated_runner="$(mktemp "${TMPDIR:-/tmp}/qwen35-tp4-16k-runner.XXXXXX")"
trap 'rm -f "${generated_runner}"' EXIT

python3 - "${FROZEN_RUNNER}" "${generated_runner}" <<'PY'
from pathlib import Path
import sys


source_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
text = source_path.read_text(encoding="utf-8")
text = text.replace(
    'REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"',
    'REPO_ROOT="${QWEN35_TP4_16K_REPO_ROOT:?}"',
)
text = text.replace(
    "qwen35_generic_speculative_tp4",
    "qwen35_generic_speculative_tp4_16k",
)
text = text.replace(
    "qwen35-generic-speculative-tp4-runs",
    "qwen35-generic-speculative-tp4-16k-runs",
)
authority_sources = (
    "    tools/qwen35_generic_speculative_tp4_16k_gate.py \\\n"
    "    tools/qwen35_generic_speculative_tp4_16k_worker.py \\\n"
    "    tools/verify_qwen35_generic_speculative_tp4_16k_gate.py\n"
)
bound_sources = (
    "    tools/qwen35_generic_speculative_tp4_gate.py \\\n"
    "    tools/qwen35_generic_speculative_tp4_worker.py \\\n"
    "    tools/verify_qwen35_generic_speculative_tp4_gate.py \\\n"
    + authority_sources
)
if authority_sources not in text:
    raise SystemExit(
        "frozen runner source archive contract changed"
    )
text = text.replace(
    authority_sources,
    bound_sources,
    1,
)
required_fragments = (
    "sitian@10.232.195.203",
    "FILE:/Users/bytedance/krb5cc_sitian",
    "ControlMaster=no",
    "ControlPath=none",
    "qwen35_generic_speculative_tp4_16k_gate.py",
    "qwen35_generic_speculative_tp4_16k_worker.py",
    "verify_qwen35_generic_speculative_tp4_16k_gate.py",
    "qwen35_generic_speculative_tp4_gate.py",
    "qwen35_generic_speculative_tp4_worker.py",
    "verify_qwen35_generic_speculative_tp4_gate.py",
    "campaign.status",
    "campaign.pid",
    "campaign.exit_code",
    "authority.failed",
    "REMOTE_COMMAND_RETRY_ATTEMPTS",
    "REMOTE_RSYNC_RETRY_ATTEMPTS",
    "POLL_INTERVAL_SECONDS",
    "head -n 4",
    "campaign already terminal",
    "campaign already running",
)
missing = [
    fragment
    for fragment in required_fragments
    if fragment not in text
]
if missing:
    raise SystemExit(
        "derived 16K runner contract is incomplete: "
        + ", ".join(missing)
    )
if "ControlMaster=" + "yes" in text:
    raise SystemExit(
        "derived 16K runner enables persistent SSH control"
    )
output_path.write_text(text, encoding="utf-8")
PY

bash "${generated_runner}" "$@"
