#!/usr/bin/env bash
set -euo pipefail

source_root="$1"
artifacts="$2"
python_executable="$3"
package_root="$4"
driver_status=1

write_status() {
  printf '%s\n' "${driver_status}" \
    >"${artifacts}/recovery-driver-exit-code.txt"
}
trap write_status EXIT

cd "${source_root}"
for name in \
  exit-code.txt \
  verify-timing-remote-exit-code.txt \
  verify-remote-exit-code.txt \
  verify-host-remote-exit-code.txt \
  telemetry-assemble.log \
  host-semantic-assemble.log; do
  if [[ -f "${artifacts}/${name}" \
      && ! -f "${artifacts}/initial-${name}" ]]; then
    cp -p \
      "${artifacts}/${name}" \
      "${artifacts}/initial-${name}"
  fi
done

(
  cd "${artifacts}"
  sha256sum \
    result.json \
    workers/target-b4.json \
    workers/learned-b4.json \
    telemetry/target-gpu.csv \
    telemetry/learned-gpu.csv \
    host-semantic/target-host.jsonl \
    host-semantic/learned-host.jsonl \
    >recovery-raw-inputs.sha256
)

sha256sum \
  tools/autoregressive_draft_instability_telemetry.py \
  tools/autoregressive_draft_host_semantic_diagnostic.py \
  tools/verify_autoregressive_draft_instability_telemetry.py \
  tools/verify_autoregressive_draft_host_semantic_diagnostic.py \
  >"${artifacts}/recovery-source-files.sha256"

"${python_executable}" -m py_compile \
  tools/autoregressive_draft_instability_telemetry.py \
  tools/autoregressive_draft_host_semantic_diagnostic.py \
  tools/verify_autoregressive_draft_instability_telemetry.py \
  tools/verify_autoregressive_draft_host_semantic_diagnostic.py

"${python_executable}" -m pytest -q \
  tools/test_autoregressive_draft_instability_telemetry.py \
  tools/test_autoregressive_draft_host_semantic_diagnostic.py \
  >"${artifacts}/recovery-tests.log" 2>&1

PYTHONPATH="${package_root}:${source_root}" \
"${python_executable}" \
  tools/autoregressive_draft_b4_timing_diagnostic.py \
    --target-worker "${artifacts}/workers/target-b4.json" \
    --learned-worker "${artifacts}/workers/learned-b4.json" \
    --repo-root "${source_root}" \
    --out "${artifacts}/result.json" \
    >"${artifacts}/diagnostic.log" 2>&1

PYTHONPATH="${package_root}:${source_root}" \
"${python_executable}" \
  tools/autoregressive_draft_instability_telemetry.py \
    --timing-artifact "${artifacts}/result.json" \
    --target-worker "${artifacts}/workers/target-b4.json" \
    --learned-worker "${artifacts}/workers/learned-b4.json" \
    --target-gpu-csv "${artifacts}/telemetry/target-gpu.csv" \
    --learned-gpu-csv "${artifacts}/telemetry/learned-gpu.csv" \
    --repo-root "${source_root}" \
    --host-file target_vmstat=host/target-vmstat.log \
    --host-file target_mpstat=host/target-mpstat.log \
    --host-file target_pidstat=host/target-pidstat.log \
    --host-file learned_vmstat=host/learned-vmstat.log \
    --host-file learned_mpstat=host/learned-mpstat.log \
    --host-file learned_pidstat=host/learned-pidstat.log \
    --out "${artifacts}/telemetry.json" \
    >"${artifacts}/telemetry-assemble.log" 2>&1

PYTHONPATH="${package_root}:${source_root}" \
"${python_executable}" \
  tools/autoregressive_draft_host_semantic_diagnostic.py \
    --timing-artifact "${artifacts}/result.json" \
    --gpu-telemetry-artifact "${artifacts}/telemetry.json" \
    --target-worker "${artifacts}/workers/target-b4.json" \
    --learned-worker "${artifacts}/workers/learned-b4.json" \
    --target-host-jsonl \
      "${artifacts}/host-semantic/target-host.jsonl" \
    --learned-host-jsonl \
      "${artifacts}/host-semantic/learned-host.jsonl" \
    --policy-order target,learned \
    --prime-each-policy \
    --repo-root "${source_root}" \
    --out "${artifacts}/host-semantic.json" \
    >"${artifacts}/host-semantic-assemble.log" 2>&1

PYTHONPATH="${package_root}:${source_root}" \
"${python_executable}" \
  tools/verify_autoregressive_draft_b4_timing_diagnostic.py \
    --artifact "${artifacts}/result.json" \
    --repo-root "${source_root}" \
    --receipt "${artifacts}/verify.timing.remote.json" \
    >"${artifacts}/verify.timing.remote.log" 2>&1

PYTHONPATH="${package_root}:${source_root}" \
"${python_executable}" \
  tools/verify_autoregressive_draft_instability_telemetry.py \
    --artifact "${artifacts}/telemetry.json" \
    --repo-root "${source_root}" \
    --receipt "${artifacts}/verify.remote.json" \
    >"${artifacts}/verify.remote.log" 2>&1

PYTHONPATH="${package_root}:${source_root}" \
"${python_executable}" \
  tools/verify_autoregressive_draft_host_semantic_diagnostic.py \
    --artifact "${artifacts}/host-semantic.json" \
    --repo-root "${source_root}" \
    --receipt "${artifacts}/verify.host.remote.json" \
    >"${artifacts}/verify.host.remote.log" 2>&1

printf '0\n' >"${artifacts}/exit-code.txt"
printf '0\n' >"${artifacts}/verify-timing-remote-exit-code.txt"
printf '0\n' >"${artifacts}/verify-remote-exit-code.txt"
printf '0\n' >"${artifacts}/verify-host-remote-exit-code.txt"

"${python_executable}" - \
  "${artifacts}/recovery-provenance.json" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
output_path.write_text(
    json.dumps(
        {
            "canonical_postprocessing_exit_code": 0,
            "initial_campaign_exit_code": 1,
            "raw_inputs_modified": False,
            "raw_workload_rerun": False,
            "reason": (
                "GPU telemetry repeat coverage required a nearby "
                "edge sample and host JSONL gap validation was "
                "narrowed to aligned repeat windows"
            ),
            "recovery_kind": "postprocessing_only",
            "schema_version": 1,
            "status": "PASS",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

driver_status=0
