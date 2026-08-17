from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TELEMETRY_PATH = (
    ROOT / "tools" / "autoregressive_draft_instability_telemetry.py"
)
VERIFIER_PATH = (
    ROOT
    / "tools"
    / "verify_autoregressive_draft_instability_telemetry.py"
)
RUNNER_PATH = (
    ROOT
    / "tools"
    / "run_autoregressive_draft_b4_instability_telemetry_remote.sh"
)
LEARNED_AA_RUNNER_PATH = (
    ROOT
    / "tools"
    / "run_autoregressive_draft_learned_aa_remote.sh"
)
PAIRED_STABILITY_RUNNER_PATH = (
    ROOT
    / "tools"
    / "run_autoregressive_draft_paired_stability_remote.sh"
)


def _load_module():
    assert TELEMETRY_PATH.exists(), (
        f"missing module: {TELEMETRY_PATH}"
    )
    spec = importlib.util.spec_from_file_location(
        "autoregressive_draft_instability_telemetry_test_module",
        TELEMETRY_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_verifier():
    assert VERIFIER_PATH.exists(), (
        f"missing verifier: {VERIFIER_PATH}"
    )
    telemetry = _load_module()
    sys.modules[
        "autoregressive_draft_instability_telemetry"
    ] = telemetry
    spec = importlib.util.spec_from_file_location(
        "verify_autoregressive_draft_instability_telemetry_test_module",
        VERIFIER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _csv_row(
    *,
    sampled_at_unix_ns=1_786_808_823_101_000_000,
    gpu_index=3,
    sm_clock_mhz=1410,
    throttle_mask="0x0000000000000001",
):
    return (
        f"{sampled_at_unix_ns}, "
        "2026/08/15 23:47:03.101, "
        f"{gpu_index}, "
        f"GPU-{gpu_index:032d}, "
        "P0, "
        f"{sm_clock_mhz}, "
        "1512, "
        "70.38, "
        "41, "
        "93, "
        "12, "
        "72455, "
        f"{throttle_mask}"
    )


def _worker():
    return {
        "warmup_runs": [
            {
                "repeat": -1,
                "campaign_interval": {
                    "started_at_unix_ns": 100,
                    "finished_at_unix_ns": 150,
                },
            }
        ],
        "measured_runs": [
            {
                "repeat": 0,
                "campaign_interval": {
                    "started_at_unix_ns": 200,
                    "finished_at_unix_ns": 300,
                },
            }
        ],
    }


def _samples():
    rows = []
    timestamps = (200, 220, 240, 260, 300)
    for gpu_index in (3, 4, 6, 7):
        for offset, timestamp in enumerate(timestamps):
            rows.append({
                "sampled_at_unix_ns": timestamp,
                "nvidia_timestamp": (
                    "2026/08/15 23:47:03.101"
                ),
                "gpu_index": gpu_index,
                "uuid": f"GPU-{gpu_index:032d}",
                "pstate": "P0",
                "sm_clock_mhz": 1000 + offset * 100,
                "memory_clock_mhz": 1512,
                "power_w": 60.0 + offset,
                "temperature_c": 40 + offset,
                "gpu_utilization_percent": 80 + offset,
                "memory_utilization_percent": 10 + offset,
                "memory_used_mib": 72_000 + offset,
                "throttle_reasons_active": (
                    1 if offset == 4 else 0
                ),
            })
    return rows


def _gpu_csv():
    rows = []
    for gpu_index in (3, 4, 6, 7):
        for timestamp in (200, 220, 240, 260, 300):
            rows.append(
                _csv_row(
                    sampled_at_unix_ns=timestamp,
                    gpu_index=gpu_index,
                    sm_clock_mhz=1410,
                    throttle_mask="0x0",
                )
            )
    return "\n".join(rows) + "\n"


def _artifact_kwargs(*, classification="UNSTABLE"):
    target_samples = copy.deepcopy(_samples())
    learned_samples = copy.deepcopy(_samples())
    for row in target_samples + learned_samples:
        row["throttle_reasons_active"] = 0
        row["sm_clock_mhz"] = 1410
        row["temperature_c"] = 41
    return {
        "timing_artifact": {
            "schema_version": 1,
            "status": "PASS",
            "classification": classification,
            "exact_parity": True,
        },
        "target_worker": _worker(),
        "learned_worker": _worker(),
        "target_gpu_samples": target_samples,
        "learned_gpu_samples": learned_samples,
        "source_files": {
            "tools/source.py": "a" * 64,
        },
        "host_files": {
            "target_vmstat": {
                "path": "host/target-vmstat.log",
                "sha256": "b" * 64,
            },
        },
    }


def test_parse_gpu_telemetry_normalizes_driver_row():
    row = _load_module().parse_gpu_telemetry(_csv_row())[0]

    assert row == {
        "sampled_at_unix_ns": 1_786_808_823_101_000_000,
        "nvidia_timestamp": "2026/08/15 23:47:03.101",
        "gpu_index": 3,
        "uuid": "GPU-00000000000000000000000000000003",
        "pstate": "P0",
        "sm_clock_mhz": 1410,
        "memory_clock_mhz": 1512,
        "power_w": 70.38,
        "temperature_c": 41,
        "gpu_utilization_percent": 93,
        "memory_utilization_percent": 12,
        "memory_used_mib": 72455,
        "throttle_reasons_active": 1,
    }


@pytest.mark.parametrize(
    "text,match",
    (
        ("", "empty"),
        ("1,too,few", "field count"),
        (
            _csv_row(sampled_at_unix_ns=0),
            "sampled_at_unix_ns",
        ),
        (
            _csv_row() + "\n" + _csv_row(),
            "duplicate",
        ),
    ),
)
def test_parse_gpu_telemetry_rejects_invalid_rows(text, match):
    with pytest.raises(ValueError, match=match):
        _load_module().parse_gpu_telemetry(text)


def test_summarize_gpu_telemetry_aligns_repeat_boundaries():
    summary = _load_module().summarize_gpu_telemetry(
        _worker(),
        _samples(),
    )

    assert summary["expected_gpu_indices"] == [3, 4, 6, 7]
    assert summary["minimum_samples_per_repeat_gpu"] == 5
    measured = summary["measured_runs"][0]
    assert measured["repeat"] == 0
    assert measured["campaign_interval"] == {
        "started_at_unix_ns": 200,
        "finished_at_unix_ns": 300,
    }
    gpu = measured["gpus"][0]
    assert gpu["gpu_index"] == 3
    assert gpu["sample_count"] == 5
    assert gpu["sm_clock_mhz"] == {
        "minimum": 1000,
        "median": 1200,
        "maximum": 1400,
    }
    assert gpu["power_w"] == {
        "minimum": 60.0,
        "median": 62.0,
        "maximum": 64.0,
    }
    assert gpu["pstates"] == ["P0"]
    assert gpu["throttle_reasons_active_or"] == 1


def test_summarize_gpu_telemetry_requires_per_gpu_coverage():
    samples = [
        row
        for row in _samples()
        if not (
            row["gpu_index"] == 7
            and row["sampled_at_unix_ns"] == 300
        )
    ]

    with pytest.raises(
        ValueError,
        match="insufficient GPU telemetry coverage",
    ):
        _load_module().summarize_gpu_telemetry(
            _worker(),
            samples,
        )


def test_summarize_gpu_telemetry_uses_nearby_edge_sample_for_coverage():
    worker = _worker()
    worker["measured_runs"][0]["campaign_interval"] = {
        "started_at_unix_ns": 1_000_000_000,
        "finished_at_unix_ns": 2_000_000_000,
    }
    samples = []
    for gpu_index in (3, 4, 6, 7):
        for offset, timestamp in enumerate((
            900_000_000,
            1_100_000_000,
            1_300_000_000,
            1_500_000_000,
            1_900_000_000,
        )):
            row = copy.deepcopy(_samples()[offset])
            row["sampled_at_unix_ns"] = timestamp
            row["gpu_index"] = gpu_index
            row["uuid"] = f"GPU-{gpu_index:032d}"
            samples.append(row)

    summary = _load_module().summarize_gpu_telemetry(
        worker,
        samples,
    )

    assert all(
        gpu["sample_count"] == 5
        for gpu in summary["measured_runs"][0]["gpus"]
    )


def test_validate_campaign_intervals_rejects_overlap():
    worker = _worker()
    worker["measured_runs"].append({
        "repeat": 1,
        "campaign_interval": {
            "started_at_unix_ns": 300,
            "finished_at_unix_ns": 400,
        },
    })

    with pytest.raises(
        ValueError,
        match="campaign intervals overlap",
    ):
        _load_module().validate_campaign_intervals(worker)


def test_build_artifact_classifies_stable_telemetry_runtime_variance():
    artifact = _load_module().build_instability_telemetry_artifact(
        **_artifact_kwargs()
    )

    assert artifact["schema_version"] == 1
    assert artifact["status"] == "PASS"
    assert artifact["timing_classification"] == "UNSTABLE"
    assert artifact["telemetry_classification"] == (
        "RUNTIME_VARIANCE_SUSPECTED"
    )
    assert artifact["exact_parity"] is True
    assert artifact["classification_reasons"] == []
    assert artifact["policies"]["target"]["gpu_samples"]
    assert artifact["policies"]["learned"]["summary"][
        "measured_runs"
    ][0]["gpus"][0]["sample_count"] == 5


def test_build_artifact_classifies_active_throttle_reason():
    kwargs = _artifact_kwargs()
    kwargs["learned_gpu_samples"][-1][
        "throttle_reasons_active"
    ] = 4

    artifact = _load_module().build_instability_telemetry_artifact(
        **kwargs
    )

    assert artifact["telemetry_classification"] == (
        "ENVIRONMENT_CORRELATED"
    )
    assert artifact["classification_reasons"] == [
        "learned repeat 0 GPU 7 active throttle mask 0x4"
    ]


def test_build_artifact_classifies_stable_timing_baseline():
    artifact = _load_module().build_instability_telemetry_artifact(
        **_artifact_kwargs(classification="STABLE")
    )

    assert artifact["telemetry_classification"] == (
        "STABLE_BASELINE"
    )
    assert artifact["classification_reasons"] == []


def test_validate_artifact_rejects_tampered_summary():
    module = _load_module()
    artifact = module.build_instability_telemetry_artifact(
        **_artifact_kwargs()
    )
    artifact["policies"]["target"]["summary"]["measured_runs"][0][
        "gpus"
    ][0]["sample_count"] = 99

    with pytest.raises(
        ValueError,
        match="telemetry artifact recomputation mismatch",
    ):
        module.validate_instability_telemetry_artifact(artifact)


def _write_verifier_fixture(tmp_path):
    repo_root = tmp_path / "repo"
    source_path = repo_root / "tools" / "source.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("SOURCE = 1\n", encoding="utf-8")

    bundle = tmp_path / "bundle"
    host_path = bundle / "host" / "target-vmstat.log"
    host_path.parent.mkdir(parents=True)
    host_path.write_text("vmstat evidence\n", encoding="utf-8")

    kwargs = _artifact_kwargs()
    kwargs["source_files"] = {
        "tools/source.py": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
    }
    kwargs["host_files"] = {
        "target_vmstat": {
            "path": "host/target-vmstat.log",
            "sha256": hashlib.sha256(
                host_path.read_bytes()
            ).hexdigest(),
        },
    }
    artifact = _load_module().build_instability_telemetry_artifact(
        **kwargs
    )
    artifact_path = bundle / "telemetry.json"
    artifact_path.write_text(
        json.dumps(artifact, sort_keys=True),
        encoding="utf-8",
    )
    return artifact_path, repo_root, source_path, host_path


def test_verifier_checks_source_and_host_hashes(tmp_path):
    artifact_path, repo_root, _, _ = _write_verifier_fixture(
        tmp_path
    )

    receipt = _load_verifier().verify_instability_telemetry(
        artifact_path,
        repo_root,
    )

    assert receipt == {
        "status": "PASS",
        "schema_version": 1,
        "timing_classification": "UNSTABLE",
        "telemetry_classification": (
            "RUNTIME_VARIANCE_SUSPECTED"
        ),
        "exact_parity": True,
        "source_files_verified": 1,
        "host_files_verified": 1,
    }


@pytest.mark.parametrize(
    "tamper,match",
    (
        ("source", "source hash mismatch"),
        ("host", "host file hash mismatch"),
    ),
)
def test_verifier_rejects_tampered_bound_files(
    tmp_path,
    tamper,
    match,
):
    (
        artifact_path,
        repo_root,
        source_path,
        host_path,
    ) = _write_verifier_fixture(tmp_path)
    if tamper == "source":
        source_path.write_text("SOURCE = 2\n", encoding="utf-8")
    else:
        host_path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        _load_verifier().verify_instability_telemetry(
            artifact_path,
            repo_root,
        )


def test_telemetry_main_writes_source_bound_artifact(tmp_path):
    bundle = tmp_path / "bundle"
    host_path = bundle / "host" / "target-vmstat.log"
    host_path.parent.mkdir(parents=True)
    host_path.write_text("vmstat evidence\n", encoding="utf-8")
    timing_path = tmp_path / "timing.json"
    target_worker_path = tmp_path / "target.json"
    learned_worker_path = tmp_path / "learned.json"
    target_gpu_path = tmp_path / "target.csv"
    learned_gpu_path = tmp_path / "learned.csv"
    output_path = bundle / "telemetry.json"
    timing_path.write_text(
        json.dumps({
            "schema_version": 1,
            "status": "PASS",
            "classification": "UNSTABLE",
            "exact_parity": True,
        }),
        encoding="utf-8",
    )
    for path in (target_worker_path, learned_worker_path):
        path.write_text(
            json.dumps(_worker()),
            encoding="utf-8",
        )
    for path in (target_gpu_path, learned_gpu_path):
        path.write_text(_gpu_csv(), encoding="utf-8")

    status = _load_module().main([
        "--timing-artifact",
        str(timing_path),
        "--target-worker",
        str(target_worker_path),
        "--learned-worker",
        str(learned_worker_path),
        "--target-gpu-csv",
        str(target_gpu_path),
        "--learned-gpu-csv",
        str(learned_gpu_path),
        "--repo-root",
        str(ROOT),
        "--host-file",
        "target_vmstat=host/target-vmstat.log",
        "--out",
        str(output_path),
    ])

    assert status == 0
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["status"] == "PASS"
    assert artifact["telemetry_classification"] == (
        "RUNTIME_VARIANCE_SUSPECTED"
    )
    receipt = _load_verifier().verify_instability_telemetry(
        output_path,
        ROOT,
    )
    assert receipt["source_files_verified"] == 5
    assert receipt["host_files_verified"] == 1


def test_remote_runner_owns_complete_telemetry_contract():
    assert RUNNER_PATH.exists(), f"missing runner: {RUNNER_PATH}"
    script = RUNNER_PATH.read_text(encoding="utf-8")

    for expected in (
        "sitian@10.232.195.203",
        "3,4,6,7",
        "date +%s%N",
        "sleep 0.2",
        "timestamp,index,uuid,pstate",
        "clocks.current.sm",
        "clocks.current.memory",
        "power.draw",
        "temperature.gpu",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "clocks_throttle_reasons.active",
        "vmstat -t 1",
        "mpstat -P ALL 1",
        "pidstat -u -r -d -h 1",
        "--warmup-runs 2",
        "--measured-runs 8",
        "sampler_pids",
        "trap stop_samplers EXIT TERM INT",
        "verify.remote.json",
        "verify.local.json",
        "manifest.sha256",
        "tools/verify_autoregressive_draft_performance_gate.py",
        "tools/run_autoregressive_draft_b4_timing_diagnostic_remote.sh",
        "tools/run_autoregressive_draft_performance_gate_remote.sh",
        'POLICY_ORDER="${POLICY_ORDER:-target,learned}"',
    ):
        assert expected in script
    assert "torch.cuda.synchronize" not in script
    assert "find . -type f ! -name manifest.sha256" in script


def test_remote_runner_supports_validated_reverse_policy_order():
    assert RUNNER_PATH.exists(), f"missing runner: {RUNNER_PATH}"
    script = RUNNER_PATH.read_text(encoding="utf-8")

    for expected in (
        'POLICY_ORDER="${POLICY_ORDER:-target,learned}"',
        "--policy-order",
        '"target,learned"|"learned,target"',
        'IFS="," read -r -a policy_order <<< "${policy_order_csv}"',
        'for policy in "${policy_order[@]}"; do',
        '--policy "${policy}"',
        '"${artifacts}/workers/${policy}-b4.json"',
        '"${artifacts}/logs/${policy}-b4.log"',
    ):
        assert expected in script


def test_remote_runner_supports_same_policy_priming_control():
    assert RUNNER_PATH.exists(), f"missing runner: {RUNNER_PATH}"
    script = RUNNER_PATH.read_text(encoding="utf-8")

    for expected in (
        'PRIME_EACH_POLICY="${PRIME_EACH_POLICY:-0}"',
        "--prime-each-policy",
        "PRIME_EACH_POLICY=1",
        "'${REMOTE_ARTIFACTS}/prime-workers'",
        "'${REMOTE_ARTIFACTS}/prime-logs'",
        'prime_policy "${policy}"',
        "--measured-runs 1",
        '"${artifacts}/prime-workers/${policy}-prime-b4.json"',
        '"${artifacts}/prime-logs/${policy}-prime-b4.log"',
        'if [[ "${prime_each_policy}" -eq 1 ]]; then',
    ):
        assert expected in script

    assert """for policy in "${policy_order[@]}"; do
  if [[ "${prime_each_policy}" -eq 1 ]]; then
    prime_policy "${policy}"
  fi
  run_policy "${policy}"
done
""" in script


def test_remote_runner_supports_optional_ssh_control_path():
    assert RUNNER_PATH.exists(), f"missing runner: {RUNNER_PATH}"
    script = RUNNER_PATH.read_text(encoding="utf-8")

    for expected in (
        'SSH_CONTROL_PATH="${SSH_CONTROL_PATH:-}"',
        "--ssh-control-path",
        'SSH_CONTROL_PATH="$2"',
        'if [[ -n "${SSH_CONTROL_PATH}" ]]; then',
        '-o "ControlPath=${SSH_CONTROL_PATH}"',
        'printf -v RSYNC_SSH',
    ):
        assert expected in script

    assert script.count('-o "ControlPath=${SSH_CONTROL_PATH}"') == 2
    assert script.count("-o ControlPath=none") == 2


def test_remote_runner_derives_run_paths_after_cli_overrides():
    assert RUNNER_PATH.exists(), f"missing runner: {RUNNER_PATH}"
    script = RUNNER_PATH.read_text(encoding="utf-8")

    initial_local = 'LOCAL_RUN="${LOCAL_RUN:-}"'
    initial_remote = 'REMOTE_RUN="${REMOTE_RUN:-}"'
    derived_local = (
        'LOCAL_RUN="${LOCAL_RUN:-${REPO_ROOT}/experiments/'
        'autoregressive_draft/${RUN_TAG}}"'
    )
    derived_remote = (
        'REMOTE_RUN="${REMOTE_RUN:-${REMOTE_BASE}/run/${RUN_TAG}}"'
    )
    parse_loop = "while [[ $# -gt 0 ]]; do"

    for initial_default in (initial_local, initial_remote):
        assert initial_default in script
        assert script.index(initial_default) < script.index(parse_loop)
    for derived_default in (derived_local, derived_remote):
        assert derived_default in script
        assert script.index(derived_default) > script.index(parse_loop)


def test_remote_runner_owns_host_semantic_alignment_contract():
    script = RUNNER_PATH.read_text(encoding="utf-8")
    for expected in (
        "tools/autoregressive_draft_host_sampler.py",
        "tools/autoregressive_draft_host_semantic_diagnostic.py",
        "tools/verify_autoregressive_draft_host_semantic_diagnostic.py",
        "tools/test_autoregressive_draft_host_sampler.py",
        "tools/test_autoregressive_draft_host_semantic_diagnostic.py",
        "'${REMOTE_ARTIFACTS}/host-semantic'",
        '--interval-seconds 0.2',
        '"${artifacts}/host-semantic/${policy}-host.jsonl"',
        '"${artifacts}/host-semantic/${policy}-host.stderr.log"',
        '--target-host-jsonl "${artifacts}/host-semantic/target-host.jsonl"',
        '--learned-host-jsonl "${artifacts}/host-semantic/learned-host.jsonl"',
        '--out "${artifacts}/host-semantic.json"',
        "verify.host.remote.json",
        "verify.host.local.json",
        "verify-host-remote-exit-code.txt",
    ):
        assert expected in script

    policy_loop_index = script.index(
        'for policy in "${policy_order[@]}"; do'
    )
    prime_index = script.index(
        'prime_policy "${policy}"',
        policy_loop_index,
    )
    run_policy_index = script.index(
        'run_policy "${policy}"',
        prime_index,
    )
    assert prime_index < run_policy_index

    run_policy_definition_index = script.index("run_policy()")
    sampler_index = script.index(
        'start_samplers "${policy}"',
        run_policy_definition_index,
    )
    measured_index = script.index(
        "tools/autoregressive_draft_performance_worker.py",
        sampler_index,
    )
    assert sampler_index < measured_index
    assert script.count(
        "tools/autoregressive_draft_b4_timing_diagnostic.py \\\n"
        '    --target-worker "${artifacts}/workers/target-b4.json"'
    ) == 1
    assert script.count(
        "tools/autoregressive_draft_instability_telemetry.py \\\n"
        '    --timing-artifact "${artifacts}/result.json"'
    ) == 1
    assert "torch.cuda.synchronize" not in script


def test_learned_aa_runner_owns_isolated_epoch_contract():
    assert LEARNED_AA_RUNNER_PATH.exists(), (
        f"missing runner: {LEARNED_AA_RUNNER_PATH}"
    )
    script = LEARNED_AA_RUNNER_PATH.read_text(encoding="utf-8")

    for expected in (
        'REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"',
        "/data00/home/sitian/miniconda3/envs/py311/bin/python",
        "/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815",
        'GPU_INDICES="${GPU_INDICES:-3,4,6,7}"',
        'EPOCH_ORDER="learned_a,learned_b"',
        "PRIME_EACH_EPOCH=1",
        "for epoch in learned_a learned_b; do",
        '--policy learned',
        '--batch-size 4',
        '--warmup-runs 2',
        '--measured-runs 1',
        '--measured-runs 8',
        'prime_epoch "${epoch}"',
        'run_epoch "${epoch}"',
        'start_samplers "${epoch}"',
        "stop_samplers",
        'workers/${slug}-b4.json',
        'prime-workers/${slug}-prime-b4.json',
        'telemetry/${slug}-gpu.csv',
        'host-semantic/${slug}-host.jsonl',
        "tools/speculative_runtime_performance_gate.py",
        "tools/verify_autoregressive_draft_performance_gate.py",
        "tools/autoregressive_draft_b4_timing_diagnostic.py",
        "tools/verify_autoregressive_draft_b4_timing_diagnostic.py",
        "tools/run_autoregressive_draft_b4_timing_diagnostic_remote.sh",
        "tools/run_autoregressive_draft_performance_gate_remote.sh",
        "tools/verify_autoregressive_draft_instability_telemetry.py",
        "tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh",
        "tools/verify_autoregressive_draft_host_semantic_diagnostic.py",
        "tools/autoregressive_draft_learned_aa_diagnostic.py",
        "tools/verify_autoregressive_draft_learned_aa_diagnostic.py",
        "verify.learned-aa.remote.json",
        "verify.learned-aa.local.json",
        "manifest.sha256",
    ):
        assert expected in script

    for forbidden in (
        "torch.cuda.synchronize",
        "--policy learned_a",
        "--policy learned_b",
        "killall",
        "pkill",
        "kill 703088",
    ):
        assert forbidden not in script


def test_learned_aa_runner_is_executable():
    assert os.access(LEARNED_AA_RUNNER_PATH, os.X_OK)


def test_learned_aa_runner_primes_before_each_measured_epoch():
    script = LEARNED_AA_RUNNER_PATH.read_text(encoding="utf-8")
    loop = """for epoch in learned_a learned_b; do
  prime_epoch "${epoch}"
  run_epoch "${epoch}"
done
"""
    assert loop in script

    run_epoch_index = script.index("run_epoch()")
    sampler_index = script.index(
        'start_samplers "${epoch}"',
        run_epoch_index,
    )
    worker_index = script.index(
        "tools/autoregressive_draft_performance_worker.py",
        sampler_index,
    )
    stop_index = script.index("stop_samplers", worker_index)
    assert sampler_index < worker_index < stop_index


def test_learned_aa_runner_owns_transport_and_failure_receipts():
    script = LEARNED_AA_RUNNER_PATH.read_text(encoding="utf-8")
    for expected in (
        'SSH_CONTROL_PATH="${SSH_CONTROL_PATH:-}"',
        "--ssh-control-path",
        '-o "ControlPath=${SSH_CONTROL_PATH}"',
        "preflight-exit-code.txt",
        "learned-a-prime-exit-code.txt",
        "learned-a-worker-exit-code.txt",
        "learned-b-prime-exit-code.txt",
        "learned-b-worker-exit-code.txt",
        "diagnostic-exit-code.txt",
        "verify-learned-aa-remote-exit-code.txt",
        "remote-status.txt",
        "find . -type f ! -name manifest.sha256",
        "trap stop_samplers EXIT TERM INT",
        "sampler_pids",
    ):
        assert expected in script

    initial_local = 'LOCAL_RUN="${LOCAL_RUN:-}"'
    initial_remote = 'REMOTE_RUN="${REMOTE_RUN:-}"'
    parse_loop = "while [[ $# -gt 0 ]]; do"
    derived_local = (
        'LOCAL_RUN="${LOCAL_RUN:-${REPO_ROOT}/experiments/'
        'autoregressive_draft/${RUN_TAG}}"'
    )
    derived_remote = (
        'REMOTE_RUN="${REMOTE_RUN:-${REMOTE_BASE}/run/${RUN_TAG}}"'
    )
    assert script.index(initial_local) < script.index(parse_loop)
    assert script.index(initial_remote) < script.index(parse_loop)
    assert script.index(derived_local) > script.index(parse_loop)
    assert script.index(derived_remote) > script.index(parse_loop)


def test_paired_stability_runner_is_executable():
    assert PAIRED_STABILITY_RUNNER_PATH.exists()
    assert os.access(PAIRED_STABILITY_RUNNER_PATH, os.X_OK)


def test_paired_stability_runner_owns_fixed_protocol():
    script = PAIRED_STABILITY_RUNNER_PATH.read_text(encoding="utf-8")
    for expected in (
        "SCHEDULE_TEXT=$'AB\\nBA\\nBA\\nAB\\n'",
        "EXPECTED_BLOCKS=4",
        "EXPECTED_EPOCHS=8",
        "MEASURED_RUNS_PER_EPOCH=5",
        "MEASURED_RUNS_TOTAL=40",
        "--warmup-runs 2",
        "--measured-runs 1",
        "--measured-runs 5",
        "sitian@10.232.195.203",
        "/data00/home/sitian/miniconda3/envs/py311/bin/python",
        "/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815",
        "target-qwen3-1.7b",
        "GPU_INDICES=3,4,6,7",
        "PROTECTED_GPU7_PID=703088",
        "paired-stability.json",
        "verify.paired-stability.remote.json",
        "verify.paired-stability.local.json",
        "manifest.sha256",
    ):
        assert expected in script
    for forbidden in (
        "--schedule",
        "SCHEDULE_TEXT=${",
        "torch.cuda.synchronize",
        "kill 703088",
        "pkill",
        "/data00/run",
        "/data00/tmp",
        "/data00/experiments",
    ):
        assert forbidden not in script


def test_paired_stability_runner_refuses_overwrite_and_replication():
    script = PAIRED_STABILITY_RUNNER_PATH.read_text(encoding="utf-8")
    assert "refusing to overwrite local run" in script
    assert "refusing to overwrite remote run" in script
    assert "--replicate" not in script
    assert "replication bundle" not in script


def test_paired_stability_runner_packages_complete_dependency_closure():
    script = PAIRED_STABILITY_RUNNER_PATH.read_text(encoding="utf-8")
    for expected in (
        "tinyvllm",
        "tools/autoregressive_draft_performance_worker.py",
        "tools/autoregressive_draft_performance_gate.py",
        "tools/autoregressive_draft_host_sampler.py",
        "tools/autoregressive_draft_host_semantic_diagnostic.py",
        "tools/autoregressive_draft_instability_telemetry.py",
        "tools/autoregressive_draft_paired_stability_diagnostic.py",
        "tools/verify_autoregressive_draft_paired_stability_diagnostic.py",
        "tools/test_autoregressive_draft_executor.py",
        "tools/test_autoregressive_draft_performance_gate.py",
        "tools/test_autoregressive_draft_host_sampler.py",
        "tools/test_autoregressive_draft_host_semantic_diagnostic.py",
        "tools/test_autoregressive_draft_instability_telemetry.py",
        "tools/test_autoregressive_draft_paired_stability_diagnostic.py",
        "tools/run_autoregressive_draft_paired_stability_remote.sh",
    ):
        assert expected in script


def test_paired_stability_runner_reaps_only_owned_processes():
    script = PAIRED_STABILITY_RUNNER_PATH.read_text(encoding="utf-8")
    for expected in (
        "worker_pid",
        "sampler_pids",
        "epoch_owned_pids",
        "stop_owned_processes",
        'wait "${worker_pid}"',
        "trap stop_owned_processes EXIT TERM INT",
        "runner_owned_pids_remaining",
    ):
        assert expected in script


def test_paired_stability_runner_materializes_fixed_epoch_layout():
    script = PAIRED_STABILITY_RUNNER_PATH.read_text(encoding="utf-8")
    for expected in (
        "0 AB A first 0 block-0-ab/a-first",
        "0 AB B second 1 block-0-ab/b-second",
        "1 BA B first 2 block-1-ba/b-first",
        "1 BA A second 3 block-1-ba/a-second",
        "2 BA B first 4 block-2-ba/b-first",
        "2 BA A second 5 block-2-ba/a-second",
        "3 AB A first 6 block-3-ab/a-first",
        "3 AB B second 7 block-3-ab/b-second",
        'nvidia-smi -L >"${epoch_dir}/gpu.${phase}.txt"',
        'ps -eo pid,ppid,user,lstart,args \\',
        "gpu-process.${phase}.csv",
        "safety-stop.json",
        "executed_epoch_keys",
        "unexecuted_epoch_keys",
    ):
        assert expected in script


def test_paired_stability_runner_preserves_partial_evidence():
    script = PAIRED_STABILITY_RUNNER_PATH.read_text(encoding="utf-8")
    assert 'printf \'{}\\n\' >"${epoch_dir}/raw.json"' in script
    diagnostic_index = script.index(
        "tools/autoregressive_draft_paired_stability_diagnostic.py",
        script.index("diagnostic_status=125"),
    )
    campaign_failure_index = script.index(
        'if [[ "${campaign_status}" -ne 0 ]]; then',
        diagnostic_index,
    )
    assert diagnostic_index < campaign_failure_index
    assert "PAIRED_PROTOCOL_UNSTABLE" in script
    assert "NO_REPRODUCIBLE_PROCESS_EFFECT" in script
    assert "CANDIDATE_PROCESS_BOUNDARY_EFFECT" in script


def test_paired_stability_runner_seals_before_detached_receipts():
    script = PAIRED_STABILITY_RUNNER_PATH.read_text(encoding="utf-8")
    manifest_index = script.index(
        "xargs -0 shasum -a 256 >manifest.sha256"
    )
    remote_receipt_index = script.index(
        '--receipt "${remote_artifacts}/'
        'verify.paired-stability.remote.json"',
        manifest_index,
    )
    assert manifest_index < remote_receipt_index
    assert "! -name 'verify.paired-stability.remote.json'" in script
    assert "! -name 'verify.paired-stability.remote.log'" in script
    assert "! -name 'verify.paired-stability.local.json'" in script
    assert "! -name 'verify.paired-stability.local.log'" in script
    lines = {line.strip().rstrip("\\").strip() for line in script.splitlines()}
    assert "manifest-exit-code.txt" not in lines
    assert "verify-paired-stability-remote-exit-code.txt" not in lines


def test_paired_stability_runner_isolates_local_artifact_root():
    script = PAIRED_STABILITY_RUNNER_PATH.read_text(encoding="utf-8")
    assert 'LOCAL_ARTIFACTS="${LOCAL_RUN}/artifacts"' in script
    assert (
        '"${REMOTE_HOST}:${REMOTE_ARTIFACTS}/" \\\n'
        '  "${LOCAL_ARTIFACTS}/"'
    ) in script
    for expected in (
        '--artifact "${LOCAL_ARTIFACTS}/paired-stability.json"',
        '--manifest "${LOCAL_ARTIFACTS}/manifest.sha256"',
        "${LOCAL_ARTIFACTS}/verify.paired-stability.local.json",
        "${LOCAL_ARTIFACTS}/verify.paired-stability.remote.json",
    ):
        assert expected in script
