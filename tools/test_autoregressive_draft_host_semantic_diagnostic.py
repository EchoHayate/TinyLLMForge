from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tools"
    / "autoregressive_draft_host_semantic_diagnostic.py"
)
VERIFIER_PATH = (
    ROOT
    / "tools"
    / "verify_autoregressive_draft_host_semantic_diagnostic.py"
)
RUNNER_PATH = (
    ROOT
    / "tools"
    / "run_autoregressive_draft_b4_instability_telemetry_remote.sh"
)


def _load_module():
    assert MODULE_PATH.exists(), f"missing module: {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location(
        "autoregressive_draft_host_semantic_test_module",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample(unix_ns, monotonic_ns, offset=0):
    return {
        "schema_version": 1,
        "sampled_at_unix_ns": unix_ns,
        "sampled_at_monotonic_ns": monotonic_ns,
        "cpu_user_ticks": 100 + offset,
        "cpu_nice_ticks": 2,
        "cpu_system_ticks": 20 + offset,
        "cpu_idle_ticks": 300 + offset,
        "cpu_iowait_ticks": 5 + offset,
        "cpu_irq_ticks": 1,
        "cpu_softirq_ticks": 2,
        "cpu_steal_ticks": 0,
        "procs_running": 2 + offset,
        "procs_blocked": 1,
        "context_switches_total": 1000 + 10 * offset,
        "processes_forked_total": 50 + offset,
        "loadavg_1m": 1.0 + offset,
        "loadavg_5m": 2.0,
        "loadavg_15m": 3.0,
        "major_faults_total": 10 + offset,
        "page_in_kib_total": 100 + 4 * offset,
        "page_out_kib_total": 200 + 8 * offset,
        "swap_in_kib_total": 0,
        "swap_out_kib_total": 0,
        "memory_available_kib": 10000 - offset,
        "memory_cached_kib": 5000,
        "memory_dirty_kib": 20 + offset,
        "memory_writeback_kib": 2 + offset,
        "cpu_psi_some_total_us": 100 + 10 * offset,
        "cpu_psi_full_total_us": None,
        "io_psi_some_total_us": 200 + 20 * offset,
        "io_psi_full_total_us": 20 + 2 * offset,
        "memory_psi_some_total_us": 300 + 30 * offset,
        "memory_psi_full_total_us": 30 + 3 * offset,
    }


def _worker():
    return {
        "policy": "learned",
        "batch_size": 4,
        "measured_runs": [{
            "repeat": 0,
            "campaign_interval": {
                "started_at_unix_ns": 1_000_000_000,
                "finished_at_unix_ns": 2_000_000_000,
            },
            "timing": {
                "per_request": [
                    {"completion_latency_s": value, "tpot_s": value / 10}
                    for value in (1.0, 2.0, 3.0, 4.0)
                ],
            },
            "runtime": {
                "draft_executor_timing": {
                    "max_rank_ms": {"proposal_forward": 55.0}
                }
            },
            "outputs": [[1], [2], [3], [4]],
        }],
    }


def test_align_repeat_uses_outer_boundary_samples():
    module = _load_module()
    samples = [
        _sample(800_000_000, 100, 0),
        _sample(1_100_000_000, 400, 1),
        _sample(1_700_000_000, 1_000, 2),
        _sample(2_200_000_000, 1_500, 3),
    ]
    aligned = module.align_repeat_samples(_worker(), samples)
    assert aligned[0]["samples"] == samples
    assert aligned[0]["host_sample_interval"] == {
        "started_at_unix_ns": 800_000_000,
        "finished_at_unix_ns": 2_200_000_000,
        "started_at_monotonic_ns": 100,
        "finished_at_monotonic_ns": 1_500,
    }


def test_alignment_ignores_sample_gap_after_repeat_finish_boundary():
    module = _load_module()
    samples = [
        _sample(800_000_000, 100, 0),
        _sample(1_100_000_000, 300_000_000, 1),
        _sample(1_700_000_000, 900_000_000, 2),
        _sample(2_200_000_000, 1_400_000_000, 3),
        _sample(3_500_000_000, 2_700_000_000, 4),
    ]
    text = "\n".join(json.dumps(sample) for sample in samples)

    parsed = module.parse_host_jsonl(text)
    aligned = module.align_repeat_samples(_worker(), parsed)

    assert aligned[0]["samples"] == samples[:-1]


def test_alignment_rejects_sample_gap_inside_repeat_boundary():
    module = _load_module()
    samples = [
        _sample(800_000_000, 100, 0),
        _sample(1_100_000_000, 300_000_000, 1),
        _sample(1_700_000_000, 1_100_000_001, 2),
        _sample(2_200_000_000, 1_400_000_000, 3),
    ]

    with pytest.raises(ValueError, match="sample gap"):
        module.align_repeat_samples(_worker(), samples)


@pytest.mark.parametrize(
    "mutate,match",
    (
        (
            lambda rows: rows.__setitem__(
                0, _sample(500_000_000, 100, 0)
            ),
            "start boundary",
        ),
        (
            lambda rows: rows.__setitem__(
                -1, _sample(2_500_000_000, 1_500, 3)
            ),
            "finish boundary",
        ),
        (
            lambda rows: rows[2].__setitem__(
                "sampled_at_monotonic_ns", 100
            ),
            "timestamp",
        ),
        (
            lambda rows: rows[2].__setitem__(
                "context_switches_total", 1
            ),
            "counter regressed",
        ),
    ),
)
def test_align_repeat_rejects_invalid_coverage(mutate, match):
    rows = [
        _sample(800_000_000, 100, 0),
        _sample(1_100_000_000, 300_000_000, 1),
        _sample(1_700_000_000, 900_000_000, 2),
        _sample(2_200_000_000, 1_400_000_000, 3),
    ]
    mutate(rows)
    with pytest.raises(ValueError, match=match):
        _load_module().align_repeat_samples(_worker(), rows)


def test_derive_repeat_metrics_matches_hand_computation():
    module = _load_module()
    rows = [
        _sample(1_000_000_000, 1_000_000_000, 0),
        _sample(2_000_000_000, 2_000_000_000, 1),
    ]
    metrics = module.derive_repeat_metrics({
        "samples": rows,
        "host_sample_interval": {
            "started_at_monotonic_ns": 1_000_000_000,
            "finished_at_monotonic_ns": 2_000_000_000,
        },
    })

    assert metrics["context_switches_per_second"] == 10.0
    assert metrics["forks_per_second"] == 1.0
    assert metrics["major_faults_per_second"] == 1.0
    assert metrics["page_in_kib_per_second"] == 4.0
    assert metrics["page_out_kib_per_second"] == 8.0
    assert metrics["run_queue_mean"] == 2.5
    assert metrics["run_queue_max"] == 3
    assert metrics["memory_available_kib_min"] == 9_999
    assert metrics["memory_dirty_kib_max"] == 21
    assert metrics["io_psi_some_fraction"] == 20 / 1_000_000


def test_extract_repeat_timing_uses_batch_medians():
    row = _load_module().extract_repeat_timing(_worker())[0]
    assert row == {
        "repeat": 0,
        "e2e_s": 2.5,
        "tpot_s": 0.25,
        "executor_proposal_forward_ms": 55.0,
    }


def _campaign_worker(policy, *, learned_scale=1.0):
    runs = []
    for repeat in range(8):
        start = 10_000_000_000 + repeat * 2_000_000_000
        scale = learned_scale if policy == "learned" else 1.0
        base = (2.0 + repeat * 0.01) * scale
        runs.append({
            "repeat": repeat,
            "campaign_interval": {
                "started_at_unix_ns": start,
                "finished_at_unix_ns": start + 1_000_000_000,
            },
            "timing": {
                "per_request": [
                    {
                        "completion_latency_s": base + offset,
                        "tpot_s": base / 10 + offset / 10,
                    }
                    for offset in (0.0, 0.1, 0.2, 0.3)
                ],
            },
            "runtime": {
                "draft_executor_timing": {
                    "max_rank_ms": {
                        "proposal_forward": (
                            50.0 + repeat
                        ) * scale
                    }
                }
            },
            "outputs": [
                [repeat, request] for request in range(4)
            ],
        })
    return {
        "policy": policy,
        "batch_size": 4,
        "tensor_parallel_size": 4,
        "measured_runs": runs,
    }


def _campaign_samples(*, pressure_scale=1.0):
    rows = []
    start = 9_800_000_000
    for index in range(78):
        row = _sample(
            start + index * 200_000_000,
            1_000_000_000 + index * 200_000_000,
            index,
        )
        row["procs_running"] = 2 + int(
            pressure_scale * (index % 4)
        )
        row["context_switches_total"] = (
            1_000 + int(pressure_scale * 10 * index)
        )
        row["major_faults_total"] = (
            10 + int(pressure_scale * index)
        )
        row["io_psi_some_total_us"] = (
            200 + int(pressure_scale * 20 * index)
        )
        row["memory_psi_some_total_us"] = (
            300 + int(pressure_scale * 30 * index)
        )
        rows.append(row)
    return rows


def _timing_artifact():
    return {
        "schema_version": 1,
        "status": "PASS",
        "classification": "STABLE",
        "exact_parity": True,
    }


def _gpu_artifact():
    return {
        "schema_version": 1,
        "status": "PASS",
        "timing_classification": "STABLE",
        "telemetry_classification": "STABLE_BASELINE",
        "exact_parity": True,
    }


def _input_files():
    names = (
        "timing_artifact",
        "gpu_telemetry_artifact",
        "target_worker",
        "learned_worker",
        "target_host_jsonl",
        "learned_host_jsonl",
    )
    paths = (
        "result.json",
        "telemetry.json",
        "workers/target-b4.json",
        "workers/learned-b4.json",
        "host-semantic/target-host.jsonl",
        "host-semantic/learned-host.jsonl",
    )
    return {
        name: {
            "path": path,
            "sha256": f"{index + 1:064x}",
        }
        for index, (name, path) in enumerate(zip(names, paths))
    }


def _campaign_artifact_kwargs(
    *,
    policy_order="target,learned",
    learned_scale=1.0,
    learned_pressure_scale=1.0,
):
    return {
        "timing_artifact": _timing_artifact(),
        "gpu_telemetry_artifact": _gpu_artifact(),
        "target_worker": _campaign_worker("target"),
        "learned_worker": _campaign_worker(
            "learned",
            learned_scale=learned_scale,
        ),
        "target_samples": _campaign_samples(),
        "learned_samples": _campaign_samples(
            pressure_scale=learned_pressure_scale
        ),
        "policy_order": policy_order,
        "prime_each_policy": True,
        "source_files": {
            "tools/source.py": "a" * 64,
        },
        "input_files": _input_files(),
    }


def test_build_campaign_artifact_is_aligned_not_cross_classified():
    module = _load_module()
    artifact = module.build_host_semantic_artifact(
        **_campaign_artifact_kwargs()
    )

    assert artifact["status"] == "PASS"
    assert artifact["classification"] == "ALIGNED_CAMPAIGN"
    assert artifact["exact_parity"] is True
    assert len(
        artifact["policies"]["learned"]["measured_runs"]
    ) == 8


def test_build_campaign_artifact_is_deterministic():
    module = _load_module()
    kwargs = _campaign_artifact_kwargs()
    assert module.build_host_semantic_artifact(
        **copy.deepcopy(kwargs)
    ) == module.build_host_semantic_artifact(
        **copy.deepcopy(kwargs)
    )


def test_average_ranks_handles_ties():
    assert _load_module().average_ranks([10.0, 20.0, 20.0, 40.0]) == [
        1.0, 2.5, 2.5, 4.0
    ]


def test_spearman_returns_none_for_zero_variance():
    assert _load_module().spearman_rho(
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 3.0],
    ) is None


@pytest.mark.parametrize(
    "position_effect,worse_count,e2e_corr,proposal_corr,expected",
    (
        (
            0.20,
            2,
            {"run_queue_mean", "io_psi_some_fraction"},
            {"run_queue_mean", "major_faults_per_second"},
            "HOST_PRESSURE_ASSOCIATED",
        ),
        (
            0.20,
            1,
            {"run_queue_mean"},
            {"run_queue_mean"},
            "HOST_PRESSURE_NOT_SUPPORTED",
        ),
        (
            0.05,
            3,
            {"run_queue_mean", "io_psi_some_fraction"},
            {"run_queue_mean", "io_psi_some_fraction"},
            "HOST_ALIGNMENT_INCONCLUSIVE",
        ),
    ),
)
def test_classify_host_comparison(
    position_effect,
    worse_count,
    e2e_corr,
    proposal_corr,
    expected,
):
    module = _load_module()
    classification, reasons = module.classify_host_comparison(
        learned_e2e_relative_delta=position_effect,
        worse_primary_metrics={
            f"metric_{index}" for index in range(worse_count)
        },
        e2e_correlated_metrics=e2e_corr,
        proposal_correlated_metrics=proposal_corr,
    )
    assert classification == expected
    assert isinstance(reasons, list)


def _comparison_campaigns():
    module = _load_module()
    learned_first_kwargs = _campaign_artifact_kwargs(
        policy_order="learned,target",
        learned_scale=1.2,
        learned_pressure_scale=2.0,
    )
    for index, row in enumerate(
        learned_first_kwargs["input_files"].values(),
        start=100,
    ):
        row["sha256"] = f"{index:064x}"
    learned_first = module.build_host_semantic_artifact(
        **learned_first_kwargs
    )
    learned_second = module.build_host_semantic_artifact(
        **_campaign_artifact_kwargs(
            policy_order="target,learned",
        )
    )
    return learned_first, learned_second


def _comparison_references():
    return (
        {
            "path": "r8/host-semantic.json",
            "sha256": "b" * 64,
            "policy_order": "learned,target",
            "role": "learned_first",
        },
        {
            "path": "r7/host-semantic.json",
            "sha256": "c" * 64,
            "policy_order": "target,learned",
            "role": "learned_second",
        },
    )


def test_build_comparison_classifies_associated_pressure():
    module = _load_module()
    learned_first, learned_second = _comparison_campaigns()
    first_reference, second_reference = _comparison_references()

    artifact = module.build_host_semantic_comparison(
        first_artifact=learned_first,
        second_artifact=learned_second,
        first_reference=first_reference,
        second_reference=second_reference,
    )

    assert artifact["status"] == "PASS"
    assert artifact["classification"] == "HOST_PRESSURE_ASSOCIATED"
    assert set(artifact["campaign_artifacts"]) == {
        "learned_first",
        "learned_second",
    }
    assert artifact["learned_position_effect"]["e2e_s"][
        "relative_delta"
    ] >= 0.10


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("path", "../r8/host-semantic.json", "path"),
        ("role", "learned_second", "role"),
    ),
)
def test_build_comparison_rejects_invalid_reference(field, value, match):
    module = _load_module()
    learned_first, learned_second = _comparison_campaigns()
    first_reference, second_reference = _comparison_references()
    first_reference[field] = value

    with pytest.raises(ValueError, match=match):
        module.build_host_semantic_comparison(
            first_artifact=learned_first,
            second_artifact=learned_second,
            first_reference=first_reference,
            second_reference=second_reference,
        )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_main_builds_campaign_artifact_from_files(tmp_path, monkeypatch):
    module = _load_module()
    repo_root = tmp_path / "repo"
    source_path = repo_root / "tools" / "source.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("SOURCE = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "SOURCE_FILE_PATHS",
        ("tools/source.py",),
    )

    campaign = tmp_path / "campaign"
    paths = {
        "timing": campaign / "result.json",
        "gpu": campaign / "telemetry.json",
        "target_worker": campaign / "workers" / "target-b4.json",
        "learned_worker": campaign / "workers" / "learned-b4.json",
        "target_host": campaign / "host-semantic" / "target-host.jsonl",
        "learned_host": campaign / "host-semantic" / "learned-host.jsonl",
        "out": campaign / "host-semantic.json",
    }
    _write_json(paths["timing"], _timing_artifact())
    _write_json(paths["gpu"], _gpu_artifact())
    _write_json(paths["target_worker"], _campaign_worker("target"))
    _write_json(paths["learned_worker"], _campaign_worker("learned"))
    _write_jsonl(paths["target_host"], _campaign_samples())
    _write_jsonl(paths["learned_host"], _campaign_samples())

    status = module.main([
        "--timing-artifact", str(paths["timing"]),
        "--gpu-telemetry-artifact", str(paths["gpu"]),
        "--target-worker", str(paths["target_worker"]),
        "--learned-worker", str(paths["learned_worker"]),
        "--target-host-jsonl", str(paths["target_host"]),
        "--learned-host-jsonl", str(paths["learned_host"]),
        "--policy-order", "target,learned",
        "--prime-each-policy",
        "--repo-root", str(repo_root),
        "--out", str(paths["out"]),
    ])

    artifact = json.loads(paths["out"].read_text(encoding="utf-8"))
    assert status == 0
    assert artifact["classification"] == "ALIGNED_CAMPAIGN"
    assert artifact["input_files"]["target_worker"]["path"] == (
        "workers/target-b4.json"
    )
    assert set(artifact["source_files"]) == {"tools/source.py"}


def test_main_builds_comparison_artifact_from_files(tmp_path):
    module = _load_module()
    learned_first, learned_second = _comparison_campaigns()
    first_path = tmp_path / "r8" / "host-semantic.json"
    second_path = tmp_path / "r7" / "host-semantic.json"
    out_path = tmp_path / "comparison.json"
    _write_json(first_path, learned_first)
    _write_json(second_path, learned_second)

    status = module.main([
        "--campaign-artifact", str(first_path),
        "--comparison-artifact", str(second_path),
        "--out", str(out_path),
    ])

    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    assert status == 0
    assert artifact["classification"] == "HOST_PRESSURE_ASSOCIATED"
    assert artifact["campaign_artifacts"]["learned_first"]["path"] == (
        "r8/host-semantic.json"
    )
    assert artifact["campaign_artifacts"]["learned_second"]["path"] == (
        "r7/host-semantic.json"
    )


def test_main_rejects_mixed_campaign_and_comparison_modes(tmp_path):
    module = _load_module()
    with pytest.raises(ValueError, match="mode"):
        module.main([
            "--campaign-artifact", str(tmp_path / "campaign.json"),
            "--timing-artifact", str(tmp_path / "timing.json"),
            "--comparison-artifact", str(tmp_path / "other.json"),
            "--out", str(tmp_path / "out.json"),
        ])


SOURCE_FIXTURE_PATHS = (
    "tools/autoregressive_draft_performance_worker.py",
    "tools/autoregressive_draft_b4_timing_diagnostic.py",
    "tools/autoregressive_draft_instability_telemetry.py",
    "tools/autoregressive_draft_host_sampler.py",
    "tools/autoregressive_draft_host_semantic_diagnostic.py",
    "tools/verify_autoregressive_draft_host_semantic_diagnostic.py",
)


def _sha256_path(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_campaign_bundle_at(
    bundle,
    repo_root,
    *,
    policy_order,
    campaign_tag,
    learned_scale,
    learned_pressure_scale,
):
    module = _load_module()
    for relative_path in SOURCE_FIXTURE_PATHS:
        source_path = repo_root / relative_path
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(
            f"SOURCE = {relative_path!r}\n",
            encoding="utf-8",
        )
    bundle.mkdir(parents=True, exist_ok=True)
    timing = {
        **_timing_artifact(),
        "campaign_tag": campaign_tag,
    }
    gpu = {
        **_gpu_artifact(),
        "campaign_tag": campaign_tag,
    }
    target_worker = {
        **_campaign_worker("target"),
        "campaign_tag": campaign_tag,
    }
    learned_worker = {
        **_campaign_worker(
            "learned",
            learned_scale=learned_scale,
        ),
        "campaign_tag": campaign_tag,
    }
    target_samples = _campaign_samples(
        pressure_scale=1.0 + 0.01 * learned_scale
    )
    learned_samples = _campaign_samples(
        pressure_scale=learned_pressure_scale
    )
    input_payloads = {
        "timing_artifact": ("result.json", timing),
        "gpu_telemetry_artifact": ("telemetry.json", gpu),
        "target_worker": (
            "workers/target-b4.json",
            target_worker,
        ),
        "learned_worker": (
            "workers/learned-b4.json",
            learned_worker,
        ),
    }
    for _, (relative_path, payload) in input_payloads.items():
        _write_json(bundle / relative_path, payload)
    host_payloads = {
        "target_host_jsonl": (
            "host-semantic/target-host.jsonl",
            target_samples,
        ),
        "learned_host_jsonl": (
            "host-semantic/learned-host.jsonl",
            learned_samples,
        ),
    }
    for _, (relative_path, rows) in host_payloads.items():
        path = bundle / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                json.dumps(row, sort_keys=True) for row in rows
            ) + "\n",
            encoding="utf-8",
        )
    input_files = {
        name: {
            "path": relative_path,
            "sha256": _sha256_path(bundle / relative_path),
        }
        for name, (relative_path, _) in {
            **input_payloads,
            **host_payloads,
        }.items()
    }
    source_files = {
        relative_path: _sha256_path(repo_root / relative_path)
        for relative_path in SOURCE_FIXTURE_PATHS
    }
    artifact = module.build_host_semantic_artifact(
        timing_artifact=timing,
        gpu_telemetry_artifact=gpu,
        target_worker=target_worker,
        learned_worker=learned_worker,
        target_samples=target_samples,
        learned_samples=learned_samples,
        policy_order=policy_order,
        prime_each_policy=True,
        source_files=source_files,
        input_files=input_files,
    )
    artifact_path = bundle / "host-semantic.json"
    _write_json(artifact_path, artifact)
    return artifact_path


def _write_campaign_bundle(tmp_path):
    repo_root = tmp_path / "repo"
    artifact_path = _write_campaign_bundle_at(
        tmp_path / "bundle",
        repo_root,
        policy_order="target,learned",
        campaign_tag="r7",
        learned_scale=1.0,
        learned_pressure_scale=1.0,
    )
    return artifact_path, repo_root


def _tamper_campaign_bundle(
    artifact_path,
    repo_root,
    target,
):
    artifact = json.loads(
        artifact_path.read_text(encoding="utf-8")
    )
    if target == "source":
        relative_path = next(iter(artifact["source_files"]))
        (repo_root / relative_path).write_text(
            "SOURCE = 'tampered'\n",
            encoding="utf-8",
        )
    elif target == "worker":
        row = artifact["input_files"]["target_worker"]
        (artifact_path.parent / row["path"]).write_text(
            "{}\n",
            encoding="utf-8",
        )
    elif target == "host":
        row = artifact["input_files"]["learned_host_jsonl"]
        with (artifact_path.parent / row["path"]).open(
            "a",
            encoding="utf-8",
        ) as output:
            output.write("{}\n")
    elif target == "summary":
        artifact["policies"]["learned"]["measured_runs"][0][
            "metrics"
        ]["run_queue_mean"] += 1.0
        _write_json(artifact_path, artifact)
    else:
        raise AssertionError(target)


def _write_comparison_bundle(tmp_path):
    module = _load_module()
    repo_root = tmp_path / "repo"
    r7_path = _write_campaign_bundle_at(
        tmp_path / "r7",
        repo_root,
        policy_order="target,learned",
        campaign_tag="r7",
        learned_scale=1.0,
        learned_pressure_scale=1.0,
    )
    r8_path = _write_campaign_bundle_at(
        tmp_path / "r8",
        repo_root,
        policy_order="learned,target",
        campaign_tag="r8",
        learned_scale=1.2,
        learned_pressure_scale=2.0,
    )
    r7 = json.loads(r7_path.read_text(encoding="utf-8"))
    r8 = json.loads(r8_path.read_text(encoding="utf-8"))
    comparison = module.build_host_semantic_comparison(
        first_artifact=r7,
        second_artifact=r8,
        first_reference={
            "role": "learned_second",
            "path": "r7/host-semantic.json",
            "sha256": _sha256_path(r7_path),
            "policy_order": "target,learned",
        },
        second_reference={
            "role": "learned_first",
            "path": "r8/host-semantic.json",
            "sha256": _sha256_path(r8_path),
            "policy_order": "learned,target",
        },
    )
    comparison_path = tmp_path / "comparison.json"
    _write_json(comparison_path, comparison)
    return comparison_path, repo_root


def _load_verifier():
    assert VERIFIER_PATH.exists(), f"missing verifier: {VERIFIER_PATH}"
    diagnostic = _load_module()
    sys.modules[
        "autoregressive_draft_host_semantic_diagnostic"
    ] = diagnostic
    spec = importlib.util.spec_from_file_location(
        "verify_autoregressive_draft_host_semantic_test_module",
        VERIFIER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_verifier_recomputes_campaign_from_raw_inputs(tmp_path):
    artifact_path, repo_root = _write_campaign_bundle(tmp_path)
    receipt = _load_verifier().verify_host_semantic_artifact(
        artifact_path,
        repo_root,
    )
    assert receipt == {
        "status": "PASS",
        "schema_version": 1,
        "classification": "ALIGNED_CAMPAIGN",
        "source_files_verified": 6,
        "input_files_verified": 6,
        "policy_repeat_coverage": {"target": 8, "learned": 8},
    }


@pytest.mark.parametrize(
    "target,match",
    (
        ("source", "source hash mismatch"),
        ("worker", "input hash mismatch"),
        ("host", "input hash mismatch"),
        ("summary", "recomputation mismatch"),
    ),
)
def test_verifier_rejects_campaign_tampering(tmp_path, target, match):
    artifact_path, repo_root = _write_campaign_bundle(tmp_path)
    _tamper_campaign_bundle(artifact_path, repo_root, target)
    with pytest.raises(ValueError, match=match):
        _load_verifier().verify_host_semantic_artifact(
            artifact_path,
            repo_root,
        )


def test_verifier_recomputes_cross_campaign_comparison(tmp_path):
    comparison_path, repo_root = _write_comparison_bundle(tmp_path)
    receipt = _load_verifier().verify_host_semantic_comparison(
        comparison_path,
        repo_root,
    )
    assert receipt["status"] == "PASS"
    assert receipt["schema_version"] == 1
    assert receipt["classification"] in {
        "HOST_PRESSURE_ASSOCIATED",
        "HOST_PRESSURE_NOT_SUPPORTED",
        "HOST_ALIGNMENT_INCONCLUSIVE",
    }
    assert receipt["campaign_artifacts_verified"] == 2


def test_comparison_verifier_rejects_tampered_campaign(tmp_path):
    comparison_path, repo_root = _write_comparison_bundle(tmp_path)
    comparison = json.loads(
        comparison_path.read_text(encoding="utf-8")
    )
    reference = comparison["campaign_artifacts"]["learned_first"]
    campaign_path = comparison_path.parent / reference["path"]
    campaign_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="campaign artifact hash mismatch"):
        _load_verifier().verify_host_semantic_comparison(
            comparison_path,
            repo_root,
        )


def test_verifier_main_writes_campaign_receipt(tmp_path):
    artifact_path, repo_root = _write_campaign_bundle(tmp_path)
    receipt_path = tmp_path / "receipts" / "campaign.json"

    status = _load_verifier().main([
        "--artifact", str(artifact_path),
        "--repo-root", str(repo_root),
        "--receipt", str(receipt_path),
    ])

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert status == 0
    assert receipt["classification"] == "ALIGNED_CAMPAIGN"


def test_verifier_main_prints_comparison_receipt(tmp_path, capsys):
    artifact_path, repo_root = _write_comparison_bundle(tmp_path)

    status = _load_verifier().main([
        "--artifact", str(artifact_path),
        "--repo-root", str(repo_root),
    ])

    receipt = json.loads(capsys.readouterr().out)
    assert status == 0
    assert receipt["campaign_artifacts_verified"] == 2
