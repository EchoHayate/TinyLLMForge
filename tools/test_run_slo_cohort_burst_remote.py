from __future__ import annotations

import io
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import weakref

import pytest

from tools import run_slo_cohort_burst_remote as remote


def _successful_cleanup_receipt() -> dict:
    return {
        "process_group_destroyed": True,
        "rank_exit_codes": [0],
        "owned_children_remaining": [],
        "rank_cleanup_receipts": [{
            "rank": 0,
            "process_group_destroyed": True,
        }],
    }


def _gpu(
    index: int,
    *,
    memory: int = 0,
    utilization: int = 0,
    processes=None,
) -> dict:
    return {
        "index": index,
        "uuid": f"GPU-{index}",
        "name": "NVIDIA A100 80GB PCIe",
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "compute_processes": [] if processes is None else processes,
    }


def test_direct_script_entrypoint_can_import_tools() -> None:
    result = subprocess.run(
        [sys.executable, str(Path(remote.__file__)), "--help"],
        cwd=Path(remote.__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_remote_paths_are_confined_to_approved_large_mount() -> None:
    paths = remote.build_remote_paths("20260913-stage0-r1")
    prefix = (
        "/data00/home/sitian/tinyllmforge-workspaces/"
        "command-timeline-20260818/slo-cohort-burst/"
    )
    assert set(paths) == {"staging", "primary", "controller"}
    assert all(path.startswith(prefix) for path in paths.values())
    assert all("/tmp/" not in path for path in paths.values())


@pytest.mark.parametrize(
    "tag",
    ("", "../escape", "/absolute", "white space", "a" * 129),
)
def test_remote_paths_reject_invalid_run_tags(tag: str) -> None:
    with pytest.raises(ValueError, match="run tag"):
        remote.build_remote_paths(tag)


def test_remote_runtime_prelude_redirects_every_cache_to_large_mount() -> None:
    paths = remote.build_remote_paths("20260913-stage0-r1")
    source = paths["staging"] + "/source"
    prelude = remote.build_remote_runtime_prelude(
        source=source,
        gpu_index=3,
        dist_port=23456,
    )
    runtime = paths["staging"] + "/runtime"
    for name in (
        "TMPDIR",
        "TMP",
        "TEMP",
        "PYTHONPYCACHEPREFIX",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "TORCH_EXTENSIONS_DIR",
    ):
        assert f"export {name}=" in prelude
    assert runtime in prelude
    assert "CUDA_VISIBLE_DEVICES=3" in prelude
    assert "MASTER_PORT=23456" in prelude
    assert "export TMPDIR=/tmp" not in prelude


def test_kerberos_guard_is_short_enough_for_fast_gpu_claim() -> None:
    assert remote.MINIMUM_KERBEROS_LIFETIME_SECONDS == 1_800


def test_strict_clean_a100_requires_zero_processes_and_low_usage() -> None:
    rows = [
        _gpu(0),
        _gpu(1, memory=1_025),
        _gpu(2, utilization=6),
        _gpu(3, processes=[{"pid": 7, "process_name": "python"}]),
        {**_gpu(4), "name": "NVIDIA H100 80GB HBM3"},
    ]

    assert remote.strict_clean_a100s(rows) == [rows[0]]


def test_committed_source_archive_contains_only_qualification_sources(
    tmp_path: Path,
) -> None:
    for relative in remote.SOURCE_FILES:
        path = tmp_path / relative
        if Path(relative).suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative + "\n")
        else:
            path.mkdir(parents=True, exist_ok=True)
            (path / "__init__.py").write_text(relative + "\n")
    unrelated = tmp_path / "experiments/raw.trace"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("large\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "add", "--", "."],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Remote Test",
            "-c",
            "user.email=remote@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()

    payload = remote.committed_source_archive(tmp_path, commit)

    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as bundle:
        names = {member.name for member in bundle.getmembers()}
    assert "source/experiments/raw.trace" not in names
    assert (
        "source/tools/run_staged_inference_benchmark_remote.py"
        in names
    )
    assert all(
        any(
            name == "source/" + relative
            or name.startswith("source/" + relative + "/")
            for name in names
        )
        for relative in remote.SOURCE_FILES
    )


def test_worker_plan_runs_profiler_and_remote_verifier_mounted_only() -> None:
    paths = remote.build_remote_paths("20260913-stage0-plan-r1")
    plan = remote.build_worker_plan(
        paths=paths,
        run_tag="20260913-stage0-plan-r1",
        source_commit="a" * 40,
        gpu=_gpu(2),
    )
    joined = "\n".join(plan["commands"])

    assert "tools.profile_slo_cohort_burst_ceiling" in joined
    assert "--mode run" in joined
    assert "--mode verify" in joined
    assert "CUDA_VISIBLE_DEVICES=2" in joined
    assert paths["primary"] in joined
    assert paths["controller"] in joined
    assert "export TMPDIR=/tmp" not in joined
    assert "cd /tmp/" not in joined
    assert "/private/tmp" not in joined
    assert "TinyLLMForge-adaptive-ngram" not in joined
    assert "pkill" not in joined
    assert "killall" not in joined


@pytest.mark.parametrize("stage", ("correctness", "canonical"))
def test_qualification_worker_plan_runs_stage_and_independent_verifier(
    stage: str,
) -> None:
    tag = f"20260914-{stage}-plan-r1"
    paths = remote.build_remote_paths(tag)
    plan = remote.build_worker_plan(
        paths=paths,
        run_tag=tag,
        source_commit="a" * 40,
        gpu=_gpu(2),
        stage=stage,
    )
    joined = "\n".join(plan["commands"])

    assert plan["stage"] == stage
    assert "tools.run_slo_cohort_burst_remote" in joined
    assert f"--worker-stage {stage}" in joined
    assert joined.count(" --tag ") == 1
    assert "tools.slo_cohort_burst_verify" in joined
    assert paths["primary"] + "/final_bundle" in joined
    assert paths["controller"] + "/remote_verify.json" in joined
    assert "profile_slo_cohort_burst_ceiling" not in joined
    assert "/tmp/" not in joined
    assert "pkill" not in joined
    assert "killall" not in joined


def test_stage_artifact_contracts_are_separate_and_closed() -> None:
    ceiling = remote.stage_artifact_contract("ceiling")
    correctness = remote.stage_artifact_contract("correctness")
    canonical = remote.stage_artifact_contract("canonical")

    assert "cost_table.json" in ceiling["required"]
    assert "final_bundle/manifest.json" in correctness["required"]
    assert "final_bundle/cost_profile_rows.jsonl" in (
        correctness["required"]
    )
    assert "final_bundle/correctness_rows.jsonl" in (
        correctness["required"]
    )
    assert "final_bundle/request_rows.jsonl" in canonical["required"]
    assert "final_bundle/cost_profile_rows.jsonl" in canonical["required"]
    assert "final_bundle/decision_rows.jsonl" in canonical["required"]
    assert "final_bundle/execution_rows.jsonl" in canonical["required"]
    assert (
        "final_bundle/canonical_graph_identities.json"
        in canonical["required"]
    )
    assert "final_bundle/summary.json" in canonical["required"]
    assert "final_bundle/remote_verify.json" in canonical["required"]
    assert correctness["local_root"].name == "slo_cohort_burst"
    assert canonical["local_root"].name == "slo_cohort_burst"
    assert ceiling["required"] != canonical["required"]


def test_parse_args_accepts_all_frozen_stages() -> None:
    for stage in ("ceiling", "correctness", "canonical"):
        args = remote.parse_args([
            "--stage",
            stage,
            "--tag",
            f"20260914-{stage}-r1",
        ])
        assert args.stage == stage


def test_parse_args_rejects_worker_output_path_traversal() -> None:
    tag = "20260913-correctness-worker-path-r1"
    with pytest.raises(SystemExit):
        remote.parse_args([
            "--worker-stage",
            "correctness",
            "--tag",
            tag,
            "--source-commit",
            "a" * 40,
            "--output-dir",
            remote.TASK_REMOTE_ROOT
            + f"/runs/{tag}/../../outside-approved-root",
        ])


def test_main_dispatches_worker_stage_without_entering_controller(
    monkeypatch,
    capsys,
) -> None:
    calls = []
    output_dir = (
        remote.TASK_REMOTE_ROOT
        + "/runs/20260913-correctness-worker-r1"
    )

    def fake_worker(args):
        calls.append(("worker", args.worker_stage, args.output_dir))
        return {
            "status": "COMPLETE",
            "stage": args.worker_stage,
        }

    def unexpected_controller(_args):
        raise AssertionError("worker mode entered controller")

    monkeypatch.setattr(remote, "run_qualification_worker", fake_worker)
    monkeypatch.setattr(remote, "run_controller", unexpected_controller)

    exit_code = remote.main([
        "--worker-stage",
        "correctness",
        "--tag",
        "20260913-correctness-worker-r1",
        "--source-commit",
        "a" * 40,
        "--output-dir",
        output_dir,
    ])

    assert exit_code == 0
    assert calls == [("worker", "correctness", output_dir)]
    assert json.loads(capsys.readouterr().out)["stage"] == "correctness"


def test_frozen_arrival_traces_are_open_loop_balanced_and_deterministic() -> None:
    saturation = {
        "decode_heavy": 20.0,
        "mixed": 10.0,
        "bursty_eos": 25.0,
    }

    first = remote.build_frozen_arrival_traces(
        source_commit="a" * 40,
        saturation_rps_by_workload=saturation,
    )
    second = remote.build_frozen_arrival_traces(
        source_commit="a" * 40,
        saturation_rps_by_workload=saturation,
    )

    assert first == second
    assert first["minimum_repetitions"] == 5
    assert first["minimum_requests_per_workload_load_arm"] == 128
    assert first["arm_order_by_repetition"][:2] == [
        ["baseline", "candidate"],
        ["candidate", "baseline"],
    ]
    assert len(first["cases"]) == 45
    mixed = [
        request["prompt_tokens"]
        for case in first["cases"]
        if case["workload"] == "mixed"
        for request in case["requests"]
    ]
    assert {
        prompt_tokens: mixed.count(prompt_tokens)
        for prompt_tokens in (256, 2048, 8192)
    } == {256: 273, 2048: 78, 8192: 39}
    assert all(
        request["ignore_eos"] is False
        for case in first["cases"]
        if case["workload"] == "bursty_eos"
        for request in case["requests"]
    )
    for case in first["cases"]:
        offsets = [
            request["arrival_offset_ns"]
            for request in case["requests"]
        ]
        assert offsets == sorted(offsets)
        assert len({
            request["request_id"] for request in case["requests"]
        }) == 26


def test_qualification_config_keeps_the_baseline_free_of_single_request_burst(
    tmp_path: Path,
) -> None:
    candidate = remote._qualification_engine_config(
        cost_table_path=tmp_path / "cost_table.json",
    )
    baseline = remote._qualification_engine_config(
        cost_table_path=tmp_path / "cost_table.json",
        cohort_enabled=False,
    )

    assert candidate["exact_greedy_decode_burst"] is True
    assert candidate["exact_greedy_cohort_burst"] is True
    assert candidate["autoregressive_draft_command_timeline"] is False
    assert baseline["exact_greedy_decode_burst"] is False
    assert baseline["exact_greedy_cohort_burst"] is False
    assert baseline["autoregressive_draft_command_timeline"] is False


def test_qualification_graph_capture_covers_exact_runtime_shapes() -> None:
    calls = []

    class ModelRunner:
        def capture_exact_greedy_cohort_burst_graph(
            self,
            batch_size,
            *,
            block_table_width,
            correctness_trace=False,
        ):
            calls.append((
                batch_size,
                block_table_width,
                correctness_trace,
            ))
            return object()

    remote._capture_qualification_graphs(
        SimpleNamespace(model_runner=ModelRunner()),
    )

    expected = {
        (batch_size, block_table_width, False)
        for batch_size in range(1, 9)
        for block_table_width in (2, 9, 33)
    } | {
        (batch_size, 2, True)
        for batch_size in (1, 2, 4, 8)
    }
    assert set(calls) == expected
    assert len(calls) == len(expected)


def test_qualification_graph_identity_inventory_is_shape_bound() -> None:
    class ModelRunner:
        def exact_greedy_cohort_burst_capability(
            self,
            *,
            batch_size,
            block_table_width,
            correctness_trace=False,
        ):
            suffix = (
                batch_size * 100
                + block_table_width * 2
                + int(correctness_trace)
            )
            return {
                "shape_supported": True,
                "graph_identity_sha256": f"{suffix:064x}",
            }

    identities = remote._graph_identity_by_shape(
        SimpleNamespace(model_runner=ModelRunner()),
    )

    assert set(identities) == {
        remote._cohort_graph_shape_key(
            batch_size=batch_size,
            block_table_width=block_table_width,
            correctness_trace=correctness_trace,
        )
        for batch_size, block_table_width, correctness_trace
        in remote.QUALIFICATION_GRAPH_SHAPES
    }
    assert len(set(identities.values())) == len(identities)


def test_qualification_environment_binds_exact_graph_shapes() -> None:
    source_identity = {
        "source_commit": "a" * 40,
        "model": "Qwen3-0.6B",
        "checkpoint_sha256": "b" * 64,
        "gpu_uuid": "GPU-0",
        "gpu_name": "NVIDIA A100 80GB PCIe",
        "tensor_parallel_size": 1,
    }
    graph_identities = {
        "b2-w2-trace0": "c" * 64,
        "b2-w2-trace1": "d" * 64,
    }

    environment = remote._environment(
        source_identity=source_identity,
        graph_identity_sha256_by_shape=graph_identities,
        eos_token_id=2,
    )

    assert environment["schema_version"] == (
        "slo-cohort-burst.environment.v2"
    )
    assert (
        environment["graph_identity_sha256_by_shape"]
        == graph_identities
    )
    assert "graph_identity_sha256_by_batch" not in environment


def test_cohort_arm_switches_base_exact_burst_with_candidate_state():
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(
            exact_greedy_cohort_burst=False,
            exact_greedy_cohort_burst_widths=(),
        ),
        model_runner=SimpleNamespace(
            config=SimpleNamespace(
                exact_greedy_decode_burst=True,
                exact_greedy_cohort_burst=True,
            ),
        ),
    )

    remote._set_cohort_arm(engine, enabled=False)

    assert engine.scheduler.exact_greedy_cohort_burst is True
    assert engine.model_runner.config.exact_greedy_decode_burst is False
    assert engine.model_runner.config.exact_greedy_cohort_burst is False

    remote._set_cohort_arm(engine, enabled=True)

    assert engine.model_runner.config.exact_greedy_decode_burst is True
    assert engine.model_runner.config.exact_greedy_cohort_burst is True


def test_correctness_matrix_executes_every_bxk_pair_in_baseline_candidate_order(
) -> None:
    calls = []

    def run_case(*, batch_size, burst_width, arm):
        calls.append((batch_size, burst_width, arm))
        rows = [{
            "output_token_ids": [row_index] * burst_width,
            "output_text_sha256": f"text-{row_index}",
            "sampled_logits_sha256": f"logits-{row_index}",
            "argmax_token_ids": [row_index] * burst_width,
        } for row_index in range(batch_size)]
        return {
            "rows": rows,
            "duplicate_forwards": 0,
            "duplicate_commits": 0,
            "unauthorized_kv_publications": 0,
            "pending_leases_after_case": 0,
            "execution_evidence": (
                {
                    "lease_identity_sha256": (
                        f"{batch_size * 100 + burst_width:064x}"
                    ),
                }
                if arm == "candidate" and burst_width > 1
                else None
            ),
        }

    rows = remote.build_correctness_matrix(run_case=run_case)

    assert len(rows) == 16
    assert calls[:4] == [
        (1, 1, "baseline"),
        (1, 1, "candidate"),
        (1, 2, "baseline"),
        (1, 2, "candidate"),
    ]
    assert calls[-2:] == [
        (8, 8, "baseline"),
        (8, 8, "candidate"),
    ]
    assert rows[-1]["batch_size"] == 8
    assert rows[-1]["burst_width"] == 8
    assert len(rows[-1]["rows"]) == 8
    assert rows[-1]["schema_version"] == (
        "slo-cohort-burst.correctness-case.v2"
    )
    assert rows[-1]["candidate_execution"] == {
        "lease_identity_sha256": f"{808:064x}",
    }
    assert rows[0]["candidate_execution"] is None


def test_canonical_matrix_obeys_frozen_paired_arm_order() -> None:
    traces = {
        "schema_version": "slo-cohort-burst.arrival-traces.v1",
        "minimum_requests_per_workload_load_arm": 1,
        "minimum_repetitions": 2,
        "arm_order_by_repetition": [
            ["baseline", "candidate"],
            ["candidate", "baseline"],
        ],
        "cases": [
            {
                "workload": workload,
                "load": "low",
                "repetition": repetition,
                "requests": [{
                    "request_id": f"{workload}-q{repetition}",
                }],
            }
            for repetition in range(2)
            for workload in ("decode_heavy", "mixed")
        ],
    }
    calls = []

    def run_case(*, trace_case, arm):
        calls.append((trace_case["repetition"], arm))
        return {
            "request_rows": [{"arm": arm}],
            "decision_rows": [{"arm": arm}] if arm == "candidate" else [],
            "execution_rows": [{"arm": arm}] if arm == "candidate" else [],
        }

    rows = remote.run_canonical_matrix(
        arrival_traces=traces,
        run_case=run_case,
    )

    assert calls == [
        (0, "baseline"),
        (0, "baseline"),
        (0, "candidate"),
        (0, "candidate"),
        (1, "candidate"),
        (1, "candidate"),
        (1, "baseline"),
        (1, "baseline"),
    ]
    assert len(rows["request_rows"]) == 8
    assert len(rows["decision_rows"]) == 4
    assert len(rows["execution_rows"]) == 4


def test_release_qualification_engine_requires_proven_engine_exit() -> None:
    calls = []

    class Engine:
        def exit(self):
            calls.append("exit")
            return _successful_cleanup_receipt()

    assert remote._release_qualification_engine(Engine()) is None
    assert calls == ["exit"]


@pytest.mark.parametrize(
    "receipt",
    (
        None,
        {
            "process_group_destroyed": False,
            "rank_exit_codes": [0],
            "owned_children_remaining": [],
        },
        {
            "process_group_destroyed": True,
            "rank_exit_codes": [],
            "owned_children_remaining": [],
        },
        {
            "process_group_destroyed": True,
            "rank_exit_codes": [1],
            "owned_children_remaining": [],
        },
        {
            "process_group_destroyed": True,
            "rank_exit_codes": [0],
            "owned_children_remaining": [123],
        },
    ),
)
def test_release_qualification_engine_rejects_incomplete_cleanup(
    receipt,
) -> None:
    engine = SimpleNamespace(exit=lambda: receipt)

    with pytest.raises(RuntimeError, match="cleanup"):
        remote._release_qualification_engine(engine)


def test_evidence_tap_retains_k1_fallback_decision() -> None:
    tap = object.__new__(remote._CohortEvidenceTap)
    tap._lease = None
    tap._result = None
    tap._publication = None
    decision = {
        "schema_version": "slo-cohort-burst.decision.v1",
        "selected_width": 1,
        "reason": "predicted_cost_exceeds_slack",
    }

    captured = tap.take(
        case={
            "workload": "decode_heavy",
            "load": "high",
            "repetition": 0,
            "arm": "candidate",
        },
        observation={
            "slo_cohort_decision_telemetry": decision,
            "exact_greedy_cohort_burst_execution_telemetry": None,
        },
    )

    assert captured == (
        {
            "schema_version": "slo-cohort-burst.decision-evidence.v1",
            "case": {
                "workload": "decode_heavy",
                "load": "high",
                "repetition": 0,
                "arm": "candidate",
            },
            "decision": decision,
        },
        None,
    )


def test_canonical_matrix_uses_separate_engine_per_repetition_arm(
    monkeypatch,
) -> None:
    traces = {
        "schema_version": "slo-cohort-burst.arrival-traces.v1",
        "minimum_requests_per_workload_load_arm": 1,
        "minimum_repetitions": 2,
        "arm_order_by_repetition": [
            ["baseline", "candidate"],
            ["candidate", "baseline"],
        ],
        "cases": [
            {
                "workload": workload,
                "load": "low",
                "repetition": repetition,
                "requests": [{"request_id": f"{workload}-{repetition}"}],
            }
            for repetition in range(2)
            for workload in ("decode_heavy", "mixed")
        ],
    }
    created = []
    released = []
    calls = []

    def engine_factory(*, arm):
        engine = SimpleNamespace(
            name=f"{arm}-{len(created)}",
            scheduler=SimpleNamespace(
                exact_greedy_cohort_burst=False,
                exact_greedy_cohort_burst_widths=(),
            ),
            model_runner=SimpleNamespace(
                config=SimpleNamespace(
                    exact_greedy_cohort_burst=False,
                ),
            ),
        )
        created.append((arm, engine))
        return engine

    monkeypatch.setattr(
        remote,
        "_graph_identity_by_shape",
        lambda engine, **_kwargs: {
            "b1-w2-trace0": (
                "a" * 64
                if engine.name == "candidate-1"
                else "b" * 64
            ),
        },
    )
    monkeypatch.setattr(
        remote,
        "_capture_canonical_graphs",
        lambda _engine: None,
    )
    monkeypatch.setattr(
        remote,
        "_release_qualification_engine",
        lambda engine: released.append(engine.name),
    )
    monkeypatch.setattr(
        remote,
        "_run_open_loop_case",
        lambda *, engine, trace_case, arm, **_kwargs: (
            calls.append((
                engine.name,
                trace_case["repetition"],
                arm,
            ))
            or {
                "request_rows": [{"engine": engine.name}],
                "decision_rows": [],
                "execution_rows": [],
            }
        ),
    )

    result = remote._run_canonical_matrix_with_engine_factory(
        engine_factory=engine_factory,
        sampling_params_factory=object,
        source_commit="a" * 40,
        arrival_traces=traces,
    )

    assert [arm for arm, _engine in created] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]
    assert [name for name, _rep, _arm in calls] == [
        "baseline-0",
        "baseline-0",
        "candidate-1",
        "candidate-1",
        "candidate-2",
        "candidate-2",
        "baseline-3",
        "baseline-3",
    ]
    assert released == [
        "baseline-0",
        "candidate-1",
        "candidate-2",
        "baseline-3",
    ]
    assert result["graph_identity_sha256_by_repetition"] == {
        "0": {"b1-w2-trace0": "a" * 64},
        "1": {"b1-w2-trace0": "b" * 64},
    }


def test_canonical_matrix_drops_previous_engine_before_next_creation(
    monkeypatch,
) -> None:
    class Engine:
        pass

    traces = {
        "schema_version": "slo-cohort-burst.arrival-traces.v1",
        "minimum_requests_per_workload_load_arm": 1,
        "minimum_repetitions": 1,
        "arm_order_by_repetition": [["baseline", "candidate"]],
        "cases": [{
            "workload": "decode_heavy",
            "load": "low",
            "repetition": 0,
            "requests": [{"request_id": "decode-heavy-0"}],
        }],
    }
    prior_engines = []

    def engine_factory(*, arm):
        assert all(reference() is None for reference in prior_engines)
        engine = Engine()
        engine.exit = _successful_cleanup_receipt
        engine.scheduler = SimpleNamespace(
            exact_greedy_cohort_burst=False,
            exact_greedy_cohort_burst_widths=(),
        )
        engine.model_runner = SimpleNamespace(
            config=SimpleNamespace(
                exact_greedy_cohort_burst=False,
            ),
        )
        prior_engines.append(weakref.ref(engine))
        return engine

    monkeypatch.setattr(
        remote,
        "_graph_identity_by_shape",
        lambda _engine, **_kwargs: {
            "b1-w2-trace0": "a" * 64,
        },
    )
    monkeypatch.setattr(
        remote,
        "_capture_canonical_graphs",
        lambda _engine: None,
    )
    monkeypatch.setattr(
        remote,
        "_run_open_loop_case",
        lambda **_kwargs: {
            "request_rows": [],
            "decision_rows": [],
            "execution_rows": [],
        },
    )

    remote._run_canonical_matrix_with_engine_factory(
        engine_factory=engine_factory,
        sampling_params_factory=object,
        source_commit="a" * 40,
        arrival_traces=traces,
    )

    assert all(reference() is None for reference in prior_engines)


def test_correctness_worker_builds_and_seals_source_bound_bundle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_identity = {
        "source_commit": "a" * 40,
        "source_patch_sha256": "b" * 64,
        "model": "Qwen3-0.6B",
        "checkpoint_sha256": "c" * 64,
        "gpu_uuid": "GPU-0",
        "gpu_name": "NVIDIA A100 80GB PCIe",
        "tensor_parallel_size": 1,
        "dtype": "torch.bfloat16",
        "config_sha256": "d" * 64,
    }
    cost_rows = [{"row": "cost"}]
    cost_table = {
        "source_identity": source_identity,
        "table_sha256": "e" * 64,
    }
    correctness_rows = [{"case": "correctness"}]
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(eos=2),
        exit=_successful_cleanup_receipt,
    )
    calls = []
    monkeypatch.setattr(
        remote,
        "_run_source_bound_calibration",
        lambda **_kwargs: (
            source_identity,
            cost_rows,
            cost_table,
        ),
    )
    monkeypatch.setattr(
        remote,
        "_create_qualification_engine",
        lambda **_kwargs: engine,
    )
    monkeypatch.setattr(
        remote,
        "_qualification_sampling_params_factory",
        lambda: object,
    )
    monkeypatch.setattr(
        remote,
        "_capture_qualification_graphs",
        lambda value: calls.append(("capture", value)),
    )
    monkeypatch.setattr(
        remote,
        "_graph_identity_by_shape",
        lambda _engine, **_kwargs: {
            "b1-w2-trace0": "f" * 64,
        },
    )
    monkeypatch.setattr(
        remote,
        "_run_correctness_matrix_on_engine",
        lambda **_kwargs: correctness_rows,
    )
    monkeypatch.setattr(
        remote,
        "_source_manifest",
        lambda **_kwargs: {"source": "manifest"},
    )
    monkeypatch.setattr(
        remote,
        "_environment",
        lambda **_kwargs: {"environment": "identity"},
    )
    monkeypatch.setattr(
        remote,
        "validate_qualification_output_dir",
        lambda value, **_kwargs: Path(value),
    )

    def write_bundle(**kwargs):
        calls.append(("write", kwargs))
        destination = Path(kwargs["output_dir"]) / "final_bundle"
        destination.mkdir(parents=True)
        return destination

    monkeypatch.setattr(remote, "write_qualification_bundle", write_bundle)
    args = SimpleNamespace(
        worker_stage="correctness",
        tag="20260913-correctness-worker-r2",
        source_commit="a" * 40,
        model="/models/qwen",
        output_dir=str(tmp_path / "attempt"),
    )

    result = remote.run_qualification_worker(args)

    assert result == {
        "status": "COMPLETE",
        "stage": "correctness",
        "run_tag": "20260913-correctness-worker-r2",
        "source_commit": "a" * 40,
        "final_bundle": str(tmp_path / "attempt" / "final_bundle"),
        "correctness_case_count": 1,
    }
    assert calls[0] == ("capture", engine)
    artifacts = calls[1][1]["artifacts"]
    assert artifacts == {
        "source_manifest.json": {"source": "manifest"},
        "environment.json": {"environment": "identity"},
        "cost_profile_rows.jsonl": cost_rows,
        "cost_table.json": cost_table,
        "correctness_rows.jsonl": correctness_rows,
    }


class _CorrectnessEngine:
    def __init__(self):
        self.scheduler = SimpleNamespace(
            exact_greedy_cohort_burst=True,
            exact_greedy_cohort_burst_widths=(1, 2, 4, 8),
            exact_greedy_cohort_burst_target_itl_ns=40_000_000,
            exact_greedy_cohort_burst_target_ttft_ns=1_000_000_000,
            exact_greedy_cohort_burst_reserve_ns=2_000_000,
        )
        self.model_runner = SimpleNamespace(
            enforce_eager=False,
            config=SimpleNamespace(exact_greedy_cohort_burst=True),
        )
        self.tokenizer = SimpleNamespace(
            decode=lambda tokens: ",".join(map(str, tokens)),
        )
        self.last_step_observation = None
        self._sequence_ids = []
        self._step = 0
        self._finished = True
        self._logits = None
        self._target_steps = None
        self.recording = []
        self.enforce_eager_states = []

    def is_finished(self):
        return self._finished

    def enable_step_logits_authority_recording(self, enabled, **_kwargs):
        self.recording.append(enabled)

    def add_request(self, _prompt, sampling):
        sequence_id = 10 + len(self._sequence_ids)
        self._sequence_ids.append(sequence_id)
        self._target_steps = sampling["max_tokens"]
        self._finished = False
        return sequence_id

    def step(self, **_kwargs):
        self.enforce_eager_states.append(
            self.model_runner.enforce_eager
        )
        self._step += 1
        if self._step == 1:
            deltas = {
                sequence_id: [0]
                for sequence_id in self._sequence_ids
            }
            outputs = []
        else:
            ordinal = self._step - 2
            deltas = {
                sequence_id: [row_index + ordinal + 1]
                for row_index, sequence_id in enumerate(
                    self._sequence_ids
                )
            }
            self._logits = []
            for token in deltas.values():
                logits = [0.0] * 4
                logits[token[0]] = 1.0
                self._logits.append(logits)
            outputs = []
            if self._step == self._target_steps:
                self._finished = True
                outputs = [
                    (sequence_id, [0, row_index + 1, row_index + 2])
                    for row_index, sequence_id in enumerate(
                        self._sequence_ids
                    )
                ]
        self.last_step_observation = {
            "new_completion_tokens_by_seq": deltas,
        }
        return outputs, len(deltas)

    def read_step_logits_authority(self):
        return self._logits


def test_correctness_case_collects_ordered_k1_logits_and_tokens() -> None:
    engine = _CorrectnessEngine()

    result = remote._run_correctness_case(
        engine=engine,
        sampling_params_factory=lambda **kwargs: kwargs,
        source_commit="a" * 40,
        batch_size=2,
        burst_width=2,
        arm="baseline",
    )

    assert [row["output_token_ids"] for row in result["rows"]] == [
        [1, 2],
        [2, 3],
    ]
    assert [row["argmax_token_ids"] for row in result["rows"]] == [
        [1, 2],
        [2, 3],
    ]
    assert engine.recording == [True, False]
    assert result["pending_leases_after_case"] == 0


def test_correctness_baseline_forces_eager_only_during_case() -> None:
    engine = _CorrectnessEngine()

    remote._run_correctness_case(
        engine=engine,
        sampling_params_factory=lambda **kwargs: kwargs,
        source_commit="a" * 40,
        batch_size=1,
        burst_width=2,
        arm="baseline",
    )

    assert engine.enforce_eager_states == [True, True, True]
    assert engine.model_runner.enforce_eager is False


def test_correctness_k1_baseline_preserves_runtime_execution_mode() -> None:
    engine = _CorrectnessEngine()

    remote._run_correctness_case(
        engine=engine,
        sampling_params_factory=lambda **kwargs: kwargs,
        source_commit="a" * 40,
        batch_size=1,
        burst_width=1,
        arm="baseline",
    )

    assert engine.enforce_eager_states == [False, False]
    assert engine.model_runner.enforce_eager is False


def test_correctness_candidate_width_contract_retains_k1(
    monkeypatch,
) -> None:
    engine = _CorrectnessEngine()
    captured_widths = []

    class StopAfterArmConfiguration(Exception):
        pass

    def capture_arm(_engine, *, enabled, widths):
        assert enabled is True
        captured_widths.append(tuple(widths))
        raise StopAfterArmConfiguration

    monkeypatch.setattr(remote, "_set_cohort_arm", capture_arm)

    with pytest.raises(StopAfterArmConfiguration):
        remote._run_correctness_case(
            engine=engine,
            sampling_params_factory=lambda **kwargs: kwargs,
            source_commit="a" * 40,
            batch_size=2,
            burst_width=4,
            arm="candidate",
        )

    assert captured_widths == [(1, 2, 4)]


def test_publication_authority_accepts_tuple_result_tokens() -> None:
    evidence = {
        "result": {
            "rows": [{
                "sequence_id": 10,
                "tokens": (101, 102, 103, 104),
            }],
        },
        "publication": {
            "ordered_sequence_ids": [10],
            "rows": [{
                "sequence_id": 10,
                "commit_tokens": [101, 102, 103, 104],
            }],
        },
    }

    assert (
        remote._count_unauthorized_kv_publications(evidence)
        == 0
    )


class _OpenLoopEngine:
    def __init__(self):
        self.scheduler = SimpleNamespace(
            exact_greedy_cohort_burst=True,
            exact_greedy_cohort_burst_widths=(1, 2, 4, 8),
        )
        self.model_runner = SimpleNamespace(
            config=SimpleNamespace(exact_greedy_cohort_burst=True),
        )
        self.last_step_observation = None
        self._finished = True
        self._sequence_id = 41
        self.arrival_ns = None

    def is_finished(self):
        return self._finished

    def add_request(self, _prompt, _sampling, *, arrival_ns=None):
        self.arrival_ns = arrival_ns
        self._finished = False
        return self._sequence_id

    def step(self, **_kwargs):
        self._finished = True
        self.last_step_observation = {
            "slo_cohort_telemetry_error": None,
            "memory": {
                "cuda_reserved_bytes": 1_024,
                "cuda_peak_reserved_bytes": 2_048,
            },
            "slo_cohort_request_telemetry": [{
                "schema_version": "slo-cohort-burst.request.v1",
                "request_id": str(self._sequence_id),
                "sequence_id": self._sequence_id,
                "service_class": "default",
                "arrival_ns": 100,
                "prefill_start_ns": 110,
                "prefill_complete_ns": 120,
                "first_token_visible_ns": 130,
                "token_visible_ns": [130, 140],
                "completion_ns": 140,
                "output_token_ids": [7, 8],
                "output_text_sha256": "f" * 64,
                "terminal_reason": "length",
            }],
        }
        return [(self._sequence_id, [7, 8])], 2


def test_open_loop_case_binds_frozen_request_identity_and_memory() -> None:
    source_commit = "a" * 40
    prompt = remote._qualification_prompt_tokens(
        source_commit=source_commit,
        workload="decode_heavy",
        ordinal=0,
        prompt_tokens=4,
    )
    trace_case = {
        "workload": "decode_heavy",
        "load": "low",
        "repetition": 0,
        "requests": [{
            "request_id": "decode-heavy-low-r0-q0",
            "prompt_sha256": hashlib.sha256(
                remote._canonical_json_bytes(prompt).rstrip(b"\n")
            ).hexdigest(),
            "arrival_offset_ns": 0,
            "prompt_tokens": 4,
            "maximum_output_tokens": 2,
            "ignore_eos": True,
        }],
    }
    now = iter(range(0, 1_000, 10)).__next__

    engine = _OpenLoopEngine()
    result = remote._run_open_loop_case(
        engine=engine,
        sampling_params_factory=lambda **kwargs: kwargs,
        source_commit=source_commit,
        trace_case=trace_case,
        arm="baseline",
        clock_ns=now,
        sleep=lambda _seconds: None,
    )

    assert result["decision_rows"] == []
    assert result["execution_rows"] == []
    assert len(result["request_rows"]) == 1
    wrapper = result["request_rows"][0]
    assert wrapper["request"]["request_id"] == (
        "decode-heavy-low-r0-q0"
    )
    assert wrapper["peak_cuda_reserved_bytes"] == 2_048
    assert wrapper["case"]["arm"] == "baseline"
    assert engine.arrival_ns == 0


def test_runner_summary_matches_independent_reconstruction_fixture() -> None:
    from tools.test_slo_cohort_burst_verify import (
        complete_synthetic_bundle,
    )

    bundle = complete_synthetic_bundle()
    summary = remote._build_canonical_summary(
        request_rows=bundle["request_rows"],
        execution_rows=bundle["execution_rows"],
        correctness_rows=bundle["correctness_rows"],
    )

    assert summary == bundle["summary"]


def test_write_qualification_bundle_closes_manifest(
    tmp_path: Path,
) -> None:
    artifacts = {
        relative: (
            [{"row": relative}]
            if relative.endswith(".jsonl")
            else {"artifact": relative}
        )
        for relative in (
            remote.QUALIFICATION_AUTHORITATIVE_FILES
            - {"manifest.json"}
        )
    }

    final_bundle = remote.write_qualification_bundle(
        output_dir=tmp_path / "attempt",
        stage="canonical",
        artifacts=artifacts,
    )

    manifest = json.loads(
        (final_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest["artifact_sha256"]) == set(artifacts)
    for relative, expected in manifest["artifact_sha256"].items():
        assert hashlib.sha256(
            (final_bundle / relative).read_bytes()
        ).hexdigest() == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("raw_rows.jsonl", True),
        ("cost_table.json", True),
        ("ceiling_summary.json", True),
        ("source_manifest.json", True),
        ("remote_verify.json", True),
        ("runner.log", True),
        ("runtime/hf-cache/blob", False),
        ("raw/nsys.sqlite", False),
    ),
)
def test_compact_artifact_allowlist(name: str, expected: bool) -> None:
    assert remote.is_compact_artifact(name) is expected


@pytest.mark.parametrize(
    "name",
    (
        "final_bundle/source_manifest.json",
        "final_bundle/environment.json",
        "final_bundle/arrival_traces.json",
        "final_bundle/cost_table.json",
        "final_bundle/decision_rows.jsonl",
        "final_bundle/execution_rows.jsonl",
        "final_bundle/request_rows.jsonl",
        "final_bundle/correctness_rows.jsonl",
        "final_bundle/summary.json",
        "final_bundle/manifest.json",
        "final_bundle/remote_verify.json",
        "runner.log",
    ),
)
def test_canonical_compact_artifact_allowlist(name: str) -> None:
    assert remote.is_compact_artifact(name, stage="canonical") is True


def test_canonical_compact_artifact_rejects_large_or_local_receipts() -> None:
    for name in (
        "raw/nsys.sqlite",
        "runtime/hf-cache/blob",
        "final_bundle/local_verify.json",
        "final_bundle/report.md",
    ):
        assert remote.is_compact_artifact(
            name,
            stage="canonical",
        ) is False


def test_dual_verifier_agreement_is_fail_closed(tmp_path: Path) -> None:
    final_bundle = tmp_path / "final_bundle"
    final_bundle.mkdir()
    (final_bundle / "remote_verify.json").write_text(
        json.dumps({
            "verified": True,
            "classification": "NO_GO_THROUGHPUT",
        })
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="verifier disagreement"):
        remote.validate_dual_verifier_agreement(
            tmp_path,
            {
                "verified": True,
                "classification": (
                    "GO_SLO_AWARE_COHORT_DECODE_BURST"
                ),
            },
            stage="canonical",
        )


def test_resume_receipt_requires_exact_source_and_terminal_hashes() -> None:
    paths = remote.build_remote_paths("20260913-stage0-resume-r1")
    receipt = {
        "schema_version": "slo-cohort-burst.remote-resume.v1",
        "status": "COMPLETE",
        "stage": "ceiling",
        "run_tag": "20260913-stage0-resume-r1",
        "source_commit": "a" * 40,
        "remote_paths": paths,
        "artifact_sha256": {
            name: str(index) * 64
            for index, name in enumerate(
                sorted(remote.REQUIRED_TERMINAL_FILES),
                start=1,
            )
        },
    }

    assert remote.validate_resume_receipt(
        receipt,
        run_tag="20260913-stage0-resume-r1",
        source_commit="a" * 40,
        paths=paths,
    )["status"] == "COMPLETE"

    mutated = json.loads(json.dumps(receipt))
    mutated["source_commit"] = "b" * 40
    with pytest.raises(ValueError, match="source identity"):
        remote.validate_resume_receipt(
            mutated,
            run_tag="20260913-stage0-resume-r1",
            source_commit="a" * 40,
            paths=paths,
        )


def test_wait_for_clean_a100_retries_transient_remote_failure(
    monkeypatch,
) -> None:
    calls = []
    selected = _gpu(3)
    monkeypatch.setattr(
        remote,
        "validate_kerberos",
        lambda **_kwargs: calls.append("kerberos"),
    )

    def query():
        calls.append("query")
        if calls.count("query") == 1:
            raise RuntimeError("transient SSH failure")
        return [selected]

    monkeypatch.setattr(remote.base, "query_remote_gpu_rows", query)
    monkeypatch.setattr(remote.time, "sleep", lambda _seconds: None)
    monotonic = iter((0.0, 0.0, 0.0)).__next__
    monkeypatch.setattr(remote.time, "monotonic", monotonic)

    inventory, gpu = remote.wait_for_clean_a100(
        timeout_seconds=60,
        poll_interval_seconds=1,
    )

    assert inventory == [selected]
    assert gpu == selected
    assert calls == ["kerberos", "query", "kerberos", "query"]


def test_worker_plan_seals_terminal_hashes_for_immutable_resume() -> None:
    paths = remote.build_remote_paths("20260913-stage0-seal-r1")
    plan = remote.build_worker_plan(
        paths=paths,
        run_tag="20260913-stage0-seal-r1",
        source_commit="a" * 40,
        gpu=_gpu(1),
    )
    joined = "\n".join(plan["commands"])

    assert paths["controller"] in joined
    assert "resume.json" in joined
    assert "slo-cohort-burst.remote-resume.v1" in joined
    assert all(name in joined for name in remote.REQUIRED_TERMINAL_FILES)


def _write_compact_bundle(path: Path) -> dict[str, str]:
    path.mkdir(parents=True)
    for name in remote.REQUIRED_TERMINAL_FILES:
        (path / name).write_text("{}\n", encoding="utf-8")
    (path / "runner.log").write_text("complete\n", encoding="utf-8")
    return {
        name: __import__("hashlib").sha256(
            (path / name).read_bytes()
        ).hexdigest()
        for name in remote.REQUIRED_TERMINAL_FILES
    }


def test_controller_fresh_run_executes_source_exact_flow(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []
    paths = remote.build_remote_paths("20260913-stage0-fresh-r1")
    selected = _gpu(2)
    receipt = {
        "schema_version": "slo-cohort-burst.remote-resume.v1",
        "status": "COMPLETE",
        "stage": "ceiling",
        "run_tag": "20260913-stage0-fresh-r1",
        "source_commit": "a" * 40,
        "remote_paths": paths,
        "artifact_sha256": {},
    }
    monkeypatch.setattr(
        remote,
        "require_pushed_head",
        lambda _root: calls.append("head") or "a" * 40,
    )
    monkeypatch.setattr(
        remote,
        "validate_kerberos",
        lambda **_kwargs: calls.append("kerberos") or {"status": "PASS"},
    )
    resume_results = iter((None, receipt))

    def probe_resume(**_kwargs):
        calls.append("resume")
        return next(resume_results)

    monkeypatch.setattr(remote, "probe_resume_receipt", probe_resume)
    monkeypatch.setattr(
        remote,
        "committed_source_archive",
        lambda *_args: calls.append("archive") or b"tar",
    )
    monkeypatch.setattr(
        remote,
        "upload_source_archive",
        lambda **_kwargs: calls.append("upload") or paths["staging"] + "/source",
    )
    monkeypatch.setattr(
        remote,
        "wait_for_clean_a100",
        lambda **_kwargs: calls.append("wait") or ([selected], selected),
    )
    monkeypatch.setattr(
        remote,
        "validate_selected_gpu_still_clean",
        lambda gpu: calls.append("recheck") or gpu,
    )
    monkeypatch.setattr(
        remote,
        "run_worker_plan",
        lambda _plan: calls.append("worker") or {"status": "COMPLETE"},
    )

    def download(**_kwargs):
        calls.append("download")
        destination = tmp_path / "20260913-stage0-fresh-r1"
        receipt["artifact_sha256"] = _write_compact_bundle(
            destination
        )
        return destination

    monkeypatch.setattr(remote, "download_compact_bundle", download)
    monkeypatch.setattr(
        remote,
        "verify_local_bundle",
        lambda _path: calls.append("verify") or {
            "verified": True,
            "classification": "CONTINUE_RUNTIME",
        },
    )
    monkeypatch.setattr(
        remote,
        "validate_dual_verifier_agreement",
        lambda *_args, **_kwargs: calls.append("agree") or {
            "verified": True,
            "classification": "CONTINUE_RUNTIME",
        },
    )
    monkeypatch.setattr(
        remote,
        "write_local_controller_receipt",
        lambda **_kwargs: calls.append("receipt") or (
            tmp_path / "controller.json"
        ),
    )
    args = SimpleNamespace(
        stage="ceiling",
        tag="20260913-stage0-fresh-r1",
        source_commit=None,
        local_artifact_root=str(tmp_path),
        gpu_timeout_seconds=60,
        poll_interval_seconds=1,
    )

    result = remote.run_controller(args)

    assert result["status"] == "COMPLETE"
    assert result["resumed"] is False
    assert result["classification"] == "CONTINUE_RUNTIME"
    assert calls == [
        "head",
        "kerberos",
        "resume",
        "archive",
        "upload",
        "wait",
        "kerberos",
        "recheck",
        "worker",
        "resume",
        "download",
        "verify",
        "agree",
        "receipt",
    ]


def test_controller_valid_resume_skips_upload_gpu_and_worker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []
    tag = "20260913-stage0-resume-r2"
    paths = remote.build_remote_paths(tag)
    receipt = {
        "schema_version": "slo-cohort-burst.remote-resume.v1",
        "status": "COMPLETE",
        "stage": "ceiling",
        "run_tag": tag,
        "source_commit": "a" * 40,
        "remote_paths": paths,
        "artifact_sha256": {},
    }
    monkeypatch.setattr(
        remote,
        "require_pushed_head",
        lambda _root: "a" * 40,
    )
    monkeypatch.setattr(
        remote,
        "validate_kerberos",
        lambda **_kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(
        remote,
        "probe_resume_receipt",
        lambda **_kwargs: calls.append("resume") or receipt,
    )
    monkeypatch.setattr(
        remote,
        "committed_source_archive",
        lambda *_args: pytest.fail("archive must be skipped"),
    )
    monkeypatch.setattr(
        remote,
        "wait_for_clean_a100",
        lambda **_kwargs: pytest.fail("GPU wait must be skipped"),
    )
    monkeypatch.setattr(
        remote,
        "run_worker_plan",
        lambda _plan: pytest.fail("worker must be skipped"),
    )

    def download(**_kwargs):
        calls.append("download")
        destination = tmp_path / tag
        receipt["artifact_sha256"] = _write_compact_bundle(
            destination
        )
        return destination

    monkeypatch.setattr(remote, "download_compact_bundle", download)
    monkeypatch.setattr(
        remote,
        "verify_local_bundle",
        lambda _path: calls.append("verify") or {
            "verified": True,
            "classification": "NO_GO_CEILING",
        },
    )
    monkeypatch.setattr(
        remote,
        "validate_dual_verifier_agreement",
        lambda *_args, **_kwargs: calls.append("agree") or {
            "verified": True,
            "classification": "NO_GO_CEILING",
        },
    )
    monkeypatch.setattr(
        remote,
        "write_local_controller_receipt",
        lambda **_kwargs: tmp_path / "controller.json",
    )

    result = remote.run_controller(SimpleNamespace(
        stage="ceiling",
        tag=tag,
        source_commit=None,
        local_artifact_root=str(tmp_path),
        gpu_timeout_seconds=60,
        poll_interval_seconds=1,
    ))

    assert result["resumed"] is True
    assert result["classification"] == "NO_GO_CEILING"
    assert calls == ["resume", "download", "verify", "agree"]


def test_canonical_controller_keeps_stage_identity_end_to_end(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []
    tag = "20260914-canonical-controller-r1"
    paths = remote.build_remote_paths(tag)
    required = remote.stage_artifact_contract("canonical")["required"]
    receipt = {
        "schema_version": "slo-cohort-burst.remote-resume.v1",
        "status": "COMPLETE",
        "stage": "canonical",
        "run_tag": tag,
        "source_commit": "a" * 40,
        "remote_paths": paths,
        "artifact_sha256": {},
    }
    selected = _gpu(2)
    monkeypatch.setattr(
        remote,
        "require_pushed_head",
        lambda _root: "a" * 40,
    )
    monkeypatch.setattr(
        remote,
        "validate_kerberos",
        lambda **_kwargs: {"status": "PASS"},
    )
    probes = iter((None, receipt))
    monkeypatch.setattr(
        remote,
        "probe_resume_receipt",
        lambda **kwargs: (
            calls.append(("probe", kwargs["stage"]))
            or next(probes)
        ),
    )
    monkeypatch.setattr(
        remote,
        "committed_source_archive",
        lambda *_args: b"tar",
    )
    monkeypatch.setattr(
        remote,
        "upload_source_archive",
        lambda **_kwargs: paths["staging"] + "/source",
    )
    monkeypatch.setattr(
        remote,
        "wait_for_clean_a100",
        lambda **_kwargs: ([selected], selected),
    )
    monkeypatch.setattr(
        remote,
        "validate_selected_gpu_still_clean",
        lambda gpu: gpu,
    )
    monkeypatch.setattr(
        remote,
        "build_worker_plan",
        lambda **kwargs: (
            calls.append(("plan", kwargs["stage"]))
            or {"commands": ["true"], "stage": kwargs["stage"]}
        ),
    )
    monkeypatch.setattr(
        remote,
        "run_worker_plan",
        lambda _plan: {"status": "COMPLETE"},
    )

    def download(**kwargs):
        calls.append(("download", kwargs["stage"]))
        destination = tmp_path / tag
        destination.mkdir()
        hashes = {}
        for name in required:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
            hashes[name] = __import__("hashlib").sha256(
                path.read_bytes()
            ).hexdigest()
        receipt["artifact_sha256"] = hashes
        return destination

    monkeypatch.setattr(remote, "download_compact_bundle", download)
    monkeypatch.setattr(
        remote,
        "verify_local_bundle",
        lambda path, *, stage: (
            calls.append(("verify", stage))
            or {
                "verified": True,
                "classification": "NO_GO_THROUGHPUT",
            }
        ),
    )
    monkeypatch.setattr(
        remote,
        "validate_dual_verifier_agreement",
        lambda _path, _verification, *, stage: (
            calls.append(("agree", stage))
            or {
                "verified": True,
                "classification": "NO_GO_THROUGHPUT",
            }
        ),
    )
    monkeypatch.setattr(
        remote,
        "write_local_controller_receipt",
        lambda **_kwargs: tmp_path / "controller.json",
    )

    result = remote.run_controller(SimpleNamespace(
        stage="canonical",
        tag=tag,
        source_commit=None,
        local_artifact_root=str(tmp_path),
        gpu_timeout_seconds=60,
        poll_interval_seconds=1,
    ))

    assert result["classification"] == "NO_GO_THROUGHPUT"
    assert ("plan", "canonical") in calls
    assert calls.count(("probe", "canonical")) == 2
    assert ("download", "canonical") in calls
    assert ("verify", "canonical") in calls
    assert ("agree", "canonical") in calls
