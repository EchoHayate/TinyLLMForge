from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = (
    ROOT / "tools" / "autoregressive_draft_source_pair_gate.py"
)
VERIFIER_PATH = (
    ROOT / "tools" / "verify_autoregressive_draft_source_pair_gate.py"
)
SOURCE_PAIR_RUNNER_PATH = (
    ROOT / "tools" / "run_autoregressive_draft_source_pair_remote.py"
)
assert GATE_PATH.exists(), f"missing module: {GATE_PATH}"

SPEC = importlib.util.spec_from_file_location(
    "autoregressive_draft_source_pair_gate_test_module",
    GATE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def _load_module(path: Path, name: str):
    assert path.exists(), f"missing module: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_source_pair_schedule_matches_frozen_protocol():
    rows = gate.expected_source_pair_schedule()

    assert [
        (
            row.pair_index,
            row.cuda_mode,
            row.first_source,
            row.second_source,
        )
        for row in rows
    ] == [
        (0, "eager", "baseline", "candidate"),
        (1, "graph", "candidate", "baseline"),
        (2, "graph", "baseline", "candidate"),
        (3, "eager", "candidate", "baseline"),
        (4, "graph", "baseline", "candidate"),
        (5, "eager", "baseline", "candidate"),
        (6, "eager", "candidate", "baseline"),
        (7, "graph", "candidate", "baseline"),
    ]


def test_source_pair_schedule_is_balanced_globally_and_per_mode():
    rows = gate.expected_source_pair_schedule()

    assert len(rows) == 8
    assert sum(row.first_source == "baseline" for row in rows) == 4
    assert sum(row.first_source == "candidate" for row in rows) == 4
    for cuda_mode in ("eager", "graph"):
        mode_rows = [
            row for row in rows if row.cuda_mode == cuda_mode
        ]
        assert len(mode_rows) == 4
        assert sum(
            row.first_source == "baseline" for row in mode_rows
        ) == 2
        assert sum(
            row.first_source == "candidate" for row in mode_rows
        ) == 2


def test_source_pair_sample_counts_match_command_timeline_authority():
    assert gate.MEASURED_REPEATS_PER_EPOCH == 5
    assert gate.EPOCH_COUNT_PER_SOURCE == 8
    assert gate.REQUESTS_PER_REPEAT == 4
    assert gate.MEASURED_REPEATS_PER_SOURCE == 40
    assert gate.REQUEST_SAMPLES_PER_SOURCE == 160


def test_source_pair_classification_precedence_is_exclusive_and_frozen():
    assert gate.CLASSIFICATION_PRECEDENCE == (
        "INCONCLUSIVE_ARTIFACT",
        "NO_GO_CORRECTNESS",
        "INCONCLUSIVE_STATIONARITY",
        "NO_GO_TPOT_P95",
        "NO_GO_TPOT_MEDIAN",
        "NO_GO_TTFT_REGRESSION",
        "NO_GO_THROUGHPUT_REGRESSION",
        "GO_TPOT_TAIL_OPTIMIZATION",
    )


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _correctness() -> dict:
    proposal_rows = [
        [[31, 32, 33, 34] for _ in range(4)],
        [[41, 42], [43], [44, 45, 46], [47, 48, 49, 50]],
    ]
    accepted_prefix_counts = [[2, 1, 3, 4], [1, 1, 2, 2]]
    accepted_rows = [
        [
            row[:prefix]
            for row, prefix in zip(call, prefixes)
        ]
        for call, prefixes in zip(
            proposal_rows,
            accepted_prefix_counts,
        )
    ]
    proposed = sum(
        len(row) for call in proposal_rows for row in call
    )
    accepted = sum(
        len(row) for call in accepted_rows for row in call
    )
    return {
        "target_token_rows": [
            list(range(request, request + 16))
            for request in range(4)
        ],
        "proposal_token_rows": proposal_rows,
        "proposal_row_lengths": [
            [len(row) for row in call] for call in proposal_rows
        ],
        "accepted_prefix_counts": accepted_prefix_counts,
        "accepted_token_rows": accepted_rows,
        "transaction_digest": "c" * 64,
        "active_transaction_count": 0,
        "acceptance": {
            "proposed_tokens": proposed,
            "accepted_tokens": accepted,
            "rate": accepted / proposed,
        },
        "rank_graph_counters": [],
        "rank_graph_resources": [],
        "rank_graph_identities": [
            {"rank": rank, "sha256": f"{rank + 1:x}" * 64}
            for rank in range(4)
        ],
    }


def _child_artifact(
    *,
    source_commit: str,
    source_tree_sha256: str,
    tpot_ns: int,
    ttft_ns: int,
    batch_elapsed_ns: int,
) -> dict:
    epochs = {}
    for row in gate.expected_source_pair_schedule():
        measured_runs = []
        for repeat in range(gate.MEASURED_REPEATS_PER_EPOCH):
            correctness = _correctness()
            measured_runs.append({
                "repeat": repeat,
                "outputs": copy.deepcopy(
                    correctness["target_token_rows"]
                ),
                "timing": {
                    "request_count": 4,
                    "total_output_tokens": 64,
                    "batch_elapsed_ns": batch_elapsed_ns,
                    "per_request": [
                        {
                            "sequence_id": sequence_id,
                            "output_tokens": 16,
                            "ttft_ns": ttft_ns,
                            "tpot_ns": tpot_ns,
                            "completion_latency_ns": (
                                ttft_ns + 15 * tpot_ns
                            ),
                        }
                        for sequence_id in range(4)
                    ],
                },
                "correctness": correctness,
            })
        epochs[f"epoch-{row.pair_index}"] = {
            "identity": {
                "epoch_index": row.pair_index,
                "label": row.cuda_mode,
            },
            "worker": {
                "source_commit": source_commit,
                "source_tree_sha256": source_tree_sha256,
                "gpu_uuids": [f"GPU-{rank}" for rank in range(4)],
                "prompt_sha256": "a" * 64,
                "prompt_rows": [
                    {
                        "prompt_index": index,
                        "token_ids": [index + 1] * 256,
                        "token_count": 256,
                        "sha256": f"{index + 1:x}" * 64,
                    }
                    for index in range(4)
                ],
                "request_order": [0, 1, 2, 3],
                "cuda_graph_mode": row.cuda_mode,
                "measured_runs": measured_runs,
            },
            "identity_correctness_passed": True,
            "timeline_conservation_passed": True,
            "stationarity_passed": True,
            "passed": True,
        }
    return {
        "schema_version": 1,
        "configuration": {
            "tensor_parallel_size": 4,
            "batch_size": 4,
            "max_proposal_tokens": 4,
            "prompt_tokens": 256,
            "output_tokens": 16,
            "temperature": 0.0,
            "proposal_kv_allocator": "direct",
            "proposal_kv_offload": False,
        },
        "provenance": {"run_tag": f"child-{source_commit[:8]}"},
        "epochs": epochs,
        "admission": {
            "identity_correctness_passed": True,
            "timeline_conservation_passed": True,
            "stationarity_passed": True,
            "measured_epoch_count": 8,
            "measured_repeat_count_total": 40,
            "passed": True,
        },
        "classification": "COMMAND_TIMELINE_STABLE",
    }


def _receipt(
    artifact: dict,
    *,
    location: str,
    manifest_sha256: str,
) -> dict:
    return {
        "schema_version": 1,
        "verified": True,
        "verified_at_utc": (
            "2026-08-20T12:00:00+00:00"
            if location == "remote"
            else "2026-08-20T12:01:00+00:00"
        ),
        "verification_location": location,
        "artifact_path": f"/{location}/command-timeline.json",
        "artifact_sha256": gate.canonical_json_sha256(artifact),
        "classification": artifact["classification"],
        "source_file_count": 10,
        "source_inventory_sha256": "4" * 64,
        "raw_input_file_count": 16,
        "raw_input_inventory_sha256": "5" * 64,
        "manifest_verified": True,
        "manifest_sha256": manifest_sha256,
        "manifest_file_count": 80,
        "verifier_source_sha256": "7" * 64,
    }


def _refresh_receipts(inputs: dict, source: str) -> None:
    artifact = inputs[f"{source}_artifact"]
    manifest_sha256 = inputs[f"{source}_manifest_sha256"]
    inputs[f"{source}_verifier_receipts"] = {
        location: _receipt(
            artifact,
            location=location,
            manifest_sha256=manifest_sha256,
        )
        for location in ("remote", "local")
    }


@pytest.fixture
def valid_source_pair_inputs() -> dict:
    baseline = _child_artifact(
        source_commit=gate.BASELINE_REVISION,
        source_tree_sha256="b" * 64,
        tpot_ns=100_000_000,
        ttft_ns=100_000_000,
        batch_elapsed_ns=800_000_000,
    )
    candidate = _child_artifact(
        source_commit="f" * 40,
        source_tree_sha256="d" * 64,
        tpot_ns=80_000_000,
        ttft_ns=102_000_000,
        batch_elapsed_ns=780_000_000,
    )
    return {
        "run_tag": "20260820-source-pair-test-r1",
        "baseline_artifact": baseline,
        "candidate_artifact": candidate,
        "baseline_manifest_sha256": "8" * 64,
        "candidate_manifest_sha256": "9" * 64,
        "baseline_verifier_receipts": {
            location: _receipt(
                baseline,
                location=location,
                manifest_sha256="8" * 64,
            )
            for location in ("remote", "local")
        },
        "candidate_verifier_receipts": {
            location: _receipt(
                candidate,
                location=location,
                manifest_sha256="9" * 64,
            )
            for location in ("remote", "local")
        },
    }


def test_source_pair_artifact_uses_all_requests_and_fresh_pair_metrics(
    valid_source_pair_inputs,
):
    artifact = gate.build_source_pair_artifact(
        **valid_source_pair_inputs
    )

    assert artifact["sample_counts"] == {
        "epochs_per_source": 8,
        "measured_repeats_per_source": 40,
        "request_samples_per_source": 160,
    }
    assert artifact["metrics"]["baseline"] == {
        "tpot_median_ns": 100_000_000,
        "tpot_p95_ns": 100_000_000,
        "ttft_p95_ns": 100_000_000,
        "median_batch_throughput_tokens_per_s": 80.0,
    }
    assert artifact["metrics"]["candidate"] == {
        "tpot_median_ns": 80_000_000,
        "tpot_p95_ns": 80_000_000,
        "ttft_p95_ns": 102_000_000,
        "median_batch_throughput_tokens_per_s": pytest.approx(
            82.05128205128206
        ),
    }
    assert artifact["regressions"]["ttft_p95"] == pytest.approx(0.02)
    assert artifact["regressions"]["throughput"] == pytest.approx(
        -0.02564102564102555
    )
    assert artifact["correctness"]["passed"] is True
    assert artifact["stationarity"]["passed"] is True
    assert artifact["classification"] == "GO_TPOT_TAIL_OPTIMIZATION"


def test_source_pair_artifact_binds_sources_manifests_and_receipts(
    valid_source_pair_inputs,
):
    artifact = gate.build_source_pair_artifact(
        **valid_source_pair_inputs
    )

    assert artifact["sources"]["baseline"]["commit"] == (
        gate.BASELINE_REVISION
    )
    assert artifact["sources"]["candidate"]["commit"] == "f" * 40
    assert artifact["sources"]["baseline"]["manifest_sha256"] == "8" * 64
    assert artifact["sources"]["candidate"]["manifest_sha256"] == "9" * 64
    assert len(artifact["sources"]["baseline"]["artifact_sha256"]) == 64
    assert len(
        artifact["sources"]["candidate"]["normalized_receipt_sha256"]
    ) == 64
    assert artifact["gpu_uuids"] == [
        "GPU-0",
        "GPU-1",
        "GPU-2",
        "GPU-3",
    ]


def test_source_pair_artifact_uses_embedded_epoch_index_not_mapping_order(
    valid_source_pair_inputs,
):
    reordered = copy.deepcopy(valid_source_pair_inputs)
    for source in ("baseline", "candidate"):
        epochs = reordered[f"{source}_artifact"]["epochs"]
        reordered[f"{source}_artifact"]["epochs"] = {
            key: epochs[key] for key in reversed(list(epochs))
        }
        _refresh_receipts(reordered, source)

    artifact = gate.build_source_pair_artifact(**reordered)

    assert artifact["sample_counts"]["epochs_per_source"] == 8
    assert artifact["classification"] == "GO_TPOT_TAIL_OPTIMIZATION"


def test_source_pair_output_or_transaction_mismatch_is_no_go_correctness(
    valid_source_pair_inputs,
):
    invalid = copy.deepcopy(valid_source_pair_inputs)
    run = invalid["candidate_artifact"]["epochs"]["epoch-3"][
        "worker"
    ]["measured_runs"][2]
    run["outputs"][0][0] += 1
    run["correctness"]["target_token_rows"][0][0] += 1
    _refresh_receipts(invalid, "candidate")

    artifact = gate.build_source_pair_artifact(**invalid)

    assert artifact["correctness"]["passed"] is False
    assert artifact["classification"] == "NO_GO_CORRECTNESS"


def test_source_pair_stationarity_is_separate_for_eager_and_graph(
    valid_source_pair_inputs,
):
    invalid = copy.deepcopy(valid_source_pair_inputs)
    for run in invalid["candidate_artifact"]["epochs"]["epoch-6"][
        "worker"
    ]["measured_runs"]:
        for request in run["timing"]["per_request"]:
            request["tpot_ns"] = 400_000_000
    _refresh_receipts(invalid, "candidate")

    artifact = gate.build_source_pair_artifact(**invalid)

    assert artifact["stationarity"]["eager"]["tpot_ratio"][
        "passed"
    ] is False
    assert artifact["stationarity"]["graph"]["tpot_ratio"][
        "passed"
    ] is True
    assert artifact["classification"] == "INCONCLUSIVE_STATIONARITY"


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("artifact_complete", False, "INCONCLUSIVE_ARTIFACT"),
        ("correctness_passed", False, "NO_GO_CORRECTNESS"),
        ("stationarity_passed", False, "INCONCLUSIVE_STATIONARITY"),
        ("candidate_tpot_p95_ns", 105_870_001, "NO_GO_TPOT_P95"),
        ("candidate_tpot_median_ns", 85_660_001, "NO_GO_TPOT_MEDIAN"),
        ("ttft_regression", 0.0300001, "NO_GO_TTFT_REGRESSION"),
        (
            "throughput_regression",
            0.0300001,
            "NO_GO_THROUGHPUT_REGRESSION",
        ),
    ],
)
def test_source_pair_classification_obeys_precedence(
    field,
    value,
    expected,
):
    inputs = {
        "artifact_complete": True,
        "correctness_passed": True,
        "stationarity_passed": True,
        "candidate_tpot_p95_ns": 80_000_000,
        "candidate_tpot_median_ns": 80_000_000,
        "ttft_regression": 0.0,
        "throughput_regression": 0.0,
    }
    inputs[field] = value
    if field not in (
        "artifact_complete",
        "correctness_passed",
        "stationarity_passed",
    ):
        inputs["correctness_passed"] = False
        assert gate.classify_source_pair(**inputs) == "NO_GO_CORRECTNESS"
        inputs["correctness_passed"] = True

    assert gate.classify_source_pair(**inputs) == expected


def test_source_pair_rejects_wrong_baseline_revision(
    valid_source_pair_inputs,
):
    invalid = copy.deepcopy(valid_source_pair_inputs)
    for epoch in invalid["baseline_artifact"]["epochs"].values():
        epoch["worker"]["source_commit"] = "0" * 40
    _refresh_receipts(invalid, "baseline")

    with pytest.raises(ValueError, match="baseline revision"):
        gate.build_source_pair_artifact(**invalid)


def test_source_pair_rejects_verifier_disagreement(
    valid_source_pair_inputs,
):
    invalid = copy.deepcopy(valid_source_pair_inputs)
    invalid["candidate_verifier_receipts"]["local"][
        "manifest_file_count"
    ] += 1

    with pytest.raises(ValueError, match="verifier receipts disagree"):
        gate.build_source_pair_artifact(**invalid)


def test_source_pair_rejects_incomplete_rank_inventory(
    valid_source_pair_inputs,
):
    invalid = copy.deepcopy(valid_source_pair_inputs)
    for epoch in invalid["candidate_artifact"]["epochs"].values():
        epoch["worker"]["gpu_uuids"].pop()
    _refresh_receipts(invalid, "candidate")

    with pytest.raises(ValueError, match="four GPU"):
        gate.build_source_pair_artifact(**invalid)


def test_source_pair_rejects_non_finite_timing(
    valid_source_pair_inputs,
):
    invalid = copy.deepcopy(valid_source_pair_inputs)
    invalid["candidate_artifact"]["epochs"]["epoch-0"]["worker"][
        "measured_runs"
    ][0]["timing"]["per_request"][0]["tpot_ns"] = math.inf

    with pytest.raises(ValueError, match="finite"):
        gate.build_source_pair_artifact(**invalid)


def test_source_pair_validation_recomputes_derived_fields(
    valid_source_pair_inputs,
):
    artifact = gate.build_source_pair_artifact(
        **valid_source_pair_inputs
    )
    artifact["metrics"]["candidate"]["tpot_p95_ns"] += 1

    with pytest.raises(ValueError, match="recomputation"):
        gate.validate_source_pair_artifact(
            artifact,
            baseline_artifact=valid_source_pair_inputs[
                "baseline_artifact"
            ],
            candidate_artifact=valid_source_pair_inputs[
                "candidate_artifact"
            ],
            baseline_verifier_receipts=valid_source_pair_inputs[
                "baseline_verifier_receipts"
            ],
            candidate_verifier_receipts=valid_source_pair_inputs[
                "candidate_verifier_receipts"
            ],
        )


def _write_canonical_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gate.canonical_json_bytes(value))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_complete_manifest(root: Path) -> Path:
    manifest = root / "manifest.sha256"
    rows = []
    for path in sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate != manifest
    ):
        rows.append(
            f"{_file_sha256(path)}  {path.relative_to(root).as_posix()}"
        )
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return manifest


@pytest.fixture
def valid_source_pair_bundle(
    tmp_path,
    valid_source_pair_inputs,
) -> dict:
    root = tmp_path / "source-pair"
    baseline_root = root / "children" / "baseline"
    candidate_root = root / "children" / "candidate"
    baseline_artifact_path = baseline_root / "command-timeline.json"
    candidate_artifact_path = candidate_root / "command-timeline.json"
    _write_canonical_json(
        baseline_artifact_path,
        valid_source_pair_inputs["baseline_artifact"],
    )
    _write_canonical_json(
        candidate_artifact_path,
        valid_source_pair_inputs["candidate_artifact"],
    )
    baseline_manifest_path = baseline_root / "manifest.sha256"
    candidate_manifest_path = candidate_root / "manifest.sha256"
    baseline_manifest_path.write_text(
        "baseline child manifest\n",
        encoding="utf-8",
    )
    candidate_manifest_path.write_text(
        "candidate child manifest\n",
        encoding="utf-8",
    )
    inputs = copy.deepcopy(valid_source_pair_inputs)
    inputs["baseline_manifest_sha256"] = _file_sha256(
        baseline_manifest_path
    )
    inputs["candidate_manifest_sha256"] = _file_sha256(
        candidate_manifest_path
    )
    _refresh_receipts(inputs, "baseline")
    _refresh_receipts(inputs, "candidate")
    receipt_paths = {"baseline": {}, "candidate": {}}
    for source in ("baseline", "candidate"):
        child_root = root / "children" / source
        for location in ("remote", "local"):
            path = child_root / f"verify-{location}.json"
            _write_canonical_json(
                path,
                inputs[f"{source}_verifier_receipts"][location],
            )
            receipt_paths[source][location] = path
    artifact = gate.build_source_pair_artifact(**inputs)
    artifact_path = root / "source-pair.json"
    _write_canonical_json(artifact_path, artifact)
    manifest_path = _write_complete_manifest(root)
    return {
        "root": root,
        "artifact": artifact,
        "artifact_path": artifact_path,
        "baseline_artifact_path": baseline_artifact_path,
        "candidate_artifact_path": candidate_artifact_path,
        "baseline_manifest_path": baseline_manifest_path,
        "candidate_manifest_path": candidate_manifest_path,
        "baseline_receipt_paths": receipt_paths["baseline"],
        "candidate_receipt_paths": receipt_paths["candidate"],
        "manifest_path": manifest_path,
    }


def _verify_bundle(verifier, bundle):
    return verifier.verify_source_pair_gate(
        artifact_path=bundle["artifact_path"],
        baseline_artifact_path=bundle["baseline_artifact_path"],
        candidate_artifact_path=bundle["candidate_artifact_path"],
        baseline_manifest_path=bundle["baseline_manifest_path"],
        candidate_manifest_path=bundle["candidate_manifest_path"],
        baseline_receipt_paths=bundle["baseline_receipt_paths"],
        candidate_receipt_paths=bundle["candidate_receipt_paths"],
        manifest_path=bundle["manifest_path"],
    )


def test_source_pair_verifier_rebuilds_and_verifies_complete_manifest(
    valid_source_pair_bundle,
):
    verifier = _load_module(
        VERIFIER_PATH,
        "source_pair_verifier_complete_manifest",
    )

    receipt = _verify_bundle(verifier, valid_source_pair_bundle)

    assert receipt["verified"] is True
    assert receipt["classification"] == "GO_TPOT_TAIL_OPTIMIZATION"
    assert receipt["manifest_verified"] is True
    assert receipt["manifest_file_count"] == 9
    assert receipt["artifact_sha256"] == _file_sha256(
        valid_source_pair_bundle["artifact_path"]
    )


def test_source_pair_verifier_rejects_derived_field_tamper(
    valid_source_pair_bundle,
):
    verifier = _load_module(
        VERIFIER_PATH,
        "source_pair_verifier_derived_tamper",
    )
    artifact = copy.deepcopy(valid_source_pair_bundle["artifact"])
    artifact["classification"] = "NO_GO_TPOT_P95"
    _write_canonical_json(
        valid_source_pair_bundle["artifact_path"],
        artifact,
    )
    valid_source_pair_bundle["manifest_path"] = _write_complete_manifest(
        valid_source_pair_bundle["root"]
    )

    with pytest.raises(ValueError, match="recomputation"):
        _verify_bundle(verifier, valid_source_pair_bundle)


def test_source_pair_verifier_rejects_incomplete_parent_manifest(
    valid_source_pair_bundle,
):
    verifier = _load_module(
        VERIFIER_PATH,
        "source_pair_verifier_incomplete_manifest",
    )
    rows = valid_source_pair_bundle["manifest_path"].read_text(
        encoding="utf-8"
    ).splitlines()
    valid_source_pair_bundle["manifest_path"].write_text(
        "\n".join(rows[:-1]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="complete file inventory"):
        _verify_bundle(verifier, valid_source_pair_bundle)


def test_source_pair_verifier_rejects_parent_manifest_symlink(
    valid_source_pair_bundle,
):
    verifier = _load_module(
        VERIFIER_PATH,
        "source_pair_verifier_manifest_symlink",
    )
    root = valid_source_pair_bundle["root"]
    (root / "result-alias.json").symlink_to("result.json")
    valid_source_pair_bundle["manifest_path"] = _write_complete_manifest(
        root
    )

    with pytest.raises(ValueError, match="symlink"):
        _verify_bundle(verifier, valid_source_pair_bundle)


def test_source_pair_verifier_rejects_bound_child_receipt_tamper(
    valid_source_pair_bundle,
):
    verifier = _load_module(
        VERIFIER_PATH,
        "source_pair_verifier_receipt_tamper",
    )
    receipt_path = valid_source_pair_bundle[
        "candidate_receipt_paths"
    ]["local"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest_file_count"] += 1
    _write_canonical_json(receipt_path, receipt)
    valid_source_pair_bundle["manifest_path"] = _write_complete_manifest(
        valid_source_pair_bundle["root"]
    )

    with pytest.raises(ValueError, match="verifier receipts disagree"):
        _verify_bundle(verifier, valid_source_pair_bundle)


def test_source_pair_runner_paths_stay_under_sitian_task_root():
    runner = _load_module(
        SOURCE_PAIR_RUNNER_PATH,
        "source_pair_runner_paths",
    )

    paths = runner.source_pair_paths("20260820-source-pair-r2")

    assert set(paths) == {
        "parent_primary",
        "parent_controller",
        "baseline_primary",
        "baseline_controller",
        "candidate_primary",
        "candidate_controller",
    }
    assert len(set(paths.values())) == len(paths)
    assert all(
        path.startswith(f"{runner.REMOTE_TASK_ROOT}/")
        for path in paths.values()
    )
    assert all(
        not path.startswith(("/tmp/", "/private/tmp/"))
        for path in paths.values()
    )


def test_source_pair_execution_plan_interleaves_members_then_finalizes():
    runner = _load_module(
        SOURCE_PAIR_RUNNER_PATH,
        "source_pair_runner_plan",
    )

    plan = runner.build_execution_plan("20260820-source-pair-r2")

    epoch_steps = [
        (
            row["pair_index"],
            row["cuda_mode"],
            row["source"],
        )
        for row in plan
        if row["action"] == "epoch"
    ]
    assert epoch_steps == [
        (0, "eager", "baseline"),
        (0, "eager", "candidate"),
        (1, "graph", "candidate"),
        (1, "graph", "baseline"),
        (2, "graph", "baseline"),
        (2, "graph", "candidate"),
        (3, "eager", "candidate"),
        (3, "eager", "baseline"),
        (4, "graph", "baseline"),
        (4, "graph", "candidate"),
        (5, "eager", "baseline"),
        (5, "eager", "candidate"),
        (6, "eager", "candidate"),
        (6, "eager", "baseline"),
        (7, "graph", "candidate"),
        (7, "graph", "baseline"),
    ]
    first_parent = next(
        index
        for index, row in enumerate(plan)
        if row["scope"] == "parent"
    )
    child_finalizations = [
        row for row in plan[:first_parent] if row["action"] == "finalize"
    ]
    assert [row["source"] for row in child_finalizations] == [
        "baseline",
        "candidate",
    ]
    assert all(
        row["action"] != "compare" for row in plan[:first_parent]
    )
    assert [row["action"] for row in plan[first_parent:]] == [
        "assemble",
        "pre-manifest-verify",
        "manifest",
        "primary-verify",
        "controller-copy",
        "controller-verify",
        "compare-receipts",
    ]


def test_child_finalization_uses_candidate_tooling_for_both_sources():
    runner = _load_module(
        SOURCE_PAIR_RUNNER_PATH,
        "source_pair_runner_child_finalization_tooling",
    )
    tag = "20260820-source-pair-r2"
    candidate_source = (
        runner.child_runner.primary_run_path(f"{tag}-candidate")
        + "/source"
    )

    for source in ("baseline", "candidate"):
        command = runner._child_finalization_remote_arguments(
            tag,
            source,
            "assemble",
        )
        script = command[-1]

        assert (
            f"{candidate_source}/tools/"
            "run_autoregressive_draft_command_timeline_remote.py"
        ) in script
        assert f"_remote-action assemble {tag}-{source}" in script


def test_source_pair_git_export_uses_exact_object_without_checkout():
    runner = _load_module(
        SOURCE_PAIR_RUNNER_PATH,
        "source_pair_runner_git_export",
    )
    commands = []

    def command_runner(command, **kwargs):
        commands.append((command, kwargs))
        if command[:3] == ["git", "ls-tree", "-r"]:
            return type("Result", (), {
                "returncode": 0,
                "stdout": (
                    "tinyvllm/engine/llm_engine.py\n"
                    "tools/run_autoregressive_draft_command_timeline_remote.py\n"
                ),
                "stderr": "",
            })()
        if command[:2] == ["git", "archive"]:
            return type("Result", (), {
                "returncode": 0,
                "stdout": b"exact-git-object-archive",
                "stderr": b"",
            })()
        raise AssertionError(f"unexpected command: {command}")

    payload = runner.export_git_revision_archive(
        ROOT,
        gate.BASELINE_REVISION,
        command_runner=command_runner,
    )

    assert payload == b"exact-git-object-archive"
    flattened = [part for command, _ in commands for part in command]
    assert "worktree" not in flattened
    assert "checkout" not in flattened
    assert "diff" not in flattened
    archive_command = commands[-1][0]
    assert gate.BASELINE_REVISION in archive_command
    assert "--prefix=source/" in archive_command


def test_source_pair_preflight_script_checks_every_destination_and_gpu_rule():
    runner = _load_module(
        SOURCE_PAIR_RUNNER_PATH,
        "source_pair_runner_preflight_script",
    )

    script = runner.build_parent_preflight_script(
        "20260820-source-pair-r2"
    )
    paths = runner.source_pair_paths("20260820-source-pair-r2")

    assert all(path in script for path in paths.values())
    assert "memory_used_mib" in script
    assert "<=1024" in script
    assert "utilization_percent" in script
    assert "<=5" in script
    assert "compute_processes" in script
    assert "nvidia-smi" in script


def test_source_pair_preflight_fails_before_remote_mutation_without_kerberos(
    monkeypatch,
):
    runner = _load_module(
        SOURCE_PAIR_RUNNER_PATH,
        "source_pair_runner_kerberos_fail_fast",
    )
    remote_calls = []
    monkeypatch.setattr(
        runner.child_runner,
        "_local_kerberos_preflight",
        lambda **_kwargs: {
            "status": "INCONCLUSIVE_ENVIRONMENT",
            "reason": "local Kerberos payload is invalid",
        },
    )
    monkeypatch.setattr(
        runner.child_runner,
        "_run_remote_command",
        lambda *args, **kwargs: remote_calls.append((args, kwargs)),
    )

    result = runner.run_preflight(
        run_tag="20260820-source-pair-r2",
        repo_root=ROOT,
    )

    assert result["status"] == "INCONCLUSIVE_ENVIRONMENT"
    assert remote_calls == []


def test_command_timeline_candidate_archive_contains_source_pair_closure():
    runner = _load_module(
        ROOT / "tools" / "run_autoregressive_draft_command_timeline_remote.py",
        "command_timeline_runner_source_pair_closure",
    )

    assert {
        "tools/autoregressive_draft_source_pair_gate.py",
        "tools/verify_autoregressive_draft_source_pair_gate.py",
        "tools/run_autoregressive_draft_source_pair_remote.py",
    } <= set(runner.SOURCE_PATHS)


def test_source_pair_campaign_prepares_interleaves_and_finalizes(
    monkeypatch,
):
    runner = _load_module(
        SOURCE_PAIR_RUNNER_PATH,
        "source_pair_runner_campaign",
    )
    tag = "20260820-source-pair-r2"
    paths = runner.source_pair_paths(tag)
    gpu_rows = [
        {
            "index": index,
            "uuid": f"GPU-{index}",
            "memory_used_mib": 0,
            "utilization_percent": 0,
            "compute_processes": [],
        }
        for index in range(4)
    ]
    preflight = {
        "status": "READY",
        "gpu_indices": [0, 1, 2, 3],
        "gpu_uuids": [f"GPU-{index}" for index in range(4)],
        "baseline_revision": gate.BASELINE_REVISION,
        "candidate_revision": "f" * 40,
        "local_kerberos": {"status": "READY"},
        **paths,
    }
    monkeypatch.setattr(
        runner,
        "run_preflight",
        lambda **_kwargs: copy.deepcopy(preflight),
    )
    exported = []

    def export(_root, revision, **_kwargs):
        exported.append(revision)
        return f"archive:{revision}".encode()

    monkeypatch.setattr(runner, "export_git_revision_archive", export)
    child_actions = []

    def child_action(action, arguments, **kwargs):
        child_actions.append({
            "action": action,
            "arguments": list(arguments),
            "input": kwargs.get("input"),
        })
        stdout = (
            json.dumps(gpu_rows)
            if action.startswith("inventory-")
            else ""
        )
        return type("Result", (), {
            "returncode": 0,
            "stdout": stdout,
            "stderr": "",
        })()

    monkeypatch.setattr(
        runner.child_runner,
        "_run_remote_action",
        child_action,
    )
    child_finalization_actions = []
    monkeypatch.setattr(
        runner,
        "_run_child_finalization_action",
        lambda run_tag, source, action, **_kwargs: (
            child_finalization_actions.append((
                run_tag,
                source,
                action,
            ))
            or type("Result", (), {
                "returncode": 0,
                "stdout": "",
                "stderr": "",
            })()
        ),
    )
    parent_actions = []
    monkeypatch.setattr(
        runner,
        "_run_parent_remote_action",
        lambda action, arguments, **_kwargs: (
            parent_actions.append((action, list(arguments)))
            or type("Result", (), {
                "returncode": 0,
                "stdout": "",
                "stderr": "",
            })()
        ),
    )

    result = runner.run_campaign(run_tag=tag, repo_root=ROOT)

    assert result["status"] == "PASS"
    assert exported == [gate.BASELINE_REVISION, "f" * 40]
    prepare = [
        row for row in child_actions if row["action"] == "prepare"
    ]
    assert len(prepare) == 2
    for row in prepare:
        _, patch = runner.child_runner._decode_prepare_payload(row["input"])
        assert patch == b""
    epoch_actions = [
        row for row in child_actions if row["action"] == "epoch"
    ]
    assert [
        (
            row["arguments"][0].removeprefix(f"{tag}-"),
            int(row["arguments"][1]),
            row["arguments"][2],
        )
        for row in epoch_actions
    ] == [
        (source, pair_index, mode)
        for pair_index, mode, source in [
            (0, "eager", "baseline"),
            (0, "eager", "candidate"),
            (1, "graph", "candidate"),
            (1, "graph", "baseline"),
            (2, "graph", "baseline"),
            (2, "graph", "candidate"),
            (3, "eager", "candidate"),
            (3, "eager", "baseline"),
            (4, "graph", "baseline"),
            (4, "graph", "candidate"),
            (5, "eager", "baseline"),
            (5, "eager", "candidate"),
            (6, "eager", "candidate"),
            (6, "eager", "baseline"),
            (7, "graph", "candidate"),
            (7, "graph", "baseline"),
        ]
    ]
    assert child_finalization_actions == [
        (tag, source, action)
        for source in ("baseline", "candidate")
        for action in runner.CHILD_FINALIZATION_ACTIONS
    ]
    assert [action for action, _ in parent_actions] == [
        "assemble",
        "pre-manifest-verify",
        "manifest",
        "primary-verify",
        "controller-copy",
        "controller-verify",
        "compare-receipts",
    ]


def test_source_pair_remote_parent_pipeline_completes_dual_verification(
    monkeypatch,
    tmp_path,
    valid_source_pair_inputs,
):
    runner = _load_module(
        SOURCE_PAIR_RUNNER_PATH,
        "source_pair_runner_parent_pipeline",
    )
    tag = "20260820-source-pair-r2"
    root = tmp_path / "remote-root"
    paths = {
        "parent_primary": str(root / "parent-primary"),
        "parent_controller": str(root / "parent-controller"),
        "baseline_primary": str(root / "baseline-primary"),
        "baseline_controller": str(root / "baseline-controller"),
        "candidate_primary": str(root / "candidate-primary"),
        "candidate_controller": str(root / "candidate-controller"),
    }
    monkeypatch.setattr(
        runner,
        "source_pair_paths",
        lambda _run_tag: copy.deepcopy(paths),
    )
    for source in ("baseline", "candidate"):
        primary = Path(paths[f"{source}_primary"])
        controller = Path(paths[f"{source}_controller"])
        primary.mkdir(parents=True)
        controller.mkdir(parents=True)
        artifact = valid_source_pair_inputs[f"{source}_artifact"]
        artifact_path = primary / "command-timeline.json"
        _write_canonical_json(artifact_path, artifact)
        manifest_path = primary / "manifest.sha256"
        manifest_path.write_text(
            f"{source} child manifest\n",
            encoding="utf-8",
        )
        inputs = copy.deepcopy(valid_source_pair_inputs)
        inputs[f"{source}_manifest_sha256"] = _file_sha256(
            manifest_path
        )
        receipt_remote = _receipt(
            artifact,
            location="remote",
            manifest_sha256=_file_sha256(manifest_path),
        )
        receipt_local = _receipt(
            artifact,
            location="local",
            manifest_sha256=_file_sha256(manifest_path),
        )
        _write_canonical_json(
            primary / "verify.command-timeline.remote.json",
            receipt_remote,
        )
        _write_canonical_json(
            controller / "verify.command-timeline.local.json",
            receipt_local,
        )

    for action in (
        "assemble",
        "pre-manifest-verify",
        "manifest",
        "primary-verify",
        "controller-copy",
        "controller-verify",
        "compare-receipts",
    ):
        assert runner._remote_parent_action(action, [tag]) == 0

    parent_primary = Path(paths["parent_primary"])
    parent_controller = Path(paths["parent_controller"])
    artifact = json.loads(
        (parent_primary / "source-pair.json").read_text(
            encoding="utf-8"
        )
    )
    assert artifact["classification"] == "GO_TPOT_TAIL_OPTIMIZATION"
    assert (
        parent_primary / "verify.source-pair.remote.json"
    ).is_file()
    assert (
        parent_controller / "verify.source-pair.local.json"
    ).is_file()
    assert (
        parent_primary / "manifest.sha256"
    ).read_bytes() == (
        parent_controller / "manifest.sha256"
    ).read_bytes()
