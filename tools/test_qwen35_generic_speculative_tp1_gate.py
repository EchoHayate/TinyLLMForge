from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / "tools/qwen35_generic_speculative_tp1_gate.py"
VERIFIER_PATH = (
    ROOT / "tools/verify_qwen35_generic_speculative_tp1_gate.py"
)
WORKER_PATH = (
    ROOT / "tools/qwen35_generic_speculative_tp1_worker.py"
)
RUNNER_PATH = (
    ROOT / "tools/run_qwen35_generic_speculative_tp1_gate_remote.sh"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _prompt_rows(batch_size: int) -> list[dict]:
    rows = []
    for prompt_index in range(batch_size):
        pattern = [17 + prompt_index, 31 + prompt_index, 17 + prompt_index]
        token_ids = (pattern * 1350)[:4048]
        rows.append({
            "prompt_index": prompt_index,
            "token_count": len(token_ids),
            "token_ids": token_ids,
            "sha256": _digest(token_ids),
        })
    return rows


def _outputs(batch_size: int) -> list[dict]:
    rows = []
    for prompt_index in range(batch_size):
        token_ids = [101 + prompt_index, 17 + prompt_index] * 4
        rows.append({
            "prompt_index": prompt_index,
            "token_count": len(token_ids),
            "token_ids": token_ids,
            "sha256": _digest(token_ids),
        })
    return rows


def _runtime(policy: str, batch_size: int) -> dict:
    if policy == "baseline":
        return {
            "proposal_rows": 0,
            "proposed_tokens": 0,
            "accepted_draft_tokens": 0,
            "rejected_draft_tokens": 0,
            "first_target_callbacks": 0,
            "verify_callbacks": 0,
            "accepted_prefix_replays": 0,
            "consumed_input_mappings": [],
            "side_state_receipts": [],
            "failure_path_rollbacks": [],
        }
    mappings = [
        {
            "sequence_id": sequence_id,
            "proposal_token_count": 4,
            "accepted_draft_count": 2,
            "verify_input_count": 3,
            "committed_tail_input_count": 2,
            "committed_input_count": 3,
        }
        for sequence_id in range(batch_size)
    ]
    receipts = []
    for sequence_id in range(batch_size):
        handle_id = f"handle-{sequence_id}"
        receipts.extend([
            {
                "sequence_id": sequence_id,
                "handle_id": handle_id,
                "operation": "prepare",
                "state": "prepared",
            },
            {
                "sequence_id": sequence_id,
                "handle_id": handle_id,
                "operation": "select",
                "state": "selected",
            },
            {
                "sequence_id": sequence_id,
                "handle_id": handle_id,
                "operation": "apply",
                "state": "applied",
            },
            {
                "sequence_id": sequence_id,
                "handle_id": handle_id,
                "operation": "seal",
                "state": "sealed",
            },
        ])
    return {
        "proposal_rows": batch_size,
        "proposed_tokens": batch_size * 4,
        "accepted_draft_tokens": batch_size * 2,
        "rejected_draft_tokens": batch_size * 2,
        "first_target_callbacks": 1,
        "verify_callbacks": 1,
        "accepted_prefix_replays": 0,
        "consumed_input_mappings": mappings,
        "side_state_receipts": receipts,
        "failure_path_rollbacks": [],
    }


def _cell(policy: str, batch_size: int) -> dict:
    return {
        "schema_version": (
            "qwen35.generic-speculative-tp1-"
            "transactional-correctness.v1"
        ),
        "policy": policy,
        "batch_size": batch_size,
        "world_size": 1,
        "gpu_index": 0,
        "model_identity": {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForCausalLM"],
            "linear_layer_count": 16,
            "full_attention_layer_count": 8,
        },
        "prompt_rows": _prompt_rows(batch_size),
        "output_rows": _outputs(batch_size),
        "runtime": _runtime(policy, batch_size),
        "lease_inventory": {
            "before": 0,
            "after": 0,
            "leaked_sequence_ids": [],
        },
        "runtime_poisoned": False,
        "cleanup_receipt": {
            "engine_exit_called": True,
            "worker_exit_code": 0,
            "owned_children_remaining": [],
        },
    }


def _result() -> dict:
    cells = {
        f"{policy}:b{batch_size}": _cell(policy, batch_size)
        for batch_size in (1, 4)
        for policy in ("baseline", "ngram")
    }
    return {
        "schema_version": (
            "qwen35.generic-speculative-tp1-"
            "transactional-correctness.v1"
        ),
        "classification": "SECOND_MODEL_TP1_ESTABLISHED",
        "model_manifest_sha256": (
            "3e650a908234771c3cf1ac4e20c4d38fe"
            "69982efedaf4a3e631ad0b14aad7dd0"
        ),
        "source_tree_sha256": "a" * 64,
        "world_size": 1,
        "gpu_index": 0,
        "cells": cells,
        "parity": {"b1": True, "b4": True},
    }


def test_contract_constants_are_frozen():
    gate = _load_module("qwen35_generic_gate_constants", GATE_PATH)

    assert gate.SCHEMA_VERSION == (
        "qwen35.generic-speculative-tp1-"
        "transactional-correctness.v1"
    )
    assert gate.POLICIES == ("baseline", "ngram")
    assert gate.BATCH_SIZES == (1, 4)
    assert gate.WORLD_SIZE == 1
    assert gate.NGRAM_SIZE == 3
    assert gate.MAX_PROPOSAL_TOKENS == 4
    assert gate.MODEL_MANIFEST_SHA256 == (
        "3e650a908234771c3cf1ac4e20c4d38fe"
        "69982efedaf4a3e631ad0b14aad7dd0"
    )


def test_model_identity_hashes_frozen_parent_manifest(tmp_path):
    gate = _load_module(
        "qwen35_generic_gate_model_manifest",
        GATE_PATH,
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type":"qwen3_5"}\n',
        encoding="utf-8",
    )
    manifest_path = tmp_path / "model_manifest.json"
    manifest_path.write_text(
        '{"files":{},"schema_version":1}\n',
        encoding="utf-8",
    )

    assert gate.model_manifest_sha256(str(model_dir)) == (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda result: result.update(
                model_manifest_sha256="0" * 64
            ),
            "model manifest",
        ),
        (
            lambda result: result["cells"]["ngram:b1"][
                "model_identity"
            ].update(model_type="qwen3"),
            "qwen3_5",
        ),
        (
            lambda result: result["cells"]["ngram:b1"][
                "model_identity"
            ].update(linear_layer_count=0),
            "linear layer",
        ),
        (
            lambda result: result["cells"]["ngram:b1"][
                "model_identity"
            ].update(full_attention_layer_count=0),
            "full attention",
        ),
        (
            lambda result: result["cells"].pop("baseline:b1"),
            "cell inventory",
        ),
        (
            lambda result: result["cells"]["ngram:b1"][
                "output_rows"
            ][0]["token_ids"].append(9),
            "output",
        ),
        (
            lambda result: result["cells"]["ngram:b1"]["runtime"].update(
                proposed_tokens=0
            ),
            "proposed",
        ),
        (
            lambda result: result["cells"]["ngram:b1"]["runtime"].update(
                accepted_draft_tokens=0
            ),
            "accepted",
        ),
        (
            lambda result: result["cells"]["ngram:b1"]["runtime"].update(
                rejected_draft_tokens=0
            ),
            "rejected",
        ),
        (
            lambda result: result["cells"]["ngram:b1"]["runtime"].update(
                first_target_callbacks=0
            ),
            "first-target",
        ),
        (
            lambda result: result["cells"]["ngram:b1"]["runtime"].update(
                verify_callbacks=0
            ),
            "verify callback",
        ),
        (
            lambda result: result["cells"]["ngram:b1"]["runtime"][
                "side_state_receipts"
            ].pop(),
            "side-state",
        ),
        (
            lambda result: result["cells"]["ngram:b1"]["runtime"].update(
                failure_path_rollbacks=[{
                    "handle_id": "failed-handle",
                    "reason": "injected",
                }]
            ),
            "rollback receipt",
        ),
        (
            lambda result: result["cells"]["ngram:b1"]["runtime"][
                "consumed_input_mappings"
            ][0].update(committed_input_count=4),
            "committed input",
        ),
        (
            lambda result: result["cells"]["ngram:b1"]["runtime"].update(
                accepted_prefix_replays=1
            ),
            "accepted-prefix replay",
        ),
        (
            lambda result: result["cells"]["ngram:b1"][
                "lease_inventory"
            ].update(after=1, leaked_sequence_ids=[0]),
            "lease",
        ),
        (
            lambda result: result["cells"]["ngram:b1"].update(
                runtime_poisoned=True
            ),
            "poisoned",
        ),
        (
            lambda result: result["cells"]["ngram:b1"][
                "cleanup_receipt"
            ].update(engine_exit_called=False),
            "cleanup",
        ),
    ],
)
def test_result_rejects_invalid_authority(mutate, match):
    gate = _load_module("qwen35_generic_gate_validation", GATE_PATH)
    result = _result()
    mutate(result)

    with pytest.raises(ValueError, match=match):
        gate.validate_result(result)


def test_result_accepts_opaque_cell_inventory_without_run_id_sorting():
    gate = _load_module("qwen35_generic_gate_opaque_ids", GATE_PATH)
    result = _result()
    result["opaque_run_id"] = "zeta-before-alpha-is-not-a-date"

    normalized = gate.validate_result(result)

    assert normalized["classification"] == (
        "SECOND_MODEL_TP1_ESTABLISHED"
    )


def _write_run(tmp_path: Path) -> Path:
    gate = _load_module("qwen35_generic_gate_write_run", GATE_PATH)
    run_dir = tmp_path / "authority"
    run_dir.mkdir()
    result = gate.validate_result(_result())
    gate.atomic_write_json(run_dir / "result.json", result)
    result_digest = gate.sha256_file(run_dir / "result.json")
    gate.atomic_write_json(
        run_dir / "source_manifest.json",
        {
            "schema_version": gate.SCHEMA_VERSION,
            "source_tree_sha256": result["source_tree_sha256"],
            "model_manifest_sha256": result[
                "model_manifest_sha256"
            ],
            "source_files": {
                "tinyvllm/engine/model_runner.py": "b" * 64,
            },
            "artifacts": {"result.json": result_digest},
        },
    )
    return run_dir


def test_independent_verifier_accepts_complete_authority(tmp_path):
    run_dir = _write_run(tmp_path)
    verifier = _load_module(
        "qwen35_generic_gate_verifier",
        VERIFIER_PATH,
    )

    verification = verifier.verify_run(run_dir)

    assert verification == {
        "classification": "SECOND_MODEL_TP1_ESTABLISHED",
        "failures": [],
    }


def test_independent_verifier_rejects_source_tree_mismatch(tmp_path):
    run_dir = _write_run(tmp_path)
    manifest_path = run_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_tree_sha256"] = "c" * 64
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    verifier = _load_module(
        "qwen35_generic_gate_source_verifier",
        VERIFIER_PATH,
    )

    verification = verifier.verify_run(run_dir)

    assert verification["classification"] == "FAIL"
    assert "source tree" in verification["failures"][0]


def _fake_worker_subprocess(gate, calls):
    def run(command, **kwargs):
        policy = command[command.index("--policy") + 1]
        batch_size = int(
            command[command.index("--batch-size") + 1]
        )
        output_path = Path(command[command.index("--out") + 1])
        calls.append((policy, batch_size, output_path))
        gate.atomic_write_json(
            output_path,
            _cell(policy, batch_size),
        )
        return SimpleNamespace(returncode=0)

    return run


def test_campaign_runs_four_fresh_cells_and_publishes_atomically(
    tmp_path,
    monkeypatch,
):
    gate = _load_module("qwen35_generic_gate_campaign", GATE_PATH)
    calls = []
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        _fake_worker_subprocess(gate, calls),
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type":"qwen3_5"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        gate,
        "model_manifest_sha256",
        lambda path: gate.MODEL_MANIFEST_SHA256,
    )

    result = gate.run_campaign(
        model_path=str(model_dir),
        gpu_index=0,
        output_dir=tmp_path / "authority",
        repo_root=ROOT,
        source_files=("tinyvllm/engine/model_runner.py",),
        verifier=lambda run_dir, source_root: {
            "classification": "SECOND_MODEL_TP1_ESTABLISHED",
            "failures": [],
        },
    )

    assert [(policy, batch) for policy, batch, _ in calls] == [
        ("baseline", 1),
        ("ngram", 1),
        ("baseline", 4),
        ("ngram", 4),
    ]
    assert len({path.parent for _, _, path in calls}) == 1
    assert result["classification"] == (
        "SECOND_MODEL_TP1_ESTABLISHED"
    )
    assert (tmp_path / "authority" / "verify.json").is_file()


def test_campaign_retains_failed_authority(tmp_path, monkeypatch):
    gate = _load_module("qwen35_generic_gate_failure", GATE_PATH)
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        _fake_worker_subprocess(gate, []),
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type":"qwen3_5"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        gate,
        "model_manifest_sha256",
        lambda path: gate.MODEL_MANIFEST_SHA256,
    )

    with pytest.raises(RuntimeError, match="authority.failed"):
        gate.run_campaign(
            model_path=str(model_dir),
            gpu_index=0,
            output_dir=tmp_path / "authority",
            repo_root=ROOT,
            source_files=("tinyvllm/engine/model_runner.py",),
            verifier=lambda run_dir, source_root: {
                "classification": "FAIL",
                "failures": ["sentinel"],
            },
        )

    assert (tmp_path / "authority.failed" / "verify.json").is_file()


def test_worker_recomputes_consumed_input_mappings():
    worker = _load_module(
        "qwen35_generic_gate_worker_summary",
        WORKER_PATH,
    )
    observations = [{
        "speculative_proposal_row_count": 1,
        "speculative_proposal_token_counts": {"7": 4},
        "speculative_accepted_draft_token_counts": {"7": 2},
        "speculative_first_target_callback_count": 1,
        "speculative_fixed_q_group_count": 1,
    }]

    runtime = worker.summarize_observations(
        observations,
        side_state_receipts=[],
    )

    assert runtime["proposed_tokens"] == 4
    assert runtime["accepted_draft_tokens"] == 2
    assert runtime["rejected_draft_tokens"] == 2
    assert runtime["consumed_input_mappings"] == [{
        "sequence_id": 7,
        "proposal_token_count": 4,
        "accepted_draft_count": 2,
        "verify_input_count": 3,
        "committed_tail_input_count": 2,
        "committed_input_count": 3,
    }]
    assert runtime["accepted_prefix_replays"] == 0


def test_worker_normalizes_real_batch_receipts_per_sequence():
    worker = _load_module(
        "qwen35_generic_gate_worker_receipts",
        WORKER_PATH,
    )
    receipts = [
        {
            "operation": operation,
            "status": status,
            "transaction_id": "qwen35-spec-3",
            "sequence_ids": [9, 4],
        }
        for operation, status in (
            ("prepare", "prepared"),
            ("select", "selected"),
            ("apply", "applied"),
            ("seal", "sealed"),
        )
    ]

    normalized = worker.normalize_side_state_receipts(receipts)

    assert [
        row["operation"]
        for row in normalized
        if row["sequence_id"] == 9
    ] == ["prepare", "select", "apply", "seal"]
    assert {
        row["handle_id"]
        for row in normalized
    } == {"qwen35-spec-3"}
    assert worker.gate._validate_side_state(
        normalized,
        [],
    ) == normalized


def test_worker_reads_hybrid_inventory_from_nested_text_config():
    worker = _load_module(
        "qwen35_generic_gate_worker_model_identity",
        WORKER_PATH,
    )
    layer_types = ["linear_attention"] * 18 + [
        "full_attention"
    ] * 6
    hf_config = SimpleNamespace(
        model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=SimpleNamespace(layer_types=layer_types),
    )
    engine = SimpleNamespace(
        config=SimpleNamespace(hf_config=hf_config),
        model_runner=SimpleNamespace(),
    )

    assert worker._model_identity(engine) == {
        "model_type": "qwen3_5",
        "architectures": [
            "Qwen3_5ForConditionalGeneration"
        ],
        "linear_layer_count": 18,
        "full_attention_layer_count": 6,
    }


def test_remote_runner_binds_serial_transport_and_authority():
    text = RUNNER_PATH.read_text(encoding="utf-8")

    assert "sitian@10.232.195.203" in text
    assert "FILE:/Users/bytedance/krb5cc_sitian" in text
    assert "ControlMaster=no" in text
    assert "ControlPath=none" in text
    assert "REMOTE_COMMAND_RETRY_ATTEMPTS" in text
    assert "REMOTE_RSYNC_RETRY_ATTEMPTS" in text
    assert "retry_remote_command" in text
    assert "retry_remote_rsync" in text
    assert "nvidia-smi" in text
    assert "memory.free" in text
    assert (
        "/data00/home/sitian/sitian-workspace01/tllm/"
        "qwen35-hybrid-state-runs/"
        "qwen35-2b-hybrid-acquire-20260723-222004/model"
    ) in text
    assert (
        "3e650a908234771c3cf1ac4e20c4d38fe"
        "69982efedaf4a3e631ad0b14aad7dd0"
    ) in text
    assert "kv_offload_mvp0=True" not in text
    assert "DEFAULT_SOURCE_FILES" in text
    assert "source.tar" in text
    assert "text_config" in text
    assert (
        "/data00/home/sitian/sitian-workspace01/tllm/"
        "env/bin/python"
    ) in text
    assert text.count(
        "verify_qwen35_generic_speculative_tp1_gate.py"
    ) >= 2
    assert "authority.failed" in text
    assert "last_completed_run_path.txt" in text
    assert "date -d" not in text


def test_remote_runner_does_not_replay_non_idempotent_campaign():
    text = RUNNER_PATH.read_text(encoding="utf-8")

    assert "campaign.status" in text
    assert "campaign.pid" in text
    assert "campaign.exit_code" in text
    assert "nohup bash" in text
    assert (
        'retry_remote_command \\\n'
        '  "cd \'${REMOTE_SOURCE}\' && \\\n'
        "   export CUDA_VISIBLE_DEVICES"
    ) not in text


