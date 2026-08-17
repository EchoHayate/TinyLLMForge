from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / "tools" / "generic_speculative_tp4_gate.py"
WORKER_PATH = (
    ROOT / "tools" / "generic_speculative_tp4_worker.py"
)
VERIFIER_PATH = (
    ROOT / "tools"
    / "verify_generic_speculative_tp4_gate.py"
)
RUNNER_PATH = (
    ROOT / "tools"
    / "run_generic_speculative_tp4_gate_remote.sh"
)


def _load_module(name, path):
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_tp4_contract_is_independent_and_not_promotable():
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )

    assert gate.SCHEMA_VERSION == 1
    assert gate.CLASSIFICATION == "NOT_PROMOTABLE"
    assert gate.WORLD_SIZE == 4
    assert gate.POLICIES == ("baseline", "ngram")
    assert gate.CONTEXT_TOKENS == 4096
    assert gate.BATCH_SIZES == (1, 4)
    assert gate.MAX_OUTPUT_TOKENS == 8
    assert gate.cell_key("ngram", 4) == "ngram:b4"


def _profile_step(rank, step_index, batch_kind):
    return {
        "rank": rank,
        "step_index": step_index,
        "batch_kind": batch_kind,
        "is_decode": True,
        "decode_ordinal": step_index,
        "active_sequence_count": 2,
        "request_set_sha256": "a" * 64,
        "dispatch": "eager",
        "wall_ns": 1000 + rank,
        "cuda_ns": 800 + rank,
        "non_cuda_upper_bound_ns": 200,
    }


def _profile_prefill_step(rank, step_index=0):
    row = _profile_step(rank, step_index, "prefill")
    row["is_decode"] = False
    row["decode_ordinal"] = None
    return row


def _profile_collective(rank, step_index, operation):
    return {
        "rank": rank,
        "step_index": step_index,
        "decode_ordinal": step_index,
        "operation": operation,
        "tensor_shape": [2, 151936],
        "tensor_dtype": "torch.float16",
        "wall_ns": 100 + rank,
        "cuda_ns": 80 + rank,
    }


def _valid_rank_profile(policy="ngram"):
    ranks = []
    for rank in range(4):
        steps = []
        collectives = []
        if policy == "ngram":
            steps = [
                _profile_step(rank, 0, "spec_first_target"),
                _profile_step(rank, 1, "spec_verify"),
            ]
            collectives = [
                _profile_collective(
                    rank,
                    0,
                    "row_parallel_all_reduce",
                ),
                _profile_collective(
                    rank,
                    1,
                    "row_parallel_all_reduce",
                ),
            ]
        ranks.append({
            "rank": rank,
            "enabled": True,
            "finalization_status": "complete",
            "steps": steps,
            "collectives": collectives,
        })
    return {
        "enabled": True,
        "rank_inventory": [0, 1, 2, 3],
        "ranks": ranks,
    }


def test_validate_rank_profile_requires_four_matching_ranks():
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )

    normalized = gate.validate_rank_profile(
        _valid_rank_profile(),
        policy="ngram",
    )

    assert normalized["rank_inventory"] == [0, 1, 2, 3]


def test_validate_rank_profile_accepts_non_speculative_prefill_steps():
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )
    profile = _valid_rank_profile(policy="baseline")
    for rank, row in enumerate(profile["ranks"]):
        row["steps"].append(_profile_prefill_step(rank))

    normalized = gate.validate_rank_profile(
        profile,
        policy="baseline",
    )

    assert all(
        rank_row["steps"][0]["is_decode"] is False
        for rank_row in normalized["ranks"]
    )


def test_validate_rank_profile_rejects_missing_rank():
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )
    profile = _valid_rank_profile()
    profile["ranks"].pop()

    with pytest.raises(ValueError, match="rank inventory"):
        gate.validate_rank_profile(profile, policy="ngram")


def test_validate_rank_profile_rejects_callback_identity_mismatch():
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )
    profile = _valid_rank_profile()
    profile["ranks"][1]["steps"][0][
        "request_set_sha256"
    ] = "b" * 64

    with pytest.raises(ValueError, match="callback identity"):
        gate.validate_rank_profile(profile, policy="ngram")


def test_validate_rank_profile_rejects_missing_collective():
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )
    profile = _valid_rank_profile()
    profile["ranks"][2]["collectives"].pop()

    with pytest.raises(ValueError, match="collective"):
        gate.validate_rank_profile(profile, policy="ngram")


def test_validate_rank_profile_rejects_collective_identity_mismatch():
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )
    profile = _valid_rank_profile()
    profile["ranks"][3]["collectives"][1][
        "operation"
    ] = "tampered"

    with pytest.raises(ValueError, match="collective identity"):
        gate.validate_rank_profile(profile, policy="ngram")


def test_validate_rank_profile_rejects_baseline_speculative_callback():
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )
    profile = _valid_rank_profile(policy="baseline")
    profile["ranks"][0]["steps"].append(
        _profile_step(0, 0, "spec_first_target")
    )

    with pytest.raises(
        ValueError,
        match="baseline speculative callback",
    ):
        gate.validate_rank_profile(
            copy.deepcopy(profile),
            policy="baseline",
        )


def _residency_status(operation):
    return {
        "prepare": "prepared",
        "precommit": "precommitted",
        "seal": "sealed",
    }[operation]


def _residency_phase(operation, ticket_id=1):
    committed = (
        []
        if operation == "prepare"
        else [[12, 3]]
    )
    rejected = (
        []
        if operation == "prepare"
        else [[13, 4]]
    )
    return {
        "ticket_id": ticket_id,
        "operation": operation,
        "status": _residency_status(operation),
        "rows": [
            {
                "ticket_id": ticket_id,
                "participant_id": rank,
                "operation": operation,
                "status": _residency_status(operation),
                "sequence_ids": [8, 4],
                "committed_block_identities": committed,
                "rejected_block_identities": rejected,
                "detail": "",
            }
            for rank in range(4)
        ],
    }


def _valid_residency_phases():
    return [
        _residency_phase("prepare"),
        _residency_phase("precommit"),
        _residency_phase("seal"),
    ]


def test_validate_residency_phases_requires_successful_order():
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )

    normalized = gate.validate_residency_phases(
        _valid_residency_phases()
    )

    assert [
        phase["operation"]
        for phase in normalized
    ] == ["prepare", "precommit", "seal"]


def test_validate_residency_phases_rejects_incomplete_rank_inventory():
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )
    phases = _valid_residency_phases()
    phases[1]["rows"].pop()

    with pytest.raises(
        ValueError,
        match="residency rank inventory",
    ):
        gate.validate_residency_phases(phases)


def test_validate_residency_phases_rejects_wrong_order():
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )
    phases = _valid_residency_phases()
    phases[1], phases[2] = phases[2], phases[1]

    with pytest.raises(
        ValueError,
        match="residency phase order",
    ):
        gate.validate_residency_phases(phases)


def _valid_cleanup_receipt():
    return {
        "process_group_destroyed": True,
        "rank_exit_codes": [0, 0, 0, 0],
        "owned_children_remaining": [],
        "rank_cleanup_receipts": [
            {
                "rank": rank,
                "process_group_destroyed": True,
            }
            for rank in range(4)
        ],
    }


def test_validate_cleanup_requires_all_four_rank_receipts():
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )

    normalized = gate.validate_cleanup_receipt(
        _valid_cleanup_receipt()
    )

    assert normalized["process_group_destroyed"] is True


def test_validate_cleanup_rejects_missing_rank_receipt():
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )
    receipt = _valid_cleanup_receipt()
    receipt["rank_cleanup_receipts"].pop()

    with pytest.raises(
        ValueError,
        match="cleanup rank inventory",
    ):
        gate.validate_cleanup_receipt(receipt)


def test_validate_cleanup_rejects_nonzero_rank_exit():
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )
    receipt = _valid_cleanup_receipt()
    receipt["rank_exit_codes"][2] = 1

    with pytest.raises(ValueError, match="rank exit codes"):
        gate.validate_cleanup_receipt(receipt)


def _prompt_row(prompt_index):
    token_ids = [prompt_index + 1] * 4096
    return {
        "prompt_index": prompt_index,
        "token_count": 4096,
        "sha256": hashlib.sha256(
            json.dumps(
                token_ids,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def _runtime_summary(policy, batch_size):
    if policy == "baseline":
        return {
            "proposal_rows": 0,
            "proposed_tokens": 0,
            "accepted_draft_tokens": 0,
            "first_target_callbacks": 0,
            "tail_callbacks": 0,
        }
    return {
        "proposal_rows": batch_size * 2,
        "proposed_tokens": batch_size * 4,
        "accepted_draft_tokens": batch_size * 2,
        "first_target_callbacks": 2,
        "tail_callbacks": 2,
    }


def _kv_rank_deltas():
    return [
        {
            "rank": rank,
            "h2d_copies": 4 + rank,
            "h2d_bytes": 4096 + rank,
            "d2h_copies": 1,
            "d2h_bytes": 1024,
            "copy_waits": 2,
            "evictions": 1,
            "evict_clean": 1,
            "speculative_residency_committed_blocks": 0,
            "speculative_residency_rejected_blocks": 0,
            "speculative_residency_rejected_d2h_copies": 0,
        }
        for rank in range(4)
    ]


def _valid_cell(policy="ngram", batch_size=1):
    return {
        "schema_version": 1,
        "classification": "NOT_PROMOTABLE",
        "policy": policy,
        "context_tokens": 4096,
        "batch_size": batch_size,
        "world_size": 4,
        "rank_inventory": [0, 1, 2, 3],
        "ack_ranks": [1, 2, 3],
        "prompt_rows": [
            _prompt_row(prompt_index)
            for prompt_index in range(batch_size)
        ],
        "outputs": [
            [
                1000 + prompt_index * 10 + token_index
                for token_index in range(8)
            ]
            for prompt_index in range(batch_size)
        ],
        "runtime": _runtime_summary(policy, batch_size),
        "kv_rank_deltas": _kv_rank_deltas(),
        "residency_phases": (
            _valid_residency_phases()
            if policy == "ngram"
            else []
        ),
        "profile": _valid_rank_profile(policy=policy),
        "tokenizer_identifier": "qwen3-fixture",
        "dtype": "torch.float16",
        "cleanup_receipt": _valid_cleanup_receipt(),
    }


def test_validate_cell_result_accepts_candidate_authority():
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )

    normalized = gate.validate_cell_result(
        _valid_cell(policy="ngram", batch_size=4)
    )

    assert normalized["policy"] == "ngram"
    assert normalized["batch_size"] == 4
    assert normalized["runtime"]["proposed_tokens"] > 0


def test_validate_cell_result_rejects_candidate_without_runtime():
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )
    cell = _valid_cell()
    cell["runtime"]["tail_callbacks"] = 0

    with pytest.raises(
        ValueError,
        match="candidate runtime evidence",
    ):
        gate.validate_cell_result(cell)


def test_validate_cell_result_rejects_rejected_speculative_d2h():
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )
    cell = _valid_cell()
    cell["kv_rank_deltas"][2][
        "speculative_residency_rejected_d2h_copies"
    ] = 1

    with pytest.raises(
        ValueError,
        match="rejected speculative blocks",
    ):
        gate.validate_cell_result(cell)


def _valid_result():
    cells = {
        "baseline:b1": _valid_cell("baseline", 1),
        "ngram:b1": _valid_cell("ngram", 1),
        "baseline:b4": _valid_cell("baseline", 4),
        "ngram:b4": _valid_cell("ngram", 4),
    }
    return {
        "schema_version": 1,
        "classification": "NOT_PROMOTABLE",
        "claim_scope": (
            "Qwen3-0.6B generic host n-gram TP4 "
            "correctness and collective authority"
        ),
        "limitations": [
            "no TP4 performance claim",
            "no second-model evidence",
        ],
        "source_tree_sha256": "c" * 64,
        "model_manifest_sha256": "d" * 64,
        "world_size": 4,
        "gpu_indices": [0, 1, 2, 3],
        "cells": cells,
        "parity": {
            "b1": True,
            "b4": True,
        },
    }


def test_validate_result_requires_exact_policy_parity():
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )
    result = _valid_result()

    assert gate.validate_result(result)["parity"] == {
        "b1": True,
        "b4": True,
    }

    result["cells"]["ngram:b1"]["outputs"][0][0] += 1
    with pytest.raises(ValueError, match="output parity"):
        gate.validate_result(result)


def test_source_tree_hash_changes_with_bound_source(tmp_path):
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )
    source = tmp_path / "a.py"
    source.write_text("one\n", encoding="utf-8")
    before = gate.source_tree_sha256(tmp_path, ("a.py",))
    source.write_text("two\n", encoding="utf-8")
    after = gate.source_tree_sha256(tmp_path, ("a.py",))

    assert before != after


def test_source_tree_hash_is_independent_of_inventory_order(
    tmp_path,
):
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )
    (tmp_path / "a.py").write_text(
        "a\n",
        encoding="utf-8",
    )
    (tmp_path / "b.py").write_text(
        "b\n",
        encoding="utf-8",
    )

    assert gate.source_tree_sha256(
        tmp_path,
        ("b.py", "a.py"),
    ) == gate.source_tree_sha256(
        tmp_path,
        ("a.py", "b.py"),
    )


def test_atomic_write_json_replaces_complete_payload(tmp_path):
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )
    path = tmp_path / "result.json"
    path.write_text('{"old":true}\n', encoding="utf-8")

    gate.atomic_write_json(path, {"new": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "new": True
    }
    assert not (tmp_path / ".result.json.tmp").exists()


class FakeTokenizer:
    name_or_path = "qwen3-fixture"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [
            (ord(character) % 97) + 1
            for character in text
        ]


class FakeTP4Engine:
    def __init__(self, *, rank_exit_codes=None):
        self.calls = []
        self.tokenizer = FakeTokenizer()
        self.config = SimpleNamespace(dtype="torch.float16")
        self.model_runner = SimpleNamespace(
            world_size=4,
            config=self.config,
        )
        self.ps = [object(), object(), object()]
        self.rank_exit_codes = (
            [0, 0, 0, 0]
            if rank_exit_codes is None
            else list(rank_exit_codes)
        )
        self._kv_snapshot_index = 0

    def activate_speculative_runtime(self, runtime):
        self.calls.append("activate_runtime")
        self.runtime = runtime

    def configure_decode_internal_profile(
        self,
        enabled,
        profile_label,
        *,
        timeout_s,
    ):
        assert enabled is True
        assert profile_label
        assert timeout_s == 60.0
        self.calls.append("configure_profile")
        return {
            "enabled": True,
            "rank_inventory": [0, 1, 2, 3],
        }

    def clear_reusable_prefix_cache(self):
        self.calls.append("clear_reusable_prefix_cache")

    def kv_offload_summaries(self, *, timeout_s):
        assert timeout_s == 60.0
        label = (
            "kv_before"
            if self._kv_snapshot_index == 0
            else "kv_after"
        )
        self.calls.append(label)
        offset = self._kv_snapshot_index
        self._kv_snapshot_index += 1
        return tuple({
            name: (
                rank + offset
                if name
                != "speculative_residency_rejected_d2h_copies"
                else 0
            )
            for name in _load_module(
                "generic_speculative_tp4_gate_test_module",
                GATE_PATH,
            ).MOVEMENT_KEYS
        } for rank in range(4))

    def _call_speculative_residency_phase(
        self,
        method_name,
        ticket_id,
        *args,
        **kwargs,
    ):
        operation = kwargs["expected_operation"]
        status = kwargs["expected_status"]
        self.calls.append(f"residency_{operation}")
        return tuple(
            _residency_phase(
                operation,
                ticket_id=ticket_id,
            )["rows"]
        )

    def finalize_decode_internal_profile(self, *, timeout_s):
        assert timeout_s == 60.0
        self.calls.append("finalize_profile")
        return _valid_rank_profile(policy="ngram")

    def exit(self):
        self.calls.append("exit")
        receipt = _valid_cleanup_receipt()
        receipt["rank_exit_codes"] = list(
            self.rank_exit_codes
        )
        return receipt


class FakeRuntime:
    def __init__(self, adapter):
        self.adapter = adapter


class FakeAdapter:
    def __init__(self, *, ngram_size, max_proposal_tokens):
        self.ngram_size = ngram_size
        self.max_proposal_tokens = max_proposal_tokens


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


def _fake_generation_runner():
    calls = {"count": 0}

    def run_generation(**kwargs):
        calls["count"] += 1
        engine = kwargs["engine"]
        prompt_rows = kwargs["prompt_rows"]
        engine.calls.append(
            "warmup_generation"
            if calls["count"] == 1
            else "recorded_generation"
        )
        if calls["count"] == 2:
            for operation in (
                "prepare",
                "precommit",
                "seal",
            ):
                status = _residency_status(operation)
                engine._call_speculative_residency_phase(
                    f"{operation}_speculative_residency_batch",
                    7,
                    expected_operation=operation,
                    expected_status=status,
                    expected_sequence_ids=tuple(
                        range(len(prompt_rows))
                    ),
                    expected_committed_block_identities=(),
                    expected_rejected_block_identities=(),
                    timeout_s=60.0,
                )
        return {
            "outputs": [
                [
                    1000 + prompt_index * 10 + token_index
                    for token_index in range(8)
                ]
                for prompt_index in range(len(prompt_rows))
            ],
            "runtime": _runtime_summary(
                "ngram",
                len(prompt_rows),
            ),
        }

    return run_generation


def _load_worker():
    return _load_module(
        "generic_speculative_tp4_worker_test_module",
        WORKER_PATH,
    )


def _run_fake_worker_cell(engine):
    worker = _load_worker()
    return worker.run_policy_cell(
        model_path="/model",
        gpu_indices=(0, 1, 2, 3),
        policy="ngram",
        batch_size=1,
        dist_port=29001,
        master_port=29002,
        engine_factory=lambda *args, **kwargs: engine,
        sampling_params_type=FakeSamplingParams,
        runtime_type=FakeRuntime,
        adapter_type=FakeAdapter,
        synchronize=lambda: None,
        run_generation_fn=_fake_generation_runner(),
    )


def test_worker_captures_profile_residency_kv_and_cleanup():
    engine = FakeTP4Engine()

    cell = _run_fake_worker_cell(engine)

    assert engine.calls == [
        "activate_runtime",
        "configure_profile",
        "warmup_generation",
        "clear_reusable_prefix_cache",
        "kv_before",
        "recorded_generation",
        "residency_prepare",
        "residency_precommit",
        "residency_seal",
        "kv_after",
        "finalize_profile",
        "exit",
    ]
    assert [
        row["rank"] for row in cell["kv_rank_deltas"]
    ] == [0, 1, 2, 3]
    assert [
        row["operation"]
        for row in cell["residency_phases"]
    ] == ["prepare", "precommit", "seal"]
    assert cell["cleanup_receipt"][
        "process_group_destroyed"
    ] is True


def test_worker_restores_distributed_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "9")
    monkeypatch.setenv("TINYVLLM_DIST_PORT", "19991")
    monkeypatch.setenv("MASTER_PORT", "19992")

    _run_fake_worker_cell(FakeTP4Engine())

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "9"
    assert os.environ["TINYVLLM_DIST_PORT"] == "19991"
    assert os.environ["MASTER_PORT"] == "19992"


def test_worker_does_not_publish_cell_without_clean_exit():
    engine = FakeTP4Engine(
        rank_exit_codes=[0, 1, 0, 0]
    )

    with pytest.raises(ValueError, match="rank exit codes"):
        _run_fake_worker_cell(engine)


def _fake_worker_subprocess(calls):
    def run(command, **kwargs):
        policy = command[
            command.index("--policy") + 1
        ]
        batch_size = int(
            command[
                command.index("--batch-size") + 1
            ]
        )
        dist_port = int(
            command[
                command.index("--dist-port") + 1
            ]
        )
        master_port = int(
            command[
                command.index("--master-port") + 1
            ]
        )
        output_path = Path(
            command[command.index("--out") + 1]
        )
        calls.append({
            "cell": f"{policy}:b{batch_size}",
            "dist_port": dist_port,
            "master_port": master_port,
        })
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_text(
            json.dumps(
                _valid_cell(policy, batch_size),
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    return run


def test_campaign_runs_each_cell_in_a_fresh_subprocess(
    tmp_path,
    monkeypatch,
):
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )
    calls = []
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        _fake_worker_subprocess(calls),
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type":"qwen3"}\n',
        encoding="utf-8",
    )

    result = gate.run_campaign(
        model_path=str(model_dir),
        gpu_indices=(0, 1, 2, 3),
        output_dir=tmp_path / "run",
        dist_port_base=29100,
        master_port_base=29200,
        repo_root=ROOT,
        verifier=lambda run_dir, source_root: {
            "classification": "PASS",
            "failures": [],
        },
    )

    assert [call["cell"] for call in calls] == [
        "baseline:b1",
        "ngram:b1",
        "baseline:b4",
        "ngram:b4",
    ]
    assert len({
        call["dist_port"]
        for call in calls
    }) == 4
    assert len({
        call["master_port"]
        for call in calls
    }) == 4
    assert result["classification"] == "NOT_PROMOTABLE"
    assert (tmp_path / "run" / "verify.json").is_file()


def test_campaign_preserves_failed_verification_artifacts(
    tmp_path,
    monkeypatch,
):
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        _fake_worker_subprocess([]),
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type":"qwen3"}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="sentinel verification failure",
    ):
        gate.run_campaign(
            model_path=str(model_dir),
            gpu_indices=(0, 1, 2, 3),
            output_dir=tmp_path / "run",
            dist_port_base=29100,
            master_port_base=29200,
            repo_root=ROOT,
            verifier=lambda run_dir, source_root: {
                "classification": "FAIL",
                "failures": [
                    "sentinel verification failure"
                ],
            },
        )

    failed_dir = tmp_path / "run.failed"
    assert (failed_dir / "verify.json").is_file()
    assert not (tmp_path / "run").exists()


def _write_valid_run(tmp_path):
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = _valid_result()
    gate.atomic_write_json(run_dir / "result.json", result)
    result_sha256 = hashlib.sha256(
        (run_dir / "result.json").read_bytes()
    ).hexdigest()
    gate.atomic_write_json(
        run_dir / "source_manifest.json",
        {
            "schema_version": 1,
            "source_tree_sha256": (
                result["source_tree_sha256"]
            ),
            "model_manifest_sha256": (
                result["model_manifest_sha256"]
            ),
            "source_files": {
                "tinyvllm/engine/model_runner.py": (
                    "e" * 64
                ),
            },
            "artifacts": {
                "result.json": result_sha256,
            },
        },
    )
    return run_dir


def _rewrite_result_and_manifest(run_dir, result):
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )
    gate.atomic_write_json(run_dir / "result.json", result)
    manifest = json.loads(
        (run_dir / "source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest["artifacts"]["result.json"] = hashlib.sha256(
        (run_dir / "result.json").read_bytes()
    ).hexdigest()
    gate.atomic_write_json(
        run_dir / "source_manifest.json",
        manifest,
    )


def _load_verifier():
    return _load_module(
        "verify_generic_speculative_tp4_gate_test_module",
        VERIFIER_PATH,
    )


def test_verifier_rejects_tampered_collective(tmp_path):
    run_dir = _write_valid_run(tmp_path)
    result = json.loads(
        (run_dir / "result.json").read_text(
            encoding="utf-8"
        )
    )
    result["cells"]["ngram:b1"]["profile"]["ranks"][
        2
    ]["collectives"][0]["operation"] = "tampered"
    _rewrite_result_and_manifest(run_dir, result)

    verification = _load_verifier().verify_run(run_dir)

    assert verification["classification"] == "FAIL"
    assert "collective identity" in verification[
        "failures"
    ][0]


def test_verifier_rejects_source_hash_mismatch(tmp_path):
    run_dir = _write_valid_run(tmp_path)
    gate = _load_module(
        "generic_speculative_tp4_gate_test_module",
        GATE_PATH,
    )
    manifest = json.loads(
        (run_dir / "source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest["source_tree_sha256"] = "0" * 64
    gate.atomic_write_json(
        run_dir / "source_manifest.json",
        manifest,
    )

    verification = _load_verifier().verify_run(run_dir)

    assert verification["classification"] == "FAIL"
    assert "source tree identity mismatch" in verification[
        "failures"
    ][0]


def test_remote_runner_uses_required_host_auth_and_no_controlmaster():
    text = RUNNER_PATH.read_text(encoding="utf-8")

    assert "sitian@10.232.195.203" in text
    assert "FILE:/Users/bytedance/krb5cc_sitian" in text
    assert "ControlMaster=no" in text
    assert "ControlPath=none" in text
    assert "GSSAPIAuthentication=yes" in text
    assert "nvidia-smi" in text
    assert "memory.free" in text
    assert "CUDA_VISIBLE_DEVICES" in text
    assert "verify_generic_speculative_tp4_gate.py" in text
    assert "last_completed_run_path.txt" in text
    assert "STALE_RECHECK_ATTEMPTS" in text
    assert "REMOTE_COMMAND_RETRY_ATTEMPTS" in text
    assert "retry_remote_command" in text
    assert "retry_remote_rsync" in text
    assert (
        'REMOTE_COMMAND_RETRY_ATTEMPTS="${'
        'REMOTE_COMMAND_RETRY_ATTEMPTS:-5}"'
    ) in text
    assert (
        'REMOTE_COMMAND_RETRY_INTERVAL_SECONDS="${'
        'REMOTE_COMMAND_RETRY_INTERVAL_SECONDS:-3}"'
    ) in text
    assert "/proc/sys/net/ipv4/ip_local_port_range" in text
    assert (
        "'/proc/sys/net/ipv4/ip_local_port_range'"
        in text
    )
    assert "ephemeral_start" in text
