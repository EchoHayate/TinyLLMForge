from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = (
    ROOT
    / "tools"
    / "qwen35_native_mtp_tp1_4k_engine_gate.py"
)
WORKER_PATH = (
    ROOT
    / "tools"
    / "qwen35_native_mtp_tp1_4k_engine_worker.py"
)
VERIFIER_PATH = (
    ROOT
    / "tools"
    / "verify_qwen35_native_mtp_tp1_4k_engine_gate.py"
)
RUNNER_PATH = (
    ROOT
    / "tools"
    / "run_qwen35_native_mtp_tp1_4k_engine_gate_remote.sh"
)

TARGET_MODEL_MANIFEST_SHA256 = (
    "3e650a908234771c3cf1ac4e20c4d38fe"
    "69982efedaf4a3e631ad0b14aad7dd0"
)
MTP_CHECKPOINT_MANIFEST_SHA256 = (
    "9a975bdcf0383774183cae560594dd60"
    "b522b83fe9c4cd595c47c12e2403702b"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_gate(name: str):
    return _load_module(name, GATE_PATH)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _token_rows(
    batch_size: int,
    *,
    token_count: int,
    start: int,
) -> list[dict]:
    rows = []
    for prompt_index in range(batch_size):
        token_ids = [
            start + prompt_index + offset % 7
            for offset in range(token_count)
        ]
        rows.append({
            "prompt_index": prompt_index,
            "token_count": token_count,
            "token_ids": token_ids,
            "sha256": _digest(token_ids),
        })
    return rows


def _native_runtime(batch_size: int) -> dict:
    finalize_receipts = []
    side_state_receipts = []
    proposal_kv_receipts = []
    lifecycle_events = []
    for sequence_id in range(batch_size):
        transaction_id = f"proposal-tx-{sequence_id}"
        finalize_receipts.extend([
            {
                "sequence_id": sequence_id,
                "transaction_id": transaction_id,
                "operation": "prepare",
            },
            {
                "sequence_id": sequence_id,
                "transaction_id": transaction_id,
                "operation": "commit",
            },
        ])
        side_state_receipts.extend(
            {
                "sequence_id": sequence_id,
                "transaction_id": f"side-tx-{sequence_id}",
                "operation": operation,
            }
            for operation in (
                "prepare",
                "select",
                "apply",
                "seal",
            )
        )
        proposal_kv_receipts.append({
            "sequence_id": sequence_id,
            "transaction_id": transaction_id,
            "accepted_token_count": 2,
            "rejected_token_count": 2,
            "accepted_slot_identity_preserved": True,
            "rejected_slots_released": True,
        })
        lifecycle_events.extend(
            {
                "sequence_id": sequence_id,
                "operation": operation,
            }
            for operation in (
                "proposal_finalize_prepare",
                "side_state_apply",
                "target_kv_commit",
                "scheduler_commit",
                "proposal_finalize_commit",
                "side_state_seal",
                "proposal_sequence_release",
            )
        )
    return {
        "native_binding": {
            "executor_id": "native_checkpoint_proposal",
            "source_type": "native_model_runner",
            "module_type": "Qwen35NativeMTP",
            "physical_store_type": "Qwen35MTPPhysicalSlotStore",
            "checkpoint_tensor_count": 15,
        },
        "proposal_rows": batch_size,
        "proposed_tokens": batch_size * 4,
        "accepted_draft_tokens": batch_size * 2,
        "rejected_draft_tokens": batch_size * 2,
        "first_target_callbacks": 1,
        "verify_callbacks": 1,
        "first_target_target_forwards": 1,
        "verify_target_forwards": 1,
        "accepted_prefix_target_replays": 0,
        "proposal_finalize_receipts": finalize_receipts,
        "side_state_receipts": side_state_receipts,
        "proposal_kv_receipts": proposal_kv_receipts,
        "lifecycle_events": lifecycle_events,
    }


def _baseline_runtime() -> dict:
    return {
        "native_binding": None,
        "proposal_rows": 0,
        "proposed_tokens": 0,
        "accepted_draft_tokens": 0,
        "rejected_draft_tokens": 0,
        "first_target_callbacks": 0,
        "verify_callbacks": 0,
        "first_target_target_forwards": 0,
        "verify_target_forwards": 0,
        "accepted_prefix_target_replays": 0,
        "proposal_finalize_receipts": [],
        "side_state_receipts": [],
        "proposal_kv_receipts": [],
        "lifecycle_events": [],
    }


def _cell(policy: str, batch_size: int) -> dict:
    return {
        "schema_version": (
            "qwen35.native-mtp-tp1-4k-engine-"
            "transactional-correctness.v1"
        ),
        "policy": policy,
        "batch_size": batch_size,
        "world_size": 1,
        "gpu_index": 0,
        "prompt_token_count": 4096,
        "max_output_tokens": 32,
        "max_proposal_tokens": 4,
        "model_identity": {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForCausalLM"],
            "target_model_manifest_sha256": (
                TARGET_MODEL_MANIFEST_SHA256
            ),
            "mtp_checkpoint_manifest_sha256": (
                MTP_CHECKPOINT_MANIFEST_SHA256
            ),
        },
        "prompt_rows": _token_rows(
            batch_size,
            token_count=4096,
            start=17,
        ),
        "output_rows": _token_rows(
            batch_size,
            token_count=32,
            start=101,
        ),
        "runtime": (
            _baseline_runtime()
            if policy == "baseline"
            else _native_runtime(batch_size)
        ),
        "cleanup": {
            "proposal_transactions_open": [],
            "proposal_finalize_tickets_open": [],
            "proposal_sequence_ids": [],
            "proposal_kv_slots_in_use": 0,
            "native_state_snapshot": (
                None
                if policy == "baseline"
                else {
                    "pending_prefix_count": 0,
                    "bootstrapped_sequence_count": 0,
                    "proposal_transaction_count": 0,
                    "batch_ticket_count": 0,
                    "batch_ticket_transaction_count": 0,
                    "allocated_physical_slot_count": 0,
                }
            ),
            "hybrid_state_leases_before": 0,
            "hybrid_state_leases_after": 0,
            "owned_children_remaining": [],
            "engine_exit_called": True,
            "worker_exit_code": 0,
        },
        "runtime_poisoned": False,
    }


def _result() -> dict:
    cells = {
        f"{policy}:b{batch_size}": _cell(
            policy,
            batch_size,
        )
        for batch_size in (1, 4)
        for policy in ("baseline", "native_mtp")
    }
    return {
        "schema_version": (
            "qwen35.native-mtp-tp1-4k-engine-"
            "transactional-correctness.v1"
        ),
        "classification": (
            "QWEN35_NATIVE_MTP_TP1_4K_ENGINE_ESTABLISHED"
        ),
        "promotion_classification": "NOT_PROMOTABLE",
        "target_model_manifest_sha256": (
            TARGET_MODEL_MANIFEST_SHA256
        ),
        "mtp_checkpoint_manifest_sha256": (
            MTP_CHECKPOINT_MANIFEST_SHA256
        ),
        "source_tree_sha256": "a" * 64,
        "world_size": 1,
        "gpu_index": 0,
        "cells": cells,
        "parity": {"b1": True, "b4": True},
        "limitations": [
            "TP1 only",
            "4K prompt only",
            "KV offload disabled",
            "eager native MTP only",
            "not production ready",
        ],
    }


def test_contract_constants_are_frozen():
    gate = _load_gate("native_mtp_engine_gate_constants")

    assert gate.SCHEMA_VERSION == (
        "qwen35.native-mtp-tp1-4k-engine-"
        "transactional-correctness.v1"
    )
    assert gate.CLASSIFICATION == (
        "QWEN35_NATIVE_MTP_TP1_4K_ENGINE_ESTABLISHED"
    )
    assert gate.PROMOTION_CLASSIFICATION == "NOT_PROMOTABLE"
    assert gate.POLICIES == ("baseline", "native_mtp")
    assert gate.BATCH_SIZES == (1, 4)
    assert gate.PROMPT_TOKENS == 4096
    assert gate.MAX_OUTPUT_TOKENS == 32
    assert gate.MAX_PROPOSAL_TOKENS == 4
    assert gate.WORLD_SIZE == 1
    assert (
        gate.TARGET_MODEL_MANIFEST_SHA256
        == TARGET_MODEL_MANIFEST_SHA256
    )
    assert (
        gate.MTP_CHECKPOINT_MANIFEST_SHA256
        == MTP_CHECKPOINT_MANIFEST_SHA256
    )


def test_complete_authority_is_accepted():
    gate = _load_gate("native_mtp_engine_gate_accept")

    normalized = gate.validate_result(_result())

    assert normalized["parity"] == {"b1": True, "b4": True}
    assert normalized["promotion_classification"] == (
        "NOT_PROMOTABLE"
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda result: result["cells"].pop("baseline:b1"),
            "cell inventory",
        ),
        (
            lambda result: result.update(
                promotion_classification="PRODUCTION_READY"
            ),
            "promotion",
        ),
        (
            lambda result: result.update(
                target_model_manifest_sha256="0" * 64
            ),
            "target model manifest",
        ),
        (
            lambda result: result.update(
                mtp_checkpoint_manifest_sha256="0" * 64
            ),
            "MTP checkpoint manifest",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"].update(
                prompt_token_count=2048
            ),
            "prompt token",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"].update(
                max_output_tokens=31
            ),
            "output token",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "prompt_rows"
            ][0]["token_ids"].pop(),
            "prompt",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "output_rows"
            ][0]["token_ids"].append(7),
            "output",
        ),
        (
            lambda result: result["cells"]["baseline:b1"]["runtime"].update(
                proposed_tokens=1
            ),
            "baseline speculative",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"]["runtime"].update(
                native_binding=None
            ),
            "native binding",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"]["runtime"][
                "native_binding"
            ].update(executor_id="other"),
            "executor",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"]["runtime"][
                "native_binding"
            ].update(module_type="other"),
            "module",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"]["runtime"][
                "native_binding"
            ].update(physical_store_type="other"),
            "physical store",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"]["runtime"].update(
                proposed_tokens=0
            ),
            "proposed",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"]["runtime"].update(
                accepted_draft_tokens=0
            ),
            "accepted",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"]["runtime"].update(
                rejected_draft_tokens=0
            ),
            "rejected",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"]["runtime"].update(
                first_target_callbacks=0
            ),
            "first-target",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"]["runtime"].update(
                verify_callbacks=0
            ),
            "verify",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"]["runtime"].update(
                first_target_target_forwards=2
            ),
            "first-target target forward",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"]["runtime"].update(
                verify_target_forwards=2
            ),
            "verify target forward",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"]["runtime"].update(
                accepted_prefix_target_replays=1
            ),
            "accepted-prefix",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"]["runtime"][
                "proposal_finalize_receipts"
            ].pop(),
            "finalize",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"]["runtime"][
                "side_state_receipts"
            ].pop(),
            "side-state",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"]["runtime"][
                "proposal_kv_receipts"
            ][0].update(rejected_slots_released=False),
            "rejected slots",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"]["runtime"][
                "lifecycle_events"
            ].reverse(),
            "lifecycle",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "cleanup"
            ]["proposal_transactions_open"].append("tx"),
            "transaction leak",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "cleanup"
            ]["proposal_finalize_tickets_open"].append("ticket"),
            "ticket leak",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "cleanup"
            ]["proposal_sequence_ids"].append(7),
            "sequence leak",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "cleanup"
            ].update(proposal_kv_slots_in_use=1),
            "slot leak",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"].update(
                runtime_poisoned=True
            ),
            "poisoned",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "cleanup"
            ].update(engine_exit_called=False),
            "cleanup",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "output_rows"
            ].__setitem__(
                0,
                _token_rows(
                    1,
                    token_count=32,
                    start=777,
                )[0],
            ),
            "parity",
        ),
    ),
)
def test_invalid_authority_is_rejected(mutate, match):
    gate = _load_gate("native_mtp_engine_gate_reject")
    result = _result()
    mutate(result)

    with pytest.raises(ValueError, match=match):
        gate.validate_result(result)


def test_assemble_authority_recomputes_parity():
    gate = _load_gate("native_mtp_engine_gate_assemble")
    result = _result()

    assembled = gate.assemble_authority(
        result["cells"],
        source_tree_sha256=result["source_tree_sha256"],
        target_model_manifest_sha256=(
            result["target_model_manifest_sha256"]
        ),
        mtp_checkpoint_manifest_sha256=(
            result["mtp_checkpoint_manifest_sha256"]
        ),
        gpu_index=0,
        limitations=result["limitations"],
    )

    assert assembled == gate.validate_result(result)


def test_source_tree_digest_binds_names_and_contents(tmp_path):
    gate = _load_gate("native_mtp_engine_gate_source")
    (tmp_path / "a.py").write_text("a\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("b\n", encoding="utf-8")

    first = gate.source_tree_sha256(
        tmp_path,
        ("a.py", "b.py"),
    )
    second = gate.source_tree_sha256(
        tmp_path,
        ("b.py", "a.py"),
    )
    (tmp_path / "a.py").write_text("changed\n", encoding="utf-8")
    changed = gate.source_tree_sha256(
        tmp_path,
        ("a.py", "b.py"),
    )

    assert first == second
    assert changed != first


def test_default_source_inventory_covers_remote_python_bundle():
    gate = _load_gate(
        "native_mtp_engine_complete_source_inventory"
    )
    expected = tuple(sorted(
        [
            str(path.relative_to(ROOT))
            for path in (ROOT / "tinyvllm").rglob("*.py")
        ]
        + [
            str(GATE_PATH.relative_to(ROOT)),
            str(WORKER_PATH.relative_to(ROOT)),
            str(VERIFIER_PATH.relative_to(ROOT)),
        ]
    ))

    assert gate.DEFAULT_SOURCE_FILES == expected


def test_run_campaign_rejects_source_change_during_execution(
    tmp_path,
    monkeypatch,
):
    gate = _load_gate(
        "native_mtp_engine_campaign_source_snapshot"
    )
    repo_root = tmp_path / "source"
    repo_root.mkdir()
    source_path = repo_root / "engine.py"
    source_path.write_text("engine-v1\n", encoding="utf-8")
    worker_script = repo_root / "worker.py"
    worker_script.write_text("", encoding="utf-8")
    worker_calls = 0

    def fake_run(command, **kwargs):
        nonlocal worker_calls
        assert kwargs["cwd"] == repo_root
        policy = command[command.index("--policy") + 1]
        batch_size = int(
            command[command.index("--batch-size") + 1]
        )
        output_path = Path(
            command[command.index("--out") + 1]
        )
        gate.atomic_write_json(
            output_path,
            _cell(policy, batch_size),
        )
        worker_calls += 1
        if worker_calls == 1:
            source_path.write_text(
                "engine-v2\n",
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    worker_module = SimpleNamespace(
        target_model_manifest_sha256=(
            lambda _model_path: TARGET_MODEL_MANIFEST_SHA256
        ),
        mtp_checkpoint_manifest_sha256=(
            lambda _model_path: MTP_CHECKPOINT_MANIFEST_SHA256
        ),
    )
    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    monkeypatch.setattr(
        gate,
        "_load_module",
        lambda _name, _path: worker_module,
    )

    with pytest.raises(
        RuntimeError,
        match="source changed during campaign",
    ):
        gate.run_campaign(
            model_path="/checkpoint",
            gpu_index=0,
            output_dir=tmp_path / "authority",
            repo_root=repo_root,
            worker_script=worker_script,
            source_files=("engine.py",),
            verifier=lambda _run_dir, _source_root: {
                "classification": "PASS",
                "failures": [],
            },
        )


def test_publish_authority_is_atomic_and_complete(tmp_path):
    gate = _load_gate("native_mtp_engine_gate_publish")
    output_dir = tmp_path / "authority"
    result = gate.validate_result(_result())

    gate.publish_authority(
        output_dir,
        result,
        source_files={
            "tinyvllm/engine/llm_engine.py": "b" * 64,
        },
    )

    assert json.loads(
        (output_dir / "result.json").read_text()
    ) == result
    manifest = json.loads(
        (output_dir / "source_manifest.json").read_text()
    )
    status = json.loads(
        (output_dir / "status.json").read_text()
    )
    assert manifest["source_tree_sha256"] == (
        result["source_tree_sha256"]
    )
    assert manifest["artifacts"]["result.json"] == (
        gate.sha256_file(output_dir / "result.json")
    )
    assert status == {
        "schema_version": gate.SCHEMA_VERSION,
        "status": "PASS",
        "classification": gate.CLASSIFICATION,
        "promotion_classification": "NOT_PROMOTABLE",
    }
    assert not tuple(tmp_path.glob(".authority.*"))


def test_publish_authority_refuses_existing_output(tmp_path):
    gate = _load_gate("native_mtp_engine_gate_existing")
    output_dir = tmp_path / "authority"
    output_dir.mkdir()

    with pytest.raises(ValueError, match="already exists"):
        gate.publish_authority(
            output_dir,
            gate.validate_result(deepcopy(_result())),
            source_files={},
        )


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        seed = sum(ord(character) for character in text) % 100
        return [seed + offset + 1 for offset in range(8)]


class _FakeRuntime:
    def __init__(self, *, model_runner_executor):
        self.model_runner_executor = model_runner_executor


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


class Qwen35MTPPhysicalSlotStore:
    def __init__(self):
        self._allocated_slot_ids = set()


class Qwen35NativeMTP:
    pass


class _FakeExecutor:
    def __init__(self, module, physical_store):
        self.module = module
        self._pending_prefixes = {}
        self._bootstrapped = {}
        self._proposal_transactions = {}
        self._batch_tickets = {}
        self._batch_ticket_transactions = {}
        self.proposal_kv_cache = SimpleNamespace(
            physical_store=physical_store
        )


class _FakeModelRunner:
    def __init__(self, *, native=True):
        self.calls = []
        self.qwen35_mtp_registration_error = None
        if not native:
            self.qwen35_mtp_executor_descriptor = None
            self.qwen35_mtp_module = None
            self.qwen35_mtp_physical_store = None
            self.qwen35_mtp_executor = None
            return
        module = Qwen35NativeMTP()
        physical_store = Qwen35MTPPhysicalSlotStore()
        executor = _FakeExecutor(module, physical_store)
        self.qwen35_mtp_module = module
        self.qwen35_mtp_physical_store = physical_store
        self.qwen35_mtp_executor = executor
        self.qwen35_mtp_executor_descriptor = SimpleNamespace(
            executor_id="native_checkpoint_proposal",
            capabilities=SimpleNamespace(
                source_type="native_model_runner",
                requires_proposal_lifecycle=True,
            ),
        )

    @staticmethod
    def _invoke(method_name, *args):
        if method_name == (
            "prepare_speculative_proposal_finalize_batch"
        ):
            return "proposal-ticket-1"
        if method_name.endswith(
            "speculative_side_state_batch"
        ):
            operation = method_name.removesuffix(
                "_speculative_side_state_batch"
            )
            statuses = {
                "prepare": "prepared",
                "select": "selected",
                "apply": "applied",
                "seal": "sealed",
                "rollback": "rolled_back",
            }
            return {
                "operation": operation,
                "status": statuses[operation],
                "transaction_id": "side-ticket-1",
                "sequence_ids": [0],
            }
        return None

    def call(self, method_name, *args, **kwargs):
        self.calls.append((method_name, args))
        return self._invoke(method_name, *args)

    def prepare_speculative_side_state_batch(self, *args):
        return self._invoke(
            "prepare_speculative_side_state_batch",
            *args,
        )

    def select_speculative_side_state_batch(self, *args):
        return self._invoke(
            "select_speculative_side_state_batch",
            *args,
        )

    def apply_speculative_side_state_batch(self, *args):
        return self._invoke(
            "apply_speculative_side_state_batch",
            *args,
        )

    def seal_speculative_side_state_batch(self, *args):
        return self._invoke(
            "seal_speculative_side_state_batch",
            *args,
        )


class _FakeAllocator:
    def observation_snapshot(self):
        return {"used_slots": 0, "owners": {}}


class _FakeEngine:
    def __init__(
        self,
        *,
        batch_size,
        native,
        output_count=32,
        step_error=None,
        accepted_prefix_replay=False,
        extra_spec_verify_forward=False,
        side_state_via_acknowledged=False,
    ):
        self.tokenizer = _FakeTokenizer()
        self.model_runner = _FakeModelRunner(native=native)
        self.scheduler = SimpleNamespace(
            hybrid_state_allocator=_FakeAllocator(),
            block_manager=SimpleNamespace(
                commit_speculative_kv_commit_batch=(
                    lambda plans: None
                )
            ),
            commit_prepared_postprocess=(
                lambda prepared: None
            ),
        )
        self.config = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="qwen3_5",
                architectures=["Qwen3_5ForCausalLM"],
            )
        )
        self.speculative_runtime_poisoned = False
        self.last_step_observation = None
        self.batch_size = batch_size
        self.native = native
        self.output_count = output_count
        self.step_error = step_error
        self.accepted_prefix_replay = accepted_prefix_replay
        self.extra_spec_verify_forward = (
            extra_spec_verify_forward
        )
        self.side_state_via_acknowledged = (
            side_state_via_acknowledged
        )
        self.requests = []
        self.activated_runtime = None
        self.finished = False
        self.exit_called = False

    def activate_speculative_runtime(self, runtime):
        self.activated_runtime = runtime

    def add_request(self, token_ids, sampling_params):
        self.requests.append((list(token_ids), sampling_params))

    def is_finished(self):
        return self.finished

    def call_model_runner_acknowledged(
        self,
        method_name,
        *args,
        timeout_s,
    ):
        assert timeout_s == 60.0
        return (
            getattr(self.model_runner, method_name)(*args),
            (),
        )

    def step(self):
        if self.step_error is not None:
            raise self.step_error
        if self.native:
            self.model_runner.call(
                "run_spec_first_target_and_proposal_batch",
                tuple(
                    SimpleNamespace(seq_id=sequence_id)
                    for sequence_id in range(self.batch_size)
                ),
                self.model_runner.qwen35_mtp_executor_descriptor,
                (),
            )
            self.model_runner.call(
                "run_spec_verify_batch",
                (),
            )
            if self.extra_spec_verify_forward:
                self.model_runner.call(
                    "run_spec_verify_batch",
                    (),
                )
            rows = tuple(
                SimpleNamespace(
                    sequence_id=sequence_id,
                    proposal_transaction_id=(
                        f"proposal-tx-{sequence_id}"
                    ),
                    accepted_proposal_tokens=2,
                )
                for sequence_id in range(self.batch_size)
            )
            self.model_runner.call(
                "observe_speculative_target_prefill_batch",
                "native_checkpoint_proposal",
                rows,
            )
            call_side_state = (
                (
                    lambda method_name, *args:
                    self.call_model_runner_acknowledged(
                        method_name,
                        *args,
                        timeout_s=60.0,
                    )[0]
                )
                if self.side_state_via_acknowledged
                else self.model_runner.call
            )
            call_side_state(
                "prepare_speculative_side_state_batch",
                tuple(
                    SimpleNamespace(seq_id=sequence_id)
                    for sequence_id in range(self.batch_size)
                ),
            )
            call_side_state(
                "select_speculative_side_state_batch",
                rows,
            )
            ticket = self.model_runner.call(
                "prepare_speculative_proposal_finalize_batch",
                "native_checkpoint_proposal",
                rows,
            )
            call_side_state(
                "apply_speculative_side_state_batch"
            )
            plans = tuple(
                SimpleNamespace(sequence_id=sequence_id)
                for sequence_id in range(self.batch_size)
            )
            self.scheduler.block_manager.commit_speculative_kv_commit_batch(
                plans
            )
            self.scheduler.commit_prepared_postprocess(object())
            self.model_runner.call(
                "commit_speculative_proposal_finalize_batch",
                "native_checkpoint_proposal",
                ticket,
            )
            call_side_state(
                "seal_speculative_side_state_batch"
            )
            for sequence_id in range(self.batch_size):
                self.model_runner.call(
                    "release_speculative_proposal_sequence",
                    "native_checkpoint_proposal",
                    sequence_id,
                    0,
                )
            proposal_counts = {
                sequence_id: 4
                for sequence_id in range(self.batch_size)
            }
            accepted_counts = {
                sequence_id: 2
                for sequence_id in range(self.batch_size)
            }
            if self.accepted_prefix_replay:
                self.model_runner.call(
                    "run",
                    tuple(
                        SimpleNamespace(seq_id=sequence_id)
                        for sequence_id in range(self.batch_size)
                    ),
                    False,
                    True,
                    None,
                )
        else:
            proposal_counts = {}
            accepted_counts = {}
        self.last_step_observation = {
            "speculative_proposal_row_count": (
                self.batch_size if self.native else 0
            ),
            "speculative_proposal_token_counts": (
                proposal_counts
            ),
            "speculative_accepted_draft_token_counts": (
                accepted_counts
            ),
            "speculative_first_target_callback_count": (
                1 if self.native else 0
            ),
            "speculative_fixed_q_group_count": (
                1 if self.native else 0
            ),
        }
        self.finished = True
        return (
            [
                (
                    sequence_id,
                    [
                        101 + sequence_id + offset % 7
                        for offset in range(self.output_count)
                    ],
                )
                for sequence_id in range(self.batch_size)
            ],
            -1,
        )

    def flush_pending_hybrid_state_releases(self, *, timeout_s):
        assert timeout_s == 60.0

    def exit(self):
        self.exit_called = True
        return {
            "owned_children_remaining": [],
        }


def _worker_dependencies(engine):
    return {
        "engine_factory": lambda *args, **kwargs: engine,
        "sampling_params_type": _FakeSamplingParams,
        "runtime_type": _FakeRuntime,
        "synchronize": lambda: None,
        "target_manifest_resolver": (
            lambda path: TARGET_MODEL_MANIFEST_SHA256
        ),
        "mtp_manifest_resolver": (
            lambda path: MTP_CHECKPOINT_MANIFEST_SHA256
        ),
    }


def test_worker_builds_exact_distinct_deterministic_prompts():
    worker = _load_module(
        "native_mtp_engine_worker_prompts",
        WORKER_PATH,
    )
    tokenizer = _FakeTokenizer()

    first = worker.build_prompt_rows(tokenizer, 4)
    second = worker.build_prompt_rows(tokenizer, 4)

    assert first == second
    assert [row["prompt_index"] for row in first] == [
        0,
        1,
        2,
        3,
    ]
    assert all(row["token_count"] == 4096 for row in first)
    assert all(len(row["token_ids"]) == 4096 for row in first)
    assert len({row["sha256"] for row in first}) == 4


def test_worker_native_activates_runtime_and_records_transactions():
    worker = _load_module(
        "native_mtp_engine_worker_native",
        WORKER_PATH,
    )
    engine = _FakeEngine(batch_size=1, native=True)
    factory_calls = []

    def factory(model_path, **kwargs):
        factory_calls.append((model_path, kwargs))
        return engine

    dependencies = _worker_dependencies(engine)
    dependencies["engine_factory"] = factory

    cell = worker.run_policy_cell(
        model_path="/checkpoint",
        gpu_index=0,
        policy="native_mtp",
        batch_size=1,
        **dependencies,
    )

    assert factory_calls == [(
        "/checkpoint",
        {
            "tensor_parallel_size": 1,
            "enforce_eager": True,
            "max_model_len": 8192,
            "max_num_batched_tokens": 16384,
            "max_num_prefill_tokens_per_step": 1024,
            "max_num_seqs": 1,
            "kv_offload_mvp0": False,
            "qwen35_mtp_enabled": True,
            "qwen35_mtp_cuda_graphs": False,
            "qwen35_mtp_max_proposal_tokens": 4,
        },
    )]
    assert isinstance(engine.activated_runtime, _FakeRuntime)
    assert (
        engine.activated_runtime.model_runner_executor
        is engine.model_runner.qwen35_mtp_executor_descriptor
    )
    runtime = cell["runtime"]
    assert runtime["first_target_target_forwards"] == 1
    assert runtime["verify_target_forwards"] == 1
    assert runtime["proposal_finalize_receipts"]
    assert runtime["side_state_receipts"]
    assert runtime["proposal_kv_receipts"]
    assert [
        row["operation"]
        for row in runtime["lifecycle_events"]
    ] == [
        "proposal_finalize_prepare",
        "side_state_apply",
        "target_kv_commit",
        "scheduler_commit",
        "proposal_finalize_commit",
        "side_state_seal",
        "proposal_sequence_release",
    ]
    assert cell["cleanup"]["native_state_snapshot"] == {
        "pending_prefix_count": 0,
        "bootstrapped_sequence_count": 0,
        "proposal_transaction_count": 0,
        "batch_ticket_count": 0,
        "batch_ticket_transaction_count": 0,
        "allocated_physical_slot_count": 0,
    }
    assert engine.exit_called is True


def test_worker_records_side_state_through_engine_acknowledgement_boundary():
    worker = _load_module(
        "native_mtp_engine_worker_ack_side_state",
        WORKER_PATH,
    )
    engine = _FakeEngine(
        batch_size=1,
        native=True,
        side_state_via_acknowledged=True,
    )

    cell = worker.run_policy_cell(
        model_path="/checkpoint",
        gpu_index=0,
        policy="native_mtp",
        batch_size=1,
        **_worker_dependencies(engine),
    )

    assert [
        row["operation"]
        for row in cell["runtime"]["side_state_receipts"]
    ] == [
        "prepare",
        "select",
        "apply",
        "seal",
    ]


def test_worker_stops_lifecycle_capture_after_side_state_seal():
    worker = _load_module(
        "native_mtp_engine_worker_lifecycle_scope",
        WORKER_PATH,
    )
    engine = _FakeEngine(batch_size=1, native=True)
    executor = engine.model_runner.qwen35_mtp_executor
    row = SimpleNamespace(
        sequence_id=0,
        proposal_transaction_id="proposal-tx-0",
        accepted_proposal_tokens=2,
    )

    with worker.capture_runtime_receipts(
        engine,
        executor,
    ) as capture:
        ticket = engine.model_runner.call(
            "prepare_speculative_proposal_finalize_batch",
            "native_checkpoint_proposal",
            (row,),
        )
        engine.model_runner.call(
            "apply_speculative_side_state_batch"
        )
        engine.scheduler.block_manager.commit_speculative_kv_commit_batch(
            (SimpleNamespace(sequence_id=0),)
        )
        engine.scheduler.commit_prepared_postprocess(object())
        engine.model_runner.call(
            "commit_speculative_proposal_finalize_batch",
            "native_checkpoint_proposal",
            ticket,
        )
        engine.model_runner.call(
            "seal_speculative_side_state_batch"
        )
        engine.model_runner.call(
            "release_speculative_proposal_sequence",
            "native_checkpoint_proposal",
            0,
            0,
        )
        engine.scheduler.commit_prepared_postprocess(object())

    assert [
        row["operation"]
        for row in capture["lifecycle_events"]
    ] == [
        "proposal_finalize_prepare",
        "side_state_apply",
        "target_kv_commit",
        "scheduler_commit",
        "proposal_finalize_commit",
        "side_state_seal",
        "proposal_sequence_release",
    ]


def test_worker_proposal_kv_receipts_exclude_bootstrap_commits():
    worker = _load_module(
        "native_mtp_engine_worker_proposal_kv_scope",
        WORKER_PATH,
    )
    engine = _FakeEngine(batch_size=1, native=True)

    class Cache:
        def __init__(self):
            self._transactions = {
                "bootstrap-tx": SimpleNamespace(
                    transaction_id="bootstrap-tx",
                    sequence_id=0,
                    staged_slot_ids=(10, 11),
                ),
                "proposal-tx": SimpleNamespace(
                    transaction_id="proposal-tx",
                    sequence_id=0,
                    staged_slot_ids=(20, 21, 22),
                ),
            }
            self._tickets = {
                "bootstrap-ticket": SimpleNamespace(
                    transaction_id="bootstrap-tx",
                    commit_entry_count=2,
                    release_slot_ids=(),
                ),
                "proposal-ticket": SimpleNamespace(
                    transaction_id="proposal-tx",
                    commit_entry_count=1,
                    release_slot_ids=(21, 22),
                ),
            }
            self._sequence_states = {
                0: SimpleNamespace(committed_slot_ids=()),
            }
            self._owned_slot_ids = {10, 11, 20, 21, 22}
            self.physical_store = SimpleNamespace(
                _allocated_slot_ids={10, 11, 20, 21, 22}
            )

        def commit_finalize(self, ticket_id):
            ticket = self._tickets[ticket_id]
            transaction = self._transactions[
                ticket.transaction_id
            ]
            state = self._sequence_states[transaction.sequence_id]
            committed = transaction.staged_slot_ids[
                :ticket.commit_entry_count
            ]
            state.committed_slot_ids += committed
            self._owned_slot_ids.difference_update(
                ticket.release_slot_ids
            )
            (
                self.physical_store._allocated_slot_ids
                .difference_update(ticket.release_slot_ids)
            )

    cache = Cache()
    executor = engine.model_runner.qwen35_mtp_executor
    executor.proposal_kv_cache = cache
    row = SimpleNamespace(
        sequence_id=0,
        proposal_transaction_id="proposal-tx",
        accepted_proposal_tokens=2,
    )

    with worker.capture_runtime_receipts(
        engine,
        executor,
    ) as capture:
        cache.commit_finalize("bootstrap-ticket")
        engine.model_runner.call(
            "prepare_speculative_proposal_finalize_batch",
            "native_checkpoint_proposal",
            (row,),
        )
        cache.commit_finalize("proposal-ticket")

    assert capture["proposal_kv_receipts"] == [{
        "sequence_id": 0,
        "transaction_id": "proposal-tx",
        "accepted_token_count": 2,
        "rejected_token_count": 2,
        "accepted_slot_identity_preserved": True,
        "rejected_slots_released": True,
    }]


def test_worker_measures_accepted_prefix_target_replay():
    worker = _load_module(
        "native_mtp_engine_worker_replay",
        WORKER_PATH,
    )
    engine = _FakeEngine(
        batch_size=1,
        native=True,
        accepted_prefix_replay=True,
    )

    with pytest.raises(
        ValueError,
        match="accepted-prefix target replay",
    ):
        worker.run_policy_cell(
            model_path="/checkpoint",
            gpu_index=0,
            policy="native_mtp",
            batch_size=1,
            **_worker_dependencies(engine),
        )


def test_worker_rejects_extra_spec_verify_target_forward():
    worker = _load_module(
        "native_mtp_engine_worker_extra_verify_forward",
        WORKER_PATH,
    )
    engine = _FakeEngine(
        batch_size=1,
        native=True,
        extra_spec_verify_forward=True,
    )

    with pytest.raises(
        ValueError,
        match="verify target forward",
    ):
        worker.run_policy_cell(
            model_path="/checkpoint",
            gpu_index=0,
            policy="native_mtp",
            batch_size=1,
            **_worker_dependencies(engine),
        )


def test_worker_baseline_disables_mtp_and_runtime():
    worker = _load_module(
        "native_mtp_engine_worker_baseline",
        WORKER_PATH,
    )
    engine = _FakeEngine(batch_size=1, native=False)
    factory_kwargs = []

    def factory(model_path, **kwargs):
        factory_kwargs.append(kwargs)
        return engine

    dependencies = _worker_dependencies(engine)
    dependencies["engine_factory"] = factory
    cell = worker.run_policy_cell(
        model_path="/checkpoint",
        gpu_index=0,
        policy="baseline",
        batch_size=1,
        **dependencies,
    )

    assert factory_kwargs[0]["qwen35_mtp_enabled"] is False
    assert engine.activated_runtime is None
    assert cell["runtime"] == _baseline_runtime()
    assert cell["cleanup"]["native_state_snapshot"] is None
    assert engine.exit_called is True


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda runner: setattr(
                runner,
                "qwen35_mtp_executor_descriptor",
                None,
            ),
            "descriptor",
        ),
        (
            lambda runner: setattr(
                runner,
                "qwen35_mtp_registration_error",
                "load failed",
            ),
            "registration",
        ),
        (
            lambda runner: setattr(
                runner.qwen35_mtp_executor,
                "module",
                object(),
            ),
            "module identity",
        ),
    ),
)
def test_worker_rejects_invalid_native_registration(
    mutate,
    match,
):
    worker = _load_module(
        "native_mtp_engine_worker_registration",
        WORKER_PATH,
    )
    runner = _FakeModelRunner(native=True)
    mutate(runner)

    with pytest.raises(RuntimeError, match=match):
        worker.validate_native_registration(runner)


def test_worker_rejects_wrong_output_length_and_exits():
    worker = _load_module(
        "native_mtp_engine_worker_output_length",
        WORKER_PATH,
    )
    engine = _FakeEngine(
        batch_size=1,
        native=True,
        output_count=31,
    )

    with pytest.raises(RuntimeError, match="output token count"):
        worker.run_policy_cell(
            model_path="/checkpoint",
            gpu_index=0,
            policy="native_mtp",
            batch_size=1,
            **_worker_dependencies(engine),
        )

    assert engine.exit_called is True


def test_worker_exits_when_engine_step_fails():
    worker = _load_module(
        "native_mtp_engine_worker_finally",
        WORKER_PATH,
    )
    engine = _FakeEngine(
        batch_size=1,
        native=True,
        step_error=RuntimeError("step failed"),
    )

    with pytest.raises(RuntimeError, match="step failed"):
        worker.run_policy_cell(
            model_path="/checkpoint",
            gpu_index=0,
            policy="native_mtp",
            batch_size=1,
            **_worker_dependencies(engine),
        )

    assert engine.exit_called is True


def _write_verifier_run(
    tmp_path,
    *,
    complete_source_inventory=True,
):
    gate = _load_gate("native_mtp_engine_verifier_fixture")
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_names = (
        gate.DEFAULT_SOURCE_FILES
        if complete_source_inventory
        else ("engine.py",)
    )
    source_files = {}
    for index, name in enumerate(source_names):
        source_path = source_root / name
        source_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        source_path.write_text(
            f"source-{index}\n",
            encoding="utf-8",
        )
        source_files[name] = gate.sha256_file(source_path)
    result = _result()
    result["source_tree_sha256"] = gate.source_tree_sha256(
        source_root,
        source_names,
    )
    run_dir = tmp_path / "authority"
    gate.publish_authority(
        run_dir,
        gate.validate_result(result),
        source_files=source_files,
    )
    return gate, run_dir, source_root


def _rewrite_result(gate, run_dir, mutate):
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    mutate(result)
    gate.atomic_write_json(result_path, result)
    manifest_path = run_dir / "source_manifest.json"
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    manifest["artifacts"]["result.json"] = gate.sha256_file(
        result_path
    )
    gate.atomic_write_json(manifest_path, manifest)


def test_independent_verifier_accepts_complete_authority(tmp_path):
    _, run_dir, source_root = _write_verifier_run(tmp_path)
    verifier = _load_module(
        "native_mtp_engine_verifier_accept",
        VERIFIER_PATH,
    )

    assert verifier.verify_run(run_dir, source_root) == {
        "classification": "PASS",
        "failures": [],
    }


def test_independent_verifier_rejects_partial_source_inventory(
    tmp_path,
):
    _, run_dir, source_root = _write_verifier_run(
        tmp_path,
        complete_source_inventory=False,
    )
    verifier = _load_module(
        "native_mtp_engine_verifier_partial_source_inventory",
        VERIFIER_PATH,
    )

    verification = verifier.verify_run(run_dir, source_root)

    assert verification["classification"] == "FAIL"
    assert any(
        "source file inventory" in failure
        for failure in verification["failures"]
    )


def test_independent_verifier_without_source_root_rejects_tree_mismatch(
    tmp_path,
):
    _, run_dir, _ = _write_verifier_run(tmp_path)
    manifest_path = run_dir / "source_manifest.json"
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    first_name = sorted(manifest["source_files"])[0]
    manifest["source_files"][first_name] = "0" * 64
    manifest_path.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    verifier = _load_module(
        "native_mtp_engine_verifier_offline_tree",
        VERIFIER_PATH,
    )

    verification = verifier.verify_run(run_dir)

    assert verification["classification"] == "FAIL"
    assert any(
        "source tree digest mismatch" in failure
        for failure in verification["failures"]
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda manifest: manifest.update(
                schema_version="other"
            ),
            "source manifest schema",
        ),
        (
            lambda manifest: manifest["artifacts"].update(
                {"extra.json": "0" * 64}
            ),
            "artifact inventory",
        ),
        (
            lambda manifest: manifest.update(
                undeclared_claim="production_ready"
            ),
            "source manifest is not canonical",
        ),
    ),
)
def test_independent_verifier_rejects_manifest_semantic_drift(
    tmp_path,
    mutate,
    match,
):
    _, run_dir, _ = _write_verifier_run(tmp_path)
    manifest_path = run_dir / "source_manifest.json"
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    mutate(manifest)
    manifest_path.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    verifier = _load_module(
        "native_mtp_engine_verifier_manifest_semantics",
        VERIFIER_PATH,
    )

    verification = verifier.verify_run(run_dir)

    assert verification["classification"] == "FAIL"
    assert any(
        match in failure
        for failure in verification["failures"]
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda result: result.update(
                mtp_checkpoint_manifest_sha256="0" * 64
            ),
            "MTP checkpoint manifest",
        ),
        (
            lambda result: result["cells"].pop("native_mtp:b4"),
            "cell inventory",
        ),
        (
            lambda result: result["parity"].update(b4=False),
            "parity",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "runtime"
            ]["lifecycle_events"].reverse(),
            "lifecycle",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "cleanup"
            ]["proposal_sequence_ids"].append(7),
            "sequence leak",
        ),
    ),
)
def test_independent_verifier_rejects_semantic_drift(
    tmp_path,
    mutate,
    match,
):
    gate, run_dir, source_root = _write_verifier_run(
        tmp_path
    )
    _rewrite_result(gate, run_dir, mutate)
    verifier = _load_module(
        "native_mtp_engine_verifier_semantic",
        VERIFIER_PATH,
    )

    verification = verifier.verify_run(run_dir, source_root)

    assert verification["classification"] == "FAIL"
    assert any(
        match in failure
        for failure in verification["failures"]
    )


def test_independent_verifier_rejects_result_digest_mismatch(
    tmp_path,
):
    _, run_dir, source_root = _write_verifier_run(tmp_path)
    result_path = run_dir / "result.json"
    result_path.write_text(
        result_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    verifier = _load_module(
        "native_mtp_engine_verifier_digest",
        VERIFIER_PATH,
    )

    verification = verifier.verify_run(run_dir, source_root)

    assert verification["classification"] == "FAIL"
    assert "artifact digest" in verification["failures"][0]


def test_independent_verifier_rejects_undeclared_result_field(
    tmp_path,
):
    gate, run_dir, source_root = _write_verifier_run(tmp_path)
    _rewrite_result(
        gate,
        run_dir,
        lambda result: result.update(
            undeclared_claim="production_ready"
        ),
    )
    verifier = _load_module(
        "native_mtp_engine_verifier_result_canonicality",
        VERIFIER_PATH,
    )

    verification = verifier.verify_run(run_dir, source_root)

    assert verification["classification"] == "FAIL"
    assert any(
        "result is not canonical" in failure
        for failure in verification["failures"]
    )


def test_independent_verifier_rejects_changed_source_file(
    tmp_path,
):
    gate, run_dir, source_root = _write_verifier_run(tmp_path)
    (source_root / gate.DEFAULT_SOURCE_FILES[0]).write_text(
        "engine-v2\n",
        encoding="utf-8",
    )
    verifier = _load_module(
        "native_mtp_engine_verifier_source",
        VERIFIER_PATH,
    )

    verification = verifier.verify_run(run_dir, source_root)

    assert verification["classification"] == "FAIL"
    assert any(
        "source" in failure
        for failure in verification["failures"]
    )


def test_remote_runner_contract_is_serial_and_non_destructive():
    text = RUNNER_PATH.read_text(encoding="utf-8")

    assert "sitian@10.232.195.203" in text
    assert (
        "KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian"
        in text
    )
    assert "ControlMaster=no" in text
    assert "ControlPath=none" in text
    assert "REMOTE_COMMAND_RETRY_ATTEMPTS" in text
    assert "REMOTE_RSYNC_RETRY_ATTEMPTS" in text
    assert "retry_remote_command" in text
    assert "retry_remote_rsync" in text
    assert 'klist -t -c "${KRB5CCNAME}"' in text
    assert text.index(
        'klist -t -c "${KRB5CCNAME}"'
    ) < text.index('mkdir -p "${LOCAL_RUN}"')
    assert 'retry_remote_command "true"' in text
    assert text.index(
        'retry_remote_command "true"'
    ) < text.index('mkdir -p "${LOCAL_RUN}"')
    assert "mktemp -d" in text
    assert "gpu-before.csv" in text
    assert "gpu-after.csv" in text
    assert "nvidia-smi" in text
    assert "baseline 1" in text
    assert "native_mtp 1" in text
    assert "baseline 4" in text
    assert "native_mtp 4" in text
    assert (
        "verify_qwen35_native_mtp_tp1_4k_engine_gate.py"
        in text
    )
    assert "local-authority" in text
    assert "rsync" in text
    assert "kill " not in text
    assert "pkill" not in text
