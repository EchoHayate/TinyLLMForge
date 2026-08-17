from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


TOOLS_DIR = Path(__file__).resolve().parent
REMOTE_WRAPPER = (
    TOOLS_DIR / "run_qwen35_mtp_real_checkpoint_gate_remote.sh"
)
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import qwen35_mtp_real_checkpoint_gate as gate_module  # noqa: E402
from qwen35_mtp_real_checkpoint_gate import (  # noqa: E402
    RealQwen35MTPGateBackend,
    REQUIRED_REPORT_FIELDS,
    checkpoint_manifest_sha256,
    main,
    parse_integer_csv,
    run_gate,
    validate_gate_report,
)


REQUIRED_Q_VALUES = (1, 2, 3, 4)
REQUIRED_BATCH_SIZES = (1, 4)


def _transaction_cases():
    return [
        {
            "q": q,
            "batch_size": batch_size,
            "accepted_proposal_tokens": accepted,
            "staged_slot_ids": list(range(q - 1)),
            "committed_slot_ids": list(
                range(max(accepted - 1, 0))
            ),
            "released_slot_ids": list(
                range(max(accepted - 1, 0), q - 1)
            ),
            "accepted_slot_identity_preserved": True,
            "rejected_slots_released": True,
            "post_rollback_continuation_equal": True,
        }
        for batch_size in REQUIRED_BATCH_SIZES
        for q in REQUIRED_Q_VALUES
        for accepted in range(q + 1)
    ]


def _valid_report():
    return {
        "schema_version": 1,
        "checkpoint_path": "/readonly/model",
        "checkpoint_manifest_sha256": "a" * 64,
        "device_name": "NVIDIA A100 80GB PCIe",
        "torch_version": "2.6.0",
        "cuda_version": "12.4",
        "q_values": list(REQUIRED_Q_VALUES),
        "batch_sizes": list(REQUIRED_BATCH_SIZES),
        "loader_passed": True,
        "shared_embedding_identity": True,
        "shared_lm_head_identity": True,
        "eager_reference_max_abs_diff": 0.0,
        "eager_reference_argmax_equal": True,
        "graph_backend_installed": True,
        "graph_capture_count": 6,
        "graph_replay_count": 6,
        "graph_eager_argmax_equal": True,
        "graph_eager_proposal_tokens_equal": True,
        "graph_transaction_commit": True,
        "graph_transaction_rollback": True,
        "replay_failure_quarantined": True,
        "replay_failure_eager_retry_count": 0,
        "transaction_cases": _transaction_cases(),
        "accepted_slot_identity_preserved": True,
        "rejected_slots_released": True,
        "post_rollback_continuation_equal": True,
        "status": "PASS",
        "promotion_classification": "NOT_PROMOTABLE",
        "limitations": [
            "TP1 only",
            "KV offload disabled",
            "single Qwen3.5 architecture",
            "no long-context coverage",
            "no performance claim",
        ],
        "coverage": {
            "tensor_parallel_sizes": [1],
            "kv_offload": False,
            "architectures": ["qwen3_5"],
            "long_context": False,
            "performance": False,
        },
    }


def test_required_schema_fields_are_frozen():
    assert REQUIRED_REPORT_FIELDS == (
        "schema_version",
        "checkpoint_path",
        "checkpoint_manifest_sha256",
        "device_name",
        "torch_version",
        "cuda_version",
        "q_values",
        "batch_sizes",
        "loader_passed",
        "shared_embedding_identity",
        "shared_lm_head_identity",
        "eager_reference_max_abs_diff",
        "eager_reference_argmax_equal",
        "graph_backend_installed",
        "graph_capture_count",
        "graph_replay_count",
        "graph_eager_argmax_equal",
        "graph_eager_proposal_tokens_equal",
        "graph_transaction_commit",
        "graph_transaction_rollback",
        "replay_failure_quarantined",
        "replay_failure_eager_retry_count",
        "transaction_cases",
        "accepted_slot_identity_preserved",
        "rejected_slots_released",
        "post_rollback_continuation_equal",
        "status",
        "promotion_classification",
        "limitations",
    )


def test_integer_csv_parser_requires_canonical_positive_values():
    assert parse_integer_csv("1,2,3,4", name="q_values") == (
        1,
        2,
        3,
        4,
    )
    for value in ("", "1,1", "2,1", "0,1", "1,-2", "1,true"):
        with pytest.raises(ValueError):
            parse_integer_csv(value, name="q_values")


def test_eager_reference_probe_cannot_observe_or_replay_graphs():
    source = inspect.getsource(
        gate_module._build_real_eager_reference_probe
    )

    assert "original_graph_runner = executor.graph_runner" in source
    assert "executor.graph_runner = None" in source
    assert "executor.graph_runner = original_graph_runner" in source


def test_transaction_probe_cannot_observe_or_capture_graphs():
    source = inspect.getsource(
        gate_module._build_real_transaction_probe
    )

    assert "original_graph_runner = executor.graph_runner" in source
    assert "executor.graph_runner = None" in source
    assert "executor.graph_runner = original_graph_runner" in source


def test_replay_failure_is_injected_after_all_graph_families_capture():
    source = inspect.getsource(
        gate_module._build_real_graph_eager_probe
    )

    assert "(q, batch_size) == (4, 4)" in source


def test_valid_pass_report_is_accepted_but_not_promoted():
    report = _valid_report()

    assert validate_gate_report(
        report,
        required_q_values=REQUIRED_Q_VALUES,
        required_batch_sizes=REQUIRED_BATCH_SIZES,
    ) is report
    assert report["status"] == "PASS"
    assert report["promotion_classification"] == "NOT_PROMOTABLE"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("q_values", [1, 2, 3]),
        ("batch_sizes", [1]),
    ),
)
def test_pass_rejects_incomplete_q_or_batch_domain(field, value):
    report = _valid_report()
    report[field] = value

    with pytest.raises(ValueError, match=field):
        validate_gate_report(
            report,
            required_q_values=REQUIRED_Q_VALUES,
            required_batch_sizes=REQUIRED_BATCH_SIZES,
        )


def test_pass_rejects_missing_acceptance_transaction_case():
    report = _valid_report()
    report["transaction_cases"] = [
        case
        for case in report["transaction_cases"]
        if not (
            case["batch_size"] == 4
            and case["q"] == 3
            and case["accepted_proposal_tokens"] == 2
        )
    ]

    with pytest.raises(
        ValueError,
        match="transaction case domain",
    ):
        validate_gate_report(
            report,
            required_q_values=REQUIRED_Q_VALUES,
            required_batch_sizes=REQUIRED_BATCH_SIZES,
        )


@pytest.mark.parametrize(
    ("coverage_field", "coverage_value", "match"),
    (
        ("tensor_parallel_sizes", [1, 4], "TP4"),
        ("kv_offload", True, "KV offload"),
        (
            "architectures",
            ["qwen3_5", "second_model"],
            "second architecture",
        ),
        ("long_context", True, "long-context"),
        ("performance", True, "performance"),
    ),
)
def test_pass_rejects_unsupported_promotion_claims(
    coverage_field,
    coverage_value,
    match,
):
    report = _valid_report()
    report["coverage"][coverage_field] = coverage_value

    with pytest.raises(ValueError, match=match):
        validate_gate_report(
            report,
            required_q_values=REQUIRED_Q_VALUES,
            required_batch_sizes=REQUIRED_BATCH_SIZES,
        )


@pytest.mark.parametrize(
    "field",
    (
        "loader_passed",
        "shared_embedding_identity",
        "shared_lm_head_identity",
        "eager_reference_argmax_equal",
        "graph_backend_installed",
        "graph_eager_argmax_equal",
        "graph_eager_proposal_tokens_equal",
        "graph_transaction_commit",
        "graph_transaction_rollback",
        "replay_failure_quarantined",
        "accepted_slot_identity_preserved",
        "rejected_slots_released",
        "post_rollback_continuation_equal",
    ),
)
def test_pass_requires_every_correctness_boolean(field):
    report = _valid_report()
    report[field] = False

    with pytest.raises(ValueError, match=field):
        validate_gate_report(
            report,
            required_q_values=REQUIRED_Q_VALUES,
            required_batch_sizes=REQUIRED_BATCH_SIZES,
        )


def test_report_never_accepts_promotable_classification():
    report = deepcopy(_valid_report())
    report["promotion_classification"] = "PROMOTABLE"

    with pytest.raises(ValueError, match="NOT_PROMOTABLE"):
        validate_gate_report(
            report,
            required_q_values=REQUIRED_Q_VALUES,
            required_batch_sizes=REQUIRED_BATCH_SIZES,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("graph_backend_installed", False),
        ("graph_capture_count", 0),
        ("graph_replay_count", 0),
        ("graph_eager_argmax_equal", False),
        ("graph_eager_proposal_tokens_equal", False),
        ("graph_transaction_commit", False),
        ("graph_transaction_rollback", False),
        ("replay_failure_quarantined", False),
        ("replay_failure_eager_retry_count", 1),
    ),
)
def test_pass_rejects_corrupt_graph_evidence(field, value):
    report = _valid_report()
    report[field] = value

    with pytest.raises(ValueError, match=field):
        validate_gate_report(
            report,
            required_q_values=REQUIRED_Q_VALUES,
            required_batch_sizes=REQUIRED_BATCH_SIZES,
        )
    assert report["promotion_classification"] == "NOT_PROMOTABLE"


class _RecordingBackend:

    def __init__(self):
        self.events = []

    def load(self, checkpoint_path):
        self.events.append(("load", checkpoint_path))
        return {
            "checkpoint_manifest_sha256": "b" * 64,
            "device_name": "NVIDIA A100 80GB PCIe",
            "torch_version": "2.6.0",
            "cuda_version": "12.4",
            "loader_passed": True,
            "shared_embedding_identity": True,
            "shared_lm_head_identity": True,
            "checkpoint_unchanged": True,
            "config_tensor_contract_passed": True,
        }

    def compare_eager_reference(self, q, batch_size):
        self.events.append(("eager_reference", q, batch_size))
        return {
            "max_abs_diff": q * batch_size * 1e-6,
            "argmax_equal": True,
        }

    def compare_graph_eager(self, q, batch_size):
        self.events.append(("graph_eager", q, batch_size))
        graph_family = q >= 2
        return {
            "backend_installed": True,
            "capture_count": int(graph_family),
            "replay_count": int(graph_family),
            "argmax_equal": True,
            "proposal_tokens_equal": True,
            "transaction_commit": True,
            "transaction_rollback": True,
            "replay_failure_quarantined": (
                q == 2 and batch_size == 1
            ),
            "replay_failure_eager_retry_count": 0,
        }

    def run_transaction_case(self, q, batch_size, accepted):
        self.events.append(
            ("transaction", q, batch_size, accepted)
        )
        staged = list(range(q - 1))
        committed_count = max(accepted - 1, 0)
        return {
            "q": q,
            "batch_size": batch_size,
            "accepted_proposal_tokens": accepted,
            "staged_slot_ids": staged,
            "committed_slot_ids": staged[:committed_count],
            "released_slot_ids": staged[committed_count:],
            "accepted_slot_identity_preserved": True,
            "rejected_slots_released": True,
            "post_rollback_continuation_equal": True,
        }


def test_gate_runs_complete_q_batch_and_transaction_matrix():
    backend = _RecordingBackend()

    report = run_gate(
        checkpoint_path="/readonly/model",
        q_values=REQUIRED_Q_VALUES,
        batch_sizes=REQUIRED_BATCH_SIZES,
        backend=backend,
    )

    assert backend.events[0] == ("load", "/readonly/model")
    assert [
        event
        for event in backend.events
        if event[0] == "eager_reference"
    ] == [
        ("eager_reference", q, batch_size)
        for batch_size in REQUIRED_BATCH_SIZES
        for q in REQUIRED_Q_VALUES
    ]
    assert [
        event
        for event in backend.events
        if event[0] == "graph_eager"
    ] == [
        ("graph_eager", q, batch_size)
        for batch_size in REQUIRED_BATCH_SIZES
        for q in REQUIRED_Q_VALUES
    ]
    assert [
        event
        for event in backend.events
        if event[0] == "transaction"
    ] == [
        ("transaction", q, batch_size, accepted)
        for batch_size in REQUIRED_BATCH_SIZES
        for q in REQUIRED_Q_VALUES
        for accepted in range(q + 1)
    ]
    assert report["status"] == "PASS"
    assert report["promotion_classification"] == "NOT_PROMOTABLE"
    assert report["checkpoint_unchanged"] is True
    assert report["config_tensor_contract_passed"] is True
    assert report["eager_reference_max_abs_diff"] == pytest.approx(
        4 * 4 * 1e-6
    )
    assert report["graph_backend_installed"] is True
    assert report["graph_capture_count"] == 6
    assert report["graph_replay_count"] == 6
    assert report["graph_eager_proposal_tokens_equal"] is True
    assert report["graph_transaction_commit"] is True
    assert report["graph_transaction_rollback"] is True
    assert report["replay_failure_quarantined"] is True
    assert report["replay_failure_eager_retry_count"] == 0
    validate_gate_report(
        report,
        required_q_values=REQUIRED_Q_VALUES,
        required_batch_sizes=REQUIRED_BATCH_SIZES,
    )


def test_gate_fails_closed_when_any_backend_check_is_false():
    class Backend(_RecordingBackend):
        def compare_graph_eager(self, q, batch_size):
            result = super().compare_graph_eager(q, batch_size)
            if (q, batch_size) == (3, 4):
                result["argmax_equal"] = False
            return result

    report = run_gate(
        checkpoint_path="/readonly/model",
        q_values=REQUIRED_Q_VALUES,
        batch_sizes=REQUIRED_BATCH_SIZES,
        backend=Backend(),
    )

    assert report["status"] == "FAIL"
    assert report["graph_eager_argmax_equal"] is False
    assert report["promotion_classification"] == "NOT_PROMOTABLE"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("backend_installed", False),
        ("capture_count", 0),
        ("replay_count", 0),
        ("argmax_equal", False),
        ("proposal_tokens_equal", False),
        ("transaction_commit", False),
        ("transaction_rollback", False),
        ("replay_failure_quarantined", False),
        ("replay_failure_eager_retry_count", 1),
    ),
)
def test_gate_fails_closed_for_corrupt_graph_probe_result(
    field,
    value,
):
    class Backend(_RecordingBackend):
        def compare_graph_eager(self, q, batch_size):
            result = super().compare_graph_eager(q, batch_size)
            if (q, batch_size) == (2, 1):
                result[field] = value
            return result

    report = run_gate(
        checkpoint_path="/readonly/model",
        q_values=REQUIRED_Q_VALUES,
        batch_sizes=REQUIRED_BATCH_SIZES,
        backend=Backend(),
    )

    assert report["status"] == "FAIL"
    assert report["promotion_classification"] == "NOT_PROMOTABLE"


def test_checkpoint_manifest_sha256_is_content_and_path_stable(
    tmp_path,
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        '{"model_type":"qwen3_5"}\n',
        encoding="utf-8",
    )
    shard = checkpoint / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"checkpoint-payload")

    first = checkpoint_manifest_sha256(checkpoint)
    shard.touch()
    second = checkpoint_manifest_sha256(checkpoint)
    shard.write_bytes(b"changed-payload")
    third = checkpoint_manifest_sha256(checkpoint)

    assert first == second
    assert third != first
    assert len(first) == 64


def test_real_backend_fails_closed_with_complete_blocked_matrix(
    tmp_path,
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        '{"model_type":"qwen3_5"}\n',
        encoding="utf-8",
    )

    def runtime_loader(_checkpoint_path):
        return SimpleNamespace(
            loader_passed=True,
            shared_embedding_identity=True,
            shared_lm_head_identity=True,
            config_tensor_contract_passed=True,
            blockers={
                "eager_reference": (
                    "MTP attention context wiring is not installed"
                ),
                "graph_eager": (
                    "Qwen3.5 MTP CUDA graph capture backend "
                    "is not installed"
                ),
                "transaction": (
                    "physical MTP KV tensor evidence is unavailable"
                ),
            },
        )

    backend = RealQwen35MTPGateBackend(
        runtime_loader=runtime_loader,
        runtime_metadata_loader=lambda: {
            "device_name": "NVIDIA A100 80GB PCIe",
            "torch_version": "2.4.1+cu121",
            "cuda_version": "12.1",
        },
    )
    report = run_gate(
        checkpoint_path=str(checkpoint),
        q_values=REQUIRED_Q_VALUES,
        batch_sizes=REQUIRED_BATCH_SIZES,
        backend=backend,
    )

    assert report["loader_passed"] is True
    assert report["eager_reference_argmax_equal"] is False
    assert report["graph_eager_argmax_equal"] is False
    assert len(report["transaction_cases"]) == sum(
        q + 1
        for _batch_size in REQUIRED_BATCH_SIZES
        for q in REQUIRED_Q_VALUES
    )
    assert report["accepted_slot_identity_preserved"] is False
    assert report["rejected_slots_released"] is False
    assert report["post_rollback_continuation_equal"] is False
    assert report["status"] == "FAIL"
    assert report["checkpoint_unchanged"] is True
    assert report["backend_failures"] == [
        {
            "domain": "eager_reference",
            "reason": "MTP attention context wiring is not installed",
        },
        {
            "domain": "graph_eager",
            "reason": (
                "Qwen3.5 MTP CUDA graph capture backend "
                "is not installed"
            ),
        },
        {
            "domain": "transaction",
            "reason": (
                "physical MTP KV tensor evidence is unavailable"
            ),
        },
    ]


def test_real_runtime_installs_eager_and_transaction_probes(
    monkeypatch,
):
    shared_embedding = object()
    shared_lm_head = object()
    target_model = SimpleNamespace(
        embed_tokens=shared_embedding,
        lm_head=shared_lm_head,
    )
    module = SimpleNamespace(
        embed_tokens=shared_embedding,
        lm_head=shared_lm_head,
    )
    executor = object()
    physical_store = object()
    eager_probe = object()
    graph_probe = object()
    transaction_probe = object()

    class FakeConfig:

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.hf_config = SimpleNamespace(model_type="qwen3_5")

    class FakeModelRunner:

        def __init__(self, config, rank, event):
            assert config.kwargs["tensor_parallel_size"] == 1
            assert config.kwargs["kv_offload_mvp0"] is False
            assert config.kwargs["qwen35_mtp_cuda_graphs"] is True
            assert rank == 0
            assert event is not None
            self.config = config
            self.model = target_model
            self.qwen35_mtp_module = module
            self.qwen35_mtp_executor = executor
            self.qwen35_mtp_physical_store = physical_store
            self.qwen35_mtp_registration_error = None

    config_module = SimpleNamespace(Config=FakeConfig)
    runner_module = SimpleNamespace(ModelRunner=FakeModelRunner)
    eager_probe_calls = []
    graph_probe_calls = []
    transaction_probe_calls = []
    monkeypatch.setitem(
        sys.modules,
        "tinyvllm.config",
        config_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "tinyvllm.engine.model_runner",
        runner_module,
    )
    monkeypatch.setattr(
        gate_module,
        "_build_real_transaction_probe",
        lambda **kwargs: (
            transaction_probe_calls.append(kwargs)
            or transaction_probe
        ),
        raising=False,
    )
    monkeypatch.setattr(
        gate_module,
        "_build_real_graph_eager_probe",
        lambda **kwargs: (
            graph_probe_calls.append(kwargs)
            or graph_probe
        ),
        raising=False,
    )
    monkeypatch.setattr(
        gate_module,
        "_build_real_eager_reference_probe",
        lambda **kwargs: (
            eager_probe_calls.append(kwargs)
            or eager_probe
        ),
        raising=False,
    )

    runtime = RealQwen35MTPGateBackend._load_real_runtime(
        "/readonly/model"
    )

    assert runtime["loader_passed"] is True
    assert runtime["eager_reference_probe"] is eager_probe
    assert runtime["graph_eager_probe"] is graph_probe
    assert runtime["transaction_probe"] is transaction_probe
    expected_probe_arguments = {
        "executor": executor,
        "module": module,
        "physical_store": physical_store,
        "hf_config": runtime["runner"].config.hf_config,
    }
    assert eager_probe_calls == [expected_probe_arguments]
    assert graph_probe_calls == [expected_probe_arguments]
    assert transaction_probe_calls == [expected_probe_arguments]
    assert "eager_reference" not in runtime["blockers"]
    assert "graph_eager" not in runtime["blockers"]
    assert "transaction" not in runtime["blockers"]
    assert runtime["blockers"] == {}


def test_cli_atomically_writes_fail_report_and_exits_zero(
    tmp_path,
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        '{"model_type":"qwen3_5"}\n',
        encoding="utf-8",
    )
    output = tmp_path / "artifacts" / "gate.json"

    class Backend(_RecordingBackend):
        def compare_graph_eager(self, q, batch_size):
            super().compare_graph_eager(q, batch_size)
            return {"argmax_equal": False}

    exit_code = main(
        [
            "--checkpoint",
            str(checkpoint),
            "--q-values",
            "1,2,3,4",
            "--batch-sizes",
            "1,4",
            "--output",
            str(output),
        ],
        backend_factory=Backend,
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert report["promotion_classification"] == "NOT_PROMOTABLE"
    assert not tuple(output.parent.glob(f".{output.name}.*.tmp"))


def test_remote_wrapper_is_isolated_serial_and_downloads_artifact():
    text = REMOTE_WRAPPER.read_text(encoding="utf-8")

    assert "set -euo pipefail" in text
    assert (
        'KRB5CCNAME="${KRB5CCNAME:-'
        'FILE:/Users/bytedance/krb5cc_sitian}"'
    ) in text
    assert (
        'REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"'
        in text
    )
    assert "ControlMaster=no" in text
    assert "ControlPath=none" in text
    assert "retry 3 " in text
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"' in text
    assert "qwen35-mtp-runs/${TAG}" in text
    assert "cp -a '${REMOTE_BASE}/.' '${REMOTE_RUN_ROOT}/'" in text
    assert "SOURCE_FILES=(" in text
    for source_file in (
        "tinyvllm/config.py",
        "tinyvllm/engine/model_runner.py",
        "tinyvllm/engine/qwen35_mtp_registration.py",
        "tinyvllm/engine/spec_verify_exact_cuda_graph_cache.py",
        "tinyvllm/engine/qwen35_mtp_executor.py",
        "tinyvllm/engine/qwen35_mtp_graph.py",
        "tinyvllm/engine/qwen35_mtp_cuda_graph_backend.py",
        "tinyvllm/engine/qwen35_mtp_graph_scratch.py",
        "tinyvllm/utils/context.py",
        "tinyvllm/layers/qwen35_full_attention.py",
        "tinyvllm/models/qwen35_mtp_checkpoint.py",
        "tinyvllm/models/qwen35_mtp.py",
        "tools/qwen35_mtp_real_checkpoint_gate.py",
    ):
        assert source_file in text
    assert "--q-values 1,2,3,4" in text
    assert "--batch-sizes 1,4" in text
    assert "artifacts/qwen35_mtp_real_checkpoint_gate.json" in text
    assert 'LOCAL_RUN_ROOT="${LOCAL_RUN_BASE}/${TAG}"' in text
    assert (
        '"${REMOTE_HOST}:${REMOTE_ARTIFACT}" '
        '"${LOCAL_RUN_ROOT}/"'
    ) in text
