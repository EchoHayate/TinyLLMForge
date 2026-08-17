from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = (
    ROOT / "tools" / "speculative_residency_boundary_gate.py"
)
VERIFIER_PATH = (
    ROOT
    / "tools"
    / "verify_speculative_residency_boundary_gate.py"
)
RUNNER_PATH = (
    ROOT
    / "tools"
    / "run_speculative_residency_boundary_gate_remote.sh"
)


def _gate():
    spec = importlib.util.spec_from_file_location(
        "speculative_residency_boundary_gate_test_module",
        GATE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _verifier():
    spec = importlib.util.spec_from_file_location(
        "speculative_residency_boundary_verifier_test_module",
        VERIFIER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _Capabilities:
    source_type: str
    supports_batch: bool
    requires_target_hidden: bool
    requires_target_logits: bool
    max_proposal_tokens: int


@dataclass(frozen=True)
class _Proposal:
    sequence_id: int
    token_ids: tuple[int, ...]
    source_type: str
    metadata: object | None = None
    timing_ms: dict[str, float] | None = None


def _context(target_token=17):
    return SimpleNamespace(
        sequence_id=7,
        first_target_token=target_token,
        max_proposal_tokens=3,
        remaining_output_tokens=3,
    )


def test_boundary_adapter_forces_accept_or_reject():
    gate = _gate()
    adapter_types = (_Capabilities, _Proposal)

    accept = gate.BoundaryDraftAdapter(
        "accept",
        accepted_token_ids=(17, 18, 19),
        adapter_types=adapter_types,
    )
    reject = gate.BoundaryDraftAdapter(
        "reject",
        accepted_token_ids=(17, 18, 19),
        adapter_types=adapter_types,
    )
    reject_zero = gate.BoundaryDraftAdapter(
        "reject",
        accepted_token_ids=(0, 18, 19),
        adapter_types=adapter_types,
    )

    accepted = accept.propose_batch((_context(17),))
    rejected = reject.propose_batch((_context(17),))
    rejected_zero = reject_zero.propose_batch((_context(0),))

    assert accept.capabilities == _Capabilities(
        source_type="boundary_fixture",
        supports_batch=True,
        requires_target_hidden=False,
        requires_target_logits=False,
        max_proposal_tokens=3,
    )
    assert accepted[0].token_ids == (17, 18, 19)
    assert rejected[0].token_ids == (0, 18, 19)
    assert rejected_zero[0].token_ids == (1, 18, 19)
    assert accepted[0].metadata == {"mode": "accept"}
    assert rejected[0].metadata == {"mode": "reject"}

    with pytest.raises(RuntimeError, match="baseline"):
        accept.propose_batch((_context(16),))


@pytest.mark.parametrize(
    "value,match",
    [
        ([1] * 253, "254"),
        ([1] * 255, "254"),
        ([1] * 253 + [True], "integer"),
        ("tokens", "list"),
    ],
)
def test_validate_boundary_prompt_rejects_wrong_shape(
    value,
    match,
):
    with pytest.raises(ValueError, match=match):
        _gate().validate_boundary_prompt_token_ids(value)


def test_validate_boundary_prompt_accepts_exact_token_list():
    token_ids = list(range(254))

    assert _gate().validate_boundary_prompt_token_ids(
        token_ids
    ) == tuple(token_ids)


def test_prefill_boundary_requires_one_live_length_255_sequence():
    gate = _gate()
    sequence = SimpleNamespace(
        seq_id=7,
        num_tokens=255,
        num_prompt_tokens=254,
        block_table=[3],
    )
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(running=[sequence]),
    )

    assert gate.require_prefill_boundary_sequence(engine) is sequence

    sequence.num_tokens = 254
    with pytest.raises(RuntimeError, match="255"):
        gate.require_prefill_boundary_sequence(engine)


def test_evict_boundary_history_orders_real_manager_operations():
    gate = _gate()
    calls = []

    class _Manager:
        logical_to_slot = {3: 0}
        bound_generations = [None, None, None, 9]
        cpu_valid = [False, False, False, False]
        dirty_logical_blocks = {3}
        pending_wait_blocks = set()

        def writeback_dirty(self, logical_blocks=None):
            calls.append(("writeback", logical_blocks))
            self.dirty_logical_blocks.clear()
            self.cpu_valid[3] = True

        def synchronize_copies(self):
            calls.append(("synchronize",))

        def evict_clean_resident_blocks(self, identities):
            calls.append(("evict", identities))
            self.logical_to_slot.clear()
            return identities

    manager = _Manager()
    engine = SimpleNamespace(
        model_runner=SimpleNamespace(kv_offload=manager),
    )
    sequence = SimpleNamespace(block_table=[3])

    identities = gate.evict_boundary_history(
        engine,
        sequence,
    )

    assert identities == ((3, 9),)
    assert calls == [
        ("writeback", [3]),
        ("synchronize",),
        ("evict", ((3, 9),)),
    ]
    assert manager.logical_to_slot == {}


@dataclass(frozen=True)
class _SamplingParams:
    temperature: float
    max_tokens: int
    ignore_eos: bool


@dataclass(frozen=True)
class _Runtime:
    draft_adapter: object


class _BoundaryManager:
    def __init__(self):
        self.logical_to_slot = {3: 0}
        self.bound_generations = [None, None, None, 9]
        self.cpu_valid = [False, False, False, False]
        self.dirty_logical_blocks = {3}
        self.pending_wait_blocks = set()
        self.calls = []

    def writeback_dirty(self, logical_blocks=None):
        self.calls.append(("writeback", logical_blocks))
        self.dirty_logical_blocks.clear()
        self.cpu_valid[3] = True

    def synchronize_copies(self):
        self.calls.append(("synchronize",))

    def evict_clean_resident_blocks(self, identities):
        self.calls.append(("evict", identities))
        self.logical_to_slot.clear()
        return identities


class _BoundaryEngine:
    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs
        self.manager = _BoundaryManager()
        self.model_runner = SimpleNamespace(
            kv_offload=self.manager,
        )
        self.sequence = SimpleNamespace(
            seq_id=7,
            num_tokens=254,
            num_prompt_tokens=254,
            block_table=[3],
        )
        self.scheduler = SimpleNamespace(running=[])
        self.runtime = None
        self.request = None
        self.step_count = 0
        self.finished = False
        self.exited = False
        self.last_step_observation = None

    def activate_speculative_runtime(self, runtime):
        self.runtime = runtime

    def add_request(self, prompt, sampling_params):
        self.request = (prompt, sampling_params)

    def step(self):
        self.step_count += 1
        if self.step_count == 1:
            self.sequence.num_tokens = 255
            self.scheduler.running = [self.sequence]
            self.last_step_observation = {
                "speculative_selected_seq_ids": [],
                "speculative_proposal_token_counts": {},
                "speculative_proposal_row_count": 0,
                "speculative_accepted_draft_token_counts": {},
                "speculative_first_target_callback_count": 0,
                "speculative_fixed_q_group_count": 0,
            }
            return [], 254

        mode = (
            None
            if self.runtime is None
            else self.runtime.draft_adapter._mode
        )
        selected = [] if mode is None else [7]
        proposal_counts = (
            {} if mode is None else {7: 3}
        )
        accepted_counts = (
            {}
            if mode is None
            else {7: 3 if mode == "accept" else 0}
        )
        self.last_step_observation = {
            "speculative_selected_seq_ids": selected,
            "speculative_proposal_token_counts": proposal_counts,
            "speculative_proposal_row_count": len(selected),
            "speculative_accepted_draft_token_counts": (
                accepted_counts
            ),
            "speculative_first_target_callback_count": (
                int(mode is not None)
            ),
            "speculative_fixed_q_group_count": (
                int(mode is not None)
            ),
        }
        self.finished = True
        self.scheduler.running = []
        return [(7, [31, 32, 33, 34])], -1

    def is_finished(self):
        return self.finished

    def kv_offload_summaries(self, timeout_s):
        mode = (
            None
            if self.runtime is None
            else self.runtime.draft_adapter._mode
        )
        return ({
            "h2d_copies": int(mode is not None),
            "d2h_copies": 1,
            "h2d_bytes": 4096 if mode is not None else 0,
            "d2h_bytes": 4096,
            "copy_waits": int(mode is not None),
            "evictions": 1,
            "evict_clean": 1,
            "evict_dirty": 0,
            "speculative_residency_prepares": (
                int(mode is not None)
            ),
            "speculative_residency_precommits": (
                int(mode is not None)
            ),
            "speculative_residency_seals": (
                int(mode is not None)
            ),
            "speculative_residency_rollbacks": 0,
            "speculative_residency_committed_blocks": (
                int(mode == "accept")
            ),
            "speculative_residency_rejected_blocks": (
                int(mode == "reject")
            ),
            "speculative_residency_rejected_d2h_copies": 0,
        },)

    def exit(self):
        self.exited = True


@pytest.mark.parametrize(
    "mode,accepted,committed,rejected",
    [
        ("accept", 3, 1, 0),
        ("reject", 0, 0, 1),
    ],
)
def test_run_boundary_case_collects_real_case_evidence(
    mode,
    accepted,
    committed,
    rejected,
):
    gate = _gate()
    engines = []

    def engine_factory(*args, **kwargs):
        engine = _BoundaryEngine(*args, **kwargs)
        engines.append(engine)
        return engine

    result = gate.run_boundary_case(
        engine_factory=engine_factory,
        model_path="/model",
        prompt_token_ids=list(range(254)),
        mode=mode,
        accepted_token_ids=(32, 33, 34),
        runtime_types=(_SamplingParams, _Runtime),
        adapter_types=(_Capabilities, _Proposal),
    )

    engine = engines[0]
    assert engine.kwargs == {
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "max_model_len": 4096,
        "max_num_seqs": 1,
        "kv_offload_mvp0": True,
        "kv_offload_gpu_blocks": 2,
        "kv_offload_logical_blocks": 64,
    }
    assert engine.request == (
        list(range(254)),
        _SamplingParams(
            temperature=0.0,
            max_tokens=4,
            ignore_eos=True,
        ),
    )
    assert engine.manager.calls == [
        ("writeback", [3]),
        ("synchronize",),
        ("evict", ((3, 9),)),
    ]
    assert result["outputs"] == [[31, 32, 33, 34]]
    assert result["evicted_block_identities"] == [[3, 9]]
    assert result["summary"]["proposed_tokens"] == 3
    assert (
        result["summary"]["accepted_draft_tokens"]
        == accepted
    )
    assert (
        result["residency"][
            "speculative_residency_committed_blocks"
        ]
        == committed
    )
    assert (
        result["residency"][
            "speculative_residency_rejected_blocks"
        ]
        == rejected
    )
    assert result["movement"]["h2d_copies"] == 1
    assert result["movement"]["h2d_bytes"] == 4096
    assert engine.exited is True


def test_run_boundary_case_baseline_skips_runtime():
    gate = _gate()
    engines = []

    def engine_factory(*args, **kwargs):
        engine = _BoundaryEngine(*args, **kwargs)
        engines.append(engine)
        return engine

    result = gate.run_boundary_case(
        engine_factory=engine_factory,
        model_path="/model",
        prompt_token_ids=list(range(254)),
        mode=None,
        accepted_token_ids=None,
        runtime_types=(_SamplingParams, _Runtime),
        adapter_types=(_Capabilities, _Proposal),
    )

    assert engines[0].runtime is None
    assert result["outputs"] == [[31, 32, 33, 34]]
    assert result["summary"]["proposed_tokens"] == 0


def _artifact_case(
    *,
    accepted=0,
    committed=0,
    rejected=0,
    proposed=None,
    rejected_d2h=0,
    h2d_copies=1,
    h2d_bytes=4096,
):
    if proposed is None:
        proposed = int(accepted > 0 or rejected > 0)
    return {
        "outputs": [[31, 32]],
        "evicted_block_identities": [[3, 9]],
        "elapsed_s": 1.0,
        "summary": {
            "proposed_tokens": proposed,
            "accepted_draft_tokens": accepted,
        },
        "movement": {
            "h2d_copies": h2d_copies,
            "d2h_copies": 1,
            "h2d_bytes": h2d_bytes,
            "d2h_bytes": 4096,
            "copy_waits": h2d_copies,
            "evictions": 1,
            "evict_clean": 1,
            "evict_dirty": 0,
        },
        "residency": {
            "speculative_residency_prepares": proposed,
            "speculative_residency_precommits": proposed,
            "speculative_residency_seals": proposed,
            "speculative_residency_rollbacks": 0,
            "speculative_residency_committed_blocks": (
                committed
            ),
            "speculative_residency_rejected_blocks": rejected,
            "speculative_residency_rejected_d2h_copies": (
                rejected_d2h
            ),
        },
    }


def test_build_boundary_artifact_accepts_required_evidence():
    artifact = _gate().build_boundary_artifact(
        baseline_case=_artifact_case(h2d_copies=0, h2d_bytes=0),
        accept_case=_artifact_case(accepted=1, committed=1),
        reject_case=_artifact_case(rejected=1),
        source_hashes={"tracked.py": "0" * 64},
        environment={"model_path": "/model"},
    )

    assert artifact["schema_version"] == 1
    assert artifact["status"] == "PASS"
    assert artifact["classification"] == "NOT_PROMOTABLE"


@pytest.mark.parametrize(
    "accept_case,reject_case,match",
    [
        (
            {
                **_artifact_case(accepted=1, committed=1),
                "outputs": [[99]],
            },
            _artifact_case(rejected=1),
            "parity",
        ),
        (
            _artifact_case(accepted=1, committed=0),
            _artifact_case(rejected=1),
            "committed",
        ),
        (
            _artifact_case(accepted=1, committed=1),
            _artifact_case(rejected=0, proposed=1),
            "rejected blocks",
        ),
        (
            _artifact_case(accepted=1, committed=1),
            _artifact_case(rejected=1, rejected_d2h=1),
            "rejected D2H",
        ),
        (
            _artifact_case(
                accepted=1,
                committed=1,
                h2d_copies=0,
            ),
            _artifact_case(rejected=1),
            "H2D",
        ),
    ],
)
def test_build_boundary_artifact_rejects_missing_evidence(
    accept_case,
    reject_case,
    match,
):
    with pytest.raises(ValueError, match=match):
        _gate().build_boundary_artifact(
            baseline_case=_artifact_case(
                h2d_copies=0,
                h2d_bytes=0,
            ),
            accept_case=accept_case,
            reject_case=reject_case,
            source_hashes={"tracked.py": "0" * 64},
            environment={"model_path": "/model"},
        )


def _verified_artifact(tmp_path):
    source_path = tmp_path / "tracked.py"
    source_path.write_text("value = 1\n")
    source_hash = hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    artifact = _gate().build_boundary_artifact(
        baseline_case=_artifact_case(h2d_copies=0, h2d_bytes=0),
        accept_case=_artifact_case(accepted=1, committed=1),
        reject_case=_artifact_case(rejected=1),
        source_hashes={"tracked.py": source_hash},
        environment={"model_path": "/model"},
    )
    artifact_path = tmp_path / "result.json"
    artifact_path.write_text(
        json.dumps(artifact, sort_keys=True)
    )
    return artifact_path, artifact


def test_independent_verifier_recomputes_source_hashes(tmp_path):
    artifact_path, _ = _verified_artifact(tmp_path)

    result = _verifier().verify_boundary_artifact(
        artifact_path,
        tmp_path,
    )

    assert result["status"] == "PASS"
    assert result["schema_version"] == 1
    assert result["verified_source_files"] == ["tracked.py"]


def test_independent_verifier_rejects_missing_source(tmp_path):
    artifact_path, _ = _verified_artifact(tmp_path)
    (tmp_path / "tracked.py").unlink()

    with pytest.raises(FileNotFoundError, match="tracked.py"):
        _verifier().verify_boundary_artifact(
            artifact_path,
            tmp_path,
        )


def test_independent_verifier_rejects_source_hash_mismatch(
    tmp_path,
):
    artifact_path, _ = _verified_artifact(tmp_path)
    (tmp_path / "tracked.py").write_text("value = 2\n")

    with pytest.raises(ValueError, match="hash"):
        _verifier().verify_boundary_artifact(
            artifact_path,
            tmp_path,
        )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda artifact: artifact["cases"][
                "accepted_boundary"
            ].update(outputs=[[99]]),
            "parity",
        ),
        (
            lambda artifact: artifact["cases"][
                "accepted_boundary"
            ]["movement"].update(h2d_copies=-1),
            "non-negative integer",
        ),
        (
            lambda artifact: artifact["cases"][
                "rejected_boundary"
            ]["residency"].update(
                speculative_residency_rejected_d2h_copies=1
            ),
            "rejected D2H",
        ),
        (
            lambda artifact: artifact.update(
                performance_claims=["faster"]
            ),
            "performance",
        ),
    ],
)
def test_independent_verifier_rejects_tampered_artifact(
    tmp_path,
    mutate,
    match,
):
    artifact_path, artifact = _verified_artifact(tmp_path)
    mutate(artifact)
    artifact_path.write_text(
        json.dumps(artifact, sort_keys=True)
    )

    with pytest.raises(ValueError, match=match):
        _verifier().verify_boundary_artifact(
            artifact_path,
            tmp_path,
        )


def test_run_live_gate_writes_complete_artifact(tmp_path):
    gate = _gate()
    source_path = tmp_path / "tracked.py"
    source_path.write_text("value = 1\n")
    output_path = tmp_path / "result.json"
    engines = []

    def engine_factory(*args, **kwargs):
        engine = _BoundaryEngine(*args, **kwargs)
        engines.append(engine)
        return engine

    artifact = gate.run_live_gate(
        engine_factory=engine_factory,
        repo_root=tmp_path,
        model_path="/model",
        prompt_token_ids=list(range(254)),
        output_path=output_path,
        command=["boundary-gate", "run"],
        source_files=("tracked.py",),
        runtime_types=(_SamplingParams, _Runtime),
        adapter_types=(_Capabilities, _Proposal),
        environment={
            "model_path": "/model",
            "device_name": "fake",
        },
    )

    assert len(engines) == 3
    assert output_path.is_file()
    assert json.loads(output_path.read_text()) == artifact
    assert artifact["status"] == "PASS"
    assert artifact["cases"]["baseline"]["mode"] == "baseline"
    assert (
        artifact["cases"]["accepted_boundary"]["mode"]
        == "accept"
    )
    assert (
        artifact["cases"]["rejected_boundary"]["mode"]
        == "reject"
    )


def test_remote_runner_has_fixed_sync_and_verification_contract():
    source = RUNNER_PATH.read_text()

    for required in (
        'REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"',
        'CONTROL_SOCKET="${CONTROL_SOCKET:-/tmp/ssh-sitian-10.232.195.203}"',
        'REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/sitian-workspace01/tllm/env/bin/python}"',
        'MODEL_PATH="${MODEL_PATH:-/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B}"',
        'REMOTE_REPO="${REMOTE_REPO:-/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge}"',
        'GPU_ID="${GPU_ID:-0}"',
        "tinyvllm/",
        "tools/speculative_residency_boundary_gate.py",
        "tools/verify_speculative_residency_boundary_gate.py",
        "tools/test_speculative_residency_boundary_gate.py",
        'CUDA_VISIBLE_DEVICES="${gpu_id}"',
        "verify.remote.json",
        "verify.json",
        "set +e",
        '"${REMOTE_HOST}:${REMOTE_OUT}/"',
    ):
        assert required in source

    remote_verify = source.index("verify.remote.json")
    download = source.index(
        '"${REMOTE_HOST}:${REMOTE_OUT}/"'
    )
    local_verify = source.index("verify.json", download)
    assert remote_verify < download < local_verify
