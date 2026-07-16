"""Dependency-light tests for native verifier oracle comparison."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import sys
import tempfile
import types
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def _load_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(
        module_name,
        os.path.join(_REPO_ROOT, relative_path),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


tinyvllm_pkg = types.ModuleType("tinyvllm")
tinyvllm_pkg.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm")]
engine_pkg = types.ModuleType("tinyvllm.engine")
engine_pkg.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm", "engine")]
utils_pkg = types.ModuleType("tinyvllm.utils")
utils_pkg.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm", "utils")]
sys.modules.setdefault("tinyvllm", tinyvllm_pkg)
sys.modules.setdefault("tinyvllm.engine", engine_pkg)
sys.modules.setdefault("tinyvllm.utils", utils_pkg)

context_mod = types.ModuleType("tinyvllm.utils.context")
context_mod.reset_context = lambda: None
sys.modules.setdefault("tinyvllm.utils.context", context_mod)

xxhash_mod = types.ModuleType("xxhash")


class _FakeXXH64:
    def __init__(self):
        self._hash = hashlib.blake2b(digest_size=8)

    def update(self, data):
        self._hash.update(data)

    def intdigest(self):
        return int.from_bytes(self._hash.digest(), "little")


xxhash_mod.xxh64 = _FakeXXH64
sys.modules.setdefault("xxhash", xxhash_mod)
_load_module(
    "tinyvllm.sampling_params",
    "tinyvllm/sampling_params.py",
)
_load_module(
    "tinyvllm.engine.sequence",
    "tinyvllm/engine/sequence.py",
)
block_manager_module = _load_module(
    "tinyvllm.engine.block_manager",
    "tinyvllm/engine/block_manager.py",
)
BlockManager = block_manager_module.BlockManager

_ORACLE_PATH = os.path.join(_THIS_DIR, "native_verifier_oracle.py")
_SPEC = importlib.util.spec_from_file_location(
    "native_verifier_oracle_under_test",
    _ORACLE_PATH,
)
oracle = importlib.util.module_from_spec(_SPEC)
sys.modules["native_verifier_oracle_under_test"] = oracle
_SPEC.loader.exec_module(oracle)

compare_native_and_oracle = oracle.compare_native_and_oracle
dtype_tolerance = oracle.dtype_tolerance
build_case_payload = oracle.build_case_payload
construct_draft_tokens = oracle.construct_draft_tokens
run_case = oracle.run_case


class _FakeTensor:
    def __init__(self, values):
        self.values = values

    def argmax(self, dim=-1):
        assert dim == -1
        rows = self.values
        if rows and not isinstance(rows[0], list):
            rows = [rows]
        return _FakeTensor([
            max(range(len(row)), key=row.__getitem__)
            for row in rows
        ])

    def detach(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def float(self):
        return self

    def tolist(self):
        return self.values


class _FakeSequence:
    block_size = 4

    def __init__(self, token_ids):
        self.token_ids = list(token_ids)
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        self.last_token = self.token_ids[-1]
        self.block_table = []
        self.num_cached_tokens = 0
        self.num_computed_tokens = 0
        self.ignore_eos = True
        self.max_tokens = 32

    def __len__(self):
        return self.num_tokens

    @property
    def num_blocks(self):
        return (
            self.num_tokens + self.block_size - 1
        ) // self.block_size

    @property
    def last_block_num_tokens(self):
        return (
            self.num_tokens
            - (self.num_blocks - 1) * self.block_size
        )

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    def block(self, index):
        start = index * self.block_size
        return self.token_ids[start:start + self.block_size]

    def append_token(self, token_id):
        self.token_ids.append(int(token_id))
        self.num_tokens += 1
        self.last_token = int(token_id)


class _SerializedModelRunner:
    def __init__(self):
        self.prepare_calls = []

    def run(self, seqs, is_prefill):
        assert is_prefill is False
        assert len(seqs) == 1
        return [11]

    def prepare_decode(self, seqs, *, flash_attn_num_splits=0):
        assert flash_attn_num_splits in (0, 16)
        for seq in seqs:
            self.prepare_calls.append({
                "last_token": int(seq.last_token),
                "block_table": list(seq.block_table),
                "num_blocks": int(seq.num_blocks),
            })
        return (
            [int(seq.last_token) for seq in seqs],
            [len(seq) for seq in seqs],
        )

    def run_model(
        self,
        input_ids,
        positions,
        *,
        is_prefill,
        execution_mode,
    ):
        assert is_prefill is False
        assert execution_mode == "decode"
        target_by_pending = {
            7: 11,
            11: 22,
            22: 33,
        }
        rows = []
        prepare_calls = self.prepare_calls[-len(input_ids):]
        for pending_token, prepare_call in zip(
            input_ids,
            prepare_calls,
        ):
            if (
                len(prepare_call["block_table"])
                != prepare_call["num_blocks"]
            ):
                target = 11
            else:
                target = target_by_pending[int(pending_token)]
            logits = [0.0] * 34
            logits[target] = 1.0
            rows.append(logits)
        return _FakeTensor(rows)

    def snapshot_kv_slots(self, physical_slots):
        slots = [int(slot) for slot in physical_slots]
        return {
            "keys": _FakeTensor([
                [[float(slot)] for slot in slots],
            ]),
            "values": _FakeTensor([
                [[float(slot + 1)] for slot in slots],
            ]),
        }


class _RowExpandedModelRunner:
    def __init__(self):
        self.prepare_batches = []

    def run(self, seqs, is_prefill):
        assert is_prefill is False
        assert len(seqs) == 1
        return [11]

    def prepare_decode(self, seqs, *, flash_attn_num_splits=0):
        assert flash_attn_num_splits == 16
        self.prepare_batches.append([
            {
                "last_token": int(seq.last_token),
                "num_tokens": len(seq),
                "block_table": list(seq.block_table),
                "num_blocks": int(seq.num_blocks),
            }
            for seq in seqs
        ])
        return (
            [int(seq.last_token) for seq in seqs],
            [len(seq) for seq in seqs],
        )

    def run_model(
        self,
        input_ids,
        positions,
        *,
        is_prefill,
        execution_mode,
    ):
        assert is_prefill is False
        assert execution_mode == "decode"
        target_by_pending = {
            11: 22,
            22: 33,
        }
        rows = []
        for pending_token in input_ids:
            target = target_by_pending[int(pending_token)]
            logits = [0.0] * 34
            logits[target] = 1.0
            rows.append(logits)
        return _FakeTensor(rows)

    def snapshot_kv_slots(self, physical_slots):
        slots = [int(slot) for slot in physical_slots]
        return {
            "keys": _FakeTensor([
                [[float(slot)] for slot in slots],
                [[float(slot + 10)] for slot in slots],
            ]),
            "values": _FakeTensor([
                [[float(slot + 20)] for slot in slots],
                [[float(slot + 30)] for slot in slots],
            ]),
        }


class _FakeScheduler:
    def __init__(self, block_manager):
        self.block_manager = block_manager
        self.eos = 999


class _FakeLLM:
    def __init__(self, block_manager, model_runner=None):
        self.scheduler = _FakeScheduler(block_manager)
        self.model_runner = (
            model_runner
            if model_runner is not None
            else _SerializedModelRunner()
        )


def test_tinyvllm_backend_has_runtime_timer_dependency():
    assert callable(oracle.time.perf_counter)


def make_comparison_fixture():
    return {
        "dtype": "torch.float16",
        "target_tokens": [4, 5, 6],
        "accepted_tokens": [4, 5],
        "sequence_tokens_after": [1, 2, 3, 4, 5],
        "block_table_after": [0],
        "continuation_tokens": list(range(16)),
        "logits": [[0.0, 1.0], [1.0, 0.0]],
        "kv": {
            "keys": [[0.0, 1.0]],
            "values": [[1.0, 0.0]],
        },
        "continuation_logits": [[[0.0, 1.0]]],
        "continuation_kv": [
            {
                "keys": [[0.0, 1.0]],
                "values": [[1.0, 0.0]],
            }
        ],
        "finite": True,
    }


def test_dtype_tolerances_are_fixed():
    fp16 = dtype_tolerance("torch.float16")
    bf16 = dtype_tolerance("torch.bfloat16")
    assert fp16.logits_rtol == 2e-3
    assert fp16.logits_atol == 2e-3
    assert fp16.kv_rtol == 2e-3
    assert fp16.kv_atol == 2e-3
    assert bf16.logits_rtol == 8e-3
    assert bf16.logits_atol == 8e-3
    assert bf16.kv_rtol == 8e-3
    assert bf16.kv_atol == 8e-3


def test_comparison_requires_tokens_acceptance_metadata_and_continuation():
    payload = make_comparison_fixture()
    comparison = compare_native_and_oracle(payload, dict(payload))

    assert comparison["status"] == "PASS"
    assert comparison["target_token_match"] is True
    assert comparison["accepted_prefix_match"] is True
    assert comparison["metadata_match"] is True
    assert comparison["continuation_token_match"] is True
    assert comparison["continuation_steps"] == 16
    assert comparison["logits_within_tolerance"] is True
    assert comparison["kv_within_tolerance"] is True


def test_token_mismatch_is_no_go_even_when_numeric_error_is_small():
    native = make_comparison_fixture()
    oracle_payload = make_comparison_fixture()
    oracle_payload["continuation_tokens"][-1] += 1

    comparison = compare_native_and_oracle(native, oracle_payload)

    assert comparison["status"] == "NO_GO"
    assert "continuation token mismatch" in comparison["reasons"]


def test_acceptance_or_metadata_mismatch_is_no_go():
    native = make_comparison_fixture()
    oracle_payload = make_comparison_fixture()
    oracle_payload["accepted_tokens"] = [4]
    oracle_payload["block_table_after"] = [0, 1]

    comparison = compare_native_and_oracle(native, oracle_payload)

    assert comparison["status"] == "NO_GO"
    assert "accepted prefix mismatch" in comparison["reasons"]
    assert "committed metadata mismatch" in comparison["reasons"]


def test_numeric_mismatch_is_no_go():
    native = make_comparison_fixture()
    oracle_payload = make_comparison_fixture()
    oracle_payload["logits"] = [[0.0, 2.0], [1.0, 0.0]]
    oracle_payload["kv"] = {
        "keys": [[0.0, 2.0]],
        "values": [[1.0, 0.0]],
    }

    comparison = compare_native_and_oracle(native, oracle_payload)

    assert comparison["status"] == "NO_GO"
    assert "logits exceed tolerance" in comparison["reasons"]
    assert "KV exceeds tolerance" in comparison["reasons"]


def test_continuation_numeric_mismatch_is_no_go():
    native = make_comparison_fixture()
    oracle_payload = make_comparison_fixture()
    oracle_payload["continuation_logits"] = [[[0.0, 2.0]]]
    oracle_payload["continuation_kv"] = [{
        "keys": [[0.0, 2.0]],
        "values": [[1.0, 0.0]],
    }]

    comparison = compare_native_and_oracle(native, oracle_payload)

    assert comparison["status"] == "NO_GO"
    assert "logits exceed tolerance" in comparison["reasons"]
    assert "KV exceeds tolerance" in comparison["reasons"]


def test_missing_or_nonfinite_evidence_is_classified_strictly():
    native = make_comparison_fixture()
    oracle_payload = make_comparison_fixture()
    native["finite"] = False
    comparison = compare_native_and_oracle(native, oracle_payload)
    assert comparison["status"] == "NO_GO"
    assert "non-finite logits or KV" in comparison["reasons"]

    native = make_comparison_fixture()
    oracle_payload = make_comparison_fixture()
    del oracle_payload["kv"]
    comparison = compare_native_and_oracle(native, oracle_payload)
    assert comparison["status"] == "INCOMPLETE"
    assert "missing oracle field: kv" in comparison["reasons"]


def test_less_than_16_continuation_steps_is_incomplete():
    native = make_comparison_fixture()
    oracle_payload = make_comparison_fixture()
    native["continuation_tokens"] = list(range(8))
    oracle_payload["continuation_tokens"] = list(range(8))

    comparison = compare_native_and_oracle(native, oracle_payload)

    assert comparison["status"] == "INCOMPLETE"
    assert "continuation coverage below 16 steps" in comparison["reasons"]


def test_build_case_payload_records_complete_evidence():
    evidence = {
        "dtype": "torch.bfloat16",
        "target_tokens": [4, 5],
        "accepted_tokens": [4],
        "sequence_tokens_after": [1, 2, 4],
        "block_table_after": [7],
        "continuation_tokens": list(range(16)),
        "logits": [[0.0, 1.0]],
        "kv": {"keys": [[0.0]], "values": [[1.0]]},
        "continuation_logits": [[[0.0, 1.0]]],
        "continuation_kv": [
            {"keys": [[0.0]], "values": [[1.0]]}
        ],
        "physical_slots": [2],
        "policy": "oracle",
        "case_id": "case-1",
    }

    payload = build_case_payload(evidence)

    assert payload["finite"] is True
    assert payload["case_id"] == "case-1"
    assert payload["policy"] == "oracle"
    assert payload["tolerance"] == {
        "logits_rtol": 8e-3,
        "logits_atol": 8e-3,
        "kv_rtol": 8e-3,
        "kv_atol": 8e-3,
    }


def test_oracle_kv_rows_are_aggregated_layer_major():
    total = {"keys": [], "values": []}
    oracle._append_kv_rows(total, {
        "keys": [
            [["l0s0"]],
            [["l1s0"]],
        ],
        "values": [
            [["v0s0"]],
            [["v1s0"]],
        ],
    })
    oracle._append_kv_rows(total, {
        "keys": [
            [["l0s1"]],
            [["l1s1"]],
        ],
        "values": [
            [["v0s1"]],
            [["v1s1"]],
        ],
    })

    assert total == {
        "keys": [
            [["l0s0"], ["l0s1"]],
            [["l1s0"], ["l1s1"]],
        ],
        "values": [
            [["v0s0"], ["v0s1"]],
            [["v1s0"], ["v1s1"]],
        ],
    }


def test_run_case_validates_input_and_writes_backend_payload():
    calls = []

    def fake_backend(**kwargs):
        calls.append(kwargs)
        return build_case_payload({
            "dtype": "torch.float16",
            "target_tokens": [4],
            "accepted_tokens": [4],
            "sequence_tokens_after": [1, 4],
            "block_table_after": [0],
            "continuation_tokens": list(range(16)),
            "logits": [[0.0, 1.0]],
            "kv": {"keys": [[0.0]], "values": [[1.0]]},
            "continuation_logits": [],
            "continuation_kv": [],
            "physical_slots": [],
            "policy": kwargs["policy"],
            "case_id": kwargs["case"]["case_id"],
        })

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "case.json"
        result = run_case(
            policy="native",
            case={
                "case_id": "case-1",
                "prompt": "hello",
                "history_len": 8,
                "draft_tokens": [4],
                "max_tokens": 32,
                "ignore_eos": True,
            },
            out_path=out_path,
            model="/model",
            continuation_steps=16,
            backend=fake_backend,
        )
        written = json.loads(out_path.read_text())

    assert result == written
    assert calls == [{
        "policy": "native",
        "case": {
            "case_id": "case-1",
            "prompt": "hello",
            "history_len": 8,
            "draft_tokens": [4],
            "max_tokens": 32,
            "ignore_eos": True,
        },
        "model": "/model",
        "continuation_steps": 16,
    }]

    invalid_cases = (
        ({}, "case_id"),
        ({
            "case_id": "x",
            "prompt": "hello",
            "history_len": 8,
            "draft_tokens": [],
            "max_tokens": 32,
            "ignore_eos": True,
        }, "draft_tokens"),
    )
    for case, expected in invalid_cases:
        try:
            run_case(
                policy="oracle",
                case=case,
                out_path=Path("/unused"),
                model="/model",
                continuation_steps=16,
                backend=fake_backend,
            )
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("invalid oracle case must fail")


def test_run_case_accepts_all_isolated_policies():
    seen = []

    def fake_backend(**kwargs):
        seen.append(kwargs["policy"])
        return {
            "policy": kwargs["policy"],
            "case_id": kwargs["case"]["case_id"],
        }

    with tempfile.TemporaryDirectory() as tmp:
        for policy in (
            "probe",
            "baseline",
            "legacy_rematerialize",
            "native",
            "oracle",
        ):
            run_case(
                policy=policy,
                case={
                    "case_id": "case-1",
                    "prompt": "hello",
                    "history_len": 8,
                    "draft_tokens": [4],
                    "max_tokens": 32,
                    "ignore_eos": True,
                },
                out_path=Path(tmp) / f"{policy}.json",
                model="/model",
                continuation_steps=16,
                backend=fake_backend,
            )
    assert seen == [
        "probe",
        "baseline",
        "legacy_rematerialize",
        "native",
        "oracle",
    ]


def test_construct_draft_tokens_is_deterministic_for_all_acceptance_cases():
    targets = [10, 20, 30, 40]
    assert construct_draft_tokens(
        targets,
        acceptance_case="full",
        vocab_size=100,
    ) == [10, 20, 30, 40]
    assert construct_draft_tokens(
        targets,
        acceptance_case="partial",
        vocab_size=100,
    ) == [10, 20, 31, 40]
    assert construct_draft_tokens(
        targets,
        acceptance_case="one",
        vocab_size=100,
    ) == [10, 21, 30, 40]
    assert construct_draft_tokens(
        targets,
        acceptance_case="zero",
        vocab_size=100,
    ) == [11, 20, 30, 40]

    try:
        construct_draft_tokens(
            [10],
            acceptance_case="partial",
            vocab_size=100,
        )
    except ValueError as exc:
        assert "partial" in str(exc)
    else:
        raise AssertionError("partial K=1 must fail")


def test_serialized_oracle_consumes_each_pending_draft_token_once():
    block_manager = BlockManager(num_blocks=8, block_size=4)
    seq = _FakeSequence([1, 2, 7])
    block_manager.allocate(seq)
    llm = _FakeLLM(block_manager)

    result = oracle._run_serialized_oracle_verify(
        llm,
        seq,
        [11, 22, 33],
    )

    assert result["target_tokens"] == [11, 22, 33]
    assert [
        call["last_token"]
        for call in llm.model_runner.prepare_calls
    ] == [11, 22]
    assert all(
        len(call["block_table"]) == call["num_blocks"]
        for call in llm.model_runner.prepare_calls
    )


def test_oracle_expands_tail_queries_into_one_decode_batch():
    block_manager = BlockManager(num_blocks=8, block_size=4)
    seq = _FakeSequence([1, 2, 7])
    block_manager.allocate(seq)
    runner = _RowExpandedModelRunner()
    llm = _FakeLLM(block_manager, runner)

    result = oracle._run_serialized_oracle_verify(
        llm,
        seq,
        [11, 22, 33],
    )

    assert result["target_tokens"] == [11, 22, 33]
    assert len(runner.prepare_batches) == 1
    assert runner.prepare_batches[0] == [
        {
            "last_token": 11,
            "num_tokens": 4,
            "block_table": [0],
            "num_blocks": 1,
        },
        {
            "last_token": 22,
            "num_tokens": 5,
            "block_table": [0, 1],
            "num_blocks": 2,
        },
    ]
    assert result["physical_slots"] == [3, 4]
    assert result["kv"] == {
        "keys": [
            [[3.0], [4.0]],
            [[13.0], [14.0]],
        ],
        "values": [
            [[23.0], [24.0]],
            [[33.0], [34.0]],
        ],
    }


def test_baseline_commits_block_metadata_before_continuation():
    block_manager = BlockManager(num_blocks=8, block_size=4)
    seq = _FakeSequence([1, 2, 7])
    block_manager.allocate(seq)
    llm = _FakeLLM(block_manager)

    result = oracle._run_baseline_verify(
        llm,
        seq,
        [11, 22, 99],
    )

    assert result["target_tokens"] == [11, 22, 33]
    assert result["accepted_tokens"] == [11, 22]
    assert seq.token_ids == [1, 2, 7, 11, 22]
    assert len(seq.block_table) == 1
    assert block_manager.blocks[seq.block_table[-1]].hash != -1

    block_manager.may_append(seq)
    assert len(seq.block_table) == 2


def main():
    test_tinyvllm_backend_has_runtime_timer_dependency()
    test_dtype_tolerances_are_fixed()
    test_comparison_requires_tokens_acceptance_metadata_and_continuation()
    test_token_mismatch_is_no_go_even_when_numeric_error_is_small()
    test_acceptance_or_metadata_mismatch_is_no_go()
    test_numeric_mismatch_is_no_go()
    test_continuation_numeric_mismatch_is_no_go()
    test_missing_or_nonfinite_evidence_is_classified_strictly()
    test_less_than_16_continuation_steps_is_incomplete()
    test_build_case_payload_records_complete_evidence()
    test_oracle_kv_rows_are_aggregated_layer_major()
    test_run_case_validates_input_and_writes_backend_payload()
    test_run_case_accepts_all_isolated_policies()
    test_construct_draft_tokens_is_deterministic_for_all_acceptance_cases()
    test_serialized_oracle_consumes_each_pending_draft_token_once()
    test_oracle_expands_tail_queries_into_one_decode_batch()
    test_baseline_commits_block_metadata_before_continuation()
    print("native verifier oracle tests passed")


if __name__ == "__main__":
    main()
