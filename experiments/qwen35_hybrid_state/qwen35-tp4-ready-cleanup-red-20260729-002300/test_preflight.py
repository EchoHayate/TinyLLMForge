from __future__ import annotations

import copy
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_PATH = (
    ROOT
    / "tools/qwen35_tp4_real_root_logit_correctness_preflight.py"
)


def _load_preflight():
    spec = importlib.util.spec_from_file_location(
        "qwen35_tp4_real_root_logit_correctness_preflight",
        PREFLIGHT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


preflight = _load_preflight()


def _expect_value_error(function, message):
    try:
        function()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")


def _manual_attention(query, key, value):
    token_count, query_heads, head_dim = query.shape
    key = key.repeat_interleave(query_heads // key.shape[1], dim=1)
    value = value.repeat_interleave(
        query_heads // value.shape[1],
        dim=1,
    )
    scores = torch.einsum(
        "thd,shd->hts",
        query.float(),
        key.float(),
    )
    scores = scores * (head_dim ** -0.5)
    mask = torch.ones(
        token_count,
        token_count,
        dtype=torch.bool,
    ).tril()
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum(
        "hts,shd->thd",
        probabilities,
        value.float(),
    )


def test_tp4_attention_matches_manual_fp32_two_query_one_kv():
    backend = preflight.Qwen35TP4CausalAttentionBackend(
        local_query_heads=2,
        local_kv_heads=1,
        head_dim=4,
    )
    query = torch.tensor(
        [
            [1, 2, 3, 4, 4, 3, 2, 1],
            [2, 1, 0, -1, 1, 0, 1, 0],
            [0, 1, 0, 1, -1, -2, -3, -4],
        ],
        dtype=torch.bfloat16,
    )
    key = torch.tensor(
        [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.bfloat16,
    )
    value = torch.tensor(
        [
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [2, 4, 6, 8],
        ],
        dtype=torch.bfloat16,
    )

    actual = backend(query, key, value)
    expected = _manual_attention(
        query.reshape(3, 2, 4),
        key.reshape(3, 1, 4),
        value.reshape(3, 1, 4),
    ).reshape(3, 8)

    assert actual.dtype == torch.bfloat16
    assert actual.shape == (3, 8)
    torch.testing.assert_close(
        actual.float(),
        expected,
        atol=2e-2,
        rtol=2e-2,
    )


def test_tp4_attention_is_causal_under_future_token_poisoning():
    backend = preflight.Qwen35TP4CausalAttentionBackend(
        local_query_heads=2,
        local_kv_heads=1,
        head_dim=2,
    )
    query = torch.arange(
        16,
        dtype=torch.float32,
    ).reshape(4, 4).to(torch.bfloat16)
    key = torch.arange(
        8,
        dtype=torch.float32,
    ).reshape(4, 2).to(torch.bfloat16)
    value = torch.arange(
        8,
        dtype=torch.float32,
    ).reshape(4, 2).to(torch.bfloat16)
    poisoned_key = key.clone()
    poisoned_value = value.clone()
    poisoned_key[3] = 1000
    poisoned_value[3] = -1000

    baseline = backend(query, key, value)
    poisoned = backend(query, poisoned_key, poisoned_value)

    torch.testing.assert_close(baseline[:3], poisoned[:3])
    assert not torch.equal(baseline[3], poisoned[3])


def test_tp4_attention_rejects_invalid_construction_and_inputs():
    invalid_constructors = (
        (
            lambda: preflight.Qwen35TP4CausalAttentionBackend(
                local_query_heads=1,
                local_kv_heads=1,
                head_dim=4,
            ),
            "local_query_heads",
        ),
        (
            lambda: preflight.Qwen35TP4CausalAttentionBackend(
                local_query_heads=2,
                local_kv_heads=2,
                head_dim=4,
            ),
            "local_kv_heads",
        ),
        (
            lambda: preflight.Qwen35TP4CausalAttentionBackend(
                local_query_heads=2,
                local_kv_heads=1,
                head_dim=0,
            ),
            "head_dim",
        ),
    )
    for function, message in invalid_constructors:
        _expect_value_error(function, message)

    backend = preflight.Qwen35TP4CausalAttentionBackend(
        local_query_heads=2,
        local_kv_heads=1,
        head_dim=4,
    )
    query = torch.zeros(3, 8, dtype=torch.bfloat16)
    key = torch.zeros(3, 4, dtype=torch.bfloat16)
    value = torch.zeros(3, 4, dtype=torch.bfloat16)
    invalid_inputs = (
        (
            lambda: backend(query[:, :7], key, value),
            "query width",
        ),
        (
            lambda: backend(query, key[:, :3], value),
            "key width",
        ),
        (
            lambda: backend(query, key, value[:2]),
            "token count",
        ),
        (
            lambda: backend(query.float(), key, value),
            "dtype",
        ),
        (
            lambda: backend(
                torch.full_like(query, math.nan),
                key,
                value,
            ),
            "finite",
        ),
    )
    for function, message in invalid_inputs:
        _expect_value_error(function, message)


class _FakePool:
    def __init__(self):
        self.device = torch.device("cpu")
        self._tensors = {
            (0, "linear_convolution"): torch.zeros(1, 2, 3),
            (0, "linear_recurrent"): torch.zeros(1, 2, 2, 2),
        }
        self._bindings = {}
        self.events = []

    def activate(self, lease):
        self.events.append(("activate", lease.generation))
        self._bindings[lease.slot_id] = (
            lease.request_id,
            lease.generation,
        )
        for tensor in self._tensors.values():
            tensor[lease.slot_id].zero_()

    def release(self, lease):
        self.events.append(("release", lease.generation))
        for tensor in self._tensors.values():
            tensor[lease.slot_id].zero_()
        del self._bindings[lease.slot_id]


class _FakeAllocator:
    def __init__(self):
        self.generation = 0
        self.events = []

    def allocate(self, request_id):
        self.generation += 1
        lease = SimpleNamespace(
            slot_id=0,
            generation=self.generation,
            request_id=request_id,
        )
        self.events.append(("allocate", request_id, self.generation))
        return lease

    def release(self, lease):
        self.events.append((
            "release",
            lease.request_id,
            lease.generation,
        ))


class _FakeNativeModel:
    def __init__(self, pool, *, rank, invalid_output=None):
        self.pool = pool
        self.rank = rank
        self.invalid_output = invalid_output
        self.events = []

    def run_step(
        self,
        leases,
        token_counts,
        input_ids,
        position_ids,
        input_embeds=None,
    ):
        self.events.append((
            "run_step",
            tuple(lease.generation for lease in leases),
            token_counts,
            input_ids.detach().cpu().tolist(),
            position_ids.detach().cpu().tolist(),
            input_embeds,
        ))
        for tensor in self.pool._tensors.values():
            tensor[leases[0].slot_id].fill_(leases[0].generation)
        normalized = torch.arange(
            input_ids.shape[0] * 3,
            dtype=torch.float32,
        ).reshape(input_ids.shape[0], 3)
        if self.invalid_output == "root_none":
            return normalized, None
        if self.invalid_output == "non_root_tensor":
            return normalized, torch.zeros(
                preflight.MODEL_VOCAB_SIZE,
                dtype=torch.float32,
            )
        if self.rank == 0:
            return normalized, torch.arange(
                preflight.MODEL_VOCAB_SIZE,
                dtype=torch.float32,
            ).reshape(1, -1)
        return normalized, None


def _fake_tp4_candidate(
    *,
    rank,
    pool,
    fingerprint=None,
    invalid_output=None,
):
    model = _FakeNativeModel(
        pool,
        rank=rank,
        invalid_output=invalid_output,
    )
    return SimpleNamespace(
        owner=SimpleNamespace(model=model, pool=pool),
        binding_plan=SimpleNamespace(
            tensor_parallel_size=4,
            tensor_parallel_rank=rank,
        ),
        model_fingerprint=(
            preflight.APPROVED_MODEL_MANIFEST_SHA256
            if fingerprint is None
            else fingerprint
        ),
    )


def _prompt(case_id, tokens):
    return SimpleNamespace(
        case_id=case_id,
        token_ids=tuple(tokens),
        token_sha256="a" * 64,
    )


def test_real_tp4_candidate_builder_calls_authorized_stack_once_for_rank_two():
    events = []
    metadata = SimpleNamespace(
        hf_config=object(),
        index_payload={"weight_map": {"x": "shard"}},
        shard_headers={"shard": {"x": {}}},
    )
    tensor_plan = object()
    layout = object()
    pool = _FakePool()
    target = SimpleNamespace(pool=pool)
    request = object()
    candidate = _fake_tp4_candidate(rank=2, pool=pool)

    class _Dependencies:
        def make_shard_identity(self, **kwargs):
            events.append(("shard", kwargs))
            return "shard-identity"

        def read_metadata(self, checkpoint_dir, **kwargs):
            events.append(("metadata", checkpoint_dir, kwargs))
            return metadata

        def build_tensor_plan(self, hf_config, index_payload, shard_headers):
            events.append((
                "plan",
                hf_config,
                index_payload,
                shard_headers,
            ))
            return tensor_plan

        def build_layout(self, hf_config, **kwargs):
            events.append(("layout", hf_config, kwargs))
            return layout

        def make_pool(self, received_layout, **kwargs):
            events.append(("pool", received_layout, kwargs))
            return pool

        def prepare_target(self, hf_config, received_plan, **kwargs):
            events.append((
                "target",
                hf_config,
                received_plan,
                kwargs,
            ))
            return target

        def build_loader(self, provider, **kwargs):
            events.append(("loader", kwargs))

            def load(received_request):
                events.append(("load", received_request))
                assert provider() is target
                return candidate

            return load

        def make_request(self, **kwargs):
            events.append(("request", kwargs))
            return request

    result = preflight.build_real_tp4_cpu_candidate(
        rank=2,
        dependencies=_Dependencies(),
    )

    assert result.candidate is candidate
    assert result.pool is pool
    assert result.metadata is metadata
    assert result.tensor_plan is tensor_plan
    assert [event[0] for event in events] == [
        "shard",
        "metadata",
        "plan",
        "layout",
        "pool",
        "target",
        "loader",
        "request",
        "load",
    ]
    assert events[3][2] == {
        "tensor_parallel_size": 4,
        "dtype": torch.bfloat16,
        "recurrent_dtype": torch.float32,
        "speculative_tokens": 1,
    }
    assert events[4][2] == {"capacity": 1, "device": "cpu"}
    assert events[5][3]["tensor_parallel_size"] == 4
    assert events[5][3]["tensor_parallel_rank"] == 2
    assert events[5][3]["parameter_device"] == "cpu"
    backend = events[5][3]["build_attention_backend"](
        3,
        2,
        1,
        128,
    )
    assert isinstance(
        backend,
        preflight.Qwen35TP4CausalAttentionBackend,
    )
    assert backend.local_query_heads == 2
    assert backend.local_kv_heads == 1
    assert backend.head_dim == 128
    assert events[6][1] == {
        "authorization_sha256": preflight.AUTHORIZATION_SHA256,
    }
    assert events[7][1] == {
        "checkpoint_dir": preflight.APPROVED_MODEL_DIR,
        "model_fingerprint": preflight.APPROVED_MODEL_MANIFEST_SHA256,
        "max_tensor_bytes": preflight.MAX_TENSOR_BYTES,
        "authorization_sha256": preflight.AUTHORIZATION_SHA256,
    }


def test_real_tp4_candidate_builder_rejects_rank_or_identity_drift():
    for rank, message in ((-1, "rank"), (4, "rank")):
        _expect_value_error(
            lambda rank=rank: preflight.build_real_tp4_cpu_candidate(
                rank=rank,
                dependencies=object(),
            ),
            message,
        )

    pool = _FakePool()

    class _Dependencies:
        def make_shard_identity(self, **_kwargs):
            return object()

        def read_metadata(self, *_args, **_kwargs):
            return SimpleNamespace(
                hf_config=object(),
                index_payload={},
                shard_headers={},
            )

        def build_tensor_plan(self, *_args):
            return object()

        def build_layout(self, *_args, **_kwargs):
            return object()

        def make_pool(self, *_args, **_kwargs):
            return pool

        def prepare_target(self, *_args, **_kwargs):
            return SimpleNamespace(pool=pool)

        def build_loader(self, provider, **_kwargs):
            def load(_request):
                provider()
                return _fake_tp4_candidate(
                    rank=1,
                    pool=pool,
                    fingerprint="1" * 64,
                )

            return load

        def make_request(self, **_kwargs):
            return object()

    _expect_value_error(
        lambda: preflight.build_real_tp4_cpu_candidate(
            rank=1,
            dependencies=_Dependencies(),
        ),
        "model fingerprint",
    )


def test_move_tp4_candidate_migrates_model_pool_and_adapter_aliases():
    pool = _FakePool()
    model = torch.nn.Linear(2, 2, bias=False)
    model.pool = pool
    adapter = SimpleNamespace(
        pool=pool,
        layer_index=0,
        convolution=pool._tensors[(0, "linear_convolution")],
        recurrent=pool._tensors[(0, "linear_recurrent")],
    )
    owner = SimpleNamespace(
        model=model,
        pool=pool,
        state_transaction=SimpleNamespace(
            pool=pool,
            adapters=(adapter,),
        ),
        runtime_bridge=SimpleNamespace(pool=pool),
    )
    candidate = SimpleNamespace(
        owner=owner,
        binding_plan=SimpleNamespace(
            tensor_parallel_size=4,
            tensor_parallel_rank=2,
        ),
        model_fingerprint=preflight.APPROVED_MODEL_MANIFEST_SHA256,
    )

    moved = preflight.move_loaded_tp4_candidate_to_device(
        candidate,
        rank=2,
        device="meta",
        expected_model_fingerprint=(
            preflight.APPROVED_MODEL_MANIFEST_SHA256
        ),
    )

    assert moved is candidate
    assert pool.device == torch.device("meta")
    assert all(tensor.device.type == "meta" for tensor in pool._tensors.values())
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
    assert adapter.convolution is pool._tensors[(0, "linear_convolution")]
    assert adapter.recurrent is pool._tensors[(0, "linear_recurrent")]


def test_gpu_selection_and_dynamic_ports_are_unique():
    rows = [
        {
            "gpu_index": index,
            "gpu_uuid": f"GPU-{index}",
            "free_bytes": preflight.MIN_GPU_FREE_BYTES + index,
            "compute_processes": [],
        }
        for index in (5, 1, 3, 2, 7)
    ]

    selected = preflight.select_tp4_gpu_resources(
        rows,
        minimum_free_bytes=preflight.MIN_GPU_FREE_BYTES,
    )

    assert tuple(row["gpu_index"] for row in selected) == (1, 2, 3, 5)
    assert tuple(row["rank"] for row in selected) == (0, 1, 2, 3)
    first, second = preflight.fresh_port_pair()
    assert isinstance(first, int)
    assert isinstance(second, int)
    assert first > 0
    assert second > 0
    assert first != second

    rows[0]["compute_processes"] = [123]
    _expect_value_error(
        lambda: preflight.select_tp4_gpu_resources(
            rows[:4],
            minimum_free_bytes=preflight.MIN_GPU_FREE_BYTES,
        ),
        "four",
    )


def test_gpu_query_maps_active_compute_processes_by_uuid():
    calls = []

    def command_runner(command, **_kwargs):
        calls.append(tuple(command))
        if "--query-gpu=index,uuid,name,memory.total,memory.free" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "0, GPU-a, A100, 81920, 70000\n"
                    "1, GPU-b, A100, 81920, 71000\n"
                ),
                stderr="",
            )
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "GPU-a, 101, python, 436\n"
                "GPU-a, 102, worker, 2048\n"
            ),
            stderr="",
        )

    rows = preflight._query_tp4_gpu_resources(
        command_runner=command_runner
    )

    assert len(calls) == 2
    assert rows[0]["compute_processes"] == [
        {
            "pid": 101,
            "process_name": "python",
            "used_bytes": 436 * 1024**2,
        },
        {
            "pid": 102,
            "process_name": "worker",
            "used_bytes": 2048 * 1024**2,
        },
    ]
    assert rows[1]["compute_processes"] == []
    selected = preflight.select_tp4_gpu_resources(
        [
            *rows,
            {
                "gpu_index": 2,
                "gpu_uuid": "GPU-c",
                "gpu_name": "A100",
                "total_bytes": 80 * 1024**3,
                "free_bytes": 70 * 1024**3,
                "compute_processes": [],
            },
            {
                "gpu_index": 3,
                "gpu_uuid": "GPU-d",
                "gpu_name": "A100",
                "total_bytes": 80 * 1024**3,
                "free_bytes": 70 * 1024**3,
                "compute_processes": [],
            },
            {
                "gpu_index": 4,
                "gpu_uuid": "GPU-e",
                "gpu_name": "A100",
                "total_bytes": 80 * 1024**3,
                "free_bytes": 70 * 1024**3,
                "compute_processes": [],
            },
        ]
    )
    assert tuple(row["gpu_index"] for row in selected) == (1, 2, 3, 4)


def test_rank_evidence_requires_complete_unique_clean_group():
    rows = [
        {
            "rank": rank,
            "world_size": 4,
            "pid": 100 + rank,
            "gpu_index": rank + 4,
            "gpu_uuid": f"GPU-{rank}",
            "process_group_nonce": "n" * 32,
            "rendezvous": "tcp://127.0.0.1:41001",
            "case_barrier_count": 3,
            "final_barrier_completed": True,
            "process_group_destroyed": True,
            "candidate_reference_dropped": True,
            "model_reference_dropped": True,
            "cuda_synchronized": True,
            "cuda_cache_emptied": True,
            "root_logits_present": rank == 0,
            "non_root_logits_none": rank != 0,
            "global_query_heads": 8,
            "global_kv_heads": 2,
            "local_query_heads": 2,
            "local_kv_heads": 1,
            "kv_head_replicas": 2,
            "source_kv_rank": rank // 2,
            "collective_events": [
                event
                for case_index in range(3)
                for event in (
                    {
                        "ordinal": case_index * 2,
                        "collective": "all_reduce",
                        "shape": [17, 2048],
                        "dtype": "torch.bfloat16",
                        "async_op": False,
                    },
                    {
                        "ordinal": case_index * 2 + 1,
                        "collective": "gather",
                        "shape": [1, 62080],
                        "dtype": "torch.bfloat16",
                        "destination": 0,
                        "receive_count": 4 if rank == 0 else None,
                        "async_op": False,
                    },
                )
            ],
            "state_rows": [
                {
                    "case_id": case_id,
                    "changed_component_count": 36,
                    "state_nonzero_after_commit": {
                        **{
                            f"{layer}:linear_convolution": True
                            for layer in range(18)
                        },
                        **{
                            f"{layer}:linear_recurrent": True
                            for layer in range(18)
                        },
                    },
                    "release_zeroed": True,
                    "pool_binding_released": True,
                }
                for case_id in ("p17", "p65", "synthetic")
            ],
        }
        for rank in range(4)
    ]

    validated = preflight.validate_rank_evidence(rows)

    assert tuple(row["rank"] for row in validated) == (0, 1, 2, 3)
    duplicate = [dict(row) for row in rows]
    duplicate[3]["gpu_uuid"] = duplicate[0]["gpu_uuid"]
    _expect_value_error(
        lambda: preflight.validate_rank_evidence(duplicate),
        "GPU UUIDs",
    )
    dirty = [dict(row) for row in rows]
    dirty[2]["process_group_destroyed"] = False
    _expect_value_error(
        lambda: preflight.validate_rank_evidence(dirty),
        "destroyed",
    )
    for field, fragment in (
        ("candidate_reference_dropped", "candidate reference"),
        ("model_reference_dropped", "model reference"),
        ("cuda_synchronized", "CUDA synchronization"),
        ("cuda_cache_emptied", "CUDA cache"),
    ):
        incomplete = copy.deepcopy(rows)
        incomplete[1][field] = False
        _expect_value_error(
            lambda: preflight.validate_rank_evidence(incomplete),
            fragment,
        )


def test_launched_process_rows_bind_exact_persisted_rank_evidence():
    persisted = list(_artifact_inputs()[4])
    launched = tuple({
        "rank": row["rank"],
        "world_size": 4,
        "pid": row["pid"],
        "exit_code": 0,
        "gpu_index": row["gpu_index"],
        "gpu_uuid": row["gpu_uuid"],
        "process_group_nonce": row["process_group_nonce"],
        "rendezvous": row["rendezvous"],
    } for row in persisted)

    bound = preflight.bind_launched_rank_evidence(
        launched,
        persisted,
    )

    assert tuple(row["rank"] for row in bound) == (0, 1, 2, 3)
    assert all(row["worker_exited"] is True for row in bound)
    mismatch = copy.deepcopy(persisted)
    mismatch[2]["pid"] += 100
    _expect_value_error(
        lambda: preflight.bind_launched_rank_evidence(
            launched,
            mismatch,
        ),
        "launch identity",
    )


def test_worker_gpu_identity_requires_visible_rank_and_matching_uuid():
    rows = (
        {
            "gpu_index": 12,
            "gpu_uuid": "GPU-physical",
            "free_bytes": 60 * 1024**3,
            "compute_processes": [],
        },
    )
    validated = preflight.validate_native_worker_gpu_identity(
        rank=2,
        expected_gpu_index=12,
        expected_gpu_uuid="GPU-physical",
        visible_devices="10,11,12,13",
        query_gpus=lambda: rows,
    )
    assert validated["local_rank"] == 2
    assert validated["physical_gpu_index"] == 12
    _expect_value_error(
        lambda: preflight.validate_native_worker_gpu_identity(
            rank=2,
            expected_gpu_index=12,
            expected_gpu_uuid="GPU-wrong",
            visible_devices="10,11,12,13",
            query_gpus=lambda: rows,
        ),
        "UUID",
    )
    _expect_value_error(
        lambda: preflight.validate_native_worker_gpu_identity(
            rank=2,
            expected_gpu_index=12,
            expected_gpu_uuid="GPU-physical",
            visible_devices="10,11,12,13",
            query_gpus=lambda: ({
                **rows[0],
                "compute_processes": [{
                    "pid": 91,
                    "process_name": "new-owner",
                    "used_bytes": 1024,
                }],
            },),
        ),
        "active compute process",
    )


def test_tp4_native_cases_preserve_root_and_non_root_output_contract():
    prompts = (
        _prompt("a", (3, 5)),
        _prompt("b", (7, 11, 13)),
    )
    for rank in range(4):
        pool = _FakePool()
        candidate = _fake_tp4_candidate(rank=rank, pool=pool)
        allocator = _FakeAllocator()
        context_events = []
        barriers = []

        results = preflight.run_tp4_native_cases(
            candidate=candidate,
            rank=rank,
            prompt_cases=prompts,
            expected_model_fingerprint=(
                preflight.APPROVED_MODEL_MANIFEST_SHA256
            ),
            device="cpu",
            allocator=allocator,
            first_request_id=100,
            set_context=lambda **kwargs: context_events.append(
                ("set", kwargs)
            ),
            reset_context=lambda: context_events.append(("reset",)),
            barrier=lambda case_id: barriers.append(case_id),
        )

        assert tuple(result.case_id for result in results) == ("a", "b")
        assert tuple(result.request_id for result in results) == (100, 101)
        assert all(result.rank == rank for result in results)
        assert all(result.release_zeroed for result in results)
        assert all(result.pool_binding_released for result in results)
        assert all(
            result.state_nonzero_after_commit == {
                "0:linear_convolution": True,
                "0:linear_recurrent": True,
            }
            for result in results
        )
        if rank == 0:
            assert all(
                result.logits is not None
                and result.logits.shape
                == (preflight.MODEL_VOCAB_SIZE,)
                and result.logits.dtype == torch.float32
                and result.logits.is_contiguous()
                for result in results
            )
        else:
            assert all(result.logits is None for result in results)
        assert pool.events == [
            ("activate", 1),
            ("release", 1),
            ("activate", 2),
            ("release", 2),
        ]
        assert barriers == ["a", "b"]
        assert tuple(event[0] for event in context_events) == (
            "set",
            "reset",
            "set",
            "reset",
        )


def test_tp4_native_case_rejects_root_none_or_non_root_tensor_and_releases():
    cases = (
        (0, "root_none", "rank zero"),
        (2, "non_root_tensor", "non-root"),
    )
    for rank, invalid_output, message in cases:
        pool = _FakePool()
        candidate = _fake_tp4_candidate(
            rank=rank,
            pool=pool,
            invalid_output=invalid_output,
        )
        allocator = _FakeAllocator()

        _expect_value_error(
            lambda: preflight.run_tp4_native_cases(
                candidate=candidate,
                rank=rank,
                prompt_cases=(_prompt("bad", (2, 3)),),
                expected_model_fingerprint=(
                    preflight.APPROVED_MODEL_MANIFEST_SHA256
                ),
                device="cpu",
                allocator=allocator,
                first_request_id=1,
                set_context=lambda **_kwargs: None,
                reset_context=lambda: None,
                barrier=lambda _case_id: None,
            ),
            message,
        )
        assert pool._bindings == {}
        assert all(
            not bool(torch.count_nonzero(tensor))
            for tensor in pool._tensors.values()
        )
        assert allocator.events == [
            ("allocate", 1, 1),
            ("release", 1, 1),
        ]


class _FakeDistributed:
    def __init__(self):
        self.calls = []
        self.initialized = False

    def init_process_group(self, **kwargs):
        self.calls.append(("init_process_group", kwargs))
        self.initialized = True

    def barrier(self):
        self.calls.append(("barrier",))

    def all_reduce(self, tensor, *args, **kwargs):
        self.calls.append(("all_reduce", tensor, args, kwargs))
        return "all-reduce-result"

    def gather(self, tensor, gather_list=None, dst=0, *args, **kwargs):
        self.calls.append((
            "gather",
            tensor,
            gather_list,
            dst,
            args,
            kwargs,
        ))
        return "gather-result"

    def is_initialized(self):
        return self.initialized

    def destroy_process_group(self):
        self.calls.append(("destroy_process_group",))
        self.initialized = False


class _FakeCuda:
    def __init__(self):
        self.calls = []

    def set_device(self, rank):
        self.calls.append(("set_device", rank))

    def synchronize(self):
        self.calls.append(("synchronize",))

    def empty_cache(self):
        self.calls.append(("empty_cache",))


def test_native_rank_scope_initializes_barriers_and_destroys_group():
    distributed = _FakeDistributed()
    cuda = _FakeCuda()
    candidate = object()
    events = []

    def run_cases(**kwargs):
        events.append(("run", kwargs))
        kwargs["barrier"]("p17")
        kwargs["barrier"]("p65")
        return (
            SimpleNamespace(case_id="p17"),
            SimpleNamespace(case_id="p65"),
        )

    row = preflight.execute_native_rank_scope(
        rank=3,
        world_size=4,
        rendezvous="tcp://127.0.0.1:41001",
        process_group_nonce="n" * 32,
        prompt_cases=(_prompt("p17", (1, 2)), _prompt("p65", (3, 4))),
        build_candidate=lambda rank: events.append(
            ("build", rank)
        ) or SimpleNamespace(candidate=candidate),
        move_candidate=lambda value, **kwargs: events.append(
            ("move", value, kwargs)
        ) or value,
        run_cases=run_cases,
        allocator_factory=lambda capacity: events.append(
            ("allocator", capacity)
        ) or object(),
        set_context=lambda **_kwargs: None,
        reset_context=lambda: None,
        distributed=distributed,
        cuda=cuda,
    )

    assert cuda.calls == [
        ("set_device", 3),
        ("synchronize",),
        ("empty_cache",),
    ]
    assert distributed.calls == [
        (
            "init_process_group",
            {
                "backend": "nccl",
                "init_method": "tcp://127.0.0.1:41001",
                "world_size": 4,
                "rank": 3,
            },
        ),
        ("barrier",),
        ("barrier",),
        ("barrier",),
        ("destroy_process_group",),
    ]
    assert row["rank"] == 3
    assert row["world_size"] == 4
    assert row["process_group_nonce"] == "n" * 32
    assert row["case_barrier_count"] == 2
    assert row["final_barrier_completed"] is True
    assert row["process_group_destroyed"] is True
    assert row["candidate_reference_dropped"] is True
    assert row["model_reference_dropped"] is True
    assert row["cuda_synchronized"] is True
    assert row["cuda_cache_emptied"] is True


def test_native_rank_scope_destroys_group_after_run_failure():
    distributed = _FakeDistributed()
    cuda = _FakeCuda()

    def fail_run(**_kwargs):
        raise RuntimeError("injected rank failure")

    try:
        preflight.execute_native_rank_scope(
            rank=1,
            world_size=4,
            rendezvous="tcp://127.0.0.1:41002",
            process_group_nonce="m" * 32,
            prompt_cases=(_prompt("p17", (1, 2)),),
            build_candidate=lambda _rank: SimpleNamespace(
                candidate=object()
            ),
            move_candidate=lambda value, **_kwargs: value,
            run_cases=fail_run,
            allocator_factory=lambda _capacity: object(),
            set_context=lambda **_kwargs: None,
            reset_context=lambda: None,
            distributed=distributed,
            cuda=cuda,
        )
    except RuntimeError as error:
        assert "injected rank failure" in str(error)
    else:
        raise AssertionError("expected injected rank failure")

    assert distributed.calls[-1] == ("destroy_process_group",)
    assert distributed.calls.count(("barrier",)) == 0
    assert cuda.calls[-2:] == [
        ("synchronize",),
        ("empty_cache",),
    ]


def test_collective_recorder_delegates_once_records_order_and_restores():
    distributed = _FakeDistributed()
    tensor = torch.zeros(2, 3)
    gather_list = [torch.empty_like(tensor) for _ in range(4)]
    original_all_reduce = distributed.all_reduce
    original_gather = distributed.gather

    with preflight.record_distributed_collectives(distributed) as events:
        assert distributed.all_reduce(tensor, op="sum") == "all-reduce-result"
        assert (
            distributed.gather(tensor, gather_list, 0)
            == "gather-result"
        )

    assert len([
        call for call in distributed.calls if call[0] == "all_reduce"
    ]) == 1
    assert len([
        call for call in distributed.calls if call[0] == "gather"
    ]) == 1
    assert [event["collective"] for event in events] == [
        "all_reduce",
        "gather",
    ]
    assert events[0]["shape"] == [2, 3]
    assert events[0]["dtype"] == "torch.float32"
    assert events[1]["destination"] == 0
    assert events[1]["receive_count"] == 4
    assert distributed.all_reduce == original_all_reduce
    assert distributed.gather == original_gather


class _FakeRankProcess:
    def __init__(
        self,
        *,
        rank,
        events,
        pid,
        exit_code=0,
        alive_after_join=False,
    ):
        self.rank = rank
        self.events = events
        self.pid = pid
        self.exitcode = None
        self._configured_exit_code = exit_code
        self._alive_after_join = alive_after_join

    def start(self):
        self.events.append(("start", self.rank))

    def join(self, timeout):
        self.events.append(("join", self.rank, timeout))
        if not self._alive_after_join:
            self.exitcode = self._configured_exit_code

    def is_alive(self):
        return self._alive_after_join

    def terminate(self):
        self.events.append(("terminate", self.rank))
        self._alive_after_join = False
        self.exitcode = -15


def _selected_gpus():
    return tuple(
        {
            "rank": rank,
            "world_size": 4,
            "gpu_index": 10 + rank,
            "gpu_uuid": f"GPU-{rank}",
            "free_bytes": 40 * 1024**3,
            "compute_processes": [],
            "minimum_free_bytes": 24 * 1024**3,
        }
        for rank in range(4)
    )


def test_launch_native_group_starts_all_ranks_before_join_and_binds_environment():
    events = []
    invocations = []

    def process_factory(**kwargs):
        invocations.append(kwargs)
        return _FakeRankProcess(
            rank=kwargs["rank"],
            events=events,
            pid=200 + kwargs["rank"],
        )

    rows = preflight.launch_native_rank_group(
        selected_gpus=_selected_gpus(),
        rendezvous="tcp://127.0.0.1:42001",
        process_group_nonce="p" * 32,
        tinyvllm_dist_port=42002,
        master_port=42003,
        process_factory=process_factory,
        timeout_seconds=90,
        pid_alive=lambda _pid: False,
        base_environment={"PYTHONPATH": "/source"},
    )

    assert events[:4] == [
        ("start", 0),
        ("start", 1),
        ("start", 2),
        ("start", 3),
    ]
    assert [event[0] for event in events[4:]] == ["join"] * 4
    assert tuple(row["rank"] for row in rows) == (0, 1, 2, 3)
    for rank, invocation in enumerate(invocations):
        assert invocation["rank"] == rank
        assert invocation["world_size"] == 4
        assert invocation["gpu_index"] == 10 + rank
        assert invocation["gpu_uuid"] == f"GPU-{rank}"
        assert invocation["rendezvous"] == "tcp://127.0.0.1:42001"
        assert invocation["process_group_nonce"] == "p" * 32
        environment = invocation["environment"]
        assert environment["CUDA_VISIBLE_DEVICES"] == "10,11,12,13"
        assert environment["TINYVLLM_DIST_PORT"] == "42002"
        assert environment["MASTER_PORT"] == "42003"
        assert environment["TINYVLLM_GATE_LOCAL_RANK"] == str(rank)
        assert environment["TINYVLLM_GATE_GPU_UUID"] == f"GPU-{rank}"


def test_launch_native_group_rejects_port_collision_timeout_and_surviving_pid():
    scenarios = (
        ("ports", 42002, 42002, False, False, "ports must be distinct"),
        ("timeout", 42002, 42003, True, False, "timed out"),
        ("pid", 42002, 42003, False, True, "PID survived"),
    )
    for _, dist_port, master_port, alive, pid_survives, message in scenarios:
        events = []

        def process_factory(**kwargs):
            return _FakeRankProcess(
                rank=kwargs["rank"],
                events=events,
                pid=300 + kwargs["rank"],
                alive_after_join=alive and kwargs["rank"] == 2,
            )

        _expect_value_error(
            lambda: preflight.launch_native_rank_group(
                selected_gpus=_selected_gpus(),
                rendezvous="tcp://127.0.0.1:42001",
                process_group_nonce="q" * 32,
                tinyvllm_dist_port=dist_port,
                master_port=master_port,
                process_factory=process_factory,
                timeout_seconds=90,
                pid_alive=(
                    (lambda pid: pid_survives and pid == 301)
                ),
                base_environment={},
            ),
            message,
        )


def test_launch_native_group_reaps_partially_started_workers_after_start_failure():
    events = []
    processes = {}

    class StartFailureProcess:
        def __init__(self, rank):
            self.rank = rank
            self.pid = 500 + rank
            self.exitcode = None
            self.alive = False
            self.terminated = False
            self.killed = False

        def start(self):
            events.append(("start", self.rank))
            self.alive = True
            if self.rank == 2:
                raise RuntimeError("rank start failed")

        def join(self, timeout):
            events.append(("join", self.rank, timeout))
            if self.killed or (self.terminated and self.rank != 1):
                self.alive = False
                self.exitcode = -9 if self.killed else -15

        def is_alive(self):
            return self.alive

        def terminate(self):
            events.append(("terminate", self.rank))
            self.terminated = True

        def kill(self):
            events.append(("kill", self.rank))
            self.killed = True

    def process_factory(**kwargs):
        process = StartFailureProcess(kwargs["rank"])
        processes[kwargs["rank"]] = process
        return process

    try:
        preflight.launch_native_rank_group(
            selected_gpus=_selected_gpus(),
            rendezvous="tcp://127.0.0.1:42001",
            process_group_nonce="r" * 32,
            tinyvllm_dist_port=42002,
            master_port=42003,
            process_factory=process_factory,
            timeout_seconds=90,
            pid_alive=lambda pid: any(
                process.pid == pid and process.alive
                for process in processes.values()
            ),
            base_environment={},
        )
    except RuntimeError as error:
        assert str(error) == "rank start failed"
    else:
        raise AssertionError("expected rank start failure")

    assert set(processes) == {0, 1, 2, 3}
    assert all(not process.alive for process in processes.values())
    assert ("terminate", 2) in events
    assert ("join", 2, 30) in events
    assert ("kill", 1) in events
    assert events.count(("join", 1, 30)) == 2


def test_reference_must_exit_before_native_and_all_native_pids_before_publication():
    events = []

    def reference_worker():
        events.append("reference")
        return {"pid": 101, "exit_code": 0}

    def native_launcher():
        events.append("native")
        return tuple(
            {"rank": rank, "pid": 201 + rank, "exit_code": 0}
            for rank in range(4)
        )

    rows = preflight.execute_reference_then_native_group(
        reference_worker=reference_worker,
        native_launcher=native_launcher,
        pid_alive=lambda _pid: False,
    )
    assert events == ["reference", "native"]
    assert tuple(row["rank"] for row in rows) == (0, 1, 2, 3)

    events.clear()
    _expect_value_error(
        lambda: preflight.execute_reference_then_native_group(
            reference_worker=reference_worker,
            native_launcher=native_launcher,
            pid_alive=lambda pid: pid == 101,
        ),
        "reference worker PID",
    )
    assert events == ["reference"]

    events.clear()
    _expect_value_error(
        lambda: preflight.execute_reference_then_native_group(
            reference_worker=reference_worker,
            native_launcher=native_launcher,
            pid_alive=lambda pid: pid == 203,
        ),
        "native worker PID",
    )
    assert events == ["reference", "native"]


def test_source_bound_run_uses_dist_port_for_rendezvous_and_environment():
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        source_root = root / "source"
        source_manifest_path = root / "source_manifest.json"
        source_root.mkdir()
        source_hashes = {"gate.py": "0" * 64}
        source_manifest = {
            "schema_version": 1,
            "source_file_sha256": source_hashes,
            "source_tree_sha256": hashlib.sha256(
                json.dumps(
                    source_hashes,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "model_manifest_sha256": (
                preflight.APPROVED_MODEL_MANIFEST_SHA256
            ),
            "config_sha256": preflight.APPROVED_CONFIG_SHA256,
            "index_sha256": preflight.APPROVED_INDEX_SHA256,
            "shard_name": preflight.APPROVED_SHARD_NAME,
            "shard_size": preflight.APPROVED_SHARD_SIZE,
            "shard_sha256": preflight.APPROVED_SHARD_SHA256,
            "prerequisites": {
                "tp1_real_root_logit_correctness": {
                    "run_tag": (
                        "qwen35-tp1-authority-20260728-195153-r2"
                    ),
                    "classification": "PASS",
                    "source_tree_sha256": (
                        "e5da50970951b32a61ecc9f85cf1cd447dc95950856345d780cd9e089bdf11ab"
                    ),
                    "artifacts": {
                        "tp1_real_root_logit_correctness.json": (
                            "39c4bbc548a82e915609dccb57101a6e64cbc92319d5ba69c44d427f8f4aa519"
                        ),
                        "reference_logits.pt": (
                            "3373ab6038d6f21cf6421aa0e1c9146cc001e6e8bcbd06e6ab59c65ecc709e5a"
                        ),
                        "native_logits.pt": (
                            "5d3fcb3a204a1b1bd673026d83f06ddc76422ca49d51d6af32671ba153ecb4d4"
                        ),
                        "source_manifest.json": (
                            "0633a6ad5913d0d8a28526c1ec05f2cb17e347c180a6c93fa58fc3674fcb2207"
                        ),
                    },
                },
            },
        }
        source_manifest_path.write_bytes(
            preflight._canonical_json_bytes(
                source_manifest
            )
            + b"\n"
        )
        selected = _selected_gpus()
        observed = {}

        def command_runner(command, **_kwargs):
            tensor_output = Path(
                command[command.index("--tensor-output") + 1]
            )
            process_output = Path(
                command[command.index("--process-output") + 1]
            )
            torch.save(_artifact_inputs()[0], tensor_output)
            process_output.write_bytes(
                preflight._canonical_json_bytes({
                    "pid": 111,
                    "exit_code": 0,
                })
                + b"\n"
            )
            return SimpleNamespace(returncode=0, stderr="")

        def process_factory_builder(**_kwargs):
            def process_factory(**kwargs):
                observed.setdefault("invocations", []).append(kwargs)
                return _FakeRankProcess(
                    rank=kwargs["rank"],
                    events=[],
                    pid=200 + kwargs["rank"],
                )

            return process_factory

        original_ports = preflight.fresh_port_pair
        original_launcher = preflight.launch_native_rank_group
        original_finalize = preflight.finalize_tp4_correctness_artifact
        original_loader = preflight._load_json
        try:
            preflight.fresh_port_pair = lambda: (46001, 46002)

            def launch_native_rank_group(**kwargs):
                observed["launch"] = kwargs
                raise RuntimeError("stop after port capture")

            preflight.launch_native_rank_group = launch_native_rank_group
            preflight.finalize_tp4_correctness_artifact = (
                lambda **_kwargs: ()
            )
            preflight._load_json = (
                lambda _path, *, label: (
                    {"pid": 111, "exit_code": 0}
                    if label == "reference"
                    else {}
                )
            )
            try:
                preflight.execute_source_bound_run(
                    run_dir=root / "run",
                    run_tag="port-binding-test",
                    source_manifest_path=source_manifest_path,
                    source_root=source_root,
                    query_gpus=lambda: selected,
                    command_runner=command_runner,
                    process_factory_builder=process_factory_builder,
                    pid_alive=lambda _pid: False,
                )
            except RuntimeError as error:
                assert str(error) == "stop after port capture"
            else:
                raise AssertionError("expected port capture stop")
        finally:
            preflight.fresh_port_pair = original_ports
            preflight.launch_native_rank_group = original_launcher
            preflight.finalize_tp4_correctness_artifact = original_finalize
            preflight._load_json = original_loader

        launch = observed["launch"]
        assert launch["rendezvous"] == "tcp://127.0.0.1:46001"
        assert launch["tinyvllm_dist_port"] == 46001
        assert launch["master_port"] == 46002


def test_native_rank_worker_atomically_writes_rank_row_and_root_logits_only():
    cases = (
        _prompt("p17", (1, 2)),
        _prompt("p65", (3, 4)),
        _prompt("synthetic", (5, 6)),
    )

    def scope_runner(**kwargs):
        rank = kwargs["rank"]
        return {
            "rank": rank,
            "world_size": 4,
            "process_group_nonce": kwargs["process_group_nonce"],
            "rendezvous": kwargs["rendezvous"],
            "case_barrier_count": 3,
            "final_barrier_completed": True,
            "process_group_destroyed": True,
            "candidate_reference_dropped": True,
            "model_reference_dropped": True,
            "cuda_synchronized": True,
            "cuda_cache_emptied": True,
            "results": tuple(
                SimpleNamespace(
                    case_id=case.case_id,
                    rank=rank,
                    token_count=len(case.token_ids),
                    logits=(
                        torch.arange(
                            preflight.MODEL_VOCAB_SIZE,
                            dtype=torch.float32,
                        ).add(index)
                        if rank == 0
                        else None
                    ),
                    state_nonzero_after_commit={"0:linear_recurrent": True},
                    release_zeroed=True,
                    pool_binding_released=True,
                )
                for index, case in enumerate(cases)
            ),
        }

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        for rank in range(4):
            row_path = root / f"rank-{rank}.json.partial"
            logits_path = (
                root / "rank0-logits.pt.partial"
                if rank == 0
                else None
            )
            row = preflight.execute_native_rank_worker(
                rank=rank,
                world_size=4,
                rendezvous="tcp://127.0.0.1:43001",
                process_group_nonce="r" * 32,
                prompt_cases=cases,
                gpu_index=20 + rank,
                gpu_uuid=f"GPU-{rank}",
                rank_output=row_path,
                logits_output=logits_path,
                process_id=500 + rank,
                scope_runner=scope_runner,
                allocator_factory=lambda _capacity: object(),
            )
            assert row_path.is_file()
            assert json.loads(row_path.read_text()) == row
            assert row["root_logits_present"] is (rank == 0)
            assert row["non_root_logits_none"] is (rank != 0)
            assert row["candidate_reference_dropped"] is True
            assert row["model_reference_dropped"] is True
            assert row["cuda_synchronized"] is True
            assert row["cuda_cache_emptied"] is True
            if rank == 0:
                payload = torch.load(
                    logits_path,
                    map_location="cpu",
                    weights_only=True,
                )
                assert tuple(payload) == ("p17", "p65", "synthetic")
                assert all(
                    tensor.shape == (preflight.MODEL_VOCAB_SIZE,)
                    for tensor in payload.values()
                )


def test_native_rank_worker_rejects_non_root_logits_and_leaves_no_partial():
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        row_path = root / "rank-2.json.partial"

        def invalid_scope(**kwargs):
            return {
                "rank": kwargs["rank"],
                "world_size": 4,
                "process_group_nonce": kwargs["process_group_nonce"],
                "rendezvous": kwargs["rendezvous"],
                "case_barrier_count": 1,
                "final_barrier_completed": True,
                "process_group_destroyed": True,
                "results": (
                    SimpleNamespace(
                        case_id="p17",
                        rank=2,
                        token_count=2,
                        logits=torch.zeros(
                            preflight.MODEL_VOCAB_SIZE,
                            dtype=torch.float32,
                        ),
                        state_nonzero_after_commit={"state": True},
                        release_zeroed=True,
                        pool_binding_released=True,
                    ),
                ),
            }

        _expect_value_error(
            lambda: preflight.execute_native_rank_worker(
                rank=2,
                world_size=4,
                rendezvous="tcp://127.0.0.1:43002",
                process_group_nonce="s" * 32,
                prompt_cases=(_prompt("p17", (1, 2)),),
                gpu_index=22,
                gpu_uuid="GPU-2",
                rank_output=row_path,
                logits_output=None,
                process_id=502,
                scope_runner=invalid_scope,
                allocator_factory=lambda _capacity: object(),
            ),
            "non-root",
        )
        assert not row_path.exists()


def _artifact_inputs(*, native_offset=0.0):
    cases = preflight._TP4_CONTRACT.prompt_cases()
    reference = {}
    native = {}
    for index, case in enumerate(cases):
        row = torch.arange(
            preflight.MODEL_VOCAB_SIZE,
            dtype=torch.float32,
        ).mul_(1e-5)
        row[-1] = 100.0 + index
        row[-2] = 90.0 + index
        reference[case.case_id] = row.contiguous()
        candidate = row.clone()
        candidate.add_(native_offset)
        native[case.case_id] = candidate.contiguous()
    rank_rows = []
    for rank in range(4):
        rank_rows.append({
            "rank": rank,
            "world_size": 4,
            "pid": 700 + rank,
            "exit_code": 0,
            "gpu_index": 30 + rank,
            "gpu_uuid": f"GPU-{rank}",
            "process_group_nonce": "t" * 32,
            "rendezvous": "tcp://127.0.0.1:44001",
            "case_ids": [case.case_id for case in cases],
            "case_barrier_count": 3,
            "final_barrier_completed": True,
            "process_group_destroyed": True,
            "candidate_reference_dropped": True,
            "model_reference_dropped": True,
            "cuda_synchronized": True,
            "cuda_cache_emptied": True,
            "root_logits_present": rank == 0,
            "non_root_logits_none": rank != 0,
            "global_query_heads": 8,
            "global_kv_heads": 2,
            "local_query_heads": 2,
            "local_kv_heads": 1,
            "kv_head_replicas": 2,
            "source_kv_rank": rank // 2,
            "collective_events": [
                event
                for case_index in range(3)
                for event in (
                    {
                        "ordinal": case_index * 2,
                        "collective": "all_reduce",
                        "shape": [17, 2048],
                        "dtype": "torch.bfloat16",
                        "async_op": False,
                    },
                    {
                        "ordinal": case_index * 2 + 1,
                        "collective": "gather",
                        "shape": [1, 62080],
                        "dtype": "torch.bfloat16",
                        "destination": 0,
                        "receive_count": 4 if rank == 0 else None,
                        "async_op": False,
                    },
                )
            ],
            "state_rows": [
                {
                    "case_id": case.case_id,
                    "changed_component_count": 36,
                    "state_nonzero_after_commit": {
                        **{
                            f"{layer}:linear_convolution": True
                            for layer in range(18)
                        },
                        **{
                            f"{layer}:linear_recurrent": True
                            for layer in range(18)
                        },
                    },
                    "release_zeroed": True,
                    "pool_binding_released": True,
                }
                for case in cases
            ],
        })
    source_manifest = {
        "schema_version": 1,
        "source_file_sha256": {
            "tools/qwen35_tp4_real_root_logit_correctness_contract.py": (
                "a" * 64
            ),
        },
        "source_tree_sha256": hashlib.sha256(
            json.dumps(
                {
                    "tools/qwen35_tp4_real_root_logit_correctness_contract.py": (
                        "a" * 64
                    ),
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "model_manifest_sha256": (
            preflight.APPROVED_MODEL_MANIFEST_SHA256
        ),
        "config_sha256": preflight.APPROVED_CONFIG_SHA256,
        "index_sha256": preflight.APPROVED_INDEX_SHA256,
        "shard_name": preflight.APPROVED_SHARD_NAME,
        "shard_size": preflight.APPROVED_SHARD_SIZE,
        "shard_sha256": preflight.APPROVED_SHARD_SHA256,
        "prerequisites": {
            "tp1_real_root_logit_correctness": {
                "run_tag": "qwen35-tp1-authority-20260728-195153-r2",
                "classification": "PASS",
                "source_tree_sha256": (
                    "e5da50970951b32a61ecc9f85cf1cd447dc95950856345d780cd9e089bdf11ab"
                ),
                "artifacts": {
                    "tp1_real_root_logit_correctness.json": (
                        "39c4bbc548a82e915609dccb57101a6e64cbc92319d5ba69c44d427f8f4aa519"
                    ),
                    "reference_logits.pt": (
                        "3373ab6038d6f21cf6421aa0e1c9146cc001e6e8bcbd06e6ab59c65ecc709e5a"
                    ),
                    "native_logits.pt": (
                        "5d3fcb3a204a1b1bd673026d83f06ddc76422ca49d51d6af32671ba153ecb4d4"
                    ),
                    "source_manifest.json": (
                        "0633a6ad5913d0d8a28526c1ec05f2cb17e347c180a6c93fa58fc3674fcb2207"
                    ),
                },
            },
        },
    }
    reference_process = {
        "worker": "reference",
        "pid": 650,
        "exit_code": 0,
        "gpu_index": 30,
        "gpu_uuid": "GPU-reference",
        "case_ids": [case.case_id for case in cases],
        "vocab_size": preflight.MODEL_VOCAB_SIZE,
        "cleanup_complete": True,
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "attn_implementation": "eager",
        "use_cache": False,
    }
    return (
        cases,
        reference,
        native,
        reference_process,
        rank_rows,
        source_manifest,
    )


def test_finalizer_publishes_exact_five_pass_artifacts():
    (
        _,
        reference,
        native,
        reference_process,
        rank_rows,
        source_manifest,
    ) = _artifact_inputs()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = Path(temporary_directory) / "authority"
        paths = preflight.finalize_tp4_correctness_artifact(
            run_dir=run_dir,
            run_tag="qwen35-tp4-test",
            reference_logits=reference,
            native_rank0_logits=native,
            reference_process=reference_process,
            rank_rows=rank_rows,
            source_manifest=source_manifest,
            forbidden_counters={
                "engine": 0,
                "model_runner": 0,
                "scheduler": 0,
                "sampler": 0,
                "generation": 0,
            },
        )
        assert tuple(path.name for path in paths) == (
            "tp4_real_root_logit_correctness.json",
            "reference_logits.pt",
            "native_rank0_logits.pt",
            "rank_evidence.json",
            "source_manifest.json",
        )
        assert {path.name for path in run_dir.iterdir()} == {
            path.name for path in paths
        }
        result = json.loads(
            (run_dir / "tp4_real_root_logit_correctness.json").read_text()
        )
        assert result["classification"] == "PASS"
        assert len(result["comparisons"]) == 3
        assert result["reference_process"]["pid"] == 650
        evidence = json.loads(
            (run_dir / "rank_evidence.json").read_text()
        )
        assert tuple(row["rank"] for row in evidence) == (0, 1, 2, 3)
        manifest = json.loads(
            (run_dir / "source_manifest.json").read_text()
        )
        assert set(manifest["artifacts"]) == {
            "tp4_real_root_logit_correctness.json",
            "reference_logits.pt",
            "native_rank0_logits.pt",
            "rank_evidence.json",
        }


def test_finalizer_rejects_bad_evidence_counters_and_non_pass_without_output():
    (
        _,
        reference,
        native,
        reference_process,
        rank_rows,
        source_manifest,
    ) = _artifact_inputs()
    scenarios = []
    duplicate = [dict(row) for row in rank_rows]
    duplicate[3]["rank"] = 2
    scenarios.append(("ranks", duplicate, native, {
        "engine": 0,
        "model_runner": 0,
        "scheduler": 0,
        "sampler": 0,
        "generation": 0,
    }, "ranks"))
    non_root = [dict(row) for row in rank_rows]
    non_root[2]["non_root_logits_none"] = False
    scenarios.append(("non-root", non_root, native, {
        "engine": 0,
        "model_runner": 0,
        "scheduler": 0,
        "sampler": 0,
        "generation": 0,
    }, "non-root"))
    cleanup = [dict(row) for row in rank_rows]
    cleanup[1]["process_group_destroyed"] = False
    scenarios.append(("cleanup", cleanup, native, {
        "engine": 0,
        "model_runner": 0,
        "scheduler": 0,
        "sampler": 0,
        "generation": 0,
    }, "destroyed"))
    scenarios.append(("counter", rank_rows, native, {
        "engine": 1,
        "model_runner": 0,
        "scheduler": 0,
        "sampler": 0,
        "generation": 0,
    }, "forbidden"))
    topology = [copy.deepcopy(row) for row in rank_rows]
    topology[1]["local_query_heads"] = 1
    scenarios.append(("topology", topology, native, {
        "engine": 0,
        "model_runner": 0,
        "scheduler": 0,
        "sampler": 0,
        "generation": 0,
    }, "local_query_heads"))
    collectives = [copy.deepcopy(row) for row in rank_rows]
    collectives[3]["collective_events"] = collectives[3][
        "collective_events"
    ][:-1]
    scenarios.append(("collective", collectives, native, {
        "engine": 0,
        "model_runner": 0,
        "scheduler": 0,
        "sampler": 0,
        "generation": 0,
    }, "gather"))
    states = [copy.deepcopy(row) for row in rank_rows]
    states[2]["state_rows"][0]["changed_component_count"] = 35
    scenarios.append(("state", states, native, {
        "engine": 0,
        "model_runner": 0,
        "scheduler": 0,
        "sampler": 0,
        "generation": 0,
    }, "state"))
    bad_source = copy.deepcopy(source_manifest)
    bad_source["prerequisites"]["tp1_real_root_logit_correctness"][
        "classification"
    ] = "NO_GO_LOGIT"
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = Path(temporary_directory) / "prerequisite"
        _expect_value_error(
            lambda: preflight.finalize_tp4_correctness_artifact(
                run_dir=run_dir,
                run_tag="qwen35-tp4-prerequisite",
                reference_logits=reference,
                native_rank0_logits=native,
                reference_process=reference_process,
                rank_rows=rank_rows,
                source_manifest=bad_source,
                forbidden_counters={
                    "engine": 0,
                    "model_runner": 0,
                    "scheduler": 0,
                    "sampler": 0,
                    "generation": 0,
                },
            ),
            "prerequisite",
        )
    bad_tree = copy.deepcopy(source_manifest)
    bad_tree["source_tree_sha256"] = "f" * 64
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = Path(temporary_directory) / "source-tree"
        _expect_value_error(
            lambda: preflight.finalize_tp4_correctness_artifact(
                run_dir=run_dir,
                run_tag="qwen35-tp4-source-tree",
                reference_logits=reference,
                native_rank0_logits=native,
                reference_process=reference_process,
                rank_rows=rank_rows,
                source_manifest=bad_tree,
                forbidden_counters={
                    "engine": 0,
                    "model_runner": 0,
                    "scheduler": 0,
                    "sampler": 0,
                    "generation": 0,
                },
            ),
            "tree hash",
        )
    no_go = {name: value.clone() for name, value in native.items()}
    no_go[next(iter(no_go))][-1] = -100.0
    scenarios.append(("no-go", rank_rows, no_go, {
        "engine": 0,
        "model_runner": 0,
        "scheduler": 0,
        "sampler": 0,
        "generation": 0,
    }, "PASS"))

    for label, rows, native_rows, counters, message in scenarios:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory) / label
            _expect_value_error(
                lambda: preflight.finalize_tp4_correctness_artifact(
                    run_dir=run_dir,
                    run_tag=f"qwen35-tp4-{label}",
                    reference_logits=reference,
                    native_rank0_logits=native_rows,
                    reference_process=reference_process,
                    rank_rows=rows,
                    source_manifest=source_manifest,
                    forbidden_counters=counters,
                ),
                message,
            )
            assert not run_dir.exists() or not any(run_dir.iterdir())


def test_finalizer_rejects_nonempty_directory_and_rolls_back_replace_failure():
    (
        _,
        reference,
        native,
        reference_process,
        rank_rows,
        source_manifest,
    ) = _artifact_inputs()
    counters = {
        "engine": 0,
        "model_runner": 0,
        "scheduler": 0,
        "sampler": 0,
        "generation": 0,
    }
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = Path(temporary_directory) / "nonempty"
        run_dir.mkdir()
        (run_dir / "foreign").write_text("x")
        _expect_value_error(
            lambda: preflight.finalize_tp4_correctness_artifact(
                run_dir=run_dir,
                run_tag="qwen35-tp4-nonempty",
                reference_logits=reference,
                native_rank0_logits=native,
                reference_process=reference_process,
                rank_rows=rank_rows,
                source_manifest=source_manifest,
                forbidden_counters=counters,
            ),
            "not empty",
        )

    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = Path(temporary_directory) / "rollback"
        replacements = 0

        def fail_second_replace(source, destination):
            nonlocal replacements
            replacements += 1
            if replacements == 2:
                raise OSError("injected replace failure")
            os.replace(source, destination)

        try:
            preflight.finalize_tp4_correctness_artifact(
                run_dir=run_dir,
                run_tag="qwen35-tp4-rollback",
                reference_logits=reference,
                native_rank0_logits=native,
                reference_process=reference_process,
                rank_rows=rank_rows,
                source_manifest=source_manifest,
                forbidden_counters=counters,
                replace=fail_second_replace,
            )
        except OSError as error:
            assert "injected replace failure" in str(error)
        else:
            raise AssertionError("expected injected replace failure")
        assert run_dir.is_dir()
        assert not any(run_dir.iterdir())


def test_cli_forwards_run_validate_reference_and_native_rank_contracts():
    calls = []
    cases = (_prompt("p17", (1, 2)),)

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        manifest = root / "source.json"
        manifest.write_text("{}")
        assert preflight.main(
            [
                "run",
                "--run-dir",
                os.fspath(root / "authority"),
                "--run-tag",
                "qwen35-tp4-cli",
                "--source-manifest",
                os.fspath(manifest),
            ],
            execute_run=lambda **kwargs: calls.append(("run", kwargs)),
            prompt_case_loader=lambda: cases,
            environment={},
        ) == 0
        assert preflight.main(
            ["validate", os.fspath(root / "authority")],
            execute_validate=lambda run_dir: calls.append(
                ("validate", Path(run_dir))
            ),
            prompt_case_loader=lambda: cases,
            environment={},
        ) == 0
        worker_environment = {
            "TINYVLLM_GATE_LOCAL_RANK": "2",
            "TINYVLLM_GATE_PHYSICAL_GPU_INDEX": "12",
            "TINYVLLM_GATE_GPU_UUID": "GPU-2",
            "TINYVLLM_GATE_PROCESS_GROUP_NONCE": "w" * 32,
            "TINYVLLM_GATE_RENDEZVOUS": "tcp://127.0.0.1:46001",
        }
        assert preflight.main(
            [
                "internal-native-rank",
                "--rank-output",
                os.fspath(root / "rank2.json.partial"),
            ],
            execute_native_rank=lambda **kwargs: calls.append(
                ("native", kwargs)
            ),
            prompt_case_loader=lambda: cases,
            environment=worker_environment,
        ) == 0

    assert calls[0] == (
        "run",
        {
            "run_dir": root / "authority",
            "run_tag": "qwen35-tp4-cli",
            "source_manifest_path": manifest,
        },
    )
    assert calls[1] == ("validate", root / "authority")
    native = calls[2][1]
    assert native["rank"] == 2
    assert native["world_size"] == 4
    assert native["gpu_index"] == 12
    assert native["gpu_uuid"] == "GPU-2"
    assert native["logits_output"] is None
    assert native["prompt_cases"] == cases


def test_subprocess_rank_factory_builds_deferred_native_rank_command():
    calls = []

    class FakePopen:
        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))
            self.pid = 1234
            self.returncode = None

        def wait(self, timeout):
            calls.append(("wait", timeout))
            self.returncode = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            calls.append(("terminate",))
            self.returncode = -15

    with tempfile.TemporaryDirectory() as temporary_directory:
        work_dir = Path(temporary_directory)
        process = preflight.make_native_rank_subprocess(
            rank=0,
            world_size=4,
            gpu_index=10,
            gpu_uuid="GPU-0",
            rendezvous="tcp://127.0.0.1:46001",
            process_group_nonce="x" * 32,
            environment={"CUDA_VISIBLE_DEVICES": "10,11,12,13"},
            script_path=Path("/source/tools/preflight.py"),
            python_executable="/env/bin/python",
            work_dir=work_dir,
            rank_output=work_dir / "rank0.json.partial",
            logits_output=work_dir / "rank0.pt.partial",
            popen=FakePopen,
        )
        assert calls == []
        process.start()
        command, kwargs = calls[0]
        assert command[:3] == (
            "/env/bin/python",
            "/source/tools/preflight.py",
            "internal-native-rank",
        )
        assert "--rank-output" in command
        assert "--logits-output" in command
        assert kwargs["cwd"] == work_dir
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "10,11,12,13"
        assert kwargs["stdout"].name.endswith("rank-0.stdout.log")
        assert kwargs["stderr"].name.endswith("rank-0.stderr.log")
        process.join(30)
        assert process.exitcode == 0
        assert process.pid == 1234
        assert process.is_alive() is False


def test_source_bound_run_orders_reference_group_and_finalizes_exact_five():
    (
        _cases,
        reference,
        native,
        reference_process,
        rank_rows,
        source_manifest,
    ) = _artifact_inputs()
    events = []
    required_sources = (
        "tools/qwen35_tp1_real_root_logit_correctness_contract.py",
        "tools/qwen35_tp1_real_root_logit_correctness_preflight.py",
        "tools/qwen35_tp4_real_root_logit_correctness_contract.py",
        "tools/qwen35_tp4_real_root_logit_correctness_preflight.py",
        "tools/verify_qwen35_tp4_real_root_logit_correctness_gate.py",
    )
    source_hashes = {
        relative: preflight._sha256_file(ROOT / relative)
        for relative in required_sources
    }
    source_manifest["source_file_sha256"] = source_hashes
    source_manifest["source_tree_sha256"] = hashlib.sha256(
        json.dumps(
            dict(sorted(source_hashes.items())),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    reference_process = dict(reference_process)
    reference_process["gpu_uuid"] = "GPU-0"

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def command_runner(command, **kwargs):
        events.append(("reference", tuple(command), kwargs["env"]))
        work_dir = Path(kwargs["cwd"])
        torch.save(reference, work_dir / "reference_logits.pt.partial")
        (work_dir / "reference_process.json.partial").write_text(
            json.dumps(reference_process)
        )
        return Completed()

    def process_factory_builder(
        *,
        work_dir,
        **_kwargs,
    ):
        def process_factory(**kwargs):
            rank = kwargs["rank"]

            class Process:
                pid = rank_rows[rank]["pid"]
                exitcode = None

                def start(self):
                    events.append(("start", rank))
                    row = dict(rank_rows[rank])
                    row["process_group_nonce"] = kwargs[
                        "process_group_nonce"
                    ]
                    row["rendezvous"] = kwargs["rendezvous"]
                    (work_dir / f"rank-{rank}.json.partial").write_text(
                        json.dumps(row)
                    )
                    if rank == 0:
                        torch.save(
                            native,
                            work_dir / "native_rank0_logits.pt.partial",
                        )

                def join(self, _timeout):
                    events.append(("join", rank))
                    self.exitcode = 0

                def is_alive(self):
                    return False

                def terminate(self):
                    events.append(("terminate", rank))

            return Process()

        return process_factory

    gpu_rows = tuple(
        {
            "gpu_index": 30 + rank,
            "gpu_uuid": f"GPU-{rank}",
            "gpu_name": "A100",
            "total_bytes": 80 * 1024**3,
            "free_bytes": 40 * 1024**3,
            "compute_processes": [],
        }
        for rank in range(4)
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        manifest_path = root / "source.json"
        manifest_path.write_text(json.dumps(source_manifest))
        run_dir = root / "authority"
        result = preflight.execute_source_bound_run(
            run_dir=run_dir,
            run_tag="qwen35-tp4-source-bound-test",
            source_manifest_path=manifest_path,
            source_root=ROOT,
            query_gpus=lambda: gpu_rows,
            command_runner=command_runner,
            process_factory_builder=process_factory_builder,
            pid_alive=lambda _pid: False,
        )
        assert result["classification"] == "PASS"
        assert events[0][0] == "reference"
        assert events[1:5] == [
            ("start", 0),
            ("start", 1),
            ("start", 2),
            ("start", 3),
        ]
        assert {path.name for path in run_dir.iterdir()} == set(
            preflight.TP4_ARTIFACT_NAMES
        )
        assert not (
            root / ".qwen35-tp4-source-bound-test.work"
        ).exists()


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"qwen35 TP4 root-logit preflight tests passed ({len(tests)} tests)")


if __name__ == "__main__":
    _run()
