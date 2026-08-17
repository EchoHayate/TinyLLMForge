"""Tests for the Qwen3.5 TP1 real root-logit correctness preflight.

Run remotely with:
  /data00/home/sitian/sitian-workspace01/tllm/env/bin/python \
    tools/test_qwen35_tp1_real_root_logit_correctness_preflight.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys
from contextlib import contextmanager
import tempfile


THIS_DIR = Path(__file__).resolve().parent
PREFLIGHT_PATH = (
    THIS_DIR / "qwen35_tp1_real_root_logit_correctness_preflight.py"
)
SPEC = importlib.util.spec_from_file_location(
    "qwen35_tp1_real_root_logit_correctness_preflight_under_test",
    os.fspath(PREFLIGHT_PATH),
)
preflight = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preflight
SPEC.loader.exec_module(preflight)

import torch


def _expect_value_error(callable_, message_fragment):
    try:
        callable_()
    except ValueError as exc:
        assert message_fragment in str(exc)
    else:
        raise AssertionError("expected ValueError")


def _manual_attention(
    query,
    key,
    value,
    *,
    query_heads,
    kv_heads,
    head_dim,
):
    tokens = query.shape[0]
    query = query.reshape(tokens, query_heads, head_dim).float()
    key = key.reshape(tokens, kv_heads, head_dim).float()
    value = value.reshape(tokens, kv_heads, head_dim).float()
    repeats = query_heads // kv_heads
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    scores = torch.einsum("thd,shd->hts", query, key)
    scores = scores / math.sqrt(head_dim)
    causal = torch.tril(
        torch.ones(tokens, tokens, dtype=torch.bool)
    )
    scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    output = torch.einsum("hts,shd->thd", weights, value)
    return output.reshape(tokens, query_heads * head_dim)


def test_causal_attention_matches_manual_single_head_fp32():
    backend = preflight.Qwen35TP1CausalAttentionBackend(
        head_dim=2,
        query_heads=1,
        kv_heads=1,
    )
    query = torch.tensor(
        [[1.0, 0.5], [0.25, -1.0], [2.0, 1.0]],
        dtype=torch.float32,
    )
    key = torch.tensor(
        [[0.5, 1.0], [-1.0, 0.25], [0.75, -0.5]],
        dtype=torch.float32,
    )
    value = torch.tensor(
        [[2.0, -1.0], [0.5, 3.0], [-2.0, 4.0]],
        dtype=torch.float32,
    )
    actual = backend(query, key, value)
    expected = _manual_attention(
        query,
        key,
        value,
        query_heads=1,
        kv_heads=1,
        head_dim=2,
    )
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float32


def test_causal_attention_replicates_one_kv_head_across_query_heads():
    backend = preflight.Qwen35TP1CausalAttentionBackend(
        head_dim=2,
        query_heads=4,
        kv_heads=1,
    )
    query = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 7
    key = torch.tensor(
        [[1.0, -1.0], [0.5, 2.0], [-0.25, 1.5]],
        dtype=torch.float32,
    )
    value = torch.tensor(
        [[2.0, 1.0], [-1.0, 3.0], [4.0, -2.0]],
        dtype=torch.float32,
    )
    actual = backend(query, key, value)
    expected = _manual_attention(
        query,
        key,
        value,
        query_heads=4,
        kv_heads=1,
        head_dim=2,
    )
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (3, 8)


def test_causal_attention_is_immune_to_future_key_value_poisoning():
    backend = preflight.Qwen35TP1CausalAttentionBackend(
        head_dim=2,
        query_heads=2,
        kv_heads=1,
    )
    query = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.5, 1.0, -1.0, 0.5],
            [2.0, -0.5, 0.25, 1.5],
        ],
        dtype=torch.float32,
    )
    key = torch.tensor(
        [[1.0, 0.5], [0.25, -1.0], [1.5, 2.0]],
        dtype=torch.float32,
    )
    value = torch.tensor(
        [[2.0, -1.0], [0.5, 3.0], [-2.0, 4.0]],
        dtype=torch.float32,
    )
    baseline = backend(query, key, value)
    poisoned_key = key.clone()
    poisoned_value = value.clone()
    poisoned_key[2].fill_(1000000.0)
    poisoned_value[2].fill_(-1000000.0)
    poisoned = backend(query, poisoned_key, poisoned_value)
    torch.testing.assert_close(poisoned[:2], baseline[:2])
    assert not torch.equal(poisoned[2], baseline[2])


def test_causal_attention_uses_fp32_math_and_restores_bfloat16():
    backend = preflight.Qwen35TP1CausalAttentionBackend(
        head_dim=4,
        query_heads=2,
        kv_heads=1,
    )
    query = (
        torch.arange(32, dtype=torch.float32).reshape(4, 8) / 11
    ).to(torch.bfloat16)
    key = (
        torch.arange(16, dtype=torch.float32).reshape(4, 4) / 13
    ).to(torch.bfloat16)
    value = (
        torch.arange(16, dtype=torch.float32).reshape(4, 4) / 17
    ).to(torch.bfloat16)
    actual = backend(query, key, value)
    expected = _manual_attention(
        query,
        key,
        value,
        query_heads=2,
        kv_heads=1,
        head_dim=4,
    ).to(torch.bfloat16)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_causal_attention_rejects_malformed_inputs_and_configuration():
    _expect_value_error(
        lambda: preflight.Qwen35TP1CausalAttentionBackend(
            head_dim=2,
            query_heads=3,
            kv_heads=2,
        ),
        "divisible",
    )
    backend = preflight.Qwen35TP1CausalAttentionBackend(
        head_dim=2,
        query_heads=2,
        kv_heads=1,
    )
    query = torch.ones(3, 4)
    key = torch.ones(3, 2)
    value = torch.ones(3, 2)
    _expect_value_error(
        lambda: backend(query[:, :3], key, value),
        "query width",
    )
    _expect_value_error(
        lambda: backend(query, key[:2], value),
        "token count",
    )
    _expect_value_error(
        lambda: backend(query.to(torch.float64), key, value),
        "dtype",
    )
    _expect_value_error(
        lambda: backend(
            query.to(torch.int64),
            key.to(torch.int64),
            value.to(torch.int64),
        ),
        "floating point",
    )


def test_backend_source_excludes_optimized_or_cached_attention_paths():
    source = PREFLIGHT_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "flash_attn",
        "flashinfer",
        "paged_attention",
        "cuda_graph",
        "sparse_attention",
    ):
        assert forbidden not in source.lower()


class _FakePool:
    def __init__(self):
        self.capacity = 1
        self.device = torch.device("cpu")
        self._tensors = {
            (0, "linear_convolution"): torch.zeros(1, 2, 3),
            (0, "linear_recurrent"): torch.zeros(1, 2, 2, 2),
        }
        self._bindings = {}
        self.events = []

    def activate(self, lease):
        self.events.append(("activate", lease.generation))
        for tensor in self._tensors.values():
            tensor[lease.slot_id].zero_()
        self._bindings[lease.slot_id] = (
            lease.request_id,
            lease.generation,
        )

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
        self.events.append(("release", lease.request_id, lease.generation))


class _FakeNativeModel:
    def __init__(self, pool, *, fail=False, final_only=False):
        self.pool = pool
        self.fail = fail
        self.final_only = final_only
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
        assert not any(
            bool(torch.count_nonzero(tensor))
            for tensor in self.pool._tensors.values()
        )
        for tensor in self.pool._tensors.values():
            tensor[leases[0].slot_id].fill_(leases[0].generation)
        if self.fail:
            raise RuntimeError("injected native failure")
        tokens = input_ids.shape[0]
        normalized = torch.arange(
            tokens * 3,
            dtype=torch.float32,
        ).reshape(tokens, 3)
        logits = torch.arange(
            tokens * 5,
            dtype=torch.float32,
        ).reshape(tokens, 5)
        if self.final_only:
            logits = logits[-1:]
        return normalized, logits


def _fake_candidate(
    *,
    fail=False,
    final_only=False,
    fingerprint="a" * 64,
):
    pool = _FakePool()
    model = _FakeNativeModel(pool, fail=fail, final_only=final_only)
    owner = SimpleNamespace(model=model, pool=pool)
    binding_plan = SimpleNamespace(
        tensor_parallel_size=1,
        tensor_parallel_rank=0,
    )
    candidate = SimpleNamespace(
        owner=owner,
        binding_plan=binding_plan,
        model_fingerprint=fingerprint,
    )
    return candidate, model, pool


def test_native_case_uses_exact_tp1_identity_and_extracts_final_row():
    candidate, model, pool = _fake_candidate()
    allocator = _FakeAllocator()
    prompt = SimpleNamespace(
        case_id="fixture",
        token_ids=(7, 11, 13),
        token_sha256="b" * 64,
    )
    result = preflight.run_native_case(
        candidate=candidate,
        prompt_case=prompt,
        expected_model_fingerprint="a" * 64,
        request_id=41,
        device="cpu",
        allocator=allocator,
    )

    assert result.case_id == "fixture"
    assert result.request_id == 41
    assert result.lease_generation == 1
    assert result.token_count == 3
    assert result.logits.shape == (5,)
    assert result.logits.dtype == torch.float32
    assert result.logits.is_contiguous()
    torch.testing.assert_close(
        result.logits,
        torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0]),
    )
    assert result.state_nonzero_after_commit == {
        "0:linear_convolution": True,
        "0:linear_recurrent": True,
    }
    assert result.release_zeroed is True
    assert result.pool_binding_released is True
    assert model.events == [(
        "run_step",
        (1,),
        (3,),
        [7, 11, 13],
        [0, 1, 2],
        None,
    )]
    assert pool.events == [("activate", 1), ("release", 1)]
    assert allocator.events == [
        ("allocate", 41, 1),
        ("release", 41, 1),
    ]


def test_native_cases_use_fresh_generations_and_release_between_cases():
    candidate, _, pool = _fake_candidate()
    allocator = _FakeAllocator()
    prompts = (
        SimpleNamespace(
            case_id="a",
            token_ids=(3, 4),
            token_sha256="c" * 64,
        ),
        SimpleNamespace(
            case_id="b",
            token_ids=(5, 6, 7),
            token_sha256="d" * 64,
        ),
    )
    results = preflight.run_native_cases(
        candidate=candidate,
        prompt_cases=prompts,
        expected_model_fingerprint="a" * 64,
        device="cpu",
        allocator=allocator,
        first_request_id=100,
    )
    assert tuple(result.case_id for result in results) == ("a", "b")
    assert tuple(result.request_id for result in results) == (100, 101)
    assert tuple(result.lease_generation for result in results) == (1, 2)
    assert pool.events == [
        ("activate", 1),
        ("release", 1),
        ("activate", 2),
        ("release", 2),
    ]


def test_native_case_releases_and_zeroes_after_model_failure():
    candidate, _, pool = _fake_candidate(fail=True)
    allocator = _FakeAllocator()
    prompt = SimpleNamespace(
        case_id="failure",
        token_ids=(7, 8),
        token_sha256="e" * 64,
    )
    try:
        preflight.run_native_case(
            candidate=candidate,
            prompt_case=prompt,
            expected_model_fingerprint="a" * 64,
            request_id=9,
            device="cpu",
            allocator=allocator,
        )
    except RuntimeError as exc:
        assert "injected native failure" in str(exc)
    else:
        raise AssertionError("expected native failure")
    assert pool._bindings == {}
    assert all(
        not bool(torch.count_nonzero(tensor))
        for tensor in pool._tensors.values()
    )
    assert allocator.events == [
        ("allocate", 9, 1),
        ("release", 9, 1),
    ]


def test_native_case_installs_final_token_prefill_context_and_resets_it():
    candidate, _, _ = _fake_candidate(final_only=True)
    allocator = _FakeAllocator()
    prompt = SimpleNamespace(
        case_id="final-only",
        token_ids=(2, 3, 5, 7),
        token_sha256="f" * 64,
    )
    context_events = []

    def set_context(**kwargs):
        context_events.append(("set", kwargs))

    def reset_context():
        context_events.append(("reset",))

    result = preflight.run_native_case(
        candidate=candidate,
        prompt_case=prompt,
        expected_model_fingerprint="a" * 64,
        request_id=12,
        device="cpu",
        allocator=allocator,
        set_context=set_context,
        reset_context=reset_context,
    )
    assert context_events[0][0] == "set"
    assert context_events[0][1]["is_prefill"] is True
    assert context_events[0][1]["mode"] == "prefill"
    assert context_events[0][1]["max_seqlen_q"] == 4
    assert context_events[0][1]["max_seqlen_k"] == 4
    torch.testing.assert_close(
        context_events[0][1]["cu_seqlens_q"],
        torch.tensor([0, 4], dtype=torch.int32),
    )
    torch.testing.assert_close(
        context_events[0][1]["cu_seqlens_k"],
        torch.tensor([0, 4], dtype=torch.int32),
    )
    torch.testing.assert_close(
        context_events[0][1]["logits_indices"],
        torch.tensor([3], dtype=torch.int64),
    )
    assert context_events[1] == ("reset",)
    torch.testing.assert_close(
        result.logits,
        torch.tensor([15.0, 16.0, 17.0, 18.0, 19.0]),
    )


def test_native_case_rejects_wrong_identity_tp_or_device_before_activation():
    prompt = SimpleNamespace(
        case_id="fixture",
        token_ids=(1,),
        token_sha256="f" * 64,
    )
    candidate, _, pool = _fake_candidate(fingerprint="1" * 64)
    _expect_value_error(
        lambda: preflight.run_native_case(
            candidate=candidate,
            prompt_case=prompt,
            expected_model_fingerprint="a" * 64,
            request_id=1,
            device="cpu",
            allocator=_FakeAllocator(),
        ),
        "model fingerprint",
    )
    candidate, _, pool = _fake_candidate()
    candidate.binding_plan.tensor_parallel_size = 4
    _expect_value_error(
        lambda: preflight.run_native_case(
            candidate=candidate,
            prompt_case=prompt,
            expected_model_fingerprint="a" * 64,
            request_id=1,
            device="cpu",
            allocator=_FakeAllocator(),
        ),
        "TP1",
    )
    candidate, _, pool = _fake_candidate()
    _expect_value_error(
        lambda: preflight.run_native_case(
            candidate=candidate,
            prompt_case=prompt,
            expected_model_fingerprint="a" * 64,
            request_id=1,
            device="cuda:0",
            allocator=_FakeAllocator(),
        ),
        "pool device",
    )
    assert pool.events == []


def test_preflight_source_does_not_construct_engine_runner_or_scheduler():
    source = PREFLIGHT_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "LLMEngine(",
        "ModelRunner(",
        "Scheduler(",
        "Sampler(",
    ):
        assert forbidden not in source


def test_reference_worker_invocation_is_offline_bf16_eager_and_single_gpu():
    with tempfile.TemporaryDirectory() as temporary_directory:
        work_dir = Path(temporary_directory)
        invocation = preflight.build_reference_worker_invocation(
            python_executable="/approved/python",
            script_path=PREFLIGHT_PATH,
            work_dir=work_dir,
            gpu_index=3,
            base_environment={
                "PATH": "/approved/bin",
                "HTTP_PROXY": "http://forbidden-proxy",
                "HTTPS_PROXY": "http://forbidden-proxy",
            },
        )

    assert invocation.argv[0] == "/approved/python"
    assert invocation.argv[1] == os.fspath(PREFLIGHT_PATH.resolve())
    assert invocation.argv[2] == "internal-reference"
    assert invocation.argv[3:] == (
        "--model-dir",
        preflight.APPROVED_MODEL_DIR,
        "--model-manifest-sha256",
        preflight.APPROVED_MODEL_MANIFEST_SHA256,
        "--tensor-output",
        os.fspath(work_dir / "reference_logits.pt.partial"),
        "--process-output",
        os.fspath(work_dir / "reference_process.json.partial"),
        "--dtype",
        "bfloat16",
        "--attn-implementation",
        "eager",
        "--local-files-only",
        "--no-trust-remote-code",
        "--no-use-cache",
    )
    assert invocation.environment["PATH"] == "/approved/bin"
    assert invocation.environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert invocation.environment["HF_HUB_OFFLINE"] == "1"
    assert invocation.environment["TRANSFORMERS_OFFLINE"] == "1"
    assert invocation.environment["HF_DATASETS_OFFLINE"] == "1"
    assert invocation.environment["TOKENIZERS_PARALLELISM"] == "false"
    for forbidden in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        assert forbidden not in invocation.environment
    assert invocation.minimum_free_bytes == 24 * 1024**3
    assert invocation.gpu_index == 3


def test_reference_gpu_preflight_requires_exact_selected_gpu_and_24_gib():
    calls = []

    def query_gpu(gpu_index):
        calls.append(gpu_index)
        return {
            "gpu_index": gpu_index,
            "gpu_uuid": "GPU-approved",
            "free_bytes": 24 * 1024**3,
        }

    result = preflight.require_reference_gpu_resource(
        gpu_index=2,
        query_gpu=query_gpu,
    )
    assert calls == [2]
    assert result == {
        "gpu_index": 2,
        "gpu_uuid": "GPU-approved",
        "free_bytes": 24 * 1024**3,
        "minimum_free_bytes": 24 * 1024**3,
    }

    def insufficient(_gpu_index):
        return {
            "gpu_index": 2,
            "gpu_uuid": "GPU-approved",
            "free_bytes": 24 * 1024**3 - 1,
        }

    _expect_value_error(
        lambda: preflight.require_reference_gpu_resource(
            gpu_index=2,
            query_gpu=insufficient,
        ),
        "24 GiB",
    )

    def wrong_gpu(_gpu_index):
        return {
            "gpu_index": 1,
            "gpu_uuid": "GPU-wrong",
            "free_bytes": 80 * 1024**3,
        }

    _expect_value_error(
        lambda: preflight.require_reference_gpu_resource(
            gpu_index=2,
            query_gpu=wrong_gpu,
        ),
        "selected GPU",
    )


def test_nvidia_smi_gpu_query_returns_exact_index_uuid_and_bytes():
    calls = []

    def command_runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            "0, GPU-zero, 24576\n1, GPU-one, 32768\n",
            "",
        )

    result = preflight.query_nvidia_smi_gpu(
        1,
        command_runner=command_runner,
    )
    assert result == {
        "gpu_index": 1,
        "gpu_uuid": "GPU-one",
        "free_bytes": 32768 * 1024**2,
    }
    assert calls == [(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free",
            "--format=csv,noheader,nounits",
        ],
        {
            "check": False,
            "text": True,
            "capture_output": True,
        },
    )]


class _FakeReferenceProcess:
    def __init__(
        self,
        *,
        pid,
        work_dir,
        returncode=0,
        timeout=False,
        write_outputs=True,
    ):
        self.pid = pid
        self.returncode = None
        self._work_dir = Path(work_dir)
        self._final_returncode = returncode
        self._timeout = timeout
        self._write_outputs = write_outputs
        self.killed = False

    def communicate(self, timeout):
        assert timeout == 37
        if self._timeout:
            raise preflight.subprocess.TimeoutExpired(
                cmd=("reference",),
                timeout=timeout,
            )
        if self._write_outputs:
            torch.save(
                {
                    "p17": torch.arange(8, dtype=torch.float32),
                    "p65": torch.arange(8, dtype=torch.float32) + 1,
                    "synthetic": torch.arange(8, dtype=torch.float32) + 2,
                },
                self._work_dir / preflight.REFERENCE_TENSOR_PARTIAL_NAME,
            )
            (
                self._work_dir / preflight.REFERENCE_PROCESS_PARTIAL_NAME
            ).write_text(
                json.dumps({
                    "worker": "reference",
                    "pid": self.pid,
                    "exit_code": 0,
                    "model_manifest_sha256": (
                        preflight.APPROVED_MODEL_MANIFEST_SHA256
                    ),
                    "gpu_index": 2,
                    "gpu_uuid": "GPU-approved",
                    "free_bytes_before": 40 * 1024**3,
                    "minimum_free_bytes": 24 * 1024**3,
                    "local_files_only": True,
                    "trust_remote_code": False,
                    "dtype": "bfloat16",
                    "attn_implementation": "eager",
                    "use_cache": False,
                    "case_ids": ["p17", "p65", "synthetic"],
                    "vocab_size": 8,
                    "cleanup_complete": True,
                }),
                encoding="utf-8",
            )
        self.returncode = self._final_returncode
        return ("reference stdout", "reference stderr")

    def kill(self):
        self.killed = True
        self.returncode = -9


def test_run_reference_worker_accepts_complete_fresh_process_outputs():
    events = []
    processes = []
    with tempfile.TemporaryDirectory() as temporary_directory:
        work_dir = Path(temporary_directory)

        def query_gpu(gpu_index):
            events.append(("resource", gpu_index))
            return {
                "gpu_index": gpu_index,
                "gpu_uuid": "GPU-approved",
                "free_bytes": 40 * 1024**3,
            }

        def launch_process(invocation, received_work_dir):
            events.append(("spawn", invocation.gpu_index))
            assert received_work_dir == work_dir
            process = _FakeReferenceProcess(
                pid=4321,
                work_dir=work_dir,
            )
            processes.append(process)
            return process

        result = preflight.run_reference_worker(
            python_executable="/approved/python",
            script_path=PREFLIGHT_PATH,
            work_dir=work_dir,
            gpu_index=2,
            timeout_seconds=37,
            query_gpu=query_gpu,
            launch_process=launch_process,
            pid_alive=lambda pid: False,
            expected_case_ids=("p17", "p65", "synthetic"),
            expected_vocab_size=8,
            base_environment={"PATH": "/approved/bin"},
        )

    assert events == [("resource", 2), ("spawn", 2)]
    assert len(processes) == 1
    assert result.pid == 4321
    assert result.stdout == "reference stdout"
    assert result.stderr == "reference stderr"
    assert result.process_row["cleanup_complete"] is True
    assert tuple(result.logits) == ("p17", "p65", "synthetic")
    assert all(
        tensor.device.type == "cpu"
        and tensor.dtype == torch.float32
        and tensor.shape == (8,)
        and tensor.is_contiguous()
        for tensor in result.logits.values()
    )
    assert result.resource["free_bytes"] == 40 * 1024**3


def test_run_reference_worker_rejects_timeout_exit_output_and_surviving_pid():
    def query_gpu(gpu_index):
        return {
            "gpu_index": gpu_index,
            "gpu_uuid": "GPU-approved",
            "free_bytes": 40 * 1024**3,
        }

    def run_case(
        *,
        returncode=0,
        timeout=False,
        write_outputs=True,
        pid_alive=lambda _pid: False,
        message,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            work_dir = Path(temporary_directory)
            processes = []

            def launch_process(_invocation, _work_dir):
                process = _FakeReferenceProcess(
                    pid=9876,
                    work_dir=work_dir,
                    returncode=returncode,
                    timeout=timeout,
                    write_outputs=write_outputs,
                )
                processes.append(process)
                return process

            _expect_value_error(
                lambda: preflight.run_reference_worker(
                    python_executable="/approved/python",
                    script_path=PREFLIGHT_PATH,
                    work_dir=work_dir,
                    gpu_index=2,
                    timeout_seconds=37,
                    query_gpu=query_gpu,
                    launch_process=launch_process,
                    pid_alive=pid_alive,
                    expected_case_ids=("p17", "p65", "synthetic"),
                    expected_vocab_size=8,
                    base_environment={"PATH": "/approved/bin"},
                ),
                message,
            )
            return processes[0]

    timed_out = run_case(timeout=True, message="timed out")
    assert timed_out.killed is True
    run_case(returncode=7, message="exit code")
    run_case(write_outputs=False, message="tensor output")
    run_case(pid_alive=lambda pid: pid == 9876, message="still alive")


def test_custom_op_annotation_compatibility_is_temporary():
    calls = []

    def operation(input: "torch.Tensor") -> "torch.Tensor":
        return input

    original_annotations = dict(operation.__annotations__)

    def infer_schema(function, mutates_args=()):
        calls.append(dict(function.__annotations__))
        return "(Tensor input) -> Tensor"

    owner = SimpleNamespace(infer_schema=infer_schema)
    with preflight.torch_custom_op_annotation_compatibility(
        infer_schema_owner=owner,
    ):
        assert owner.infer_schema(operation, ()) == (
            "(Tensor input) -> Tensor"
        )
        assert operation.__annotations__ == original_annotations
    assert owner.infer_schema is infer_schema
    assert calls == [{
        "input": torch.Tensor,
        "return": torch.Tensor,
    }]


class _FakeOfficialModel:
    def __init__(self, *, vocab_size=8, non_finite=False):
        self.eval_calls = 0
        self.to_calls = []
        self.forward_calls = []
        self.vocab_size = vocab_size
        self.non_finite = non_finite

    def eval(self):
        self.eval_calls += 1
        return self

    def to(self, *, device):
        self.to_calls.append(device)
        return self

    def __call__(self, **kwargs):
        assert torch.is_grad_enabled() is False
        self.forward_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        token_count = input_ids.shape[1]
        logits = torch.arange(
            token_count * self.vocab_size,
            dtype=torch.bfloat16,
            device=input_ids.device,
        ).reshape(1, token_count, self.vocab_size)
        if self.non_finite:
            logits[0, -1, 0] = float("nan")
        return SimpleNamespace(logits=logits)


class _FakeAutoModel:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def from_pretrained(self, model_dir, **kwargs):
        self.calls.append((Path(model_dir), kwargs))
        return self.model


class _FakeCuda:
    def __init__(self):
        self.empty_cache_calls = 0
        self.reset_calls = 0

    def is_available(self):
        return True

    def mem_get_info(self):
        return 40 * 1024**3, 80 * 1024**3

    def reset_peak_memory_stats(self):
        self.reset_calls += 1

    def max_memory_allocated(self):
        return 5 * 1024**3

    def max_memory_reserved(self):
        return 6 * 1024**3

    def synchronize(self):
        return None

    def empty_cache(self):
        self.empty_cache_calls += 1


def test_execute_reference_worker_uses_official_eager_no_cache_forward():
    fake_model = _FakeOfficialModel()
    auto_model = _FakeAutoModel(fake_model)
    cuda = _FakeCuda()
    compatibility_events = []

    @contextmanager
    def compatibility():
        compatibility_events.append("enter")
        yield
        compatibility_events.append("exit")

    prompts = (
        SimpleNamespace(case_id="p17", token_ids=(1, 2)),
        SimpleNamespace(case_id="p65", token_ids=(3, 4, 5)),
        SimpleNamespace(case_id="synthetic", token_ids=(6,)),
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        tensor_output = root / "reference_logits.pt.partial"
        process_output = root / "reference_process.json.partial"
        row = preflight.execute_reference_worker(
            model_dir=Path(preflight.APPROVED_MODEL_DIR),
            model_manifest_sha256=(
                preflight.APPROVED_MODEL_MANIFEST_SHA256
            ),
            tensor_output=tensor_output,
            process_output=process_output,
            prompt_cases=prompts,
            expected_vocab_size=8,
            auto_model=auto_model,
            cuda=cuda,
            process_id=2468,
            gpu_index=2,
            gpu_uuid="GPU-approved",
            verify_model_identity=lambda *_args, **_kwargs: None,
            process_memory_reader=lambda: {
                "vmrss_kib": 100,
                "vmhwm_kib": 200,
            },
            version_reader=lambda name: {
                "torch": "2.4.1",
                "transformers": "5.8.1",
            }[name],
            timestamp_reader=iter(
                ("2026-07-28T10:00:00Z", "2026-07-28T10:00:01Z")
            ).__next__,
            custom_op_compatibility=compatibility,
        )
        tensors = torch.load(
            tensor_output,
            map_location="cpu",
            weights_only=True,
        )
        persisted = json.loads(process_output.read_text(encoding="utf-8"))

    assert auto_model.calls == [(
        Path(preflight.APPROVED_MODEL_DIR),
        {
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": torch.bfloat16,
            "attn_implementation": "eager",
        },
    )]
    assert compatibility_events == ["enter", "exit"]
    assert fake_model.eval_calls == 1
    assert fake_model.to_calls == [torch.device("cuda:0")]
    assert [call["input_ids"].tolist() for call in fake_model.forward_calls] == [
        [[1, 2]],
        [[3, 4, 5]],
        [[6]],
    ]
    assert all(
        call["input_ids"].dtype == torch.int64
        and call["input_ids"].device == torch.device("cuda:0")
        and call["use_cache"] is False
        and call["return_dict"] is True
        for call in fake_model.forward_calls
    )
    assert tuple(tensors) == ("p17", "p65", "synthetic")
    assert torch.equal(
        tensors["p17"],
        torch.arange(8, 16, dtype=torch.float32),
    )
    assert torch.equal(
        tensors["p65"],
        torch.arange(16, 24, dtype=torch.float32),
    )
    assert torch.equal(
        tensors["synthetic"],
        torch.arange(0, 8, dtype=torch.float32),
    )
    assert persisted == row
    assert row["pid"] == 2468
    assert row["case_ids"] == ["p17", "p65", "synthetic"]
    assert row["vocab_size"] == 8
    assert row["free_bytes_before"] == 40 * 1024**3
    assert row["max_memory_allocated"] == 5 * 1024**3
    assert row["max_memory_reserved"] == 6 * 1024**3
    assert row["cleanup_complete"] is True
    assert cuda.reset_calls == 1
    assert cuda.empty_cache_calls == 1


def test_execute_reference_worker_rejects_identity_nonfinite_and_wrong_vocab():
    prompts = (SimpleNamespace(case_id="p17", token_ids=(1, 2)),)

    def execute_with(model, *, verify_model_identity=lambda *_args: None):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            return preflight.execute_reference_worker(
                model_dir=Path(preflight.APPROVED_MODEL_DIR),
                model_manifest_sha256=(
                    preflight.APPROVED_MODEL_MANIFEST_SHA256
                ),
                tensor_output=root / "reference_logits.pt.partial",
                process_output=root / "reference_process.json.partial",
                prompt_cases=prompts,
                expected_vocab_size=8,
                auto_model=_FakeAutoModel(model),
                cuda=_FakeCuda(),
                process_id=2468,
                gpu_index=2,
                gpu_uuid="GPU-approved",
                verify_model_identity=verify_model_identity,
                process_memory_reader=lambda: {
                    "vmrss_kib": 100,
                    "vmhwm_kib": 200,
                },
                version_reader=lambda _name: "test",
                timestamp_reader=iter(("start", "finish")).__next__,
                custom_op_compatibility=lambda: contextmanager(
                    lambda: (yield)
                )(),
            )

    _expect_value_error(
        lambda: execute_with(
            _FakeOfficialModel(),
            verify_model_identity=lambda *_args: (
                (_ for _ in ()).throw(ValueError("identity mismatch"))
            ),
        ),
        "identity mismatch",
    )
    _expect_value_error(
        lambda: execute_with(_FakeOfficialModel(non_finite=True)),
        "shape or values",
    )
    _expect_value_error(
        lambda: execute_with(_FakeOfficialModel(vocab_size=7)),
        "shape or values",
    )


def test_internal_reference_cli_forwards_only_frozen_contract():
    calls = []

    def execute_reference(**kwargs):
        calls.append(kwargs)
        return {"worker": "reference"}

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        exit_code = preflight.main(
            [
                "internal-reference",
                "--model-dir",
                preflight.APPROVED_MODEL_DIR,
                "--model-manifest-sha256",
                preflight.APPROVED_MODEL_MANIFEST_SHA256,
                "--tensor-output",
                os.fspath(root / "reference_logits.pt.partial"),
                "--process-output",
                os.fspath(root / "reference_process.json.partial"),
                "--dtype",
                "bfloat16",
                "--attn-implementation",
                "eager",
                "--local-files-only",
                "--no-trust-remote-code",
                "--no-use-cache",
            ],
            execute_reference=execute_reference,
            prompt_case_loader=lambda: (
                SimpleNamespace(case_id="p17", token_ids=(1,)),
                SimpleNamespace(case_id="p65", token_ids=(2,)),
                SimpleNamespace(case_id="synthetic", token_ids=(3,)),
            ),
            environment={
                "TINYVLLM_GATE_PHYSICAL_GPU_INDEX": "2",
                "TINYVLLM_GATE_GPU_UUID": "GPU-approved",
            },
        )

    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0]["model_dir"] == Path(preflight.APPROVED_MODEL_DIR)
    assert calls[0]["model_manifest_sha256"] == (
        preflight.APPROVED_MODEL_MANIFEST_SHA256
    )
    assert calls[0]["expected_vocab_size"] == 248320
    assert calls[0]["gpu_index"] == 2
    assert calls[0]["gpu_uuid"] == "GPU-approved"
    assert tuple(case.case_id for case in calls[0]["prompt_cases"]) == (
        "p17",
        "p65",
        "synthetic",
    )


def test_internal_native_cli_forwards_only_frozen_contract():
    calls = []

    def execute_native(**kwargs):
        calls.append(kwargs)
        return {"worker": "native"}

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        exit_code = preflight.main(
            [
                "internal-native",
                "--tensor-output",
                os.fspath(root / "native_logits.pt.partial"),
                "--process-output",
                os.fspath(root / "native_process.json.partial"),
                "--state-output",
                os.fspath(root / "native_state.json.partial"),
                "--dtype",
                "bfloat16",
                "--recurrent-dtype",
                "float32",
                "--tensor-parallel-size",
                "1",
                "--tensor-parallel-rank",
                "0",
            ],
            execute_native=execute_native,
            prompt_case_loader=lambda: (
                SimpleNamespace(case_id="p17", token_ids=(1,)),
                SimpleNamespace(case_id="p65", token_ids=(2,)),
                SimpleNamespace(case_id="synthetic", token_ids=(3,)),
            ),
            environment={
                "TINYVLLM_GATE_PHYSICAL_GPU_INDEX": "2",
                "TINYVLLM_GATE_GPU_UUID": "GPU-approved",
            },
        )

    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0]["expected_vocab_size"] == 248320
    assert calls[0]["gpu_index"] == 2
    assert calls[0]["gpu_uuid"] == "GPU-approved"
    assert tuple(case.case_id for case in calls[0]["prompt_cases"]) == (
        "p17",
        "p65",
        "synthetic",
    )


def test_run_cli_forwards_source_bound_coordinator_contract():
    calls = []

    def execute_run(**kwargs):
        calls.append(kwargs)
        return {"classification": "PASS"}

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        source_manifest = root / "source_manifest.input.json"
        source_manifest.write_text("{}")
        exit_code = preflight.main(
            [
                "run",
                "--run-dir",
                os.fspath(root / "authority"),
                "--run-tag",
                "qwen35-tp1-authority-test",
                "--source-manifest",
                os.fspath(source_manifest),
            ],
            execute_run=execute_run,
            environment={},
        )

    assert exit_code == 0
    assert calls == [{
        "run_dir": root / "authority",
        "run_tag": "qwen35-tp1-authority-test",
        "source_manifest_path": source_manifest,
    }]


def test_validate_cli_forwards_read_only_artifact_directory():
    calls = []

    def execute_validate(run_dir):
        calls.append(Path(run_dir))
        return {"classification": "PASS", "checks": 123}

    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = Path(temporary_directory) / "authority"
        exit_code = preflight.main(
            ["validate", os.fspath(run_dir)],
            execute_validate=execute_validate,
            environment={},
        )

    assert exit_code == 0
    assert calls == [run_dir]


def test_source_bound_run_orders_workers_and_publishes_exact_artifact():
    events = []
    source_root = PREFLIGHT_PATH.parents[1]
    source_paths = (
        "tools/qwen35_tp1_real_root_logit_correctness_contract.py",
        "tools/qwen35_tp1_real_root_logit_correctness_preflight.py",
        "tools/verify_qwen35_tp1_real_root_logit_correctness_gate.py",
    )
    source_hashes = {
        name: preflight._sha256_file(source_root / name)
        for name in source_paths
    }
    source_manifest = {
        "schema_version": 1,
        "source_file_sha256": source_hashes,
        "source_tree_sha256": preflight.hashlib.sha256(
            preflight.json.dumps(
                dict(sorted(source_hashes.items())),
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
    }
    cases = preflight._load_frozen_prompt_cases()
    case_ids = [case.case_id for case in cases]
    logits = {
        case.case_id: torch.arange(
            248320,
            dtype=torch.float32,
        ).add_(index).contiguous()
        for index, case in enumerate(cases)
    }

    def process_row(worker, pid):
        return {
            "worker": worker,
            "pid": pid,
            "exit_code": 0,
            "model_manifest_sha256": (
                preflight.APPROVED_MODEL_MANIFEST_SHA256
            ),
            "gpu_index": 1,
            "gpu_uuid": "GPU-selected",
            "free_bytes_before": 40 * 1024**3,
            "minimum_free_bytes": 24 * 1024**3,
            "case_ids": case_ids,
            "vocab_size": 248320,
            "cleanup_complete": True,
            "start_timestamp": f"{worker}-start",
            "finish_timestamp": f"{worker}-finish",
            "torch_version": "2.4.1",
            "vmrss_kib": 100,
            "vmhwm_kib": 200,
            "max_memory_allocated": 1024,
            "max_memory_reserved": 2048,
            **({
                "local_files_only": True,
                "trust_remote_code": False,
                "dtype": "bfloat16",
                "attn_implementation": "eager",
                "use_cache": False,
            } if worker == "reference" else {
                "tensor_parallel_size": 1,
                "tensor_parallel_rank": 0,
                "dtype": "bfloat16",
                "recurrent_dtype": "float32",
                "engine_constructed": False,
                "model_runner_constructed": False,
                "scheduler_constructed": False,
                "sampler_constructed": False,
            }),
        }

    def command_runner(command, **kwargs):
        mode = command[2]
        events.append(mode)
        environment = kwargs["env"]
        assert environment["TINYVLLM_DIST_PORT"].isdigit()
        assert environment["MASTER_PORT"].isdigit()
        events.append((
            environment["TINYVLLM_DIST_PORT"],
            environment["MASTER_PORT"],
        ))
        work_dir = Path(kwargs["cwd"])
        if mode == "internal-reference":
            torch.save(logits, work_dir / "reference_logits.pt.partial")
            (work_dir / "reference_process.json.partial").write_text(
                json.dumps(process_row("reference", 101))
            )
        else:
            assert events[0] == "internal-reference"
            assert events[2] == "internal-native"
            assert events[1] != events[3]
            torch.save(logits, work_dir / "native_logits.pt.partial")
            (work_dir / "native_process.json.partial").write_text(
                json.dumps(process_row("native", 202))
            )
            (work_dir / "native_state.json.partial").write_text(
                json.dumps([{
                    "case_id": case.case_id,
                    "prepare_read_only": True,
                    "linear_layer_count": 18,
                    "changed_component_count": 36,
                    "full_attention_state_component_count": 0,
                    "commit_count": 1,
                    "release_zeroed": True,
                    "pool_binding_released": True,
                } for case in cases])
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        manifest_path = root / "source_manifest.input.json"
        manifest_path.write_text(json.dumps(source_manifest))
        run_dir = root / "authority"
        result = preflight.execute_source_bound_run(
            run_dir=run_dir,
            run_tag="qwen35-tp1-source-bound-test",
            source_manifest_path=manifest_path,
            query_gpus=lambda: (
                {
                    "gpu_index": 0,
                    "gpu_uuid": "GPU-small",
                    "free_bytes": 8 * 1024**3,
                },
                {
                    "gpu_index": 1,
                    "gpu_uuid": "GPU-selected",
                    "free_bytes": 40 * 1024**3,
                },
            ),
            command_runner=command_runner,
            pid_alive=lambda _pid: False,
        )

        assert result["classification"] == "PASS"
        assert result["gpu"]["gpu_index"] == 1
        assert [events[0], events[2]] == [
            "internal-reference",
            "internal-native",
        ]
        assert events[1] != events[3]
        assert {path.name for path in run_dir.iterdir()} == {
            "tp1_real_root_logit_correctness.json",
            "reference_logits.pt",
            "native_logits.pt",
            "source_manifest.json",
        }
        assert not (root / ".qwen35-tp1-source-bound-test.work").exists()


def test_frozen_prompt_loader_registers_dataclass_module_and_returns_contract():
    cases = preflight._load_frozen_prompt_cases()
    assert tuple(case.case_id for case in cases) == (
        "p17",
        "p65",
        "synthetic",
    )
    assert tuple(len(case.token_ids) for case in cases) == (17, 65, 11)


def test_execute_native_worker_loads_moves_runs_and_writes_state_evidence():
    events = []
    cuda = _FakeCuda()
    candidate = SimpleNamespace(owner=SimpleNamespace(
        model=SimpleNamespace(),
        pool=SimpleNamespace(),
    ))
    prompts = (
        SimpleNamespace(case_id="p17", token_ids=(1, 2)),
        SimpleNamespace(case_id="p65", token_ids=(3, 4, 5)),
        SimpleNamespace(case_id="synthetic", token_ids=(6,)),
    )

    def build_candidate():
        events.append("build")
        return SimpleNamespace(candidate=candidate)

    def move_candidate(received, **kwargs):
        events.append(("move", received, kwargs))
        return received

    def allocator_factory(capacity):
        events.append(("allocator", capacity))
        return object()

    def run_cases(**kwargs):
        events.append(("run", kwargs))
        return tuple(
            SimpleNamespace(
                case_id=case.case_id,
                request_id=100 + index,
                lease_generation=index + 1,
                token_count=len(case.token_ids),
                logits=torch.arange(
                    8,
                    dtype=torch.float32,
                ) + index,
                state_nonzero_after_commit={
                    **{
                        f"{layer}:linear_convolution": True
                        for layer in range(18)
                    },
                    **{
                        f"{layer}:linear_recurrent": True
                        for layer in range(18)
                    },
                },
                release_zeroed=True,
                pool_binding_released=True,
            )
            for index, case in enumerate(prompts)
        )

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        row = preflight.execute_native_worker(
            tensor_output=root / "native_logits.pt.partial",
            process_output=root / "native_process.json.partial",
            state_output=root / "native_state.json.partial",
            prompt_cases=prompts,
            expected_vocab_size=8,
            build_candidate=build_candidate,
            move_candidate=move_candidate,
            run_cases=run_cases,
            allocator_factory=allocator_factory,
            cuda=cuda,
            process_id=1357,
            gpu_index=2,
            gpu_uuid="GPU-approved",
            process_memory_reader=lambda: {
                "vmrss_kib": 100,
                "vmhwm_kib": 200,
            },
            version_reader=lambda _name: "2.4.1",
            timestamp_reader=iter(("start", "finish")).__next__,
        )
        tensors = torch.load(
            root / "native_logits.pt.partial",
            map_location="cpu",
            weights_only=True,
        )
        states = json.loads(
            (root / "native_state.json.partial").read_text()
        )
        persisted = json.loads(
            (root / "native_process.json.partial").read_text()
        )

    assert events[0] == "build"
    assert events[1][0] == "move"
    assert events[1][2] == {
        "device": "cuda:0",
        "expected_model_fingerprint": (
            preflight.APPROVED_MODEL_MANIFEST_SHA256
        ),
    }
    assert events[2] == ("allocator", 1)
    assert events[3][0] == "run"
    assert events[3][1]["first_request_id"] == 100
    assert tuple(tensors) == ("p17", "p65", "synthetic")
    assert all(
        tensor.dtype == torch.float32
        and tensor.device.type == "cpu"
        and tensor.shape == (8,)
        for tensor in tensors.values()
    )
    assert len(states) == 3
    assert all(row["changed_component_count"] == 36 for row in states)
    assert all(row["linear_layer_count"] == 18 for row in states)
    assert all(row["release_zeroed"] is True for row in states)
    assert persisted == row
    assert row["worker"] == "native"
    assert row["pid"] == 1357
    assert row["tensor_parallel_size"] == 1
    assert row["recurrent_dtype"] == "float32"
    assert row["cleanup_complete"] is True
    assert cuda.empty_cache_calls == 1


class _MigratableModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))


def test_move_loaded_candidate_migrates_model_and_pool_as_one_owner():
    model = _MigratableModel()
    pool = _FakePool()
    adapter = SimpleNamespace(
        pool=pool,
        layer_index=0,
        convolution=pool._tensors[(0, "linear_convolution")],
        recurrent=pool._tensors[(0, "linear_recurrent")],
    )
    transaction = SimpleNamespace(adapters=(adapter,), pool=pool)
    owner = SimpleNamespace(
        model=model,
        pool=pool,
        state_transaction=transaction,
        runtime_bridge=SimpleNamespace(pool=pool),
    )
    candidate = SimpleNamespace(
        owner=owner,
        binding_plan=SimpleNamespace(
            tensor_parallel_size=1,
            tensor_parallel_rank=0,
        ),
        model_fingerprint="a" * 64,
    )
    moved = preflight.move_loaded_candidate_to_device(
        candidate,
        device="meta",
        expected_model_fingerprint="a" * 64,
    )
    assert moved is candidate
    assert model.weight.device.type == "meta"
    assert pool.device.type == "meta"
    assert all(
        tensor.device.type == "meta"
        for tensor in pool._tensors.values()
    )
    assert adapter.pool is pool
    assert adapter.convolution is pool._tensors[
        (0, "linear_convolution")
    ]
    assert adapter.recurrent is pool._tensors[(0, "linear_recurrent")]
    assert adapter.convolution.device.type == "meta"
    assert adapter.recurrent.device.type == "meta"
    assert owner.runtime_bridge.pool is pool


def test_real_cpu_candidate_builder_calls_authorized_stack_once_in_order():
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
    candidate, _, _ = _fake_candidate(
        fingerprint=preflight.APPROVED_MODEL_MANIFEST_SHA256,
    )
    candidate.owner.pool = pool
    candidate.owner.model.pool = pool

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

    result = preflight.build_real_tp1_cpu_candidate(
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
    metadata_event = events[1]
    assert metadata_event[1] == preflight.APPROVED_MODEL_DIR
    assert metadata_event[2] == {
        "shards": ("shard-identity",),
        "expected_config_sha256": preflight.APPROVED_CONFIG_SHA256,
        "expected_index_sha256": preflight.APPROVED_INDEX_SHA256,
        "expected_config_index_header_sha256": (
            preflight.APPROVED_COMPOSITE_SHA256
        ),
    }
    assert events[3][2] == {
        "tensor_parallel_size": 1,
        "dtype": torch.bfloat16,
        "recurrent_dtype": torch.float32,
        "speculative_tokens": 1,
    }
    assert events[4][2] == {"capacity": 1, "device": "cpu"}
    assert events[5][3]["tensor_parallel_size"] == 1
    assert events[5][3]["tensor_parallel_rank"] == 0
    assert events[5][3]["parameter_device"] == "cpu"
    backend = events[5][3]["build_attention_backend"](3, 16, 4, 128)
    assert isinstance(
        backend,
        preflight.Qwen35TP1CausalAttentionBackend,
    )
    assert backend.query_heads == 16
    assert backend.kv_heads == 4
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


def test_real_cpu_candidate_builder_rejects_identity_or_pool_drift():
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
            return _FakePool()

        def prepare_target(self, *_args, **kwargs):
            return SimpleNamespace(pool=kwargs["pool"])

        def build_loader(self, provider, **_kwargs):
            def load(_request):
                target = provider()
                candidate, _, _ = _fake_candidate(
                    fingerprint="1" * 64
                )
                candidate.owner.pool = _FakePool()
                assert candidate.owner.pool is not target.pool
                return candidate

            return load

        def make_request(self, **_kwargs):
            return object()

    _expect_value_error(
        lambda: preflight.build_real_tp1_cpu_candidate(
            dependencies=_Dependencies(),
        ),
        "model fingerprint",
    )


def test_main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and name != "test_main"
    ]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} tests")


if __name__ == "__main__":
    test_main()
