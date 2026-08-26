from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/run_qwen38_tp_correctness.py"
VERIFIER_PATH = ROOT / "tools/qwen38_tp_correctness.py"


def _load():
    assert MODULE_PATH.is_file(), "Qwen3.8 correctness runner is missing"
    spec = importlib.util.spec_from_file_location(
        "run_qwen38_tp_correctness_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load()


class _TorchCudaUuid:

    def __str__(self):
        return "7dc22583-df04-6c76-4ba5-ea32c428c130"


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "qwen38_tp_correctness_for_runner_test",
        VERIFIER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verifier = _load_verifier()


def test_production_root_is_the_approved_remote_data_mount():
    assert runner.APPROVED_REMOTE_ROOT == Path(
        "/data00/home/sitian/tinyllmforge-workspaces/"
        "command-timeline-20260818"
    )


def test_cuda_device_identity_normalizes_torch_cuuid(monkeypatch):
    cuda = SimpleNamespace(
        current_device=lambda: 0,
        get_device_properties=lambda _index: (
            SimpleNamespace(uuid=_TorchCudaUuid())
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=cuda),
    )

    assert runner._read_cuda_device_identity() == {
        "gpu_index": 0,
        "gpu_uuid": "GPU-7dc22583-df04-6c76-4ba5-ea32c428c130",
    }


def _plan(tmp_path, monkeypatch):
    approved = tmp_path / "approved"
    source = approved / "source"
    model = approved / "models" / "qwen38"
    source.mkdir(parents=True)
    model.mkdir(parents=True)
    manifest = model.parent / "model_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "APPROVED_REMOTE_ROOT",
        approved.resolve(),
    )
    return runner.build_correctness_plan(
        attempt_root=approved / "attempts" / "correctness-r1",
        source_root=source,
        model_root=model,
        model_manifest_path=manifest,
        source_tree_sha256="1" * 64,
        model_manifest_sha256="2" * 64,
        python_executable=approved / "env" / "bin" / "python",
        torchrun_executable=approved / "env" / "bin" / "torchrun",
        gpu_indices=(2, 3, 4, 5),
        rendezvous_ports=(32101, 32102),
        prompt_token_ids=(11, 22, 33, 44),
        generated_tokens=2,
        topk=8,
        timeout_s=900,
    )


def test_plan_emits_three_text_only_greedy_commands(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)

    assert plan["schema_version"] == runner.PLAN_SCHEMA
    assert plan["command_order"] == [
        "official_tp1",
        "tinyllmforge_tp1",
        "tinyllmforge_tp4",
        "assemble",
        "verify",
    ]
    commands = plan["commands"]
    assert commands["official_tp1"]["argv"][0].endswith("/python")
    assert commands["tinyllmforge_tp1"]["argv"][0].endswith("/python")
    assert commands["tinyllmforge_tp4"]["argv"][0].endswith("/python")
    assert "--tensor-parallel-size=1" in (
        commands["tinyllmforge_tp1"]["argv"]
    )
    assert "--tensor-parallel-size=4" in (
        commands["tinyllmforge_tp4"]["argv"]
    )
    assert "--dist-port=32101" in commands["tinyllmforge_tp1"]["argv"]
    assert "--dist-port=32102" in commands["tinyllmforge_tp4"]["argv"]
    assert plan["rendezvous_ports"] == [32101, 32102]
    assert all(
        "torchrun" not in argument
        and not argument.startswith("--nproc-per-node")
        for name in ("tinyllmforge_tp1", "tinyllmforge_tp4")
        for argument in commands[name]["argv"]
    )

    for name in ("official_tp1", "tinyllmforge_tp1", "tinyllmforge_tp4"):
        argv = commands[name]["argv"]
        assert "--text-only" in argv
        assert "--greedy" in argv
        assert "--temperature=0" in argv
        assert "--generated-tokens=2" in argv
        assert "--prompt-token-ids=[11,22,33,44]" in argv
        assert "--disable-profiler" in argv
    assert commands["official_tp1"]["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert commands["tinyllmforge_tp1"]["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert commands["tinyllmforge_tp4"]["env"]["CUDA_VISIBLE_DEVICES"] == (
        "2,3,4,5"
    )


def test_plan_only_writes_below_approved_root_and_has_no_forbidden_actions(
    tmp_path,
    monkeypatch,
):
    plan = _plan(tmp_path, monkeypatch)
    approved = runner.APPROVED_REMOTE_ROOT
    attempt = Path(plan["attempt_root"])
    assert attempt.is_relative_to(approved)
    for path in plan["write_paths"]:
        assert Path(path).is_relative_to(attempt)

    serialized = json.dumps(plan, sort_keys=True)
    for forbidden in (
        "pkill",
        "killall",
        "kinit",
        "krenew",
        "adaptive-ngram",
        "/private/tmp",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "mutator,match",
    (
        (
            lambda kwargs, root: kwargs.update(
                attempt_root=root.parent / "outside"
            ),
            "approved remote root",
        ),
        (
            lambda kwargs, root: kwargs.update(
                rendezvous_ports=(32101, 32101)
            ),
            "distinct",
        ),
        (
            lambda kwargs, root: kwargs.update(gpu_indices=(0, 1, 2, 2)),
            "four distinct",
        ),
    ),
)
def test_plan_rejects_unsafe_scope_ports_or_gpu_identity(
    tmp_path,
    monkeypatch,
    mutator,
    match,
):
    approved = tmp_path / "approved"
    source = approved / "source"
    model = approved / "models" / "qwen38"
    source.mkdir(parents=True)
    model.mkdir(parents=True)
    manifest = model.parent / "model_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "APPROVED_REMOTE_ROOT", approved.resolve())
    kwargs = {
        "attempt_root": approved / "attempts" / "r1",
        "source_root": source,
        "model_root": model,
        "model_manifest_path": manifest,
        "source_tree_sha256": "1" * 64,
        "model_manifest_sha256": "2" * 64,
        "python_executable": approved / "env/bin/python",
        "torchrun_executable": approved / "env/bin/torchrun",
        "gpu_indices": (0, 1, 2, 3),
        "rendezvous_ports": (32101, 32102),
        "prompt_token_ids": (1, 2),
        "generated_tokens": 2,
        "topk": 8,
        "timeout_s": 900,
    }
    mutator(kwargs, approved)

    with pytest.raises(ValueError, match=match):
        runner.build_correctness_plan(**kwargs)


def test_executor_records_owned_processes_cleanup_and_verification(
    tmp_path,
    monkeypatch,
):
    plan = _plan(tmp_path, monkeypatch)
    calls = []

    def run_command(command, *, timeout_s):
        calls.append(command["name"])
        return {
            "returncode": 0,
            "pid": 1000 + len(calls),
            "pgid": 2000 + len(calls),
            "process_group_destroyed": True,
            "stdout": "",
            "stderr": "",
        }

    receipt = runner.execute_correctness_plan(
        plan,
        run_command=run_command,
        verify_bundle=lambda root: {"classification": "PASS"},
    )

    assert receipt["classification"] == "PASS"
    assert receipt["failed_stage"] is None
    assert calls == plan["command_order"][:-1]
    assert [row["pid"] for row in receipt["processes"]] == [
        1001,
        1002,
        1003,
        1004,
    ]
    assert all(
        row["process_group_destroyed"]
        for row in receipt["processes"]
    )
    assert receipt["owned_children_remaining"] == []
    receipt_path = Path(plan["attempt_root"]) / "runner_receipt.json"
    assert json.loads(receipt_path.read_text()) == receipt


def test_executor_returns_stage_specific_failure_receipt(
    tmp_path,
    monkeypatch,
):
    plan = _plan(tmp_path, monkeypatch)

    def run_command(command, *, timeout_s):
        return {
            "returncode": 7 if command["name"] == "tinyllmforge_tp1" else 0,
            "pid": 101,
            "pgid": 201,
            "process_group_destroyed": True,
            "stdout": "",
            "stderr": "injected failure",
        }

    receipt = runner.execute_correctness_plan(
        plan,
        run_command=run_command,
        verify_bundle=lambda root: {"classification": "PASS"},
    )

    assert receipt["classification"] == "FAIL"
    assert receipt["failed_stage"] == "tinyllmforge_tp1"
    assert receipt["verification"] is None
    assert receipt["owned_children_remaining"] == []


def test_executor_rejects_tampered_output_path_before_launch(
    tmp_path,
    monkeypatch,
):
    plan = _plan(tmp_path, monkeypatch)
    plan["commands"]["official_tp1"]["output_paths"][0] = str(
        runner.APPROVED_REMOTE_ROOT.parent / "escaped.jsonl"
    )
    calls = []

    with pytest.raises(ValueError, match="attempt_root"):
        runner.execute_correctness_plan(
            plan,
            run_command=lambda command, *, timeout_s: calls.append(command),
            verify_bundle=lambda root: {"classification": "PASS"},
        )

    assert calls == []


def test_executor_rejects_tampered_command_before_launch(
    tmp_path,
    monkeypatch,
):
    plan = _plan(tmp_path, monkeypatch)
    plan["commands"]["official_tp1"]["argv"] = [
        "/bin/sh",
        "-c",
        "true",
    ]
    calls = []

    with pytest.raises(ValueError, match="command executable"):
        runner.execute_correctness_plan(
            plan,
            run_command=lambda command, *, timeout_s: calls.append(command),
            verify_bundle=lambda root: {"classification": "PASS"},
        )

    assert calls == []


def test_executor_persists_failure_for_malformed_command_result(
    tmp_path,
    monkeypatch,
):
    plan = _plan(tmp_path, monkeypatch)

    receipt = runner.execute_correctness_plan(
        plan,
        run_command=lambda command, *, timeout_s: {
            "returncode": 0,
            "pid": 101,
            "pgid": 201,
            "process_group_destroyed": True,
            "owned_children_remaining": None,
        },
        verify_bundle=lambda root: {"classification": "PASS"},
    )

    assert receipt["classification"] == "FAIL"
    assert receipt["failed_stage"] == "official_tp1"
    assert "owned_children_remaining" in receipt["failure_reason"]
    assert json.loads(
        (
            Path(plan["attempt_root"]) / "runner_receipt.json"
        ).read_text()
    ) == receipt


class _FakeEngine:

    def __init__(self, *, tp_size, fail_decode=False):
        self.model_runner = SimpleNamespace(world_size=tp_size, rank=0)
        self.fail_decode = fail_decode
        self.recording_calls = []
        self.added_requests = []
        self.step_index = 0
        self.last_step_observation = None
        self.exited = False
        self._generated = [2, 1]
        self._logits = (
            torch.tensor([0.0, 1.0, 5.0, 2.0]),
            torch.tensor([0.0, 6.0, 1.0, 2.0]),
        )

    def enable_step_logits_authority_recording(
        self,
        enabled,
        *,
        timeout_s,
    ):
        self.recording_calls.append((enabled, timeout_s))
        return {
            "enabled": enabled,
            "rank_inventory": list(range(self.model_runner.world_size)),
        }

    def add_request(self, prompt, sampling_params):
        self.added_requests.append(
            (list(prompt), sampling_params.temperature,
             sampling_params.max_tokens, sampling_params.ignore_eos)
        )

    def is_finished(self):
        return self.step_index == len(self._generated)

    def step(self):
        if self.fail_decode:
            raise RuntimeError("decode failed")
        token_id = self._generated[self.step_index]
        self.step_index += 1
        self.last_step_observation = {
            "do_sample": True,
            "new_completion_tokens_by_seq": {0: [token_id]},
        }
        return [(0, self._generated[:self.step_index])], 1

    def read_step_logits_authority(self):
        return self._logits[self.step_index - 1].clone()

    def exit(self):
        self.exited = True
        world_size = self.model_runner.world_size
        return {
            "process_group_destroyed": True,
            "rank_exit_codes": [0] * world_size,
            "owned_children_remaining": [],
            "rank_cleanup_receipts": [
                {"rank": rank, "process_group_destroyed": True}
                for rank in range(world_size)
            ],
        }


def _rank_identities(tp_size):
    return [
        {
            "rank": rank,
            "gpu_index": 2 + rank,
            "gpu_uuid": f"GPU-{rank}",
            "expected_weight_shard_sha256": str(rank + 1) * 64,
            "loaded_weight_shard_sha256": str(rank + 1) * 64,
        }
        for rank in range(tp_size)
    ]


def test_tinyllmforge_worker_uses_engine_owned_tp_and_rank0_logits(
    tmp_path,
):
    attempt = tmp_path / "attempt"
    output = attempt / "rows" / "tinyllmforge_tp4.jsonl"
    process_output = attempt / "processes" / "tinyllmforge_tp4.json"
    (tmp_path / "model").mkdir()
    engine = _FakeEngine(tp_size=4)
    factory_calls = []

    def engine_factory(model_root, **kwargs):
        factory_calls.append((Path(model_root), kwargs))
        return engine

    receipt = runner.run_tinyllmforge_worker(
        mode="tinyllmforge_tp4",
        attempt_root=attempt,
        output_path=output,
        process_output_path=process_output,
        model_root=tmp_path / "model",
        model_repository="Qwen/Qwen3.8-27B",
        model_revision="a" * 40,
        source_tree_sha256="1" * 64,
        model_manifest_sha256="2" * 64,
        prompt_token_ids=(11, 22, 33),
        generated_tokens=2,
        topk=3,
        tensor_parallel_size=4,
        timeout_s=30.0,
        engine_factory=engine_factory,
        sampling_params_factory=lambda **kwargs: SimpleNamespace(**kwargs),
        rank_identity_reader=lambda value: _rank_identities(4),
        process_identity_reader=lambda: {"pid": 101, "pgid": 202},
    )

    assert factory_calls == [
        (
            (tmp_path / "model").resolve(),
            {
                "tensor_parallel_size": 4,
                "enforce_eager": True,
                "max_num_seqs": 1,
                "max_model_len": 5,
            },
        )
    ]
    assert engine.added_requests == [([11, 22, 33], 0.0, 2, True)]
    assert engine.recording_calls == [(True, 30.0), (False, 30.0)]
    assert engine.exited is True
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["rank"] for row in rows] == [0, 1, 2, 3]
    assert rows[0]["generated_token_ids"] == [2, 1]
    assert rows[0]["logits_authority"] == "rank0_full"
    assert rows[0]["positions"] == [
        {
            "position": 0,
            "topk_token_ids": [2, 3, 1],
            "topk_logits": [5.0, 2.0, 1.0],
        },
        {
            "position": 1,
            "topk_token_ids": [1, 3, 2],
            "topk_logits": [6.0, 2.0, 1.0],
        },
    ]
    for row in rows[1:]:
        assert row["generated_token_ids"] == [2, 1]
        assert row["logits_authority"] == (
            "unavailable_non_root_by_tp_design"
        )
        assert row["positions"] is None
        assert row["finite_logits"] is None
    assert receipt["classification"] == "PASS"
    assert receipt["rank_inventory"] == [0, 1, 2, 3]
    assert receipt["rank_exit_codes"] == [0, 0, 0, 0]
    assert receipt["process_group_destroyed"] is True
    assert receipt["owned_children_remaining"] == []
    assert json.loads(process_output.read_text()) == receipt
    assert not list(attempt.rglob("*.tmp"))


def test_tinyllmforge_worker_failure_atomically_records_cleanup(tmp_path):
    attempt = tmp_path / "attempt"
    output = attempt / "rows" / "tinyllmforge_tp4.jsonl"
    process_output = attempt / "processes" / "tinyllmforge_tp4.json"
    (tmp_path / "model").mkdir()
    engine = _FakeEngine(tp_size=4, fail_decode=True)

    with pytest.raises(RuntimeError, match="decode failed"):
        runner.run_tinyllmforge_worker(
            mode="tinyllmforge_tp4",
            attempt_root=attempt,
            output_path=output,
            process_output_path=process_output,
            model_root=tmp_path / "model",
            model_repository="Qwen/Qwen3.8-27B",
            model_revision="a" * 40,
            source_tree_sha256="1" * 64,
            model_manifest_sha256="2" * 64,
            prompt_token_ids=(11, 22, 33),
            generated_tokens=2,
            topk=3,
            tensor_parallel_size=4,
            timeout_s=30.0,
            engine_factory=lambda model_root, **kwargs: engine,
            sampling_params_factory=lambda **kwargs: SimpleNamespace(
                **kwargs
            ),
            rank_identity_reader=lambda value: _rank_identities(4),
            process_identity_reader=lambda: {"pid": 101, "pgid": 202},
        )

    receipt = json.loads(process_output.read_text())
    assert receipt["classification"] == "FAIL"
    assert receipt["failed_stage"] == "decode"
    assert receipt["failure_reason"] == "RuntimeError: decode failed"
    assert receipt["process_group_destroyed"] is True
    assert receipt["rank_exit_codes"] == [0, 0, 0, 0]
    assert receipt["owned_children_remaining"] == []
    assert engine.exited is True
    assert not output.exists()
    assert not list(attempt.rglob("*.tmp"))


def test_tinyllmforge_worker_rejects_writes_outside_attempt(tmp_path):
    attempt = tmp_path / "attempt"

    with pytest.raises(ValueError, match="attempt_root"):
        runner.run_tinyllmforge_worker(
            mode="tinyllmforge_tp1",
            attempt_root=attempt,
            output_path=tmp_path / "escaped.jsonl",
            process_output_path=attempt / "processes" / "tp1.json",
            model_root=tmp_path / "model",
            model_repository="Qwen/Qwen3.8-27B",
            model_revision="a" * 40,
            source_tree_sha256="1" * 64,
            model_manifest_sha256="2" * 64,
            prompt_token_ids=(11,),
            generated_tokens=1,
            topk=1,
            tensor_parallel_size=1,
            timeout_s=30.0,
            engine_factory=lambda model_root, **kwargs: object(),
            sampling_params_factory=lambda **kwargs: object(),
            rank_identity_reader=lambda value: _rank_identities(1),
            process_identity_reader=lambda: {"pid": 101, "pgid": 202},
        )


def test_engine_rank_identity_reader_collects_all_acknowledged_ranks():
    local = _rank_identities(4)[0]
    acknowledgements = [
        SimpleNamespace(rank=rank, result=_rank_identities(4)[rank])
        for rank in range(1, 4)
    ]

    class Engine:
        model_runner = SimpleNamespace(
            world_size=4,
            qwen38_correctness_rank_identity=lambda: local,
        )

        def call_model_runner_acknowledged(
            self,
            method_name,
            *,
            timeout_s,
        ):
            assert method_name == "qwen38_correctness_rank_identity"
            assert timeout_s == 45.0
            return local, acknowledgements

    rows = runner.read_engine_rank_identities(
        Engine(),
        timeout_s=45.0,
    )

    assert rows == _rank_identities(4)


def test_worker_cli_dispatches_real_tinyllmforge_adapter(
    tmp_path,
    monkeypatch,
):
    attempt = tmp_path / "attempt"
    model = tmp_path / "model"
    model.mkdir()
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text(
        json.dumps({
            "repository": "Qwen/Qwen3.8-27B",
            "resolved_revision": "a" * 40,
        }),
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    called = {}

    def run_worker(**kwargs):
        called.update(kwargs)
        return {"classification": "PASS"}

    monkeypatch.setattr(runner, "run_tinyllmforge_worker", run_worker)
    monkeypatch.setattr(
        runner,
        "_default_engine_factory",
        object(),
    )
    monkeypatch.setattr(
        runner,
        "_default_sampling_params_factory",
        object(),
    )
    monkeypatch.setattr(
        runner,
        "read_engine_rank_identities",
        object(),
    )
    monkeypatch.setattr(
        runner,
        "_read_process_identity",
        object(),
    )

    result = runner.main([
        "worker",
        "--mode=tinyllmforge_tp4",
        f"--attempt-root={attempt}",
        f"--output={attempt / 'rows/tp4.jsonl'}",
        f"--process-output={attempt / 'processes/tp4.json'}",
        f"--source-root={tmp_path}",
        f"--model-root={model}",
        f"--model-manifest={manifest}",
        f"--source-tree-sha256={'1' * 64}",
        f"--model-manifest-sha256={manifest_sha256}",
        "--text-only",
        "--greedy",
        "--temperature=0",
        "--generated-tokens=2",
        "--prompt-token-ids=[11,22,33]",
        "--topk=3",
        "--dtype=bfloat16",
        "--disable-profiler",
        "--tensor-parallel-size=4",
        "--dist-port=32102",
        "--timeout-s=30",
    ])

    assert result == 0
    assert called["mode"] == "tinyllmforge_tp4"
    assert called["attempt_root"] == attempt.resolve()
    assert called["model_root"] == model.resolve()
    assert called["model_repository"] == "Qwen/Qwen3.8-27B"
    assert called["model_revision"] == "a" * 40
    assert called["prompt_token_ids"] == (11, 22, 33)
    assert called["tensor_parallel_size"] == 4
    assert called["timeout_s"] == 30.0
    assert called["engine_factory"] is runner._default_engine_factory
    assert (
        called["rank_identity_reader"]
        is runner.read_engine_rank_identities
    )


def test_worker_cli_dispatches_real_official_adapter(
    tmp_path,
    monkeypatch,
):
    attempt = tmp_path / "attempt"
    source = tmp_path / "source"
    model = tmp_path / "model"
    source.mkdir()
    model.mkdir()
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text(
        json.dumps({
            "repository": "Qwen/Qwen3.8-27B",
            "resolved_revision": "a" * 40,
        }),
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    called = {}

    def run_worker(**kwargs):
        called.update(kwargs)
        return {"classification": "PASS"}

    monkeypatch.setattr(runner, "run_official_worker", run_worker)
    monkeypatch.setattr(
        runner,
        "_default_official_reference_factory",
        object(),
    )
    monkeypatch.setattr(
        runner,
        "_read_cuda_device_identity",
        object(),
    )
    monkeypatch.setattr(
        runner,
        "_read_process_identity",
        object(),
    )

    result = runner.main([
        "worker",
        "--mode=official_tp1",
        f"--attempt-root={attempt}",
        f"--output={attempt / 'rows/official_tp1.jsonl'}",
        f"--process-output={attempt / 'processes/official_tp1.json'}",
        f"--source-root={source}",
        f"--model-root={model}",
        f"--model-manifest={manifest}",
        f"--source-tree-sha256={'1' * 64}",
        f"--model-manifest-sha256={manifest_sha256}",
        "--text-only",
        "--greedy",
        "--temperature=0",
        "--generated-tokens=2",
        "--prompt-token-ids=[11,22,33]",
        "--topk=3",
        "--dtype=bfloat16",
        "--disable-profiler",
    ])

    assert result == 0
    assert called["attempt_root"] == attempt.resolve()
    assert called["model_root"] == model.resolve()
    assert called["model_repository"] == "Qwen/Qwen3.8-27B"
    assert called["model_revision"] == "a" * 40
    assert called["prompt_token_ids"] == (11, 22, 33)
    assert called["generated_tokens"] == 2
    assert called["topk"] == 3
    assert (
        called["reference_factory"]
        is runner._default_official_reference_factory
    )
    assert (
        called["device_identity_reader"]
        is runner._read_cuda_device_identity
    )
    assert (
        called["process_identity_reader"]
        is runner._read_process_identity
    )


class _FakeOfficialReference:

    def __init__(self):
        self.closed = False

    def generate_step_logits(
        self,
        *,
        prompt_token_ids,
        generated_tokens,
    ):
        assert prompt_token_ids == (11, 22, 33)
        assert generated_tokens == 2
        return {
            "generated_token_ids": [2, 1],
            "step_logits": [
                torch.tensor([0.0, 1.0, 5.0, 2.0]),
                torch.tensor([0.0, 6.0, 1.0, 2.0]),
            ],
        }

    def close(self):
        self.closed = True


def test_official_worker_records_step_logits_identity_and_cleanup(tmp_path):
    attempt = tmp_path / "attempt"
    output = attempt / "rows" / "official_tp1.jsonl"
    process_output = attempt / "processes" / "official_tp1.json"
    model = tmp_path / "model"
    model.mkdir()
    reference = _FakeOfficialReference()

    receipt = runner.run_official_worker(
        attempt_root=attempt,
        output_path=output,
        process_output_path=process_output,
        model_root=model,
        model_repository="Qwen/Qwen3.8-27B",
        model_revision="a" * 40,
        source_tree_sha256="1" * 64,
        model_manifest_sha256="2" * 64,
        prompt_token_ids=(11, 22, 33),
        generated_tokens=2,
        topk=3,
        reference_factory=lambda model_root: reference,
        device_identity_reader=lambda: {
            "gpu_index": 0,
            "gpu_uuid": "GPU-official",
        },
        process_identity_reader=lambda: {"pid": 101, "pgid": 202},
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["mode"] == "official_tp1"
    assert row["tp_size"] == 1
    assert row["rank"] == 0
    assert row["generated_token_ids"] == [2, 1]
    assert row["positions"] == [
        {
            "position": 0,
            "topk_token_ids": [2, 3, 1],
            "topk_logits": [5.0, 2.0, 1.0],
        },
        {
            "position": 1,
            "topk_token_ids": [1, 3, 2],
            "topk_logits": [6.0, 2.0, 1.0],
        },
    ]
    assert row["expected_weight_shard_sha256"] == "2" * 64
    assert row["loaded_weight_shard_sha256"] == "2" * 64
    assert receipt["classification"] == "PASS"
    assert receipt["rank_inventory"] == [0]
    assert receipt["process_group_destroyed"] is True
    assert receipt["rank_exit_codes"] == [0]
    assert receipt["owned_children_remaining"] == []
    assert reference.closed is True
    assert json.loads(process_output.read_text()) == receipt


def _run_fake_tiny_worker(
    *,
    attempt,
    mode,
    tensor_parallel_size,
):
    output = attempt / "rows" / f"{mode}.jsonl"
    process_output = attempt / "processes" / f"{mode}.json"
    engine = _FakeEngine(tp_size=tensor_parallel_size)
    runner.run_tinyllmforge_worker(
        mode=mode,
        attempt_root=attempt,
        output_path=output,
        process_output_path=process_output,
        model_root=attempt.parent / "model",
        model_repository="Qwen/Qwen3.8-27B",
        model_revision="a" * 40,
        source_tree_sha256="1" * 64,
        model_manifest_sha256="2" * 64,
        prompt_token_ids=(11, 22, 33),
        generated_tokens=2,
        topk=3,
        tensor_parallel_size=tensor_parallel_size,
        timeout_s=30.0,
        engine_factory=lambda model_root, **kwargs: engine,
        sampling_params_factory=lambda **kwargs: SimpleNamespace(**kwargs),
        rank_identity_reader=lambda value: _rank_identities(
            tensor_parallel_size
        ),
        process_identity_reader=lambda: {"pid": 101, "pgid": 202},
    )
    return output, process_output


def test_assembler_materializes_independently_verifiable_bundle(tmp_path):
    approved = tmp_path / "approved"
    attempt = approved / "attempt"
    model = approved / "model"
    model.mkdir(parents=True)
    monkeypatch_manifest = {
        "schema_version": verifier.MODEL_SCHEMA,
        "repository": "Qwen/Qwen3.8-27B",
        "resolved_revision": "a" * 40,
    }
    model_manifest = approved / "model_manifest.json"
    model_manifest.write_bytes(runner._canonical_bytes(monkeypatch_manifest))
    model_manifest_sha256 = hashlib.sha256(
        model_manifest.read_bytes()
    ).hexdigest()
    reference = _FakeOfficialReference()
    official_rows = attempt / "rows" / "official_tp1.jsonl"
    official_process = attempt / "processes" / "official_tp1.json"
    runner.run_official_worker(
        attempt_root=attempt,
        output_path=official_rows,
        process_output_path=official_process,
        model_root=model,
        model_repository="Qwen/Qwen3.8-27B",
        model_revision="a" * 40,
        source_tree_sha256="1" * 64,
        model_manifest_sha256=model_manifest_sha256,
        prompt_token_ids=(11, 22, 33),
        generated_tokens=2,
        topk=3,
        reference_factory=lambda model_root: reference,
        device_identity_reader=lambda: {
            "gpu_index": 0,
            "gpu_uuid": "GPU-official",
        },
        process_identity_reader=lambda: {"pid": 101, "pgid": 202},
    )
    tp1_rows, tp1_process = _run_fake_tiny_worker(
        attempt=attempt,
        mode="tinyllmforge_tp1",
        tensor_parallel_size=1,
    )
    tp4_rows, tp4_process = _run_fake_tiny_worker(
        attempt=attempt,
        mode="tinyllmforge_tp4",
        tensor_parallel_size=4,
    )
    for path in (tp1_rows, tp4_rows):
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        for row in rows:
            row["model_manifest_sha256"] = model_manifest_sha256
        path.write_bytes(
            b"".join(runner._canonical_bytes(row) for row in rows)
        )

    result = runner.assemble_correctness_bundle(
        attempt_root=attempt,
        model_manifest_path=model_manifest,
        official_rows_path=official_rows,
        tinyllmforge_tp1_rows_path=tp1_rows,
        tinyllmforge_tp4_rows_path=tp4_rows,
        official_process_path=official_process,
        tinyllmforge_tp1_process_path=tp1_process,
        tinyllmforge_tp4_process_path=tp4_process,
        source_tree_sha256="1" * 64,
        model_manifest_sha256=model_manifest_sha256,
        prompt_token_ids=(11, 22, 33),
        generated_tokens=2,
        topk=3,
        atol=0.02,
        rtol=0.01,
        bundle_verifier=verifier.validate_correctness_bundle,
    )

    assert result["classification"] == "PASS"
    assert (attempt / "model_manifest.json").read_bytes() == (
        model_manifest.read_bytes()
    )
    assert verifier.validate_correctness_bundle(attempt)[
        "classification"
    ] == "PASS"
    cleanup = json.loads(
        (attempt / "cleanup_receipt.json").read_text(encoding="utf-8")
    )
    assert cleanup["process_groups_destroyed"] == {
        "official_tp1": [0],
        "tinyllmforge_tp1": [0],
        "tinyllmforge_tp4": [0, 1, 2, 3],
    }


def test_assemble_cli_dispatches_all_source_bound_inputs(
    tmp_path,
    monkeypatch,
):
    attempt = tmp_path / "attempt"
    model_manifest = tmp_path / "model_manifest.json"
    official_rows = attempt / "rows" / "official_tp1.jsonl"
    tp1_rows = attempt / "rows" / "tinyllmforge_tp1.jsonl"
    tp4_rows = attempt / "rows" / "tinyllmforge_tp4.jsonl"
    official_process = attempt / "processes" / "official_tp1.json"
    tp1_process = attempt / "processes" / "tinyllmforge_tp1.json"
    tp4_process = attempt / "processes" / "tinyllmforge_tp4.json"
    called = {}

    def assemble(**kwargs):
        called.update(kwargs)
        return {"classification": "PASS"}

    monkeypatch.setattr(runner, "assemble_correctness_bundle", assemble)

    result = runner.main([
        "assemble",
        f"--attempt-root={attempt}",
        f"--model-manifest={model_manifest}",
        f"--official-rows={official_rows}",
        f"--tinyllmforge-tp1-rows={tp1_rows}",
        f"--tinyllmforge-tp4-rows={tp4_rows}",
        f"--official-process={official_process}",
        f"--tinyllmforge-tp1-process={tp1_process}",
        f"--tinyllmforge-tp4-process={tp4_process}",
        f"--source-tree-sha256={'1' * 64}",
        f"--model-manifest-sha256={'2' * 64}",
        "--prompt-token-ids=[11,22,33]",
        "--generated-tokens=2",
        "--topk=3",
    ])

    assert result == 0
    assert called == {
        "attempt_root": attempt.resolve(),
        "model_manifest_path": model_manifest.resolve(),
        "official_rows_path": official_rows.resolve(),
        "tinyllmforge_tp1_rows_path": tp1_rows.resolve(),
        "tinyllmforge_tp4_rows_path": tp4_rows.resolve(),
        "official_process_path": official_process.resolve(),
        "tinyllmforge_tp1_process_path": tp1_process.resolve(),
        "tinyllmforge_tp4_process_path": tp4_process.resolve(),
        "source_tree_sha256": "1" * 64,
        "model_manifest_sha256": "2" * 64,
        "prompt_token_ids": (11, 22, 33),
        "generated_tokens": 2,
        "topk": 3,
    }
