from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name, filename):
    path = TOOLS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stage_runner = _load(
    "qwen35_tp4_controlled_shared_root_stage_runner",
    "qwen35_tp4_controlled_shared_root_stage_runner.py",
)


def _plan(root):
    baseline = root / "resource_baseline.json"
    baseline.write_text(json.dumps({
        "schema_version": (
            "qwen35.tp4-controlled-shared-resource-baseline.v1"
        ),
        "classification": "READY",
        "ssh_target": "sitian@10.232.195.203",
        "captured_at": "2026-07-29T15:00:00+08:00",
        "gpu_indices": [2, 4, 5, 6],
        "selected": [
            {
                "gpu_index": index,
                "gpu_uuid": uuid,
                "free_bytes": 64 * 1024**3,
                "compute_processes": [{
                    "pid": 1000 + index,
                    "process_name": "python3",
                    "used_memory_mib": 400,
                    "start_time_ticks": 2000 + index,
                }],
            }
            for index, uuid in zip(
                [2, 4, 5, 6],
                ["GPU-2", "GPU-4", "GPU-5", "GPU-6"],
            )
        ],
        "minimum_free_bytes_per_gpu": 24 * 1024**3,
        "benchmark_execution_authorized": False,
    }, sort_keys=True, separators=(",", ":")) + "\n")
    plan = {
        "resource_policy": "controlled_shared",
        "resource_baseline_path": str(baseline),
        "resource_baseline_sha256": hashlib.sha256(
            baseline.read_bytes()
        ).hexdigest(),
        "gpu_indices": [2, 4, 5, 6],
        "gpu_uuids": ["GPU-2", "GPU-4", "GPU-5", "GPU-6"],
        "run_tag": "root-shared-r1",
        "repo_root": str(root / "repo"),
        "local_run_dir": str(root / "repo/experiments/root-shared-r1"),
        "remote_run_dir": "/remote/root-shared-r1",
        "frozen_source_tree_sha256": "b" * 64,
        "frozen_source_tag": "frozen-r1",
        "exact_artifact_names": ["a", "b"],
    }
    return plan


def _guard(plan):
    return {
        "classification": "READY",
        "resource_policy": "controlled_shared",
        "baseline_sha256": plan["resource_baseline_sha256"],
        "benchmark_execution_authorized": False,
        "selected": [
            {
                "gpu_index": index,
                "gpu_uuid": uuid,
                "free_bytes": 32 * 1024**3,
                "compute_processes": [{
                    "pid": 1000 + index,
                    "process_name": "python3",
                    "used_memory_mib": 400,
                    "start_time_ticks": 2000 + index,
                }],
            }
            for index, uuid in zip(
                plan["gpu_indices"],
                plan["gpu_uuids"],
            )
        ],
    }


def test_preflight_uploads_once_guards_and_persists_real_processes():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = _plan(root)
        events = []

        def command_runner(command, **kwargs):
            events.append(command)
            if command[0] == "scp":
                return SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr="",
                )
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(_guard(plan)),
                stderr="",
            )

        result = stage_runner.run_stage(
            name="preflight",
            plan=plan,
            execution_env={
                "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian"
            },
            command_runner=command_runner,
        )
        assert [command[0] for command in events] == ["ssh", "scp", "ssh"]
        upload_command = events[1]
        assert "ServerAliveInterval=30" in upload_command
        assert "ServerAliveCountMax=3" in upload_command
        assert result["status"] == "READY"
        assert result["resource_policy"] == "controlled_shared"
        assert result["selected"][0]["compute_processes"][0][
            "pid"
        ] == 1002
        assert Path(plan["local_run_dir"]).joinpath(
            "remote_resource_preflight.json"
        ).is_file()


def test_preflight_retries_transient_baseline_upload_failure():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = _plan(root)
        events = []
        upload_attempts = 0

        def command_runner(command, **kwargs):
            nonlocal upload_attempts
            events.append(command)
            if command[0] == "scp":
                upload_attempts += 1
                if upload_attempts == 1:
                    return SimpleNamespace(
                        returncode=255,
                        stdout="",
                        stderr="Connection closed by UNKNOWN port 65535",
                    )
                return SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr="",
                )
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(_guard(plan)),
                stderr="",
            )

        result = stage_runner.run_stage(
            name="preflight",
            plan=plan,
            execution_env={
                "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian"
            },
            command_runner=command_runner,
        )
        assert [command[0] for command in events] == [
            "ssh",
            "scp",
            "scp",
            "ssh",
        ]
        assert result["status"] == "READY"


def test_preflight_retries_transient_remote_reservation_failure():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = _plan(root)
        events = []
        reservation_attempts = 0

        def command_runner(command, **kwargs):
            nonlocal reservation_attempts
            events.append(command)
            if command[0] == "scp":
                return SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr="",
                )
            if reservation_attempts < 2:
                reservation_attempts += 1
                return SimpleNamespace(
                    returncode=255,
                    stdout="",
                    stderr="Connection closed by UNKNOWN port 65535",
                )
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(_guard(plan)),
                stderr="",
            )

        result = stage_runner.run_stage(
            name="preflight",
            plan=plan,
            execution_env={
                "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian"
            },
            command_runner=command_runner,
        )

        assert [command[0] for command in events] == [
            "ssh",
            "ssh",
            "ssh",
            "scp",
            "ssh",
        ]
        assert result["status"] == "READY"


def test_default_command_runner_injects_exact_kerberos_environment():
    calls = []
    original = stage_runner.subprocess.run

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    stage_runner.subprocess.run = fake_run
    try:
        stage_runner._run(["ssh", "sitian@10.232.195.203", "true"])
    finally:
        stage_runner.subprocess.run = original

    assert calls[0][1]["env"]["KRB5CCNAME"] == (
        "FILE:/Users/bytedance/krb5cc_sitian"
    )


def test_preflight_retries_transient_resource_guard_failure():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = _plan(root)
        events = []
        guard_attempts = 0

        def command_runner(command, **kwargs):
            nonlocal guard_attempts
            events.append(command)
            if command[0] == "scp":
                return SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr="",
                )
            if len(events) == 1:
                return SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr="",
                )
            guard_attempts += 1
            if guard_attempts == 1:
                return SimpleNamespace(
                    returncode=255,
                    stdout="",
                    stderr="Connection closed by UNKNOWN port 65535",
                )
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(_guard(plan)),
                stderr="",
            )

        result = stage_runner.run_stage(
            name="preflight",
            plan=plan,
            execution_env={
                "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian"
            },
            command_runner=command_runner,
        )

        assert [command[0] for command in events] == [
            "ssh",
            "scp",
            "ssh",
            "ssh",
        ]
        assert result["status"] == "READY"


def test_run_uses_remote_wrapper_then_final_guard():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = _plan(root)
        local_run = Path(plan["local_run_dir"])
        local_run.mkdir(parents=True)
        local_run.joinpath("remote_resource_preflight.json").write_text(
            json.dumps({
                "status": "READY",
                "selected": _guard(plan)["selected"],
            })
        )
        events = []

        def command_runner(command, **kwargs):
            events.append(command)
            if len(events) == 2:
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps({
                        "classification": "PASS",
                        "artifact_names": ["a", "b"],
                    }),
                    stderr="",
                )
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(_guard(plan)),
                stderr="",
            )

        result = stage_runner.run_stage(
            name="run",
            plan=plan,
            execution_env={
                "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian"
            },
            command_runner=command_runner,
        )
        assert len(events) == 3
        assert all(command[0] == "ssh" for command in events)
        wrapper_text = events[1][-1]
        assert "execute_source_bound_run" in wrapper_text
        assert "process_factory_builder" in wrapper_text
        assert "query_gpus" in wrapper_text
        assert result["status"] == "REMOTE_PASS"
        assert result["final_resource"]["resource_policy"] == (
            "controlled_shared"
        )


def test_wrapper_injects_frozen_deferred_subprocess_popen_dependency():
    with tempfile.TemporaryDirectory() as temporary:
        plan = _plan(Path(temporary))
        wrapper = stage_runner._wrapper_script(plan, _guard(plan))
        assert "import importlib.util,json,os,subprocess,sys" in wrapper
        assert "popen=subprocess.Popen" in wrapper


def test_wrapper_persists_native_rank_failure_diagnostics():
    with tempfile.TemporaryDirectory() as temporary:
        plan = _plan(Path(temporary))
        wrapper = stage_runner._wrapper_script(plan, _guard(plan))
        assert "traceback.format_exc()" in wrapper
        assert ".native-rank-failure-" in wrapper
        assert "native rank diagnostics:" in wrapper


def test_wrapper_persists_non_pass_comparison_diagnostics():
    with tempfile.TemporaryDirectory() as temporary:
        plan = _plan(Path(temporary))
        wrapper = stage_runner._wrapper_script(plan, _guard(plan))
        assert ".comparison-diagnostics.json" in wrapper
        assert "original_compare_logits" in wrapper
        assert "original_classify_rows" in wrapper
        assert "'comparisons':comparison_rows" in wrapper
        assert "'classification':classification" in wrapper
        assert "comparison diagnostics:" in wrapper


def test_wrapper_adapts_keyword_only_candidate_builder():
    with tempfile.TemporaryDirectory() as temporary:
        plan = _plan(Path(temporary))
        wrapper = stage_runner._wrapper_script(plan, _guard(plan))
        assert (
            "def build_candidate(rank): return "
            "module.build_real_tp4_cpu_candidate(rank=rank)"
        ) in wrapper
        assert (
            "build_candidate=build_candidate,run_cases=run_cases,"
            "query_gpus=query_gpus"
        ) in wrapper


def test_wrapper_canonicalizes_only_rank_zero_lm_head_logits():
    with tempfile.TemporaryDirectory() as temporary:
        plan = _plan(Path(temporary))
        wrapper = stage_runner._wrapper_script(plan, _guard(plan))
        assert (
            "if kwargs['rank']!=0: return canonicalize_results("
            "module.run_tp4_native_cases(**kwargs))"
        ) in wrapper
        assert (
            "return logits.to(dtype=module.torch.float32)"
        ) in wrapper
        assert (
            "logical_state={f'{logical_layer}:{role}'"
        ) in wrapper
        assert "finally: hook.remove()" in wrapper


def test_root_run_cases_adapter_canonicalizes_lm_head_logits_to_float32():
    @dataclass(frozen=True)
    class FakeResult:
        logits: object
        state_nonzero_after_commit: dict[str, bool]

    class FakeTensor:
        def __init__(self, dtype):
            self.dtype = dtype

        def to(self, *, dtype):
            return FakeTensor(dtype)

    class FakeHook:
        def __init__(self, owner, hook_id):
            self.owner = owner
            self.hook_id = hook_id

        def remove(self):
            self.owner._forward_hooks.pop(self.hook_id)

    class FakeLMHead:
        def __init__(self):
            self._forward_hooks = {}

        def register_forward_hook(self, hook):
            self._forward_hooks[1] = hook
            return FakeHook(self, 1)

        def forward(self, value):
            output = value.to(dtype="bfloat16")
            for hook in tuple(self._forward_hooks.values()):
                output = hook(self, (value,), output)
            return output

        __call__ = forward

    lm_head = FakeLMHead()
    candidate = SimpleNamespace(
        owner=SimpleNamespace(
            model=SimpleNamespace(lm_head=lm_head),
        ),
    )

    class FrozenModule:
        torch = SimpleNamespace(float32="float32")

        @staticmethod
        def run_tp4_native_cases(**kwargs):
            logits = kwargs["candidate"].owner.model.lm_head(
                FakeTensor("float32")
            )
            return (FakeResult(
                logits=logits,
                state_nonzero_after_commit={
                    f"{layer}:{role}": True
                    for layer in (
                        0, 1, 2, 4, 5, 6, 8, 9, 10,
                        12, 13, 14, 16, 17, 18, 20, 21, 22,
                    )
                    for role in (
                        "linear_convolution",
                        "linear_recurrent",
                    )
                },
            ),)

    result = stage_runner._run_cases_with_float32_root_logits(
        module=FrozenModule,
        candidate=candidate,
        rank=0,
    )

    assert result[0].logits.dtype == "float32"
    assert not lm_head._forward_hooks


def test_root_run_cases_adapter_removes_hook_after_failure():
    class FakeHook:
        def __init__(self, owner):
            self.owner = owner

        def remove(self):
            self.owner._forward_hooks.clear()

    class FakeLMHead:
        def __init__(self):
            self._forward_hooks = {}

        def register_forward_hook(self, hook):
            self._forward_hooks[1] = hook
            return FakeHook(self)

    lm_head = FakeLMHead()
    candidate = SimpleNamespace(
        owner=SimpleNamespace(
            model=SimpleNamespace(lm_head=lm_head),
        ),
    )

    class FrozenModule:
        torch = SimpleNamespace(float32="float32")

        @staticmethod
        def run_tp4_native_cases(**_kwargs):
            raise RuntimeError("expected failure")

    try:
        stage_runner._run_cases_with_float32_root_logits(
            module=FrozenModule,
            candidate=candidate,
            rank=0,
        )
    except RuntimeError as error:
        assert str(error) == "expected failure"
    else:
        raise AssertionError("expected adapter failure")

    assert not lm_head._forward_hooks


def test_run_cases_adapter_canonicalizes_sparse_physical_layer_evidence():
    @dataclass(frozen=True)
    class FakeResult:
        state_nonzero_after_commit: dict[str, bool]

    physical_layers = (
        0, 1, 2, 4, 5, 6, 8, 9, 10,
        12, 13, 14, 16, 17, 18, 20, 21, 22,
    )
    state = {
        f"{layer}:{role}": True
        for layer in physical_layers
        for role in ("linear_convolution", "linear_recurrent")
    }
    candidate = SimpleNamespace(
        owner=SimpleNamespace(
            model=SimpleNamespace(lm_head=None),
        ),
    )

    class FrozenModule:
        @staticmethod
        def run_tp4_native_cases(**_kwargs):
            return (FakeResult(state),)

    result = stage_runner._run_cases_with_float32_root_logits(
        module=FrozenModule,
        candidate=candidate,
        rank=1,
    )

    assert set(result[0].state_nonzero_after_commit) == {
        f"{layer}:{role}"
        for layer in range(18)
        for role in ("linear_convolution", "linear_recurrent")
    }
    assert all(result[0].state_nonzero_after_commit.values())
    assert set(state) == {
        f"{layer}:{role}"
        for layer in physical_layers
        for role in ("linear_convolution", "linear_recurrent")
    }


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 controlled-shared root stage runner tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
