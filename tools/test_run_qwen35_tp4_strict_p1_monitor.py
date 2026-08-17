from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import signal
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
ROLLOVER = TOOLS / "run_qwen35_tp4_strict_p1_monitor_rollover.sh"


def _load_runner():
    path = TOOLS / "run_qwen35_tp4_strict_p1_monitor.py"
    spec = importlib.util.spec_from_file_location(
        "run_qwen35_tp4_strict_p1_monitor",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_uses_frozen_schema_resource_threshold():
    runner = _load_runner()
    assert runner.GPU_INDICES == (2, 4, 5, 6)
    assert runner.MINIMUM_FREE_BYTES == 25769803776
    assert runner.WORKERS_TIMEOUT_S == 12 * 60 * 60
    assert (
        runner.MINIMUM_FREE_BYTES
        == runner.benchmark.MIN_GPU_FREE_BYTES
    )


def test_benchmark_remote_query_returns_canonical_gpu_payload():
    runner = _load_runner()
    rows = [
        {
            "gpu_index": index,
            "gpu_uuid": f"GPU-{index}",
            "free_bytes": runner.MINIMUM_FREE_BYTES,
            "compute_processes": [
                {
                    "pid": 1000 + index,
                    "process_name": "unrelated-service",
                    "used_bytes": 1024**3,
                },
            ],
            "utilization_percent": runner.MAXIMUM_GPU_UTILIZATION_PERCENT,
        }
        for index in runner.GPU_INDICES
    ]

    def command_runner(**kwargs):
        return {
            "returncode": 0,
            "stdout": json.dumps({"gpus": rows}),
            "stderr": "",
        }

    payload = runner._benchmark_remote_query(command_runner)
    assert payload == {"gpus": rows}


def test_shared_gpu_selector_allows_processes_at_utilization_limit():
    runner = _load_runner()
    rows = [
        {
            "gpu_index": index,
            "gpu_uuid": f"GPU-{index}",
            "free_bytes": runner.MINIMUM_FREE_BYTES,
            "compute_processes": [
                {
                    "pid": 1000 + index,
                    "process_name": "unrelated-service",
                    "used_bytes": 1024**3,
                },
            ],
            "utilization_percent": runner.MAXIMUM_GPU_UTILIZATION_PERCENT,
        }
        for index in runner.GPU_INDICES
    ]

    selected = runner._select_shared_tp4_gpu_resources(rows)

    assert [row["gpu_index"] for row in selected] == list(
        runner.GPU_INDICES
    )


def test_shared_gpu_selector_rejects_utilization_above_limit():
    runner = _load_runner()
    rows = [
        {
            "gpu_index": index,
            "gpu_uuid": f"GPU-{index}",
            "free_bytes": runner.MINIMUM_FREE_BYTES,
            "compute_processes": [],
            "utilization_percent": (
                runner.MAXIMUM_GPU_UTILIZATION_PERCENT
                + (1 if index == 6 else 0)
            ),
        }
        for index in runner.GPU_INDICES
    ]

    try:
        runner._select_shared_tp4_gpu_resources(rows)
    except ValueError as error:
        assert "utilization" in str(error)
    else:
        raise AssertionError("over-utilized fixed GPU was admitted")


def test_scoped_cleanup_script_requires_both_run_tag_and_remote_root():
    runner = _load_runner()
    script = runner._cleanup_script("strict-p1-run-r1")
    assert "strict-p1-run-r1" in script
    assert runner.REMOTE_ROOT in script
    assert "run_tag in row" in script
    assert "remote_root in row" in script
    assert "cmdline" in script
    assert "root_pids" in script
    assert "target_pids" in script
    assert "descendants" in script
    assert "target_start_times" in script
    assert "killall" not in script
    assert "pkill" not in script
    assert "nvidia-smi --gpu-reset" not in script
    assert "--query-compute-apps=pid" in script
    assert "'gpu_process_pids'" in script
    assert "'root_pids'" in script
    assert "'target_pids'" in script
    assert "'remaining_target_pids'" in script
    assert "'matched_gpu_pids_after_cleanup'" in script


def test_termination_signal_is_converted_to_system_exit():
    runner = _load_runner()
    for signal_number in (signal.SIGTERM, signal.SIGINT):
        try:
            runner._raise_signal_exit(signal_number, None)
        except SystemExit as error:
            assert error.code == 128 + signal_number
        else:
            raise AssertionError("termination signal was not converted")


def test_main_installs_cleanup_compatible_signal_handlers():
    runner = _load_runner()
    calls = []
    original = runner.signal.signal
    runner.signal.signal = lambda number, handler: calls.append(
        (number, handler)
    )
    try:
        runner._install_signal_handlers()
    finally:
        runner.signal.signal = original
    assert calls == [
        (signal.SIGTERM, runner._raise_signal_exit),
        (signal.SIGINT, runner._raise_signal_exit),
    ]


def test_main_launches_benchmark_with_shared_resource_policy():
    runner = _load_runner()
    calls = []
    launch_results = []
    original_install = runner._install_signal_handlers
    original_command_runner = (
        runner.transport.controlmaster_command_runner
    )
    original_monitor = runner.monitor.monitor_until_launch
    original_execute = runner.benchmark.execute_benchmark_launch
    original_cleanup = runner._cleanup_run
    runner._install_signal_handlers = lambda: None
    runner.transport.controlmaster_command_runner = (
        lambda **kwargs: object()
    )
    runner.benchmark.execute_benchmark_launch = (
        lambda **kwargs: (
            calls.append(kwargs)
            or {"classification": "PASS"}
        )
    )
    runner._cleanup_run = lambda *args: {"classification": "CLEAN"}

    def fake_monitor_until_launch(**kwargs):
        launch_result = kwargs["launch_fn"]()
        launch_results.append(launch_result)
        return {
            "classification": launch_result["classification"],
        }

    runner.monitor.monitor_until_launch = fake_monitor_until_launch
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = runner.main([
                "--monitor-tag",
                "shared-policy-monitor",
                "--run-tag",
                "shared-policy-run",
                "--control-path",
                str(root / "control"),
                "--prerequisites",
                str(root / "prerequisites.json"),
                "--local-model-manifest",
                str(root / "model_manifest.json"),
                "--remote-model-dir",
                "/remote/model",
                "--remote-model-manifest",
                "/remote/model_manifest.json",
                "--output-root",
                str(root / "output"),
            ])
    finally:
        runner._install_signal_handlers = original_install
        runner.transport.controlmaster_command_runner = (
            original_command_runner
        )
        runner.monitor.monitor_until_launch = original_monitor
        runner.benchmark.execute_benchmark_launch = original_execute
        runner._cleanup_run = original_cleanup

    assert result == 0
    assert len(calls) == 1
    assert calls[0]["resource_policy"] == (
        runner.benchmark.SHARED_LOW_UTILIZATION_RESOURCE_POLICY
    )
    assert calls[0]["maximum_gpu_utilization_percent"] == (
        runner.MAXIMUM_GPU_UTILIZATION_PERCENT
    )
    assert launch_results == [{
        "classification": "PASS",
        "resource_sharing_policy": "shared-low-utilization",
        "performance_claim_boundary": (
            "non-exclusive shared-GPU observation; not an "
            "uncontended strict-P1 performance baseline"
        ),
    }]


def test_rollover_owns_controlmaster_with_fixed_kerberos_cache():
    script = ROLLOVER.read_text(encoding="utf-8")
    assert (
        "export KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian"
        in script
    )
    assert "CONTROL_PATH=/tmp/ssh-sitian-10.232.195.203" in script
    assert 'ssh -S "$CONTROL_PATH" -O check' in script
    assert 'ssh -MN -S "$CONTROL_PATH"' in script
    assert "ControlMaster=yes" in script
    assert "ControlPersist=7d" in script
    assert "--control-path \"$CONTROL_PATH\"" in script
    assert "MONITOR_OUTPUT=" in script
    assert "monitor_result.json" in script
    assert "monitor_failure.json" in script
    assert "exit \"$monitor_exit\"" in script


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 strict-P1 live monitor tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
