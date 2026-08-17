from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT / "tools" / "run_spec_verify_cuda_graph_gate_remote.py"
)


def _load_runner():
    assert RUNNER_PATH.is_file(), (
        f"missing remote runner: {RUNNER_PATH}"
    )
    spec = importlib.util.spec_from_file_location(
        "run_spec_verify_cuda_graph_gate_remote_test_module",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_ssh_command_uses_sitian_control_socket():
    runner = _load_runner()

    command = runner.build_ssh_command([
        "printf",
        "%s",
        "hello world",
    ])

    assert command[:2] == ["ssh", "-S"]
    assert runner.CONTROL_SOCKET in command
    assert runner.REMOTE_TARGET in command
    assert command[-1] == "printf %s 'hello world'"


def test_classify_preflight_selects_existing_checkpoint_and_gpu():
    runner = _load_runner()
    payload = {
        "python_exists": True,
        "repo_exists": True,
        "cuda_available": True,
        "torch_version": "2.4.1",
        "cuda_version": "12.4",
        "device_count": 2,
        "devices": [
            {
                "index": 0,
                "name": "A100",
                "compute_capability": [8, 0],
            },
            {
                "index": 1,
                "name": "A100",
                "compute_capability": [8, 0],
            },
        ],
        "idle_gpu_indices": [1],
        "checkpoint_candidates": [
            {
                "path": "/missing",
                "exists": False,
            },
            {
                "path": "/models/Qwen3-0.6B",
                "exists": True,
            },
        ],
    }

    result = runner.classify_preflight_payload(payload)

    assert result["status"] == "READY"
    assert result["gpu_index"] == 1
    assert result["checkpoint"] == "/models/Qwen3-0.6B"


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("python_exists", False, "python"),
        ("repo_exists", False, "repo"),
        ("cuda_available", False, "CUDA"),
        ("idle_gpu_indices", [], "idle GPU"),
        ("checkpoint_candidates", [], "checkpoint"),
    ),
)
def test_classify_preflight_fails_closed(
    field,
    value,
    match,
):
    runner = _load_runner()
    payload = {
        "python_exists": True,
        "repo_exists": True,
        "cuda_available": True,
        "torch_version": "2.4.1",
        "cuda_version": "12.4",
        "device_count": 1,
        "devices": [{
            "index": 0,
            "name": "A100",
            "compute_capability": [8, 0],
        }],
        "idle_gpu_indices": [0],
        "checkpoint_candidates": [{
            "path": "/models/Qwen3-0.6B",
            "exists": True,
        }],
    }
    payload[field] = value

    with pytest.raises(ValueError, match=match):
        runner.classify_preflight_payload(payload)


def test_build_remote_smoke_command_preserves_exact_matrix():
    runner = _load_runner()

    command = runner.build_remote_smoke_command(
        remote_source_root="/remote/source",
        checkpoint="/models/Qwen3-0.6B",
        remote_artifact="/remote/result.json",
        context_length=4096,
        batch_sizes=(1, 4),
        query_lengths=(1, 3),
        page_table_widths=(1, 2),
        gpu_index=2,
    )

    assert command[:3] == [
        "env",
        "CUDA_VISIBLE_DEVICES=2",
        "PYTHONPATH=/remote/source",
    ]
    assert "--context-length" in command
    assert command[command.index("--context-length") + 1] == "4096"
    assert command[
        command.index("--batch-sizes") + 1:
        command.index("--query-lengths")
    ] == ["1", "4"]
    assert command[
        command.index("--query-lengths") + 1:
        command.index("--page-table-widths")
    ] == ["1", "3"]
    assert command[
        command.index("--page-table-widths") + 1:
    ] == ["1", "2"]


def test_build_remote_smoke_command_enables_performance_mode():
    runner = _load_runner()

    command = runner.build_remote_smoke_command(
        remote_source_root="/remote/source",
        checkpoint="/models/Qwen3-0.6B",
        remote_artifact="/remote/result.json",
        context_length=4096,
        batch_sizes=(1, 4),
        query_lengths=(1, 3),
        page_table_widths=(1, 2),
        gpu_index=2,
        measure_performance=True,
    )

    assert command.count("--measure-performance") == 1


def test_build_remote_preflight_command_checks_live_resources():
    runner = _load_runner()

    command = runner.build_remote_preflight_command()

    assert command[:2] == [runner.REMOTE_PYTHON, "-c"]
    script = command[2]
    assert runner.REMOTE_REPO in script
    assert all(
        candidate in script
        for candidate in runner.CHECKPOINT_CANDIDATES
    )
    assert "torch.cuda.is_available()" in script
    assert "--query-compute-apps=gpu_uuid" in script


def test_build_scp_command_reuses_control_socket(tmp_path):
    runner = _load_runner()
    local_path = tmp_path / "source.tar"

    command = runner.build_scp_command(
        local_path,
        "/remote/source.tar",
    )

    assert command[:3] == ["scp", "-o", (
        f"ControlPath={runner.CONTROL_SOCKET}"
    )]
    assert command[-2:] == [
        str(local_path),
        f"{runner.REMOTE_TARGET}:/remote/source.tar",
    ]


def test_build_source_archive_contains_runtime_tree(tmp_path):
    runner = _load_runner()
    repo_root = tmp_path / "repo"
    (repo_root / "tinyvllm").mkdir(parents=True)
    (repo_root / "tools").mkdir()
    (repo_root / "tinyvllm" / "__init__.py").write_text("")
    (repo_root / "tinyvllm" / "config.py").write_text("CONFIG = 1\n")
    (repo_root / "tinyvllm" / "__pycache__").mkdir()
    (repo_root / "tinyvllm" / "__pycache__" / "x.pyc").write_bytes(
        b"cache"
    )
    for filename in runner.SOURCE_TOOL_FILES:
        (repo_root / "tools" / filename).write_text(
            f"# {filename}\n"
        )
    archive_path = tmp_path / "source.tar"

    runner.build_source_archive(repo_root, archive_path)

    with tarfile.open(archive_path, "r:") as archive:
        names = set(archive.getnames())
    assert "source/tinyvllm/__init__.py" in names
    assert "source/tinyvllm/config.py" in names
    assert all(
        f"source/tools/{filename}" in names
        for filename in runner.SOURCE_TOOL_FILES
    )
    assert not any("__pycache__" in name for name in names)
    assert not any(name.endswith(".pyc") for name in names)


def test_execute_gate_orders_fail_closed_stages(tmp_path):
    runner = _load_runner()
    events = []
    output_path = tmp_path / "artifact.json"
    preflight = {
        "status": "READY",
        "gpu_index": 1,
        "checkpoint": "/models/Qwen3-0.6B",
    }

    result = runner.execute_gate(
        output_path=output_path,
        context_length=4096,
        batch_sizes=(1, 4),
        query_lengths=(1, 3),
        page_table_widths=(1, 2),
        preflight_executor=lambda: (
            events.append("preflight") or preflight
        ),
        upload_executor=lambda **kwargs: (
            events.append(("upload", kwargs)) or {
                "remote_source_root": "/remote/source",
                "remote_artifact": "/remote/artifact.json",
            }
        ),
        smoke_executor=lambda **kwargs: (
            events.append(("run", kwargs))
        ),
        download_executor=lambda **kwargs: (
            events.append(("download", kwargs))
            or output_path.write_text("{}\n")
        ),
        verify_executor=lambda **kwargs: (
            events.append(("verify", kwargs))
            or {"status": "PASS"}
        ),
    )

    assert [event if isinstance(event, str) else event[0]
            for event in events] == [
        "preflight",
        "upload",
        "run",
        "download",
        "verify",
    ]
    assert result["status"] == "PASS"
    assert result["preflight"] == preflight


def test_execute_gate_stops_after_blocked_preflight(tmp_path):
    runner = _load_runner()
    events = []

    def blocked():
        events.append("preflight")
        raise ValueError("no idle GPU is available")

    with pytest.raises(ValueError, match="idle GPU"):
        runner.execute_gate(
            output_path=tmp_path / "artifact.json",
            context_length=4096,
            batch_sizes=(1, 4),
            query_lengths=(1, 3),
            page_table_widths=(1, 2),
            preflight_executor=blocked,
            upload_executor=lambda **kwargs: events.append("upload"),
            smoke_executor=lambda **kwargs: events.append("run"),
            download_executor=lambda **kwargs: events.append("download"),
            verify_executor=lambda **kwargs: events.append("verify"),
        )

    assert events == ["preflight"]


def test_normalize_gate_configuration_accepts_only_mvp_matrix():
    runner = _load_runner()

    result = runner.normalize_gate_configuration(
        context_length=4096,
        batch_sizes=(1, 4),
        query_lengths=(1, 3),
        page_table_widths=(1, 2),
    )

    assert result == {
        "context_length": 4096,
        "batch_sizes": (1, 4),
        "query_lengths": (1, 3),
        "page_table_widths": (1, 2),
    }


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("context_length", 8192, "context_length"),
        ("batch_sizes", (1,), "batch_sizes"),
        ("query_lengths", (1, 2, 3), "query_lengths"),
        ("page_table_widths", (1,), "page_table_widths"),
    ),
)
def test_normalize_gate_configuration_rejects_scope_expansion(
    field,
    value,
    match,
):
    runner = _load_runner()
    arguments = {
        "context_length": 4096,
        "batch_sizes": (1, 4),
        "query_lengths": (1, 3),
        "page_table_widths": (1, 2),
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=match):
        runner.normalize_gate_configuration(**arguments)


def test_build_preflight_record_preserves_blocker_payload():
    runner = _load_runner()
    payload = {
        "python_exists": True,
        "repo_exists": True,
        "cuda_available": True,
        "torch_version": "2.4.1+cu121",
        "cuda_version": "12.1",
        "device_count": 1,
        "devices": [{
            "index": 0,
            "name": "A100",
            "compute_capability": [8, 0],
        }],
        "idle_gpu_indices": [],
        "checkpoint_candidates": [{
            "path": "/models/Qwen3-0.6B",
            "exists": True,
        }],
    }

    record = runner.build_preflight_record(payload)

    assert record["schema_version"] == 1
    assert record["status"] == "BLOCKED"
    assert record["error"] == "no idle GPU is available"
    assert record["payload"] == payload
    assert record["source_upload_started"] is False
    assert record["cuda_gate_started"] is False


def test_execute_preflight_authority_writes_blocked_record(
    tmp_path,
):
    runner = _load_runner()
    output_path = tmp_path / "preflight.json"
    payload = {
        "python_exists": True,
        "repo_exists": True,
        "cuda_available": True,
        "torch_version": "2.4.1+cu121",
        "cuda_version": "12.1",
        "device_count": 1,
        "devices": [{
            "index": 0,
            "name": "A100",
            "compute_capability": [8, 0],
        }],
        "idle_gpu_indices": [],
        "checkpoint_candidates": [{
            "path": "/models/Qwen3-0.6B",
            "exists": True,
        }],
    }

    record = runner.execute_preflight_authority(
        output_path=output_path,
        payload_executor=lambda: payload,
    )

    assert record["status"] == "BLOCKED"
    assert output_path.is_file()
    assert runner.json.loads(
        output_path.read_text(encoding="utf-8")
    ) == record


def test_preflight_only_exit_code_is_fail_closed(tmp_path):
    runner = _load_runner()

    assert runner.preflight_exit_code({
        "status": "READY",
    }) == 0
    assert runner.preflight_exit_code({
        "status": "BLOCKED",
    }) == 2
    with pytest.raises(ValueError, match="status"):
        runner.preflight_exit_code({
            "status": "UNKNOWN",
        })
