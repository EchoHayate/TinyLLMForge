from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tools import run_cross_engine_k8_remote as controller_module
from tools.run_cross_engine_k8_remote import (
    COMMITTED_SOURCE_PATHS,
    KRB5_CACHE,
    ControllerConfig,
    RemoteController,
    build_committed_source_archive,
    build_byte_ranges,
    build_vllm_install_argv,
    build_worker_plan,
    pending_vllm_versions,
    resolve_vllm_wheel,
    stable_versions_from_pip_index,
    select_admitted_gpu,
)
from tools.cross_engine_k8_workload import build_workload_manifest


class RecordingRunner:
    def __init__(self, stdout=""):
        self.calls = []
        self.stdout = stdout

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs)))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=self.stdout,
            stderr="",
        )


class FlakyRunner(RecordingRunner):
    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs)))
        return subprocess.CompletedProcess(
            argv,
            255 if len(self.calls) == 1 else 0,
            stdout="" if len(self.calls) == 1 else "ok\n",
            stderr=(
                "Connection closed by UNKNOWN port 65535"
                if len(self.calls) == 1
                else ""
            ),
        )


class SequencedRunner(RecordingRunner):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = iter(outputs)

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs)))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=next(self.outputs),
            stderr="",
        )


class BinaryRecordingRunner:
    def __init__(self, stdout=b"archive"):
        self.calls = []
        self.stdout = stdout

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs)))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=self.stdout,
            stderr=b"",
        )


def _gpu(
    *,
    index=0,
    uuid=None,
    name="NVIDIA A100 80GB PCIe",
    memory=0,
    utilization=0,
    processes=(),
):
    return {
        "index": index,
        "uuid": f"GPU-{index}" if uuid is None else uuid,
        "name": name,
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "compute_processes": list(processes),
    }


def _config():
    return ControllerConfig(
        run_tag="20260829-cross-engine-k8-qwen3-06b-r1",
        source_revision="a" * 40,
    )


def test_controller_injects_file_cache_into_every_ssh_command():
    runner = RecordingRunner(stdout="n232-195-203\n")
    controller = RemoteController(
        _config(),
        command_runner=runner,
        sleep=lambda _seconds: None,
    )

    result = controller.remote(["hostname"])

    assert result.stdout == "n232-195-203\n"
    assert runner.calls[0][1]["env"]["KRB5CCNAME"] == KRB5_CACHE
    assert runner.calls[0][0][-1] == "hostname"


def test_controller_retries_transient_ssh_transport_failure():
    runner = FlakyRunner()
    controller = RemoteController(
        _config(),
        command_runner=runner,
        sleep=lambda _seconds: None,
    )

    result = controller.remote(["true"])

    assert result.stdout == "ok\n"
    assert len(runner.calls) == 2


def test_committed_source_archive_contains_only_runtime_paths(tmp_path):
    runner = BinaryRecordingRunner()

    archive = build_committed_source_archive(
        tmp_path,
        "a" * 40,
        command_runner=runner,
    )

    assert archive == b"archive"
    assert runner.calls[0][0] == [
        "git",
        "archive",
        "--format=tar",
        "a" * 40,
        *COMMITTED_SOURCE_PATHS,
    ]


def test_remote_binary_input_uses_ssh_stdin_without_text_mode():
    runner = BinaryRecordingRunner(stdout=b"ok\n")
    controller = RemoteController(
        _config(),
        command_runner=runner,
        sleep=lambda _seconds: None,
    )

    result = controller.remote_with_input(
        ["python3", "-c", "print('ok')"],
        b"payload",
    )

    assert result.stdout == b"ok\n"
    assert runner.calls[0][1]["input"] == b"payload"
    assert runner.calls[0][1]["text"] is False
    assert runner.calls[0][1]["env"]["KRB5CCNAME"] == KRB5_CACHE


def test_existing_frozen_source_is_verified_without_restreaming(
    monkeypatch,
):
    controller = RemoteController(
        _config(),
        command_runner=RecordingRunner(stdout="a" * 40 + "\n"),
        attempt_exists=lambda path: path.endswith("a" * 40),
        sleep=lambda _seconds: None,
    )

    def fail_if_archived(*_args, **_kwargs):
        raise AssertionError("existing frozen source must not be archived")

    monkeypatch.setattr(
        controller_module,
        "build_committed_source_archive",
        fail_if_archived,
    )

    source_root = controller.ensure_remote_source()

    assert source_root.endswith("a" * 40)
    assert controller._command_runner.calls[0][0][-1].endswith(
        "/.source_revision"
    )


def test_pip_index_versions_are_filtered_stable_and_newest_first():
    output = "\n".join(
        (
            "vllm (0.28.0)",
            "Available versions: 0.28.0, 0.27.1, 0.28.0rc1, "
            "0.10.2, garbage, 0.9.9",
        )
    )

    assert stable_versions_from_pip_index(output) == (
        "0.28.0",
        "0.27.1",
        "0.10.2",
        "0.9.9",
    )


def test_pip_index_versions_rejects_output_without_stable_release():
    with pytest.raises(
        RuntimeError,
        match="VLLM_STABLE_VERSION_DISCOVERY_FAILED",
    ):
        stable_versions_from_pip_index(
            "Available versions: 0.28.0rc1, nightly"
        )


def test_candidate_resume_skips_versions_already_journaled():
    assert pending_vllm_versions(
        ("0.28.0", "0.27.1", "0.27.0"),
        [
            {"version": "0.28.0", "compatible": False},
            {"version": "0.27.1", "compatible": False},
        ],
    ) == ("0.27.0",)


def test_candidate_resume_retries_timeout_after_download_strategy_change():
    assert pending_vllm_versions(
        ("0.28.0", "0.27.1"),
        [
            {
                "version": "0.28.0",
                "compatible": False,
                "phase": "binary_install",
                "returncode": 124,
            },
        ],
    ) == ("0.28.0", "0.27.1")


def test_resolve_vllm_wheel_selects_linux_abi3_and_sha256():
    digest = "a" * 64
    html = (
        '<a href="../../packages/wrong/vllm-0.28.0-cp312-cp312-'
        'macosx_14_0_arm64.whl#sha256='
        + "b" * 64
        + '">wrong</a>'
        '<a href="../../packages/right/vllm-0.28.0-cp38-abi3-'
        'manylinux_2_28_x86_64.whl#sha256='
        + digest
        + '">right</a>'
    )

    wheel = resolve_vllm_wheel(
        html,
        version="0.28.0",
        index_url="https://mirrors.aliyun.com/pypi/simple/vllm/",
    )

    assert wheel == {
        "filename": (
            "vllm-0.28.0-cp38-abi3-manylinux_2_28_x86_64.whl"
        ),
        "sha256": digest,
        "url": (
            "https://mirrors.aliyun.com/pypi/packages/right/"
            "vllm-0.28.0-cp38-abi3-manylinux_2_28_x86_64.whl"
        ),
    }


def test_byte_ranges_cover_file_without_overlap():
    assert build_byte_ranges(total_bytes=10, chunk_bytes=4) == (
        (0, 3),
        (4, 7),
        (8, 9),
    )


def test_vllm_candidate_install_is_binary_only_bounded_and_no_deps():
    argv = build_vllm_install_argv(
        candidate_build="/campaign/envs/vllm-0.28.0.building",
        candidate_version="0.28.0",
        environment={"PIP_CACHE_DIR": "/campaign/cache"},
    )

    assert argv[:3] == ["env", "PIP_CACHE_DIR=/campaign/cache", "timeout"]
    assert "--only-binary=:all:" in argv
    assert "--no-deps" in argv
    assert "https://mirrors.aliyun.com/pypi/simple" in argv
    assert argv[-1] == "vllm==0.28.0"


def test_vllm_candidate_install_accepts_verified_local_wheel():
    argv = build_vllm_install_argv(
        candidate_build="/campaign/envs/vllm-0.28.0.building",
        candidate_version="0.28.0",
        environment={"PIP_CACHE_DIR": "/campaign/cache"},
        wheel_path="/campaign/wheelhouse/vllm-0.28.0.whl",
    )

    assert "--index-url" not in argv
    assert argv[-1] == "/campaign/wheelhouse/vllm-0.28.0.whl"
    assert "900s" in argv


def test_controller_downloads_verified_wheel_under_campaign_root():
    digest = "a" * 64
    filename = (
        "vllm-0.28.0-cp38-abi3-manylinux_2_28_x86_64.whl"
    )
    html = (
        f'<a href="../../packages/right/{filename}'
        f'#sha256={digest}">wheel</a>'
    )
    wheel_path = (
        "/data00/home/sitian/tinyllmforge-workspaces/"
        "command-timeline-20260818/cross-engine-k8-qwen3-06b/"
        f"shared/wheelhouse/{filename}"
    )
    runner = SequencedRunner(
        [
            html,
            (
                '{"download_strategy":"parallel_range_v1",'
                f'"filename":"{filename}","path":"{wheel_path}",'
                f'"sha256":"{digest}","size_bytes":314500103,'
                '"url":"https://mirrors.aliyun.com/pypi/packages/right/'
                f'{filename}"}}'
            ),
        ]
    )
    controller = RemoteController(
        _config(),
        command_runner=runner,
        sleep=lambda _seconds: None,
    )

    receipt = controller.ensure_vllm_wheel("0.28.0")

    assert receipt["path"] == wheel_path
    assert receipt["sha256"] == digest
    assert receipt["url"].endswith(filename)
    assert len(runner.calls) == 2
    assert "/simple/vllm/" in runner.calls[0][0][-1]
    assert "shared/wheelhouse" in runner.calls[1][0][-1]
    assert "/tmp" not in runner.calls[1][0][-1]
    assert "missing_ok" not in runner.calls[1][0][-1]


def test_select_admitted_gpu_requires_exactly_clean_a100_80gb_pcie():
    rows = [
        _gpu(index=0, utilization=1),
        _gpu(index=1, name="NVIDIA H100 80GB HBM3"),
        _gpu(index=2, memory=1025),
        _gpu(index=3, processes=[{"pid": 9, "process_name": "python"}]),
        _gpu(index=4, uuid="GPU-clean"),
    ]

    selected = select_admitted_gpu(rows)

    assert selected["index"] == 4
    assert selected["uuid"] == "GPU-clean"


def test_controller_waits_for_two_matching_strict_clean_samples():
    samples = iter([
        [_gpu(utilization=2)],
        [_gpu()],
        [_gpu()],
    ])
    controller = RemoteController(
        _config(),
        command_runner=RecordingRunner(),
        gpu_inventory=lambda: next(samples),
        sleep=lambda _seconds: None,
    )

    admission = controller.wait_for_admitted_gpu(
        timeout_seconds=10,
        interval_seconds=5,
    )

    assert admission["sample_count"] == 3
    assert admission["admitted"] is True
    assert admission["gpu"]["uuid"] == "GPU-0"


def test_controller_never_kills_foreign_processes():
    signaled = []
    controller = RemoteController(
        _config(),
        command_runner=RecordingRunner(),
        signal_process_group=lambda pgid, signal_name: signaled.append(
            (pgid, signal_name)
        ),
        sleep=lambda _seconds: None,
    )
    controller.owned_process_group = 400

    controller.cleanup_owned_processes()

    assert signaled == [(400, "TERM")]


def test_controller_without_owned_process_group_sends_no_signal():
    signaled = []
    controller = RemoteController(
        _config(),
        command_runner=RecordingRunner(),
        signal_process_group=lambda pgid, signal_name: signaled.append(
            (pgid, signal_name)
        ),
        sleep=lambda _seconds: None,
    )

    controller.cleanup_owned_processes()

    assert signaled == []


def test_existing_attempt_directory_is_never_overwritten():
    controller = RemoteController(
        _config(),
        command_runner=RecordingRunner(),
        attempt_exists=lambda _path: True,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(RuntimeError, match="IMMUTABLE_ATTEMPT_EXISTS"):
        controller.require_new_attempt()


def test_controller_rejects_wrong_source_revision_shape():
    with pytest.raises(ValueError, match="source revision"):
        ControllerConfig(
            run_tag="20260829-cross-engine-k8-qwen3-06b-r1",
            source_revision="bad",
        )


def test_worker_plan_carries_frozen_prompts_and_gpu_identity():
    workload = build_workload_manifest("b" * 64)

    plan = build_worker_plan(
        config=_config(),
        workload=workload,
        arm="tinyllmforge_exact_k8",
        repetition=3,
        gpu={"index": 4, "uuid": "GPU-clean"},
        expected_tokens={"short": list(range(128))},
        smoke=True,
    )

    assert plan["gpu_uuid"] == "GPU-clean"
    assert plan["gpu_index"] == 4
    assert plan["warmups"] == 2
    assert len(plan["cases"]) == 1
    assert len(plan["cases"][0]["prompt_token_ids"]) == 256
    assert plan["expected_tokens"]["short"] == list(range(128))


@pytest.mark.parametrize(
    ("stage", "method_name"),
    [
        ("prepare-environments", "prepare_environments"),
        ("smoke", "run_stage"),
        ("canonical", "run_stage"),
        ("finalize", "finalize"),
    ],
)
def test_main_dispatches_each_post_preflight_stage(
    monkeypatch,
    capsys,
    stage,
    method_name,
):
    calls = []

    class FakeController:
        def __init__(self, config):
            calls.append(("init", config))

        def prepare_environments(self):
            calls.append(("prepare_environments",))
            return {"stage": "prepare-environments"}

        def run_stage(self, selected_stage):
            calls.append(("run_stage", selected_stage))
            return {"stage": selected_stage}

        def finalize(self):
            calls.append(("finalize",))
            return {"stage": "finalize"}

    monkeypatch.setattr(controller_module, "RemoteController", FakeController)

    exit_code = controller_module.main(
        [
            "--stage",
            stage,
            "--run-tag",
            "20260829-cross-engine-k8-qwen3-06b-r1",
            "--source-revision",
            "a" * 40,
        ]
    )

    assert exit_code == 0
    expected_call = (
        ("run_stage", stage)
        if method_name == "run_stage"
        else (method_name,)
    )
    assert expected_call in calls
    assert f'"stage": "{stage}"' in capsys.readouterr().out


def test_controller_supports_documented_direct_script_entrypoint():
    repository_root = Path(__file__).resolve().parent.parent

    result = subprocess.run(
        [
            sys.executable,
            str(repository_root / "tools" / "run_cross_engine_k8_remote.py"),
            "--help",
        ],
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--stage" in result.stdout
