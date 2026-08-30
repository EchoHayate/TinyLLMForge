from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import tarfile
from tempfile import TemporaryDirectory

import pytest

from tools import run_exact_burst_octet_folded_graph_remote as remote


def _gpu_row(
    index: int,
    *,
    memory: int = 0,
    utilization: int = 0,
    processes=None,
) -> dict:
    return {
        "index": index,
        "uuid": f"GPU-{index}",
        "name": "NVIDIA A100 80GB PCIe",
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "compute_processes": [] if processes is None else processes,
    }


def test_remote_paths_and_runtime_prelude_are_mounted_only() -> None:
    paths = remote.remote_paths("20260830-octet-folded-r1")
    assert set(paths) == {"staging", "primary", "controller"}
    assert all(
        path.startswith(remote.TASK_REMOTE_ROOT + "/")
        for path in paths.values()
    )
    prelude = remote.remote_runtime_prelude(
        source=paths["staging"] + "/source",
        run_root=paths["primary"],
        gpu_index=2,
    )
    for variable in (
        "TMPDIR",
        "TMP",
        "TEMP",
        "PYTHONPYCACHEPREFIX",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "TORCH_EXTENSIONS_DIR",
    ):
        assert f"export {variable}=" in prelude
    assert "export CUDA_VISIBLE_DEVICES=2" in prelude
    assert "=/tmp" not in prelude
    assert "=/private/tmp" not in prelude
    assert "set -e" not in prelude


def test_strict_clean_a100_policy_and_source_identity() -> None:
    rows = [
        _gpu_row(0, memory=1024, utilization=5),
        _gpu_row(1, memory=1025),
        _gpu_row(2, utilization=6),
        _gpu_row(3, processes=[{"pid": 7}]),
        {
            **_gpu_row(4),
            "name": "NVIDIA H100 80GB HBM3",
        },
    ]
    assert remote.strict_clean_a100s(rows) == [rows[0]]
    assert remote.validate_source_commit(
        "a" * 40,
        pushed_head="a" * 40,
    ) == "a" * 40
    with pytest.raises(ValueError, match="pushed"):
        remote.validate_source_commit(
            "a" * 40,
            pushed_head="b" * 40,
        )


def test_remote_command_runs_profiler_and_independent_verifier() -> None:
    paths = remote.remote_paths("20260830-octet-folded-r2")
    commands = remote.remote_execution_commands(
        source=paths["staging"] + "/source",
        primary=paths["primary"],
        controller=paths["controller"],
        model=remote.MODEL_PATH,
        gpu_index=1,
        run_tag="20260830-octet-folded-r2",
        source_commit="a" * 40,
        source_patch_sha256=remote.EMPTY_PATCH_SHA256,
    )
    joined = "\n".join(commands)
    assert "tools.profile_exact_burst_octet_folded_graph" in joined
    assert "tools.exact_burst_octet_folded_graph_verify" in joined
    assert "producer_exitcode" in joined
    assert "verifier_exitcode" in joined
    assert "worker_pid" in joined
    assert "worker_pgid" in joined
    assert "=/tmp" not in joined
    assert "=/private/tmp" not in joined


def test_controller_source_contains_no_ticket_or_process_mutation() -> None:
    source = Path(remote.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "kinit",
        "krenew",
        "pkill",
        "killall",
        "nvidia-smi --gpu-reset",
        "os.kill(",
        "os.killpg(",
    ):
        assert forbidden not in source


def test_committed_archive_is_head_only_and_rooted_under_source() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "tinyvllm").mkdir()
        (root / "tools").mkdir()
        (root / "tinyvllm/config.py").write_text(
            "committed = True\n",
            encoding="utf-8",
        )
        (root / "tools/probe.py").write_text(
            "value = 1\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(
            ["git", "add", "--", "tinyvllm", "tools"],
            cwd=root,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Octet Test",
                "-c",
                "user.email=octet@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            cwd=root,
            check=True,
        )
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        (root / "tinyvllm/config.py").write_text(
            "dirty = True\n",
            encoding="utf-8",
        )
        payload = remote.committed_source_archive(
            root,
            source_commit,
        )

    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as bundle:
        names = {member.name for member in bundle.getmembers()}
        member = bundle.extractfile("source/tinyvllm/config.py")
        assert member is not None
        assert member.read() == b"committed = True\n"
    assert "source/tools/probe.py" in names


def test_source_upload_is_scoped_to_octet_folded_staging(
    monkeypatch,
) -> None:
    captured = {}

    def fake_upload(command: str, payload: bytes):
        captured["command"] = command
        captured["payload"] = payload
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=b"",
            stderr=b"",
        )

    monkeypatch.setattr(
        remote,
        "_run_remote_with_input",
        fake_upload,
        raising=False,
    )
    staging = remote.remote_paths(
        "20260830-octet-folded-r3"
    )["staging"]

    assert remote.upload_source_archive(
        staging=staging,
        archive=b"archive",
    ) == staging + "/source"
    assert captured["payload"] == b"archive"
    assert staging in captured["command"]
    assert "unsafe source archive member" in captured["command"]
    with pytest.raises(ValueError, match="staging"):
        remote.upload_source_archive(
            staging=remote.APPROVED_ROOT + "/other/task",
            archive=b"archive",
        )


def test_remote_requirements_enforce_python_model_and_free_space(
    monkeypatch,
) -> None:
    payload = {
        "python": {
            "path": remote.REMOTE_PYTHON,
            "is_file": True,
            "is_executable": True,
        },
        "model": {
            "path": remote.MODEL_PATH,
            "is_dir": True,
            "config_path": remote.MODEL_PATH + "/config.json",
            "config_is_file": True,
        },
        "approved_root": {
            "path": remote.APPROVED_ROOT,
            "is_dir": True,
            "free_bytes": remote.MINIMUM_REMOTE_FREE_BYTES,
        },
    }

    def fake_remote(_command: str):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(remote, "_run_remote", fake_remote)
    assert remote.probe_remote_requirements() == payload

    payload["approved_root"]["free_bytes"] -= 1
    with pytest.raises(ValueError, match="requirements"):
        remote.probe_remote_requirements()


def test_controller_runs_all_preconditions_before_pipeline(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []
    selected = _gpu_row(2)
    heads = iter(("a" * 40, "a" * 40))
    monkeypatch.setenv("KRB5CCNAME", remote.KRB5_CACHE)
    monkeypatch.setattr(
        remote,
        "require_pushed_head",
        lambda _root: calls.append("head") or next(heads),
    )
    monkeypatch.setattr(
        remote,
        "validate_kerberos",
        lambda **_kwargs: calls.append("kerberos") or {"status": "PASS"},
    )
    monkeypatch.setattr(
        remote,
        "establish_ssh_control_master",
        lambda: calls.append("ssh"),
    )
    monkeypatch.setattr(
        remote,
        "probe_remote_requirements",
        lambda: calls.append("requirements") or {"status": "PASS"},
    )
    monkeypatch.setattr(
        remote,
        "require_remote_destinations_absent",
        lambda _paths: calls.append("destinations"),
    )
    monkeypatch.setattr(
        remote,
        "wait_for_clean_a100",
        lambda **_kwargs: (
            calls.append("wait") or ([selected], selected)
        ),
    )
    monkeypatch.setattr(
        remote,
        "committed_source_archive",
        lambda _root, _commit: calls.append("archive") or b"archive",
    )
    monkeypatch.setattr(
        remote,
        "_prepare_remote_source",
        lambda **kwargs: (
            calls.append("upload")
            or kwargs["paths"]["staging"] + "/source"
        ),
    )
    monkeypatch.setattr(
        remote,
        "run_remote_preflight",
        lambda **_kwargs: calls.append("preflight"),
        raising=False,
    )
    monkeypatch.setattr(
        remote,
        "validate_selected_gpu_still_clean",
        lambda gpu: calls.append("recheck") or gpu,
    )
    monkeypatch.setattr(
        remote,
        "create_remote_controller_manifest",
        lambda **_kwargs: calls.append("manifest"),
        raising=False,
    )
    monkeypatch.setattr(
        remote,
        "run_remote_pipeline",
        lambda **_kwargs: (
            calls.append("pipeline")
            or {"producer_exitcode": 0, "verifier_exitcode": 0}
        ),
    )
    monkeypatch.setattr(
        remote,
        "download_bundle",
        lambda **_kwargs: (
            calls.append("download")
            or {"verified": True, "classification": "GO_CEILING"}
        ),
    )

    result = remote.run_controller(
        remote.parse_args([
            "--run-tag",
            "20260830-octet-folded-r4",
            "--source-commit",
            "a" * 40,
            "--local-artifact-root",
            str(tmp_path),
        ])
    )

    assert result["status"] == "COMPLETE"
    assert calls == [
        "head",
        "kerberos",
        "ssh",
        "requirements",
        "destinations",
        "wait",
        "head",
        "archive",
        "upload",
        "kerberos",
        "preflight",
        "recheck",
        "manifest",
        "pipeline",
        "download",
    ]


def test_download_requires_worker_identity_and_receipt_agreement(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = remote.remote_paths("20260830-octet-folded-r5")
    remote_receipt = {
        "verified": True,
        "classification": "GO_CEILING",
        "run_tag": "20260830-octet-folded-r5",
        "source_commit": "a" * 40,
        "source_patch_sha256": remote.EMPTY_PATCH_SHA256,
    }
    local_receipt = dict(remote_receipt)

    def install_bundle(
        destination: Path,
        *,
        include_workers: bool,
        include_manifest: bool,
    ):
        def fake_download(remote_path: str, local_path: Path) -> None:
            local_path.mkdir(parents=True, exist_ok=True)
            if remote_path == paths["controller"]:
                (local_path / "producer_exitcode").write_text(
                    "0\n",
                    encoding="utf-8",
                )
                (local_path / "verifier_exitcode").write_text(
                    "0\n",
                    encoding="utf-8",
                )
                if include_workers:
                    for stage in ("producer", "verifier"):
                        (local_path / f"{stage}_worker_pid").write_text(
                            "123\n",
                            encoding="utf-8",
                        )
                        (local_path / f"{stage}_worker_pgid").write_text(
                            "123\n",
                            encoding="utf-8",
                        )
                if include_manifest:
                    (
                        local_path / "controller_manifest.json"
                    ).write_text(
                        json.dumps({
                            "schema_version":
                                "exact-burst-octet-folded.controller.v1",
                            "run_tag":
                                "20260830-octet-folded-r5",
                            "source_commit": "a" * 40,
                            "source_patch_sha256":
                                remote.EMPTY_PATCH_SHA256,
                            "remote_paths": paths,
                        }),
                        encoding="utf-8",
                    )
                (local_path / "remote-verification.json").write_text(
                    json.dumps(remote_receipt),
                    encoding="utf-8",
                )

        monkeypatch.setattr(
            remote.download,
            "download_remote_tree_preserving_partial",
            fake_download,
        )
        monkeypatch.setattr(
            remote,
            "verify_artifact_directory",
            lambda *_args, **_kwargs: dict(local_receipt),
        )
        return destination

    missing_workers = install_bundle(
        tmp_path / "missing-workers",
        include_workers=False,
        include_manifest=True,
    )
    with pytest.raises(ValueError, match="worker"):
        remote.download_bundle(
            paths=paths,
            local_destination=missing_workers,
        )

    missing_manifest = install_bundle(
        tmp_path / "missing-manifest",
        include_workers=True,
        include_manifest=False,
    )
    with pytest.raises(ValueError, match="manifest"):
        remote.download_bundle(
            paths=paths,
            local_destination=missing_manifest,
        )

    local_receipt["classification"] = "NO_GO_CEILING"
    mismatched = install_bundle(
        tmp_path / "mismatched",
        include_workers=True,
        include_manifest=True,
    )
    with pytest.raises(ValueError, match="disagree"):
        remote.download_bundle(
            paths=paths,
            local_destination=mismatched,
        )
