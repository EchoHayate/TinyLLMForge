from __future__ import annotations

import ast
import copy
from collections import deque
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for package_name in ("tools",):
    package = types.ModuleType(package_name)
    package.__path__ = [
        str(ROOT / package_name.replace(".", "/"))
    ]
    sys.modules.setdefault(package_name, package)

from tools.autoregressive_draft_cuda_graph_contract import (
    build_gate_payload,
    canonical_json_bytes,
    canonical_json_sha256,
    validate_gate_payload,
)

VERIFIER_PATH = (
    ROOT
    / "tools"
    / "verify_autoregressive_draft_cuda_graph_gate.py"
)
RUNNER_PATH = (
    ROOT
    / "tools"
    / "run_autoregressive_draft_cuda_graph_gate_remote.py"
)
COMMAND_TIMELINE_RUNNER_PATH = (
    ROOT
    / "tools"
    / "run_autoregressive_draft_command_timeline_remote.py"
)
GATE_PATH = (
    ROOT / "tools" / "autoregressive_draft_cuda_graph_gate.py"
)
COLLECTIVE_DIAGNOSTIC_PATH = (
    ROOT
    / "tools"
    / "diagnose_autoregressive_draft_tp4_cuda_graph_collective.py"
)


def _load_path(path, module_name):
    assert path.is_file(), f"missing module: {path}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rank_graph_counters(*, graph, replays=None):
    if replays is None:
        replays = 4 if graph else 0
    return [
        {
            "rank": rank,
            "capture_attempts": 1 if graph else 0,
            "captures": 1 if graph else 0,
            "replays": replays,
            "quarantines": 0,
            "fallback_pre_replay": 0,
        }
        for rank in range(4)
    ]


def _rank_graph_resources(*, graph):
    return [
        {
            "rank": rank,
            "ready_entry_count": 1 if graph else 0,
            "static_bytes": 53_408 if graph else 0,
            "reserved_bytes": 8_520_704 if graph else 0,
            "total_capture_ns": (
                1_500_000_000 if graph else 0
            ),
        }
        for rank in range(4)
    ]


def _mode_row(mode, pair_index):
    graph = mode == "graph"
    e2e_ns = (
        80_000_000 + pair_index * 100_000
        if graph
        else 100_000_000 + pair_index * 100_000
    )
    return {
        "mode": mode,
        "target_token_rows": [
            [
                100 + row * 100 + token
                for token in range(16)
            ]
            for row in range(4)
        ],
        "proposal_token_rows": [
            [
                [
                    10 + row * 10 + token
                    for token in range(4)
                ]
                for row in range(4)
            ],
        ],
        "accepted_prefix_counts": [4, 3, 2, 1],
        "transaction_digest": "a" * 64,
        "active_transaction_count": 0,
        "warmup_rank_graph_counters": _rank_graph_counters(
            graph=graph,
            replays=3 if graph else 0,
        ),
        "rank_graph_counters": _rank_graph_counters(
            graph=graph
        ),
        "warmup_rank_graph_resources": _rank_graph_resources(
            graph=graph
        ),
        "rank_graph_resources": _rank_graph_resources(
            graph=graph
        ),
        "rank_memory_rows": [
            {
                "rank": rank,
                "peak_allocated_bytes": (
                    12_000_000 + rank
                ),
                "peak_reserved_bytes": (
                    16_000_000 + rank
                ),
            }
            for rank in range(4)
        ],
        "timing": {
            "e2e_ns": e2e_ns,
            "throughput_tokens_per_second": (
                50.0 if graph else 40.0
            ),
            "ttft_ns": (
                20_000_000 if graph else 22_000_000
            ),
            "tpot_ns": (
                9_000_000 if graph else 10_000_000
            ),
            "proposal_forward_ns": (
                35_000_000 if graph else 53_000_000
            ),
            "proposal_detail_ns": {
                "setup": 2_000_000,
                "backend_submit": (
                    25_000_000 if graph else 45_000_000
                ),
                "selection_collective": 5_000_000,
                "decode_authority": 1_000_000,
                "token_readback": (
                    1_000_000 if graph else 3_000_000
                ),
                "materialize_register": 1_000_000,
            },
        },
        "acceptance": {
            "proposed_tokens": 16,
            "accepted_tokens": 10,
            "accepted_tokens_per_target_call": 2.5,
            "rate": 0.625,
        },
    }


def _ragged_logical_proposal_token_rows():
    return [
        [
            [10 + row * 10 + token for token in range(4)]
            for row in range(4)
        ],
        [
            [20 + row * 10 + token for token in range(4)]
            for row in range(4)
        ],
        [
            [30 + row * 10 + token for token in range(4)]
            for row in range(4)
        ],
        [
            [40, 41, 42],
            [50, 51, 52, 53],
            [60, 61, 62, 63],
            [70, 71, 72],
        ],
        [
            [80, 81, 82, 83],
            [90, 91, 92, 93],
        ],
    ]


def _gate_rows():
    warmups = []
    for warmup_index in range(2):
        warmups.append({
            "warmup_index": warmup_index,
            "eager": _mode_row("eager", warmup_index),
            "graph": _mode_row("graph", warmup_index),
        })
    pairs = []
    for pair_index in range(8):
        pairs.append({
            "pair_index": pair_index,
            "order": (
                "eager_graph"
                if pair_index % 2 == 0
                else "graph_eager"
            ),
            "eager": _mode_row("eager", pair_index),
            "graph": _mode_row("graph", pair_index),
        })
    return warmups, pairs


def _payload():
    warmups, pairs = _gate_rows()
    return build_gate_payload(
        provenance={
            "source_commit": "b" * 40,
            "source_patch_sha256": "c" * 64,
            "source_tree_sha256": "d" * 64,
            "python_version": "3.11.9",
            "torch_version": "2.5.1",
            "cuda_version": "12.4",
            "nccl_version": "2.21.5",
            "target_model_fingerprint": "e" * 64,
            "draft_model_fingerprint": "f" * 64,
            "tokenizer_fingerprint": "1" * 64,
            "gpu_uuids": [
                f"GPU-{rank:064x}" for rank in range(4)
            ],
        },
        environment={
            "host": "n232-195-203",
            "interference_detected": False,
            "gpu_before": [{"rank": rank} for rank in range(4)],
            "gpu_after": [{"rank": rank} for rank in range(4)],
        },
        warmups=warmups,
        pairs=pairs,
    )


def test_valid_payload_recomputes_go_classification():
    payload = _payload()

    result = validate_gate_payload(payload)

    assert payload["schema_version"] == 2
    assert payload["configuration"][
        "in_process_warmup_runs"
    ] == 1
    assert result["classification"] == "GO"
    assert result["correctness_passed"] is True
    assert result["every_rank_replayed"] is True
    assert result["measured_pair_count"] == 8
    assert result["order_counts"] == {
        "eager_graph": 4,
        "graph_eager": 4,
    }
    assert result["median_graph_throughput"] > (
        result["median_eager_throughput"]
    )
    assert result["paired_throughput_delta_ci_low"] > 0


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        (
            lambda value: value["pairs"][0]["graph"][
                "target_token_rows"
            ][0].pop(),
            "target token",
        ),
        (
            lambda value: value["pairs"][0]["graph"][
                "proposal_token_rows"
            ][0][0].pop(),
            "proposal token",
        ),
        (
            lambda value: value["pairs"][0]["graph"][
                "accepted_prefix_counts"
            ].append(0),
            "accepted prefix",
        ),
        (
            lambda value: value["pairs"][0]["graph"].__setitem__(
                "transaction_digest",
                "0" * 64,
            ),
            "transaction digest",
        ),
        (
            lambda value: value["pairs"][0]["graph"][
                "rank_graph_counters"
            ][2].__setitem__("replays", 0),
            "replay",
        ),
        (
            lambda value: value["provenance"].__setitem__(
                "source_tree_sha256",
                "tampered",
            ),
            "source_tree_sha256",
        ),
        (
            lambda value: value["pairs"][1].__setitem__(
                "order",
                "eager_graph",
            ),
            "position balanced",
        ),
        (
            lambda value: value["pairs"][0]["graph"][
                "rank_memory_rows"
            ][0].__setitem__("peak_reserved_bytes", -1),
            "memory",
        ),
        (
            lambda value: value["summary"].__setitem__(
                "median_graph_throughput",
                999.0,
            ),
            "summary",
        ),
        (
            lambda value: value["pairs"][0]["graph"].__setitem__(
                "active_transaction_count",
                1,
            ),
            "active transaction",
        ),
    ),
)
def test_verifier_rejects_semantic_and_aggregate_tampering(
    tamper,
    message,
):
    payload = copy.deepcopy(_payload())
    tamper(payload)

    with pytest.raises(ValueError, match=message):
        validate_gate_payload(payload)


def test_environment_interference_is_inconclusive():
    payload = _payload()
    payload["environment"]["interference_detected"] = True
    payload["summary"] = None
    rebuilt = build_gate_payload(
        provenance=payload["provenance"],
        environment=payload["environment"],
        warmups=payload["warmups"],
        pairs=payload["pairs"],
    )

    result = validate_gate_payload(rebuilt)

    assert result["classification"] == (
        "INCONCLUSIVE_ENVIRONMENT"
    )


def test_contract_accepts_ragged_logical_proposal_rows_within_b4_q4():
    payload = _payload()
    proposal_rows = _ragged_logical_proposal_token_rows()
    for mode in ("eager", "graph"):
        payload["pairs"][0][mode]["proposal_token_rows"] = (
            copy.deepcopy(proposal_rows)
        )
    payload["summary"] = None

    rebuilt = build_gate_payload(
        provenance=payload["provenance"],
        environment=payload["environment"],
        warmups=payload["warmups"],
        pairs=payload["pairs"],
    )

    assert rebuilt["pairs"][0]["eager"][
        "proposal_token_rows"
    ] == proposal_rows
    assert rebuilt["pairs"][0]["graph"][
        "proposal_token_rows"
    ] == proposal_rows


@pytest.mark.parametrize(
    "mutate",
    (
        lambda rows: rows.append([]),
        lambda rows: rows[0].append([]),
        lambda rows: rows[0].append([1, 2, 3, 4]),
        lambda rows: rows[0][0].append(5),
    ),
    ids=(
        "empty-call",
        "empty-row",
        "more-than-b4-rows",
        "more-than-q4-tokens",
    ),
)
def test_contract_rejects_invalid_logical_proposal_row_bounds(
    mutate,
):
    payload = _payload()
    for mode in ("eager", "graph"):
        mutate(payload["pairs"][0][mode]["proposal_token_rows"])
    payload["summary"] = None

    with pytest.raises(ValueError, match="proposal token"):
        build_gate_payload(
            provenance=payload["provenance"],
            environment=payload["environment"],
            warmups=payload["warmups"],
            pairs=payload["pairs"],
        )


@pytest.mark.parametrize(
    ("field", "mutate", "message"),
    (
        (
            "target_token_rows",
            lambda rows: rows[0].pop(),
            "target token shape",
        ),
        (
            "proposal_token_rows",
            lambda rows: rows[0].append([1, 2, 3, 4]),
            "proposal token shape",
        ),
    ),
)
def test_verifier_rejects_equally_malformed_exact_shapes(
    field,
    mutate,
    message,
):
    payload = _payload()
    for mode in ("eager", "graph"):
        mutate(payload["pairs"][0][mode][field])
    payload["summary"] = None

    with pytest.raises(ValueError, match=message):
        build_gate_payload(
            provenance=payload["provenance"],
            environment=payload["environment"],
            warmups=payload["warmups"],
            pairs=payload["pairs"],
        )


def _write_source_bound_bundle(tmp_path):
    source_root = tmp_path / "source"
    source_path = source_root / "tinyvllm" / "config.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("CONFIG = 1\n", encoding="utf-8")
    source_hashes = {
        "tinyvllm/config.py": (
            __import__("hashlib").sha256(
                source_path.read_bytes()
            ).hexdigest()
        ),
    }
    patch_path = tmp_path / "source.patch"
    patch_path.write_text("diff --git a/x b/x\n", encoding="utf-8")
    payload = _payload()
    payload["provenance"]["source_patch_sha256"] = (
        __import__("hashlib").sha256(
            patch_path.read_bytes()
        ).hexdigest()
    )
    payload["provenance"]["source_tree_sha256"] = (
        canonical_json_sha256(source_hashes)
    )
    payload = build_gate_payload(
        provenance=payload["provenance"],
        environment=payload["environment"],
        warmups=payload["warmups"],
        pairs=payload["pairs"],
    )
    payload_path = tmp_path / "result.json"
    payload_path.write_bytes(canonical_json_bytes(payload))
    manifest = {
        "schema_version": 1,
        "payload_sha256": canonical_json_sha256(payload),
        "source_patch": "source.patch",
        "source_files": source_hashes,
    }
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return payload_path, source_root, patch_path, manifest_path


def test_source_bound_verifier_recomputes_payload_and_source_hashes(
    tmp_path,
):
    verifier = _load_path(
        VERIFIER_PATH,
        "autoregressive_draft_cuda_graph_verifier_test",
    )
    (
        payload_path,
        source_root,
        patch_path,
        manifest_path,
    ) = _write_source_bound_bundle(tmp_path)

    receipt = verifier.verify_gate_bundle(
        payload_path=payload_path,
        source_root=source_root,
        source_patch_path=patch_path,
        source_manifest_path=manifest_path,
    )

    assert receipt["classification"] == "GO"
    assert receipt["schema_version"] == 2
    assert receipt["source_files_verified"] == 1
    assert receipt["payload_sha256"] == canonical_json_sha256(
        json.loads(payload_path.read_text())
    )


def test_source_bound_verifier_rejects_changed_source(tmp_path):
    verifier = _load_path(
        VERIFIER_PATH,
        "autoregressive_draft_cuda_graph_verifier_tamper_test",
    )
    (
        payload_path,
        source_root,
        patch_path,
        manifest_path,
    ) = _write_source_bound_bundle(tmp_path)
    (source_root / "tinyvllm" / "config.py").write_text(
        "CONFIG = 2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source hash mismatch"):
        verifier.verify_gate_bundle(
            payload_path=payload_path,
            source_root=source_root,
            source_patch_path=patch_path,
            source_manifest_path=manifest_path,
        )


def test_remote_preflight_requires_four_clean_gpus():
    runner = _load_path(
        RUNNER_PATH,
        "autoregressive_draft_cuda_graph_runner_preflight_test",
    )
    rows = [
        {
            "index": index,
            "uuid": f"GPU-{index}",
            "memory_used_mib": 0,
            "utilization_percent": 0,
            "compute_process_count": 0,
        }
        for index in range(8)
    ]

    ready = runner.classify_gpu_preflight(rows)
    blocked = runner.classify_gpu_preflight([
        {
            **row,
            "compute_process_count": (
                1 if row["index"] < 5 else 0
            ),
        }
        for row in rows
    ])

    assert ready == {
        "status": "READY",
        "gpu_indices": [0, 1, 2, 3],
    }
    assert blocked["status"] == "INCONCLUSIVE_ENVIRONMENT"
    assert blocked["gpu_indices"] == []


def _kerberos_payload(
    *,
    expires,
    principal="sitian@BYTEDANCE.COM",
):
    return {
        "version": 1,
        "cache": "API:redacted",
        "principal": principal,
        "tickets": [{
            "Issued": "20260817120000",
            "Expires": expires,
            "Principal": (
                "krbtgt/BYTEDANCE.COM@BYTEDANCE.COM"
            ),
        }],
    }


def _kerberos_now():
    return datetime(
        2026,
        8,
        17,
        20,
        0,
        tzinfo=timezone(timedelta(hours=8)),
    )


def test_local_kerberos_payload_accepts_expected_tgt_with_sufficient_lifetime():
    runner = _load_path(
        RUNNER_PATH,
        "autoregressive_draft_cuda_graph_runner_kerberos_ready_test",
    )

    result = runner.classify_local_kerberos_payload(
        _kerberos_payload(expires="20260817220001"),
        now=_kerberos_now(),
    )

    assert result == {
        "status": "READY",
        "principal": "sitian@BYTEDANCE.COM",
        "tgt_principal": (
            "krbtgt/BYTEDANCE.COM@BYTEDANCE.COM"
        ),
        "expires_at": "2026-08-17T22:00:01+08:00",
        "remaining_lifetime_seconds": 7201,
        "minimum_required_lifetime_seconds": 5400,
    }


@pytest.mark.parametrize(
    ("payload", "reason"),
    (
        (
            _kerberos_payload(expires="20260817195959"),
            "local Kerberos TGT is expired",
        ),
        (
            _kerberos_payload(expires="20260817212959"),
            "local Kerberos TGT lifetime is insufficient",
        ),
        (
            {
                **_kerberos_payload(expires="20260817220001"),
                "tickets": [],
            },
            "local Kerberos TGT is missing",
        ),
        (
            _kerberos_payload(
                expires="20260817220001",
                principal="someone@BYTEDANCE.COM",
            ),
            "local Kerberos principal is unexpected",
        ),
        (
            None,
            "local Kerberos payload is invalid",
        ),
        (
            _kerberos_payload(expires="not-a-timestamp"),
            "local Kerberos payload is invalid",
        ),
    ),
)
def test_local_kerberos_payload_rejects_invalid_or_short_credentials(
    payload,
    reason,
):
    runner = _load_path(
        RUNNER_PATH,
        "autoregressive_draft_cuda_graph_runner_kerberos_reject_test",
    )

    result = runner.classify_local_kerberos_payload(
        payload,
        now=_kerberos_now(),
    )

    assert result["status"] == "INCONCLUSIVE_ENVIRONMENT"
    assert result["reason"] == reason
    assert result["minimum_required_lifetime_seconds"] == 5400
    assert "cache" not in result


def test_local_kerberos_preflight_runs_klist_and_classifies_payload():
    runner = _load_path(
        RUNNER_PATH,
        "autoregressive_draft_cuda_graph_runner_kerberos_command_test",
    )
    commands = []

    def command_runner(command, **_kwargs):
        commands.append(command)
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                _kerberos_payload(expires="20260817220001")
            ),
            stderr="",
        )

    result = runner._local_kerberos_preflight(
        command_runner=command_runner,
        now=_kerberos_now(),
    )

    assert commands == [["klist", "--json"]]
    assert result["status"] == "READY"
    assert result["remaining_lifetime_seconds"] == 7201
    assert "cache" not in result


@pytest.mark.parametrize(
    ("returncode", "stdout", "reason"),
    (
        (
            1,
            "",
            "local Kerberos cache is unavailable",
        ),
        (
            0,
            "not-json",
            "local Kerberos payload is invalid",
        ),
    ),
)
def test_local_kerberos_preflight_rejects_command_or_json_failure(
    returncode,
    stdout,
    reason,
):
    runner = _load_path(
        RUNNER_PATH,
        "autoregressive_draft_cuda_graph_runner_kerberos_failure_test",
    )

    def command_runner(_command, **_kwargs):
        return types.SimpleNamespace(
            returncode=returncode,
            stdout=stdout,
            stderr="secret detail must not leak",
        )

    result = runner._local_kerberos_preflight(
        command_runner=command_runner,
        now=_kerberos_now(),
    )

    assert result == {
        "status": "INCONCLUSIVE_ENVIRONMENT",
        "reason": reason,
        "minimum_required_lifetime_seconds": 5400,
    }


def test_local_kerberos_preflight_rejects_subprocess_start_failure():
    runner = _load_path(
        RUNNER_PATH,
        "autoregressive_draft_cuda_graph_runner_kerberos_oserror_test",
    )

    def command_runner(_command, **_kwargs):
        raise FileNotFoundError("klist is unavailable")

    result = runner._local_kerberos_preflight(
        command_runner=command_runner,
        now=_kerberos_now(),
    )

    assert result == {
        "status": "INCONCLUSIVE_ENVIRONMENT",
        "reason": "local Kerberos cache is unavailable",
        "minimum_required_lifetime_seconds": 5400,
    }


def test_preflight_only_avoids_ssh_when_local_kerberos_is_expired():
    runner = _load_path(
        RUNNER_PATH,
        "autoregressive_draft_cuda_graph_runner_preflight_auth_test",
    )
    kerberos_commands = []
    remote_commands = []

    def expired_kerberos(command, **_kwargs):
        kerberos_commands.append(command)
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                _kerberos_payload(expires="20260817195959")
            ),
            stderr="",
        )

    def forbidden_remote(command, **_kwargs):
        remote_commands.append(command)
        raise AssertionError("SSH must not run")

    result = runner._preflight_only(
        target_model="/models/target",
        draft_model="/models/draft",
        kerberos_command_runner=expired_kerberos,
        command_runner=forbidden_remote,
        now=_kerberos_now(),
    )

    assert result["status"] == "INCONCLUSIVE_ENVIRONMENT"
    assert result["reason"] == "local Kerberos TGT is expired"
    assert kerberos_commands == [["klist", "--json"]]
    assert remote_commands == []


def test_execute_remote_gate_avoids_side_effects_when_local_kerberos_is_expired(
    tmp_path,
):
    runner = _load_path(
        RUNNER_PATH,
        "autoregressive_draft_cuda_graph_runner_execute_auth_test",
    )
    local_run = tmp_path / "must-not-exist"
    remote_commands = []

    def expired_kerberos(_command, **_kwargs):
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                _kerberos_payload(expires="20260817195959")
            ),
            stderr="",
        )

    def forbidden_remote(command, **_kwargs):
        remote_commands.append(command)
        raise AssertionError("remote command must not run")

    result = runner.execute_remote_gate(
        repo_root=tmp_path,
        run_tag="fresh-tag",
        local_run=local_run,
        target_model="/models/target",
        draft_model="/models/draft",
        kerberos_command_runner=expired_kerberos,
        command_runner=forbidden_remote,
        now=_kerberos_now(),
    )

    assert result["classification"] == "INCONCLUSIVE_ENVIRONMENT"
    assert result["preflight"]["reason"] == (
        "local Kerberos TGT is expired"
    )
    assert not local_run.exists()
    assert remote_commands == []


def test_ready_kerberos_creates_local_run_before_remote_preflight(
    tmp_path,
):
    runner = _load_path(
        RUNNER_PATH,
        "autoregressive_draft_cuda_graph_runner_execute_order_test",
    )
    local_run = tmp_path / "ready-run"
    observations = []
    gpu_output = "\n".join((
        "0, GPU-a, 0, 0",
        "1, GPU-b, 0, 0",
        "2, GPU-c, 0, 0",
        runner.GPU_PROCESS_MARKER,
    ))

    def valid_kerberos(_command, **_kwargs):
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                _kerberos_payload(expires="20260817220001")
            ),
            stderr="",
        )

    def remote_runner(_command, **_kwargs):
        observations.append(
            (
                "local-run-exists-before-ssh"
                if local_run.is_dir()
                else "local-run-missing-before-ssh"
            )
        )
        return types.SimpleNamespace(
            returncode=0,
            stdout=gpu_output,
            stderr="",
        )

    result = runner.execute_remote_gate(
        repo_root=tmp_path,
        run_tag="ready-tag",
        local_run=local_run,
        target_model="/models/target",
        draft_model="/models/draft",
        kerberos_command_runner=valid_kerberos,
        command_runner=remote_runner,
        now=_kerberos_now(),
    )

    assert result["classification"] == "INCONCLUSIVE_ENVIRONMENT"
    assert observations == ["local-run-exists-before-ssh"]
    preflight = json.loads(
        (local_run / "preflight.json").read_text()
    )
    assert preflight["local_kerberos"]["status"] == "READY"
    assert "cache" not in preflight["local_kerberos"]


def test_collective_diagnostic_is_source_bound_and_uses_exact_tp4():
    runner = _load_path(
        RUNNER_PATH,
        "autoregressive_draft_cuda_graph_runner_collective_command_test",
    )

    command = runner.build_remote_collective_diagnostic_command(
        source_root="/remote/source",
        output_root="/remote/output",
        gpu_indices=(0, 1, 2, 3),
        pythonpath_extra="/remote/site-packages",
    )

    assert (
        "tools/"
        "diagnose_autoregressive_draft_tp4_cuda_graph_collective.py"
        in runner.SOURCE_PATHS
    )
    assert command[:3] == [
        "env",
        "CUDA_VISIBLE_DEVICES=0,1,2,3",
        "PYTHONPATH=/remote/site-packages:/remote/source",
    ]
    assert command[3:9] == [
        runner.REMOTE_PYTHON,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        "4",
    ]
    assert command[9:] == [
        "/remote/source/tools/"
        "diagnose_autoregressive_draft_tp4_cuda_graph_collective.py",
        "--out-root",
        "/remote/output",
    ]


def test_collective_diagnostic_summary_requires_all_rank_replay_parity():
    diagnostic = _load_path(
        COLLECTIVE_DIAGNOSTIC_PATH,
        "autoregressive_draft_collective_diagnostic_contract_test",
    )
    rows = [
        {
            "rank": rank,
            "world_size": 4,
            "device_index": rank,
            "capture_completed": True,
            "replay_completed": True,
            "all_reduce_value": 10.0,
            "broadcast_value": 7.0,
        }
        for rank in range(4)
    ]

    summary = diagnostic.summarize_rank_rows(rows)

    assert summary["classification"] == "PASS"
    assert summary["rank_count"] == 4
    assert summary["capture_completed"] is True
    assert summary["replay_completed"] is True

    rows[2]["broadcast_value"] = 0.0
    with pytest.raises(ValueError, match="broadcast replay parity"):
        diagnostic.summarize_rank_rows(rows)


def test_collective_diagnostic_resets_graph_before_final_synchronize():
    diagnostic = _load_path(
        COLLECTIVE_DIAGNOSTIC_PATH,
        "autoregressive_draft_collective_diagnostic_cleanup_test",
    )
    calls = []

    class Graph:

        def reset(self):
            calls.append("reset")

    diagnostic.release_captured_graph(
        Graph(),
        synchronize=lambda: calls.append("synchronize"),
    )

    assert calls == ["reset", "synchronize"]


def test_remote_gpu_query_parser_does_not_require_python():
    runner = _load_path(
        RUNNER_PATH,
        "autoregressive_draft_cuda_graph_runner_gpu_parser_test",
    )
    output = "\n".join((
        "0, GPU-a, 10, 0",
        "1, GPU-b, 20, 1",
        runner.GPU_PROCESS_MARKER,
        "GPU-b",
        "GPU-b",
    ))

    rows = runner.parse_gpu_query_output(output)

    assert rows == [
        {
            "index": 0,
            "uuid": "GPU-a",
            "memory_used_mib": 10,
            "utilization_percent": 0,
            "compute_process_count": 0,
        },
        {
            "index": 1,
            "uuid": "GPU-b",
            "memory_used_mib": 20,
            "utilization_percent": 1,
            "compute_process_count": 2,
        },
    ]


def test_remote_preflight_reports_missing_prerequisites_without_python():
    runner = _load_path(
        RUNNER_PATH,
        "autoregressive_draft_cuda_graph_runner_shell_preflight_test",
    )
    gpu_output = "\n".join([
        *(
            f"{index}, GPU-{index}, 0, 0"
            for index in range(8)
        ),
        runner.GPU_PROCESS_MARKER,
    ])
    commands = []

    def command_runner(command, **_kwargs):
        commands.append(command)
        if len(commands) == 1:
            return types.SimpleNamespace(
                returncode=0,
                stdout=gpu_output,
                stderr="",
            )
        if "bash -c" not in command[-1]:
            return types.SimpleNamespace(
                returncode=127,
                stdout="",
                stderr="remote python does not exist",
            )
        return types.SimpleNamespace(
            returncode=0,
            stdout="\n".join((
                "n232-195-203",
                "false",
                "false",
                "false",
                "false",
            )) + "\n",
            stderr="",
        )

    result = runner._remote_preflight(
        target_model="/missing/target",
        draft_model="/missing/draft",
        command_runner=command_runner,
    )

    assert result["status"] == "INCONCLUSIVE_ENVIRONMENT"
    assert result["gpu_indices"] == []
    assert result["python_executable"] is False
    assert result["target_exists"] is False
    assert result["draft_exists"] is False
    assert result["package_root_exists"] is False
    assert len(commands) == 2
    assert commands[1][-1].startswith("bash -c ")


def test_remote_runner_defaults_bind_current_qwen3_environment():
    runner = _load_path(
        RUNNER_PATH,
        "autoregressive_draft_cuda_graph_runner_defaults_test",
    )

    assert runner.REMOTE_PYTHON == (
        "/data00/home/sitian/tllm/env/bin/python"
    )
    assert runner.REMOTE_PACKAGE_ROOT == (
        "/data00/home/sitian/tllm/env/lib/python3.11/site-packages"
    )
    assert runner.DEFAULT_TARGET_MODEL == (
        "/data00/home/sitian/.ms_cache/Qwen/Qwen3-8B"
    )
    assert runner.DEFAULT_DRAFT_MODEL == (
        "/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B"
    )


def test_remote_command_binds_exact_tp4_b4_q4_gate():
    runner = _load_path(
        RUNNER_PATH,
        "autoregressive_draft_cuda_graph_runner_command_test",
    )

    command = runner.build_remote_gate_command(
        source_root="/remote/source",
        target_model="/models/target",
        draft_model="/models/draft",
        output_path="/remote/result.json",
        gpu_indices=(1, 3, 5, 7),
    )

    assert command[:3] == [
        "env",
        "CUDA_VISIBLE_DEVICES=1,3,5,7",
        "PYTHONPATH=/remote/source",
    ]
    assert "--tensor-parallel-size" in command
    assert command[
        command.index("--tensor-parallel-size") + 1
    ] == "4"
    assert command[command.index("--batch-size") + 1] == "4"
    assert command[command.index("--max-proposal-tokens") + 1] == "4"
    assert command[command.index("--warmup-pairs") + 1] == "2"
    assert command[command.index("--measured-pairs") + 1] == "8"


def test_source_archive_contains_only_allowlisted_runtime_files(
    tmp_path,
):
    runner = _load_path(
        RUNNER_PATH,
        "autoregressive_draft_cuda_graph_runner_archive_test",
    )
    repo_root = tmp_path / "repo"
    for relative in runner.SOURCE_PATHS:
        path = repo_root / relative
        if relative.endswith("/"):
            path.mkdir(parents=True)
            (path / "kept.py").write_text("VALUE = 1\n")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {relative}\n", encoding="utf-8")
    experiment = repo_root / "experiments" / "large.bin"
    experiment.parent.mkdir(parents=True)
    experiment.write_bytes(b"do not archive")
    archive_path = tmp_path / "source.tar"

    runner.build_source_archive(repo_root, archive_path)

    with tarfile.open(archive_path, "r:") as archive:
        names = set(archive.getnames())
    assert "source/experiments/large.bin" not in names
    assert all(
        any(
            name == f"source/{relative.rstrip('/')}"
            or name.startswith(
                f"source/{relative.rstrip('/')}/"
            )
            for name in names
        )
        for relative in runner.SOURCE_PATHS
    )


def test_source_archive_allowlist_includes_tp4_transitive_gate_modules():
    runner = _load_path(
        RUNNER_PATH,
        "autoregressive_draft_cuda_graph_runner_dependencies_test",
    )

    assert (
        "tools/autoregressive_draft_tp1_engine_gate.py"
        in runner.SOURCE_PATHS
    )
    assert (
        "tools/autoregressive_draft_tp4_local_gate.py"
        in runner.SOURCE_PATHS
    )


def test_remote_runner_source_has_no_process_destruction():
    source = RUNNER_PATH.read_text(encoding="utf-8")

    for forbidden in (
        "pkill",
        "killall",
        "rm -rf",
        "git reset",
        "git clean",
        "kinit",
        "--keychain",
    ):
        assert forbidden not in source


def test_gate_pair_schedule_is_position_balanced():
    gate = _load_path(
        GATE_PATH,
        "autoregressive_draft_cuda_graph_gate_schedule_test",
    )

    schedule = gate.build_pair_schedule(
        warmup_pairs=2,
        measured_pairs=8,
    )

    assert schedule["warmups"] == [
        "eager_graph",
        "graph_eager",
    ]
    assert schedule["measured"] == [
        "eager_graph",
        "graph_eager",
        "eager_graph",
        "graph_eager",
        "eager_graph",
        "graph_eager",
        "eager_graph",
        "graph_eager",
    ]


def test_gate_worker_command_binds_same_engine_warmup_and_single_run():
    gate = _load_path(
        GATE_PATH,
        "autoregressive_draft_cuda_graph_gate_command_test",
    )

    command = gate.build_worker_command(
        python="/python",
        worker_script="/source/tools/worker.py",
        target_model="/models/target",
        draft_model="/models/draft",
        mode="graph",
        output_path="/run/graph.json",
    )

    assert command[:2] == ["/python", "/source/tools/worker.py"]
    assert command[command.index("--policy") + 1] == "learned"
    assert command[command.index("--batch-size") + 1] == "4"
    assert command[
        command.index("--cuda-graph-mode") + 1
    ] == "graph"
    assert command[command.index("--warmup-runs") + 1] == "1"
    assert command[command.index("--measured-runs") + 1] == "1"


def test_gate_worker_command_supports_command_timeline_five_repeat_epoch():
    gate = _load_path(
        GATE_PATH,
        "autoregressive_draft_cuda_graph_gate_timeline_command_test",
    )

    command = gate.build_worker_command(
        python="/python",
        worker_script="/source/tools/worker.py",
        target_model="/models/target",
        draft_model="/models/draft",
        mode="eager",
        output_path="/run/eager.json",
        warmup_runs=1,
        measured_runs=5,
        command_timeline=True,
    )

    assert command[command.index("--warmup-runs") + 1] == "1"
    assert command[command.index("--measured-runs") + 1] == "5"
    assert "--command-timeline" in command


def _load_command_timeline_runner(name):
    return _load_path(COMMAND_TIMELINE_RUNNER_PATH, name)


def test_command_timeline_runner_schedule_and_worker_commands_are_exact():
    runner = _load_command_timeline_runner(
        "command_timeline_runner_schedule_test"
    )

    schedule = runner.build_epoch_schedule()

    assert schedule == [
        ("block-0", "eager", "first"),
        ("block-0", "graph", "second"),
        ("block-1", "graph", "first"),
        ("block-1", "eager", "second"),
        ("block-2", "graph", "first"),
        ("block-2", "eager", "second"),
        ("block-3", "eager", "first"),
        ("block-3", "graph", "second"),
    ]
    for epoch_index, (_block, mode, _position) in enumerate(schedule):
        command = runner.build_epoch_worker_command(
            source_root="/remote/run/source",
            output_path=f"/remote/run/workers/{epoch_index}.json",
            target_model="/models/target",
            draft_model="/models/draft",
            mode=mode,
            gpu_indices=(1, 3, 5, 7),
        )
        assert command[command.index("--policy") + 1] == "learned"
        assert command[command.index("--batch-size") + 1] == "4"
        assert command[command.index("--warmup-runs") + 1] == "1"
        assert command[command.index("--measured-runs") + 1] == "5"
        assert command[command.index("--cuda-graph-mode") + 1] == mode
        assert "--command-timeline" in command


def test_command_timeline_runner_uses_sitian_only_paths_and_safe_ssh():
    runner = _load_command_timeline_runner(
        "command_timeline_runner_paths_test"
    )

    assert runner.REMOTE_TASK_ROOT == (
        "/data00/home/sitian/tinyllmforge-workspaces/"
        "command-timeline-20260818"
    )
    assert runner.REMOTE_PYTHON == (
        "/data00/home/sitian/tllm/env/bin/python"
    )
    assert runner.primary_run_path("tag_1") == (
        f"{runner.REMOTE_TASK_ROOT}/runs/tag_1"
    )
    assert runner.controller_run_path("tag_1") == (
        f"{runner.REMOTE_TASK_ROOT}/controller-verification/tag_1"
    )
    command = runner.build_ssh_command(["true"])
    assert "ControlMaster=no" in command
    assert "ControlPath=none" in command
    assert all("/tmp" not in argument for argument in command)
    remote_shell = command[-1]
    for name in (
        "TMPDIR",
        "TMP",
        "TEMP",
        "PYTHONPYCACHEPREFIX",
        "XDG_CACHE_HOME",
    ):
        assert f"{name}={runner.REMOTE_TASK_ROOT}/runtime/" in remote_shell


def test_command_timeline_runner_supports_optional_existing_control_path(
    monkeypatch,
):
    control_path = (
        "/Users/bytedance/.ssh/"
        "tinyllmforge-command-timeline-20260819.sock"
    )
    monkeypatch.setenv(
        "TINYLLMFORGE_SSH_CONTROL_PATH",
        control_path,
    )
    runner = _load_command_timeline_runner(
        "command_timeline_runner_control_path_test"
    )

    command = runner.build_ssh_command(["true"])

    assert "ControlMaster=no" in command
    assert f"ControlPath={control_path}" in command
    assert "ControlPath=none" not in command


def test_command_timeline_source_archive_is_exact_and_safe(tmp_path):
    runner = _load_command_timeline_runner(
        "command_timeline_runner_archive_test"
    )
    repo_root = tmp_path / "repo"
    for relative in runner.SOURCE_PATHS:
        path = repo_root / relative.rstrip("/")
        if relative.endswith("/"):
            path.mkdir(parents=True)
            (path / "kept.py").write_text("VALUE = 1\n")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {relative}\n", encoding="utf-8")
    archive_path = tmp_path / "source.tar"

    runner.build_source_archive(repo_root, archive_path)

    with tarfile.open(archive_path, "r:") as archive:
        members = archive.getmembers()
    assert members
    assert all(
        member.name == "source"
        or member.name.startswith("source/")
        for member in members
    )
    assert all(not member.issym() and not member.islnk() for member in members)
    assert len({member.name for member in members}) == len(members)
    assert set(runner.SOURCE_PATHS) == {
        "tinyvllm/",
        "tools/autoregressive_draft_performance_worker.py",
        "tools/autoregressive_draft_performance_gate.py",
        "tools/speculative_runtime_performance_gate.py",
        "tools/autoregressive_draft_tp1_engine_gate.py",
        "tools/autoregressive_draft_tp4_engine_gate.py",
        "tools/autoregressive_draft_tp4_local_gate.py",
        "tools/autoregressive_draft_cuda_graph_contract.py",
        "tools/autoregressive_draft_cuda_graph_gate.py",
        "tools/autoregressive_draft_command_timeline_diagnostic.py",
        "tools/verify_autoregressive_draft_command_timeline_diagnostic.py",
        "tools/autoregressive_draft_paired_stability_diagnostic.py",
        "tools/autoregressive_draft_instability_telemetry.py",
        "tools/autoregressive_draft_host_semantic_diagnostic.py",
        "tools/autoregressive_draft_host_sampler.py",
        "tools/run_autoregressive_draft_command_timeline_remote.py",
    }


def test_command_timeline_source_archive_includes_worker_transitive_dependency():
    runner = _load_command_timeline_runner(
        "command_timeline_runner_worker_dependency_test"
    )

    assert (
        "tools/speculative_runtime_performance_gate.py"
        in runner.SOURCE_PATHS
    )


def test_command_timeline_source_archive_covers_worker_local_import_closure(
    tmp_path,
):
    runner = _load_command_timeline_runner(
        "command_timeline_runner_worker_import_closure_test"
    )
    tools_root = ROOT / "tools"
    pending = deque([
        tools_root / "autoregressive_draft_performance_worker.py",
    ])
    discovered = set()

    while pending:
        source_path = pending.popleft().resolve()
        if source_path in discovered:
            continue
        discovered.add(source_path)
        syntax_tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
        )
        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.Import):
                module_names = [
                    alias.name
                    for alias in node.names
                ]
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_names = [node.module]
            else:
                continue
            for module_name in module_names:
                if module_name.startswith("tools."):
                    candidate = ROOT / (
                        module_name.replace(".", "/") + ".py"
                    )
                elif "." not in module_name:
                    candidate = tools_root / f"{module_name}.py"
                else:
                    continue
                if candidate.is_file():
                    pending.append(candidate)

    archived_files = {
        ROOT / relative
        for relative in runner.SOURCE_PATHS
        if not relative.endswith("/")
    }
    missing = sorted(
        str(path.relative_to(ROOT))
        for path in discovered - archived_files
    )

    assert missing == []

    archive_path = tmp_path / "source.tar"
    archive_path.write_bytes(runner.build_source_archive_bytes(ROOT))
    extracted_root = tmp_path / "extracted"
    with tarfile.open(archive_path, "r:") as archive:
        for member in archive.getmembers():
            destination = extracted_root / member.name
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            assert member.isreg()
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            assert source is not None
            destination.write_bytes(source.read())
    source_root = extracted_root / "source"
    import_probe = """
import sys
import types

torch = types.ModuleType("torch")
torch.cuda = types.SimpleNamespace(synchronize=lambda: None)
tinyvllm = types.ModuleType("tinyvllm")
tinyvllm.SamplingParams = object
sys.modules["torch"] = torch
sys.modules["tinyvllm"] = tinyvllm

import autoregressive_draft_performance_worker as worker

dependencies = worker._default_dependencies()
assert dependencies["engine_factory"].__module__ == (
    "autoregressive_draft_tp4_engine_gate"
)
"""
    environment = dict(os.environ)
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join((
            str(source_root / "tools"),
            str(source_root),
        )),
    })
    completed = subprocess.run(
        [sys.executable, "-c", import_probe],
        cwd=source_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_command_timeline_source_archive_rejects_symlink_component(tmp_path):
    runner = _load_command_timeline_runner(
        "command_timeline_runner_archive_symlink_test"
    )
    repo_root = tmp_path / "repo"
    real_tools = tmp_path / "real-tools"
    real_tools.mkdir()
    (repo_root / "tinyvllm").mkdir(parents=True)
    (repo_root / "tinyvllm" / "kept.py").write_text("VALUE = 1\n")
    (repo_root / "tools").symlink_to(real_tools, target_is_directory=True)
    for relative in runner.SOURCE_PATHS:
        if relative.startswith("tools/"):
            path = real_tools / Path(relative).name
            path.write_text("# unsafe through symlink\n", encoding="utf-8")

    with pytest.raises(ValueError, match="symlink"):
        runner.build_source_archive(repo_root, tmp_path / "source.tar")


def test_command_timeline_source_archive_can_stream_without_local_file(
    tmp_path,
):
    runner = _load_command_timeline_runner(
        "command_timeline_runner_archive_stream_test"
    )
    repo_root = tmp_path / "repo"
    for relative in runner.SOURCE_PATHS:
        path = repo_root / relative.rstrip("/")
        if relative.endswith("/"):
            path.mkdir(parents=True)
            (path / "kept.py").write_text(
                "STREAMED_SOURCE = True\n",
                encoding="utf-8",
            )
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {relative}\n", encoding="utf-8")

    archive_bytes = runner.build_source_archive_bytes(repo_root)

    assert isinstance(archive_bytes, bytes)
    assert archive_bytes
    archive_path = tmp_path / "remote-source.tar"
    archive_path.write_bytes(archive_bytes)
    extracted = runner.extract_source_archive(
        archive_path,
        tmp_path / "extract",
    )
    assert (extracted / "tinyvllm" / "kept.py").read_text(
        encoding="utf-8"
    ) == "STREAMED_SOURCE = True\n"


def test_command_timeline_epoch_samplers_capture_stderr_without_pipes(
    tmp_path,
    monkeypatch,
):
    runner = _load_command_timeline_runner(
        "command_timeline_runner_sampler_stderr_test"
    )
    popen_calls = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.pid = 1000 + len(popen_calls)
            popen_calls.append((command, kwargs))

    monkeypatch.setattr(runner.subprocess, "Popen", FakeProcess)
    gpu_path = tmp_path / "gpu.jsonl"
    host_path = tmp_path / "host.jsonl"

    _processes, handles = runner._start_epoch_samplers(
        gpu_indices=[0, 1, 2, 3],
        gpu_uuids=["GPU-0", "GPU-1", "GPU-2", "GPU-3"],
        gpu_path=gpu_path,
        host_path=host_path,
    )

    try:
        assert len(popen_calls) == 2
        assert len(handles) == 4
        assert all(
            kwargs["stderr"] is not runner.subprocess.PIPE
            for _command, kwargs in popen_calls
        )
        assert all(
            getattr(kwargs["stderr"], "name", "").endswith(".stderr")
            for _command, kwargs in popen_calls
        )
    finally:
        for handle in handles:
            handle.close()


def test_command_timeline_epoch_samplers_retain_full_gpu_and_host_fields(
    tmp_path,
    monkeypatch,
):
    runner = _load_command_timeline_runner(
        "command_timeline_runner_sampler_schema_test"
    )
    popen_calls = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.pid = 2000 + len(popen_calls)
            popen_calls.append((command, kwargs))

    monkeypatch.setattr(runner.subprocess, "Popen", FakeProcess)
    _processes, handles = runner._start_epoch_samplers(
        gpu_indices=[0, 1, 2, 3],
        gpu_uuids=["GPU-0", "GPU-1", "GPU-2", "GPU-3"],
        gpu_path=tmp_path / "gpu.jsonl",
        host_path=tmp_path / "host.jsonl",
        source_root="/frozen/source",
    )

    try:
        gpu_command = " ".join(popen_calls[0][0])
        host_command = popen_calls[1][0]
        for field in (
            "nvmlDeviceGetPerformanceState",
            "nvmlDeviceGetClockInfo",
            "nvmlDeviceGetPowerUsage",
            "nvmlDeviceGetTemperature",
            "nvmlDeviceGetUtilizationRates",
            "nvmlDeviceGetMemoryInfo",
            "nvmlDeviceGetCurrentClocksThrottleReasons",
        ):
            assert field in gpu_command
        assert '"index": 0' in gpu_command
        assert '"uuid": "GPU-0"' in gpu_command
        assert host_command == [
            runner.REMOTE_PYTHON,
            "/frozen/source/tools/autoregressive_draft_host_sampler.py",
            "--interval-seconds",
            "0.2",
        ]
    finally:
        for handle in handles:
            handle.close()


def _fake_nvml_prelude(
    *,
    shutdown_path,
    query_failure=None,
    uuid_mismatch=False,
):
    return f"""
import ctypes
from pathlib import Path

class FakeFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None
    def __call__(self, *arguments):
        return self.callback(*arguments)

class FakeNvml:
    def __init__(self):
        self.nvmlInit_v2 = FakeFunction(lambda: 0)
        self.nvmlShutdown = FakeFunction(self.shutdown)
        self.nvmlErrorString = FakeFunction(
            lambda _result: b"fake NVML failure"
        )
        self.nvmlDeviceGetHandleByIndex_v2 = FakeFunction(
            self.get_handle
        )
        self.nvmlDeviceGetUUID = FakeFunction(self.get_uuid)
        self.nvmlDeviceGetPerformanceState = FakeFunction(
            self.get_pstate
        )
        self.nvmlDeviceGetClockInfo = FakeFunction(self.get_clock)
        self.nvmlDeviceGetPowerUsage = FakeFunction(self.get_power)
        self.nvmlDeviceGetTemperature = FakeFunction(
            self.get_temperature
        )
        self.nvmlDeviceGetUtilizationRates = FakeFunction(
            self.get_utilization
        )
        self.nvmlDeviceGetMemoryInfo = FakeFunction(self.get_memory)
        self.nvmlDeviceGetCurrentClocksThrottleReasons = FakeFunction(
            self.get_throttle
        )

    def shutdown(self):
        path = Path({str(shutdown_path)!r})
        count = int(path.read_text()) if path.exists() else 0
        path.write_text(str(count + 1))
        return 0

    def get_handle(self, index, output):
        output._obj.value = int(index) + 100
        return 0

    def get_uuid(self, device, output, _length):
        index = int(device.value) - 100
        value = (
            f"GPU-{{index + 1}}" if {uuid_mismatch!r}
            else f"GPU-{{index}}"
        )
        output.value = value.encode()
        return 0

    def get_pstate(self, _device, output):
        output._obj.value = 0
        return 0

    def get_clock(self, _device, clock_type, output):
        output._obj.value = 1410 if int(clock_type) == 1 else 1512
        return 0

    def get_power(self, _device, output):
        if {query_failure!r} == "nvmlDeviceGetPowerUsage":
            return 3
        output._obj.value = 70000
        return 0

    def get_temperature(self, _device, _sensor, output):
        output._obj.value = 40
        return 0

    def get_utilization(self, _device, output):
        output._obj.gpu = 50
        output._obj.memory = 10
        return 0

    def get_memory(self, _device, output):
        output._obj.total = 1024 * 1024 * 1024
        output._obj.free = 924 * 1024 * 1024
        output._obj.used = 100 * 1024 * 1024
        return 0

    def get_throttle(self, _device, output):
        output._obj.value = 0
        return 0

ctypes.CDLL = lambda _name: FakeNvml()
"""


def _direct_nvml_inventory():
    return [
        {"index": index, "uuid": f"GPU-{index}"}
        for index in (2, 3, 4, 6)
    ]


def _direct_nvml_sampler_command(
    runner,
    *,
    shutdown_path,
    query_failure=None,
    uuid_mismatch=False,
    inventory=None,
):
    program = (
        _fake_nvml_prelude(
            shutdown_path=shutdown_path,
            query_failure=query_failure,
            uuid_mismatch=uuid_mismatch,
        )
        + "\n"
        + runner._build_gpu_sampler_script()
    )
    return [
        sys.executable,
        "-c",
        program,
        json.dumps(
            _direct_nvml_inventory()
            if inventory is None
            else inventory
        ),
    ]


def test_command_timeline_direct_nvml_sampler_emits_complete_snapshot(
    tmp_path,
):
    runner = _load_command_timeline_runner(
        "command_timeline_runner_direct_nvml_sampler_test"
    )
    sampler = subprocess.Popen(
        _direct_nvml_sampler_command(
            runner,
            shutdown_path=tmp_path / "shutdown-count",
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert sampler.stdout is not None
    rows = [json.loads(sampler.stdout.readline()) for _ in range(4)]
    sampler.terminate()
    _stdout, stderr = sampler.communicate(timeout=5)

    assert sampler.returncode == 0, stderr
    assert [row["gpu_index"] for row in rows] == [2, 3, 4, 6]
    assert [row["gpu_uuid"] for row in rows] == [
        "GPU-2",
        "GPU-3",
        "GPU-4",
        "GPU-6",
    ]
    assert len({row["sampled_at_unix_ns"] for row in rows}) == 1
    assert len({row["sampled_at_monotonic_ns"] for row in rows}) == 1
    assert all(row["pstate"] == "P0" for row in rows)
    assert all(row["power_w"] == 70.0 for row in rows)
    assert all(row["memory_used_mib"] == 100 for row in rows)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"uuid_mismatch": True}, "GPU UUID inventory changed"),
        (
            {"query_failure": "nvmlDeviceGetPowerUsage"},
            "nvmlDeviceGetPowerUsage",
        ),
        (
            {"inventory": [
                {"index": 2, "uuid": "GPU-2"},
                {"index": 3, "uuid": "GPU-3"},
                {"index": 4, "uuid": "GPU-4"},
                {"index": 4, "uuid": "GPU-6"},
            ]},
            "GPU inventory is invalid",
        ),
    ),
)
def test_command_timeline_direct_nvml_sampler_fails_closed(
    tmp_path,
    kwargs,
    message,
):
    runner = _load_command_timeline_runner(
        "command_timeline_runner_direct_nvml_failure_test"
        + message.replace(" ", "_")
    )
    result = subprocess.run(
        _direct_nvml_sampler_command(
            runner,
            shutdown_path=tmp_path / "shutdown-count",
            **kwargs,
        ),
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode != 0
    assert message in result.stderr
    assert result.stdout == ""


def test_command_timeline_direct_nvml_sampler_shutdowns_once_on_sigterm(
    tmp_path,
):
    runner = _load_command_timeline_runner(
        "command_timeline_runner_direct_nvml_shutdown_test"
    )
    shutdown_path = tmp_path / "shutdown-count"
    sampler = subprocess.Popen(
        _direct_nvml_sampler_command(
            runner,
            shutdown_path=shutdown_path,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert sampler.stdout is not None
    for _ in range(4):
        sampler.stdout.readline()
    sampler.terminate()
    _stdout, stderr = sampler.communicate(timeout=5)

    assert sampler.returncode == 0, stderr
    assert shutdown_path.read_text() == "1"


def test_command_timeline_telemetry_attachment_rejects_boundary_only_gpu_rows(
    tmp_path,
):
    runner = _load_command_timeline_runner(
        "command_timeline_runner_strict_telemetry_coverage_test"
    )
    start = 1_800_000_000_000_000_000
    finish = start + 1_000_000_000
    gpu_uuids = [f"GPU-{index}" for index in range(4)]
    gpu_path = tmp_path / "gpu.jsonl"
    host_path = tmp_path / "host.jsonl"
    gpu_path.write_text(
        "".join(
            json.dumps({
                "sampled_at_unix_ns": start - 1,
                "sampled_at_monotonic_ns": 1,
                "gpu_uuid": uuid,
            }) + "\n"
            for uuid in gpu_uuids
        ),
        encoding="utf-8",
    )
    host_path.write_text(
        json.dumps({
            "sampled_at_unix_ns": start + 1,
            "sampled_at_monotonic_ns": 2,
        }) + "\n",
        encoding="utf-8",
    )
    worker = {
        "measured_runs": [
            {
                "campaign_interval": {
                    "started_at_unix_ns": (
                        start + repeat * 2_000_000_000
                    ),
                    "finished_at_unix_ns": (
                        finish + repeat * 2_000_000_000
                    ),
                },
                "command_timeline_repeat_index": repeat,
            }
            for repeat in range(5)
        ],
    }

    with pytest.raises(
        ValueError,
        match="telemetry coverage is incomplete",
    ):
        runner._attach_epoch_telemetry(
            worker,
            gpu_path=gpu_path,
            host_path=host_path,
            gpu_uuids=gpu_uuids,
        )


def test_command_timeline_telemetry_alignment_uses_real_unix_only_worker_bounds(
    tmp_path,
):
    runner = _load_command_timeline_runner(
        "command_timeline_runner_unix_telemetry_test"
    )
    gpu_uuids = [f"GPU-{index}" for index in range(4)]
    measured_runs = []
    gpu_rows = []
    host_rows = []
    for repeat in range(5):
        start = 1_800_000_000_000_000_000 + repeat * 1_000_000_000
        finish = start + 900_000_000
        measured_runs.append({
            "repeat": repeat,
            "command_timeline_repeat_index": repeat,
            "campaign_interval": {
                "started_at_unix_ns": start,
                "finished_at_unix_ns": finish,
            },
        })
        for gpu_index, gpu_uuid in enumerate(gpu_uuids):
            gpu_rows.append({
                "sampled_at_unix_ns": start + 400_000_000,
                "sampled_at_monotonic_ns": 10_000_000_000
                + repeat * 1_000_000_000
                + gpu_index,
                "gpu_index": gpu_index,
                "gpu_uuid": gpu_uuid,
                "nvidia_timestamp": "2026/08/18 12:00:00.000",
                "pstate": "P0",
                "sm_clock_mhz": 1000,
                "memory_clock_mhz": 1200,
                "power_w": 200.0,
                "temperature_c": 60,
                "gpu_utilization_percent": 50,
                "memory_utilization_percent": 25,
                "memory_used_mib": 100,
                "throttle_reasons_active": 0,
            })
        host_rows.append({
            "schema_version": 1,
            "sampled_at_unix_ns": start + 500_000_000,
            "sampled_at_monotonic_ns": (
                10_000_000_000 + repeat * 1_000_000_000 + 100
            ),
            "cpu_user_ticks": repeat,
        })
    gpu_path = tmp_path / "gpu.jsonl"
    host_path = tmp_path / "host.jsonl"
    gpu_path.write_text(
        "".join(json.dumps(row) + "\n" for row in gpu_rows),
        encoding="utf-8",
    )
    host_path.write_text(
        "".join(json.dumps(row) + "\n" for row in host_rows),
        encoding="utf-8",
    )

    attached = runner._attach_epoch_telemetry(
        {"measured_runs": measured_runs},
        gpu_path=gpu_path,
        host_path=host_path,
        gpu_uuids=gpu_uuids,
    )

    for repeat, run in enumerate(attached["measured_runs"]):
        assert run["campaign_interval"]["started_at_unix_ns"] == (
            measured_runs[repeat]["campaign_interval"][
                "started_at_unix_ns"
            ]
        )
        assert run["campaign_interval"]["finished_at_unix_ns"] == (
            measured_runs[repeat]["campaign_interval"][
                "finished_at_unix_ns"
            ]
        )
        assert {
            row["gpu_uuid"] for row in run["telemetry"]["gpu_rows"]
        } == set(gpu_uuids)
        assert len(run["telemetry"]["host_rows"]) == 1


def test_command_timeline_remote_command_rechecks_kerberos_and_injects_cache_env():
    runner = _load_command_timeline_runner(
        "command_timeline_runner_remote_wrapper_test"
    )
    commands = []

    def command_runner(command, **_kwargs):
        commands.append(command)
        if command == ["klist", "--json"]:
            return types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    _kerberos_payload(expires="20260817220001")
                ),
                stderr="",
            )
        assert command[0] == "ssh"
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

    result = runner._run_remote_command(
        ["true"],
        command_runner=command_runner,
        context="test remote command",
        now=_kerberos_now(),
        text=True,
        capture_output=True,
    )

    assert result.stdout == "ok"
    assert commands[0] == ["klist", "--json"]
    assert commands[1][0] == "ssh"
    remote_shell = commands[1][-1]
    assert f"TMPDIR={runner.REMOTE_TASK_ROOT}/runtime/" in remote_shell
    assert f"XDG_CACHE_HOME={runner.REMOTE_TASK_ROOT}/runtime/" in remote_shell
    assert "mkdir -p" in remote_shell
    assert remote_shell.index("mkdir -p") < remote_shell.index("env")


def test_command_timeline_owned_gpu_binding_rejects_unowned_rank_process():
    runner = _load_command_timeline_runner(
        "command_timeline_runner_owned_gpu_test"
    )
    rows = [
        {
            "index": index,
            "uuid": f"GPU-{index}",
            "compute_processes": [{"pid": 100 + index}],
        }
        for index in range(4)
    ]

    assert runner.validate_owned_gpu_processes(
        rows,
        owned_pids={100, 101, 102, 103},
        gpu_uuids=[f"GPU-{index}" for index in range(4)],
    ) == {
        f"GPU-{index}": 100 + index for index in range(4)
    }
    rows[3]["compute_processes"] = [{"pid": 999}]
    with pytest.raises(ValueError, match="unowned"):
        runner.validate_owned_gpu_processes(
            rows,
            owned_pids={100, 101, 102, 103},
            gpu_uuids=[f"GPU-{index}" for index in range(4)],
        )


def test_command_timeline_worker_launch_uses_dedicated_process_group(
    monkeypatch,
):
    runner = _load_command_timeline_runner(
        "command_timeline_runner_worker_group_test"
    )
    calls = []

    class FakeProcess:
        pid = 4321

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    process = runner._launch_owned_worker(
        ["python", "worker.py"],
        stdout=runner.subprocess.DEVNULL,
        stderr=runner.subprocess.DEVNULL,
    )

    assert process.pid == 4321
    assert calls == [(
        ["python", "worker.py"],
        {
            "stdout": runner.subprocess.DEVNULL,
            "stderr": runner.subprocess.DEVNULL,
            "text": True,
            "start_new_session": True,
        },
    )]
    assert runner.WORKER_TIMEOUT_SECONDS > 0


def test_command_timeline_worker_monitor_refreshes_owned_pids_after_gpu_snapshot(
    monkeypatch,
):
    runner = _load_command_timeline_runner(
        "command_timeline_runner_worker_monitor_spawn_race_test"
    )
    gpu_uuids = [f"GPU-{index}" for index in range(4)]
    gpu_pids = [5101, 5102, 5103, 5104]
    gpu_snapshot_taken = False

    class FakeProcess:
        pid = 5000

        def __init__(self):
            self.polls = iter((None, 0))

        def poll(self):
            return next(self.polls)

        def wait(self, timeout):
            assert timeout == 30
            return 0

    def fake_gpu_rows():
        nonlocal gpu_snapshot_taken
        gpu_snapshot_taken = True
        return [
            {
                "index": index,
                "uuid": uuid,
                "compute_processes": [{"pid": pid}],
            }
            for index, (uuid, pid) in enumerate(zip(gpu_uuids, gpu_pids))
        ]

    def fake_owned_process_group_pids(process_group_id):
        assert process_group_id == 5000
        if not gpu_snapshot_taken:
            return {5000}
        return {5000, *gpu_pids}

    monkeypatch.setattr(runner, "_remote_gpu_rows", fake_gpu_rows)
    monkeypatch.setattr(
        runner,
        "_owned_process_group_pids",
        fake_owned_process_group_pids,
    )

    returncode, binding, observed_owned = runner._monitor_owned_worker(
        FakeProcess(),
        process_group_id=5000,
        gpu_uuids=gpu_uuids,
        monotonic=iter((0, 1)).__next__,
        sleep=lambda _seconds: None,
    )

    assert returncode == 0
    assert binding == dict(zip(gpu_uuids, gpu_pids))
    assert observed_owned == {5000, *gpu_pids}


def test_command_timeline_worker_monitor_still_rejects_external_gpu_pid(
    monkeypatch,
):
    runner = _load_command_timeline_runner(
        "command_timeline_runner_worker_monitor_external_pid_test"
    )
    gpu_uuids = [f"GPU-{index}" for index in range(4)]
    gpu_pids = [5201, 5202, 5203, 9999]

    class FakeProcess:
        pid = 5000

        def poll(self):
            return None

    monkeypatch.setattr(
        runner,
        "_remote_gpu_rows",
        lambda: [
            {
                "index": index,
                "uuid": uuid,
                "compute_processes": [{"pid": pid}],
            }
            for index, (uuid, pid) in enumerate(zip(gpu_uuids, gpu_pids))
        ],
    )
    monkeypatch.setattr(
        runner,
        "_owned_process_group_pids",
        lambda process_group_id: {5000, 5201, 5202, 5203},
    )

    with pytest.raises(ValueError, match="unowned process"):
        runner._monitor_owned_worker(
            FakeProcess(),
            process_group_id=5000,
            gpu_uuids=gpu_uuids,
            monotonic=iter((0, 1)).__next__,
            sleep=lambda _seconds: None,
        )


def test_command_timeline_cleanup_signals_group_after_leader_exit(
    monkeypatch,
):
    runner = _load_command_timeline_runner(
        "command_timeline_runner_worker_group_cleanup_test"
    )
    signals = []

    class FakeProcess:
        def poll(self):
            return 0

    monkeypatch.setattr(
        runner.os,
        "killpg",
        lambda process_group_id, signal_number: signals.append(
            (process_group_id, signal_number)
        ),
    )
    runner._terminate_owned_process_group(FakeProcess(), 4321)

    assert signals
    assert signals[0] == (4321, runner.signal.SIGTERM)


def test_command_timeline_frozen_source_drives_workload_and_both_verifiers():
    source = COMMAND_TIMELINE_RUNNER_PATH.read_text(encoding="utf-8")

    assert "build_source_archive(Path(REMOTE_CURRENT_SOURCE)" not in source
    assert "source_root=Path(REMOTE_CURRENT_SOURCE)" not in source
    assert "build_source_archive_bytes" in source
    assert 'source_root=primary / "source"' in source
    assert 'source_root=controller / "source"' in source


@pytest.mark.parametrize("tag", ["", "../escape", "space tag", "tag.dot"])
def test_command_timeline_run_tag_fails_before_commands_or_writes(
    tmp_path,
    tag,
):
    runner = _load_command_timeline_runner(
        f"command_timeline_runner_tag_test_{len(tag)}"
    )
    commands = []

    def command_runner(command, **_kwargs):
        commands.append(command)
        raise AssertionError("invalid tags must fail before commands")

    with pytest.raises(ValueError, match=r"\[A-Za-z0-9_-\]"):
        runner.run_bundle(run_tag=tag, command_runner=command_runner)
    assert commands == []
    assert list(tmp_path.iterdir()) == []


def test_command_timeline_gpu_preflight_requires_exactly_four_idle_devices():
    runner = _load_command_timeline_runner(
        "command_timeline_runner_gpu_test"
    )
    rows = [
        {
            "index": index,
            "uuid": f"GPU-{index}",
            "memory_used_mib": 0,
            "utilization_percent": 0,
            "compute_processes": [],
        }
        for index in range(4)
    ]

    assert runner.classify_gpu_preflight(rows) == {
        "status": "READY",
        "gpu_indices": [0, 1, 2, 3],
        "gpu_uuids": ["GPU-0", "GPU-1", "GPU-2", "GPU-3"],
    }
    with pytest.raises(ValueError, match="exactly four"):
        runner.classify_gpu_preflight(rows + [{
            **rows[-1],
            "index": 4,
            "uuid": "GPU-4",
        }])
    with pytest.raises(ValueError, match="unrelated"):
        runner.classify_gpu_preflight([
            *rows[:3],
            {**rows[3], "compute_processes": [{"pid": 99}]},
        ])


def _command_timeline_ready_gpu_payload():
    return {
        "primary_exists": False,
        "controller_exists": False,
        "gpu_rows": [
            {
                "index": index,
                "uuid": f"GPU-{index}",
                "memory_used_mib": 0,
                "utilization_percent": 0,
                "compute_processes": [],
            }
            for index in range(4)
        ],
    }


def test_command_timeline_preflight_is_read_only_and_orders_fail_fast_gates():
    runner = _load_command_timeline_runner(
        "command_timeline_runner_preflight_test"
    )
    commands = []

    def command_runner(command, **_kwargs):
        commands.append(command)
        if command == ["klist", "--json"]:
            return types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    _kerberos_payload(expires="20260817220001")
                ),
                stderr="",
            )
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return types.SimpleNamespace(
                returncode=0,
                stdout="8cf39121ffbe357812941e2e05628ed8ab1153ac\n",
                stderr="",
            )
        if command[:2] == ["git", "rev-parse"]:
            return types.SimpleNamespace(
                returncode=0,
                stdout="8cf39121ffbe357812941e2e05628ed8ab1153ac\n",
                stderr="",
            )
        assert command[0] == "ssh"
        assert "TASK7_ACTION=preflight" in command[-1]
        assert (
            f"mkdir -p {runner.REMOTE_RUNTIME_ROOT}/scratch "
            f"{runner.REMOTE_RUNTIME_ROOT}/pycache "
            f"{runner.REMOTE_RUNTIME_ROOT}/xdg"
        ) in command[-1]
        assert (
            f"mkdir -p {runner.primary_run_path('task7_ready')}"
            not in command[-1]
        )
        assert (
            f"mkdir -p {runner.controller_run_path('task7_ready')}"
            not in command[-1]
        )
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps(_command_timeline_ready_gpu_payload()),
            stderr="",
        )

    result = runner.run_preflight(
        run_tag="task7_ready",
        command_runner=command_runner,
        now=_kerberos_now(),
    )

    assert result["status"] == "READY"
    assert result["source_commit"] == (
        "8cf39121ffbe357812941e2e05628ed8ab1153ac"
    )
    assert result["gpu_indices"] == [0, 1, 2, 3]
    assert commands[0] == ["klist", "--json"]
    assert commands[1][:3] == ["git", "rev-parse", "HEAD"]
    assert commands[2][:2] == ["git", "rev-parse"]
    assert commands[3][0] == "ssh"


def test_command_timeline_preflight_reports_insufficient_idle_gpus(
    monkeypatch,
):
    runner = _load_command_timeline_runner(
        "command_timeline_runner_insufficient_gpu_test"
    )
    payload = _command_timeline_ready_gpu_payload()
    payload["gpu_rows"] = payload["gpu_rows"][:3]
    kerberos = {
        "status": "READY",
        "remaining_lifetime_seconds": 7200,
    }
    source_commit = "9" * 40

    monkeypatch.setattr(
        runner,
        "_local_kerberos_preflight",
        lambda **_kwargs: kerberos,
    )
    monkeypatch.setattr(
        runner,
        "_local_source_commit",
        lambda **_kwargs: source_commit,
    )
    monkeypatch.setattr(
        runner,
        "_run_remote_command",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    )

    result = runner.run_preflight(run_tag="task7_gpu_busy")

    assert result == {
        "status": "INCONCLUSIVE_ENVIRONMENT",
        "reason": "GPU preflight requires exactly four rows",
        "available_idle_gpu_count": 3,
        "gpu_indices": [],
        "gpu_uuids": [],
        "source_commit": source_commit,
        "local_kerberos": kerberos,
        "primary_run": runner.primary_run_path("task7_gpu_busy"),
        "controller_run": runner.controller_run_path(
            "task7_gpu_busy"
        ),
    }


def test_command_timeline_bundle_stops_after_first_worker_failure_and_copies_partial():
    runner = _load_command_timeline_runner(
        "command_timeline_runner_partial_failure_test"
    )
    actions = []
    prepare_inputs = []

    def command_runner(command, **_kwargs):
        if command == ["klist", "--json"]:
            return types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    _kerberos_payload(expires="20260817220001")
                ),
                stderr="",
            )
        if command[:2] == ["git", "rev-parse"]:
            return types.SimpleNamespace(
                returncode=0,
                stdout="8cf39121ffbe357812941e2e05628ed8ab1153ac\n",
                stderr="",
            )
        if command[:2] == ["git", "diff"]:
            return types.SimpleNamespace(
                returncode=0,
                stdout=b"diff --git a/source b/source\n",
                stderr=b"",
            )
        remote = command[-1]
        marker = next(
            (
                name
                for name in (
                    "preflight",
                    "prepare",
                    "inventory-before",
                    "epoch",
                    "inventory-after",
                    "partial-copy",
                )
                if f"TASK7_ACTION={name}" in remote
            ),
            None,
        )
        assert marker is not None
        actions.append(marker)
        if marker == "prepare":
            prepare_inputs.append(_kwargs.get("input"))
        if marker == "preflight":
            stdout = json.dumps(_command_timeline_ready_gpu_payload())
            returncode = 0
        elif marker.startswith("inventory"):
            stdout = json.dumps(
                _command_timeline_ready_gpu_payload()["gpu_rows"]
            )
            returncode = 0
        elif marker == "epoch":
            stdout = "worker failed"
            returncode = 9
        else:
            stdout = ""
            returncode = 0
        return types.SimpleNamespace(
            returncode=returncode,
            stdout=stdout,
            stderr="worker stderr" if returncode else "",
        )

    result = runner.run_bundle(
        run_tag="task7_partial",
        command_runner=command_runner,
        now=_kerberos_now(),
    )

    assert result["status"] == "FAILED"
    assert result["failed_epoch"] == "block-0:eager:first"
    assert actions == [
        "preflight",
        "prepare",
        "inventory-before",
        "epoch",
        "partial-copy",
    ]
    assert len(prepare_inputs) == 1
    source_archive, source_patch = runner._decode_prepare_payload(
        prepare_inputs[0]
    )
    assert source_archive
    assert source_patch == b"diff --git a/source b/source\n"


def test_command_timeline_bundle_copies_partial_when_inventory_query_fails():
    runner = _load_command_timeline_runner(
        "command_timeline_runner_inventory_failure_test"
    )
    actions = []

    def command_runner(command, **_kwargs):
        if command == ["klist", "--json"]:
            return types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    _kerberos_payload(expires="20260817220001")
                ),
                stderr="",
            )
        if command[:2] == ["git", "rev-parse"]:
            return types.SimpleNamespace(
                returncode=0,
                stdout="8cf39121ffbe357812941e2e05628ed8ab1153ac\n",
                stderr="",
            )
        if command[:2] == ["git", "diff"]:
            return types.SimpleNamespace(
                returncode=0,
                stdout=b"",
                stderr=b"",
            )
        remote = command[-1]
        marker = next(
            name
            for name in (
                "preflight",
                "prepare",
                "inventory-before",
                "partial-copy",
            )
            if f"TASK7_ACTION={name}" in remote
        )
        actions.append(marker)
        if marker == "preflight":
            return types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps(_command_timeline_ready_gpu_payload()),
                stderr="",
            )
        if marker == "inventory-before":
            return types.SimpleNamespace(
                returncode=7,
                stdout="",
                stderr="inventory failed",
            )
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    result = runner.run_bundle(
        run_tag="task7_inventory_failure",
        command_runner=command_runner,
        now=_kerberos_now(),
    )

    assert result["status"] == "FAILED"
    assert result["failed_epoch"] == "block-0:eager:first"
    assert actions == [
        "preflight",
        "prepare",
        "inventory-before",
        "partial-copy",
    ]


def test_command_timeline_runner_source_forbids_broad_process_or_tree_actions():
    source = COMMAND_TIMELINE_RUNNER_PATH.read_text(encoding="utf-8")

    for forbidden in (
        "pkill",
        "killall",
        "fuser -k",
        "git clean",
        "git reset",
        "rm -rf",
        "sudo",
        '"/tmp',
        "'/tmp",
    ):
        assert forbidden not in source


def test_gate_converts_worker_evidence_without_synthetic_rows():
    gate = _load_path(
        GATE_PATH,
        "autoregressive_draft_cuda_graph_gate_convert_test",
    )
    run = {
        "outputs": [[1, 2], [3, 4]],
        "timing": {
            "latency_s": 0.1,
            "throughput_tokens_per_second": 40.0,
            "ttft_s": 0.02,
            "tpot_s": 0.01,
        },
        "runtime": {
            "proposed_tokens": 8,
            "accepted_draft_tokens": 4,
            "acceptance_rate": 0.5,
            "draft_executor_timing": {
                "max_rank_ms": {"proposal_forward": 5.0},
            },
            "draft_executor_proposal_detail": {
                "critical_rank_ms": {
                    "setup": 0.2,
                    "backend_submit": 3.0,
                    "selection_collective": 0.5,
                    "decode_authority": 0.2,
                    "token_readback": 0.6,
                    "materialize_register": 0.5,
                },
            },
        },
        "memory": {
            "ranks": [
                {
                    "rank": rank,
                    "cuda_peak_allocated_bytes": 100 + rank,
                    "cuda_peak_reserved_bytes": 200 + rank,
                }
                for rank in range(4)
            ],
        },
        "correctness": {
            "proposal_token_rows": [[[9, 8, 7, 6]]],
            "accepted_prefix_counts": [4],
            "transaction_digest": "a" * 64,
            "active_transaction_count": 0,
            "rank_graph_counters": _rank_graph_counters(
                graph=True
            ),
            "rank_graph_resources": _rank_graph_resources(
                graph=True
            ),
        },
    }
    warmup = copy.deepcopy(run)
    warmup["correctness"]["rank_graph_counters"] = (
        _rank_graph_counters(graph=True, replays=3)
    )
    worker = {
        "policy": "learned",
        "batch_size": 4,
        "cuda_graph_mode": "graph",
        "warmup_runs": [warmup],
        "measured_runs": [run],
    }

    row = gate.mode_row_from_worker(worker, mode="graph")

    assert row["mode"] == "graph"
    assert row["target_token_rows"] == run["outputs"]
    assert (
        row["proposal_token_rows"]
        == run["correctness"]["proposal_token_rows"]
    )
    assert row["warmup_rank_graph_counters"] == warmup[
        "correctness"
    ]["rank_graph_counters"]
    assert row["warmup_rank_graph_resources"] == warmup[
        "correctness"
    ]["rank_graph_resources"]
    assert row["rank_graph_resources"] == run[
        "correctness"
    ]["rank_graph_resources"]
    assert row["timing"]["e2e_ns"] == 100_000_000
    assert row["timing"]["proposal_forward_ns"] == 5_000_000
    assert row["rank_memory_rows"][3][
        "peak_reserved_bytes"
    ] == 203


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        (
            lambda row: row["rank_graph_counters"][0].__setitem__(
                "captures",
                2,
            ),
            "capture",
        ),
        (
            lambda row: row["rank_graph_counters"][0].__setitem__(
                "replays",
                row["warmup_rank_graph_counters"][0]["replays"],
            ),
            "replay",
        ),
        (
            lambda row: row["rank_graph_resources"][0].__setitem__(
                "total_capture_ns",
                row["warmup_rank_graph_resources"][0][
                    "total_capture_ns"
                ] + 1,
            ),
            "capture resource",
        ),
        (
            lambda row: row["rank_graph_resources"][0].__setitem__(
                "reserved_bytes",
                row["warmup_rank_graph_resources"][0][
                    "reserved_bytes"
                ] + 1,
            ),
            "capture resource",
        ),
    ),
)
def test_contract_rejects_non_steady_state_graph_measurement(
    tamper,
    message,
):
    payload = _payload()
    raw_warmups, raw_pairs = _gate_rows()
    tamper(raw_pairs[0]["graph"])

    with pytest.raises(ValueError, match=message):
        build_gate_payload(
            provenance=payload["provenance"],
            environment=payload["environment"],
            warmups=raw_warmups,
            pairs=raw_pairs,
        )
