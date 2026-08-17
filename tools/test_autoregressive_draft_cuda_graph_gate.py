from __future__ import annotations

import copy
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import types
from zoneinfo import ZoneInfo

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
        tzinfo=ZoneInfo("Asia/Shanghai"),
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
