from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/qwen35_tp4_decode_internal_profile.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "qwen35_tp4_decode_internal_profile",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _step(
    rank,
    step_index,
    *,
    is_decode,
    decode_ordinal,
    wall_ns,
    cuda_ns,
    request_set_sha256="a" * 64,
):
    return {
        "rank": rank,
        "step_index": step_index,
        "batch_kind": "decode" if is_decode else "prefill",
        "is_decode": is_decode,
        "decode_ordinal": decode_ordinal,
        "active_sequence_count": 4,
        "request_set_sha256": request_set_sha256,
        "wall_ns": wall_ns,
        "cuda_ns": cuda_ns,
        "non_cuda_upper_bound_ns": max(0, wall_ns - cuda_ns),
        "dispatch": "eager",
    }


def _collective(rank, step_index, decode_ordinal, duration_ns):
    return {
        "rank": rank,
        "step_index": step_index,
        "decode_ordinal": decode_ordinal,
        "operation": "row_parallel_all_reduce",
        "tensor_shape": [4, 2048],
        "tensor_dtype": "torch.bfloat16",
        "wall_ns": duration_ns + 10,
        "cuda_ns": duration_ns,
    }


def _payload(
    *,
    policy="recompute",
    repetition=0,
    phase="measured",
    wall_scale=1.0,
    cuda_scale=1.0,
    collective_scale=1.0,
    request_count=1,
):
    case_id = (
        f"w2_long_reuse__{phase}__r{repetition}__{policy}"
    )
    ranks = []
    for rank in range(4):
        steps = [
            _step(
                rank,
                0,
                is_decode=False,
                decode_ordinal=None,
                wall_ns=100_000_000,
                cuda_ns=90_000_000,
            )
        ]
        collectives = []
        step_index = 1
        for request_index in range(request_count):
            request_digest = (
                f"{request_index + 1:064x}"
            )
            for decode_ordinal in range(7):
                base_wall = (
                    80_000_000
                    if decode_ordinal == 0
                    else 60_000_000
                )
                base_cuda = (
                    70_000_000
                    if decode_ordinal == 0
                    else 50_000_000
                )
                steps.append(
                    _step(
                        rank,
                        step_index,
                        is_decode=True,
                        decode_ordinal=decode_ordinal,
                        wall_ns=round(
                            base_wall * wall_scale
                        ) + rank * 1000,
                        cuda_ns=round(
                            base_cuda * cuda_scale
                        ) + rank * 1000,
                        request_set_sha256=request_digest,
                    )
                )
                collective = _collective(
                    rank,
                    step_index,
                    decode_ordinal,
                    round(
                        5_000_000 * collective_scale
                    ) + rank * 100,
                )
                collectives.append(collective)
                step_index += 1
        ranks.append({
            "rank": rank,
            "steps": steps,
            "collectives": collectives,
        })
    return {
        "schema_version": (
            "qwen35.tp4-decode-internal-case.v1"
        ),
        "variant": "decode_internal",
        "resource_policy": "shared-low-utilization",
        "exclusive": False,
        "source_tree_sha256": "b" * 64,
        "workload_manifest_sha256": "c" * 64,
        "case_id": case_id,
        "workload": "w2_long_reuse",
        "policy": policy,
        "phase": phase,
        "repetition": repetition,
        "generated_tokens": 8,
        "units": "nanoseconds",
        "rank_inventory": [0, 1, 2, 3],
        "finalization_status": "complete",
        "ranks": ranks,
    }


def _write_case(
    root,
    policy,
    repetition,
    *,
    wall_scale,
    cuda_scale,
    collective_scale,
):
    payload = _payload(
        policy=policy,
        repetition=repetition,
        wall_scale=wall_scale,
        cuda_scale=cuda_scale,
        collective_scale=collective_scale,
    )
    case_dir = root / payload["case_id"]
    case_dir.mkdir()
    (case_dir / "decode_profile.json").write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_ids = list(range(8))
    rows = [
        {
            "request_id": f"request-{index}",
            "generated_tokens": 8,
            "output_token_ids": output_ids,
        }
        for index in range(4)
    ]
    (case_dir / "case_rows.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_validate_accepts_aligned_four_rank_profile():
    profile = _load()

    validated = profile.validate_decode_profile(_payload())

    assert validated["rank_inventory"] == [0, 1, 2, 3]
    assert validated["decode_ordinals"] == list(range(7))
    assert validated["first_decode_ordinal"] == 0
    assert validated["steady_decode_ordinals"] == list(range(1, 7))


def test_validate_accepts_four_serial_request_groups():
    profile = _load()

    validated = profile.validate_decode_profile(
        _payload(request_count=4)
    )

    assert validated["request_group_count"] == 4
    assert validated["decode_ordinals"] == list(range(7))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["ranks"].pop(),
            "rank",
        ),
        (
            lambda payload: payload["ranks"][1]["steps"].reverse(),
            "step",
        ),
        (
            lambda payload: payload["ranks"][2]["steps"][2].update(
                {"request_set_sha256": "d" * 64}
            ),
            "request",
        ),
        (
            lambda payload: payload["ranks"][0]["steps"][1].update(
                {"wall_ns": -1}
            ),
            "wall_ns",
        ),
        (
            lambda payload: payload["ranks"][3]["collectives"][0].update(
                {"step_index": 999}
            ),
            "collective",
        ),
        (
            lambda payload: payload.update({"generated_tokens": 7}),
            "generated",
        ),
        (
            lambda payload: payload.update(
                {"finalization_status": "active"}
            ),
            "finalization",
        ),
    ],
)
def test_validate_rejects_invalid_profile(mutation, message):
    profile = _load()
    payload = _payload()
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        profile.validate_decode_profile(payload)


def test_aggregate_reports_paired_decode_regions_and_representative_pair(
    tmp_path,
):
    profile = _load()
    exact_scales = [1.02, 1.04, 1.08, 1.06, 1.10]
    for repetition, exact_scale in enumerate(exact_scales):
        _write_case(
            tmp_path,
            "recompute",
            repetition,
            wall_scale=1.0,
            cuda_scale=1.0,
            collective_scale=1.0,
        )
        _write_case(
            tmp_path,
            "exact_restore",
            repetition,
            wall_scale=exact_scale,
            cuda_scale=exact_scale,
            collective_scale=(
                1.0 + 2.0 * (exact_scale - 1.0)
            ),
        )

    summary = profile.aggregate_decode_profiles(tmp_path)

    assert summary["measured_pairs"] == 5
    assert summary["generated_tokens"] == 8
    assert len(summary["first_step"]["paired_ratios"]) == 5
    assert len(summary["steady_state"]["paired_ratios"]) == 5
    assert summary["representative_repetition"] == 3
    assert summary["classification"] == "collective_regression"
    assert summary["by_policy"]["exact_restore"][
        "median_collective_cuda_ns"
    ] > summary["by_policy"]["recompute"][
        "median_collective_cuda_ns"
    ]


def test_aggregate_rejects_output_parity_mismatch(tmp_path):
    profile = _load()
    for repetition in range(5):
        for policy in ("recompute", "exact_restore"):
            _write_case(
                tmp_path,
                policy,
                repetition,
                wall_scale=1.0,
                cuda_scale=1.0,
                collective_scale=1.0,
            )
    exact_case = (
        tmp_path
        / "w2_long_reuse__measured__r2__exact_restore"
        / "case_rows.jsonl"
    )
    rows = [
        json.loads(line)
        for line in exact_case.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    rows[1]["output_token_ids"][-1] = 999
    exact_case.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="parity"):
        profile.aggregate_decode_profiles(tmp_path)
