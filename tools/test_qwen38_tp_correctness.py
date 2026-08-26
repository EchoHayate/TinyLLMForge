from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/qwen38_tp_correctness.py"


def _load():
    assert MODULE_PATH.is_file(), "Qwen3.8 correctness verifier is missing"
    spec = importlib.util.spec_from_file_location(
        "qwen38_tp_correctness_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


correctness = _load()

SOURCE_SHA = "1" * 64
REVISION = "2" * 40
PROMPT = [11, 22, 33, 44]
GENERATED = [7, 8]


def _canonical_bytes(payload):
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _prompt_sha256(tokens):
    return _sha256_bytes(_canonical_bytes(tokens))


def _positions(*, first_token=7, delta=0.0):
    return [
        {
            "position": 0,
            "topk_token_ids": [first_token, 8, 9],
            "topk_logits": [5.0 + delta, 4.0, 3.0],
        },
        {
            "position": 1,
            "topk_token_ids": [8, 7, 9],
            "topk_logits": [6.0 + delta, 4.5, 2.5],
        },
    ]


def _row(
    mode,
    *,
    rank,
    tp_size,
    model_manifest_sha256,
    shard_sha,
    first_token=7,
    delta=0.0,
):
    return {
        "schema_version": correctness.ROW_SCHEMA,
        "source_tree_sha256": SOURCE_SHA,
        "model_manifest_sha256": model_manifest_sha256,
        "model_repository": "Qwen/Qwen3.8-27B",
        "model_revision": REVISION,
        "prompt_sha256": _prompt_sha256(PROMPT),
        "mode": mode,
        "dtype": "bfloat16",
        "tp_size": tp_size,
        "rank": rank,
        "prompt_token_ids": PROMPT,
        "generated_token_ids": GENERATED,
        "positions": _positions(
            first_token=first_token,
            delta=delta,
        ),
        "finite_logits": True,
        "expected_weight_shard_sha256": shard_sha,
        "loaded_weight_shard_sha256": shard_sha,
    }


def _write_json(path, payload):
    path.write_bytes(_canonical_bytes(payload))


def _write_bundle(root: Path):
    model_manifest = {
        "schema_version": "tinyllmforge.qwen38-model-manifest.v1",
        "repository": "Qwen/Qwen3.8-27B",
        "resolved_revision": REVISION,
    }
    model_manifest_bytes = _canonical_bytes(model_manifest)
    model_manifest_sha256 = _sha256_bytes(model_manifest_bytes)
    (root / "model_manifest.json").write_bytes(model_manifest_bytes)
    _write_json(root / "source_manifest.json", {
        "schema_version": "tinyllmforge.source-manifest.v1",
        "source_tree_sha256": SOURCE_SHA,
        "model_manifest_sha256": model_manifest_sha256,
    })
    _write_json(root / "correctness_manifest.json", {
        "schema_version": correctness.BUNDLE_SCHEMA,
        "source_tree_sha256": SOURCE_SHA,
        "model_manifest_sha256": model_manifest_sha256,
        "model_repository": "Qwen/Qwen3.8-27B",
        "model_revision": REVISION,
        "prompt_token_ids": PROMPT,
        "prompt_sha256": _prompt_sha256(PROMPT),
        "dtype": "bfloat16",
        "generated_token_count": len(GENERATED),
        "topk": 3,
        "atol": 0.02,
        "rtol": 0.01,
    })
    rows = [
        _row(
            "official_tp1",
            rank=0,
            tp_size=1,
            model_manifest_sha256=model_manifest_sha256,
            shard_sha="a" * 64,
        ),
        _row(
            "tinyllmforge_tp1",
            rank=0,
            tp_size=1,
            model_manifest_sha256=model_manifest_sha256,
            shard_sha="b" * 64,
            delta=0.001,
        ),
        *[
            _row(
                "tinyllmforge_tp4",
                rank=rank,
                tp_size=4,
                model_manifest_sha256=model_manifest_sha256,
                shard_sha=str(rank + 3) * 64,
                delta=0.002 + rank * 0.001,
            )
            for rank in range(4)
        ],
    ]
    (root / "correctness_rows.jsonl").write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    _write_json(root / "cleanup_receipt.json", {
        "schema_version": correctness.CLEANUP_SCHEMA,
        "process_groups_destroyed": {
            "official_tp1": [0],
            "tinyllmforge_tp1": [0],
            "tinyllmforge_tp4": [0, 1, 2, 3],
        },
        "rank_exit_codes": {
            "official_tp1": [0],
            "tinyllmforge_tp1": [0],
            "tinyllmforge_tp4": [0, 0, 0, 0],
        },
        "owned_children_remaining": [],
    })
    return rows


def _rewrite_rows(root, rows):
    (root / "correctness_rows.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_complete_source_bound_tp1_tp4_bundle_passes(tmp_path):
    _write_bundle(tmp_path)

    result = correctness.validate_correctness_bundle(tmp_path)

    assert result["classification"] == "PASS"
    assert result["exact_prompt_tokens"] is True
    assert result["exact_generated_tokens"] is True
    assert result["exact_argmax_positions"] is True
    assert result["exact_topk_token_ids"] is True
    assert result["within_numeric_tolerance"] is True
    assert result["finite_logits_all_ranks"] is True
    assert result["rank_inventory"] == [0, 1, 2, 3]
    assert result["distinct_expected_shards"] is True
    assert result["process_groups_destroyed"] is True
    assert result["owned_children_remaining"] == []
    assert result["max_abs_logit_error"] == pytest.approx(0.005)
    assert result["max_rel_logit_error"] > 0.0


def test_numerically_close_different_argmax_is_a_hard_failure(tmp_path):
    rows = _write_bundle(tmp_path)
    tp4_rank_2 = next(
        row
        for row in rows
        if row["mode"] == "tinyllmforge_tp4" and row["rank"] == 2
    )
    tp4_rank_2["positions"][0] = {
        "position": 0,
        "topk_token_ids": [8, 7, 9],
        "topk_logits": [5.0001, 4.0, 3.0],
    }
    _rewrite_rows(tmp_path, rows)

    result = correctness.validate_correctness_bundle(tmp_path)

    assert result["classification"] == "FAIL"
    assert result["within_numeric_tolerance"] is True
    assert result["exact_argmax_positions"] is False


@pytest.mark.parametrize(
    "field,value",
    (
        ("source_tree_sha256", "f" * 64),
        ("model_manifest_sha256", "e" * 64),
        ("prompt_sha256", "d" * 64),
        ("model_revision", "c" * 40),
        ("dtype", "float16"),
    ),
)
def test_every_row_must_repeat_the_frozen_identity(
    tmp_path,
    field,
    value,
):
    rows = _write_bundle(tmp_path)
    rows[-1][field] = value
    _rewrite_rows(tmp_path, rows)

    with pytest.raises(ValueError, match=field):
        correctness.validate_correctness_bundle(tmp_path)


def test_rank_inventory_shards_and_cleanup_fail_closed(tmp_path):
    rows = _write_bundle(tmp_path)
    rows.pop()
    _rewrite_rows(tmp_path, rows)
    with pytest.raises(ValueError, match="rank inventory"):
        correctness.validate_correctness_bundle(tmp_path)

    rows = _write_bundle(tmp_path)
    tp4_rows = [
        row for row in rows if row["mode"] == "tinyllmforge_tp4"
    ]
    tp4_rows[-1]["expected_weight_shard_sha256"] = (
        tp4_rows[0]["expected_weight_shard_sha256"]
    )
    tp4_rows[-1]["loaded_weight_shard_sha256"] = (
        tp4_rows[0]["loaded_weight_shard_sha256"]
    )
    _rewrite_rows(tmp_path, rows)
    result = correctness.validate_correctness_bundle(tmp_path)
    assert result["classification"] == "FAIL"
    assert result["distinct_expected_shards"] is False

    _write_bundle(tmp_path)
    cleanup_path = tmp_path / "cleanup_receipt.json"
    cleanup = json.loads(cleanup_path.read_text(encoding="utf-8"))
    cleanup["owned_children_remaining"] = [12345]
    _write_json(cleanup_path, cleanup)
    result = correctness.validate_correctness_bundle(tmp_path)
    assert result["classification"] == "FAIL"
    assert result["owned_children_remaining"] == [12345]


def test_every_mode_must_load_its_expected_weight_shard(tmp_path):
    rows = _write_bundle(tmp_path)
    official = next(
        row for row in rows if row["mode"] == "official_tp1"
    )
    official["loaded_weight_shard_sha256"] = "f" * 64
    _rewrite_rows(tmp_path, rows)

    result = correctness.validate_correctness_bundle(tmp_path)

    assert result["classification"] == "FAIL"
    assert result["loaded_expected_shards"] is False


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("generated_token_count", 3, "generated token count"),
        ("topk", 4, "top-k"),
    ),
)
def test_rows_must_match_manifest_decode_shape(
    tmp_path,
    field,
    value,
    match,
):
    _write_bundle(tmp_path)
    manifest_path = tmp_path / "correctness_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match=match):
        correctness.validate_correctness_bundle(tmp_path)


def test_model_manifest_bytes_are_bound_to_source_manifest(tmp_path):
    _write_bundle(tmp_path)
    (tmp_path / "model_manifest.json").write_text(
        '{"tampered":true}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model manifest SHA-256"):
        correctness.validate_correctness_bundle(tmp_path)


def test_compare_decode_rows_rejects_nonfinite_or_incomplete_positions():
    reference = {
        "prompt_token_ids": PROMPT,
        "generated_token_ids": GENERATED,
        "positions": _positions(),
        "finite_logits": True,
    }
    tp1 = {
        **reference,
        "positions": _positions(delta=0.001),
    }
    tp4 = {
        "rank_rows": [
            {
                **reference,
                "rank": rank,
                "expected_weight_shard_sha256": str(rank + 3) * 64,
                "loaded_weight_shard_sha256": str(rank + 3) * 64,
            }
            for rank in range(4)
        ]
    }
    result = correctness.compare_decode_rows(
        reference,
        tp1,
        tp4,
        atol=0.02,
        rtol=0.01,
    )
    assert result["classification"] == "PASS"

    tp4["rank_rows"][1]["positions"][0]["topk_logits"][0] = float("nan")
    tp4["rank_rows"][1]["finite_logits"] = False
    result = correctness.compare_decode_rows(
        reference,
        tp1,
        tp4,
        atol=0.02,
        rtol=0.01,
    )
    assert result["classification"] == "FAIL"
    assert result["finite_logits_all_ranks"] is False
