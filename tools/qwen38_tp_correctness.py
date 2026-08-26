from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


BUNDLE_SCHEMA = "tinyllmforge.qwen38-tp-correctness-bundle.v1"
ROW_SCHEMA = "tinyllmforge.qwen38-tp-correctness-row.v1"
CLEANUP_SCHEMA = "tinyllmforge.qwen38-tp-correctness-cleanup.v1"
SOURCE_SCHEMA = "tinyllmforge.source-manifest.v1"
MODEL_SCHEMA = "tinyllmforge.qwen38-model-manifest.v1"
REPOSITORY = "Qwen/Qwen3.8-27B"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_EXPECTED_MODES = {
    "official_tp1": (1, (0,)),
    "tinyllmforge_tp1": (1, (0,)),
    "tinyllmforge_tp4": (4, (0, 1, 2, 3)),
}
_ROW_FIELDS = {
    "schema_version",
    "source_tree_sha256",
    "model_manifest_sha256",
    "model_repository",
    "model_revision",
    "prompt_sha256",
    "mode",
    "dtype",
    "tp_size",
    "rank",
    "prompt_token_ids",
    "generated_token_ids",
    "positions",
    "finite_logits",
    "expected_weight_shard_sha256",
    "loaded_weight_shard_sha256",
}


def _canonical_bytes(payload) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_sha256(value, label) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _load_json(path, label):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing")

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _load_jsonl(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("correctness rows are missing")
    rows = []
    try:
        for line_number, text in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not text.strip():
                raise ValueError(
                    f"correctness row {line_number} is empty"
                )
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(
                    f"correctness row {line_number} must be an object"
                )
            rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("correctness rows are invalid JSONL") from error
    return rows


def _validate_token_ids(value, label):
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in value
        )
    ):
        raise ValueError(f"{label} must contain non-negative integers")
    return tuple(value)


def _validate_nonnegative_float(value, label) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be a finite non-negative number")
    return float(value)


def _validate_positions(row, *, generated_count, topk):
    positions = row.get("positions")
    if (
        not isinstance(positions, list)
        or len(positions) != generated_count
    ):
        raise ValueError("positions must cover every generated token")
    normalized = []
    actual_finite = True
    for expected_position, position in enumerate(positions):
        if (
            not isinstance(position, dict)
            or set(position)
            != {"position", "topk_token_ids", "topk_logits"}
            or position.get("position") != expected_position
        ):
            raise ValueError("position inventory is invalid")
        token_ids = position["topk_token_ids"]
        logits = position["topk_logits"]
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != topk
            or len(set(token_ids)) != len(token_ids)
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in token_ids
            )
        ):
            raise ValueError("top-k token IDs are invalid")
        if (
            not isinstance(logits, list)
            or len(logits) != topk
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                for value in logits
            )
        ):
            raise ValueError("top-k logits are invalid")
        numeric_logits = tuple(float(value) for value in logits)
        actual_finite = actual_finite and all(
            math.isfinite(value) for value in numeric_logits
        )
        normalized.append({
            "token_ids": tuple(token_ids),
            "logits": numeric_logits,
        })
    return tuple(normalized), actual_finite


def _validate_comparison_row(row, *, generated_count=None, topk=None):
    prompt = _validate_token_ids(
        row.get("prompt_token_ids"),
        "prompt_token_ids",
    )
    generated = _validate_token_ids(
        row.get("generated_token_ids"),
        "generated_token_ids",
    )
    if generated_count is None:
        generated_count = len(generated)
    if len(generated) != generated_count:
        raise ValueError("generated token count mismatch")
    positions_value = row.get("positions")
    if topk is None:
        if not isinstance(positions_value, list) or not positions_value:
            raise ValueError("positions must be a non-empty list")
        first = positions_value[0]
        if not isinstance(first, dict):
            raise ValueError("position inventory is invalid")
        ids = first.get("topk_token_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError("top-k token IDs are invalid")
        topk = len(ids)
    positions, actual_finite = _validate_positions(
        row,
        generated_count=generated_count,
        topk=topk,
    )
    declared_finite = row.get("finite_logits")
    if type(declared_finite) is not bool:
        raise ValueError("finite_logits must be a bool")
    generated_matches_argmax = all(
        generated[index] == position["token_ids"][0]
        for index, position in enumerate(positions)
    )
    return {
        "prompt": prompt,
        "generated": generated,
        "positions": positions,
        "finite": declared_finite and actual_finite,
        "generated_matches_argmax": generated_matches_argmax,
    }


def compare_decode_rows(
    reference: dict,
    tp1: dict,
    tp4: dict,
    *,
    atol: float,
    rtol: float,
) -> dict:
    atol = _validate_nonnegative_float(atol, "atol")
    rtol = _validate_nonnegative_float(rtol, "rtol")
    reference_row = _validate_comparison_row(reference)
    generated_count = len(reference_row["generated"])
    topk = len(reference_row["positions"][0]["token_ids"])
    candidates = [
        _validate_comparison_row(
            tp1,
            generated_count=generated_count,
            topk=topk,
        )
    ]
    rank_rows = tp4.get("rank_rows") if isinstance(tp4, dict) else None
    if not isinstance(rank_rows, list) or len(rank_rows) != 4:
        raise ValueError("TP4 rank rows must contain four entries")
    candidates.extend(
        _validate_comparison_row(
            row,
            generated_count=generated_count,
            topk=topk,
        )
        for row in rank_rows
    )

    exact_prompt_tokens = all(
        row["prompt"] == reference_row["prompt"] for row in candidates
    )
    exact_generated_tokens = (
        reference_row["generated_matches_argmax"]
        and all(
            row["generated"] == reference_row["generated"]
            and row["generated_matches_argmax"]
            for row in candidates
        )
    )
    exact_argmax_positions = all(
        tuple(
            position["token_ids"][0]
            for position in row["positions"]
        )
        == tuple(
            position["token_ids"][0]
            for position in reference_row["positions"]
        )
        for row in candidates
    )
    exact_topk_token_ids = all(
        tuple(
            position["token_ids"]
            for position in row["positions"]
        )
        == tuple(
            position["token_ids"]
            for position in reference_row["positions"]
        )
        for row in candidates
    )
    finite_logits_all_ranks = (
        reference_row["finite"]
        and all(row["finite"] for row in candidates)
    )
    max_abs_error = 0.0
    max_rel_error = 0.0
    within_numeric_tolerance = finite_logits_all_ranks
    for candidate in candidates:
        for reference_position, candidate_position in zip(
            reference_row["positions"],
            candidate["positions"],
        ):
            for reference_logit, candidate_logit in zip(
                reference_position["logits"],
                candidate_position["logits"],
            ):
                if not (
                    math.isfinite(reference_logit)
                    and math.isfinite(candidate_logit)
                ):
                    max_abs_error = math.inf
                    max_rel_error = math.inf
                    within_numeric_tolerance = False
                    continue
                absolute = abs(candidate_logit - reference_logit)
                relative = absolute / max(abs(reference_logit), 1e-12)
                max_abs_error = max(max_abs_error, absolute)
                max_rel_error = max(max_rel_error, relative)
                if absolute > atol + rtol * abs(reference_logit):
                    within_numeric_tolerance = False
    checks = (
        exact_prompt_tokens,
        exact_generated_tokens,
        exact_argmax_positions,
        exact_topk_token_ids,
        within_numeric_tolerance,
        finite_logits_all_ranks,
    )
    return {
        "classification": "PASS" if all(checks) else "FAIL",
        "exact_prompt_tokens": exact_prompt_tokens,
        "exact_generated_tokens": exact_generated_tokens,
        "exact_argmax_positions": exact_argmax_positions,
        "exact_topk_token_ids": exact_topk_token_ids,
        "within_numeric_tolerance": within_numeric_tolerance,
        "finite_logits_all_ranks": finite_logits_all_ranks,
        "max_abs_logit_error": max_abs_error,
        "max_rel_logit_error": max_rel_error,
        "atol": atol,
        "rtol": rtol,
    }


def _validate_bundle_manifest(payload, model_manifest_sha256):
    required = {
        "schema_version",
        "source_tree_sha256",
        "model_manifest_sha256",
        "model_repository",
        "model_revision",
        "prompt_token_ids",
        "prompt_sha256",
        "dtype",
        "generated_token_count",
        "topk",
        "atol",
        "rtol",
    }
    if set(payload) != required or payload.get("schema_version") != (
        BUNDLE_SCHEMA
    ):
        raise ValueError("correctness manifest schema mismatch")
    _validate_sha256(payload["source_tree_sha256"], "source_tree_sha256")
    _validate_sha256(
        payload["model_manifest_sha256"],
        "model_manifest_sha256",
    )
    if payload["model_manifest_sha256"] != model_manifest_sha256:
        raise ValueError("model manifest SHA-256 mismatch")
    if payload["model_repository"] != REPOSITORY:
        raise ValueError("model_repository mismatch")
    if (
        not isinstance(payload["model_revision"], str)
        or _REVISION.fullmatch(payload["model_revision"]) is None
    ):
        raise ValueError("model_revision mismatch")
    prompt = _validate_token_ids(
        payload["prompt_token_ids"],
        "prompt_token_ids",
    )
    prompt_sha256 = _digest_bytes(_canonical_bytes(list(prompt)))
    if payload["prompt_sha256"] != prompt_sha256:
        raise ValueError("prompt_sha256 mismatch")
    if payload["dtype"] != "bfloat16":
        raise ValueError("dtype mismatch")
    for name in ("generated_token_count", "topk"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    _validate_nonnegative_float(payload["atol"], "atol")
    _validate_nonnegative_float(payload["rtol"], "rtol")
    return payload


def _validate_row(row, manifest):
    if set(row) != _ROW_FIELDS or row.get("schema_version") != ROW_SCHEMA:
        raise ValueError("correctness row schema mismatch")
    for field in (
        "source_tree_sha256",
        "model_manifest_sha256",
        "model_repository",
        "model_revision",
        "prompt_sha256",
        "dtype",
    ):
        if row[field] != manifest[field]:
            raise ValueError(f"correctness row {field} mismatch")
    mode = row["mode"]
    if mode not in _EXPECTED_MODES:
        raise ValueError("correctness row mode mismatch")
    expected_tp, expected_ranks = _EXPECTED_MODES[mode]
    if row["tp_size"] != expected_tp or row["rank"] not in expected_ranks:
        raise ValueError("correctness row rank identity mismatch")
    if row["prompt_token_ids"] != manifest["prompt_token_ids"]:
        raise ValueError("correctness row prompt_token_ids mismatch")
    _validate_comparison_row(
        row,
        generated_count=manifest["generated_token_count"],
        topk=manifest["topk"],
    )
    _validate_sha256(
        row["expected_weight_shard_sha256"],
        "expected_weight_shard_sha256",
    )
    _validate_sha256(
        row["loaded_weight_shard_sha256"],
        "loaded_weight_shard_sha256",
    )
    return row


def validate_correctness_bundle(root: Path) -> dict:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("correctness bundle root must be a directory")
    model_manifest_path = root / "model_manifest.json"
    if (
        not model_manifest_path.is_file()
        or model_manifest_path.is_symlink()
    ):
        raise ValueError("model manifest is missing")
    model_manifest_bytes = model_manifest_path.read_bytes()
    model_manifest_sha256 = _digest_bytes(model_manifest_bytes)
    source_manifest = _load_json(
        root / "source_manifest.json",
        "source manifest",
    )
    if (
        set(source_manifest)
        != {
            "schema_version",
            "source_tree_sha256",
            "model_manifest_sha256",
        }
        or source_manifest.get("schema_version") != SOURCE_SCHEMA
    ):
        raise ValueError("source manifest schema mismatch")
    _validate_sha256(
        source_manifest["source_tree_sha256"],
        "source_tree_sha256",
    )
    if source_manifest["model_manifest_sha256"] != model_manifest_sha256:
        raise ValueError("model manifest SHA-256 mismatch")
    model_manifest = _load_json(model_manifest_path, "model manifest")
    if (
        model_manifest.get("schema_version") != MODEL_SCHEMA
        or model_manifest.get("repository") != REPOSITORY
        or _REVISION.fullmatch(
            str(model_manifest.get("resolved_revision", ""))
        )
        is None
    ):
        raise ValueError("model manifest identity mismatch")

    manifest = _validate_bundle_manifest(
        _load_json(
            root / "correctness_manifest.json",
            "correctness manifest",
        ),
        model_manifest_sha256,
    )
    if (
        manifest["source_tree_sha256"]
        != source_manifest["source_tree_sha256"]
    ):
        raise ValueError("source_tree_sha256 mismatch")
    if (
        manifest["model_revision"]
        != model_manifest["resolved_revision"]
    ):
        raise ValueError("model_revision mismatch")

    rows = [
        _validate_row(row, manifest)
        for row in _load_jsonl(root / "correctness_rows.jsonl")
    ]
    rows_by_mode = {}
    for row in rows:
        key = (row["mode"], row["rank"])
        if key in rows_by_mode:
            raise ValueError("duplicate correctness row identity")
        rows_by_mode[key] = row
    for mode, (tp_size, expected_ranks) in _EXPECTED_MODES.items():
        observed = tuple(sorted(
            rank
            for row_mode, rank in rows_by_mode
            if row_mode == mode
        ))
        if observed != expected_ranks:
            raise ValueError(f"{mode} rank inventory mismatch")
        if any(
            rows_by_mode[(mode, rank)]["tp_size"] != tp_size
            for rank in expected_ranks
        ):
            raise ValueError(f"{mode} TP size mismatch")
    if len(rows_by_mode) != 6:
        raise ValueError("correctness row inventory mismatch")

    tp4_rows = [
        rows_by_mode[("tinyllmforge_tp4", rank)]
        for rank in range(4)
    ]
    comparison = compare_decode_rows(
        rows_by_mode[("official_tp1", 0)],
        rows_by_mode[("tinyllmforge_tp1", 0)],
        {"rank_rows": tp4_rows},
        atol=manifest["atol"],
        rtol=manifest["rtol"],
    )
    rank_inventory = [row["rank"] for row in tp4_rows]
    expected_shards = [
        row["expected_weight_shard_sha256"] for row in tp4_rows
    ]
    distinct_expected_shards = len(set(expected_shards)) == 4
    loaded_expected_shards = all(
        row["loaded_weight_shard_sha256"]
        == row["expected_weight_shard_sha256"]
        for row in rows
    )

    cleanup = _load_json(
        root / "cleanup_receipt.json",
        "cleanup receipt",
    )
    if cleanup.get("schema_version") != CLEANUP_SCHEMA:
        raise ValueError("cleanup receipt schema mismatch")
    expected_cleanup_ranks = {
        mode: list(ranks)
        for mode, (_, ranks) in _EXPECTED_MODES.items()
    }
    process_groups_destroyed = (
        cleanup.get("process_groups_destroyed")
        == expected_cleanup_ranks
        and cleanup.get("rank_exit_codes")
        == {
            "official_tp1": [0],
            "tinyllmforge_tp1": [0],
            "tinyllmforge_tp4": [0, 0, 0, 0],
        }
    )
    owned_children_remaining = cleanup.get(
        "owned_children_remaining"
    )
    if not isinstance(owned_children_remaining, list):
        raise ValueError("owned_children_remaining must be a list")

    result = {
        **comparison,
        "rank_inventory": rank_inventory,
        "distinct_expected_shards": distinct_expected_shards,
        "loaded_expected_shards": loaded_expected_shards,
        "process_groups_destroyed": process_groups_destroyed,
        "owned_children_remaining": owned_children_remaining,
        "source_tree_sha256": manifest["source_tree_sha256"],
        "model_manifest_sha256": manifest["model_manifest_sha256"],
        "prompt_sha256": manifest["prompt_sha256"],
    }
    result["classification"] = (
        "PASS"
        if comparison["classification"] == "PASS"
        and rank_inventory == [0, 1, 2, 3]
        and distinct_expected_shards
        and loaded_expected_shards
        and process_groups_destroyed
        and owned_children_remaining == []
        else "FAIL"
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_root")
    args = parser.parse_args(argv)
    result = validate_correctness_bundle(Path(args.bundle_root))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["classification"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
