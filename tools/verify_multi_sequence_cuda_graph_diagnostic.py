#!/usr/bin/env python3
"""Independently verify multi-sequence CUDA Graph diagnostic artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "tools" / "multi_sequence_cuda_graph_contract.py"
SPLIT_POLICY_PATH = (
    ROOT / "tinyvllm" / "engine" / "flash_attn_split_policy.py"
)


def _load_contract():
    spec = importlib.util.spec_from_file_location(
        "multi_sequence_cuda_graph_contract_for_verifier",
        CONTRACT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_contract()


def _load_split_policy():
    spec = importlib.util.spec_from_file_location(
        "flash_attn_split_policy_for_verifier",
        SPLIT_POLICY_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


split_policy = _load_split_policy()


POLICY_EVIDENCE_FIELDS = (
    "split_policy_name",
    "flash_attn_version",
    "page_table_width",
    "effective_num_splits",
    "heuristic_batch_size",
    "heuristic_num_query_heads",
    "heuristic_num_kv_heads",
    "heuristic_head_dim",
    "heuristic_page_block_size",
    "heuristic_max_seqlen_q",
    "heuristic_multi_processor_count",
    "graph_batch_size",
    "graph_identity_sha256",
)


def _policy_row_key(row: dict) -> tuple[str, int] | None:
    case_id = row.get("case_id")
    step_id = row.get("step_id")
    if not isinstance(case_id, str) or not isinstance(step_id, int):
        return None
    return case_id, step_id


def _expected_policy_from_row(row: dict):
    inputs = split_policy.FlashAttentionSplitInputs(
        batch_size=int(row["heuristic_batch_size"]),
        num_query_heads=int(row["heuristic_num_query_heads"]),
        num_kv_heads=int(row["heuristic_num_kv_heads"]),
        head_dim=int(row["heuristic_head_dim"]),
        page_block_size=int(row["heuristic_page_block_size"]),
        page_table_width=int(row["page_table_width"]),
        max_seqlen_q=int(row["heuristic_max_seqlen_q"]),
        multi_processor_count=int(
            row["heuristic_multi_processor_count"]
        ),
    )
    identity = split_policy.build_flash_attn_263_graph_identity(
        graph_batch_size=int(row["graph_batch_size"]),
        inputs=inputs,
        flash_attn_version=str(row["flash_attn_version"]),
    )
    return inputs, identity


def _identity_summary_entry(identity) -> dict:
    return {
        "sha256": identity.sha256,
        "page_table_width": identity.page_table_width,
        "effective_num_splits": identity.effective_num_splits,
        "graph_batch_size": identity.graph_batch_size,
    }


def verify_policy_integrity(
    *,
    raw_rows: list[dict],
    layer_rows: list[dict],
    kv_rows: list[dict],
    process_rows: dict[str, dict],
) -> dict:
    failures = []
    incomplete = False
    indexed_sets = {}
    for evidence_name, rows in (
        ("raw_rows", raw_rows),
        ("layer_rows", layer_rows),
        ("kv_rows", kv_rows),
    ):
        indexed = {}
        for row_index, row in enumerate(rows):
            key = _policy_row_key(row)
            if key is None:
                failures.append(
                    f"{evidence_name}: row {row_index} missing key"
                )
                incomplete = True
                continue
            process = process_rows.get(key[0], {})
            if process.get("policy") == "legacy_eager_auto":
                continue
            if key in indexed:
                failures.append(f"{evidence_name}: duplicate {key}")
                incomplete = True
                continue
            missing = [
                field
                for field in POLICY_EVIDENCE_FIELDS
                if field not in row
            ]
            if missing:
                failures.append(
                    f"{evidence_name}: {key} missing {missing}"
                )
                incomplete = True
            indexed[key] = row
        indexed_sets[evidence_name] = indexed

    expected_keys = set(indexed_sets["raw_rows"])
    for evidence_name in ("layer_rows", "kv_rows"):
        actual_keys = set(indexed_sets[evidence_name])
        if actual_keys != expected_keys:
            failures.append(
                f"{evidence_name}: policy row keys disagree"
            )
            incomplete = True

    identities_by_case = {}
    splits_by_case_role = {}
    for key in sorted(expected_keys):
        rows = [
            indexed_sets[evidence_name].get(key)
            for evidence_name in ("raw_rows", "layer_rows", "kv_rows")
        ]
        if any(row is None for row in rows):
            continue
        raw = rows[0]
        if any(
            tuple(row.get(field) for field in POLICY_EVIDENCE_FIELDS)
            != tuple(raw.get(field) for field in POLICY_EVIDENCE_FIELDS)
            for row in rows[1:]
        ):
            failures.append(f"policy evidence disagreement for {key}")
            continue
        if any(field not in raw for field in POLICY_EVIDENCE_FIELDS):
            continue
        if raw["split_policy_name"] != contract.HEURISTIC_POLICY_NAME:
            failures.append(f"{key}: split_policy_name drift")
            continue
        try:
            inputs, identity = _expected_policy_from_row(raw)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"{key}: invalid policy inputs: {exc}")
            continue
        expected_values = {
            "page_table_width": inputs.page_table_width,
            "effective_num_splits": identity.effective_num_splits,
            "heuristic_batch_size": inputs.batch_size,
            "heuristic_num_query_heads": inputs.num_query_heads,
            "heuristic_num_kv_heads": inputs.num_kv_heads,
            "heuristic_head_dim": inputs.head_dim,
            "heuristic_page_block_size": inputs.page_block_size,
            "heuristic_max_seqlen_q": inputs.max_seqlen_q,
            "heuristic_multi_processor_count": (
                inputs.multi_processor_count
            ),
            "graph_batch_size": identity.graph_batch_size,
            "graph_identity_sha256": identity.sha256,
        }
        for field, expected in expected_values.items():
            if raw.get(field) != expected:
                failures.append(
                    f"{key}: {field}={raw.get(field)!r}, "
                    f"expected {expected!r}"
                )
        if raw.get("effective_num_splits") == 0:
            failures.append(f"{key}: heuristic candidate uses auto split")
        identities_by_case.setdefault(key[0], []).append(identity)

        process = process_rows.get(key[0], {})
        role = process.get("mode")
        if role in {
            "candidate_eager_heuristic",
            "exact_graph_heuristic",
            "rounded_graph_heuristic",
        }:
            comparison_key = (
                raw["heuristic_batch_size"],
                key[1],
                raw["page_table_width"],
            )
            splits_by_case_role.setdefault(comparison_key, {})[role] = (
                raw["effective_num_splits"]
            )

    for comparison_key, role_splits in splits_by_case_role.items():
        eager_split = role_splits.get("candidate_eager_heuristic")
        if eager_split is None:
            continue
        for role in ("exact_graph_heuristic", "rounded_graph_heuristic"):
            graph_split = role_splits.get(role)
            if graph_split is not None and graph_split != eager_split:
                failures.append(
                    f"{comparison_key}: eager/graph split disagreement"
                )

    for case_id, identities in identities_by_case.items():
        process = process_rows.get(case_id)
        if process is None:
            failures.append(f"{case_id}: process row missing")
            incomplete = True
            continue
        expected_summary = []
        seen = set()
        for identity in identities:
            if identity in seen:
                continue
            seen.add(identity)
            expected_summary.append(_identity_summary_entry(identity))
        if process.get("graph_identities") != expected_summary:
            failures.append(f"{case_id}: graph identity summary drift")

    return {
        "classification": (
            "INCOMPLETE"
            if incomplete
            else "POLICY_DRIFT"
            if failures
            else "POLICY_EXACT"
        ),
        "failures": failures,
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise ValueError(f"{path.name} lacks final newline")
    rows = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path.name} line {line_number} is invalid JSON"
            ) from exc
    return rows


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_bytes(path, contract.canonical_json_bytes(value) + b"\n")


def _case_identity(case) -> tuple[int, str, int]:
    return case.batch_size, case.trajectory, case.repetition


def _expected_same_policy_cases():
    return {
        case.case_id: case for case in contract.build_diagnostic_matrix()
    }


def _expected_compatibility_cases():
    return {
        case.case_id: case
        for case in contract.build_legacy_compatibility_matrix()
    }


def _all_expected_cases():
    return {
        **_expected_same_policy_cases(),
        **_expected_compatibility_cases(),
    }


def _manifest_case_domain(
    manifest: dict,
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    all_same_policy = _expected_same_policy_cases()
    all_compatibility = _expected_compatibility_cases()
    failures = []

    def select(field: str, known: dict[str, object]) -> dict[str, object]:
        values = manifest.get(field)
        if not isinstance(values, list) or any(
            not isinstance(case_id, str) for case_id in values
        ):
            failures.append(f"manifest {field} must be a string list")
            return {}
        if len(values) != len(set(values)):
            failures.append(f"manifest {field} contains duplicates")
        unknown = [case_id for case_id in values if case_id not in known]
        if unknown:
            failures.append(f"manifest {field} unknown {unknown}")
        return {
            case_id: known[case_id]
            for case_id in values
            if case_id in known
        }

    same_policy = select("same_policy_case_ids", all_same_policy)
    compatibility = select(
        "compatibility_case_ids",
        all_compatibility,
    )
    case_ids = manifest.get("case_ids")
    expected_case_ids = list(same_policy) + list(compatibility)
    if case_ids != expected_case_ids:
        failures.append(
            f"manifest case_ids={case_ids!r}, "
            f"expected {expected_case_ids!r}"
        )
    if manifest.get("legacy_compatibility_case_ids") != list(compatibility):
        failures.append(
            "manifest legacy_compatibility_case_ids disagree with "
            "compatibility_case_ids"
        )
    if manifest.get("canonical") is True:
        if list(same_policy) != list(all_same_policy):
            failures.append("canonical manifest same-policy matrix incomplete")
        if list(compatibility) != list(all_compatibility):
            failures.append(
                "canonical manifest compatibility matrix incomplete"
            )
    elif manifest.get("canonical") is not False:
        failures.append("manifest canonical must be boolean")
    return same_policy, compatibility, failures


def _comparison_policy_name(case) -> str:
    return (
        "same_policy_heuristic_exact_width"
        if hasattr(case, "mode")
        else "legacy_auto_vs_heuristic"
    )


def _policy_identity(case, flash_attn_version: str) -> dict:
    return {
        "flash_attn_version": flash_attn_version,
        "split_policy_name": case.split_policy_name,
        "flash_attn_num_splits": case.flash_attn_num_splits,
        "comparison_policy_name": _comparison_policy_name(case),
    }


def _index_unique(
    rows: list[dict],
    *,
    evidence_name: str,
) -> tuple[dict[str, dict], list[str]]:
    indexed = {}
    failures = []
    for row_index, row in enumerate(rows):
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            failures.append(
                f"{evidence_name}: row {row_index} missing case_id"
            )
            continue
        if case_id in indexed:
            failures.append(
                f"{evidence_name}: duplicate case_id {case_id}"
            )
            continue
        indexed[case_id] = row
    return indexed, failures


def _index_step_rows(
    rows: list[dict],
    *,
    evidence_name: str,
) -> tuple[dict[tuple[str, int], dict], list[str]]:
    indexed = {}
    failures = []
    for row_index, row in enumerate(rows):
        case_id = row.get("case_id")
        step_id = row.get("step_id")
        if not isinstance(case_id, str) or not isinstance(step_id, int):
            failures.append(
                f"{evidence_name}: row {row_index} missing case_id/step_id"
            )
            continue
        key = (case_id, step_id)
        if key in indexed:
            failures.append(f"{evidence_name}: duplicate row {key}")
            continue
        indexed[key] = row
    return indexed, failures


def _parse_sha256sums(run_dir: Path) -> tuple[dict[str, str], list[str]]:
    path = run_dir / "sha256sums.txt"
    if not path.is_file():
        return {}, ["sha256sums.txt missing"]
    sums = {}
    failures = []
    payload = path.read_text(encoding="utf-8")
    if payload and not payload.endswith("\n"):
        failures.append("sha256sums.txt lacks final newline")
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if "  " not in line:
            failures.append(
                f"sha256sums.txt line {line_number} malformed"
            )
            continue
        digest, relative = line.split("  ", 1)
        if relative in sums:
            failures.append(f"sha256sums.txt duplicate path {relative}")
            continue
        sums[relative] = digest
    return sums, failures


def _resolve_artifact(
    run_dir: Path,
    record: dict,
    *,
    label: str,
    sha256sums: dict[str, str],
    failures: list[str],
) -> Path | None:
    if not isinstance(record, dict):
        failures.append(f"{label}: artifact record missing")
        return None
    relative_value = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(relative_value, str) or not relative_value:
        failures.append(f"{label}: artifact path missing")
        return None
    if not isinstance(expected_hash, str) or not expected_hash:
        failures.append(f"{label}: missing artifact hash")
        return None
    relative = Path(relative_value)
    if relative.is_absolute():
        failures.append(f"{label}: artifact path is absolute")
        return None
    resolved = (run_dir / relative).resolve()
    try:
        resolved.relative_to(run_dir.resolve())
    except ValueError:
        failures.append(f"{label}: artifact path escapes run directory")
        return None
    if not resolved.is_file():
        failures.append(f"{label}: artifact missing at {relative_value}")
        return None
    actual_hash = contract.sha256_file(resolved)
    if actual_hash != expected_hash:
        failures.append(f"{label}: artifact hash mismatch")
    listed_hash = sha256sums.get(relative.as_posix())
    if listed_hash is None:
        failures.append(f"{label}: missing from sha256sums.txt")
    elif listed_hash != actual_hash:
        failures.append(f"{label}: sha256sums.txt mismatch")
    return resolved


def _validate_manifest(
    manifest: dict,
    source_evidence: dict,
    environment: dict,
    prompt_manifest: dict,
    *,
    same_policy_cases: dict[str, object] | None = None,
    compatibility_cases: dict[str, object] | None = None,
) -> list[str]:
    failures = []
    if same_policy_cases is None or compatibility_cases is None:
        (
            parsed_same_policy,
            parsed_compatibility,
            domain_failures,
        ) = _manifest_case_domain(manifest)
        failures.extend(domain_failures)
        if same_policy_cases is None:
            same_policy_cases = parsed_same_policy
        if compatibility_cases is None:
            compatibility_cases = parsed_compatibility
    same_policy_ids = list(same_policy_cases)
    compatibility_ids = list(compatibility_cases)
    expected_ids = same_policy_ids + compatibility_ids
    expected_values = {
        "schema_version": 1,
        "kind": "heuristic_exact_width_recovery",
        "source_tree_sha256": source_evidence.get("tree_sha256"),
        "environment_sha256": contract.canonical_json_sha256(environment),
        "prompt_manifest_sha256": contract.canonical_json_sha256(
            prompt_manifest
        ),
        "case_ids": expected_ids,
        "same_policy_case_ids": same_policy_ids,
        "compatibility_case_ids": compatibility_ids,
        "legacy_compatibility_case_ids": compatibility_ids,
        "same_policy_process_count": len(same_policy_ids),
        "compatibility_process_count": len(compatibility_ids),
        "compatibility_pair_count": len(compatibility_ids) // 2,
        "flash_attn_version": environment.get("flash_attention"),
        "policy_name": contract.HEURISTIC_POLICY_NAME,
        "auto_split_count": contract.AUTO_FLASH_ATTN_NUM_SPLITS,
        "warmup_steps": contract.WARMUP_STEPS,
        "measured_steps": contract.MEASURED_STEPS,
        "logit_rtol": contract.LOGIT_RTOL,
        "logit_atol": contract.LOGIT_ATOL,
    }
    for field, expected in expected_values.items():
        if manifest.get(field) != expected:
            failures.append(
                f"manifest {field}={manifest.get(field)!r}, "
                f"expected {expected!r}"
            )
    if (
        environment.get("source_tree_sha256")
        != source_evidence.get("tree_sha256")
    ):
        failures.append(
            "environment source_tree_sha256 does not match source evidence"
        )
    if environment.get("bf16_supported") is not True:
        failures.append("environment does not report BF16 support")
    return failures


def _validate_process_rows(
    rows: list[dict],
    *,
    manifest: dict,
    environment: dict,
    prompt_manifest: dict,
    expected_cases: dict[str, object],
) -> tuple[dict[str, dict], list[str]]:
    indexed, failures = _index_unique(
        rows,
        evidence_name="process_rows",
    )
    ports = {}
    for row_index, row in enumerate(rows):
        owner = row.get("case_id", f"row {row_index}")
        for port_field in ("tinyvllm_dist_port", "master_port"):
            port = row.get(port_field)
            if not isinstance(port, int):
                failures.append(
                    f"process_rows: {owner} {port_field} missing"
                )
                continue
            if port in ports:
                failures.append(
                    f"process_rows: reused port {port} by "
                    f"{ports[port]} and {owner}"
                )
            else:
                ports[port] = owner
        if row.get("tinyvllm_dist_port") == row.get("master_port"):
            failures.append(f"process_rows: {owner} ports are identical")

    actual_ids = set(indexed)
    expected_ids = set(expected_cases)
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    if missing:
        failures.append(f"process_rows: missing {missing}")
    if unexpected:
        failures.append(f"process_rows: unexpected {unexpected}")

    source_hash = manifest.get("source_tree_sha256")
    environment_hash = manifest.get("environment_sha256")
    for case_id, row in indexed.items():
        case = expected_cases.get(case_id)
        if case is None:
            continue
        for field, expected in {
            **asdict(case),
            **_policy_identity(
                case,
                str(environment.get("flash_attention")),
            ),
            "case_id": case.case_id,
            "status": "PASS",
            "source_tree_sha256": source_hash,
            "environment_sha256": environment_hash,
        }.items():
            if field not in row or row.get(field) != expected:
                failures.append(
                    f"process_rows: {case_id} {field}="
                    f"{row.get(field)!r}, expected {expected!r}"
                )
        expected_prompt_hash = (
            prompt_manifest.get("trajectories", {})
            .get(case.trajectory, {})
            .get(str(case.batch_size))
        )
        if row.get("prompt_sha256") != expected_prompt_hash:
            failures.append(
                f"process_rows: {case_id} prompt_sha256 mismatch"
            )
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, dict):
            failures.append(f"process_rows: {case_id} artifacts missing")
        if not isinstance(row.get("reference_tokens"), dict):
            failures.append(
                f"process_rows: {case_id} reference_tokens missing"
            )
    if environment_hash != contract.canonical_json_sha256(environment):
        failures.append("environment_sha256 mismatch")
    return indexed, failures


def _validate_step_evidence(
    rows: list[dict],
    *,
    evidence_name: str,
    process_rows: dict[str, dict],
    expected_cases: dict[str, object],
) -> tuple[dict[tuple[str, int], dict], list[str]]:
    indexed, failures = _index_step_rows(
        rows,
        evidence_name=evidence_name,
    )
    expected_keys = {
        (case.case_id, step_id)
        for case in expected_cases.values()
        for step_id in range(contract.MEASURED_STEPS)
    }
    actual_keys = set(indexed)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing:
        failures.append(f"{evidence_name}: missing {missing[:8]}")
    if unexpected:
        failures.append(f"{evidence_name}: unexpected {unexpected[:8]}")
    policy_fields = (
        "flash_attn_version",
        "split_policy_name",
        "flash_attn_num_splits",
        "comparison_policy_name",
    )
    for (case_id, step_id), row in indexed.items():
        process = process_rows.get(case_id)
        if process is None:
            continue
        for field in policy_fields:
            if row.get(field) != process.get(field):
                failures.append(
                    f"{evidence_name}: {case_id} step {step_id} "
                    f"{field}={row.get(field)!r}, "
                    f"expected {process.get(field)!r}"
                )
    return indexed, failures


def _load_tensor_shard(
    path: Path,
    *,
    case,
    label: str,
    failures: list[str],
) -> dict | None:
    import torch

    try:
        shard = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        failures.append(f"{label}: failed to load tensor shard: {exc}")
        return None
    required = {
        "schema_version",
        "case_id",
        "step_ids",
        "row_ids",
    }
    missing = sorted(required - set(shard))
    if missing:
        failures.append(f"{label}: shard missing {missing}")
        return None
    if shard.get("schema_version") != 1:
        failures.append(f"{label}: schema_version mismatch")
    if shard.get("case_id") != case.case_id:
        failures.append(f"{label}: case_id mismatch")
    for field, expected in _policy_identity(
        case,
        str(shard.get("flash_attn_version")),
    ).items():
        if field == "flash_attn_version":
            if not isinstance(shard.get(field), str) or not shard.get(field):
                failures.append(f"{label}: {field} missing")
        elif shard.get(field) != expected:
            failures.append(f"{label}: {field} mismatch")
    if shard.get("step_ids") != list(range(contract.MEASURED_STEPS)):
        failures.append(f"{label}: step_ids mismatch")
    if shard.get("row_ids") != list(range(case.batch_size)):
        failures.append(f"{label}: row_ids mismatch")
    return shard


def _validate_reference_tokens(
    run_dir: Path,
    process_rows: dict[str, dict],
    sha256sums: dict[str, str],
    failures: list[str],
    expected_cases: dict[str, object],
) -> None:
    expected_by_comparison = {}
    for case in expected_cases.values():
        row = process_rows.get(case.case_id)
        if row is None:
            continue
        path = _resolve_artifact(
            run_dir,
            row.get("reference_tokens"),
            label=f"{case.case_id} reference_tokens",
            sha256sums=sha256sums,
            failures=failures,
        )
        if path is None:
            continue
        try:
            tokens = _read_json(path)
        except Exception as exc:
            failures.append(
                f"{case.case_id} reference_tokens invalid: {exc}"
            )
            continue
        expected_shape = (
            contract.WARMUP_STEPS + contract.MEASURED_STEPS,
            case.batch_size,
        )
        if (
            not isinstance(tokens, list)
            or len(tokens) != expected_shape[0]
            or any(
                not isinstance(item, list)
                or len(item) != expected_shape[1]
                for item in tokens
            )
        ):
            failures.append(
                f"{case.case_id} reference token shape mismatch"
            )
            continue
        digest = contract.canonical_json_sha256(tokens)
        if row.get("reference_token_sha256") != digest:
            failures.append(
                f"process_rows: {case.case_id} "
                "reference_token_sha256 mismatch"
            )
        identity = (_comparison_policy_name(case), _case_identity(case))
        previous = expected_by_comparison.setdefault(identity, digest)
        if previous != digest:
            failures.append(
                f"process_rows: {case.case_id} reference token drift"
            )


def _first_tensor_divergence(
    eager,
    candidate,
    *,
    evidence: str,
    case_id: str,
) -> dict | None:
    import torch

    if eager.shape != candidate.shape:
        return {
            "evidence": evidence,
            "case_id": case_id,
            "kind": "shape_mismatch",
        }
    finite_mask = torch.isfinite(eager) & torch.isfinite(candidate)
    if not bool(finite_mask.all().item()):
        index = tuple(
            int(item)
            for item in (
                ~finite_mask
            ).nonzero(as_tuple=False)[0].tolist()
        )
        detail = {
            "evidence": evidence,
            "case_id": case_id,
            "kind": "nonfinite",
            "step_id": index[0],
        }
        if evidence == "logits":
            detail["row_id"] = index[1]
        else:
            detail.update(
                {
                    "component_id": index[1],
                    "layer_id": index[2],
                    "row_id": index[3],
                }
            )
        return detail
    if evidence == "logits":
        eager_argmax = torch.argmax(eager, dim=-1)
        candidate_argmax = torch.argmax(candidate, dim=-1)
        mismatches = (eager_argmax != candidate_argmax).nonzero(
            as_tuple=False
        )
        if mismatches.numel():
            step_id, row_id = [
                int(item) for item in mismatches[0].tolist()
            ]
            return {
                "evidence": evidence,
                "case_id": case_id,
                "kind": "argmax_mismatch",
                "step_id": step_id,
                "row_id": row_id,
            }
    close_mask = torch.isclose(
        eager.float(),
        candidate.float(),
        rtol=contract.LOGIT_RTOL,
        atol=contract.LOGIT_ATOL,
    )
    mismatches = (~close_mask).nonzero(as_tuple=False)
    if mismatches.numel():
        index = [int(item) for item in mismatches[0].tolist()]
        detail = {
            "evidence": evidence,
            "case_id": case_id,
            "kind": "close_failure",
            "step_id": index[0],
        }
        if evidence == "logits":
            detail["row_id"] = index[1]
        else:
            detail.update(
                {
                    "component_id": index[1],
                    "layer_id": index[2],
                    "row_id": index[3],
                }
            )
        return detail
    return None


def _compare_logits_and_layers(
    run_dir: Path,
    process_rows: dict[str, dict],
    sha256sums: dict[str, str],
    failures: list[str],
    expected_cases: dict[str, object],
) -> tuple[list[dict], list[dict], list[dict]]:
    eager_shards = {}
    logit_results = []
    layer_results = []
    divergences = []
    for case in expected_cases.values():
        row = process_rows.get(case.case_id)
        if row is None:
            continue
        artifacts = row.get("artifacts", {})
        loaded = {}
        for evidence in ("logits", "layers"):
            path = _resolve_artifact(
                run_dir,
                artifacts.get(evidence),
                label=f"{case.case_id} {evidence}",
                sha256sums=sha256sums,
                failures=failures,
            )
            if path is None:
                loaded[evidence] = None
                continue
            shard = _load_tensor_shard(
                path,
                case=case,
                label=f"{case.case_id} {evidence}",
                failures=failures,
            )
            loaded[evidence] = shard
            if shard is None:
                continue
            tensor = shard.get("tensor")
            if tensor is None:
                failures.append(
                    f"{case.case_id} {evidence}: tensor missing"
                )
                continue
            if shard.get("dtype") != str(tensor.dtype):
                failures.append(
                    f"{case.case_id} {evidence}: dtype metadata mismatch"
                )
            if shard.get("shape") != list(tensor.shape):
                failures.append(
                    f"{case.case_id} {evidence}: shape metadata mismatch"
                )
        layer_shard = loaded.get("layers")
        if layer_shard is not None and layer_shard.get("tensor") is not None:
            layer_tensor = layer_shard["tensor"]
            if layer_tensor.ndim != 5 or layer_tensor.shape[1] != 2:
                failures.append(
                    f"{case.case_id} layers: expected "
                    "[step, component, layer, row, hidden]"
                )
            layer_ids = layer_shard.get("layer_ids")
            if layer_ids is not None and layer_ids != list(
                range(layer_tensor.shape[2])
            ):
                failures.append(f"{case.case_id} layers: layer_ids mismatch")
            component_ids = layer_shard.get("component_ids")
            if component_ids is not None and component_ids != [
                "hidden_states",
                "residual",
            ]:
                failures.append(
                    f"{case.case_id} layers: component_ids mismatch"
                )
        identity = _case_identity(case)
        if case.mode == "candidate_eager_heuristic":
            eager_shards[identity] = loaded
            continue
        eager = eager_shards.get(identity)
        if eager is None:
            failures.append(f"{case.case_id}: eager predecessor missing")
            continue
        common = {
            "case_id": case.case_id,
            "mode": case.mode,
            "batch_size": case.batch_size,
            "graph_size": case.graph_size,
        }
        logit_comparison = {
            "shape_equal": False,
            "dtype_equal": False,
            "finite": False,
            "argmax_equal": False,
            "close": False,
        }
        if (
            loaded.get("logits") is not None
            and eager.get("logits") is not None
            and loaded["logits"].get("tensor") is not None
            and eager["logits"].get("tensor") is not None
        ):
            logit_comparison = contract.compare_tensor_pair(
                eager["logits"]["tensor"],
                loaded["logits"]["tensor"],
            )
            detail = _first_tensor_divergence(
                eager["logits"]["tensor"],
                loaded["logits"]["tensor"],
                evidence="logits",
                case_id=case.case_id,
            )
            if detail is not None:
                divergences.append(detail)
        logit_results.append({**common, **logit_comparison})

        layer_finite = False
        layer_close = False
        observed_layers = 0
        required_layers = 0
        if (
            loaded.get("layers") is not None
            and eager.get("layers") is not None
            and loaded["layers"].get("tensor") is not None
            and eager["layers"].get("tensor") is not None
        ):
            candidate_tensor = loaded["layers"]["tensor"]
            eager_tensor = eager["layers"]["tensor"]
            observed_layers = (
                int(candidate_tensor.shape[2])
                if candidate_tensor.ndim == 5
                else 0
            )
            required_layers = (
                int(eager_tensor.shape[2])
                if eager_tensor.ndim == 5
                else 0
            )
            candidate_layer_ids = loaded["layers"].get("layer_ids")
            if candidate_layer_ids is None:
                candidate_layer_ids = list(range(observed_layers))
            if (
                observed_layers != required_layers
                or candidate_layer_ids != list(range(required_layers))
            ):
                failures.append(
                    f"{case.case_id} layers: layer_ids incomplete"
                )
            comparison = contract.compare_tensor_pair(
                eager_tensor,
                candidate_tensor,
            )
            layer_finite = comparison["finite"]
            layer_close = comparison["close"]
            detail = _first_tensor_divergence(
                eager_tensor,
                candidate_tensor,
                evidence="layers",
                case_id=case.case_id,
            )
            if detail is not None:
                divergences.append(detail)
        layer_results.append(
            {
                **common,
                "required_layer_count": required_layers,
                "observed_layer_count": observed_layers,
                "finite": layer_finite,
                "close": layer_close,
            }
        )
    return logit_results, layer_results, divergences


def _kv_tensor_equal(left, right) -> bool:
    import torch

    return bool(torch.equal(left, right))


def _compare_kv(
    run_dir: Path,
    process_rows: dict[str, dict],
    sha256sums: dict[str, str],
    failures: list[str],
    expected_cases: dict[str, object],
) -> tuple[list[dict], list[dict]]:
    eager_shards = {}
    results = []
    divergences = []
    for case in expected_cases.values():
        row = process_rows.get(case.case_id)
        if row is None:
            continue
        path = _resolve_artifact(
            run_dir,
            row.get("artifacts", {}).get("kv"),
            label=f"{case.case_id} kv",
            sha256sums=sha256sums,
            failures=failures,
        )
        if path is None:
            shard = None
        else:
            shard = _load_tensor_shard(
                path,
                case=case,
                label=f"{case.case_id} kv",
                failures=failures,
            )
        if shard is not None:
            required = {
                "slot_ids",
                "plans",
                "keys_before",
                "values_before",
                "keys_after",
                "values_after",
            }
            missing = sorted(required - set(shard))
            if missing:
                failures.append(f"{case.case_id} kv: missing {missing}")
                shard = None
            elif (
                len(shard["slot_ids"]) != contract.MEASURED_STEPS
                or len(shard["plans"]) != contract.MEASURED_STEPS
            ):
                failures.append(f"{case.case_id} kv: step metadata mismatch")
                shard = None
        identity = _case_identity(case)
        if case.mode == "candidate_eager_heuristic":
            eager_shards[identity] = shard
            continue
        common = {
            "case_id": case.case_id,
            "mode": case.mode,
            "batch_size": case.batch_size,
            "graph_size": case.graph_size,
        }
        eager = eager_shards.get(identity)
        active_equal = False
        unexpected = []
        if shard is not None and eager is not None:
            active_equal = True
            for step_id in range(contract.MEASURED_STEPS):
                slots = shard["slot_ids"][step_id]
                plan = shard["plans"][step_id]
                eager_slots = eager["slot_ids"][step_id]
                active_slots = plan.get("active_write_slots", [])
                for slot in active_slots:
                    if slot not in slots or slot not in eager_slots:
                        failures.append(
                            f"{case.case_id} kv: active slot {slot} missing"
                        )
                        active_equal = False
                        continue
                    candidate_index = slots.index(slot)
                    eager_index = eager_slots.index(slot)
                    for tensor_name in ("keys_after", "values_after"):
                        if not _kv_tensor_equal(
                            shard[tensor_name][step_id, :, candidate_index],
                            eager[tensor_name][step_id, :, eager_index],
                        ):
                            active_equal = False
                            if not divergences:
                                divergences.append(
                                    {
                                        "evidence": "kv",
                                        "case_id": case.case_id,
                                        "kind": "active_kv_mismatch",
                                        "step_id": step_id,
                                        "slot_id": slot,
                                    }
                                )
                protected = []
                protected.append((plan.get("slot_zero"), "slot_zero_mutation"))
                protected.extend(
                    (slot, "inactive_slot_mutation")
                    for slot in plan.get("inactive_declared_slots", [])
                )
                protected.extend(
                    (slot, "sentinel_mutation")
                    for slot in plan.get("sentinel_slots", [])
                )
                for slot, kind in protected:
                    if slot is None or slot not in slots:
                        failures.append(
                            f"{case.case_id} kv: protected slot {slot} missing"
                        )
                        continue
                    slot_index = slots.index(slot)
                    mutated = any(
                        not _kv_tensor_equal(
                            shard[after_name][step_id, :, slot_index],
                            shard[before_name][step_id, :, slot_index],
                        )
                        for before_name, after_name in (
                            ("keys_before", "keys_after"),
                            ("values_before", "values_after"),
                        )
                    )
                    if mutated:
                        unexpected.append(slot)
                        if not divergences:
                            divergences.append(
                                {
                                    "evidence": "kv",
                                    "case_id": case.case_id,
                                    "kind": kind,
                                    "step_id": step_id,
                                    "slot_id": slot,
                                }
                            )
            unexpected = list(dict.fromkeys(unexpected))
        results.append(
            {
                **common,
                "active_slots_equal": active_equal,
                "unexpected_slot_mutations": unexpected,
            }
        )
    return results, divergences


def _load_compatibility_artifacts(
    run_dir: Path,
    process_rows: dict[str, dict],
    sha256sums: dict[str, str],
    failures: list[str],
    expected_cases: dict[str, object],
) -> dict[str, dict[str, dict | None]]:
    loaded_by_pair = {}
    for case in expected_cases.values():
        row = process_rows.get(case.case_id)
        if row is None:
            continue
        artifacts = row.get("artifacts", {})
        loaded = {}
        for evidence in ("logits", "kv"):
            path = _resolve_artifact(
                run_dir,
                artifacts.get(evidence),
                label=f"{case.case_id} {evidence}",
                sha256sums=sha256sums,
                failures=failures,
            )
            if path is None:
                loaded[evidence] = None
                continue
            shard = _load_tensor_shard(
                path,
                case=case,
                label=f"{case.case_id} {evidence}",
                failures=failures,
            )
            loaded[evidence] = shard
        logits = loaded.get("logits")
        if logits is not None:
            tensor = logits.get("tensor")
            if tensor is None:
                failures.append(f"{case.case_id} logits: tensor missing")
                loaded["logits"] = None
            else:
                if logits.get("dtype") != str(tensor.dtype):
                    failures.append(
                        f"{case.case_id} logits: dtype metadata mismatch"
                    )
                if logits.get("shape") != list(tensor.shape):
                    failures.append(
                        f"{case.case_id} logits: shape metadata mismatch"
                    )
        kv = loaded.get("kv")
        if kv is not None:
            required = {
                "slot_ids",
                "plans",
                "keys_before",
                "values_before",
                "keys_after",
                "values_after",
            }
            missing = sorted(required - set(kv))
            if missing:
                failures.append(f"{case.case_id} kv: missing {missing}")
                loaded["kv"] = None
            elif (
                len(kv["slot_ids"]) != contract.MEASURED_STEPS
                or len(kv["plans"]) != contract.MEASURED_STEPS
            ):
                failures.append(f"{case.case_id} kv: step metadata mismatch")
                loaded["kv"] = None
        loaded_by_pair.setdefault(case.pair_id, {})[case.policy] = loaded
    return loaded_by_pair


def _protected_kv_mutations(case_id: str, shard: dict) -> list[dict]:
    mutations = []
    for step_id in range(contract.MEASURED_STEPS):
        slots = shard["slot_ids"][step_id]
        plan = shard["plans"][step_id]
        protected = [(plan.get("slot_zero"), "slot_zero_mutation")]
        protected.extend(
            (slot, "inactive_slot_mutation")
            for slot in plan.get("inactive_declared_slots", [])
        )
        protected.extend(
            (slot, "sentinel_mutation")
            for slot in plan.get("sentinel_slots", [])
        )
        for slot, kind in protected:
            if slot is None or slot not in slots:
                mutations.append(
                    {
                        "case_id": case_id,
                        "kind": "protected_slot_missing",
                        "step_id": step_id,
                        "slot_id": slot,
                    }
                )
                continue
            slot_index = slots.index(slot)
            mutated = any(
                not _kv_tensor_equal(
                    shard[after_name][step_id, :, slot_index],
                    shard[before_name][step_id, :, slot_index],
                )
                for before_name, after_name in (
                    ("keys_before", "keys_after"),
                    ("values_before", "values_after"),
                )
            )
            if mutated:
                mutations.append(
                    {
                        "case_id": case_id,
                        "kind": kind,
                        "step_id": step_id,
                        "slot_id": slot,
                    }
                )
    return mutations


def _compare_legacy_compatibility(
    run_dir: Path,
    process_rows: dict[str, dict],
    raw_rows: dict[tuple[str, int], dict],
    sha256sums: dict[str, str],
    failures: list[str],
    expected_cases: dict[str, object],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    loaded_by_pair = _load_compatibility_artifacts(
        run_dir,
        process_rows,
        sha256sums,
        failures,
        expected_cases,
    )
    logit_results = []
    token_results = []
    kv_results = []
    divergences = []
    candidate_cases = (
        case
        for case in expected_cases.values()
        if case.policy == "candidate_eager_heuristic"
    )
    for case in candidate_cases:
        pair = loaded_by_pair.get(case.pair_id, {})
        legacy = pair.get("legacy_eager_auto", {})
        candidate = pair.get("candidate_eager_heuristic", {})
        common = {
            "pair_id": case.pair_id,
            "batch_size": case.batch_size,
            "trajectory": case.trajectory,
            "repetition": case.repetition,
            "comparison_policy_name": "legacy_auto_vs_heuristic",
        }

        logit_comparison = {
            "shape_equal": False,
            "dtype_equal": False,
            "finite": False,
            "argmax_equal": False,
            "close": False,
        }
        legacy_logits = legacy.get("logits")
        candidate_logits = candidate.get("logits")
        if (
            legacy_logits is not None
            and candidate_logits is not None
            and legacy_logits.get("tensor") is not None
            and candidate_logits.get("tensor") is not None
        ):
            logit_comparison = contract.compare_tensor_pair(
                legacy_logits["tensor"],
                candidate_logits["tensor"],
            )
            detail = _first_tensor_divergence(
                legacy_logits["tensor"],
                candidate_logits["tensor"],
                evidence="logits",
                case_id=case.case_id,
            )
            if detail is not None:
                detail["evidence"] = "legacy_compatibility_logits"
                detail["pair_id"] = case.pair_id
                divergences.append(detail)
        logit_results.append({**common, **logit_comparison})

        legacy_case_id = next(
            item.case_id
            for item in expected_cases.values()
            if item.pair_id == case.pair_id
            and item.policy == "legacy_eager_auto"
        )
        legacy_tokens = [
            raw_rows.get((legacy_case_id, step_id), {}).get(
                "observed_argmax_token_ids"
            )
            for step_id in range(contract.MEASURED_STEPS)
        ]
        candidate_tokens = [
            raw_rows.get((case.case_id, step_id), {}).get(
                "observed_argmax_token_ids"
            )
            for step_id in range(contract.MEASURED_STEPS)
        ]
        tokens_equal = legacy_tokens == candidate_tokens
        token_results.append({**common, "tokens_equal": tokens_equal})
        if not tokens_equal:
            mismatch_step = next(
                step_id
                for step_id, (legacy_row, candidate_row) in enumerate(
                    zip(legacy_tokens, candidate_tokens)
                )
                if legacy_row != candidate_row
            )
            divergences.append(
                {
                    "evidence": "legacy_compatibility_tokens",
                    "case_id": case.case_id,
                    "pair_id": case.pair_id,
                    "kind": "token_mismatch",
                    "step_id": mismatch_step,
                }
            )

        legacy_kv = legacy.get("kv")
        candidate_kv = candidate.get("kv")
        touched_slot_sets_equal = False
        unexpected_mutations = []
        if legacy_kv is not None and candidate_kv is not None:
            legacy_touched = [
                set(plan.get("active_write_slots", []))
                for plan in legacy_kv["plans"]
            ]
            candidate_touched = [
                set(plan.get("active_write_slots", []))
                for plan in candidate_kv["plans"]
            ]
            touched_slot_sets_equal = legacy_touched == candidate_touched
            unexpected_mutations = _protected_kv_mutations(
                legacy_case_id,
                legacy_kv,
            ) + _protected_kv_mutations(case.case_id, candidate_kv)
            if not touched_slot_sets_equal:
                mismatch_step = next(
                    step_id
                    for step_id, (legacy_slots, candidate_slots) in enumerate(
                        zip(legacy_touched, candidate_touched)
                    )
                    if legacy_slots != candidate_slots
                )
                divergences.append(
                    {
                        "evidence": "legacy_compatibility_kv",
                        "case_id": case.case_id,
                        "pair_id": case.pair_id,
                        "kind": "touched_slot_set_mismatch",
                        "step_id": mismatch_step,
                    }
                )
            divergences.extend(
                {
                    "evidence": "legacy_compatibility_kv",
                    "pair_id": case.pair_id,
                    **mutation,
                }
                for mutation in unexpected_mutations
            )
        kv_results.append(
            {
                **common,
                "touched_slot_sets_equal": touched_slot_sets_equal,
                "unexpected_slot_mutations": unexpected_mutations,
            }
        )
    return logit_results, token_results, kv_results, divergences


def _verify_layer_rows(
    layer_rows: dict[tuple[str, int], dict],
    process_rows: dict[str, dict],
    expected_cases: dict[str, object],
) -> list[str]:
    failures = []
    for case in expected_cases.values():
        process = process_rows.get(case.case_id)
        if process is None:
            continue
        for step_id in range(contract.MEASURED_STEPS):
            row = layer_rows.get((case.case_id, step_id))
            if row is None:
                continue
            required = row.get("required_layer_count")
            observed = row.get("observed_layer_count")
            layer_ids = row.get("layer_ids")
            if not isinstance(required, int) or required <= 0:
                failures.append(
                    f"layer_rows: {case.case_id} step {step_id} "
                    "required_layer_count invalid"
                )
            if observed != required:
                failures.append(
                    f"layer_rows: {case.case_id} step {step_id} "
                    "observed_layer_count mismatch"
                )
            if layer_ids is not None and layer_ids != list(range(required)):
                failures.append(
                    f"layer_rows: {case.case_id} step {step_id} "
                    "layer_ids mismatch"
                )
            if row.get("finite") is not True:
                failures.append(
                    f"layer_rows: {case.case_id} step {step_id} nonfinite"
                )
    return failures


def _verify_raw_rows_against_logits(
    raw_rows: dict[tuple[str, int], dict],
    process_rows: dict[str, dict],
    run_dir: Path,
    expected_cases: dict[str, object],
) -> list[str]:
    failures = []
    for case in expected_cases.values():
        process = process_rows.get(case.case_id)
        if process is None:
            continue
        record = process.get("artifacts", {}).get("logits")
        if not isinstance(record, dict) or not isinstance(
            record.get("path"), str
        ):
            continue
        path = run_dir / record["path"]
        if not path.is_file():
            continue
        for step_id in range(contract.MEASURED_STEPS):
            row = raw_rows.get((case.case_id, step_id))
            if row is None:
                continue
            observed = row.get("observed_argmax_token_ids")
            if not isinstance(observed, list) or len(observed) != (
                case.batch_size
            ):
                failures.append(
                    f"raw_rows: {case.case_id} step {step_id} "
                    "observed argmax shape mismatch"
                )
            reference = row.get("reference_next_input_token_ids")
            if not isinstance(reference, list) or len(reference) != (
                case.batch_size
            ):
                failures.append(
                    f"raw_rows: {case.case_id} step {step_id} "
                    "reference row shape mismatch"
                )
    return failures


def _report_markdown(summary: dict) -> str:
    lines = [
        "# Multi-Sequence CUDA Graph Independent Verification",
        "",
        f"- Classification: `{summary['classification']}`",
        (
            "- Rounded classification: "
            f"`{summary['rounded_classification']}`"
        ),
        (
            "- Legacy compatibility: "
            f"`{summary['legacy_compatibility']}`"
        ),
        f"- Policy integrity: `{summary['policy_integrity']}`",
        f"- Case count: `{summary['case_count']}`",
        (
            "- Same-policy case count: "
            f"`{summary['same_policy_case_count']}`"
        ),
        (
            "- Compatibility process count: "
            f"`{summary['compatibility_process_count']}`"
        ),
        (
            "- Compatibility pair count: "
            f"`{summary['compatibility_pair_count']}`"
        ),
        f"- Structural failures: `{len(summary['failures'])}`",
    ]
    if summary.get("first_divergence") is not None:
        lines.extend(
            [
                "",
                "## First Divergence",
                "",
                "```json",
                json.dumps(
                    summary["first_divergence"],
                    indent=2,
                    sort_keys=True,
                ),
                "```",
            ]
        )
    if summary["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {item}" for item in summary["failures"])
    return "\n".join(lines) + "\n"


def _write_verification_outputs(run_dir: Path, summary: dict) -> None:
    output_dir = run_dir / "independent-verification"
    _atomic_write_json(output_dir / "summary.json", summary)
    _atomic_write_bytes(
        output_dir / "report.md",
        _report_markdown(summary).encode("utf-8"),
    )
    exitcode = 0 if (
        summary["classification"] == "EXACT_REPLAY_CORRECT"
        and summary["legacy_compatibility"] == "LEGACY_COMPATIBLE"
        and summary["policy_integrity"] == "POLICY_EXACT"
        and not summary["failures"]
    ) else 1
    _atomic_write_bytes(
        output_dir / "verify.exitcode",
        f"{exitcode}\n".encode("ascii"),
    )


def verify_diagnostic(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    failures = []
    divergences = []
    required_json = (
        "manifest.json",
        "source_evidence.json",
        "environment.json",
        "prompt_manifest.json",
        "summary.json",
    )
    loaded = {}
    for name in required_json:
        path = run_dir / name
        if not path.is_file():
            failures.append(f"{name} missing")
            loaded[name] = {}
            continue
        try:
            loaded[name] = _read_json(path)
        except Exception as exc:
            failures.append(f"{name} invalid: {exc}")
            loaded[name] = {}
    manifest = loaded["manifest.json"]
    source_evidence = loaded["source_evidence.json"]
    environment = loaded["environment.json"]
    prompt_manifest = loaded["prompt_manifest.json"]
    (
        same_policy_cases,
        compatibility_cases,
        domain_failures,
    ) = _manifest_case_domain(manifest)
    failures.extend(domain_failures)
    expected_cases = {
        **same_policy_cases,
        **compatibility_cases,
    }
    failures.extend(
        _validate_manifest(
            manifest,
            source_evidence,
            environment,
            prompt_manifest,
            same_policy_cases=same_policy_cases,
            compatibility_cases=compatibility_cases,
        )
    )

    sha256sums, checksum_failures = _parse_sha256sums(run_dir)
    failures.extend(checksum_failures)
    for relative, expected_hash in sha256sums.items():
        path = (run_dir / relative).resolve()
        try:
            path.relative_to(run_dir.resolve())
        except ValueError:
            failures.append(f"sha256sums.txt path escapes run dir: {relative}")
            continue
        if not path.is_file():
            failures.append(f"sha256sums.txt path missing: {relative}")
            continue
        if contract.sha256_file(path) != expected_hash:
            failures.append(f"sha256sums.txt hash mismatch: {relative}")

    jsonl_names = (
        "process_rows.jsonl",
        "raw_rows.jsonl",
        "layer_observations.jsonl",
        "kv_observations.jsonl",
    )
    jsonl_rows = {}
    for name in jsonl_names:
        path = run_dir / name
        if not path.is_file():
            failures.append(f"{name} missing")
            jsonl_rows[name] = []
            continue
        try:
            jsonl_rows[name] = _read_jsonl(path)
        except Exception as exc:
            failures.append(f"{name} invalid: {exc}")
            jsonl_rows[name] = []
        listed_hash = sha256sums.get(name)
        if listed_hash is None:
            failures.append(f"{name} missing from sha256sums.txt")
        elif path.is_file() and contract.sha256_file(path) != listed_hash:
            failures.append(f"{name} sha256sums.txt mismatch")

    process_rows, process_failures = _validate_process_rows(
        jsonl_rows["process_rows.jsonl"],
        manifest=manifest,
        environment=environment,
        prompt_manifest=prompt_manifest,
        expected_cases=expected_cases,
    )
    failures.extend(process_failures)
    raw_rows, raw_failures = _validate_step_evidence(
        jsonl_rows["raw_rows.jsonl"],
        evidence_name="raw_rows",
        process_rows=process_rows,
        expected_cases=expected_cases,
    )
    failures.extend(raw_failures)
    layer_rows, layer_failures = _validate_step_evidence(
        jsonl_rows["layer_observations.jsonl"],
        evidence_name="layer_rows",
        process_rows=process_rows,
        expected_cases=expected_cases,
    )
    failures.extend(layer_failures)
    kv_rows, kv_row_failures = _validate_step_evidence(
        jsonl_rows["kv_observations.jsonl"],
        evidence_name="kv_rows",
        process_rows=process_rows,
        expected_cases=expected_cases,
    )
    failures.extend(kv_row_failures)
    policy_integrity = verify_policy_integrity(
        raw_rows=list(raw_rows.values()),
        layer_rows=list(layer_rows.values()),
        kv_rows=list(kv_rows.values()),
        process_rows=process_rows,
    )
    failures.extend(
        _verify_layer_rows(
            layer_rows,
            process_rows,
            expected_cases,
        )
    )
    failures.extend(
        _verify_raw_rows_against_logits(
            raw_rows,
            process_rows,
            run_dir,
            expected_cases,
        )
    )
    _validate_reference_tokens(
        run_dir,
        process_rows,
        sha256sums,
        failures,
        expected_cases,
    )

    logit_results, layer_results, tensor_divergences = (
        _compare_logits_and_layers(
            run_dir,
            process_rows,
            sha256sums,
            failures,
            same_policy_cases,
        )
    )
    divergences.extend(tensor_divergences)
    kv_results, kv_divergences = _compare_kv(
        run_dir,
        process_rows,
        sha256sums,
        failures,
        same_policy_cases,
    )
    divergences.extend(kv_divergences)
    matrix_rows = [
        row
        for case_id, row in process_rows.items()
        if case_id in same_policy_cases
    ]
    classification = contract.classify_diagnostic(
        matrix_rows=matrix_rows,
        logit_results=logit_results,
        layer_results=layer_results,
        kv_results=kv_results,
        expected_cases=same_policy_cases.values(),
    )
    failures.extend(classification.get("failures", []))
    (
        compatibility_logit_results,
        compatibility_token_results,
        compatibility_kv_results,
        compatibility_divergences,
    ) = _compare_legacy_compatibility(
        run_dir,
        process_rows,
        raw_rows,
        sha256sums,
        failures,
        compatibility_cases,
    )
    divergences.extend(compatibility_divergences)
    compatibility_process_rows = [
        row
        for case_id, row in process_rows.items()
        if case_id in compatibility_cases
    ]
    compatibility = contract.classify_legacy_compatibility(
        process_rows=compatibility_process_rows,
        logit_results=compatibility_logit_results,
        kv_results=compatibility_kv_results,
        token_results=compatibility_token_results,
        expected_cases=compatibility_cases.values(),
    )
    failures.extend(compatibility.get("failures", []))

    if failures:
        final_classification = "INCOMPLETE"
        rounded_classification = "INCOMPLETE"
        legacy_compatibility = "INCOMPLETE"
    else:
        final_classification = classification["classification"]
        rounded_classification = classification["rounded_classification"]
        legacy_compatibility = compatibility["classification"]
    producer = loaded["summary.json"]
    if manifest.get("canonical") is True:
        if (
            final_classification == "EXACT_REPLAY_CORRECT"
            and producer.get("classification") != final_classification
        ):
            failures.append("producer classification does not match evidence")
        if (
            rounded_classification == "ROUNDED_REPLAY_CORRECT"
            and producer.get("rounded_classification")
            != rounded_classification
        ):
            failures.append(
                "producer rounded classification does not match evidence"
            )
        if (
            legacy_compatibility == "LEGACY_COMPATIBLE"
            and producer.get("legacy_compatibility")
            != legacy_compatibility
        ):
            failures.append(
                "producer legacy compatibility does not match evidence"
            )
    expected_counts = {
        "case_count": len(process_rows),
        "same_policy_case_count": len(same_policy_cases),
        "compatibility_process_count": len(compatibility_cases),
        "compatibility_pair_count": len(compatibility_cases) // 2,
    }
    for field, expected in expected_counts.items():
        if producer.get(field) != expected:
            failures.append(
                f"producer {field}={producer.get(field)!r}, "
                f"expected {expected!r}"
            )
    if failures:
        final_classification = "INCOMPLETE"
        rounded_classification = "INCOMPLETE"
        legacy_compatibility = "INCOMPLETE"

    ordered_divergences = sorted(
        divergences,
        key=lambda item: (
            item.get("case_id", ""),
            item.get("step_id", -1),
            item.get("row_id", -1),
            item.get("layer_id", -1),
            item.get("slot_id", -1),
        ),
    )
    summary = {
        "schema_version": 1,
        "classification": final_classification,
        "rounded_classification": rounded_classification,
        "legacy_compatibility": legacy_compatibility,
        "policy_integrity": policy_integrity["classification"],
        "policy_failures": policy_integrity["failures"],
        "case_count": len(process_rows),
        "same_policy_case_count": len(same_policy_cases),
        "compatibility_process_count": len(compatibility_cases),
        "compatibility_pair_count": len(compatibility_cases) // 2,
        "failures": list(dict.fromkeys(failures)),
        "corrupt_exact_case_ids": classification.get(
            "corrupt_exact_case_ids",
            [],
        ),
        "corrupt_rounded_case_ids": classification.get(
            "corrupt_rounded_case_ids",
            [],
        ),
        "incompatible_pair_ids": compatibility.get(
            "incompatible_pair_ids",
            [],
        ),
        "first_divergence": (
            ordered_divergences[0] if ordered_divergences else None
        ),
    }
    _write_verification_outputs(run_dir, summary)
    return summary


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Independently verify CUDA Graph diagnostic evidence",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=("diagnostic", "production"),
        default="diagnostic",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.kind != "diagnostic":
        raise SystemExit("production verification is not implemented yet")
    summary = verify_diagnostic(args.run_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if (
        summary["classification"] == "EXACT_REPLAY_CORRECT"
        and summary["legacy_compatibility"] == "LEGACY_COMPATIBLE"
        and summary["policy_integrity"] == "POLICY_EXACT"
        and not summary["failures"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
