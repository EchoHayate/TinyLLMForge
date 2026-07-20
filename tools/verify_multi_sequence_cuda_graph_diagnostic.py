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


def _case_from_row(row: dict):
    return contract.DiagnosticCase(
        batch_size=int(row["batch_size"]),
        trajectory=str(row["trajectory"]),
        mode=str(row["mode"]),
        repetition=int(row["repetition"]),
        graph_size=int(row["graph_size"]),
    )


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
) -> list[str]:
    failures = []
    expected_ids = [
        case.case_id for case in contract.build_diagnostic_matrix()
    ]
    expected_values = {
        "schema_version": 1,
        "kind": "diagnostic",
        "canonical": True,
        "source_tree_sha256": source_evidence.get("tree_sha256"),
        "environment_sha256": contract.canonical_json_sha256(environment),
        "prompt_manifest_sha256": contract.canonical_json_sha256(
            prompt_manifest
        ),
        "case_ids": expected_ids,
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

    expected_cases = {
        case.case_id: case for case in contract.build_diagnostic_matrix()
    }
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
            "case_id": case.case_id,
            "status": "PASS",
            "source_tree_sha256": source_hash,
            "environment_sha256": environment_hash,
        }.items():
            if row.get(field) != expected:
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
) -> tuple[dict[tuple[str, int], dict], list[str]]:
    indexed, failures = _index_step_rows(
        rows,
        evidence_name=evidence_name,
    )
    expected_keys = {
        (case.case_id, step_id)
        for case in contract.build_diagnostic_matrix()
        for step_id in range(contract.MEASURED_STEPS)
    }
    actual_keys = set(indexed)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing:
        failures.append(f"{evidence_name}: missing {missing[:8]}")
    if unexpected:
        failures.append(f"{evidence_name}: unexpected {unexpected[:8]}")
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
) -> None:
    expected_by_identity = {}
    for case in contract.build_diagnostic_matrix():
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
        identity = _case_identity(case)
        previous = expected_by_identity.setdefault(identity, digest)
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
) -> tuple[list[dict], list[dict], list[dict]]:
    eager_shards = {}
    logit_results = []
    layer_results = []
    divergences = []
    for case in contract.build_diagnostic_matrix():
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
        if case.mode == "eager":
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
) -> tuple[list[dict], list[dict]]:
    eager_shards = {}
    results = []
    divergences = []
    for case in contract.build_diagnostic_matrix():
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
        if case.mode == "eager":
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


def _verify_layer_rows(
    layer_rows: dict[tuple[str, int], dict],
    process_rows: dict[str, dict],
) -> list[str]:
    failures = []
    for case in contract.build_diagnostic_matrix():
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
) -> list[str]:
    failures = []
    for case in contract.build_diagnostic_matrix():
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
        f"- Case count: `{summary['case_count']}`",
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
        and summary["rounded_classification"] == "ROUNDED_REPLAY_CORRECT"
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
    failures.extend(
        _validate_manifest(
            manifest,
            source_evidence,
            environment,
            prompt_manifest,
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
    )
    failures.extend(process_failures)
    raw_rows, raw_failures = _validate_step_evidence(
        jsonl_rows["raw_rows.jsonl"],
        evidence_name="raw_rows",
    )
    failures.extend(raw_failures)
    layer_rows, layer_failures = _validate_step_evidence(
        jsonl_rows["layer_observations.jsonl"],
        evidence_name="layer_rows",
    )
    failures.extend(layer_failures)
    _, kv_row_failures = _validate_step_evidence(
        jsonl_rows["kv_observations.jsonl"],
        evidence_name="kv_rows",
    )
    failures.extend(kv_row_failures)
    failures.extend(_verify_layer_rows(layer_rows, process_rows))
    failures.extend(
        _verify_raw_rows_against_logits(raw_rows, process_rows, run_dir)
    )
    _validate_reference_tokens(
        run_dir,
        process_rows,
        sha256sums,
        failures,
    )

    logit_results, layer_results, tensor_divergences = (
        _compare_logits_and_layers(
            run_dir,
            process_rows,
            sha256sums,
            failures,
        )
    )
    divergences.extend(tensor_divergences)
    kv_results, kv_divergences = _compare_kv(
        run_dir,
        process_rows,
        sha256sums,
        failures,
    )
    divergences.extend(kv_divergences)
    matrix_rows = list(process_rows.values())
    classification = contract.classify_diagnostic(
        matrix_rows=matrix_rows,
        logit_results=logit_results,
        layer_results=layer_results,
        kv_results=kv_results,
    )
    failures.extend(classification.get("failures", []))

    if failures:
        final_classification = "INCOMPLETE"
        rounded_classification = "INCOMPLETE"
    else:
        final_classification = classification["classification"]
        rounded_classification = classification["rounded_classification"]
    producer = loaded["summary.json"]
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
    if failures:
        final_classification = "INCOMPLETE"
        rounded_classification = "INCOMPLETE"

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
        "case_count": len(process_rows),
        "failures": list(dict.fromkeys(failures)),
        "corrupt_exact_case_ids": classification.get(
            "corrupt_exact_case_ids",
            [],
        ),
        "corrupt_rounded_case_ids": classification.get(
            "corrupt_rounded_case_ids",
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
        and summary["rounded_classification"] == "ROUNDED_REPLAY_CORRECT"
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
