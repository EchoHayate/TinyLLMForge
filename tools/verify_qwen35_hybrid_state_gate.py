"""Independent verifier for Qwen3.5 hybrid-state gate artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path, PurePosixPath


THIS_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = THIS_DIR / "qwen35_hybrid_state_contract.py"
REQUIRED_INPUT_FILES = {
    "manifest.json",
    "source_manifest.json",
    "model_manifest.json",
    "environment.json",
    "case_rows.jsonl",
    "state_snapshots.jsonl",
    "state_components.jsonl",
    "memory_snapshots.jsonl",
    "processes.json",
    "ports.json",
    "summary.json",
}
REQUIRED_INPUT_DIRECTORIES = {"stdout", "stderr"}
OUTPUT_FILES = {
    "independent_verification.json",
    "report.md",
    "local_verifier_process.json",
    "smoke_evidence.json",
    "stdout/local_verifier.log",
    "stderr/local_verifier.log",
}
SMOKE_CASE_IDS = {
    "environment_preflight",
    "architecture_verification",
    "same_path_repeatability__cached_repeatability__p17__r0__c17",
    "same_path_repeatability__cached_repeatability__p17__r1__c17",
    "one_shot_vs_cached__one_shot_vs_cached__p17__r0__c17",
    "fp32_path_control__cached_vs_one_shot__p17__r0__c17",
    "state_export_import__state_export_import__p17__r0__c17",
    "post_run_audit",
}


def _load_contract():
    spec = importlib.util.spec_from_file_location(
        "qwen35_hybrid_state_contract_for_verifier",
        os.fspath(CONTRACT_PATH),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_contract()


class VerificationError(ValueError):
    pass


class SemanticFailure(ValueError):
    pass


def _fail(detail):
    raise VerificationError(detail)


def _semantic_fail(detail):
    raise SemanticFailure(detail)


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"invalid JSON artifact {path.name}: {exc}")


def _read_jsonl(path):
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        _fail(f"invalid JSONL artifact {path.name}: {exc}")
    for line_number, line in enumerate(lines, start=1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            _fail(
                f"invalid JSONL artifact {path.name}:{line_number}: {exc}"
            )
    return rows


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(value):
    path = PurePosixPath(str(value))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        _fail(f"unsafe artifact path: {value}")
    return path


def _verify_inventory(run_dir, manifest):
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        _fail("manifest artifacts must be a list")
    listed = {}
    for entry in artifacts:
        if not isinstance(entry, dict):
            _fail("manifest artifact entry must be an object")
        relative = _safe_relative_path(entry.get("path"))
        relative_text = relative.as_posix()
        if relative_text in listed:
            _fail(f"duplicate manifest artifact: {relative_text}")
        path = run_dir.joinpath(*relative.parts)
        if not path.is_file():
            _fail(f"missing listed artifact: {relative_text}")
        actual_size = path.stat().st_size
        actual_hash = _sha256(path)
        if entry.get("size") != actual_size:
            _fail(f"artifact size mismatch: {relative_text}")
        if entry.get("sha256") != actual_hash:
            _fail(f"artifact sha256 mismatch: {relative_text}")
        listed[relative_text] = entry
    actual_files = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.relative_to(run_dir).as_posix() not in OUTPUT_FILES
        and path.name != "manifest.json"
    }
    if set(listed) != actual_files:
        missing = sorted(actual_files - set(listed))
        extra = sorted(set(listed) - actual_files)
        if missing:
            _fail(f"unlisted artifact: {missing[0]}")
        _fail(f"listed artifact is absent: {extra[0]}")
    listed_root = {
        PurePosixPath(relative).parts[0]
        for relative in listed
    }
    if not REQUIRED_INPUT_FILES - {"manifest.json"} <= listed_root:
        _fail("canonical artifact inventory is incomplete")
    for directory in REQUIRED_INPUT_DIRECTORIES:
        if not (run_dir / directory).is_dir():
            _fail(f"missing canonical directory: {directory}")


def _verify_source(source_manifest, manifest):
    if source_manifest.get("clean") is not True:
        _fail("source snapshot is not clean")
    commit = source_manifest.get("commit")
    if not isinstance(commit, str) or len(commit) != 40:
        _fail("source commit is not immutable")
    if manifest.get("source_commit") != commit:
        _fail("source commit mismatch")
    local_hashes = source_manifest.get("local_file_sha256")
    remote_hashes = source_manifest.get("remote_file_sha256")
    if not isinstance(local_hashes, dict) or not local_hashes:
        _fail("local source hashes are missing")
    if local_hashes != remote_hashes:
        _fail("source hash mismatch")


def _verify_model(model_manifest, manifest):
    revision = model_manifest.get("resolved_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        _fail("model revision is not immutable")
    if model_manifest.get("repository") != contract.MODEL_REPOSITORY:
        _fail("model repository mismatch")
    if manifest.get("model_repository") != contract.MODEL_REPOSITORY:
        _fail("manifest model repository mismatch")
    if manifest.get("model_resolved_revision") != revision:
        _fail("model revision mismatch")
    files = model_manifest.get("files")
    if not isinstance(files, dict) or not files:
        _fail("model file inventory is missing")
    for name, entry in files.items():
        if not isinstance(name, str) or not name:
            _fail("model file name is invalid")
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("size"), int)
            or entry["size"] < 0
            or not isinstance(entry.get("sha256"), str)
            or len(entry["sha256"]) != 64
        ):
            _fail(f"model file identity is invalid: {name}")
    if model_manifest.get("trust_remote_code") is not False:
        _fail("trust_remote_code must be false")


def _verify_environment(environment):
    if environment.get("host") != "10.232.195.203":
        _fail("remote host mismatch")
    if environment.get("user") != "sitian":
        _fail("remote user mismatch")
    variables = environment.get("environment")
    if not isinstance(variables, dict):
        _fail("execution environment is missing")
    if variables.get("CUDA_VISIBLE_DEVICES") != "0":
        _fail("CUDA_VISIBLE_DEVICES must be 0")
    tiny_port = variables.get("TINYVLLM_DIST_PORT")
    master_port = variables.get("MASTER_PORT")
    if tiny_port == master_port:
        _fail("process ports must be distinct")


def _verify_processes(processes, ports):
    process_rows = processes.get("processes")
    port_rows = ports.get("pairs")
    if not isinstance(process_rows, list) or not process_rows:
        _fail("process inventory is missing")
    if not isinstance(port_rows, list) or not port_rows:
        _fail("port inventory is missing")
    used_ports = set()
    expected_pairs = set()
    for row in process_rows:
        if row.get("exit_code") != 0:
            _fail("model process did not exit cleanly")
        pair = (
            row.get("name"),
            row.get("attempt"),
            row.get("tinyvllm_dist_port"),
            row.get("master_port"),
        )
        expected_pairs.add(pair)
        for value in pair[-2:]:
            if not isinstance(value, int) or not 1 <= value <= 65535:
                _fail("invalid process port")
            if value in used_ports:
                _fail("port reuse")
            used_ports.add(value)
    actual_pairs = {
        (
            row.get("process"),
            row.get("attempt"),
            row.get("tinyvllm_dist_port"),
            row.get("master_port"),
        )
        for row in port_rows
    }
    if actual_pairs != expected_pairs:
        _fail("process and port inventories disagree")


def _verify_case_domain(case_rows, domain):
    expected = {
        case.case_id: case
        for case in contract.build_case_matrix()
        if domain == "canonical" or case.case_id in SMOKE_CASE_IDS
    }
    observed = {}
    for row in case_rows:
        case_id = row.get("case_id")
        if case_id in observed:
            _fail(f"duplicate {domain} case: {case_id}")
        if case_id not in expected:
            _fail(f"unknown {domain} case: {case_id}")
        observed[case_id] = row
        case = expected[case_id]
        expected_fields = {
            "phase": case.phase,
            "execution_mode": case.execution_mode,
            "prompt_length": case.prompt_length,
            "chunk_schedule": list(case.chunk_schedule),
            "request_count": case.request_count,
            "decode_steps": case.decode_steps,
            "repeat_index": case.repeat_index,
            "execution_dtype": case.execution_dtype,
            "comparison_policy": case.comparison_policy,
        }
        for field, value in expected_fields.items():
            if row.get(field) != value:
                _fail(f"{domain} case mismatch: {case_id}.{field}")
        if row.get("complete") is not True:
            _fail(f"incomplete {domain} case: {case_id}")
    missing = sorted(set(expected) - set(observed))
    if missing:
        _fail(f"missing {domain} case: {missing[0]}")
    return len(expected)


def _expected_logit_domain(row):
    if row["phase"] == "interleaved_multi_request":
        return [
            (f"slot-{request_index}", 0, step_index, prompt_length)
            for step_index in range(row["decode_steps"])
            for request_index, prompt_length in enumerate(
                contract.MULTI_REQUEST_LENGTHS
            )
        ]
    if row["phase"] == "completion_release_slot_reuse":
        request_domains = (
            ("slot-0", 0, contract.MULTI_REQUEST_LENGTHS[0], 2),
            (
                "slot-1",
                0,
                contract.MULTI_REQUEST_LENGTHS[1],
                row["decode_steps"],
            ),
            (
                "slot-2",
                0,
                contract.MULTI_REQUEST_LENGTHS[2],
                row["decode_steps"],
            ),
            (
                "slot-0",
                1,
                contract.SLOT_REUSE_PROMPT_LENGTH,
                row["decode_steps"],
            ),
        )
        return [
            (
                request_id,
                request_generation,
                step_index,
                prompt_length,
            )
            for (
                request_id,
                request_generation,
                prompt_length,
                step_count,
            ) in request_domains
            for step_index in range(step_count)
        ]
    return [
        (
            row["request_ids"][0],
            row["request_generations"][0],
            step_index,
            row["prompt_length"],
        )
        for step_index in range(row["decode_steps"])
    ]


def _verify_logit_record(row, record, expected):
    (
        expected_request_id,
        expected_generation,
        expected_step,
        expected_prompt_length,
    ) = expected
    if set(record) != set(contract.LOGIT_RECORD_FIELDS):
        _fail(f"logit record schema mismatch: {row['case_id']}")
    if record.get("step_index") != expected_step:
        _fail(f"missing logit step: {row['case_id']}:{expected_step}")
    if record.get("request_id") != expected_request_id:
        _fail(f"logit request identity mismatch: {row['case_id']}")
    if record.get("request_generation") != expected_generation:
        _fail(f"logit request generation mismatch: {row['case_id']}")
    expected_sequence_length = expected_prompt_length + expected_step
    if record.get("sequence_length") != expected_sequence_length:
        _fail(f"logit sequence length mismatch: {row['case_id']}")
    digest = record.get("full_logit_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        _fail(f"invalid full-logit hash: {row['case_id']}")
    position_metadata = record.get("position_metadata")
    if not isinstance(position_metadata, dict):
        _fail(f"invalid position metadata: {row['case_id']}")
    actual_token = position_metadata.get("actual_greedy_token_id")
    oracle_token = position_metadata.get("oracle_greedy_token_id")
    if (
        not isinstance(actual_token, int)
        or isinstance(actual_token, bool)
        or not isinstance(oracle_token, int)
        or isinstance(oracle_token, bool)
    ):
        _fail(f"invalid greedy token evidence: {row['case_id']}")
    actual_digest = position_metadata.get("actual_full_logit_sha256")
    oracle_digest = position_metadata.get("oracle_full_logit_sha256")
    for name, value in (
        ("actual", actual_digest),
        ("oracle", oracle_digest),
    ):
        if not isinstance(value, str) or len(value) != 64:
            _fail(
                f"invalid {name} full-logit hash: {row['case_id']}"
            )
    if digest != actual_digest:
        _fail(f"actual full-logit hash mismatch: {row['case_id']}")
    actual_ids = record.get("actual_topk_token_ids")
    actual_logits = record.get("actual_topk_logits")
    oracle_ids = record.get("oracle_topk_token_ids")
    oracle_logits = record.get("oracle_topk_logits")
    try:
        contract.validate_ranked_topk(actual_ids, actual_logits)
        contract.validate_ranked_topk(oracle_ids, oracle_logits)
    except (TypeError, ValueError) as exc:
        if "strict positive margin" in str(exc):
            _semantic_fail(
                f"winner margin boundary crossed: {row['case_id']}"
            )
        _fail(f"invalid top-k evidence: {row['case_id']}: {exc}")
    if record.get("topk_token_ids") != actual_ids:
        _fail(f"actual top-k token alias mismatch: {row['case_id']}")
    if record.get("topk_logits") != actual_logits:
        _fail(f"actual top-k logit alias mismatch: {row['case_id']}")
    actual_winner = contract.winner_margin(actual_ids, actual_logits)
    oracle_winner = contract.winner_margin(oracle_ids, oracle_logits)
    expected_decision = {
        "actual_winner_token_id": actual_winner["winner_token_id"],
        "oracle_winner_token_id": oracle_winner["winner_token_id"],
        "actual_runner_up_token_id": actual_winner["runner_up_token_id"],
        "oracle_runner_up_token_id": oracle_winner["runner_up_token_id"],
        "actual_winner_logit": actual_winner["winner_logit"],
        "oracle_winner_logit": oracle_winner["winner_logit"],
        "actual_runner_up_logit": actual_winner["runner_up_logit"],
        "oracle_runner_up_logit": oracle_winner["runner_up_logit"],
        "actual_winner_margin": actual_winner["winner_margin"],
        "oracle_winner_margin": oracle_winner["winner_margin"],
    }
    for field, value in expected_decision.items():
        if record.get(field) != value:
            _fail(f"decision evidence mismatch: {row['case_id']}.{field}")
    if (
        actual_winner["winner_token_id"] not in oracle_ids
        or oracle_winner["winner_token_id"] not in actual_ids
    ):
        _fail(f"cross-path winner missing from top-k: {row['case_id']}")
    intersection = len(set(actual_ids).intersection(oracle_ids))
    if record.get("topk_intersection_size") != intersection:
        _fail(f"top-k intersection mismatch: {row['case_id']}")
    if record.get("oracle_topk_recall") != (
        intersection / contract.DECISION_TOPK
    ):
        _fail(f"oracle top-k recall mismatch: {row['case_id']}")
    if actual_winner["winner_margin"] <= 0 or oracle_winner[
        "winner_margin"
    ] <= 0:
        _semantic_fail(f"winner margin boundary crossed: {row['case_id']}")
    if actual_winner["winner_token_id"] != oracle_winner[
        "winner_token_id"
    ]:
        _semantic_fail(f"winner token mismatch: {row['case_id']}")
    if actual_token != actual_winner["winner_token_id"]:
        _fail(f"actual greedy winner mismatch: {row['case_id']}")
    if oracle_token != oracle_winner["winner_token_id"]:
        _fail(f"oracle greedy winner mismatch: {row['case_id']}")
    percentiles = record.get("abs_diff_percentiles")
    if (
        not isinstance(percentiles, dict)
        or tuple(percentiles) != contract.ABS_DIFF_PERCENTILE_FIELDS
    ):
        _fail(f"invalid difference percentiles: {row['case_id']}")
    percentile_values = [
        percentiles[name]
        for name in contract.ABS_DIFF_PERCENTILE_FIELDS
    ]
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value < 0
        for value in percentile_values
    ) or any(
        left > right
        for left, right in zip(
            percentile_values,
            percentile_values[1:],
        )
    ):
        _fail(f"invalid difference percentiles: {row['case_id']}")
    for field in (
        "max_abs_diff",
        "mean_abs_diff",
        "max_rel_diff",
        "mean_rel_diff",
        "winner_logit_abs_diff",
        "runner_up_logit_abs_diff",
        "winner_margin_abs_diff",
        "max_allclose_scaled_error",
    ):
        value = record.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value < 0
        ):
            _fail(f"invalid logit difference: {row['case_id']}.{field}")
    cosine = record.get("cosine_similarity")
    if (
        not isinstance(cosine, (int, float))
        or isinstance(cosine, bool)
        or not math.isfinite(float(cosine))
        or not -1.0 <= float(cosine) <= 1.0
    ):
        _fail(f"invalid cosine similarity: {row['case_id']}")
    violations = record.get("allclose_violation_count")
    if (
        not isinstance(violations, int)
        or isinstance(violations, bool)
        or violations < 0
    ):
        _fail(f"invalid allclose violation count: {row['case_id']}")
    if position_metadata.get("comparison_policy") != row.get(
        "comparison_policy"
    ):
        _fail(f"comparison policy mismatch: {row['case_id']}")


def _verify_correctness(case_rows):
    rows_by_phase = {}
    for row in case_rows:
        rows_by_phase.setdefault(row["phase"], []).append(row)
        records = row.get("logit_records")
        decoded = row.get("decoded_token_ids")
        if not isinstance(records, list) or not isinstance(decoded, list):
            _fail(f"correctness rows are malformed: {row['case_id']}")
        expected_domain = _expected_logit_domain(row)
        if len(records) != len(expected_domain):
            _fail(f"missing logit step: {row['case_id']}")
        if len(decoded) != len(expected_domain):
            _fail(f"missing decoded token: {row['case_id']}")
        for decoded_token, record, expected in zip(
            decoded,
            records,
            expected_domain,
        ):
            _verify_logit_record(row, record, expected)
            metadata = record["position_metadata"]
            if decoded_token != metadata["actual_greedy_token_id"]:
                _fail(
                    f"decoded token evidence mismatch: {row['case_id']}"
                )
            if (
                metadata["actual_greedy_token_id"]
                != metadata["oracle_greedy_token_id"]
            ):
                _semantic_fail(
                    f"oracle greedy token mismatch: {row['case_id']}"
                )
    repeatability = rows_by_phase.get("same_path_repeatability", [])
    repeatability_by_prompt = {}
    for row in repeatability:
        repeatability_by_prompt.setdefault(
            row["prompt_length"],
            [],
        ).append(row)
    for prompt_length, rows in repeatability_by_prompt.items():
        baseline = min(rows, key=lambda item: item["repeat_index"])
        for row in rows:
            if row["decoded_token_ids"] != baseline["decoded_token_ids"]:
                _semantic_fail(
                    f"decoded token mismatch: {row['case_id']}"
                )
            for actual, expected in zip(
                row["logit_records"],
                baseline["logit_records"],
            ):
                if (
                    actual["full_logit_sha256"]
                    != expected["full_logit_sha256"]
                ):
                    _fail(
                        "same-path full-logit hash mismatch: "
                        f"{row['case_id']}"
                    )
    baselines = {}
    for row in sorted(
        repeatability,
        key=lambda item: (
            item["prompt_length"],
            item["repeat_index"],
        ),
    ):
        prompt_length = row["prompt_length"]
        baseline = baselines.setdefault(prompt_length, row)
    for phase in (
        "one_shot_vs_cached",
        "one_shot_vs_chunked",
        "state_export_import",
    ):
        for row in rows_by_phase.get(phase, []):
            baseline = baselines.get(row["prompt_length"])
            if baseline is None:
                _fail(
                    f"missing repeatability baseline: {row['case_id']}"
                )
            if row["decoded_token_ids"] != baseline["decoded_token_ids"]:
                if phase == "state_export_import":
                    _semantic_fail(
                        f"export/import token mismatch: {row['case_id']}"
                    )
                _semantic_fail(
                    f"decoded token mismatch: {row['case_id']}"
                )
    fp32_rows = rows_by_phase.get("fp32_path_control", [])
    if len(fp32_rows) != 1:
        _fail("FP32 control case is missing")
    for record in fp32_rows[0]["logit_records"]:
        if (
            record["allclose_violation_count"] != 0
            or record["max_allclose_scaled_error"] > 1.0
            or record["mean_abs_diff"] > contract.FP32_MEAN_ABS_CAP
        ):
            _fail("FP32 control failed strict elementwise limits")
    return {
        "fp32_atol": contract.FP32_ATOL,
        "fp32_rtol": contract.FP32_RTOL,
        "fp32_mean_abs_cap": contract.FP32_MEAN_ABS_CAP,
    }


def _verify_dtype_profiles(
    case_rows,
    state_snapshots,
    state_components,
    model_manifest,
    environment,
    summary,
):
    profiles = model_manifest.get("dtype_profiles")
    if (
        not isinstance(profiles, dict)
        or profiles != environment.get("dtype_profiles")
        or profiles != summary.get("dtype_profiles")
    ):
        _fail("dtype profile manifests disagree")
    snapshot_epochs = {
        snapshot["snapshot_id"]: snapshot["lifetime_epoch"]
        for snapshot in state_snapshots
    }
    components_by_epoch = {}
    for component in state_components:
        components_by_epoch.setdefault(
            component["lifetime_epoch"],
            [],
        ).append(component)
    for execution_dtype in ("bfloat16", "float32"):
        profile = profiles.get(execution_dtype)
        if not isinstance(profile, dict):
            _fail(f"dtype profile is missing: {execution_dtype}")
        expected_fields = {
            "requested_model_dtype",
            "dominant_parameter_dtype",
            "logit_dtype_before_comparison",
            "comparison_accumulator_dtype",
            "recurrent_state_dtypes",
            "kv_state_dtypes",
        }
        if set(profile) != expected_fields:
            _fail(f"dtype profile schema mismatch: {execution_dtype}")
        if profile["requested_model_dtype"] != execution_dtype:
            _fail(f"dtype profile request mismatch: {execution_dtype}")
        if profile["dominant_parameter_dtype"] != execution_dtype:
            _fail(f"dtype profile parameter mismatch: {execution_dtype}")
        if profile["logit_dtype_before_comparison"] != execution_dtype:
            _fail(f"dtype profile logit mismatch: {execution_dtype}")
        if profile["comparison_accumulator_dtype"] != "float32":
            _fail(f"dtype profile accumulator mismatch: {execution_dtype}")
        epochs = {
            snapshot_epochs[snapshot_id]
            for row in case_rows
            if row["execution_dtype"] == execution_dtype
            for snapshot_id in row["state_snapshot_ids"]
            if snapshot_id in snapshot_epochs
        }
        components = [
            component
            for epoch in epochs
            for component in components_by_epoch.get(epoch, [])
        ]
        recurrent = sorted({
            component["dtype"]
            for component in components
            if component["state_role"] in {
                "linear_recurrent_state",
                "linear_convolution_state",
            }
        })
        kv = sorted({
            component["dtype"]
            for component in components
            if component["state_role"] in {
                "full_attention_key",
                "full_attention_value",
            }
        })
        if profile["recurrent_state_dtypes"] != recurrent:
            _fail(f"dtype profile recurrent state mismatch: {execution_dtype}")
        if profile["kv_state_dtypes"] != kv:
            _fail(f"dtype profile KV state mismatch: {execution_dtype}")
    return profiles


def _component_key(component):
    return (
        component["request_id"],
        component["request_generation"],
        component["layer_index"],
        component["state_role"],
        component["tensor_path"],
    )


def _verify_state_lifecycle(
    case_rows,
    state_snapshots,
    state_components,
    summary,
    domain,
):
    expected_component_fields = set(
        contract.StateComponent.__dataclass_fields__
    )
    snapshots_by_id = {}
    snapshots_by_epoch = {}
    case_by_snapshot_id = {}
    for row in case_rows:
        for snapshot_id in row.get("state_snapshot_ids", []):
            if snapshot_id in case_by_snapshot_id:
                _fail(f"state snapshot reused by cases: {snapshot_id}")
            case_by_snapshot_id[snapshot_id] = row["case_id"]
    for snapshot in state_snapshots:
        snapshot_id = snapshot.get("snapshot_id")
        epoch = snapshot.get("lifetime_epoch")
        if snapshot_id in snapshots_by_id:
            _fail(f"duplicate state snapshot: {snapshot_id}")
        if epoch in snapshots_by_epoch:
            _fail(f"duplicate lifetime epoch: {epoch}")
        if snapshot_id not in case_by_snapshot_id:
            _fail(f"unreferenced state snapshot: {snapshot_id}")
        snapshots_by_id[snapshot_id] = snapshot
        snapshots_by_epoch[epoch] = snapshot
    if set(snapshots_by_id) != set(case_by_snapshot_id):
        _fail("state snapshot inventory does not match case rows")
    components_by_epoch = {}
    for component in state_components:
        if set(component) != expected_component_fields:
            _fail("state component schema mismatch")
        role = component.get("state_role")
        if role == "other_persistent_state":
            _fail("unexplained state role")
        if role not in contract.STATE_ROLES:
            _fail(f"unsupported state role: {role}")
        update_kind = component.get("update_kind")
        if update_kind not in contract.UPDATE_KINDS:
            _fail(f"unsupported update kind: {update_kind}")
        layer_index = component.get("layer_index")
        if (
            not isinstance(layer_index, int)
            or isinstance(layer_index, bool)
            or layer_index < 0
            or layer_index >= contract.EXPECTED_NUM_HIDDEN_LAYERS
        ):
            _fail("ambiguous layer identity")
        snapshot = snapshots_by_epoch.get(component.get("lifetime_epoch"))
        if snapshot is None:
            _fail("component references unknown lifetime epoch")
        if (
            component["request_id"] != snapshot["request_id"]
            or component["request_generation"]
            != snapshot["request_generation"]
        ):
            _fail("cross-request state mutation")
        if component["sequence_length"] != snapshot["sequence_length"]:
            _fail("component sequence length mismatch")
        components_by_epoch.setdefault(
            component["lifetime_epoch"],
            [],
        ).append(component)
    schedule = summary.get("architecture", {}).get("layer_schedule")
    if not isinstance(schedule, dict):
        _fail("architecture layer schedule is missing")
    evolution = {}
    for epoch in sorted(components_by_epoch):
        snapshot = snapshots_by_epoch[epoch]
        case_id = case_by_snapshot_id[snapshot["snapshot_id"]]
        for component in components_by_epoch[epoch]:
            declared = component["declared_layer_type"]
            observed = schedule.get(str(component["layer_index"]))
            if observed != declared:
                _fail("ambiguous layer identity")
            role = component["state_role"]
            if role.startswith("full_attention_"):
                if declared != "full_attention":
                    _fail("full-attention role is bound to wrong layer")
                if (
                    len(component["shape"]) < 3
                    or component["shape"][2]
                    != component["sequence_length"]
                ):
                    _fail("full-attention state did not grow")
            if role in {
                "linear_recurrent_state",
                "linear_convolution_state",
            } and declared != "linear_attention":
                _fail("linear state is bound to wrong layer")
            key = (case_id, *_component_key(component))
            previous = evolution.get(key)
            if previous is not None:
                if role in {
                    "linear_recurrent_state",
                    "linear_convolution_state",
                } and (
                    component["update_kind"] != "released"
                    and component["shape"] != previous["shape"]
                ):
                    _fail("recurrent state shape grew")
                if (
                    role.startswith("full_attention_")
                    and component["sequence_length"]
                    > previous["sequence_length"]
                    and component["shape"][2] <= previous["shape"][2]
                ):
                    _fail("full-attention state did not grow")
            evolution[key] = component
    for snapshot in state_snapshots:
        components = components_by_epoch.get(
            snapshot["lifetime_epoch"],
            [],
        )
        if snapshot.get("component_count") != len(components):
            _fail("state snapshot component count mismatch")
        if snapshot.get("component_sha256") != (
            contract.canonical_json_sha256(components)
        ):
            _fail("state snapshot component hash mismatch")
    if domain == "smoke":
        return {
            "state_snapshot_count": len(state_snapshots),
            "state_component_count": len(state_components),
            "released_component_count": 0,
        }
    lifecycle_row = next(
        row
        for row in case_rows
        if row["phase"] == "completion_release_slot_reuse"
    )
    if lifecycle_row.get("request_ids") != [
        "slot-0",
        "slot-1",
        "slot-2",
        "slot-0",
    ]:
        _fail("slot lifecycle request identity mismatch")
    if lifecycle_row.get("request_generations") != [0, 0, 0, 1]:
        _fail("slot generation did not increment")
    lifecycle_epochs = {
        snapshots_by_id[snapshot_id]["lifetime_epoch"]
        for snapshot_id in lifecycle_row["state_snapshot_ids"]
    }
    lifecycle_components = [
        component
        for component in state_components
        if component["lifetime_epoch"] in lifecycle_epochs
    ]
    released = [
        component
        for component in lifecycle_components
        if (
            component["request_id"] == "slot-0"
            and component["request_generation"] == 0
            and component["update_kind"] == "released"
        )
    ]
    if not released:
        _fail("missing release")
    replacement = [
        component
        for component in lifecycle_components
        if (
            component["request_id"] == "slot-0"
            and component["request_generation"] == 1
        )
    ]
    if not replacement:
        _fail("slot generation did not increment")
    release_epoch = min(
        component["lifetime_epoch"] for component in released
    )
    replacement_epoch = min(
        component["lifetime_epoch"] for component in replacement
    )
    if replacement_epoch <= release_epoch:
        _fail("slot reuse occurred before release")
    released_storage = {
        (component["device"], component["storage_identity"])
        for component in released
    }
    replacement_storage = {
        (component["device"], component["storage_identity"])
        for component in replacement
    }
    if released_storage & replacement_storage:
        _fail("stale storage identity")
    released_content = {
        component["content_sha256"] for component in released
    }
    replacement_content = {
        component["content_sha256"] for component in replacement
    }
    if released_content & replacement_content:
        _fail("stale content")
    return {
        "state_snapshot_count": len(state_snapshots),
        "state_component_count": len(state_components),
        "released_component_count": len(released),
    }


def _verify_storage_ledger(
    case_rows,
    state_snapshots,
    state_components,
    memory_snapshots,
    model_manifest,
    summary,
    domain,
):
    components_by_epoch = {}
    for component in state_components:
        expected_logical = contract.logical_bytes(
            tuple(component["shape"]),
            component["dtype"],
        )
        if component["logical_numel"] * contract.DTYPE_SIZES[
            component["dtype"]
        ] != expected_logical:
            _fail("logical numel and dtype disagree")
        if component["logical_bytes"] != expected_logical:
            _fail("incorrect logical bytes")
        components_by_epoch.setdefault(
            component["lifetime_epoch"],
            [],
        ).append(component)
    snapshot_by_id = {
        snapshot["snapshot_id"]: snapshot
        for snapshot in state_snapshots
    }
    expected_memory_ids = {
        memory_id
        for row in case_rows
        for memory_id in row.get("memory_snapshot_ids", [])
    }
    memory_by_id = {}
    observed_phases = set()
    for memory in memory_snapshots:
        memory_id = memory.get("snapshot_id")
        if memory_id in memory_by_id:
            _fail(f"duplicate memory epoch: {memory_id}")
        memory_by_id[memory_id] = memory
        observed_phases.add(memory.get("phase"))
        for field in ("cuda_allocated_bytes", "cuda_reserved_bytes"):
            value = memory.get(field)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                _fail("negative allocator bytes")
        if memory["cuda_reserved_bytes"] < memory["cuda_allocated_bytes"]:
            _fail("reserved allocator bytes are below allocated bytes")
        if memory_id.startswith("memory:"):
            state_id = memory_id.removeprefix("memory:")
            snapshot = snapshot_by_id.get(state_id)
            if snapshot is None:
                _fail("memory snapshot references unknown state epoch")
            components = components_by_epoch.get(
                snapshot["lifetime_epoch"],
                [],
            )
            if memory.get("phase") == "after_request_release":
                components = []
            logical = sum(
                component["logical_bytes"] for component in components
            )
            try:
                unique = contract.unique_storage_bytes(components)
            except ValueError as exc:
                _fail(str(exc))
            if memory.get("logical_state_bytes") != logical:
                _fail("logical state bytes mismatch")
            if memory.get("unique_storage_bytes") != unique:
                _fail("unique storage bytes mismatch")
        else:
            if (
                memory.get("logical_state_bytes") != 0
                or memory.get("unique_storage_bytes") != 0
            ):
                _fail("lifecycle allocator row contains state bytes")
    missing_memory = expected_memory_ids - set(memory_by_id)
    if missing_memory:
        _fail(f"missing memory epoch: {sorted(missing_memory)[0]}")
    unreferenced_case_memory = {
        memory_id
        for memory_id in memory_by_id
        if memory_id.startswith("memory:")
    } - expected_memory_ids
    if unreferenced_case_memory:
        _fail(
            "unreferenced memory epoch: "
            f"{sorted(unreferenced_case_memory)[0]}"
        )
    required_phases = {
        "before_model_load",
        "after_model_load",
        "after_model_release",
    }
    if domain == "canonical":
        required_phases.update({
            "before_prefill",
            "after_prefill",
            "after_request_release",
            "after_slot_reuse",
        })
    missing_phases = required_phases - observed_phases
    if missing_phases:
        _fail(f"missing memory epoch: {sorted(missing_phases)[0]}")
    if not any(
        phase.startswith("after_decode_step_")
        for phase in observed_phases
    ):
        _fail("missing memory epoch: after_decode_step_N")
    parameter_dtypes = model_manifest.get("parameter_dtypes")
    if not isinstance(parameter_dtypes, dict) or not parameter_dtypes:
        _fail("model parameter dtype inventory is missing")
    parameter_bytes = 0
    for dtype, count in parameter_dtypes.items():
        if dtype not in contract.DTYPE_SIZES:
            _fail(f"unknown parameter dtype: {dtype}")
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
        ):
            _fail("invalid parameter count")
        parameter_bytes += count * contract.DTYPE_SIZES[dtype]
    if summary.get("parameter_bytes") != parameter_bytes:
        _fail("parameter-byte mismatch")
    logical_total = sum(
        component["logical_bytes"] for component in state_components
    )
    try:
        unique_total = contract.unique_storage_bytes(state_components)
    except ValueError as exc:
        _fail(str(exc))
    if summary.get("state_logical_bytes") != logical_total:
        _fail("worker aggregate disagreement: state logical bytes")
    if summary.get("state_unique_storage_bytes") != unique_total:
        _fail("worker aggregate disagreement: state unique storage bytes")
    if summary.get("state_snapshot_count") != len(state_snapshots):
        _fail("worker aggregate disagreement: state snapshot count")
    if summary.get("state_component_count") != len(state_components):
        _fail("worker aggregate disagreement: state component count")
    if summary.get("memory_snapshot_count") != len(memory_snapshots):
        _fail("worker aggregate disagreement: memory snapshot count")
    return {
        "parameter_bytes": parameter_bytes,
        "state_logical_bytes": logical_total,
        "state_unique_storage_bytes": unique_total,
        "memory_snapshot_count": len(memory_snapshots),
    }


def _write_atomic(path, content):
    partial = path.with_name(path.name + ".partial")
    partial.write_text(content, encoding="utf-8")
    os.replace(partial, path)


def _write_outputs(run_dir, result):
    _write_atomic(
        run_dir / "independent_verification.json",
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )
    report = "\n".join([
        "# Qwen3.5 Hybrid-State Independent Verification",
        "",
        f"Classification: `{result['classification']}`",
        "",
        "## Claim Boundary",
        "",
        (
            "Compatibility evidence only. This result does not claim native "
            "TinyLLMForge support, compression safety, quality retention, "
            "latency, throughput, speedup, or physical memory reduction."
        ),
        "",
    ])
    _write_atomic(run_dir / "report.md", report)


def _observed_case_count(destination):
    try:
        return len(_read_jsonl(destination / "case_rows.jsonl"))
    except VerificationError:
        return 0


def _verify_complete_run(destination, domain):
    manifest = _read_json(destination / "manifest.json")
    if manifest.get("schema_version") != contract.SCHEMA_VERSION:
        _fail("schema-v2 evidence is required")
    _verify_inventory(destination, manifest)
    source_manifest = _read_json(destination / "source_manifest.json")
    model_manifest = _read_json(destination / "model_manifest.json")
    environment = _read_json(destination / "environment.json")
    processes = _read_json(destination / "processes.json")
    ports = _read_json(destination / "ports.json")
    summary = _read_json(destination / "summary.json")
    if summary.get("schema_version") != contract.SCHEMA_VERSION:
        _fail("summary schema version mismatch")
    case_rows = _read_jsonl(destination / "case_rows.jsonl")
    state_snapshots = _read_jsonl(
        destination / "state_snapshots.jsonl"
    )
    state_components = _read_jsonl(
        destination / "state_components.jsonl"
    )
    memory_snapshots = _read_jsonl(
        destination / "memory_snapshots.jsonl"
    )
    _verify_source(source_manifest, manifest)
    _verify_model(model_manifest, manifest)
    _verify_environment(environment)
    _verify_processes(processes, ports)
    expected_case_count = _verify_case_domain(case_rows, domain)
    tolerance = _verify_correctness(case_rows)
    dtype_profiles = _verify_dtype_profiles(
        case_rows,
        state_snapshots,
        state_components,
        model_manifest,
        environment,
        summary,
    )
    lifecycle = _verify_state_lifecycle(
        case_rows,
        state_snapshots,
        state_components,
        summary,
        domain,
    )
    storage_ledger = _verify_storage_ledger(
        case_rows,
        state_snapshots,
        state_components,
        memory_snapshots,
        model_manifest,
        summary,
        domain,
    )
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "classification": (
            "GO" if domain == "canonical" else "SMOKE_PASS"
        ),
        "expected_case_count": expected_case_count,
        "observed_case_count": len(case_rows),
        "guards": {
            "artifact_inventory_pass": True,
            "source_provenance_pass": True,
            "model_provenance_pass": True,
            "environment_pass": True,
            "process_and_port_pass": True,
            "case_domain_pass": True,
            "correctness_pass": True,
            "state_lifecycle_pass": True,
            "storage_ledger_pass": True,
        },
        "reasons": [],
        "logit_tolerance": tolerance,
        "dtype_profiles": dtype_profiles,
        "state_lifecycle": lifecycle,
        "storage_ledger": storage_ledger,
        "claim_boundary": (
            "Compatibility only; no native support, compression, quality, "
            "latency, throughput, speedup, or memory-reduction claim."
        ),
    }


def verify_run(run_dir, write_report=False, domain="canonical"):
    if domain not in {"canonical", "smoke"}:
        raise ValueError("verification domain must be canonical or smoke")
    destination = Path(run_dir)
    expected_case_count = (
        len(contract.build_case_matrix())
        if domain == "canonical"
        else len(SMOKE_CASE_IDS)
    )
    if not destination.is_dir():
        return {
            "schema_version": contract.SCHEMA_VERSION,
            "classification": "INCOMPLETE",
            "expected_case_count": expected_case_count,
            "observed_case_count": 0,
            "guards": {},
            "reasons": ["run directory does not exist"],
            "claim_boundary": (
                "Compatibility only; no native support, compression, "
                "quality, latency, throughput, speedup, or "
                "memory-reduction claim."
            ),
        }
    try:
        result = _verify_complete_run(destination, domain)
    except VerificationError as exc:
        result = {
            "schema_version": contract.SCHEMA_VERSION,
            "classification": "INCOMPLETE",
            "expected_case_count": expected_case_count,
            "observed_case_count": _observed_case_count(destination),
            "guards": {},
            "reasons": [str(exc)],
            "claim_boundary": (
                "Compatibility only; no native support, compression, "
                "quality, latency, throughput, speedup, or "
                "memory-reduction claim."
            ),
        }
    except SemanticFailure as exc:
        result = {
            "schema_version": contract.SCHEMA_VERSION,
            "classification": (
                "NO_GO" if domain == "canonical" else "INCOMPLETE"
            ),
            "expected_case_count": expected_case_count,
            "observed_case_count": _observed_case_count(destination),
            "guards": {"correctness_pass": False},
            "reasons": [str(exc)],
            "claim_boundary": (
                "Compatibility only; no native support, compression, "
                "quality, latency, throughput, speedup, or "
                "memory-reduction claim."
            ),
        }
    if write_report:
        _write_outputs(destination, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument(
        "--domain",
        choices=("canonical", "smoke"),
        default="canonical",
    )
    arguments = parser.parse_args(argv)
    result = verify_run(
        Path(arguments.run_dir),
        write_report=arguments.write_report,
        domain=arguments.domain,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
