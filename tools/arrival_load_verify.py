"""Independent verifier for production arrival-load gate artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import tarfile
import tempfile
from pathlib import Path


REQUIRED_FILES = (
    "run_manifest.json",
    "cost_calibration_capacity.json",
    "cost_calibration_manifest.jsonl",
    "cost_calibration_rows.jsonl",
    "cost_calibration_summary.json",
    "calibration_manifest.jsonl",
    "calibration_rows.jsonl",
    "workload_manifest.jsonl",
    "request_timeline.jsonl",
    "scheduler_trace.jsonl",
    "memory_trace.jsonl",
    "case_rows.jsonl",
    "summary.json",
    "report.md",
    "source_evidence.json",
    "source.patch",
    "source_snapshot.tar.gz",
    "artifact_hashes.json",
)

P4_FIELDS = (
    "chunked_prefill_adaptive_mixed",
    "chunked_prefill_adaptive_enter_waiting",
    "chunked_prefill_adaptive_exit_waiting",
    "chunked_prefill_adaptive_transition_steps",
    "chunked_prefill_adaptive_max_mixed_steps",
)

EXPECTED_P4 = {
    "chunked_prefill_decode_first": True,
    "chunked_prefill_max_consecutive_chunks": 0,
    "chunked_prefill_mixed_batch": False,
    "chunked_prefill_mixed_min_prompt_tokens": 0,
    "chunked_prefill_adaptive_mixed": True,
    "chunked_prefill_adaptive_enter_waiting": 8,
    "chunked_prefill_adaptive_exit_waiting": 2,
    "chunked_prefill_adaptive_transition_steps": 2,
    "chunked_prefill_adaptive_max_mixed_steps": 2,
}

EXPECTED_P5 = {
    "chunked_prefill_decode_first": True,
    "chunked_prefill_max_consecutive_chunks": 0,
    "chunked_prefill_mixed_batch": False,
    "chunked_prefill_mixed_min_prompt_tokens": 0,
    "chunked_prefill_adaptive_mixed": False,
    "chunked_prefill_slo_mixed": True,
    "chunked_prefill_slo_target_gap_ns": 64_000_000,
    "chunked_prefill_slo_reserve_ns": 8_000_000,
    "chunked_prefill_slo_min_chunk_tokens": 16,
    "chunked_prefill_slo_token_ladder": [
        128, 112, 96, 80, 64, 48, 32, 16,
    ],
}

EXPECTED_COST_ENGINE_CONFIG = {
    "max_num_batched_tokens": 16384,
    "max_num_seqs": 512,
    "max_num_prefill_tokens_per_step": 128,
    "enforce_eager": False,
    "chunked_prefill_decode_first": False,
    "chunked_prefill_max_consecutive_chunks": 0,
    "chunked_prefill_mixed_batch": True,
    "chunked_prefill_mixed_min_prompt_tokens": 0,
    "chunked_prefill_adaptive_mixed": False,
    "chunked_prefill_slo_mixed": False,
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_identity(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_cost_calibration_module():
    path = Path(__file__).with_name(
        "arrival_load_cost_calibration.py"
    )
    spec = importlib.util.spec_from_file_location(
        "arrival_load_cost_calibration_for_verifier",
        path,
    )
    if spec is None or spec.loader is None:
        raise ValueError("could not load cost calibration module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_policy_manifest(manifest: dict) -> dict:
    names = ("P0", "P4", "P5")
    aliases = manifest.get("canonical_policy_by_name")
    identities = manifest.get("policy_identity_by_name")
    resolved = manifest.get("resolved_policy_config_by_name")
    if (
        not isinstance(aliases, dict)
        or not isinstance(identities, dict)
        or not isinstance(resolved, dict)
        or tuple(aliases) != names
        or tuple(identities) != names
        or tuple(resolved) != names
        or any(aliases[name] != name for name in names)
    ):
        raise ValueError("invalid policy or case manifest")
    recomputed = {
        name: _canonical_identity(resolved[name])
        for name in names
    }
    if recomputed != identities:
        raise ValueError("policy identity mismatch")
    if len(set(identities.values())) != len(names):
        raise ValueError("unexpected policy identity collision")
    p4 = resolved["P4"]
    if any(p4.get(key) != value for key, value in EXPECTED_P4.items()):
        raise ValueError("invalid P4 resolved policy")
    if any(field not in p4 for field in P4_FIELDS):
        raise ValueError("invalid P4 resolved policy")
    p5 = resolved["P5"]
    if any(p5.get(key) != value for key, value in EXPECTED_P5.items()):
        raise ValueError("invalid P5 resolved policy")
    for field in (
        "chunked_prefill_slo_cost_intercept_ns",
        "chunked_prefill_slo_cost_per_prefill_token_ns",
    ):
        value = p5.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError("invalid P5 resolved policy")
    artifact_sha256 = p5.get("cost_calibration_artifact_sha256")
    if (
        not isinstance(artifact_sha256, str)
        or len(artifact_sha256) != 64
    ):
        raise ValueError("invalid P5 resolved policy")
    return p5


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file: {path.name}") from exc


def _read_jsonl(path: Path) -> list[dict]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"missing JSONL file: {path.name}") from exc
    if payload and not payload.endswith(b"\n"):
        raise ValueError(f"JSONL missing final newline: {path.name}")
    rows = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        if not line:
            raise ValueError(
                f"blank JSONL record: {path.name}:{line_number}"
            )
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"malformed JSONL: {path.name}:{line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"JSONL record must be object: {path.name}:{line_number}"
            )
        rows.append(row)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    return normalized


def _nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires samples")
    normalized = [_finite(value, "percentile sample") for value in values]
    normalized.sort()
    return normalized[math.ceil(len(normalized) * percentile) - 1]


def _verify_hashes(run_dir: Path) -> None:
    for name in REQUIRED_FILES:
        if not (run_dir / name).is_file():
            raise ValueError(f"missing artifact: {name}")
    hashes = _read_json(run_dir / "artifact_hashes.json")
    expected_names = set(REQUIRED_FILES) - {"artifact_hashes.json"}
    if set(hashes) != expected_names:
        raise ValueError("artifact hash manifest path set mismatch")
    for name in sorted(expected_names):
        if hashes[name] != _sha256_file(run_dir / name):
            raise ValueError(f"artifact hash mismatch: {name}")


def _verify_cost_calibration(
    run_dir: Path,
    manifest: dict,
    p5_config: dict,
) -> dict:
    calibration = _load_cost_calibration_module()
    capacity = _read_json(
        run_dir / "cost_calibration_capacity.json"
    )
    expected_base_sha256 = _canonical_identity(
        EXPECTED_COST_ENGINE_CONFIG
    )
    if (
        capacity.get("schema_version") != 1
        or capacity.get("base_engine_config_sha256")
        != expected_base_sha256
    ):
        raise ValueError(
            "cost calibration capacity identity mismatch"
        )
    num_kvcache_blocks = capacity.get("num_kvcache_blocks")
    block_size = capacity.get("block_size")
    expected_engine_config = {
        **EXPECTED_COST_ENGINE_CONFIG,
        "num_kvcache_blocks": num_kvcache_blocks,
    }
    if capacity.get("resolved_engine_config") != expected_engine_config:
        raise ValueError(
            "cost calibration resolved engine mismatch"
        )
    required_shapes = _read_jsonl(
        run_dir / "cost_calibration_manifest.jsonl"
    )
    expected_shapes = calibration.build_required_shapes(
        max_num_seqs=EXPECTED_COST_ENGINE_CONFIG["max_num_seqs"],
        max_prefill_tokens=EXPECTED_COST_ENGINE_CONFIG[
            "max_num_prefill_tokens_per_step"
        ],
        num_kvcache_blocks=num_kvcache_blocks,
        block_size=block_size,
    )
    if required_shapes != expected_shapes:
        raise ValueError("cost calibration shape manifest mismatch")
    raw_rows = _read_jsonl(
        run_dir / "cost_calibration_rows.jsonl"
    )
    source_sha256 = manifest.get("source_tree_sha256")
    environment_sha256 = manifest.get("environment_sha256")
    engine_config_sha256 = _canonical_identity(
        expected_engine_config
    )
    recomputed = calibration.build_cost_calibration_summary(
        source_tree_sha256=source_sha256,
        environment_sha256=environment_sha256,
        engine_config_sha256=engine_config_sha256,
        required_shapes=required_shapes,
        raw_rows=raw_rows,
    )
    recomputed["purpose"] = "authoritative"
    recorded = _read_json(
        run_dir / "cost_calibration_summary.json"
    )
    if recorded != recomputed:
        raise ValueError("cost calibration summary disagreement")
    artifact_sha256 = _sha256_file(
        run_dir / "cost_calibration_summary.json"
    )
    if (
        p5_config.get("cost_calibration_artifact_sha256")
        != artifact_sha256
        or p5_config.get(
            "chunked_prefill_slo_cost_intercept_ns"
        ) != recomputed["cost_intercept_ns"]
        or p5_config.get(
            "chunked_prefill_slo_cost_per_prefill_token_ns"
        ) != recomputed["cost_per_prefill_token_ns"]
    ):
        raise ValueError("P5 cost calibration identity mismatch")
    marker = manifest.get("cost_calibration_verification")
    if not isinstance(marker, dict):
        raise ValueError("missing cost calibration verification")
    expected_marker = {
        "status": "PASS",
        "run_tag": marker.get("run_tag"),
        "artifact_sha256": artifact_sha256,
        "source_tree_sha256": source_sha256,
        "environment_sha256": environment_sha256,
    }
    if marker != expected_marker or not isinstance(
        marker["run_tag"], str
    ):
        raise ValueError("cost calibration verification mismatch")
    return recomputed


def _safe_extract_snapshot(archive_path: Path, output_dir: Path) -> Path:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            path = Path(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or member.issym()
                or member.islnk()
            ):
                raise ValueError("unsafe source snapshot member")
        archive.extractall(output_dir)
    source_root = output_dir / "source"
    if not source_root.is_dir():
        raise ValueError("source snapshot is missing source root")
    return source_root


def _verify_source(run_dir: Path, manifest: dict) -> None:
    evidence = _read_json(run_dir / "source_evidence.json")
    if evidence.get("schema_version") != 1:
        raise ValueError("unsupported source evidence schema")
    patch_payload = (run_dir / "source.patch").read_bytes()
    if evidence.get("patch_size_bytes") != len(patch_payload):
        raise ValueError("source patch size mismatch")
    if (
        evidence.get("patch_sha256")
        != hashlib.sha256(patch_payload).hexdigest()
    ):
        raise ValueError("source patch hash mismatch")
    expected_files = evidence.get("files")
    if not isinstance(expected_files, list):
        raise ValueError("invalid source evidence files")
    with tempfile.TemporaryDirectory() as temporary:
        source_root = _safe_extract_snapshot(
            run_dir / "source_snapshot.tar.gz",
            Path(temporary),
        )
        actual_files = []
        for path in sorted(source_root.rglob("*")):
            if path.is_symlink():
                raise ValueError("source snapshot contains symlink")
            if path.is_file():
                actual_files.append({
                    "path": path.relative_to(source_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                })
        if actual_files != expected_files:
            raise ValueError("source snapshot file identity mismatch")
        tree_sha256 = hashlib.sha256(
            _canonical_bytes(actual_files)
        ).hexdigest()
        if tree_sha256 != evidence.get("tree_sha256"):
            raise ValueError("source tree hash mismatch")
        if tree_sha256 != manifest.get("source_tree_sha256"):
            raise ValueError("manifest source tree hash mismatch")


def _verify_ports(manifest: dict) -> None:
    rows = manifest.get("process_port_pairs")
    if not isinstance(rows, list):
        raise ValueError("invalid process port records")
    pairs = []
    case_ids = []
    for row in rows:
        case_id = row.get("case_id")
        dist_port = row.get("tinyvllm_dist_port")
        master_port = row.get("master_port")
        if (
            not isinstance(case_id, str)
            or not isinstance(dist_port, int)
            or not isinstance(master_port, int)
            or dist_port <= 0
            or master_port <= 0
            or dist_port == master_port
        ):
            raise ValueError("invalid process port record")
        case_ids.append(case_id)
        pairs.append((dist_port, master_port))
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate process port pair")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate process case id")
    if set(case_ids) != set(manifest.get("expected_case_ids", [])):
        raise ValueError("process case matrix mismatch")


def _verify_p4_scheduler_trace(
    rows: list[dict],
    *,
    enter_waiting: int,
    exit_waiting: int,
    transition_steps: int,
    max_mixed_steps: int,
) -> None:
    if not rows:
        raise ValueError("missing P4 scheduler trace")
    expected_state = "inactive"
    expected_high = 0
    expected_low = 0
    expected_mixed = 0
    previous_controller_after = None
    controller_fields = (
        "adaptive_mixed_state",
        "adaptive_high_streak",
        "adaptive_low_streak",
        "adaptive_consecutive_mixed_steps",
    )
    required_fields = controller_fields + (
        "waiting_seq_ids",
        "prefilling_seq_ids",
        "running_seq_ids",
    )
    for expected_step, row in enumerate(rows):
        if row.get("step_index") != expected_step:
            raise ValueError("invalid P4 scheduler step sequence")
        before = row.get("queue_before")
        after = row.get("queue_after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError("missing P4 queue snapshot")
        if any(field not in before for field in required_fields):
            raise ValueError("missing P4 controller field")
        if any(field not in after for field in required_fields):
            raise ValueError("missing P4 controller field")
        controller_before = tuple(before[field] for field in controller_fields)
        if (
            previous_controller_after is not None
            and controller_before != previous_controller_after
        ):
            raise ValueError("P4 controller snapshots are not contiguous")
        for snapshot in (before, after):
            if snapshot["adaptive_mixed_state"] not in {
                "inactive",
                "active",
                "draining",
            }:
                raise ValueError("illegal adaptive state")
            counters = tuple(snapshot[field] for field in controller_fields[1:])
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in counters
            ):
                raise ValueError("invalid adaptive counter")
            if (
                counters[0] >= transition_steps
                or counters[1] >= transition_steps
                or counters[2] > max_mixed_steps
            ):
                raise ValueError("invalid adaptive counter")
            queue_sets = [
                set(snapshot[name])
                for name in (
                    "waiting_seq_ids",
                    "prefilling_seq_ids",
                    "running_seq_ids",
                )
            ]
            if (
                queue_sets[0] & queue_sets[1]
                or queue_sets[0] & queue_sets[2]
                or queue_sets[1] & queue_sets[2]
            ):
                raise ValueError("duplicate P4 queue ownership")
        if controller_before != (
            expected_state,
            expected_high,
            expected_low,
            expected_mixed,
        ):
            raise ValueError("P4 controller continuity mismatch")

        waiting_depth = len(before["waiting_seq_ids"])
        eligible = bool(
            before["running_seq_ids"]
            and (
                before["waiting_seq_ids"]
                or before["prefilling_seq_ids"]
            )
        )
        state = expected_state
        high = expected_high
        low = expected_low
        if not eligible:
            high = 0
            low = 0
        elif state == "inactive":
            low = 0
            high = high + 1 if waiting_depth >= enter_waiting else 0
            if high >= transition_steps:
                state = "active"
                high = 0
        elif state == "active":
            high = 0
            low = low + 1 if waiting_depth <= exit_waiting else 0
            if low >= transition_steps:
                state = (
                    "draining"
                    if before["prefilling_seq_ids"]
                    else "inactive"
                )
                low = 0
        else:
            low = 0
            high = high + 1 if waiting_depth >= enter_waiting else 0
            if high >= transition_steps:
                state = "active"
                high = 0
            elif not before["prefilling_seq_ids"]:
                state = "inactive"
                expected_mixed = 0

        if state == "draining" and not before["prefilling_seq_ids"]:
            state = "inactive"
            expected_mixed = 0

        branch = row.get("policy_branch")
        scheduled = row.get("scheduled")
        if not isinstance(scheduled, list):
            raise ValueError("invalid P4 scheduled rows")
        has_prefill = any(
            item.get("is_decode") is False for item in scheduled
        )
        has_decode = any(
            item.get("is_decode") is True for item in scheduled
        )
        if branch == "adaptive_mixed_prefill_decode":
            if not has_prefill or not has_decode:
                raise ValueError("adaptive mixed branch role mismatch")
            expected_mixed += 1
            if expected_mixed > max_mixed_steps:
                raise ValueError("adaptive mixed service bound exceeded")
        elif branch in {
            "adaptive_mixed_decode_first",
            "adaptive_mixed_decode_yield",
            "adaptive_mixed_decode_fallback",
        }:
            if has_prefill:
                raise ValueError("decode-only adaptive branch has prefill")
            expected_mixed = 0
        elif branch == "adaptive_mixed_chunked_prefill":
            if before["running_seq_ids"] or has_decode:
                raise ValueError("adaptive chunked prefill has decode")
            expected_mixed = 0
        else:
            raise ValueError("illegal P4 policy branch")

        if state == "draining":
            newly_prefilling = (
                set(after["prefilling_seq_ids"])
                - set(before["prefilling_seq_ids"])
            )
            if newly_prefilling & set(before["waiting_seq_ids"]):
                raise ValueError("new waiting admission during draining")

        if (
            not after["waiting_seq_ids"]
            and not after["prefilling_seq_ids"]
            and not after["running_seq_ids"]
        ):
            state = "inactive"
            high = 0
            low = 0
            expected_mixed = 0

        if after["adaptive_mixed_state"] != state:
            raise ValueError("adaptive state transition mismatch")
        if after["adaptive_high_streak"] != high:
            raise ValueError("adaptive high streak mismatch")
        if after["adaptive_low_streak"] != low:
            raise ValueError("adaptive low streak mismatch")
        if after["adaptive_consecutive_mixed_steps"] != expected_mixed:
            raise ValueError("adaptive mixed counter mismatch")
        expected_state = state
        expected_high = high
        expected_low = low
        previous_controller_after = tuple(
            after[field] for field in controller_fields
        )


def _integer(value, label: str, *, nullable: bool = False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _scheduled_shape(row: dict) -> tuple[list[int], int, int]:
    scheduled = row.get("scheduled")
    if not isinstance(scheduled, list):
        raise ValueError("invalid P5 scheduled rows")
    scheduled_ids = []
    decode_ids = []
    prefill_tokens = 0
    for item in scheduled:
        if not isinstance(item, dict):
            raise ValueError("invalid P5 scheduled row")
        seq_id = _integer(item.get("seq_id"), "scheduled seq id")
        scheduled_ids.append(seq_id)
        if item.get("is_decode") is True:
            decode_ids.append(seq_id)
            continue
        if item.get("is_decode") is not False:
            raise ValueError("invalid P5 scheduled role")
        start = _integer(
            item.get("prefill_chunk_start"),
            "prefill chunk start",
        )
        end = _integer(
            item.get("prefill_chunk_end"),
            "prefill chunk end",
        )
        if end <= start:
            raise ValueError("invalid P5 prefill chunk")
        prefill_tokens += end - start
    if len(scheduled_ids) != len(set(scheduled_ids)):
        raise ValueError("duplicate P5 scheduled sequence")
    return decode_ids, prefill_tokens, len(scheduled) - len(decode_ids)


def _next_demand_state(
    *,
    state: str,
    high_streak: int,
    low_streak: int,
    queue_before: dict,
    enter_waiting: int,
    exit_waiting: int,
    transition_steps: int,
) -> tuple[str, int, int]:
    running = queue_before["running_seq_ids"]
    waiting = queue_before["waiting_seq_ids"]
    prefilling = queue_before["prefilling_seq_ids"]
    if not running or not (waiting or prefilling):
        return state, 0, 0
    waiting_depth = len(waiting)
    if state == "inactive":
        low_streak = 0
        high_streak = (
            high_streak + 1
            if waiting_depth >= enter_waiting
            else 0
        )
        if high_streak >= transition_steps:
            return "active", 0, low_streak
        return state, high_streak, low_streak
    if state == "active":
        high_streak = 0
        low_streak = (
            low_streak + 1
            if waiting_depth <= exit_waiting
            else 0
        )
        if low_streak >= transition_steps:
            return (
                "draining" if prefilling else "inactive",
                high_streak,
                0,
            )
        return state, high_streak, low_streak
    low_streak = 0
    high_streak = (
        high_streak + 1
        if waiting_depth >= enter_waiting
        else 0
    )
    if high_streak >= transition_steps:
        return "active", 0, low_streak
    if not prefilling:
        return "inactive", high_streak, low_streak
    return state, high_streak, low_streak


def _select_p5_chunk(
    remaining_slack_ns: int,
    *,
    cost_intercept_ns: int,
    cost_per_prefill_token_ns: int,
    token_ladder: list[int],
) -> tuple[int | None, int | None]:
    for tokens in token_ladder:
        predicted = (
            cost_intercept_ns
            + tokens * cost_per_prefill_token_ns
        )
        if predicted <= remaining_slack_ns:
            return tokens, predicted
    return None, None


def _verify_p5_scheduler_trace(
    rows: list[dict],
    *,
    config: dict,
) -> dict:
    if not rows:
        raise ValueError("missing P5 scheduler trace")
    progress_by_seq_id = {}
    demand_state = "inactive"
    high_streak = 0
    low_streak = 0
    last_decision_now_ns = None
    previous_queue_after = None
    histogram = {}
    diagnostics = {
        "mixed_decision_count": 0,
        "slo_suppression_count": 0,
        "draining_decision_count": 0,
        "selected_chunk_histogram": histogram,
        "envelope_underprediction_count": 0,
        "missing_progress_count": 0,
        "clock_invalid_count": 0,
    }
    target_gap_ns = config["chunked_prefill_slo_target_gap_ns"]
    reserve_ns = config["chunked_prefill_slo_reserve_ns"]
    intercept_ns = config[
        "chunked_prefill_slo_cost_intercept_ns"
    ]
    per_token_ns = config[
        "chunked_prefill_slo_cost_per_prefill_token_ns"
    ]
    ladder = config["chunked_prefill_slo_token_ladder"]
    for expected_step, row in enumerate(rows):
        if row.get("step_index") != expected_step:
            raise ValueError("invalid P5 scheduler step sequence")
        before = row.get("queue_before")
        after = row.get("queue_after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError("missing P5 queue snapshot")
        queue_fields = (
            "waiting_seq_ids",
            "prefilling_seq_ids",
            "running_seq_ids",
        )
        if any(
            not isinstance(snapshot.get(field), list)
            for snapshot in (before, after)
            for field in queue_fields
        ):
            raise ValueError("invalid P5 queue snapshot")
        if (
            previous_queue_after is not None
            and before != previous_queue_after
        ):
            raise ValueError("P5 queue snapshots are not contiguous")
        for snapshot in (before, after):
            queue_sets = [set(snapshot[field]) for field in queue_fields]
            if (
                queue_sets[0] & queue_sets[1]
                or queue_sets[0] & queue_sets[2]
                or queue_sets[1] & queue_sets[2]
            ):
                raise ValueError("duplicate P5 queue ownership")

        decision_now_ns = _integer(
            row.get("decision_now_ns"),
            "P5 decision time",
        )
        step_end_ns = _integer(
            row.get("step_end_ns"),
            "P5 step end",
        )
        if (
            step_end_ns < decision_now_ns
            or (
                last_decision_now_ns is not None
                and decision_now_ns < last_decision_now_ns
            )
        ):
            raise ValueError("invalid P5 decision clock")
        last_decision_now_ns = decision_now_ns
        if row.get("clock_invalid") is not False:
            raise ValueError("P5 clock violation")
        if row.get("clock_invalid_reason") is not None:
            raise ValueError("P5 clock reason mismatch")
        if (
            row.get("target_gap_ns") != target_gap_ns
            or row.get("reserve_ns") != reserve_ns
            or row.get("cost_intercept_ns") != intercept_ns
            or row.get("cost_per_prefill_token_ns") != per_token_ns
            or row.get("candidate_chunk_tokens") != ladder
        ):
            raise ValueError("P5 decision coefficient mismatch")
        if row.get("demand_state_before") != demand_state:
            raise ValueError("P5 demand state continuity mismatch")

        decode_ids, actual_prefill_tokens, prefill_rows = (
            _scheduled_shape(row)
        )
        running_ids = list(before["running_seq_ids"])
        suppression = row.get("suppression_reason")
        expected_selected = None
        expected_predicted = None
        expected_oldest = None
        expected_progress = None
        expected_age = None
        expected_slack = None
        expected_suppression = None

        if running_ids:
            missing = [
                seq_id for seq_id in running_ids
                if seq_id not in progress_by_seq_id
            ]
            if missing:
                expected_suppression = "missing_decode_progress"
                diagnostics["missing_progress_count"] += 1
            else:
                expected_progress, expected_oldest = min(
                    (progress_by_seq_id[seq_id], seq_id)
                    for seq_id in running_ids
                )
                if expected_progress > decision_now_ns:
                    raise ValueError("P5 progress timestamp in future")
                expected_age = decision_now_ns - expected_progress
                expected_slack = (
                    target_gap_ns - reserve_ns - expected_age
                )
                (
                    demand_state,
                    high_streak,
                    low_streak,
                ) = _next_demand_state(
                    state=demand_state,
                    high_streak=high_streak,
                    low_streak=low_streak,
                    queue_before=before,
                    enter_waiting=config[
                        "chunked_prefill_adaptive_enter_waiting"
                    ],
                    exit_waiting=config[
                        "chunked_prefill_adaptive_exit_waiting"
                    ],
                    transition_steps=config[
                        "chunked_prefill_adaptive_transition_steps"
                    ],
                )
                if demand_state == "inactive":
                    expected_suppression = "inactive"
                elif expected_slack <= 0:
                    expected_suppression = "no_slack"
                else:
                    (
                        expected_selected,
                        expected_predicted,
                    ) = _select_p5_chunk(
                        expected_slack,
                        cost_intercept_ns=intercept_ns,
                        cost_per_prefill_token_ns=per_token_ns,
                        token_ladder=ladder,
                    )
                    if expected_selected is None:
                        expected_suppression = "cost_suppressed"
        else:
            (
                demand_state,
                high_streak,
                low_streak,
            ) = _next_demand_state(
                state=demand_state,
                high_streak=high_streak,
                low_streak=low_streak,
                queue_before=before,
                enter_waiting=config[
                    "chunked_prefill_adaptive_enter_waiting"
                ],
                exit_waiting=config[
                    "chunked_prefill_adaptive_exit_waiting"
                ],
                transition_steps=config[
                    "chunked_prefill_adaptive_transition_steps"
                ],
            )

        expected_fields = {
            "oldest_decode_seq_id": expected_oldest,
            "oldest_decode_progress_ns": expected_progress,
            "oldest_decode_age_ns": expected_age,
            "remaining_slack_ns": expected_slack,
            "predicted_step_ns": expected_predicted,
            "selected_chunk_tokens": expected_selected,
            "suppression_reason": expected_suppression,
            "demand_state_after": demand_state,
        }
        mismatches = {
            field: {
                "recorded": row.get(field),
                "expected": value,
            }
            for field, value in expected_fields.items()
            if row.get(field) != value
        }
        if mismatches:
            raise ValueError(
                "P5 decision reconstruction mismatch at step "
                f"{expected_step}: "
                + json.dumps(mismatches, sort_keys=True)
            )
        if row.get("actual_prefill_tokens") != actual_prefill_tokens:
            raise ValueError("P5 actual prefill mismatch")
        if row.get("scheduled_decode_seq_ids") != decode_ids:
            raise ValueError("P5 scheduled decode mismatch")
        if expected_suppression is not None and (
            actual_prefill_tokens != 0 or prefill_rows != 0
        ):
            raise ValueError("suppressed P5 decision admitted prefill")
        if expected_selected is not None:
            if (
                not decode_ids
                or prefill_rows <= 0
                or expected_oldest not in decode_ids
                or actual_prefill_tokens <= 0
                or actual_prefill_tokens > expected_selected
            ):
                raise ValueError("invalid P5 mixed admission")
            diagnostics["mixed_decision_count"] += 1
            key = str(expected_selected)
            histogram[key] = histogram.get(key, 0) + 1
            if row.get("actual_step_duration_ns") > expected_predicted:
                diagnostics["envelope_underprediction_count"] += 1
        if expected_suppression in ("no_slack", "cost_suppressed"):
            diagnostics["slo_suppression_count"] += 1
        if demand_state == "draining":
            diagnostics["draining_decision_count"] += 1

        duration_ns = _integer(
            row.get("actual_step_duration_ns"),
            "P5 actual step duration",
        )
        if duration_ns != step_end_ns - decision_now_ns:
            raise ValueError("P5 step duration mismatch")
        token_deltas = row.get("new_completion_tokens_by_seq")
        progress_updates = row.get("decode_progress_updates")
        if not isinstance(token_deltas, dict) or not isinstance(
            progress_updates, dict
        ):
            raise ValueError("invalid P5 progress evidence")
        scheduled_ids = {
            item["seq_id"] for item in row["scheduled"]
        }
        expected_progress_updates = {}
        for raw_seq_id, tokens in token_deltas.items():
            try:
                seq_id = int(raw_seq_id)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid P5 progress sequence") from exc
            if (
                not isinstance(tokens, list)
                or seq_id not in scheduled_ids
            ):
                raise ValueError("invalid P5 token delta")
            if tokens:
                expected_progress_updates[str(seq_id)] = step_end_ns
        if progress_updates != expected_progress_updates:
            raise ValueError("invalid P5 progress update")
        for raw_seq_id, timestamp_ns in progress_updates.items():
            seq_id = int(raw_seq_id)
            progress_by_seq_id[seq_id] = timestamp_ns
        finished = row.get("finished_progress_entries_removed")
        if not isinstance(finished, list):
            raise ValueError("invalid P5 finished progress evidence")
        queue_after_ids = {
            seq_id
            for field in queue_fields
            for seq_id in after[field]
        }
        expected_finished = sorted(
            int(raw_seq_id)
            for raw_seq_id in expected_progress_updates
            if int(raw_seq_id) not in queue_after_ids
        )
        if finished != expected_finished:
            raise ValueError("invalid P5 finished progress evidence")
        for seq_id in finished:
            progress_by_seq_id.pop(seq_id, None)
        previous_queue_after = after
    diagnostics["selected_chunk_histogram"] = dict(sorted(
        histogram.items(),
        key=lambda item: int(item[0]),
    ))
    return diagnostics


def _request_metrics(workload: dict, timeline: dict) -> dict:
    names = (
        "scheduled_arrival_ns",
        "actual_arrival_ns",
        "first_scheduled_ns",
        "first_token_ns",
        "completion_ns",
    )
    times = {
        name: _finite(timeline.get(name), name)
        for name in names
    }
    if not (
        times["scheduled_arrival_ns"]
        <= times["actual_arrival_ns"]
        <= times["first_scheduled_ns"]
        <= times["first_token_ns"]
        <= times["completion_ns"]
    ):
        raise ValueError("impossible timestamp ordering")
    token_times = [
        _finite(value, "token timestamp")
        for value in timeline.get("token_timestamps_ns", [])
    ]
    output_ids = timeline.get("output_token_ids")
    if (
        not isinstance(output_ids, list)
        or len(output_ids) != workload.get("requested_output_tokens")
        or len(token_times) != len(output_ids)
        or not token_times
    ):
        raise ValueError("token timestamp or output count mismatch")
    if token_times[0] != times["first_token_ns"]:
        raise ValueError("first token timestamp mismatch")
    if token_times[-1] > times["completion_ns"]:
        raise ValueError("token timestamp exceeds completion")
    if any(
        current < previous
        for previous, current in zip(token_times, token_times[1:])
    ):
        raise ValueError("non-monotonic token timestamps")
    if timeline.get("finish_reason") != "length":
        raise ValueError("unexpected finish reason")
    if timeline.get("error") is not None:
        raise ValueError("request error")
    itl = [
        current - previous
        for previous, current in zip(token_times, token_times[1:])
    ]
    return {
        **workload,
        "output_token_ids": list(output_ids),
        "injection_lag_ns": (
            times["actual_arrival_ns"]
            - times["scheduled_arrival_ns"]
        ),
        "queue_delay_ns": (
            times["first_scheduled_ns"]
            - times["actual_arrival_ns"]
        ),
        "ttft_ns": (
            times["first_token_ns"]
            - times["scheduled_arrival_ns"]
        ),
        "e2e_ns": (
            times["completion_ns"]
            - times["scheduled_arrival_ns"]
        ),
        "itl_ns": itl,
        "maximum_decode_gap_ns": max(itl) if itl else None,
        "scheduled_arrival_ns": times["scheduled_arrival_ns"],
        "completion_ns": times["completion_ns"],
    }


def _percentiles(values: list[float], prefix: str) -> dict:
    return {
        f"p50_{prefix}": _nearest_rank(values, 0.50),
        f"p95_{prefix}": _nearest_rank(values, 0.95),
        f"p99_{prefix}": _nearest_rank(values, 0.99),
    }


def _jain_index(values: list[float]) -> float:
    if not values or any(value < 0.0 for value in values):
        raise ValueError("invalid Jain index samples")
    denominator = len(values) * sum(value * value for value in values)
    if denominator == 0.0:
        return 0.0
    return (sum(values) ** 2) / denominator


def _recompute_case(
    case_id: str,
    timeline_rows: list[dict],
    scheduler_rows: list[dict],
    memory_rows: list[dict],
    workload_by_id: dict[str, dict],
    p5_policy: dict | None = None,
) -> dict:
    case_timeline = [
        row for row in timeline_rows if row.get("case_id") == case_id
    ]
    if not case_timeline:
        raise ValueError(f"missing request timeline: {case_id}")
    if len({
        row.get("request_id") for row in case_timeline
    }) != len(case_timeline):
        raise ValueError(f"duplicate request timeline: {case_id}")
    if len({row.get("seq_id") for row in case_timeline}) != len(
        case_timeline
    ):
        raise ValueError(f"duplicate request binding: {case_id}")
    case_scheduler = [
        row for row in scheduler_rows if row.get("case_id") == case_id
    ]
    if not case_scheduler:
        raise ValueError(f"missing scheduler trace: {case_id}")
    step_indices = [row.get("step_index") for row in case_scheduler]
    if step_indices != list(range(len(step_indices))):
        raise ValueError(f"invalid scheduler step sequence: {case_id}")
    case_memory = [
        row for row in memory_rows if row.get("case_id") == case_id
    ]
    if not case_memory:
        raise ValueError(f"missing memory trace: {case_id}")

    request_rows = []
    for timeline in case_timeline:
        request_id = timeline.get("request_id")
        if request_id not in workload_by_id:
            raise ValueError(f"unexpected request id: {request_id}")
        workload = workload_by_id[request_id]
        metrics = _request_metrics(workload, timeline)
        if workload.get("warmup", False):
            continue
        request_rows.append(metrics)
    if not request_rows:
        raise ValueError(f"case has no measured requests: {case_id}")
    start_ns = min(row["scheduled_arrival_ns"] for row in request_rows)
    end_ns = max(row["completion_ns"] for row in request_rows)
    duration_s = (end_ns - start_ns) / 1_000_000_000.0
    if duration_s <= 0.0:
        raise ValueError(f"invalid measurement duration: {case_id}")
    itl_values = [
        value
        for row in request_rows
        for value in row["itl_ns"]
    ]
    if not itl_values:
        raise ValueError(f"case has no ITL samples: {case_id}")
    metrics = {
        "request_throughput_rps": len(request_rows) / duration_s,
        "output_token_throughput_tps": sum(
            len(row["output_token_ids"]) for row in request_rows
        ) / duration_s,
        "maximum_injection_lag_ns": max(
            row["injection_lag_ns"] for row in request_rows
        ),
        **_percentiles(
            [row["injection_lag_ns"] for row in request_rows],
            "injection_lag_ns",
        ),
        **_percentiles(
            [row["queue_delay_ns"] for row in request_rows],
            "queue_delay_ns",
        ),
        **_percentiles(
            [row["ttft_ns"] for row in request_rows],
            "ttft_ns",
        ),
        **_percentiles(itl_values, "itl_ns"),
        **_percentiles(
            [row["e2e_ns"] for row in request_rows],
            "e2e_ns",
        ),
        "maximum_decode_gap_ns": max(
            row["maximum_decode_gap_ns"]
            for row in request_rows
            if row["maximum_decode_gap_ns"] is not None
        ),
        "peak_cuda_allocated_bytes": int(max(
            _finite(row.get("cuda_allocated_bytes"), "allocated memory")
            for row in case_memory
        )),
        "peak_cuda_reserved_bytes": int(max(
            _finite(row.get("cuda_reserved_bytes"), "reserved memory")
            for row in case_memory
        )),
        "peak_used_kv_blocks": int(max(
            _finite(row.get("used_kv_blocks"), "used KV blocks")
            for row in case_memory
        )),
        "peak_kv_bytes": int(max(
            _finite(row.get("used_kv_blocks"), "used KV blocks")
            * _finite(row.get("kv_block_bytes"), "KV block bytes")
            for row in case_memory
        )),
    }
    service_buckets = {}
    service_rates = []
    for bucket in sorted({
        row["service_time_bucket"] for row in request_rows
    }):
        bucket_rows = [
            row for row in request_rows
            if row["service_time_bucket"] == bucket
        ]
        bucket_metrics = {
            "completed_requests": len(bucket_rows),
            "request_throughput_rps": len(bucket_rows) / duration_s,
            "worst_e2e_ns": max(
                row["e2e_ns"] for row in bucket_rows
            ),
            **_percentiles(
                [row["e2e_ns"] for row in bucket_rows],
                "e2e_ns",
            ),
        }
        service_buckets[bucket] = bucket_metrics
        service_rates.append(
            bucket_metrics["request_throughput_rps"]
        )
    metrics["service_buckets"] = service_buckets
    metrics["jain_service_rate_index"] = _jain_index(
        service_rates
    )
    first = case_timeline[0]
    result = {
        "case_id": case_id,
        "policy": first["policy"],
        "scenario": first["scenario"],
        "repetition": first["repetition"],
        "status": "PASS",
        "correctness": {
            "exact_outputs": True,
            "complete_requests": True,
            "no_starvation": True,
            "valid_lifecycle": True,
            "stable_p0_outputs": True,
        },
        "metrics": metrics,
    }
    if p5_policy is not None:
        result["p5_policy"] = p5_policy
    return result


def _ratio(candidate: dict, baseline: dict, metric: str) -> float:
    baseline_value = _finite(
        baseline["metrics"].get(metric),
        f"baseline {metric}",
    )
    candidate_value = _finite(
        candidate["metrics"].get(metric),
        f"candidate {metric}",
    )
    if baseline_value <= 0.0:
        raise ValueError(f"baseline {metric} must be positive")
    return candidate_value / baseline_value


def _candidate_result(
    policy: str,
    paired: list[tuple[dict, dict]],
) -> dict:
    metric_names = (
        "request_throughput_rps",
        "p95_ttft_ns",
        "p95_itl_ns",
        "p99_ttft_ns",
        "p99_itl_ns",
        "p99_e2e_ns",
        "maximum_decode_gap_ns",
        "peak_cuda_reserved_bytes",
        "peak_kv_bytes",
    )
    ratios = {
        metric: [
            _ratio(candidate, baseline, metric)
            for baseline, candidate in paired
        ]
        for metric in metric_names
    }
    bucket_ratios = []
    for baseline, candidate in paired:
        baseline_buckets = baseline["metrics"]["service_buckets"]
        candidate_buckets = candidate["metrics"]["service_buckets"]
        if set(baseline_buckets) != set(candidate_buckets):
            raise ValueError("service bucket set mismatch")
        for bucket in sorted(baseline_buckets):
            baseline_value = _finite(
                baseline_buckets[bucket]["p95_e2e_ns"],
                "baseline bucket p95",
            )
            candidate_value = _finite(
                candidate_buckets[bucket]["p95_e2e_ns"],
                "candidate bucket p95",
            )
            bucket_ratios.append(candidate_value / baseline_value)
    median = {
        metric: statistics.median(values)
        for metric, values in ratios.items()
    }
    worst = {
        "request_throughput_rps": min(
            ratios["request_throughput_rps"]
        ),
        **{
            metric: max(values)
            for metric, values in ratios.items()
            if metric != "request_throughput_rps"
        },
    }
    guard_failures = []
    for metric in (
        "p99_ttft_ns",
        "p99_itl_ns",
        "p99_e2e_ns",
        "maximum_decode_gap_ns",
    ):
        if worst[metric] > 1.10:
            guard_failures.append(f"{metric} regression exceeds 10%")
    if bucket_ratios and max(bucket_ratios) > 1.10:
        guard_failures.append(
            "service bucket p95 E2E regression exceeds 10%"
        )
    if policy == "P5":
        guard_failures.extend(
            _p5_guard_failures(paired)
        )
    median_paths = {
        "throughput": (
            median["request_throughput_rps"] >= 1.05
            and median["p95_ttft_ns"] <= 1.05
            and median["p95_itl_ns"] <= 1.05
        ),
        "latency": (
            (
                median["p95_ttft_ns"] <= 0.90
                and median["p95_itl_ns"] <= 1.05
            )
            or (
                median["p95_itl_ns"] <= 0.90
                and median["p95_ttft_ns"] <= 1.05
            )
        ) and median["request_throughput_rps"] >= 0.98,
        "memory": (
            min(
                median["peak_cuda_reserved_bytes"],
                median["peak_kv_bytes"],
            ) <= 0.95
            and median["request_throughput_rps"] >= 0.98
            and median["p95_ttft_ns"] <= 1.02
            and median["p95_itl_ns"] <= 1.02
        ),
    }
    worst_paths = {
        "throughput": (
            worst["request_throughput_rps"] >= 1.05
            and worst["p95_ttft_ns"] <= 1.05
            and worst["p95_itl_ns"] <= 1.05
        ),
        "latency": (
            (
                worst["p95_ttft_ns"] <= 0.90
                and worst["p95_itl_ns"] <= 1.05
            )
            or (
                worst["p95_itl_ns"] <= 0.90
                and worst["p95_ttft_ns"] <= 1.05
            )
        ) and worst["request_throughput_rps"] >= 0.98,
        "memory": (
            min(
                worst["peak_cuda_reserved_bytes"],
                worst["peak_kv_bytes"],
            ) <= 0.95
            and worst["request_throughput_rps"] >= 0.98
            and worst["p95_ttft_ns"] <= 1.02
            and worst["p95_itl_ns"] <= 1.02
        ),
    }
    benefit_path = next(
        (
            path
            for path in ("throughput", "latency", "memory")
            if median_paths[path] and worst_paths[path]
        ),
        None,
    )
    favorable = (
        median["request_throughput_rps"] > 1.0
        or median["p95_ttft_ns"] < 1.0
        or median["p95_itl_ns"] < 1.0
        or median["peak_cuda_reserved_bytes"] < 1.0
        or median["peak_kv_bytes"] < 1.0
    )
    if guard_failures:
        classification = "NO_GO"
    elif benefit_path is not None:
        classification = "GO"
    elif favorable:
        classification = "PROMISING_NOT_PROVEN"
    else:
        classification = "NO_GO"
    return {
        "policy": policy,
        "classification": classification,
        "benefit_path": benefit_path,
        "median_ratios": median,
        "worst_repetition_ratios": worst,
        "guard_failures": guard_failures,
    }


def _p5_guard_failures(
    paired: list[tuple[dict, dict]],
) -> list[str]:
    scenarios = {
        candidate["scenario"] for _, candidate in paired
    }
    if scenarios != {
        "steady_moderate",
        "steady_high",
        "burst",
        "long_prompt_pressure",
        "decode_heavy",
        "mixed_service_fairness",
    }:
        return []
    failures = []
    burst_ratios = []
    burst_has_three_chunk_sizes = False
    non_burst_suppression_count = 0
    envelope_underprediction_count = 0
    missing_progress_count = 0
    clock_invalid_count = 0
    for baseline, candidate in paired:
        scenario = candidate["scenario"]
        if (
            scenario == "long_prompt_pressure"
            and _ratio(candidate, baseline, "p95_itl_ns") > 1.05
        ):
            failures.append(
                "long_prompt_pressure p95 ITL exceeds 5%"
            )
        if scenario == "burst":
            burst_ratios.append(_ratio(
                candidate,
                baseline,
                "request_throughput_rps",
            ))
        policy = candidate.get("p5_policy")
        if not isinstance(policy, dict):
            raise ValueError("P5 canonical row missing p5_policy")
        histogram = policy.get("selected_chunk_histogram")
        if not isinstance(histogram, dict):
            raise ValueError("invalid P5 selected chunk histogram")
        if scenario == "burst" and len(histogram) >= 3:
            burst_has_three_chunk_sizes = True
        if scenario != "burst":
            non_burst_suppression_count += int(
                policy.get("slo_suppression_count", 0)
            )
        envelope_underprediction_count += int(
            policy.get("envelope_underprediction_count", 0)
        )
        missing_progress_count += int(
            policy.get("missing_progress_count", 0)
        )
        clock_invalid_count += int(
            policy.get("clock_invalid_count", 0)
        )
    if not burst_ratios or statistics.median(burst_ratios) < 1.25:
        failures.append("burst median throughput below 1.25x")
    if not burst_has_three_chunk_sizes:
        failures.append(
            "no burst repetition selected three chunk sizes"
        )
    if non_burst_suppression_count <= 0:
        failures.append("no non-burst SLO suppression")
    if envelope_underprediction_count > 0:
        failures.append("P5 envelope underprediction detected")
    if missing_progress_count > 0:
        failures.append("P5 missing decode progress detected")
    if clock_invalid_count > 0:
        failures.append("P5 clock violation detected")
    return failures


def _classify(manifest: dict, rows: list[dict]) -> dict:
    required_scenarios = manifest.get("required_scenarios")
    repetitions = manifest.get("measured_repetitions")
    aliases = manifest.get("canonical_policy_by_name")
    identities = manifest.get("policy_identity_by_name")
    if (
        not isinstance(required_scenarios, list)
        or not required_scenarios
        or not isinstance(repetitions, int)
        or repetitions < 3
        or set(aliases or {}) != {"P0", "P4", "P5"}
        or set(identities or {}) != {"P0", "P4", "P5"}
    ):
        raise ValueError("invalid policy or case manifest")
    if any(aliases[name] != name for name in ("P0", "P4", "P5")):
        raise ValueError("invalid canonical policy mapping")
    if len(set(identities.values())) != 3:
        raise ValueError("unexpected policy identity collision")
    canonical_policies = [
        name for name in ("P0", "P4", "P5")
        if aliases[name] == name
    ]
    expected = {
        (policy, scenario, repetition)
        for policy in canonical_policies
        for scenario in required_scenarios
        for repetition in range(repetitions)
    }
    by_key = {}
    for row in rows:
        key = (
            row.get("policy"),
            row.get("scenario"),
            row.get("repetition"),
        )
        if key in by_key:
            raise ValueError("duplicate case rows")
        by_key[key] = row
    if set(by_key) != expected:
        raise ValueError("missing or unexpected case rows")
    candidate_results = {}
    for policy in canonical_policies:
        if policy == "P0":
            continue
        paired = []
        for scenario in required_scenarios:
            for repetition in range(repetitions):
                paired.append((
                    by_key[("P0", scenario, repetition)],
                    by_key[(policy, scenario, repetition)],
                ))
        candidate_results[policy] = _candidate_result(policy, paired)
    classification = candidate_results["P5"]["classification"]
    return {
        "classification": classification,
        "structural_failures": [],
        "correctness_failures": [],
        "candidate_results": candidate_results,
    }


def _smoke_summary(rows: list[dict]) -> dict:
    lifecycle_complete = bool(rows) and all(
        row.get("status") == "PASS"
        and row.get("correctness", {}).get(
            "complete_requests"
        ) is True
        and row.get("correctness", {}).get(
            "no_starvation"
        ) is True
        and row.get("correctness", {}).get(
            "valid_lifecycle"
        ) is True
        for row in rows
    )
    exact_outputs = bool(rows) and all(
        row.get("correctness", {}).get("exact_outputs") is True
        for row in rows
    )
    return {
        "classification": (
            "SMOKE_ONLY"
            if lifecycle_complete and exact_outputs
            else "INCOMPLETE"
        ),
        "lifecycle_complete": lifecycle_complete,
        "exact_outputs": exact_outputs,
        "case_count": len(rows),
    }


def _render_report(summary: dict) -> str:
    return (
        "# Production Arrival-Load Gate\n\n"
        f"Classification: `{summary['classification']}`\n"
    )


def _verify_output_equality(
    timeline_rows: list[dict],
    manifest: dict,
) -> None:
    by_case = {}
    for row in timeline_rows:
        key = (
            row.get("policy"),
            row.get("scenario"),
            row.get("repetition"),
        )
        requests = by_case.setdefault(key, {})
        request_id = row.get("request_id")
        if request_id in requests:
            raise ValueError("duplicate request timeline")
        requests[request_id] = row
    for scenario in manifest["required_scenarios"]:
        for repetition in range(manifest["measured_repetitions"]):
            baseline_rows = by_case.get(
                ("P0", scenario, repetition),
                {},
            )
            if not baseline_rows:
                raise ValueError("missing baseline request timeline")
            candidate_policies = (
                [
                    policy
                    for policy in manifest.get(
                        "smoke_policies",
                        [],
                    )
                    if policy != "P0"
                ]
                if manifest.get("run_type") == "smoke"
                else ["P4", "P5"]
            )
            for policy in candidate_policies:
                candidate_rows = by_case.get(
                    (policy, scenario, repetition),
                    {},
                )
                if set(candidate_rows) != set(baseline_rows):
                    raise ValueError("request-set mismatch")
                for request_id in baseline_rows:
                    if (
                        candidate_rows[request_id]["output_token_ids"]
                        != baseline_rows[request_id]["output_token_ids"]
                    ):
                        raise ValueError("output token mismatch")


def verify_run(
    run_dir: Path,
    *,
    write_output: bool = True,
) -> dict:
    run_dir = Path(run_dir)
    _verify_hashes(run_dir)
    manifest = _read_json(run_dir / "run_manifest.json")
    _verify_source(run_dir, manifest)
    _verify_ports(manifest)
    p5_config = _verify_policy_manifest(manifest)
    if manifest.get("run_type") != "smoke":
        _verify_cost_calibration(
            run_dir,
            manifest,
            p5_config,
        )
    _read_jsonl(run_dir / "calibration_manifest.jsonl")
    _read_jsonl(run_dir / "calibration_rows.jsonl")
    workload_rows = _read_jsonl(
        run_dir / "workload_manifest.jsonl"
    )
    if manifest.get("workload_sha256") != _canonical_identity(
        workload_rows
    ):
        raise ValueError("workload identity mismatch")
    workload_by_id = {}
    for row in workload_rows:
        request_id = row.get("request_id")
        if request_id in workload_by_id:
            raise ValueError("duplicate workload request")
        workload_by_id[request_id] = row
    timeline_rows = _read_jsonl(
        run_dir / "request_timeline.jsonl"
    )
    scheduler_rows = _read_jsonl(
        run_dir / "scheduler_trace.jsonl"
    )
    p5_policy_by_case = {}
    for case_id in manifest["expected_case_ids"]:
        case_rows = [
            row for row in scheduler_rows
            if row.get("case_id") == case_id
        ]
        policy = case_rows[0].get("policy") if case_rows else None
        if policy == "P5":
            p5_policy_by_case[case_id] = (
                _verify_p5_scheduler_trace(
                    case_rows,
                    config=p5_config,
                )
            )
        elif policy == "P4":
            p4_config = manifest[
                "resolved_policy_config_by_name"
            ]["P4"]
            _verify_p4_scheduler_trace(
                case_rows,
                enter_waiting=p4_config[
                    "chunked_prefill_adaptive_enter_waiting"
                ],
                exit_waiting=p4_config[
                    "chunked_prefill_adaptive_exit_waiting"
                ],
                transition_steps=p4_config[
                    "chunked_prefill_adaptive_transition_steps"
                ],
                max_mixed_steps=p4_config[
                    "chunked_prefill_adaptive_max_mixed_steps"
                ],
            )
    memory_rows = _read_jsonl(run_dir / "memory_trace.jsonl")
    recorded_case_rows = _read_jsonl(run_dir / "case_rows.jsonl")
    _verify_output_equality(timeline_rows, manifest)
    recomputed_case_rows = [
        _recompute_case(
            case_id,
            timeline_rows,
            scheduler_rows,
            memory_rows,
            workload_by_id,
            p5_policy=p5_policy_by_case.get(case_id),
        )
        for case_id in manifest["expected_case_ids"]
    ]
    if recorded_case_rows != recomputed_case_rows:
        raise ValueError("case row disagreement")
    if manifest.get("run_type") == "smoke":
        computed = _smoke_summary(recomputed_case_rows)
    else:
        computed = _classify(manifest, recomputed_case_rows)
    recorded = _read_json(run_dir / "summary.json")
    if recorded != computed:
        raise ValueError("classification disagreement")
    report = _render_report(computed)
    if (run_dir / "report.md").read_text() != report:
        raise ValueError("report disagreement")
    if write_output:
        output_dir = run_dir / "independent-verify"
        output_dir.mkdir(exist_ok=True)
        _write_json(output_dir / "summary.json", computed)
        (output_dir / "report.md").write_text(report)
        (output_dir / "verify.stdout").write_text(
            json.dumps(computed, sort_keys=True) + "\n"
        )
        (output_dir / "verify.stderr").write_text("")
        (output_dir / "verify.exitcode").write_text("0\n")
    return computed


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    output_dir = args.run_dir / "independent-verify"
    output_dir.mkdir(exist_ok=True)
    try:
        result = verify_run(args.run_dir, write_output=True)
    except Exception as exc:
        (output_dir / "verify.stdout").write_text("")
        (output_dir / "verify.stderr").write_text(f"{exc}\n")
        (output_dir / "verify.exitcode").write_text("1\n")
        raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
