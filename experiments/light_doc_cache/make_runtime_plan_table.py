"""Generate runtime planning/metrics tables for adaptive Light Doc Cache policies."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_PATH = _REPO_ROOT / "tinyvllm" / "engine" / "light_doc_cache_runtime.py"
_SPEC = importlib.util.spec_from_file_location("light_doc_cache_runtime", _RUNTIME_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RUNTIME = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RUNTIME
_SPEC.loader.exec_module(_RUNTIME)

build_config_from_policy_dirs = _RUNTIME.build_config_from_policy_dirs
build_light_doc_cache_runtime_plan = _RUNTIME.build_light_doc_cache_runtime_plan
load_light_doc_cache_policy = _RUNTIME.load_light_doc_cache_policy
summarize_planned_kv_storage = _RUNTIME.summarize_planned_kv_storage
summarize_planned_kv_storage_from_shape = _RUNTIME.summarize_planned_kv_storage_from_shape


def _parse_task(value: str) -> tuple[str | None, str]:
    if ":" not in value:
        return None, value
    doc_id, task_id = value.split(":", 1)
    if not task_id:
        raise ValueError(f"invalid task spec: {value!r}")
    return doc_id or None, task_id


def _load_task_file(value: str) -> list[tuple[str | None, str]]:
    if "=" in value:
        doc_id, path = value.split("=", 1)
        doc_id = doc_id or None
    else:
        doc_id, path = None, value
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = payload.get("tasks", payload) if isinstance(payload, dict) else payload
    if not isinstance(tasks, list):
        raise ValueError(f"task file must contain a list or tasks list: {path}")
    parsed = []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError(f"task entries must be objects: {path}")
        task_id = task.get("id") or task.get("task_id")
        if not task_id:
            raise ValueError(f"task entry missing id/task_id: {path}")
        parsed.append((doc_id, str(task_id)))
    return parsed


def _parse_shape_csv(value: str) -> list[int]:
    shape = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(shape) != 6:
        raise ValueError("--kv-cache-shape must contain 6 comma-separated integers")
    return shape


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-file", required=True)
    parser.add_argument("--repo-root", default=str(_REPO_ROOT))
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--num-blocks", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=0)
    parser.add_argument("--head-dim", type=int, default=0)
    parser.add_argument("--element-size-bytes", type=int, default=0)
    parser.add_argument("--kv-cache-shape", default="", help="Optional real kv_cache shape: 2,L,B,block,H,D.")
    parser.add_argument("--task", action="append", default=[], help="Task spec as doc_id:task_id or task_id.")
    parser.add_argument("--task-file", action="append", default=[], help="Task file spec as doc_id=path or path.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    policy = load_light_doc_cache_policy(args.policy_file)
    config = build_config_from_policy_dirs(
        policy,
        repo_root=args.repo_root,
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        enabled=True,
    )
    task_specs = [_parse_task(task_spec) for task_spec in args.task]
    for task_file in args.task_file:
        task_specs.extend(_load_task_file(task_file))
    if not task_specs:
        raise ValueError("provide at least one --task or --task-file")
    include_shape_storage = bool(args.kv_cache_shape)
    include_storage = include_shape_storage or any(
        value > 0 for value in (args.num_blocks, args.block_size, args.head_dim, args.element_size_bytes)
    )
    if include_shape_storage and args.element_size_bytes <= 0:
        raise ValueError("--kv-cache-shape requires --element-size-bytes")
    if include_storage and not include_shape_storage and not all(
        value > 0 for value in (args.num_blocks, args.block_size, args.head_dim, args.element_size_bytes)
    ):
        raise ValueError("storage summary requires all of --num-blocks/--block-size/--head-dim/--element-size-bytes")
    kv_cache_shape = _parse_shape_csv(args.kv_cache_shape) if include_shape_storage else None

    rows = []
    for doc_id, task_id in task_specs:
        plan = build_light_doc_cache_runtime_plan(
            config,
            task_id=task_id,
            doc_id=doc_id,
            seq_len=args.seq_len,
        )
        row = plan.as_summary()
        if include_storage:
            if kv_cache_shape is not None:
                row["storage"] = summarize_planned_kv_storage_from_shape(
                    plan,
                    full_cache_shape=kv_cache_shape,
                    element_size_bytes=args.element_size_bytes,
                )
            else:
                row["storage"] = summarize_planned_kv_storage(
                    plan,
                    num_blocks=args.num_blocks,
                    block_size=args.block_size,
                    head_dim=args.head_dim,
                    element_size_bytes=args.element_size_bytes,
                )
        rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "runtime_plan_table.jsonl"
    md_path = output_dir / "runtime_plan_table.md"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    lines = [
        "# Light Doc Cache Runtime Plan Table",
        "",
        "Boundary: planning/metrics only; this is not a runtime KV-cache compression or latency result.",
        "",
        "| Doc | Task | Fallback | Recovered Heads | Recovered KV-Head Eq | Effective Saving | Compression Ratio |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.get('doc_id') or ''} | "
            f"{row['task_id']} | "
            f"{row.get('fallback_reason') or ''} | "
            f"{row['recovered_kv_heads']} | "
            f"{row['recovered_kv_head_equivalent']:.2f} | "
            f"{row['effective_saving_fraction']:.2%} | "
            f"{row['compression_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Tasks: `{len(rows)}`",
            f"- Average Effective Saving: `{_mean([row['effective_saving_fraction'] for row in rows]):.2%}`",
            f"- Average Compression Ratio: `{_mean([row['compression_ratio'] for row in rows]):.4f}`",
        ]
    )
    if include_storage:
        lines.extend(
            [
                f"- Full KV Bytes: `{rows[0]['storage']['full_kv_bytes']:,}`",
                "- Average Planned Recovered KV Bytes: "
                f"`{int(round(_mean([row['storage']['planned_recovered_kv_bytes'] for row in rows]))):,}`",
                "- Average Planned Stored KV Bytes: "
                f"`{int(round(_mean([row['storage']['planned_stored_kv_bytes'] for row in rows]))):,}`",
            ]
        )
    lines.extend(
        [
            "",
            "Use this table to audit planned storage/recovery accounting before any ModelRunner hot-path integration.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
