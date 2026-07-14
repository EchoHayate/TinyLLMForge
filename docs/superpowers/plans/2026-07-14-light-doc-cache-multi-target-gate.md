# Light Doc Cache Multi-Target Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an eight-target, fixed-calibration Light Doc Cache read-path gate that compares `calibration_holdout` with inexpensive baselines and produces a reproducible `GO` or `NO_GO` decision.

**Architecture:** Add a versioned prompt dataset, a CPU-only report/decision module, and a one-model-instance TinyLLM batch driver. Refactor the existing calibrated KV script only enough to expose fixed-bank fitting, reuse the existing restored-sidecar read-path function for every target, and keep all runtime mutation default-off and temporary.

**Tech Stack:** Python 3.11, standard-library `argparse`/`csv`/`hashlib`/`json`/`math`/`pathlib`/`statistics`, PyTorch/TinyLLM on the remote GPU host, existing plain-assert/pytest-compatible tool tests, Bash/SSH/rsync for remote execution.

## Global Constraints

- The workflow remains default-off and must not modify attention kernels, KV allocation lifetime, slot mapping, CUDA Graph behavior, or the serving scheduler.
- Use exactly eight versioned targets covering short factual prose, long document QA, source code, mathematical reasoning, structured text, repetitive text, cross-paragraph dependency, and out-of-distribution prose.
- Cover actual tokenizer length buckets `short=16-48`, `medium=49-160`, and `long=161-384`, with at least two targets in each bucket.
- Fit or load one immutable `calibration_holdout` bank before target evaluation; target prompts must not affect source selection, fitting, thresholds, or budgets.
- Compare `repeat_last_target`, `correlated_same_layer_target`, and `calibration_holdout` for every target.
- Continue after target-level failures, but any missing required paired row makes the final decision `NO_GO`.
- Logical byte savings are accounting evidence only; do not claim observed GPU-memory reduction, serving throughput, or task-level quality.
- Use Qwen3-0.6B on `sitian@10.232.195.203` with an available GPU and dynamic `TINYVLLM_DIST_PORT`/`MASTER_PORT`.
- Preserve all existing uncommitted Light Doc Cache and `ModelRunner` work; stage and commit only files owned by the task being completed.

---

## File Structure

- Create `experiments/light_doc_cache/read_path_multi_target_prompts_v1.json`
  - Versioned eight-target evaluation dataset.
- Create `experiments/light_doc_cache/make_multi_target_read_path_report.py`
  - Dataset validation, artifact normalization, aggregate statistics, gate decision, and CSV/JSON/Markdown writers.
- Create `experiments/light_doc_cache/run_tinyllm_read_path_multi_target.py`
  - One-model-instance calibration and target orchestration.
- Create `experiments/light_doc_cache/run_tinyllm_read_path_multi_target_remote.sh`
  - Remote synchronization, GPU/port selection, smoke/full execution, and artifact mirroring.
- Create `tools/test_light_doc_cache_multi_target.py`
  - CPU-only tests for dataset validation, aggregation, gate boundaries, orchestration failures, and output determinism.
- Modify `experiments/light_doc_cache/run_tinyllm_calibrated_kv_smoke.py`
  - Extract reusable fixed-bank fitting without changing existing CLI behavior.
- Modify `experiments/light_doc_cache/README.md`
  - Record commands, output schema, gate result, and claim boundaries.
- Modify `AGENT_HANDOFF_STATE.md`
  - Record validated commands, artifacts, decision, and next branch.

---

### Task 1: Versioned Target Dataset and Validation

**Files:**
- Create: `experiments/light_doc_cache/read_path_multi_target_prompts_v1.json`
- Create: `experiments/light_doc_cache/make_multi_target_read_path_report.py`
- Create: `tools/test_light_doc_cache_multi_target.py`

**Interfaces:**
- Produces: `load_target_dataset(path: str | Path) -> dict[str, object]`
- Produces: `validate_target_dataset(payload: dict[str, object]) -> list[dict[str, str]]`
- Later tasks consume validated target dictionaries with keys `id`, `category`, `length_bucket`, and `prompt`.

- [ ] **Step 1: Write failing dataset-validation tests**

Add the initial test module:

```python
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "experiments" / "light_doc_cache" / "make_multi_target_read_path_report.py"
SPEC = importlib.util.spec_from_file_location("make_multi_target_read_path_report", REPORT_PATH)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


def _target(target_id: str, category: str, bucket: str) -> dict[str, str]:
    return {
        "id": target_id,
        "category": category,
        "length_bucket": bucket,
        "prompt": f"Prompt for {target_id} with enough non-empty text.",
    }


def _valid_payload() -> dict[str, object]:
    return {
        "version": 1,
        "targets": [
            _target("short_fact", "short_factual", "short"),
            _target("structured", "structured_text", "short"),
            _target("code", "source_code", "medium"),
            _target("math", "mathematical_reasoning", "medium"),
            _target("ood", "out_of_distribution", "medium"),
            _target("document_qa", "long_document_qa", "long"),
            _target("repetitive", "repetitive_text", "long"),
            _target("cross_paragraph", "cross_paragraph_dependency", "long"),
        ],
    }


def test_validate_target_dataset_accepts_required_matrix() -> None:
    targets = REPORT.validate_target_dataset(_valid_payload())
    assert [target["id"] for target in targets] == [
        "short_fact",
        "structured",
        "code",
        "math",
        "ood",
        "document_qa",
        "repetitive",
        "cross_paragraph",
    ]


def test_validate_target_dataset_rejects_duplicate_ids() -> None:
    payload = _valid_payload()
    payload["targets"][1]["id"] = "short_fact"
    try:
        REPORT.validate_target_dataset(payload)
    except ValueError as exc:
        assert "duplicate target id" in str(exc)
    else:
        raise AssertionError("duplicate target IDs must fail")


def test_validate_target_dataset_requires_bucket_coverage() -> None:
    payload = _valid_payload()
    for target in payload["targets"]:
        if target["length_bucket"] == "long":
            target["length_bucket"] = "medium"
    try:
        REPORT.validate_target_dataset(payload)
    except ValueError as exc:
        assert "at least two targets in each length bucket" in str(exc)
    else:
        raise AssertionError("missing long targets must fail")


def test_repository_target_dataset_is_valid() -> None:
    payload = REPORT.load_target_dataset(
        ROOT / "experiments" / "light_doc_cache" / "read_path_multi_target_prompts_v1.json"
    )
    assert payload["version"] == 1
    assert len(payload["targets"]) == 8
```

- [ ] **Step 2: Run the tests and verify the expected failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
  python3 -m pytest -q tools/test_light_doc_cache_multi_target.py
```

Expected: import or attribute failure because
`make_multi_target_read_path_report.py` and its validation functions do not
exist.

- [ ] **Step 3: Add the exact eight-target dataset**

Create `read_path_multi_target_prompts_v1.json` with this shape and content:

```json
{
  "version": 1,
  "targets": [
    {
      "id": "short_fact",
      "category": "short_factual",
      "length_bucket": "short",
      "prompt": "The Pacific Ocean is larger than the Atlantic Ocean, and Mount Everest is the highest mountain above sea level. State which ocean is larger and name the mountain."
    },
    {
      "id": "structured",
      "category": "structured_text",
      "length_bucket": "short",
      "prompt": "Read this JSON record and state the service name and retry limit: {\"service\":\"tinyllm\",\"enabled\":true,\"retry_limit\":3,\"owners\":[\"runtime\",\"infra\"]}."
    },
    {
      "id": "code",
      "category": "source_code",
      "length_bucket": "medium",
      "prompt": "Explain what this Python function returns and identify its time complexity.\\n\\ndef stable_unique(values):\\n    seen = set()\\n    output = []\\n    for value in values:\\n        if value not in seen:\\n            seen.add(value)\\n            output.append(value)\\n    return output\\n\\nUse the input [3, 1, 3, 2, 1, 4] in your explanation."
    },
    {
      "id": "math",
      "category": "mathematical_reasoning",
      "length_bucket": "medium",
      "prompt": "A cache contains 240 blocks. Forty percent are reserved for active requests. Of the remaining blocks, one third are moved to CPU storage. How many blocks remain immediately available on the GPU? Show the sequence of arithmetic operations before giving the result."
    },
    {
      "id": "ood",
      "category": "out_of_distribution",
      "length_bucket": "medium",
      "prompt": "A conservator describes a seventeenth-century blue pigment made from ground mineral, while a botanist describes nocturnal pollination in desert flowers and a musician explains a suspended fourth chord. Summarize the three unrelated subjects without inventing a connection among them."
    },
    {
      "id": "document_qa",
      "category": "long_document_qa",
      "length_bucket": "long",
      "prompt": "A runtime team tested three configurations. Configuration Amber used full attention and achieved the best recall, but its long-context decode latency was highest. Configuration Birch quantized the KV cache to eight bits, reducing KV storage by half while keeping recall close to Amber. Configuration Cedar combined eight-bit KV with sparse block selection; it improved decode throughput but lost two retrieval cases at the earliest document depth. The team decided that Birch should be the conservative default and Cedar should remain an opt-in throughput profile until the two failures are understood. They also noted that weight offload was inappropriate for the small model because transfer overhead dominated. Based only on this document, which configuration became the conservative default, why was Cedar not selected as the default, and what did the team conclude about weight offload?"
    },
    {
      "id": "repetitive",
      "category": "repetitive_text",
      "length_bucket": "long",
      "prompt": "alpha beta gamma delta repeats to test prefix structure. alpha beta gamma delta repeats to test prefix structure. alpha beta gamma delta repeats to test prefix structure. alpha beta gamma delta repeats to test prefix structure. alpha beta gamma delta repeats to test prefix structure. alpha beta gamma delta repeats to test prefix structure. alpha beta gamma delta repeats to test prefix structure. alpha beta gamma delta repeats to test prefix structure. After the repeated sequence, report the four repeated Greek-letter words in order and state how many complete repetitions appear."
    },
    {
      "id": "cross_paragraph",
      "category": "cross_paragraph_dependency",
      "length_bucket": "long",
      "prompt": "Paragraph one: Project Orion stores hot KV blocks on GPU and cold blocks on CPU. Its migration manager labels every transferred block with a monotonically increasing generation number.\\n\\nParagraph two: Project Lyra never moves KV blocks, but it compresses values to eight bits and keeps all block identifiers unchanged.\\n\\nParagraph three: During a failure investigation, engineers found a stale asynchronous copy completing after a GPU slot had already been reassigned. The fix compared the pending copy generation with the slot's current generation and discarded mismatches.\\n\\nUsing information across the paragraphs, identify which project required the generation-number fix, explain why Lyra would not encounter that particular migration race, and name the state comparison used to reject the stale completion."
    }
  ]
}
```

- [ ] **Step 4: Implement minimal validation**

Create `make_multi_target_read_path_report.py` with:

```python
"""Aggregate Light Doc Cache multi-target read-path artifacts."""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_CATEGORIES = {
    "short_factual",
    "long_document_qa",
    "source_code",
    "mathematical_reasoning",
    "structured_text",
    "repetitive_text",
    "cross_paragraph_dependency",
    "out_of_distribution",
}
LENGTH_BUCKETS = {"short", "medium", "long"}


def load_target_dataset(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    targets = validate_target_dataset(payload)
    return {"version": int(payload["version"]), "targets": targets}


def validate_target_dataset(payload: dict[str, object]) -> list[dict[str, str]]:
    if int(payload.get("version", 0)) != 1:
        raise ValueError("target dataset version must be 1")
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or len(raw_targets) != 8:
        raise ValueError("target dataset must contain exactly eight targets")
    targets: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    category_counts = {category: 0 for category in REQUIRED_CATEGORIES}
    bucket_counts = {bucket: 0 for bucket in LENGTH_BUCKETS}
    for index, raw in enumerate(raw_targets):
        if not isinstance(raw, dict):
            raise ValueError(f"target {index} must be an object")
        target = {
            "id": str(raw.get("id", "")).strip(),
            "category": str(raw.get("category", "")).strip(),
            "length_bucket": str(raw.get("length_bucket", "")).strip(),
            "prompt": str(raw.get("prompt", "")).strip(),
        }
        if not target["id"] or not target["prompt"]:
            raise ValueError(f"target {index} requires non-empty id and prompt")
        if target["id"] in seen_ids:
            raise ValueError(f"duplicate target id: {target['id']}")
        if target["category"] not in REQUIRED_CATEGORIES:
            raise ValueError(f"unknown target category: {target['category']}")
        if target["length_bucket"] not in LENGTH_BUCKETS:
            raise ValueError(f"unknown length bucket: {target['length_bucket']}")
        seen_ids.add(target["id"])
        category_counts[target["category"]] += 1
        bucket_counts[target["length_bucket"]] += 1
        targets.append(target)
    if set(category for category, count in category_counts.items() if count) != REQUIRED_CATEGORIES:
        raise ValueError("target dataset must cover every required category exactly once")
    if any(count < 2 for count in bucket_counts.values()):
        raise ValueError("target dataset requires at least two targets in each length bucket")
    return targets
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
  python3 -m pytest -q tools/test_light_doc_cache_multi_target.py
```

Expected: four tests pass.

- [ ] **Step 6: Commit the dataset slice**

```bash
git add \
  experiments/light_doc_cache/read_path_multi_target_prompts_v1.json \
  experiments/light_doc_cache/make_multi_target_read_path_report.py \
  tools/test_light_doc_cache_multi_target.py
git commit -m "Add Light Doc Cache target matrix"
```

---

### Task 2: Deterministic Aggregation and Go/No-Go Gate

**Files:**
- Modify: `experiments/light_doc_cache/make_multi_target_read_path_report.py`
- Modify: `tools/test_light_doc_cache_multi_target.py`

**Interfaces:**
- Consumes: validated targets from `load_target_dataset`.
- Produces: `nearest_rank_percentile(values: list[float], percentile: float) -> float`
- Produces: `normalize_summary_row(...) -> dict[str, object]`
- Produces: `aggregate_rows(rows: list[dict[str, object]]) -> dict[str, object]`
- Produces: `evaluate_gate(rows: list[dict[str, object]]) -> dict[str, object]`
- Produces: `write_outputs(output_dir: str | Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None`

- [ ] **Step 1: Add failing aggregation and gate tests**

Append:

```python
def _row(
    target_index: int,
    mode: str,
    mean_diff: float,
    *,
    argmax_match: bool = True,
    status: str = "success",
) -> dict[str, object]:
    return {
        "target_id": f"target_{target_index}",
        "category": "short_factual",
        "length_bucket": "short",
        "mode": mode,
        "role": "trained" if mode == "calibration_holdout" else "baseline",
        "status": status,
        "error": "",
        "prompt_tokens": 20 + target_index,
        "calibration_bank_sha256": "a" * 64 if mode == "calibration_holdout" else "",
        "logical_byte_saving_fraction": 0.1763,
        "missing_tokens": 100 + target_index,
        "missing_mse": 10.0 + target_index,
        "missing_mae": 2.0,
        "missing_max_abs": 20.0,
        "max_abs_logit_diff": mean_diff * 4,
        "mean_abs_logit_diff": mean_diff,
        "argmax_match": argmax_match,
        "original_argmax": 100,
        "restored_argmax": 100 if argmax_match else 101,
        "artifact": f"targets/target_{target_index}/{mode}",
    }


def test_nearest_rank_percentile_is_deterministic() -> None:
    assert REPORT.nearest_rank_percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.90) == 5.0
    assert REPORT.nearest_rank_percentile([5.0, 1.0, 3.0, 2.0], 0.50) == 2.0


def test_gate_passes_only_when_every_condition_holds() -> None:
    rows = []
    correlated = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
    holdout = [0.80, 0.85, 0.90, 0.90, 0.95, 1.00, 1.05, 1.10]
    for index in range(8):
        rows.append(_row(index, "correlated_same_layer_target", correlated[index]))
        rows.append(_row(index, "calibration_holdout", holdout[index]))
        rows.append(_row(index, "repeat_last_target", 1.20))
    gate = REPORT.evaluate_gate(rows)
    assert gate["decision"] == "GO"
    assert gate["paired_targets"] == 8
    assert gate["holdout_win_count"] == 5
    assert gate["holdout_win_rate"] == 0.625


def test_gate_fails_on_missing_pair_or_argmax_regression() -> None:
    rows = []
    for index in range(8):
        rows.append(_row(index, "correlated_same_layer_target", 1.0))
        if index != 7:
            rows.append(
                _row(
                    index,
                    "calibration_holdout",
                    0.8,
                    argmax_match=index != 6,
                )
            )
    gate = REPORT.evaluate_gate(rows)
    assert gate["decision"] == "NO_GO"
    assert "all eight paired targets completed" in gate["failed_conditions"]
    assert "no correlated argmax match regressed" in gate["failed_conditions"]


def test_write_outputs_keeps_per_target_setup_fields(tmp_path: Path) -> None:
    rows = [
        _row(0, "correlated_same_layer_target", 1.0),
        _row(1, "correlated_same_layer_target", 0.8),
    ]
    summary = REPORT.aggregate_rows(rows)
    REPORT.write_outputs(tmp_path, rows, summary)
    csv_text = (tmp_path / "multi_target_rows.csv").read_text(encoding="utf-8")
    assert "target_0" in csv_text
    assert "target_1" in csv_text
    assert ",20," in csv_text
    assert ",21," in csv_text
    assert (tmp_path / "multi_target_summary.json").exists()
    assert "# Light Doc Cache Multi-Target Gate" in (
        tmp_path / "multi_target_report.md"
    ).read_text(encoding="utf-8")
```

- [ ] **Step 2: Run tests and verify failure**

Run the same pytest command.

Expected: failures for undefined percentile, aggregation, gate, and writer
functions.

- [ ] **Step 3: Implement statistics and gate logic**

Add these constants and functions:

```python
import csv
import math
import statistics
from typing import Any

REQUIRED_MODES = (
    "repeat_last_target",
    "correlated_same_layer_target",
    "calibration_holdout",
)
ROW_FIELDS = (
    "target_id",
    "category",
    "length_bucket",
    "mode",
    "role",
    "status",
    "error",
    "prompt_tokens",
    "calibration_bank_sha256",
    "logical_byte_saving_fraction",
    "missing_tokens",
    "missing_mse",
    "missing_mae",
    "missing_max_abs",
    "max_abs_logit_diff",
    "mean_abs_logit_diff",
    "argmax_match",
    "original_argmax",
    "restored_argmax",
    "artifact",
)


def nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _metric_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": nearest_rank_percentile(values, 0.90),
        "worst": max(values),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    modes: dict[str, dict[str, Any]] = {}
    for mode in REQUIRED_MODES:
        mode_rows = [row for row in rows if row["mode"] == mode]
        successful = [row for row in mode_rows if row["status"] == "success"]
        modes[mode] = {
            "attempted_targets": len(mode_rows),
            "completed_targets": len(successful),
            "failed_targets": len(mode_rows) - len(successful),
            "argmax_match_count": sum(bool(row["argmax_match"]) for row in successful),
            "argmax_match_rate": (
                sum(bool(row["argmax_match"]) for row in successful) / len(successful)
                if successful
                else 0.0
            ),
            "mean_abs_logit_diff": _metric_summary(
                [float(row["mean_abs_logit_diff"]) for row in successful]
            ),
            "max_abs_logit_diff": _metric_summary(
                [float(row["max_abs_logit_diff"]) for row in successful]
            ),
            "missing_mse": _metric_summary(
                [float(row["missing_mse"]) for row in successful]
            ),
            "mean_logical_byte_saving_fraction": (
                statistics.fmean(
                    float(row["logical_byte_saving_fraction"]) for row in successful
                )
                if successful
                else 0.0
            ),
        }
    gate = evaluate_gate(rows)
    return {
        "claim_boundary": "default_off_multi_target_read_path_gate",
        "row_count": len(rows),
        "modes": modes,
        "gate": gate,
    }


def evaluate_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {
        (str(row["target_id"]), str(row["mode"])): row
        for row in rows
        if row["status"] == "success"
    }
    target_ids = sorted(
        {
            str(row["target_id"])
            for row in rows
            if row["mode"] in {
                "correlated_same_layer_target",
                "calibration_holdout",
            }
        }
    )
    pairs = []
    for target_id in target_ids:
        correlated = by_key.get((target_id, "correlated_same_layer_target"))
        holdout = by_key.get((target_id, "calibration_holdout"))
        if correlated is not None and holdout is not None:
            pairs.append((target_id, correlated, holdout))
    correlated_mean = (
        statistics.fmean(float(correlated["mean_abs_logit_diff"]) for _, correlated, _ in pairs)
        if pairs
        else math.inf
    )
    holdout_mean = (
        statistics.fmean(float(holdout["mean_abs_logit_diff"]) for _, _, holdout in pairs)
        if pairs
        else math.inf
    )
    improvement = (
        (correlated_mean - holdout_mean) / correlated_mean
        if pairs and correlated_mean > 0.0
        else 0.0
    )
    relative_changes = [
        (
            float(holdout["mean_abs_logit_diff"])
            - float(correlated["mean_abs_logit_diff"])
        )
        / float(correlated["mean_abs_logit_diff"])
        for _, correlated, holdout in pairs
        if float(correlated["mean_abs_logit_diff"]) > 0.0
    ]
    holdout_wins = sum(
        float(holdout["mean_abs_logit_diff"])
        < float(correlated["mean_abs_logit_diff"])
        for _, correlated, holdout in pairs
    )
    correlated_argmax = sum(bool(correlated["argmax_match"]) for _, correlated, _ in pairs)
    holdout_argmax = sum(bool(holdout["argmax_match"]) for _, _, holdout in pairs)
    argmax_regressions = [
        target_id
        for target_id, correlated, holdout in pairs
        if bool(correlated["argmax_match"]) and not bool(holdout["argmax_match"])
    ]
    conditions = {
        "all eight paired targets completed": len(pairs) == 8,
        "holdout argmax rate not lower": holdout_argmax >= correlated_argmax,
        "holdout wins at least five targets": holdout_wins >= 5,
        "mean logit diff improves at least five percent": improvement >= 0.05,
        "worst relative regression no more than twenty five percent": (
            bool(relative_changes) and max(relative_changes) <= 0.25
        ),
        "no correlated argmax match regressed": not argmax_regressions,
    }
    return {
        "decision": "GO" if all(conditions.values()) else "NO_GO",
        "conditions": conditions,
        "failed_conditions": [name for name, passed in conditions.items() if not passed],
        "paired_targets": len(pairs),
        "holdout_win_count": holdout_wins,
        "holdout_win_rate": holdout_wins / len(pairs) if pairs else 0.0,
        "correlated_mean_abs_logit_diff": correlated_mean,
        "holdout_mean_abs_logit_diff": holdout_mean,
        "aggregate_relative_improvement": improvement,
        "worst_relative_regression": max(relative_changes) if relative_changes else None,
        "argmax_regressions": argmax_regressions,
    }
```

- [ ] **Step 4: Implement deterministic output writers and CLI**

Add:

```python
def write_outputs(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_rows = sorted(rows, key=lambda row: (str(row["target_id"]), str(row["mode"])))
    with (output_dir / "multi_target_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in ROW_FIELDS} for row in ordered_rows)
    (output_dir / "multi_target_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    gate = summary["gate"]
    lines = [
        "# Light Doc Cache Multi-Target Gate",
        "",
        "Boundary: default-off restored-sidecar next-token comparison; no physical KV allocation or attention hot-path change.",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Paired targets: `{gate['paired_targets']}/8`",
        f"- Holdout wins: `{gate['holdout_win_count']}`",
        f"- Aggregate relative improvement: `{gate['aggregate_relative_improvement']:.2%}`",
        f"- Worst relative regression: `{gate['worst_relative_regression']}`",
        "",
        "## Conditions",
        "",
    ]
    for name, passed in gate["conditions"].items():
        lines.append(f"- [{'x' if passed else ' '}] {name}")
    lines.extend(
        [
            "",
            "## Per-Target Rows",
            "",
            "| Target | Mode | Status | Tokens | Mean Logit Diff | Argmax Match |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for row in ordered_rows:
        mean_diff = (
            f"{float(row['mean_abs_logit_diff']):.6g}"
            if row["status"] == "success"
            else "-"
        )
        lines.append(
            f"| `{row['target_id']}` | `{row['mode']}` | {row['status']} | "
            f"{row.get('prompt_tokens', '-')} | {mean_diff} | {row.get('argmax_match', '-')} |"
        )
    (output_dir / "multi_target_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
```

Add an `argparse` CLI accepting repeated `--summary target_id:category:bucket:mode:path`,
`--output-dir`, loading each existing TinyLLM summary through a
`normalize_summary_row(...)` helper, and calling `aggregate_rows` plus
`write_outputs`.

- [ ] **Step 5: Run tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
  python3 -m pytest -q tools/test_light_doc_cache_multi_target.py
```

Expected: all dataset and aggregation tests pass.

- [ ] **Step 6: Commit the reporting slice**

```bash
git add \
  experiments/light_doc_cache/make_multi_target_read_path_report.py \
  tools/test_light_doc_cache_multi_target.py
git commit -m "Add Light Doc Cache multi-target gate"
```

---

### Task 3: Extract Fixed Calibration-Bank Fitting

**Files:**
- Modify: `experiments/light_doc_cache/run_tinyllm_calibrated_kv_smoke.py`
- Modify: `tools/test_light_doc_cache_recovery_probe.py`

**Interfaces:**
- Produces: `fit_calibration_recovery_bank(...) -> tuple[object, object, dict[tuple[int, int], list[tuple[int, int]]]]`
- Existing `run_calibrated_smoke(...)` continues to produce identical CLI artifacts and delegates fitting to the new helper.
- The batch driver consumes the fitted bank and saves it with `_RUNTIME.save_multi_source_recovery_bank`.

- [ ] **Step 1: Add a failing delegation/static-contract test**

Append to `tools/test_light_doc_cache_recovery_probe.py`:

```python
def test_calibrated_kv_smoke_exposes_fixed_bank_fit_boundary() -> None:
    script = LIGHT_DOC_CACHE / "run_tinyllm_calibrated_kv_smoke.py"
    text = script.read_text(encoding="utf-8")
    for needle in [
        "def fit_calibration_recovery_bank(",
        "bank, calibration_plan, source_heads = fit_calibration_recovery_bank(",
        "source_map=source_map",
        "source_count=int(source_count)",
    ]:
        assert needle in text
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
  python3 -m pytest -q \
  tools/test_light_doc_cache_recovery_probe.py::test_calibrated_kv_smoke_exposes_fixed_bank_fit_boundary
```

Expected: assertion failure because the function does not exist.

- [ ] **Step 3: Extract the fitting helper**

Add:

```python
def fit_calibration_recovery_bank(
    *,
    calibration_kv,
    calibration_tokens: int,
    policy_file: str,
    repo_root: str | Path,
    task_id: str,
    doc_id: str | None,
    source_count: int,
    source_map: str,
    recover_ridge: float,
):
    policy = _RUNTIME.load_light_doc_cache_policy(policy_file)
    config = _RUNTIME.build_config_from_policy_dirs(
        policy,
        repo_root=repo_root,
        num_layers=int(calibration_kv.shape[1]),
        num_kv_heads=int(calibration_kv.shape[4]),
        enabled=True,
    )
    calibration_plan = _RUNTIME.build_light_doc_cache_runtime_plan(
        config,
        task_id=task_id,
        doc_id=doc_id,
        seq_len=int(calibration_tokens),
    )
    if source_map == "calibration_fit":
        source_heads = build_calibration_fit_source_heads(
            calibration_kv,
            calibration_plan,
            source_count=int(source_count),
            ridge=recover_ridge,
        )
    elif source_map == "calibration_holdout":
        source_heads = build_calibration_holdout_source_heads(
            calibration_kv,
            calibration_plan,
            source_count=int(source_count),
            ridge=recover_ridge,
        )
    elif source_map == "same_layer":
        source_heads = _build_multi_source_heads(
            calibration_plan,
            int(source_count),
        )
    else:
        raise ValueError(f"unsupported source_map: {source_map}")
    bank = _RUNTIME.fit_multi_source_recovery_bank(
        calibration_kv,
        calibration_plan,
        source_heads=source_heads,
        ridge=recover_ridge,
    )
    return bank, calibration_plan, source_heads
```

Change `run_calibrated_smoke(...)` to build the target config/plan as before,
then call:

```python
bank, calibration_plan, source_heads = fit_calibration_recovery_bank(
    calibration_kv=calibration_kv,
    calibration_tokens=calibration_tokens,
    policy_file=policy_file,
    repo_root=repo_root,
    task_id=task_id,
    doc_id=doc_id,
    source_count=int(source_count),
    source_map=source_map,
    recover_ridge=recover_ridge,
)
```

Remove the duplicated selector/fitting branch from `run_calibrated_smoke`.
Do not change its parameters, summary keys, filenames, or CLI output.

- [ ] **Step 4: Run focused and existing recovery tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
  python3 -m pytest -q \
  tools/test_light_doc_cache_recovery_probe.py \
  tools/test_light_doc_cache_runtime.py
```

Expected: all existing tests pass, including the new contract test.

- [ ] **Step 5: Run syntax validation**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  experiments/light_doc_cache/run_tinyllm_calibrated_kv_smoke.py
```

Expected: exit code 0.

- [ ] **Step 6: Commit the refactor**

```bash
git add \
  experiments/light_doc_cache/run_tinyllm_calibrated_kv_smoke.py \
  tools/test_light_doc_cache_recovery_probe.py
git commit -m "Extract fixed Light Doc Cache calibration bank"
```

---

### Task 4: One-Model Multi-Target Driver

**Files:**
- Create: `experiments/light_doc_cache/run_tinyllm_read_path_multi_target.py`
- Modify: `tools/test_light_doc_cache_multi_target.py`

**Interfaces:**
- Consumes: `load_target_dataset`, `aggregate_rows`, and `write_outputs`.
- Consumes: `fit_calibration_recovery_bank`, `stack_calibration_kv_samples`, and `copy_kv_prompt_prefix`.
- Consumes: `run_read_path_smoke(...)`.
- Produces: `run_target_matrix(...) -> tuple[list[dict[str, object]], dict[str, object]]`
- Produces: `sha256_file(path: str | Path) -> str`
- Produces the manifest and directory layout defined in the design spec.

- [ ] **Step 1: Add failing CPU-only orchestration tests**

Append:

```python
def test_run_target_matrix_attempts_all_modes_and_records_failures(tmp_path: Path) -> None:
    driver_path = (
        ROOT
        / "experiments"
        / "light_doc_cache"
        / "run_tinyllm_read_path_multi_target.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_tinyllm_read_path_multi_target",
        driver_path,
    )
    assert spec is not None and spec.loader is not None
    driver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = driver
    spec.loader.exec_module(driver)
    targets = _valid_payload()["targets"][:2]
    calls = []

    def fake_run_mode(*, target, mode, output_dir):
        calls.append((target["id"], mode))
        if target["id"] == "structured" and mode == "calibration_holdout":
            raise RuntimeError("synthetic target failure")
        return _row(
            0 if target["id"] == "short_fact" else 1,
            mode,
            0.5,
        )

    rows, manifest = driver.run_target_matrix(
        targets=targets,
        output_dir=tmp_path,
        calibration_bank_sha256="b" * 64,
        run_mode=fake_run_mode,
    )
    assert len(calls) == 6
    assert len(rows) == 6
    failed = [
        row
        for row in rows
        if row["target_id"] == "structured"
        and row["mode"] == "calibration_holdout"
    ][0]
    assert failed["status"] == "failed"
    assert failed["error"] == "RuntimeError: synthetic target failure"
    assert manifest["calibration_bank_sha256"] == "b" * 64


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "bank.json"
    path.write_text('{"kind":"test"}\\n', encoding="utf-8")
    assert len(REPORT.hashlib_sha256_file(path)) == 64
```

Use the report module to expose the shared `hashlib_sha256_file` helper, so the
driver and tests use one implementation.

- [ ] **Step 2: Run tests and verify failure**

Run the multi-target test file.

Expected: import failure because the driver does not exist and hash helper is
undefined.

- [ ] **Step 3: Implement the CPU-testable orchestration boundary**

Create the driver with imports and constants:

```python
"""Run a fixed-bank Light Doc Cache read-path matrix in one TinyLLM process."""

from __future__ import annotations

import argparse
import atexit
import importlib.util
import json
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
LIGHT_DOC_CACHE = ROOT / "experiments" / "light_doc_cache"

MODE_CONFIGS = {
    "repeat_last_target": {
        "role": "baseline",
        "recover_mode": "repeat_last",
    },
    "correlated_same_layer_target": {
        "role": "baseline",
        "recover_mode": "correlated",
        "correlated_source_map": "same_layer",
    },
    "calibration_holdout": {
        "role": "trained",
        "recover_mode": "calibrated_multi_correlated",
    },
}
```

Implement:

```python
def run_target_matrix(
    *,
    targets: list[dict[str, str]],
    output_dir: str | Path,
    calibration_bank_sha256: str,
    run_mode: Callable[..., dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    output_dir = Path(output_dir)
    rows: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    for target in targets:
        for mode in MODE_CONFIGS:
            mode_dir = output_dir / "targets" / target["id"] / mode
            try:
                row = dict(
                    run_mode(
                        target=target,
                        mode=mode,
                        output_dir=mode_dir,
                    )
                )
                row.update(
                    {
                        "target_id": target["id"],
                        "category": target["category"],
                        "length_bucket": target["length_bucket"],
                        "mode": mode,
                        "role": MODE_CONFIGS[mode]["role"],
                        "status": "success",
                        "error": "",
                        "calibration_bank_sha256": (
                            calibration_bank_sha256
                            if mode == "calibration_holdout"
                            else ""
                        ),
                        "artifact": str(mode_dir),
                    }
                )
            except Exception as exc:  # target-level failures must not abort the matrix
                row = {
                    "target_id": target["id"],
                    "category": target["category"],
                    "length_bucket": target["length_bucket"],
                    "mode": mode,
                    "role": MODE_CONFIGS[mode]["role"],
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "prompt_tokens": "",
                    "calibration_bank_sha256": (
                        calibration_bank_sha256
                        if mode == "calibration_holdout"
                        else ""
                    ),
                    "logical_byte_saving_fraction": "",
                    "missing_tokens": "",
                    "missing_mse": "",
                    "missing_mae": "",
                    "missing_max_abs": "",
                    "max_abs_logit_diff": "",
                    "mean_abs_logit_diff": "",
                    "argmax_match": "",
                    "original_argmax": "",
                    "restored_argmax": "",
                    "artifact": str(mode_dir),
                }
            mode_dir.mkdir(parents=True, exist_ok=True)
            (mode_dir / "summary.json").write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            rows.append(row)
            attempts.append(
                {
                    "target_id": target["id"],
                    "mode": mode,
                    "status": row["status"],
                    "summary_file": str(mode_dir / "summary.json"),
                }
            )
    manifest = {
        "version": 1,
        "calibration_bank_sha256": calibration_bank_sha256,
        "modes": list(MODE_CONFIGS),
        "targets": targets,
        "attempts": attempts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows, manifest
```

- [ ] **Step 4: Implement real TinyLLM setup and row normalization**

The CLI must accept:

```text
--model
--policy-file
--target-file
--repo-root
--calibration-prompt
--calibration-prompt-extra
--calibration-prompts-file
--task-id
--doc-id
--max-model-len
--gpu-memory-utilization
--source-count
--recover-ridge
--output-dir
--target-limit
```

Use one `LLM(...)` instance. Build calibration samples with the existing
prefix-only helpers, call:

```python
bank, calibration_plan, source_heads = fit_calibration_recovery_bank(
    calibration_kv=calibration_kv,
    calibration_tokens=calibration_tokens,
    policy_file=args.policy_file,
    repo_root=args.repo_root,
    task_id=args.task_id,
    doc_id=args.doc_id,
    source_count=args.source_count,
    source_map="calibration_holdout",
    recover_ridge=args.recover_ridge,
)
```

Save the bank once to
`output_dir/calibration/multi_source_recovery_bank.json`, compute its SHA256,
and write `calibration/summary.json` containing calibration token count,
plan summary, sorted source-head mapping, bank path, and hash.

Implement the real `run_mode` closure by calling `run_read_path_smoke(...)`.
Convert its nested summary to the normalized row fields:

```python
error_metrics = summary["sidecar"]["error_metrics"]
compare = summary["logit_compare"]
return {
    "prompt_tokens": int(summary["prompt_tokens"]),
    "logical_byte_saving_fraction": float(
        summary["sidecar"]["logical_byte_saving_fraction"]
    ),
    "missing_tokens": int(error_metrics["num_missing_compact_tokens"]),
    "missing_mse": float(error_metrics["mse_missing_compact_tokens"]),
    "missing_mae": float(error_metrics["mae_missing_compact_tokens"]),
    "missing_max_abs": float(error_metrics["max_abs_missing_compact_tokens"]),
    "max_abs_logit_diff": float(compare["max_abs_logit_diff"]),
    "mean_abs_logit_diff": float(compare["mean_abs_logit_diff"]),
    "argmax_match": bool(compare["argmax_match"]),
    "original_argmax": int(compare["original_argmax"]),
    "restored_argmax": int(compare["restored_argmax"]),
}
```

After `run_target_matrix`, call `aggregate_rows` and `write_outputs`. Add actual
token-bucket validation to the final report: record a failed condition when a
successful target's token count falls outside its intended bucket, using
`short=16..48`, `medium=49..160`, and `long=161..384`.

Ensure `llm.exit()` runs once in `finally`, unregistering the atexit handler as
the existing scripts do.

- [ ] **Step 5: Run CPU-only tests and syntax checks**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
  python3 -m pytest -q tools/test_light_doc_cache_multi_target.py

PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  experiments/light_doc_cache/run_tinyllm_read_path_multi_target.py \
  experiments/light_doc_cache/make_multi_target_read_path_report.py
```

Expected: tests pass and compilation exits 0.

- [ ] **Step 6: Commit the driver**

```bash
git add \
  experiments/light_doc_cache/run_tinyllm_read_path_multi_target.py \
  experiments/light_doc_cache/make_multi_target_read_path_report.py \
  tools/test_light_doc_cache_multi_target.py
git commit -m "Add Light Doc Cache multi-target driver"
```

---

### Task 5: Remote Runner and Two-Target Smoke

**Files:**
- Create: `experiments/light_doc_cache/run_tinyllm_read_path_multi_target_remote.sh`
- Modify: `tools/test_light_doc_cache_multi_target.py`

**Interfaces:**
- Consumes the driver, reporter, calibration script, read-path script, runtime module, policy files/directories, and target JSON.
- Produces local mirrored smoke/full artifact directories.

- [ ] **Step 1: Add failing shell-contract test**

Append:

```python
def test_multi_target_remote_runner_has_required_safety_contract() -> None:
    script = (
        ROOT
        / "experiments"
        / "light_doc_cache"
        / "run_tinyllm_read_path_multi_target_remote.sh"
    )
    text = script.read_text(encoding="utf-8")
    for needle in [
        "set -euo pipefail",
        "sitian@10.232.195.203",
        "CONTROL_PATH",
        "rsync",
        "--relative",
        "nvidia-smi --query-gpu=index,memory.used",
        "TINYVLLM_DIST_PORT",
        "MASTER_PORT",
        "TARGET_LIMIT",
        "multi_target_summary.json",
    ]:
        assert needle in text
```

- [ ] **Step 2: Run the focused test and verify failure**

Expected: failure because the shell runner does not exist.

- [ ] **Step 3: Implement the remote runner**

Create a strict Bash script with defaults:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REMOTE="${REMOTE:-sitian@10.232.195.203}"
CONTROL_PATH="${CONTROL_PATH:-/tmp/ssh-sitian-10.232.195.203}"
REMOTE_REPO="${REMOTE_REPO:-/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge}"
REMOTE_PY="${REMOTE_PY:-/data00/home/sitian/sitian-workspace01/tllm/env/bin/python}"
MODEL="${MODEL:-/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B}"
GPU="${GPU:-auto}"
TARGET_LIMIT="${TARGET_LIMIT:-0}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
REMOTE_OUTPUT="${REMOTE_OUTPUT:-$REMOTE_REPO/profile_out/light_doc_cache_multi_target_$TAG}"
LOCAL_OUTPUT="${LOCAL_OUTPUT:-$SCRIPT_DIR/read_path_multi_target_$TAG}"
POLICY_FILE="${POLICY_FILE:-experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json}"
TARGET_FILE="${TARGET_FILE:-experiments/light_doc_cache/read_path_multi_target_prompts_v1.json}"
```

Build `SSH=(ssh ...)` and `RSYNC_RSH="ssh ..."` with the ControlMaster only
when the socket exists. Synchronize these paths with `rsync -av --relative`:

```text
experiments/light_doc_cache/run_tinyllm_read_path_multi_target.py
experiments/light_doc_cache/make_multi_target_read_path_report.py
experiments/light_doc_cache/run_tinyllm_calibrated_kv_smoke.py
experiments/light_doc_cache/run_tinyllm_sidecar_read_path_smoke.py
experiments/light_doc_cache/read_path_multi_target_prompts_v1.json
experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json
tinyvllm/engine/light_doc_cache_runtime.py
```

Also parse the adaptive policy and synchronize every referenced
`policy_rows.csv` with relative paths. Verify remote Python compilation before
running.

For `GPU=auto`, select the least-used GPU:

```bash
GPU="$("${SSH[@]}" "$REMOTE" \
  "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
   | sort -t, -k2,2n | head -1 | cut -d, -f1 | tr -d ' ')"
```

Allocate a dynamic remote port:

```bash
PORT="$("${SSH[@]}" "$REMOTE" \
  "$REMOTE_PY -c 'import socket; s=socket.socket(); s.bind((\"\",0)); print(s.getsockname()[1]); s.close()'")"
```

Run the driver with the three historical calibration prompts:

```bash
CUDA_VISIBLE_DEVICES="$GPU" \
TINYVLLM_DIST_PORT="$PORT" \
MASTER_PORT="$PORT" \
PYTHONPATH="$REMOTE_REPO" \
"$REMOTE_PY" experiments/light_doc_cache/run_tinyllm_read_path_multi_target.py \
  --model "$MODEL" \
  --policy-file "$POLICY_FILE" \
  --target-file "$TARGET_FILE" \
  --calibration-prompt "Light Doc Cache TinyLLM calibration prompt." \
  --calibration-prompt-extra "Light Doc Cache second calibration prompt for trained recovery." \
  --calibration-prompt-extra "Light Doc Cache third calibration prompt for Qwen KV recovery." \
  --source-count 2 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.30 \
  --target-limit "$TARGET_LIMIT" \
  --output-dir "$REMOTE_OUTPUT"
```

Mirror the entire output directory with rsync and print the local
`multi_target_report.md`.

- [ ] **Step 4: Run local shell and test validation**

```bash
bash -n experiments/light_doc_cache/run_tinyllm_read_path_multi_target_remote.sh

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
  python3 -m pytest -q tools/test_light_doc_cache_multi_target.py
```

Expected: shell syntax and tests pass.

- [ ] **Step 5: Commit the runner**

```bash
git add \
  experiments/light_doc_cache/run_tinyllm_read_path_multi_target_remote.sh \
  tools/test_light_doc_cache_multi_target.py
git commit -m "Add remote Light Doc Cache target runner"
```

- [ ] **Step 6: Run the remote two-target smoke**

First inspect connectivity and GPU state:

```bash
ssh -S /tmp/ssh-sitian-10.232.195.203 \
  -o BatchMode=yes -o ConnectTimeout=10 \
  sitian@10.232.195.203 \
  'hostname; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits'
```

Then run:

```bash
CONTROL_PATH=/tmp/ssh-sitian-10.232.195.203 \
TARGET_LIMIT=2 \
TAG=20260714_smoke \
LOCAL_OUTPUT=experiments/light_doc_cache/read_path_multi_target_qwen3_0_6b_20260714_smoke \
  experiments/light_doc_cache/run_tinyllm_read_path_multi_target_remote.sh
```

Expected evidence:

- one calibration bank file and one SHA256;
- two targets × three modes = six attempted rows;
- distinct target IDs and tokenizer counts;
- all successful holdout rows carry the same bank hash;
- `multi_target_rows.csv`, `multi_target_summary.json`, and
  `multi_target_report.md` exist locally;
- target-limit smoke may be `NO_GO` because the full eight-pair condition is
  intentionally unmet.

If the smoke fails because actual token counts miss intended buckets, edit only
the dataset prompts, rerun Task 1 validation, and repeat the smoke before the
full run.

---

### Task 6: Full Eight-Target Gate, Audit, and Documentation

**Files:**
- Modify: `experiments/light_doc_cache/README.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Create by execution:
  `experiments/light_doc_cache/read_path_multi_target_qwen3_0_6b_20260714/`

**Interfaces:**
- Consumes the validated remote runner and all earlier artifacts.
- Produces the final decision and handoff evidence.

- [ ] **Step 1: Run the complete local validation set**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
  python3 -m pytest -q \
  tools/test_light_doc_cache_multi_target.py \
  tools/test_light_doc_cache_recovery_probe.py \
  tools/test_light_doc_cache_runtime.py

PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  experiments/light_doc_cache/run_tinyllm_read_path_multi_target.py \
  experiments/light_doc_cache/make_multi_target_read_path_report.py \
  experiments/light_doc_cache/run_tinyllm_calibrated_kv_smoke.py \
  experiments/light_doc_cache/run_tinyllm_sidecar_read_path_smoke.py \
  tools/test_light_doc_cache_multi_target.py

bash -n experiments/light_doc_cache/run_tinyllm_read_path_multi_target_remote.sh
git diff --check
```

Expected: all tests pass, syntax checks exit 0, and `git diff --check` is clean.

- [ ] **Step 2: Run the full remote matrix**

```bash
CONTROL_PATH=/tmp/ssh-sitian-10.232.195.203 \
TARGET_LIMIT=0 \
TAG=20260714_full \
LOCAL_OUTPUT=experiments/light_doc_cache/read_path_multi_target_qwen3_0_6b_20260714 \
  experiments/light_doc_cache/run_tinyllm_read_path_multi_target_remote.sh
```

Do not reuse the two-target smoke output directory.

- [ ] **Step 3: Independently audit generated artifacts**

Run:

```bash
python3 - <<'PY'
import csv
import hashlib
import json
from pathlib import Path

root = Path("experiments/light_doc_cache/read_path_multi_target_qwen3_0_6b_20260714")
manifest = json.loads((root / "manifest.json").read_text())
summary = json.loads((root / "multi_target_summary.json").read_text())
rows = list(csv.DictReader((root / "multi_target_rows.csv").open()))
assert len(manifest["targets"]) == 8
assert len(manifest["attempts"]) == 24
assert len(rows) == 24
assert {row["mode"] for row in rows} == {
    "repeat_last_target",
    "correlated_same_layer_target",
    "calibration_holdout",
}
bank = root / "calibration" / "multi_source_recovery_bank.json"
bank_sha = hashlib.sha256(bank.read_bytes()).hexdigest()
assert bank_sha == manifest["calibration_bank_sha256"]
holdout_hashes = {
    row["calibration_bank_sha256"]
    for row in rows
    if row["mode"] == "calibration_holdout" and row["status"] == "success"
}
assert holdout_hashes == {bank_sha}
assert summary["gate"]["decision"] in {"GO", "NO_GO"}
print("MULTI_TARGET_AUDIT_OK", summary["gate"]["decision"])
PY
```

Expected: `MULTI_TARGET_AUDIT_OK GO` or
`MULTI_TARGET_AUDIT_OK NO_GO`.

- [ ] **Step 4: Perform the prompt-to-artifact completion audit**

Create a checklist while inspecting real files:

```text
[ ] Eight required categories -> target JSON and manifest targets
[ ] Two targets per actual length bucket -> per-row prompt_tokens
[ ] One immutable bank -> bank SHA, manifest, every successful holdout row
[ ] Three modes for every target -> 24 manifest attempts and CSV rows
[ ] Per-target metrics -> each targets/<id>/<mode>/summary.json
[ ] Aggregate mean/median/P90/worst -> multi_target_summary.json
[ ] Win rate and relative regressions -> summary gate fields
[ ] GO/NO_GO conditions -> report checkboxes and independent audit
[ ] Default-off boundary -> raw summaries and report boundary text
[ ] No physical-memory or speed claim -> README and handoff wording
```

Treat any unchecked item as incomplete and fix or rerun before documenting the
result.

- [ ] **Step 5: Document the actual result**

Append to `experiments/light_doc_cache/README.md`:

- exact local and remote commands;
- target dataset and calibration-bank hash;
- artifact directory;
- a compact per-mode aggregate table;
- every gate condition and final `GO`/`NO_GO`;
- failures or weak targets;
- explicit claim boundary.

Append to `AGENT_HANDOFF_STATE.md`:

- commit IDs for implementation slices;
- remote GPU and dynamic port used;
- smoke and full artifact paths;
- `MULTI_TARGET_AUDIT_OK` output;
- final decision;
- next action:
  - `GO`: design a separate default-off hot-path/storage benchmark.
  - `NO_GO`: stop tuning this selector on the same targets and move to APC
    shared-prefix benchmarking or adaptive speculative decoding.

- [ ] **Step 6: Re-run documentation and repository checks**

```bash
git diff --check
git status --short
```

Inspect `git status` and stage only task-owned files and the final artifact
directory. Do not stage `.agents/`, `.codex/`, checkpoints, unrelated
`needle_sq_results/`, or pre-existing uncommitted model-runner work.

- [ ] **Step 7: Commit the evidence and decision**

```bash
git add \
  AGENT_HANDOFF_STATE.md \
  experiments/light_doc_cache/README.md \
  experiments/light_doc_cache/read_path_multi_target_qwen3_0_6b_20260714
git commit -m "Record Light Doc Cache multi-target gate"
```

- [ ] **Step 8: Final verification before any completion claim**

```bash
git log -6 --oneline --decorate
git status --short
```

Verify that:

- implementation commits are present;
- the final evidence commit is present;
- remaining status entries are only known pre-existing unrelated changes;
- the objective is not declared achieved globally, because this gate is one
  bounded step in the continuing inference-performance program.
