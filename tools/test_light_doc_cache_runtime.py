"""Runtime planning tests for adaptive light doc cache metrics.

Run:
  PYTHONPATH=$PWD python3 -m pytest -q tools/test_light_doc_cache_runtime.py
"""

from __future__ import annotations

import json
import importlib.util
import os
import sys
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_MODULE_PATH = Path(_REPO_ROOT) / "tinyvllm" / "engine" / "light_doc_cache_runtime.py"
_SPEC = importlib.util.spec_from_file_location("light_doc_cache_runtime", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

LightDocCacheRuntimeConfig = _MODULE.LightDocCacheRuntimeConfig
build_config_from_policy_dirs = _MODULE.build_config_from_policy_dirs
build_light_doc_cache_runtime_plan = _MODULE.build_light_doc_cache_runtime_plan
load_light_doc_cache_policy = _MODULE.load_light_doc_cache_policy
main = _MODULE.main


def _write_policy(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "kind": "task_adaptive_light_doc_cache_policy",
                "default_policy_dir": "experiments/light_doc_cache/policy_taskaware_budget75_from79_add_l6h6",
                "base_safe_policy_dir": "experiments/light_doc_cache/policy_recovery_seed78_add_l11h3_b050",
                "doc_top_tasks": {"first": 4, "second": 3},
                "overrides": {
                    "topk8_quality": {"drop_added_heads": ["6:6"], "policy_dir": "base"},
                    "smoothquant_status": {"drop_added_heads": ["6:6"], "policy_dir": "base"},
                },
            }
        ),
        encoding="utf-8",
    )


def test_default_off_plan_reports_no_compression_without_policy() -> None:
    cfg = LightDocCacheRuntimeConfig(enabled=False, num_layers=24, num_kv_heads=4)

    plan = build_light_doc_cache_runtime_plan(
        cfg,
        task_id="topk8_quality",
        doc_id="first",
        seq_len=1024,
    )

    assert plan.enabled is False
    assert plan.total_head_token_entries == 24 * 4 * 1024
    assert plan.stored_head_token_entries == plan.total_head_token_entries
    assert plan.recovered_head_token_entries == 0
    assert plan.effective_saving_fraction == 0.0
    assert plan.applied_added_heads == []
    assert plan.fallback_reason == "disabled"


def test_policy_default_adds_one_recovery_head_and_reports_entry_savings(tmp_path: Path) -> None:
    policy_path = tmp_path / "adaptive_policy.json"
    _write_policy(policy_path)
    policy = load_light_doc_cache_policy(policy_path)
    cfg = LightDocCacheRuntimeConfig(
        enabled=True,
        num_layers=24,
        num_kv_heads=8,
        policy=policy,
        base_recovered_heads=[(11, 3)],
        base_budget_fraction=0.5,
    )

    plan = build_light_doc_cache_runtime_plan(
        cfg,
        task_id="ordinary_task",
        doc_id="first",
        seq_len=100,
    )

    assert plan.enabled is True
    assert plan.total_head_token_entries == 24 * 8 * 100
    assert plan.recovered_heads == [(6, 6), (11, 3)]
    assert plan.applied_added_heads == [(6, 6)]
    assert plan.stored_head_token_entries == int((24 * 8 - 0.75) * 100)
    assert plan.recovered_head_token_entries == int(0.75 * 100)
    assert plan.effective_saving_fraction == 0.75 / (24 * 8)
    assert plan.fallback_reason is None


def test_policy_override_drops_added_head_but_keeps_base_recovery(tmp_path: Path) -> None:
    policy_path = tmp_path / "adaptive_policy.json"
    _write_policy(policy_path)
    policy = load_light_doc_cache_policy(policy_path)
    cfg = LightDocCacheRuntimeConfig(
        enabled=True,
        num_layers=24,
        num_kv_heads=8,
        policy=policy,
        base_recovered_heads=[(11, 3)],
        base_budget_fraction=0.5,
    )

    plan = build_light_doc_cache_runtime_plan(
        cfg,
        task_id="smoothquant_status",
        doc_id="second",
        seq_len=100,
    )

    assert plan.recovered_heads == [(11, 3)]
    assert plan.applied_added_heads == []
    assert plan.dropped_added_heads == [(6, 6)]
    assert plan.stored_head_token_entries == int((24 * 8 - 0.5) * 100)
    assert plan.recovered_head_token_entries == int(0.5 * 100)
    assert plan.effective_saving_fraction == 0.5 / (24 * 8)
    assert plan.fallback_reason == "task_override"


def test_policy_override_accepts_real_drop_heads_field(tmp_path: Path) -> None:
    policy_path = tmp_path / "adaptive_policy.json"
    _write_policy(policy_path)
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    payload["overrides"]["smoothquant_status"] = {"drop_heads": ["6:6"]}
    policy_path.write_text(json.dumps(payload), encoding="utf-8")
    policy = load_light_doc_cache_policy(policy_path)
    cfg = LightDocCacheRuntimeConfig(
        enabled=True,
        num_layers=24,
        num_kv_heads=8,
        policy=policy,
        base_recovered_heads=[(11, 3)],
        base_budget_fraction=0.5,
    )

    plan = build_light_doc_cache_runtime_plan(
        cfg,
        task_id="smoothquant_status",
        doc_id="second",
        seq_len=100,
    )

    assert plan.applied_added_heads == []
    assert plan.dropped_added_heads == [(6, 6)]
    assert plan.recovered_kv_head_equivalent == 0.5


def test_plan_summary_matches_79_to_80_head_frontier_counts(tmp_path: Path) -> None:
    policy_path = tmp_path / "adaptive_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "kind": "task_adaptive_light_doc_cache_policy",
                "default_policy_dir": "experiments/light_doc_cache/policy_taskaware_budget75_from79_add_l6h4",
                "base_safe_policy_dir": "experiments/light_doc_cache/policy_recovery_seed78_add_l11h3_b050",
                "overrides": {
                    "topk8_quality": {"drop_added_heads": ["6:4"], "policy_dir": "base"},
                },
            }
        ),
        encoding="utf-8",
    )
    policy = load_light_doc_cache_policy(policy_path)
    cfg = LightDocCacheRuntimeConfig(
        enabled=True,
        num_layers=16,
        num_kv_heads=5,
        policy=policy,
        base_recovered_heads=[(11, 3)],
        base_budget_fraction=0.5,
    )

    default_plan = build_light_doc_cache_runtime_plan(
        cfg,
        task_id="ordinary_task",
        doc_id="first",
        seq_len=2048,
    )
    fallback_plan = build_light_doc_cache_runtime_plan(
        cfg,
        task_id="topk8_quality",
        doc_id="first",
        seq_len=2048,
    )

    assert default_plan.total_kv_heads == 80
    assert default_plan.stored_kv_heads_equivalent == 79.25
    assert default_plan.recovered_kv_heads == 2
    assert default_plan.recovered_kv_head_equivalent == 0.75
    assert default_plan.compression_ratio == 80 / 79.25
    assert fallback_plan.total_kv_heads == 80
    assert fallback_plan.stored_kv_heads_equivalent == 79.5
    assert fallback_plan.recovered_kv_heads == 1
    assert fallback_plan.recovered_kv_head_equivalent == 0.5
    assert fallback_plan.compression_ratio == 80 / 79.5
    assert fallback_plan.as_summary()["stored_kv_heads_equivalent"] == 79.5


def test_invalid_head_coordinates_are_rejected(tmp_path: Path) -> None:
    policy_path = tmp_path / "adaptive_policy.json"
    _write_policy(policy_path)
    policy = load_light_doc_cache_policy(policy_path)
    cfg = LightDocCacheRuntimeConfig(
        enabled=True,
        num_layers=4,
        num_kv_heads=4,
        policy=policy,
        base_recovered_heads=[],
    )

    try:
        build_light_doc_cache_runtime_plan(
            cfg,
            task_id="ordinary_task",
            doc_id="first",
            seq_len=100,
        )
    except ValueError as exc:
        assert "layer/head outside model shape" in str(exc)
    else:
        raise AssertionError("expected invalid layer/head coordinates to fail")


def test_cli_writes_json_summary(tmp_path: Path, capsys) -> None:
    policy_path = tmp_path / "adaptive_policy.json"
    _write_policy(policy_path)

    exit_code = main(
        [
            "--enabled",
            "--policy-file",
            str(policy_path),
            "--task-id",
            "smoothquant_status",
            "--doc-id",
            "second",
            "--seq-len",
            "100",
            "--num-layers",
            "24",
            "--num-kv-heads",
            "8",
            "--base-recovered-heads",
            "11:3",
            "--base-budget-fraction",
            "0.5",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert summary["fallback_reason"] == "task_override"
    assert summary["stored_kv_heads_equivalent"] == 191.5
    assert summary["recovered_kv_head_equivalent"] == 0.5


def test_build_config_from_real_policy_rows_uses_mixed_budgets() -> None:
    policy = load_light_doc_cache_policy(
        Path(_REPO_ROOT)
        / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
    )
    cfg = build_config_from_policy_dirs(
        policy,
        repo_root=Path(_REPO_ROOT),
        num_layers=28,
        num_kv_heads=8,
        enabled=True,
    )

    default_plan = build_light_doc_cache_runtime_plan(
        cfg,
        task_id="ordinary_task",
        doc_id="first",
        seq_len=1536,
    )
    fallback_plan = build_light_doc_cache_runtime_plan(
        cfg,
        task_id="smoothquant_status",
        doc_id="second",
        seq_len=1536,
    )

    assert default_plan.recovered_kv_heads == 80
    assert default_plan.recovered_kv_head_equivalent == 39.75
    assert default_plan.effective_saving_fraction == 39.75 / 224
    assert fallback_plan.recovered_kv_heads == 79
    assert fallback_plan.recovered_kv_head_equivalent == 39.5
    assert fallback_plan.effective_saving_fraction == 39.5 / 224


def test_cli_can_read_real_policy_dirs(capsys) -> None:
    exit_code = main(
        [
            "--enabled",
            "--from-policy-dirs",
            "--repo-root",
            _REPO_ROOT,
            "--policy-file",
            str(
                Path(_REPO_ROOT)
                / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
            ),
            "--task-id",
            "smoothquant_status",
            "--doc-id",
            "second",
            "--seq-len",
            "1536",
            "--num-layers",
            "28",
            "--num-kv-heads",
            "8",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert summary["recovered_kv_heads"] == 79
    assert summary["recovered_kv_head_equivalent"] == 39.5
    assert summary["effective_saving_fraction"] == 39.5 / 224


def test_make_runtime_plan_table_writes_task_rows(tmp_path: Path) -> None:
    script_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/make_runtime_plan_table.py"
    spec = importlib.util.spec_from_file_location("make_runtime_plan_table", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    output_dir = tmp_path / "runtime_table"
    exit_code = module.main(
        [
            "--policy-file",
            str(
                Path(_REPO_ROOT)
                / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
            ),
            "--repo-root",
            _REPO_ROOT,
            "--num-layers",
            "28",
            "--num-kv-heads",
            "8",
            "--seq-len",
            "1536",
            "--task",
            "first:ordinary_task",
            "--task",
            "second:smoothquant_status",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    jsonl_path = output_dir / "runtime_plan_table.jsonl"
    md_path = output_dir / "runtime_plan_table.md"
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    markdown = md_path.read_text(encoding="utf-8")
    assert rows[0]["task_id"] == "ordinary_task"
    assert rows[0]["recovered_kv_head_equivalent"] == 39.75
    assert rows[1]["task_id"] == "smoothquant_status"
    assert rows[1]["recovered_kv_head_equivalent"] == 39.5
    assert "planning/metrics only" in markdown
    assert "smoothquant_status" in markdown


def test_make_runtime_plan_table_accepts_task_files_and_writes_average(tmp_path: Path) -> None:
    script_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/make_runtime_plan_table.py"
    spec = importlib.util.spec_from_file_location("make_runtime_plan_table", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    output_dir = tmp_path / "runtime_table"
    exit_code = module.main(
        [
            "--policy-file",
            str(
                Path(_REPO_ROOT)
                / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
            ),
            "--repo-root",
            _REPO_ROOT,
            "--num-layers",
            "28",
            "--num-kv-heads",
            "8",
            "--seq-len",
            "1536",
            "--task-file",
            f"first={Path(_REPO_ROOT) / 'experiments/light_doc_cache/task_quality_tasks_kv_sparse_v3_stable.json'}",
            "--task-file",
            f"second={Path(_REPO_ROOT) / 'experiments/light_doc_cache/task_quality_tasks_qwen3_8b_fixes_stable.json'}",
            "--output-dir",
            str(output_dir),
        ]
    )

    rows = [
        json.loads(line)
        for line in (output_dir / "runtime_plan_table.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    markdown = (output_dir / "runtime_plan_table.md").read_text(encoding="utf-8")
    assert exit_code == 0
    assert len(rows) == 19
    assert "Average Effective Saving" in markdown
    assert "17.70%" in markdown


def test_kv_storage_summary_maps_plan_to_full_cache_bytes() -> None:
    plan = build_light_doc_cache_runtime_plan(
        LightDocCacheRuntimeConfig(enabled=False, num_layers=2, num_kv_heads=4),
        task_id="noop",
        doc_id=None,
        seq_len=16,
    )
    summary = _MODULE.summarize_planned_kv_storage(
        plan,
        num_blocks=2,
        block_size=8,
        head_dim=4,
        element_size_bytes=2,
    )

    assert summary["full_kv_bytes"] == 2 * 2 * 2 * 8 * 4 * 4 * 2
    assert summary["planned_stored_kv_bytes"] == summary["full_kv_bytes"]
    assert summary["planned_recovered_kv_bytes"] == 0
    assert summary["planned_byte_saving_fraction"] == 0.0
    assert summary["claim_boundary"] == "planning_only_not_allocated"


def test_kv_storage_summary_reports_mixed_budget_bytes_from_real_policy() -> None:
    policy = load_light_doc_cache_policy(
        Path(_REPO_ROOT)
        / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
    )
    cfg = build_config_from_policy_dirs(
        policy,
        repo_root=Path(_REPO_ROOT),
        num_layers=28,
        num_kv_heads=8,
        enabled=True,
    )
    plan = build_light_doc_cache_runtime_plan(
        cfg,
        task_id="smoothquant_status",
        doc_id="second",
        seq_len=1536,
    )
    summary = _MODULE.summarize_planned_kv_storage(
        plan,
        num_blocks=6,
        block_size=256,
        head_dim=128,
        element_size_bytes=2,
    )

    bytes_per_head_token = 2 * 128 * 2
    assert summary["full_kv_bytes"] == 224 * 1536 * bytes_per_head_token
    assert summary["planned_recovered_kv_bytes"] == int(39.5 * 1536 * bytes_per_head_token)
    assert summary["planned_stored_kv_bytes"] == summary["full_kv_bytes"] - summary["planned_recovered_kv_bytes"]
    assert summary["planned_byte_saving_fraction"] == 39.5 / 224
    assert summary["full_cache_shape"] == [2, 28, 6, 256, 8, 128]


def test_make_runtime_plan_table_can_include_storage_bytes(tmp_path: Path) -> None:
    script_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/make_runtime_plan_table.py"
    spec = importlib.util.spec_from_file_location("make_runtime_plan_table", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    output_dir = tmp_path / "runtime_table"
    exit_code = module.main(
        [
            "--policy-file",
            str(
                Path(_REPO_ROOT)
                / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
            ),
            "--repo-root",
            _REPO_ROOT,
            "--num-layers",
            "28",
            "--num-kv-heads",
            "8",
            "--seq-len",
            "1536",
            "--num-blocks",
            "6",
            "--block-size",
            "256",
            "--head-dim",
            "128",
            "--element-size-bytes",
            "2",
            "--task",
            "second:smoothquant_status",
            "--output-dir",
            str(output_dir),
        ]
    )

    rows = [json.loads(line) for line in (output_dir / "runtime_plan_table.jsonl").read_text(encoding="utf-8").splitlines()]
    markdown = (output_dir / "runtime_plan_table.md").read_text(encoding="utf-8")
    assert exit_code == 0
    assert rows[0]["storage"]["full_kv_bytes"] == 176160768
    assert rows[0]["storage"]["planned_recovered_kv_bytes"] == 31064064
    assert "Average Planned Recovered KV Bytes" in markdown
    assert "31,064,064" in markdown


def test_kv_storage_summary_from_full_cache_shape_matches_direct_summary() -> None:
    policy = load_light_doc_cache_policy(
        Path(_REPO_ROOT)
        / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
    )
    cfg = build_config_from_policy_dirs(
        policy,
        repo_root=Path(_REPO_ROOT),
        num_layers=28,
        num_kv_heads=8,
        enabled=True,
    )
    plan = build_light_doc_cache_runtime_plan(
        cfg,
        task_id="smoothquant_status",
        doc_id="second",
        seq_len=1536,
    )

    summary = _MODULE.summarize_planned_kv_storage_from_shape(
        plan,
        full_cache_shape=(2, 28, 6, 256, 8, 128),
        element_size_bytes=2,
    )

    assert summary["full_cache_shape"] == [2, 28, 6, 256, 8, 128]
    assert summary["full_kv_bytes"] == 176160768
    assert summary["planned_recovered_kv_bytes"] == 31064064
    assert summary["planned_stored_kv_bytes"] == 145096704
    assert summary["shape_source"] == "kv_cache_shape"


def test_kv_storage_summary_from_shape_rejects_mismatched_layer_head() -> None:
    plan = build_light_doc_cache_runtime_plan(
        LightDocCacheRuntimeConfig(enabled=False, num_layers=2, num_kv_heads=4),
        task_id="noop",
        doc_id=None,
        seq_len=16,
    )

    try:
        _MODULE.summarize_planned_kv_storage_from_shape(
            plan,
            full_cache_shape=(2, 3, 2, 8, 4, 4),
            element_size_bytes=2,
        )
    except ValueError as exc:
        assert "shape layer/head mismatch" in str(exc)
    else:
        raise AssertionError("expected mismatched shape to fail")


def test_make_runtime_plan_table_can_take_kv_cache_shape(tmp_path: Path) -> None:
    script_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/make_runtime_plan_table.py"
    spec = importlib.util.spec_from_file_location("make_runtime_plan_table", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    output_dir = tmp_path / "runtime_table"
    exit_code = module.main(
        [
            "--policy-file",
            str(
                Path(_REPO_ROOT)
                / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
            ),
            "--repo-root",
            _REPO_ROOT,
            "--num-layers",
            "28",
            "--num-kv-heads",
            "8",
            "--seq-len",
            "1536",
            "--kv-cache-shape",
            "2,28,6,256,8,128",
            "--element-size-bytes",
            "2",
            "--task",
            "second:smoothquant_status",
            "--output-dir",
            str(output_dir),
        ]
    )

    row = json.loads((output_dir / "runtime_plan_table.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert exit_code == 0
    assert row["storage"]["shape_source"] == "kv_cache_shape"
    assert row["storage"]["full_cache_shape"] == [2, 28, 6, 256, 8, 128]


class _FakeKVCache:
    shape = (2, 28, 6, 256, 8, 128)

    def element_size(self):
        return 2


class _FakeModelRunner:
    def __init__(self):
        self.kv_cache = _FakeKVCache()


def test_model_runner_summary_helper_reads_kv_cache_shape() -> None:
    policy = load_light_doc_cache_policy(
        Path(_REPO_ROOT)
        / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
    )
    cfg = build_config_from_policy_dirs(
        policy,
        repo_root=Path(_REPO_ROOT),
        num_layers=28,
        num_kv_heads=8,
        enabled=True,
    )
    plan = build_light_doc_cache_runtime_plan(
        cfg,
        task_id="smoothquant_status",
        doc_id="second",
        seq_len=1536,
    )

    summary = _MODULE.build_model_runner_light_doc_cache_summary(_FakeModelRunner(), plan)

    assert summary["enabled"] is True
    assert summary["claim_boundary"] == "planning_only_not_allocated"
    assert summary["storage"]["shape_source"] == "kv_cache_shape"
    assert summary["storage"]["full_cache_shape"] == [2, 28, 6, 256, 8, 128]
    assert summary["storage"]["planned_recovered_kv_bytes"] == 31064064
    assert summary["next_step"] == "wire_compressed_storage_before_claiming_runtime_savings"


def test_model_runner_summary_helper_returns_none_without_kv_cache() -> None:
    plan = build_light_doc_cache_runtime_plan(
        LightDocCacheRuntimeConfig(enabled=False, num_layers=2, num_kv_heads=4),
        task_id="noop",
        doc_id=None,
        seq_len=16,
    )

    assert _MODULE.build_model_runner_light_doc_cache_summary(object(), plan) is None


def test_model_runner_exposes_summary_wrapper_without_hot_path_changes() -> None:
    source = (Path(_REPO_ROOT) / "tinyvllm/engine/model_runner.py").read_text(encoding="utf-8")
    assert "build_model_runner_light_doc_cache_summary" in source
    assert "materialize_model_runner_light_doc_cache_sidecar" in source
    assert "def light_doc_cache_planning_summary(self, plan)" in source
    assert "return build_model_runner_light_doc_cache_summary(self, plan)" in source
    assert "def light_doc_cache_materialize_sidecar(" in source
    assert "materialize_model_runner_light_doc_cache_sidecar(" in source
    assert "light_doc_cache_planning_summary" in source.split("def _get_pinned", 1)[0]
    assert "light_doc_cache_materialize_sidecar" in source.split("def _get_pinned", 1)[0]



def test_cpu_compressed_kv_storage_stores_selected_tokens_and_restores_shape() -> None:
    import numpy as np

    policy_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
    policy = load_light_doc_cache_policy(policy_path)
    cfg = build_config_from_policy_dirs(
        policy,
        repo_root=Path(_REPO_ROOT),
        num_layers=28,
        num_kv_heads=8,
        enabled=True,
    )
    plan = build_light_doc_cache_runtime_plan(cfg, task_id="smoothquant_status", doc_id="second", seq_len=8)
    kv = np.arange(2 * 28 * 1 * 8 * 8 * 2, dtype=np.float32).reshape(2, 28, 1, 8, 8, 2)

    storage = _MODULE.LightDocCacheCompressedKVStorage.from_full_kv(kv, plan, fill_value=-1.0)
    restored = storage.restore_to_full_shape()

    assert restored.shape == kv.shape
    assert storage.summary()["stored_tensor_bytes"] < storage.summary()["full_tensor_bytes"]
    # Fallback plan drops 6:4, but base compact head 11:3 stores first 4 tokens at b050.
    assert np.array_equal(restored[:, 11, :, :4, 3], kv[:, 11, :, :4, 3])
    assert np.array_equal(restored[:, 11, :, 4:, 3], np.full_like(kv[:, 11, :, 4:, 3], -1.0))
    # Non-compact heads stay full.
    assert np.array_equal(restored[:, 0, :, :, 6], kv[:, 0, :, :, 6])


def test_materialize_light_doc_cache_sidecar_reports_logical_bytes_and_readback() -> None:
    import numpy as np

    policy_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
    policy = load_light_doc_cache_policy(policy_path)
    cfg = build_config_from_policy_dirs(
        policy,
        repo_root=Path(_REPO_ROOT),
        num_layers=28,
        num_kv_heads=8,
        enabled=True,
    )
    plan = build_light_doc_cache_runtime_plan(cfg, task_id="smoothquant_status", doc_id="second", seq_len=12)
    kv = np.arange(2 * 28 * 1 * 16 * 8 * 2, dtype=np.float32).reshape(2, 28, 1, 16, 8, 2)

    storage, summary = _MODULE.materialize_light_doc_cache_sidecar(
        kv,
        plan,
        fill_value=-1.0,
        recover_missing_fn=_MODULE.make_repeat_last_recovery_callback(),
        evaluate_readback=True,
    )

    assert storage.summary()["compact_heads"] == 79
    assert summary["claim_boundary"] == "sidecar_materialized_not_attention_hot_path"
    assert summary["logical_full_kv_bytes"] == 2 * 28 * 12 * 8 * 2 * 4
    assert summary["logical_stored_kv_bytes"] == summary["sidecar_storage"]["stored_tensor_bytes"]
    assert 0.0 < summary["logical_byte_saving_fraction"] < 1.0
    assert summary["error_metrics"]["num_missing_compact_tokens"] == 474


def test_correlated_head_recovery_callback_uses_retained_full_head_signal() -> None:
    import numpy as np

    plan = build_light_doc_cache_runtime_plan(
        LightDocCacheRuntimeConfig(
            enabled=True,
            num_layers=2,
            num_kv_heads=2,
            policy=_MODULE.LightDocCachePolicy(
                kind="toy",
                default_policy_dir="toy_b050",
                default_budget_fraction=0.5,
            ),
            base_recovered_heads=[(1, 1)],
            base_budget_fraction=0.5,
        ),
        task_id="toy",
        doc_id=None,
        seq_len=8,
    )
    kv = np.zeros((2, 2, 1, 8, 2, 1), dtype=np.float32)
    source = np.arange(16, dtype=np.float32).reshape(2, 8, 1)
    kv[:, 0, :, :, 0, :] = source.reshape(2, 1, 8, 1)
    kv[:, 1, :, :, 1, :] = (2.0 * source + 3.0).reshape(2, 1, 8, 1)
    storage = _MODULE.LightDocCacheCompressedKVStorage.from_full_kv(kv, plan, fill_value=-1.0)

    callback = _MODULE.make_correlated_head_recovery_callback(
        storage,
        source_heads={(1, 1): (0, 0)},
        ridge=1e-6,
    )
    restored = storage.restore_to_full_shape(recover_missing_fn=callback)

    assert np.allclose(restored[:, 1, :, :, 1, :], kv[:, 1, :, :, 1, :], atol=1e-4)
    assert _MODULE.evaluate_restored_kv_error(kv, restored, plan)["mse_missing_compact_tokens"] < 1e-8


def test_correlated_head_recovery_callback_requires_retained_source_head() -> None:
    import numpy as np
    import pytest

    plan = build_light_doc_cache_runtime_plan(
        LightDocCacheRuntimeConfig(
            enabled=True,
            num_layers=2,
            num_kv_heads=2,
            policy=_MODULE.LightDocCachePolicy(
                kind="toy",
                default_policy_dir="toy_b050",
                default_budget_fraction=0.5,
            ),
            base_recovered_heads=[(1, 1)],
            base_budget_fraction=0.5,
        ),
        task_id="toy",
        doc_id=None,
        seq_len=8,
    )
    kv = np.zeros((2, 2, 1, 8, 2, 1), dtype=np.float32)
    storage = _MODULE.LightDocCacheCompressedKVStorage.from_full_kv(kv, plan, fill_value=-1.0)
    callback = _MODULE.make_correlated_head_recovery_callback(
        storage,
        source_heads={(1, 1): (1, 1)},
        ridge=1e-6,
    )

    with pytest.raises(ValueError, match="not retained as a full head"):
        storage.restore_to_full_shape(recover_missing_fn=callback)


def test_correlated_source_head_map_uses_prefix_fit_not_same_layer_first() -> None:
    import numpy as np

    plan = build_light_doc_cache_runtime_plan(
        LightDocCacheRuntimeConfig(
            enabled=True,
            num_layers=2,
            num_kv_heads=3,
            policy=_MODULE.LightDocCachePolicy(
                kind="toy",
                default_policy_dir="toy_b050",
                default_budget_fraction=0.5,
            ),
            base_recovered_heads=[(1, 1)],
            base_budget_fraction=0.5,
        ),
        task_id="toy",
        doc_id=None,
        seq_len=8,
    )
    kv = np.zeros((2, 2, 1, 8, 3, 1), dtype=np.float32)
    true_source = np.arange(16, dtype=np.float32).reshape(2, 8, 1)
    same_layer_bad_source = (np.arange(16, dtype=np.float32) ** 2).reshape(2, 8, 1)
    kv[:, 0, :, :, 0, :] = true_source.reshape(2, 1, 8, 1)
    kv[:, 1, :, :, 0, :] = same_layer_bad_source.reshape(2, 1, 8, 1)
    kv[:, 1, :, :, 1, :] = (3.0 * true_source - 2.0).reshape(2, 1, 8, 1)
    storage = _MODULE.LightDocCacheCompressedKVStorage.from_full_kv(kv, plan, fill_value=-1.0)

    assert _MODULE.build_correlated_source_head_map(storage, ridge=1e-6) == {(1, 1): (0, 0)}


def test_multi_source_correlated_recovery_combines_retained_heads() -> None:
    import numpy as np

    plan = build_light_doc_cache_runtime_plan(
        LightDocCacheRuntimeConfig(
            enabled=True,
            num_layers=1,
            num_kv_heads=3,
            policy=_MODULE.LightDocCachePolicy(
                kind="toy",
                default_policy_dir="toy_b050",
                default_budget_fraction=0.5,
            ),
            base_recovered_heads=[(0, 2)],
            base_budget_fraction=0.5,
        ),
        task_id="toy",
        doc_id=None,
        seq_len=8,
    )
    kv = np.zeros((2, 1, 1, 8, 3, 1), dtype=np.float32)
    source_a = np.arange(16, dtype=np.float32).reshape(2, 8, 1)
    source_b = (np.arange(16, dtype=np.float32) ** 2).reshape(2, 8, 1)
    target = 1.5 * source_a - 0.25 * source_b + 7.0
    kv[:, 0, :, :, 0, :] = source_a.reshape(2, 1, 8, 1)
    kv[:, 0, :, :, 1, :] = source_b.reshape(2, 1, 8, 1)
    kv[:, 0, :, :, 2, :] = target.reshape(2, 1, 8, 1)
    storage = _MODULE.LightDocCacheCompressedKVStorage.from_full_kv(kv, plan, fill_value=-1.0)

    callback = _MODULE.make_multi_source_correlated_head_recovery_callback(
        storage,
        source_heads={(0, 2): [(0, 0), (0, 1)]},
        ridge=1e-6,
    )
    restored = storage.restore_to_full_shape(recover_missing_fn=callback)

    assert np.allclose(restored[:, 0, :, :, 2, :], kv[:, 0, :, :, 2, :], atol=1e-3)
    assert _MODULE.evaluate_restored_kv_error(kv, restored, plan)["mse_missing_compact_tokens"] < 1e-6


def test_multi_source_correlated_recovery_handles_multi_dim_heads() -> None:
    import numpy as np

    plan = build_light_doc_cache_runtime_plan(
        LightDocCacheRuntimeConfig(
            enabled=True,
            num_layers=1,
            num_kv_heads=3,
            policy=_MODULE.LightDocCachePolicy(
                kind="toy",
                default_policy_dir="toy_b050",
                default_budget_fraction=0.5,
            ),
            base_recovered_heads=[(0, 2)],
            base_budget_fraction=0.5,
        ),
        task_id="toy",
        doc_id=None,
        seq_len=8,
    )
    kv = np.zeros((2, 1, 1, 8, 3, 2), dtype=np.float32)
    token_index = np.arange(16, dtype=np.float32).reshape(2, 8, 1)
    source_a = np.concatenate([token_index, token_index + 10.0], axis=2)
    source_b = np.concatenate([token_index ** 2, token_index * 0.5], axis=2)
    target = np.empty_like(source_a)
    target[..., 0:1] = 1.5 * source_a[..., 0:1] - 0.25 * source_b[..., 0:1] + 7.0
    target[..., 1:2] = -2.0 * source_a[..., 1:2] + 0.75 * source_b[..., 1:2] - 3.0
    kv[:, 0, :, :, 0, :] = source_a.reshape(2, 1, 8, 2)
    kv[:, 0, :, :, 1, :] = source_b.reshape(2, 1, 8, 2)
    kv[:, 0, :, :, 2, :] = target.reshape(2, 1, 8, 2)
    storage = _MODULE.LightDocCacheCompressedKVStorage.from_full_kv(kv, plan, fill_value=-1.0)

    callback = _MODULE.make_multi_source_correlated_head_recovery_callback(
        storage,
        source_heads={(0, 2): [(0, 0), (0, 1)]},
        ridge=1e-6,
    )
    restored = storage.restore_to_full_shape(recover_missing_fn=callback)

    assert np.allclose(restored[:, 0, :, :, 2, :], kv[:, 0, :, :, 2, :], atol=1e-3)


def test_calibrated_multi_source_recovery_bank_reuses_offline_weights() -> None:
    import numpy as np

    plan = build_light_doc_cache_runtime_plan(
        LightDocCacheRuntimeConfig(
            enabled=True,
            num_layers=1,
            num_kv_heads=3,
            policy=_MODULE.LightDocCachePolicy(
                kind="toy",
                default_policy_dir="toy_b050",
                default_budget_fraction=0.5,
            ),
            base_recovered_heads=[(0, 2)],
            base_budget_fraction=0.5,
        ),
        task_id="toy",
        doc_id=None,
        seq_len=8,
    )

    def make_kv(offset: float) -> np.ndarray:
        kv = np.zeros((2, 1, 1, 8, 3, 2), dtype=np.float32)
        token_index = (np.arange(16, dtype=np.float32).reshape(2, 8, 1) + offset)
        source_a = np.concatenate([token_index, token_index + 3.0], axis=2)
        source_b = np.concatenate([token_index ** 2, token_index * 0.25 - 2.0], axis=2)
        target = np.empty_like(source_a)
        target[..., 0:1] = 0.5 * source_a[..., 0:1] - 0.125 * source_b[..., 0:1] + 9.0
        target[..., 1:2] = -1.25 * source_a[..., 1:2] + 0.5 * source_b[..., 1:2] - 4.0
        kv[:, 0, :, :, 0, :] = source_a.reshape(2, 1, 8, 2)
        kv[:, 0, :, :, 1, :] = source_b.reshape(2, 1, 8, 2)
        kv[:, 0, :, :, 2, :] = target.reshape(2, 1, 8, 2)
        return kv

    calibration_kv = make_kv(offset=0.0)
    runtime_kv = make_kv(offset=5.0)
    bank = _MODULE.fit_multi_source_recovery_bank(
        calibration_kv,
        plan,
        source_heads={(0, 2): [(0, 0), (0, 1)]},
        ridge=1e-6,
    )
    storage = _MODULE.LightDocCacheCompressedKVStorage.from_full_kv(runtime_kv, plan, fill_value=-1.0)
    callback = _MODULE.make_calibrated_multi_source_recovery_callback(storage, bank)
    restored = storage.restore_to_full_shape(recover_missing_fn=callback)

    assert np.allclose(restored[:, 0, :, :, 2, :], runtime_kv[:, 0, :, :, 2, :], atol=1e-3)


def test_calibrated_multi_source_recovery_bank_roundtrips_json(tmp_path: Path) -> None:
    import numpy as np

    plan = build_light_doc_cache_runtime_plan(
        LightDocCacheRuntimeConfig(
            enabled=True,
            num_layers=1,
            num_kv_heads=3,
            policy=_MODULE.LightDocCachePolicy(
                kind="toy",
                default_policy_dir="toy_b050",
                default_budget_fraction=0.5,
            ),
            base_recovered_heads=[(0, 2)],
            base_budget_fraction=0.5,
        ),
        task_id="toy",
        doc_id=None,
        seq_len=8,
    )
    kv = np.zeros((2, 1, 1, 8, 3, 1), dtype=np.float32)
    source_a = np.arange(16, dtype=np.float32).reshape(2, 8, 1)
    source_b = (np.arange(16, dtype=np.float32) ** 2).reshape(2, 8, 1)
    target = 2.0 * source_a - 0.5 * source_b + 1.0
    kv[:, 0, :, :, 0, :] = source_a.reshape(2, 1, 8, 1)
    kv[:, 0, :, :, 1, :] = source_b.reshape(2, 1, 8, 1)
    kv[:, 0, :, :, 2, :] = target.reshape(2, 1, 8, 1)
    bank = _MODULE.fit_multi_source_recovery_bank(
        kv,
        plan,
        source_heads={(0, 2): [(0, 0), (0, 1)]},
        ridge=1e-6,
    )
    bank_path = tmp_path / "bank.json"
    _MODULE.save_multi_source_recovery_bank(bank, bank_path)
    loaded_bank = _MODULE.load_multi_source_recovery_bank(bank_path)
    storage = _MODULE.LightDocCacheCompressedKVStorage.from_full_kv(kv, plan, fill_value=-1.0)
    restored = storage.restore_to_full_shape(
        recover_missing_fn=_MODULE.make_calibrated_multi_source_recovery_callback(storage, loaded_bank)
    )

    payload = json.loads(bank_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "multi_source_recovery_bank"
    assert np.allclose(restored[:, 0, :, :, 2, :], kv[:, 0, :, :, 2, :], atol=1e-3)


def test_cpu_compressed_kv_storage_noop_plan_restores_full_array() -> None:
    import numpy as np

    plan = build_light_doc_cache_runtime_plan(
        LightDocCacheRuntimeConfig(enabled=False, num_layers=2, num_kv_heads=2),
        task_id="noop",
        doc_id=None,
        seq_len=8,
    )
    kv = np.arange(2 * 2 * 1 * 8 * 2 * 1, dtype=np.float32).reshape(2, 2, 1, 8, 2, 1)
    storage = _MODULE.LightDocCacheCompressedKVStorage.from_full_kv(kv, plan)
    assert storage.summary()["compact_heads"] == 0
    assert np.array_equal(storage.restore_to_full_shape(), kv)



def test_cpu_compressed_kv_storage_handles_selected_tokens_across_blocks() -> None:
    import numpy as np

    policy = load_light_doc_cache_policy(
        Path(_REPO_ROOT)
        / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
    )
    cfg = build_config_from_policy_dirs(
        policy,
        repo_root=Path(_REPO_ROOT),
        num_layers=28,
        num_kv_heads=8,
        enabled=True,
    )
    plan = build_light_doc_cache_runtime_plan(cfg, task_id="smoothquant_status", doc_id="second", seq_len=12)
    kv = np.arange(2 * 28 * 2 * 8 * 8 * 1, dtype=np.float32).reshape(2, 28, 2, 8, 8, 1)

    storage = _MODULE.LightDocCacheCompressedKVStorage.from_full_kv(kv, plan, fill_value=-1.0)
    restored = storage.restore_to_full_shape()
    original_flat = kv[:, 11, :, :, 3, :].reshape(2, 16, 1)
    restored_flat = restored[:, 11, :, :, 3, :].reshape(2, 16, 1)

    assert np.array_equal(restored_flat[:, :6], original_flat[:, :6])
    assert np.array_equal(restored_flat[:, 6:12], np.full_like(original_flat[:, 6:12], -1.0))
    # Capacity beyond seq_len should remain fill-value for compact heads too.
    assert np.array_equal(restored_flat[:, 12:], np.full_like(original_flat[:, 12:], -1.0))


def test_storage_prototype_smoke_script_writes_report(tmp_path: Path) -> None:
    script_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/run_storage_prototype_smoke.py"
    spec = importlib.util.spec_from_file_location("run_storage_prototype_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    output_dir = tmp_path / "storage_smoke"
    exit_code = module.main(
        [
            "--policy-file",
            str(
                Path(_REPO_ROOT)
                / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
            ),
            "--repo-root",
            _REPO_ROOT,
            "--output-dir",
            str(output_dir),
        ]
    )

    summary = json.loads((output_dir / "storage_prototype_summary.json").read_text(encoding="utf-8"))
    report = (output_dir / "storage_prototype_report.md").read_text(encoding="utf-8")
    assert exit_code == 0
    assert summary["storage"]["claim_boundary"] == "storage_prototype_fill_missing_not_recovered"
    assert summary["storage"]["stored_tensor_bytes"] < summary["storage"]["full_tensor_bytes"]
    assert "missing compact-head tokens are fill/repeat-last/linear-tail/oracle baselines" in report


def test_cpu_compressed_kv_storage_restore_with_recovery_callback() -> None:
    import numpy as np

    policy = load_light_doc_cache_policy(
        Path(_REPO_ROOT)
        / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
    )
    cfg = build_config_from_policy_dirs(
        policy,
        repo_root=Path(_REPO_ROOT),
        num_layers=28,
        num_kv_heads=8,
        enabled=True,
    )
    plan = build_light_doc_cache_runtime_plan(cfg, task_id="smoothquant_status", doc_id="second", seq_len=8)
    kv = np.arange(2 * 28 * 1 * 8 * 8 * 2, dtype=np.float32).reshape(2, 28, 1, 8, 8, 2)
    storage = _MODULE.LightDocCacheCompressedKVStorage.from_full_kv(kv, plan, fill_value=-1.0)
    calls = []

    def recover_missing(*, layer, kv_head, selected_tokens, missing_tokens, stored_tokens, head_dim, dtype):
        calls.append((layer, kv_head, selected_tokens, missing_tokens, stored_tokens.shape, head_dim, str(dtype)))
        return np.full((2, missing_tokens, head_dim), 42.0, dtype=dtype)

    restored = storage.restore_to_full_shape(recover_missing_fn=recover_missing)

    assert np.array_equal(restored[:, 11, :, :4, 3], kv[:, 11, :, :4, 3])
    assert np.array_equal(restored[:, 11, :, 4:, 3], np.full_like(kv[:, 11, :, 4:, 3], 42.0))
    assert any(call[:4] == (11, 3, 4, 4) for call in calls)


def test_cpu_compressed_kv_storage_recovery_callback_shape_is_checked() -> None:
    import numpy as np

    plan = build_light_doc_cache_runtime_plan(
        LightDocCacheRuntimeConfig(
            enabled=True,
            num_layers=1,
            num_kv_heads=1,
            policy=_MODULE.LightDocCachePolicy(
                kind="task_adaptive_light_doc_cache_policy",
                default_policy_dir="policy_add_l0h0_budget50",
                default_added_heads=((0, 0),),
                default_budget_fraction=0.5,
            ),
        ),
        task_id="task",
        doc_id=None,
        seq_len=8,
    )
    kv = np.zeros((2, 1, 1, 8, 1, 2), dtype=np.float32)
    storage = _MODULE.LightDocCacheCompressedKVStorage.from_full_kv(kv, plan)

    try:
        storage.restore_to_full_shape(recover_missing_fn=lambda **kwargs: np.zeros((2, 1, 2), dtype=np.float32))
    except ValueError as exc:
        assert "recovered missing tokens shape" in str(exc)
    else:
        raise AssertionError("expected bad recovery callback shape to fail")


def test_storage_prototype_smoke_script_supports_recovery_fill_mode(tmp_path: Path) -> None:
    script_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/run_storage_prototype_smoke.py"
    spec = importlib.util.spec_from_file_location("run_storage_prototype_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    output_dir = tmp_path / "storage_smoke"
    exit_code = module.main(
        [
            "--policy-file",
            str(
                Path(_REPO_ROOT)
                / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
            ),
            "--repo-root",
            _REPO_ROOT,
            "--recover-fill-value",
            "42.0",
            "--output-dir",
            str(output_dir),
        ]
    )

    summary = json.loads((output_dir / "storage_prototype_summary.json").read_text(encoding="utf-8"))
    report = (output_dir / "storage_prototype_report.md").read_text(encoding="utf-8")
    assert exit_code == 0
    assert summary["recovery_mode"] == "fill"
    assert "Recovery mode: `fill`" in report


def test_recovery_error_eval_and_oracle_callback_verify_layout() -> None:
    import numpy as np

    policy = load_light_doc_cache_policy(
        Path(_REPO_ROOT)
        / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
    )
    cfg = build_config_from_policy_dirs(
        policy,
        repo_root=Path(_REPO_ROOT),
        num_layers=28,
        num_kv_heads=8,
        enabled=True,
    )
    plan = build_light_doc_cache_runtime_plan(cfg, task_id="smoothquant_status", doc_id="second", seq_len=12)
    kv = np.arange(2 * 28 * 2 * 8 * 8 * 1, dtype=np.float32).reshape(2, 28, 2, 8, 8, 1)
    storage = _MODULE.LightDocCacheCompressedKVStorage.from_full_kv(kv, plan, fill_value=-1.0)

    fill_restored = storage.restore_to_full_shape(recover_missing_fn=lambda **kwargs: np.zeros((2, kwargs["missing_tokens"], kwargs["head_dim"]), dtype=np.float32))
    fill_metrics = _MODULE.evaluate_restored_kv_error(kv, fill_restored, plan)
    assert fill_metrics["mse_missing_compact_tokens"] > 0.0
    assert fill_metrics["num_missing_compact_tokens"] > 0

    oracle = _MODULE.make_oracle_recovery_callback(kv, plan)
    oracle_restored = storage.restore_to_full_shape(recover_missing_fn=oracle)
    oracle_metrics = _MODULE.evaluate_restored_kv_error(kv, oracle_restored, plan)
    assert oracle_metrics["mse_missing_compact_tokens"] == 0.0
    assert oracle_metrics["max_abs_missing_compact_tokens"] == 0.0
    assert oracle_metrics["num_missing_compact_tokens"] == fill_metrics["num_missing_compact_tokens"]


def test_repeat_last_recovery_callback_is_non_oracle_baseline() -> None:
    import numpy as np

    policy = load_light_doc_cache_policy(
        Path(_REPO_ROOT)
        / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
    )
    cfg = build_config_from_policy_dirs(
        policy,
        repo_root=Path(_REPO_ROOT),
        num_layers=28,
        num_kv_heads=8,
        enabled=True,
    )
    plan = build_light_doc_cache_runtime_plan(cfg, task_id="smoothquant_status", doc_id="second", seq_len=12)
    kv = np.arange(2 * 28 * 2 * 8 * 8 * 1, dtype=np.float32).reshape(2, 28, 2, 8, 8, 1)
    storage = _MODULE.LightDocCacheCompressedKVStorage.from_full_kv(kv, plan, fill_value=-1.0)

    fill_restored = storage.restore_to_full_shape(
        recover_missing_fn=lambda **kwargs: np.zeros((2, kwargs["missing_tokens"], kwargs["head_dim"]), dtype=np.float32)
    )
    repeat_restored = storage.restore_to_full_shape(
        recover_missing_fn=_MODULE.make_repeat_last_recovery_callback()
    )
    oracle_restored = storage.restore_to_full_shape(
        recover_missing_fn=_MODULE.make_oracle_recovery_callback(kv, plan)
    )

    fill_metrics = _MODULE.evaluate_restored_kv_error(kv, fill_restored, plan)
    repeat_metrics = _MODULE.evaluate_restored_kv_error(kv, repeat_restored, plan)
    oracle_metrics = _MODULE.evaluate_restored_kv_error(kv, oracle_restored, plan)

    assert repeat_metrics["num_missing_compact_tokens"] == fill_metrics["num_missing_compact_tokens"]
    assert repeat_metrics["mse_missing_compact_tokens"] > oracle_metrics["mse_missing_compact_tokens"]
    assert repeat_metrics["mse_missing_compact_tokens"] < fill_metrics["mse_missing_compact_tokens"]
    repeated_flat = repeat_restored[:, 11, :, :, 3, :].reshape(2, 16, 1)
    original_flat = kv[:, 11, :, :, 3, :].reshape(2, 16, 1)
    assert np.array_equal(repeated_flat[:, 6:12], np.repeat(original_flat[:, 5:6], 6, axis=1))


def test_linear_tail_recovery_callback_fits_prefix_trend_without_oracle() -> None:
    import numpy as np

    plan = build_light_doc_cache_runtime_plan(
        LightDocCacheRuntimeConfig(
            enabled=True,
            num_layers=1,
            num_kv_heads=1,
            policy=_MODULE.LightDocCachePolicy(
                kind="task_adaptive_light_doc_cache_policy",
                default_policy_dir="policy_add_l0h0_budget50",
                default_added_heads=((0, 0),),
                default_budget_fraction=0.5,
            ),
        ),
        task_id="task",
        doc_id=None,
        seq_len=8,
    )
    kv = np.zeros((2, 1, 1, 8, 1, 2), dtype=np.float32)
    token_positions = np.arange(8, dtype=np.float32)
    kv[0, 0, 0, :, 0, 0] = 2.0 * token_positions + 1.0
    kv[0, 0, 0, :, 0, 1] = -3.0 * token_positions + 4.0
    kv[1, 0, 0, :, 0, 0] = 0.5 * token_positions - 2.0
    kv[1, 0, 0, :, 0, 1] = 4.0 * token_positions + 0.25
    storage = _MODULE.LightDocCacheCompressedKVStorage.from_full_kv(kv, plan, fill_value=-1.0)

    repeat_restored = storage.restore_to_full_shape(
        recover_missing_fn=_MODULE.make_repeat_last_recovery_callback()
    )
    linear_restored = storage.restore_to_full_shape(
        recover_missing_fn=_MODULE.make_linear_tail_recovery_callback(ridge=1e-6)
    )
    oracle_restored = storage.restore_to_full_shape(
        recover_missing_fn=_MODULE.make_oracle_recovery_callback(kv, plan)
    )

    repeat_metrics = _MODULE.evaluate_restored_kv_error(kv, repeat_restored, plan)
    linear_metrics = _MODULE.evaluate_restored_kv_error(kv, linear_restored, plan)
    oracle_metrics = _MODULE.evaluate_restored_kv_error(kv, oracle_restored, plan)

    assert linear_metrics["mse_missing_compact_tokens"] < repeat_metrics["mse_missing_compact_tokens"]
    assert linear_metrics["mse_missing_compact_tokens"] > oracle_metrics["mse_missing_compact_tokens"]
    assert linear_metrics["max_abs_missing_compact_tokens"] < 1e-3


def test_storage_prototype_smoke_script_oracle_mode_has_zero_error(tmp_path: Path) -> None:
    script_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/run_storage_prototype_smoke.py"
    spec = importlib.util.spec_from_file_location("run_storage_prototype_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    output_dir = tmp_path / "storage_smoke"
    exit_code = module.main(
        [
            "--policy-file",
            str(
                Path(_REPO_ROOT)
                / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
            ),
            "--repo-root",
            _REPO_ROOT,
            "--recover-mode",
            "oracle",
            "--output-dir",
            str(output_dir),
        ]
    )

    summary = json.loads((output_dir / "storage_prototype_summary.json").read_text(encoding="utf-8"))
    report = (output_dir / "storage_prototype_report.md").read_text(encoding="utf-8")
    assert exit_code == 0
    assert summary["recovery_mode"] == "oracle"
    assert summary["error_metrics"]["mse_missing_compact_tokens"] == 0.0
    assert "Missing-token MSE: `0`" in report


def test_storage_prototype_smoke_script_repeat_last_mode(tmp_path: Path) -> None:
    script_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/run_storage_prototype_smoke.py"
    spec = importlib.util.spec_from_file_location("run_storage_prototype_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    output_dir = tmp_path / "storage_smoke"
    exit_code = module.main(
        [
            "--policy-file",
            str(
                Path(_REPO_ROOT)
                / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
            ),
            "--repo-root",
            _REPO_ROOT,
            "--recover-mode",
            "repeat_last",
            "--output-dir",
            str(output_dir),
        ]
    )

    summary = json.loads((output_dir / "storage_prototype_summary.json").read_text(encoding="utf-8"))
    report = (output_dir / "storage_prototype_report.md").read_text(encoding="utf-8")
    assert exit_code == 0
    assert summary["recovery_mode"] == "repeat_last"
    assert summary["error_metrics"]["mse_missing_compact_tokens"] > 0.0
    assert "Recovery mode: `repeat_last`" in report


def test_storage_prototype_smoke_script_linear_tail_mode(tmp_path: Path) -> None:
    script_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/run_storage_prototype_smoke.py"
    spec = importlib.util.spec_from_file_location("run_storage_prototype_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    output_dir = tmp_path / "storage_smoke"
    exit_code = module.main(
        [
            "--policy-file",
            str(
                Path(_REPO_ROOT)
                / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
            ),
            "--repo-root",
            _REPO_ROOT,
            "--recover-mode",
            "linear_tail",
            "--output-dir",
            str(output_dir),
        ]
    )

    summary = json.loads((output_dir / "storage_prototype_summary.json").read_text(encoding="utf-8"))
    report = (output_dir / "storage_prototype_report.md").read_text(encoding="utf-8")
    assert exit_code == 0
    assert summary["recovery_mode"] == "linear_tail"
    assert summary["error_metrics"]["mse_missing_compact_tokens"] > 0.0
    assert "Recovery mode: `linear_tail`" in report


def test_storage_prototype_smoke_script_nonlinear_pattern_exposes_linear_tail_error(tmp_path: Path) -> None:
    script_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/run_storage_prototype_smoke.py"
    spec = importlib.util.spec_from_file_location("run_storage_prototype_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    linear_output_dir = tmp_path / "linear_tail_storage_smoke"
    repeat_output_dir = tmp_path / "repeat_last_storage_smoke"
    common_args = [
        "--policy-file",
        str(
            Path(_REPO_ROOT)
            / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
        ),
        "--repo-root",
        _REPO_ROOT,
        "--kv-pattern",
        "nonlinear",
    ]

    assert module.main([*common_args, "--recover-mode", "linear_tail", "--output-dir", str(linear_output_dir)]) == 0
    assert module.main([*common_args, "--recover-mode", "repeat_last", "--output-dir", str(repeat_output_dir)]) == 0

    linear_summary = json.loads((linear_output_dir / "storage_prototype_summary.json").read_text(encoding="utf-8"))
    repeat_summary = json.loads((repeat_output_dir / "storage_prototype_summary.json").read_text(encoding="utf-8"))
    assert linear_summary["kv_pattern"] == "nonlinear"
    assert linear_summary["error_metrics"]["mse_missing_compact_tokens"] > 1e-6
    assert repeat_summary["error_metrics"]["mse_missing_compact_tokens"] > 1e-6
    assert (
        linear_summary["error_metrics"]["mse_missing_compact_tokens"]
        != repeat_summary["error_metrics"]["mse_missing_compact_tokens"]
    )


def test_tinyllm_kv_summary_smoke_writes_model_runner_accounting(tmp_path: Path) -> None:
    script_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/run_tinyllm_kv_summary_smoke.py"
    spec = importlib.util.spec_from_file_location("run_tinyllm_kv_summary_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class FakeKVCache:
        shape = (2, 28, 4, 256, 8, 128)

        def element_size(self):
            return 2

    class FakeModelRunner:
        kv_cache = FakeKVCache()

    output_dir = tmp_path / "tinyllm_summary"
    summary = module.write_tinyllm_kv_summary(
        model_runner=FakeModelRunner(),
        policy_file=str(
            Path(_REPO_ROOT)
            / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
        ),
        repo_root=_REPO_ROOT,
        task_id="smoothquant_status",
        doc_id="second",
        seq_len=512,
        output_dir=output_dir,
        model="fake-model",
        prompt_tokens=12,
    )

    written = json.loads((output_dir / "tinyllm_kv_summary.json").read_text(encoding="utf-8"))
    report = (output_dir / "tinyllm_kv_summary_report.md").read_text(encoding="utf-8")
    assert summary["claim_boundary"] == "tinyllm_allocated_kv_summary_only"
    assert written["storage"]["shape_source"] == "kv_cache_shape"
    assert written["allocated_kv_cache_bytes"] == 2 * 28 * 4 * 256 * 8 * 128 * 2
    assert written["storage"]["full_kv_bytes"] == 2 * 28 * 512 * 8 * 128 * 2
    assert written["storage"]["planned_recovered_kv_bytes"] > 0
    assert "TinyLLM KV Summary Smoke" in report


def test_tinyllm_sidecar_storage_helper_restores_from_fake_kv(tmp_path: Path) -> None:
    import numpy as np

    script_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/run_tinyllm_kv_summary_smoke.py"
    spec = importlib.util.spec_from_file_location("run_tinyllm_kv_summary_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class FakeModelRunner:
        def __init__(self):
            self.kv_cache = np.arange(2 * 28 * 1 * 16 * 8 * 2, dtype=np.float32).reshape(2, 28, 1, 16, 8, 2)
            self.sidecar_called = False

        def light_doc_cache_materialize_sidecar(
            self,
            plan,
            *,
            fill_value=-1.0,
            recover_missing_fn=None,
            evaluate_readback=False,
        ):
            self.sidecar_called = True
            _, sidecar_summary = _MODULE.materialize_light_doc_cache_sidecar(
                self.kv_cache,
                plan,
                fill_value=fill_value,
                recover_missing_fn=recover_missing_fn,
                evaluate_readback=evaluate_readback,
            )
            return sidecar_summary

    output_dir = tmp_path / "tinyllm_sidecar"
    model_runner = FakeModelRunner()
    summary = module.write_tinyllm_sidecar_storage_summary(
        model_runner=model_runner,
        policy_file=str(
            Path(_REPO_ROOT)
            / "experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json"
        ),
        repo_root=_REPO_ROOT,
        task_id="smoothquant_status",
        doc_id="second",
        seq_len=12,
        output_dir=output_dir,
        model="fake-model",
        prompt_tokens=12,
        recover_mode="linear_tail",
        recover_ridge=1e-6,
    )

    written = json.loads((output_dir / "tinyllm_sidecar_storage_summary.json").read_text(encoding="utf-8"))
    report = (output_dir / "tinyllm_sidecar_storage_report.md").read_text(encoding="utf-8")
    assert summary["claim_boundary"] == "tinyllm_sidecar_storage_readback_not_hot_path"
    assert written["sidecar_storage"]["stored_tensor_bytes"] < written["sidecar_storage"]["full_tensor_bytes"]
    assert written["logical_full_kv_bytes"] == 2 * 28 * 12 * 8 * 2 * 4
    assert written["logical_stored_kv_bytes"] == written["sidecar_storage"]["stored_tensor_bytes"]
    assert 0.0 < written["logical_byte_saving_fraction"] < 1.0
    assert written["error_metrics"]["num_missing_compact_tokens"] == 474
    assert written["recovery_mode"] == "linear_tail"
    assert model_runner.sidecar_called is True
    assert "TinyLLM Sidecar Storage Smoke" in report


def test_tinyllm_sidecar_read_path_smoke_exposes_default_off_compare_helpers() -> None:
    import numpy as np

    script_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/run_tinyllm_sidecar_read_path_smoke.py"
    spec = importlib.util.spec_from_file_location("run_tinyllm_sidecar_read_path_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class FakeModule:
        def __init__(self):
            self.k_cache = "original_k"
            self.v_cache = "original_v"

    class FakeModel:
        def __init__(self):
            self.layer = FakeModule()

        def modules(self):
            return [self.layer]

    class FakeRunner:
        def __init__(self):
            self.model = FakeModel()
            self.kv_cache = np.zeros((2, 1, 1, 2, 1, 1), dtype=np.float32)

    runner = FakeRunner()
    restored = np.ones((2, 1, 1, 2, 1, 1), dtype=np.float32)

    with module._temporary_model_runner_kv_cache(runner, restored):
        assert runner.kv_cache is restored
        assert np.array_equal(runner.model.layer.k_cache, restored[0, 0])
        assert np.array_equal(runner.model.layer.v_cache, restored[1, 0])

    assert isinstance(runner.kv_cache, np.ndarray)
    assert runner.kv_cache.sum() == 0.0
    assert runner.model.layer.k_cache == "original_k"
    assert runner.model.layer.v_cache == "original_v"
    source = script_path.read_text(encoding="utf-8")
    assert "default_off_restored_sidecar_read_path_logits_compare" in source
    assert "no hot-path code or KV allocation lifetime is changed" in source
    assert '"correlated"' in source
    assert '"multi_correlated"' in source
    assert '"calibrated_multi_correlated"' in source
    assert "--correlated-source-map" in source
    assert 'default="same_layer"' in source
    assert "--multi-correlated-source-count" in source
    assert "--recovery-bank-file" in source


def test_tinyllm_sidecar_read_path_correlated_mode_uses_runtime_source_map() -> None:
    script_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/run_tinyllm_sidecar_read_path_smoke.py"
    spec = importlib.util.spec_from_file_location("run_tinyllm_sidecar_read_path_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    source = script_path.read_text(encoding="utf-8")
    assert "build_correlated_source_head_map(storage" in source
    assert "_build_correlated_source_heads(plan)" in source


def test_read_path_recovery_matrix_generator_summarizes_artifacts(tmp_path: Path) -> None:
    script_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/make_read_path_recovery_matrix.py"
    spec = importlib.util.spec_from_file_location("make_read_path_recovery_matrix", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    def write_summary(path: Path, mode: str, mse: float, max_logit: float, mean_logit: float, argmax_match: bool) -> None:
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "prompt_tokens": 13,
                    "recovery_mode": mode,
                    "sidecar": {
                        "logical_byte_saving_fraction": 0.1899,
                        "error_metrics": {
                            "num_missing_compact_tokens": 553,
                            "mse_missing_compact_tokens": mse,
                            "mae_missing_compact_tokens": mse / 2,
                            "max_abs_missing_compact_tokens": mse * 10,
                        },
                    },
                    "logit_compare": {
                        "max_abs_logit_diff": max_logit,
                        "mean_abs_logit_diff": mean_logit,
                        "argmax_match": argmax_match,
                        "original_argmax": 1815,
                        "restored_argmax": 1815 if argmax_match else 3491,
                    },
                }
            ),
            encoding="utf-8",
        )

    repeat_summary = tmp_path / "repeat" / "tinyllm_sidecar_read_path_summary.json"
    oracle_summary = tmp_path / "oracle" / "tinyllm_sidecar_read_path_summary.json"
    write_summary(repeat_summary, "repeat_last", 13.0, 5.0, 0.7, False)
    write_summary(oracle_summary, "oracle", 0.0, 0.0, 0.0, True)
    output_dir = tmp_path / "matrix"

    exit_code = module.main(
        [
            "--artifact",
            f"repeat_last:baseline:{repeat_summary}",
            "--artifact",
            f"oracle:upper_bound:{oracle_summary}",
            "--output-dir",
            str(output_dir),
        ]
    )

    csv_text = (output_dir / "read_path_recovery_matrix.csv").read_text(encoding="utf-8")
    markdown = (output_dir / "read_path_recovery_matrix.md").read_text(encoding="utf-8")
    assert exit_code == 0
    assert "repeat_last,baseline,13" in csv_text
    assert "`oracle` | upper_bound | 0" in markdown
    assert "No non-oracle mode preserves argmax" in markdown


def test_real_kv_calibrated_recovery_smoke_exposes_bank_artifacts() -> None:
    script_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/run_real_kv_calibrated_recovery_smoke.py"
    spec = importlib.util.spec_from_file_location("run_real_kv_calibrated_recovery_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    source = script_path.read_text(encoding="utf-8")
    assert "--calibration-prompt" in source
    assert "--target-prompt" in source
    assert "fit_multi_source_recovery_bank" in source
    assert "save_multi_source_recovery_bank" in source
    assert "make_calibrated_multi_source_recovery_callback" in source
    assert "real_kv_calibrated_recovery_summary.json" in source


def test_tinyllm_calibrated_kv_smoke_exposes_bank_artifacts() -> None:
    script_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/run_tinyllm_calibrated_kv_smoke.py"
    spec = importlib.util.spec_from_file_location("run_tinyllm_calibrated_kv_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    source = script_path.read_text(encoding="utf-8")
    assert "--calibration-prompt" in source
    assert "--calibration-prompt-extra" in source
    assert "--calibration-prompts-file" in source
    assert "--target-prompt" in source
    assert "--source-map" in source
    assert "calibration_holdout" in source
    assert "stack_calibration_kv_samples" in source
    assert "fit_multi_source_recovery_bank" in source
    assert "save_multi_source_recovery_bank" in source
    assert "make_calibrated_multi_source_recovery_callback" in source
    assert "tinyllm_calibrated_kv_summary.json" in source


def test_tinyllm_calibrated_kv_smoke_stacks_multiple_calibration_kv_samples() -> None:
    import numpy as np

    script_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/run_tinyllm_calibrated_kv_smoke.py"
    spec = importlib.util.spec_from_file_location("run_tinyllm_calibrated_kv_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    first = np.zeros((2, 1, 1, 4, 3, 2), dtype=np.float32)
    second = np.zeros((2, 1, 1, 4, 3, 2), dtype=np.float32)
    first[:, 0, 0, :3, :, :] = 1.0
    second[:, 0, 0, :2, :, :] = 2.0

    stacked = module.stack_calibration_kv_samples([(first, 3), (second, 2)], block_size=4)

    assert stacked.shape == (2, 1, 2, 4, 3, 2)
    assert np.allclose(stacked[:, 0, 0, :3, :, :], 1.0)
    assert np.allclose(stacked[:, 0, 0, 3:, :, :], 2.0)
    assert np.allclose(stacked[:, 0, 1, :1, :, :], 2.0)
    assert np.allclose(stacked[:, 0, 1, 1:, :, :], 0.0)


def test_tinyllm_calibrated_kv_smoke_copies_only_prompt_prefix_blocks() -> None:
    import numpy as np

    script_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/run_tinyllm_calibrated_kv_smoke.py"
    spec = importlib.util.spec_from_file_location("run_tinyllm_calibrated_kv_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    kv_cache = np.zeros((2, 1, 5, 4, 3, 2), dtype=np.float32)
    kv_cache[:, :, 0, :, :, :] = 1.0
    kv_cache[:, :, 1, :1, :, :] = 2.0
    kv_cache[:, :, 2:, :, :, :] = 9.0

    prefix = module.copy_kv_prompt_prefix(kv_cache, prompt_tokens=5)

    assert prefix.shape == (2, 1, 2, 4, 3, 2)
    assert np.allclose(prefix[:, :, 0, :, :, :], 1.0)
    assert np.allclose(prefix[:, :, 1, :1, :, :], 2.0)
    assert np.allclose(prefix[:, :, 1, 1:, :, :], 0.0)


def test_tinyllm_calibrated_kv_smoke_uses_full_stacked_calibration_tokens(tmp_path: Path) -> None:
    import numpy as np

    script_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/run_tinyllm_calibrated_kv_smoke.py"
    spec = importlib.util.spec_from_file_location("run_tinyllm_calibrated_kv_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    policy_path = tmp_path / "policy.json"
    policy_dir = tmp_path / "toy_b050"
    policy_dir.mkdir()
    (policy_dir / "policy_rows.csv").write_text(
        "threshold,layer,kv_head,action,selected_budget,budget_fraction,quality,direct_val_r2,fitv_val_r2,reason\n"
        "0.5,0,2,compact,2,0.5,0,,,toy\n",
        encoding="utf-8",
    )
    policy_path.write_text(
        json.dumps(
            {
                "kind": "task_adaptive_light_doc_cache_policy",
                "default_policy_dir": "toy_b050",
                "overrides": {},
            }
        ),
        encoding="utf-8",
    )
    calibration_kv = np.zeros((2, 1, 2, 4, 3, 1), dtype=np.float32)
    target_kv = np.zeros((2, 1, 1, 4, 3, 1), dtype=np.float32)
    summary = module.run_calibrated_smoke(
        calibration_kv=calibration_kv,
        target_kv=target_kv,
        calibration_tokens=7,
        target_tokens=4,
        policy_file=str(policy_path),
        repo_root=tmp_path,
        task_id="toy",
        doc_id=None,
        model="toy-model",
        source_count=1,
        recover_ridge=1e-6,
        output_dir=tmp_path / "out",
    )

    assert summary["calibration_tokens"] == 7
    assert summary["target_tokens"] == 4
    assert summary["effective_plan_tokens"] == 4
    assert summary["calibration_plan_tokens"] == 7


def test_tinyllm_calibrated_kv_smoke_selects_sources_by_calibration_prefix_fit() -> None:
    import numpy as np

    script_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/run_tinyllm_calibrated_kv_smoke.py"
    spec = importlib.util.spec_from_file_location("run_tinyllm_calibrated_kv_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    plan = build_light_doc_cache_runtime_plan(
        LightDocCacheRuntimeConfig(
            enabled=True,
            num_layers=1,
            num_kv_heads=4,
            policy=_MODULE.LightDocCachePolicy(
                kind="toy",
                default_policy_dir="toy_b050",
                default_budget_fraction=0.5,
            ),
            base_recovered_heads=[(0, 3)],
            base_budget_fraction=0.5,
        ),
        task_id="toy",
        doc_id=None,
        seq_len=8,
    )
    calibration_kv = np.zeros((2, 1, 1, 8, 4, 1), dtype=np.float32)
    token_index = np.arange(16, dtype=np.float32).reshape(2, 8, 1)
    calibration_kv[:, 0, :, :, 0, :] = (100.0 - token_index).reshape(2, 1, 8, 1)
    calibration_kv[:, 0, :, :, 1, :] = (token_index * 3.0 + 7.0).reshape(2, 1, 8, 1)
    calibration_kv[:, 0, :, :, 2, :] = (token_index * -2.0).reshape(2, 1, 8, 1)
    calibration_kv[:, 0, :, :, 3, :] = (token_index * 3.0 + 7.0).reshape(2, 1, 8, 1)

    source_heads = module.build_calibration_fit_source_heads(
        calibration_kv,
        plan,
        source_count=1,
        ridge=1e-6,
    )

    assert source_heads[(0, 3)] == [(0, 1)]


def test_tinyllm_calibrated_kv_smoke_selects_sources_by_calibration_holdout_fit() -> None:
    import numpy as np

    script_path = Path(_REPO_ROOT) / "experiments/light_doc_cache/run_tinyllm_calibrated_kv_smoke.py"
    spec = importlib.util.spec_from_file_location("run_tinyllm_calibrated_kv_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    plan = build_light_doc_cache_runtime_plan(
        LightDocCacheRuntimeConfig(
            enabled=True,
            num_layers=1,
            num_kv_heads=4,
            policy=_MODULE.LightDocCachePolicy(
                kind="toy",
                default_policy_dir="toy_b050",
                default_budget_fraction=0.5,
            ),
            base_recovered_heads=[(0, 3)],
            base_budget_fraction=0.5,
        ),
        task_id="toy",
        doc_id=None,
        seq_len=8,
    )
    calibration_kv = np.zeros((2, 1, 1, 8, 4, 1), dtype=np.float32)
    token_index = np.arange(16, dtype=np.float32).reshape(2, 8, 1)
    target = token_index * 2.0 + 3.0
    good_holdout_source = token_index * 2.0 + 3.0
    prefix_only_source = target.copy()
    prefix_only_source[:, 4:, :] = -1000.0
    calibration_kv[:, 0, :, :, 0, :] = prefix_only_source.reshape(2, 1, 8, 1)
    calibration_kv[:, 0, :, :, 1, :] = good_holdout_source.reshape(2, 1, 8, 1)
    calibration_kv[:, 0, :, :, 2, :] = (token_index * -1.0).reshape(2, 1, 8, 1)
    calibration_kv[:, 0, :, :, 3, :] = target.reshape(2, 1, 8, 1)

    source_heads = module.build_calibration_holdout_source_heads(
        calibration_kv,
        plan,
        source_count=1,
        ridge=1e-6,
    )

    assert source_heads[(0, 3)] == [(0, 1)]
