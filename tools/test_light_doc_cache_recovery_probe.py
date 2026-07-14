from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIGHT_DOC_CACHE = ROOT / "experiments" / "light_doc_cache"
sys.path.insert(0, str(LIGHT_DOC_CACHE))

try:
    import torch
except ModuleNotFoundError:
    torch = None


def test_train_recovery_probe_static_contract():
    script = LIGHT_DOC_CACHE / "train_recovery_probe.py"
    text = script.read_text(encoding="utf-8")
    for needle in [
        "def train_head_recovery(",
        "def summarize_recovery(",
        "def write_outputs(",
        "recovery_head_rows.csv",
        "summary.json",
        "report.md",
        "RECOVERY_PROMISING",
        "mlp_residual",
        "learned_compact_values",
        "fused_residual",
        "torch.cat",
        "--start-layer",
        "--end-layer",
    ]:
        assert needle in text


def test_summarize_recovery_sweep_collects_and_sorts_runs(tmp_path):
    script = LIGHT_DOC_CACHE / "summarize_recovery_sweep.py"
    text = script.read_text(encoding="utf-8")
    assert "def collect_summaries(" in text
    assert "def write_sweep_outputs(" in text

    sys.path.insert(0, str(LIGHT_DOC_CACHE))
    import summarize_recovery_sweep as sweep  # noqa: E402

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "summary.json").write_text(
        json.dumps(
            {
                "decision": "RECOVERY_WEAK",
                "heads": 16,
                "mean_budget_fraction": 0.125,
                "mean_direct_val_r2": -0.6,
                "mean_fitv_val_r2": 0.08,
                "mean_recovery_val_r2": 0.08,
                "settings": {"start_layer": 10, "budgets": "64"},
            }
        ),
        encoding="utf-8",
    )
    (run_b / "summary.json").write_text(
        json.dumps(
            {
                "decision": "RECOVERY_PROMISING",
                "heads": 16,
                "mean_budget_fraction": 0.25,
                "mean_direct_val_r2": 0.1,
                "mean_fitv_val_r2": 0.4,
                "mean_recovery_val_r2": 0.55,
                "settings": {"start_layer": 16, "budgets": "128"},
            }
        ),
        encoding="utf-8",
    )

    rows = sweep.collect_summaries([tmp_path])
    assert [row["run_name"] for row in rows] == ["run_b", "run_a"]
    assert rows[0]["start_layer"] == 16
    assert rows[0]["mean_recovery_val_r2"] == 0.55

    out = tmp_path / "out"
    sweep.write_sweep_outputs(out, rows)
    assert (out / "recovery_sweep_summary.csv").read_text(encoding="utf-8").startswith("run_name,decision")
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "run_b" in report
    assert "0.5500" in report


def test_task_quality_smoke_supports_learned_value_banks():
    script = LIGHT_DOC_CACHE / "task_quality_smoke.py"
    text = script.read_text(encoding="utf-8")
    for needle in [
        "--bank-method",
        "learned_values",
        "train_compact_values",
        "--bank-train-epochs",
        "--bank-train-lr",
    ]:
        assert needle in text


def test_task_quality_smoke_supports_adaptive_policy_overrides(tmp_path):
    script = LIGHT_DOC_CACHE / "task_quality_smoke.py"
    text = script.read_text(encoding="utf-8")
    for needle in [
        "--adaptive-policy-file",
        "def load_adaptive_policy(",
        "def apply_adaptive_policy(",
        "adaptive_policy_file",
        "effective_entry_saving_fraction",
    ]:
        assert needle in text

    if torch is None:
        return

    sys.path.insert(0, str(LIGHT_DOC_CACHE))
    import task_quality_smoke as smoke  # noqa: E402

    policy = {
        "default_policy_dir": "base_policy",
        "overrides": {
            "fragile_task": {
                "drop_heads": ["1:2", "3:4"],
                "full_heads": ["5:6"],
            }
        },
    }
    policy_file = tmp_path / "adaptive_policy.json"
    policy_file.write_text(json.dumps(policy), encoding="utf-8")

    loaded = smoke.load_adaptive_policy(policy_file)
    assert loaded["default_policy_dir"] == "base_policy"

    heads = [
        {"layer": 1, "kv_head": 2, "budget": 768, "quality": 0.0},
        {"layer": 3, "kv_head": 4, "budget": 768, "quality": 0.0},
        {"layer": 5, "kv_head": 6, "budget": 768, "quality": 0.0},
        {"layer": 7, "kv_head": 0, "budget": 768, "quality": 0.0},
    ]
    filtered = smoke.apply_adaptive_policy(heads, loaded, "fragile_task")
    assert [(row["layer"], row["kv_head"]) for row in filtered] == [(7, 0)]

    unchanged = smoke.apply_adaptive_policy(heads, loaded, "safe_task")
    assert unchanged == heads


def test_make_recovery_task_policy_static_contract():
    script = LIGHT_DOC_CACHE / "make_recovery_task_policy.py"
    text = script.read_text(encoding="utf-8")
    for needle in [
        "--compact-layer",
        "all",
        "--compact-layer-range",
        "--compact-heads",
        "--budget-fraction",
        "policy_rows.csv",
        "policy_summary.csv",
        "layer_recovery_task_policy",
    ]:
        assert needle in text


def test_task_quality_remote_runner_syncs_local_policy_dir():
    script = LIGHT_DOC_CACHE / "run_task_quality_smoke_remote.sh"
    text = script.read_text(encoding="utf-8")
    for needle in [
        "LOCAL_POLICY_DIR",
        "LOCAL_ADAPTIVE_POLICY_FILE",
        "adaptive_policy.json",
        "adaptive_policy_sha",
        "task_quality_policy",
        "policy_rows_sha",
        "remote_policy_dir",
    ]:
        assert needle in text


def test_make_head_addition_policies_static_contract():
    script = LIGHT_DOC_CACHE / "make_head_addition_policies.py"
    text = script.read_text(encoding="utf-8")
    for needle in [
        "--seed-heads",
        "--candidate-heads",
        "add_l",
        "manifest.json",
        "make_recovery_task_policy.py",
    ]:
        assert needle in text


def test_make_adaptive_task_policy_generates_fragile_task_overrides(tmp_path):
    script = LIGHT_DOC_CACHE / "make_adaptive_task_policy.py"
    text = script.read_text(encoding="utf-8")
    for needle in [
        "--failure-diagnostics",
        "--default-policy-dir",
        "--drop-heads",
        "--top-tasks",
        "task_adaptive_light_doc_cache_policy",
    ]:
        assert needle in text

    diagnostics = tmp_path / "failures.tsv"
    diagnostics.write_text(
        "\n".join(
            [
                "candidate\tstatus\tdoc\ttask_id\tbaseline_pred\tcompact_pred\tanswer",
                "6:6\tfirst_fail\tfirst\ttopk8_quality\tB\tA\tB",
                "5:5\tfirst_fail\tfirst\ttopk8_quality\tB\tA\tB",
                "15:4\tfirst_fail\tfirst\tsweet_spot\tB\tA\tB",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "adaptive.json"
    import subprocess

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--failure-diagnostics",
            str(diagnostics),
            "--default-policy-dir",
            "default_policy",
            "--base-safe-policy-dir",
            "base_policy",
            "--drop-heads",
            "15:5",
            "--top-tasks",
            "2",
            "--output",
            str(output),
        ],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["kind"] == "task_adaptive_light_doc_cache_policy"
    assert payload["default_policy_dir"] == "default_policy"
    assert payload["base_safe_policy_dir"] == "base_policy"
    assert set(payload["overrides"]) == {"topk8_quality", "sweet_spot"}
    assert payload["overrides"]["topk8_quality"]["drop_heads"] == ["15:5"]


def test_make_adaptive_task_policy_combines_per_doc_failure_sources(tmp_path):
    script = LIGHT_DOC_CACHE / "make_adaptive_task_policy.py"
    diagnostics = tmp_path / "round_failures.tsv"
    diagnostics.write_text(
        "\n".join(
            [
                "candidate\tstatus\tdoc\ttask_id\tbaseline_pred\tcompact_pred\tanswer",
                "6:6\tfirst_fail\tfirst\ttopk8_quality\tB\tA\tB",
                "5:5\tfirst_fail\tfirst\ttopk8_quality\tB\tA\tB",
                "15:4\tfirst_fail\tfirst\tsweet_spot\tB\tA\tB",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    adaptive_results = tmp_path / "adaptive_results.tsv"
    adaptive_results.write_text(
        "\n".join(
            [
                "candidate\tdoc\tpass_gate\tfail_tasks",
                "6h6\tsecond\tFalse\ttp_true_weight_split,gpu_utilization_semantics,smoothquant_status",
                "5h5\tsecond\tFalse\tsmoothquant_status",
                "22h6\tsecond\tTrue\t",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "adaptive_two_doc.json"
    import subprocess

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--failure-diagnostics",
            str(diagnostics),
            "--failure-diagnostics",
            str(adaptive_results),
            "--default-policy-dir",
            "default_policy",
            "--base-safe-policy-dir",
            "base_policy",
            "--drop-heads",
            "6:6",
            "--per-doc-top-tasks",
            "first=2,second=2",
            "--output",
            str(output),
        ],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["failure_diagnostics"] == [str(diagnostics), str(adaptive_results)]
    assert payload["doc_top_tasks"] == {"first": 2, "second": 2}
    assert set(payload["overrides"]) == {
        "topk8_quality",
        "sweet_spot",
        "smoothquant_status",
        "tp_true_weight_split",
    }
    assert payload["overrides"]["smoothquant_status"]["reason"] == "doc=second fragile_task_count=2"


def test_make_adaptive_frontier_table_writes_paper_ready_outputs(tmp_path):
    script = LIGHT_DOC_CACHE / "make_adaptive_frontier_table.py"
    first_summary = tmp_path / "first_summary.json"
    second_summary = tmp_path / "second_summary.json"
    first_summary.write_text(
        json.dumps(
            {
                "summary": [
                    {
                        "tasks": 13,
                        "compressed_heads": 80,
                        "compact_accuracy": 1.0,
                        "agreement": 1.0,
                        "mean_answer_score_delta": 0.75,
                        "mean_bank_build_s": 0.29,
                        "effective_entry_saving_fraction": 0.1771,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    second_summary.write_text(
        json.dumps(
            {
                "summary": [
                    {
                        "tasks": 6,
                        "compressed_heads": 80,
                        "compact_accuracy": 1.0,
                        "agreement": 1.0,
                        "mean_answer_score_delta": 1.63,
                        "mean_bank_build_s": 0.30,
                        "effective_entry_saving_fraction": 0.1769,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "frontier"
    import subprocess

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--entry",
            f"name=two_doc_adaptive,kind=adaptive,heads=80,first={first_summary},second={second_summary},fallback_tasks=7,claim=quality_only",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    csv_text = (output_dir / "frontier_table.csv").read_text(encoding="utf-8")
    assert "two_doc_adaptive,adaptive,80,13/13,6/6,17.71%,17.69%" in csv_text
    report = (output_dir / "frontier_table.md").read_text(encoding="utf-8")
    assert "quality-only" in report
    assert "two_doc_adaptive" in report


if torch is not None:
    import train_recovery_probe as trp  # noqa: E402


def test_train_recovery_module_improves_over_direct_compact_output():
    if torch is None:
        return
    torch.manual_seed(0)
    q_group = torch.randn(2, 48, 8)
    keys = torch.randn(48, 8)
    values = torch.randn(48, 8)
    positions = torch.arange(48)
    train_idx = torch.arange(0, 32)
    val_idx = torch.arange(32, 48)
    selected = torch.arange(0, 48, 4)
    teacher = trp.attention_output(q_group, keys, values, positions, positions)

    result = trp.train_head_recovery(
        q_group=q_group,
        keys=keys,
        values=values,
        positions=positions,
        train_idx=train_idx,
        val_idx=val_idx,
        selected=selected,
        teacher_all=teacher,
        hidden_dim=16,
        epochs=80,
        lr=0.05,
        weight_decay=0.0,
        device=torch.device("cpu"),
    )

    assert result["method"] == "mlp_residual"
    assert result["budget"] == int(selected.numel())
    assert result["budget_fraction"] == selected.numel() / keys.shape[0]
    assert math.isfinite(result["direct_val_r2"])
    assert math.isfinite(result["recovery_val_r2"])
    assert result["recovery_train_r2"] > result["direct_train_r2"]
    assert result["recovery_val_r2"] >= result["direct_val_r2"] - 1e-5
    assert result["train_loss_final"] < result["train_loss_initial"]


def test_summarize_and_write_outputs(tmp_path):
    if torch is None:
        return
    rows = [
        {
            "layer": 0,
            "kv_head": 0,
            "method": "mlp_residual",
            "budget": 12,
            "budget_fraction": 0.25,
            "direct_val_r2": 0.10,
            "fitv_val_r2": 0.40,
            "recovery_val_r2": 0.55,
        },
        {
            "layer": 0,
            "kv_head": 1,
            "method": "mlp_residual",
            "budget": 12,
            "budget_fraction": 0.25,
            "direct_val_r2": 0.20,
            "fitv_val_r2": 0.45,
            "recovery_val_r2": 0.65,
        },
    ]

    summary = trp.summarize_recovery(rows, sampled_tokens=48)
    assert summary["decision"] == "RECOVERY_PROMISING"
    assert summary["heads"] == 2
    assert summary["mean_budget_fraction"] == 0.25
    assert summary["mean_recovery_val_r2"] > summary["mean_fitv_val_r2"]
    assert summary["mean_recovery_gain_vs_direct"] > 0

    trp.write_outputs(
        output_dir=tmp_path,
        rows=rows,
        summary=summary,
        metadata={"sampled_tokens": 48, "num_layers": 1, "num_kv_heads": 2},
        settings={"budgets": "12"},
        train_seconds=1.25,
    )

    assert (tmp_path / "recovery_head_rows.csv").read_text(encoding="utf-8").startswith("layer,kv_head")
    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["decision"] == "RECOVERY_PROMISING"
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "# Trainable Light Doc Cache Recovery Probe" in report
    assert "RECOVERY_PROMISING" in report


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


if __name__ == "__main__":
    test_train_recovery_probe_static_contract()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_summarize_recovery_sweep_collects_and_sorts_runs(Path(tmp))
    test_task_quality_smoke_supports_learned_value_banks()
    with tempfile.TemporaryDirectory() as tmp:
        test_task_quality_smoke_supports_adaptive_policy_overrides(Path(tmp))
    test_make_recovery_task_policy_static_contract()
    test_task_quality_remote_runner_syncs_local_policy_dir()
    test_make_head_addition_policies_static_contract()
    with tempfile.TemporaryDirectory() as tmp:
        test_make_adaptive_task_policy_generates_fragile_task_overrides(Path(tmp))
    if torch is not None:
        test_train_recovery_module_improves_over_direct_compact_output()

        with tempfile.TemporaryDirectory() as tmp:
            test_summarize_and_write_outputs(Path(tmp))
