"""Run a fixed-bank Light Doc Cache read-path matrix in one TinyLLM process."""

from __future__ import annotations

import argparse
import atexit
import importlib.util
import json
from pathlib import Path
import sys
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
LIGHT_DOC_CACHE = ROOT / "experiments" / "light_doc_cache"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_REPORT = _load_module(
    "light_doc_cache_multi_target_report",
    LIGHT_DOC_CACHE / "make_multi_target_read_path_report.py",
)

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
            except Exception as exc:
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
            summary_file = mode_dir / "summary.json"
            summary_file.write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            rows.append(row)
            attempts.append(
                {
                    "target_id": target["id"],
                    "mode": mode,
                    "status": row["status"],
                    "summary_file": str(summary_file),
                }
            )
    manifest = {
        "version": 1,
        "calibration_bank_sha256": calibration_bank_sha256,
        "modes": list(MODE_CONFIGS),
        "targets": targets,
        "attempts": attempts,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows, manifest


def _collect_calibration_prompts(
    primary: str,
    extras: list[str],
    prompts_file: str,
) -> list[str]:
    prompts = [primary, *(extras or [])]
    if prompts_file:
        prompts.extend(
            line.strip()
            for line in Path(prompts_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return prompts


def _normalized_read_path_row(summary: dict[str, object]) -> dict[str, object]:
    sidecar = summary["sidecar"]
    error_metrics = sidecar["error_metrics"]
    compare = summary["logit_compare"]
    return {
        "prompt_tokens": int(summary["prompt_tokens"]),
        "logical_byte_saving_fraction": float(
            sidecar["logical_byte_saving_fraction"]
        ),
        "missing_tokens": int(error_metrics["num_missing_compact_tokens"]),
        "missing_mse": float(error_metrics["mse_missing_compact_tokens"]),
        "missing_mae": float(error_metrics["mae_missing_compact_tokens"]),
        "missing_max_abs": float(
            error_metrics["max_abs_missing_compact_tokens"]
        ),
        "max_abs_logit_diff": float(compare["max_abs_logit_diff"]),
        "mean_abs_logit_diff": float(compare["mean_abs_logit_diff"]),
        "argmax_match": bool(compare["argmax_match"]),
        "original_argmax": int(compare["original_argmax"]),
        "restored_argmax": int(compare["restored_argmax"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy-file", required=True)
    parser.add_argument("--target-file", required=True)
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument(
        "--calibration-prompt",
        default="Light Doc Cache TinyLLM calibration prompt.",
    )
    parser.add_argument("--calibration-prompt-extra", action="append", default=[])
    parser.add_argument("--calibration-prompts-file", default="")
    parser.add_argument("--task-id", default="smoothquant_status")
    parser.add_argument("--doc-id", default="second")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--source-count", type=int, default=2)
    parser.add_argument("--recover-ridge", type=float, default=1e-6)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-limit", type=int, default=0)
    args = parser.parse_args(argv)

    calibrated = _load_module(
        "light_doc_cache_calibrated_kv_smoke",
        LIGHT_DOC_CACHE / "run_tinyllm_calibrated_kv_smoke.py",
    )
    read_path = _load_module(
        "light_doc_cache_sidecar_read_path_smoke",
        LIGHT_DOC_CACHE / "run_tinyllm_sidecar_read_path_smoke.py",
    )

    from tinyvllm import LLM, SamplingParams

    output_dir = Path(args.output_dir)
    dataset = _REPORT.load_target_dataset(args.target_file)
    targets = dataset["targets"]
    if args.target_limit > 0:
        targets = targets[: args.target_limit]

    llm = LLM(
        args.model,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    try:
        sampling_params = SamplingParams(max_tokens=1, ignore_eos=True)
        calibration_samples = [
            calibrated._run_prompt_and_copy_kv(llm, prompt, sampling_params)
            for prompt in _collect_calibration_prompts(
                args.calibration_prompt,
                args.calibration_prompt_extra,
                args.calibration_prompts_file,
            )
        ]
        calibration_kv = calibrated.stack_calibration_kv_samples(
            calibration_samples,
            block_size=int(llm.model_runner.kv_cache.shape[3]),
        )
        calibration_tokens = sum(
            int(tokens) for _, tokens in calibration_samples
        )
        bank, calibration_plan, source_heads = (
            calibrated.fit_calibration_recovery_bank(
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
        )
        calibration_dir = output_dir / "calibration"
        calibration_dir.mkdir(parents=True, exist_ok=True)
        bank_path = calibration_dir / "multi_source_recovery_bank.json"
        calibrated._RUNTIME.save_multi_source_recovery_bank(bank, bank_path)
        bank_sha256 = _REPORT.hashlib_sha256_file(bank_path)
        calibration_summary = {
            "calibration_tokens": calibration_tokens,
            "calibration_plan": calibration_plan.as_summary(),
            "source_heads": {
                f"{target[0]}:{target[1]}": [
                    [int(layer), int(head)] for layer, head in sources
                ]
                for target, sources in sorted(source_heads.items())
            },
            "recovery_bank_file": str(bank_path),
            "calibration_bank_sha256": bank_sha256,
        }
        (calibration_dir / "summary.json").write_text(
            json.dumps(calibration_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        def run_mode(
            *,
            target: dict[str, str],
            mode: str,
            output_dir: Path,
        ) -> dict[str, object]:
            config = MODE_CONFIGS[mode]
            summary = read_path.run_read_path_smoke(
                llm=llm,
                policy_file=args.policy_file,
                repo_root=args.repo_root,
                prompt=target["prompt"],
                task_id=args.task_id,
                doc_id=args.doc_id,
                output_dir=output_dir,
                model=args.model,
                recover_mode=config["recover_mode"],
                recover_ridge=args.recover_ridge,
                correlated_source_map=config.get(
                    "correlated_source_map", "same_layer"
                ),
                multi_correlated_source_count=args.source_count,
                recovery_bank_file=(
                    str(bank_path) if mode == "calibration_holdout" else None
                ),
            )
            return _normalized_read_path_row(summary)

        rows, _ = run_target_matrix(
            targets=targets,
            output_dir=output_dir,
            calibration_bank_sha256=bank_sha256,
            run_mode=run_mode,
        )
        summary = _REPORT.aggregate_rows(rows)
        summary["dataset_version"] = int(dataset["version"])
        summary["calibration"] = calibration_summary
        _REPORT.write_outputs(output_dir, rows, summary)
    finally:
        try:
            atexit.unregister(llm.exit)
        except Exception:
            pass
        llm.exit()

    print(
        json.dumps(
            {
                "decision": summary["gate"]["decision"],
                "output_dir": str(output_dir),
                "paired_targets": summary["gate"]["paired_targets"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
