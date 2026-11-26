from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import sys
import time


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _select_idle_a100() -> dict:
    import subprocess

    gpu_csv = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    apps_csv = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    busy = {line.split(",")[0].strip() for line in apps_csv.splitlines() if line.strip()}
    for line in gpu_csv.splitlines():
        cols = [seg.strip() for seg in line.split(",")]
        if len(cols) != 5:
            continue
        idx_s, uuid_s, name_s, free_s, used_s = cols
        if "A100" in name_s and uuid_s not in busy and int(free_s) >= 65536 and int(used_s) <= 512:
            return {"index": int(idx_s), "name": name_s, "free_mib": int(free_s), "used_mib": int(used_s)}
    raise RuntimeError("No idle A100 80GB GPU found")


def _chat_ids(tokenizer, messages: list[dict]) -> list[int]:
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        truncation=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(ids, "get") and "input_ids" in ids:
        ids = ids["input_ids"]
    return [int(x) for x in ids]


def _evaluate_python_candidate_locally_safe(source_code: str, entry_point: str, cases: list[dict]) -> dict:
    namespace: dict = {}
    try:
        exec(compile(source_code, "<candidate>", "exec"), namespace)
    except Exception as exc:
        return {"valid": False, "passed": 0, "total": len(cases), "error": type(exc).__name__}
    fn_obj = namespace.get(entry_point)
    if not callable(fn_obj):
        return {"valid": False, "passed": 0, "total": len(cases), "error": "missing_entry"}

    passed = 0
    for case in cases:
        expected = case["expected"]
        try:
            import copy
            val = fn_obj(*copy.deepcopy(case["args"]))
            if expected["kind"] == "return" and val == expected["value"] and type(val) is type(expected["value"]):
                passed += 1
        except ValueError:
            if expected["kind"] == "raises" and expected.get("type") == "ValueError":
                passed += 1
        except Exception:
            pass
    return {"valid": True, "passed": passed, "total": len(cases), "all_passed": passed == len(cases)}


def _drain_engine_to_completion(engine, seq_order: list[int], tokenizer) -> dict[int, dict]:
    outputs_by_sid: dict[int, dict] = {}
    step_count = 0
    eos_id = getattr(engine.scheduler, "eos", getattr(tokenizer, "eos_token_id", None))
    t0 = time.perf_counter_ns()
    while not engine.is_finished():
        finished_pairs, num_toks = engine.step(completion_only=True)
        step_count += 1
        for sid, comp_ids in finished_pairs:
            decode_ids = list(comp_ids[:-1]) if (comp_ids and comp_ids[-1] == eos_id) else list(comp_ids)
            decoded = tokenizer.decode(
                decode_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            outputs_by_sid[int(sid)] = {
                "completion_ids": list(comp_ids),
                "completion_tokens": len(comp_ids),
                "text": decoded,
            }
    elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
    for sid in seq_order:
        if sid not in outputs_by_sid:
            raise RuntimeError(f"Sequence {sid} missing from completed outputs")
    return {
        "by_seq_id": outputs_by_sid,
        "steps": step_count,
        "elapsed_ms": elapsed_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--g8-adapter-path", required=True)
    parser.add_argument("--g8-expected-sha256", required=True)
    parser.add_argument("--g9-adapter-path", required=True)
    parser.add_argument("--g9-expected-sha256", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    selected_gpu = _select_idle_a100()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu["index"])

    import torch

    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "experiments/lora_necessity"))

    g8_raw = Path(args.g8_adapter_path).read_bytes()
    g9_raw = Path(args.g9_adapter_path).read_bytes()
    if _sha256_bytes(g8_raw) != args.g8_expected_sha256:
        raise ValueError("G8 adapter SHA256 mismatch")
    if _sha256_bytes(g9_raw) != args.g9_expected_sha256:
        raise ValueError("G9 adapter SHA256 mismatch")

    g8_tensors = torch.load(io.BytesIO(g8_raw), map_location="cpu", weights_only=True)["adapter"]
    g9_tensors = torch.load(io.BytesIO(g9_raw), map_location="cpu", weights_only=True)["adapter"]

    conf_dir = repo_root / "experiments/lora_necessity/data/confirmatory_v2_r1"
    supervision_raw = (conf_dir / "supervision.json").read_bytes()
    prompts = json.loads((conf_dir / "prompts.json").read_text("utf-8"))
    oracles = {item["id"]: item for item in json.loads((conf_dir / "oracles.json").read_text("utf-8"))}
    prompts_by_id = {item["id"]: item for item in prompts}

    from confirmatory_v2_contract import confirmatory_first_round
    from tinyvllm.engine.llm_engine import LLMEngine
    from tinyvllm.sampling_params import SamplingParams

    engine = LLMEngine(
        args.model_path,
        max_num_batched_tokens=65536,
        max_num_seqs=64,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        enforce_eager=False,
        prefix_lora_max_slots=4,
        prefix_lora_rank=8,
    )
    try:
        tokenizer = engine.tokenizer
        cuda_graphs_captured = sorted(int(bs) for bs in getattr(engine.model_runner, "graphs", {}).keys())
        has_lora_graph_var = "lora_token_slot_ids" in getattr(engine.model_runner, "graph_vars", {})

        sample_task = prompts_by_id["WT05"]
        long_msgs = confirmatory_first_round(sample_task, "text", supervision_raw)
        short_msgs = confirmatory_first_round(sample_task, "lora", supervision_raw)
        long_ids = _chat_ids(tokenizer, long_msgs)
        short_ids = _chat_ids(tokenizer, short_msgs)

        suffix_len = 0
        while (
            suffix_len < min(len(long_ids), len(short_ids))
            and long_ids[len(long_ids) - 1 - suffix_len] == short_ids[len(short_ids) - 1 - suffix_len]
        ):
            suffix_len += 1
        if suffix_len < 50:
            raise RuntimeError("Unexpected short common task suffix between text and lora prompts")

        long_prefix_ids = long_ids[:len(long_ids) - suffix_len]
        short_prefix_ids = short_ids[:len(short_ids) - suffix_len]

        slot1_receipt = engine.register_promoted_prefix_lora(
            1,
            g8_tensors,
            long_prefix_ids=long_prefix_ids,
            short_prefix_ids=short_prefix_ids,
            alpha=16.0,
            rank=8,
            name="lora_diff_kl_v2",
        )
        slot2_receipt = engine.register_promoted_prefix_lora(
            2,
            g9_tensors,
            long_prefix_ids=None,
            alpha=16.0,
            rank=8,
            name="lora_csd_v2",
        )

        test_specs = [
            ("WR03", "base", 0),
            ("WR04", "base", 0),
            ("WT05", "text_auto_promoted", 0),
            ("WT06", "text_auto_promoted", 0),
            ("WT01", "explicit_slot2", 2),
            ("WR11", "explicit_slot2", 2),
        ]

        single_outputs: dict[str, str] = {}
        single_Rewrite_meta: dict[str, dict] = {}
        for tid, mode_tag, exp_slot in test_specs:
            t_obj = prompts_by_id[tid]
            if mode_tag == "text_auto_promoted":
                raw_ids = _chat_ids(tokenizer, confirmatory_first_round(t_obj, "text", supervision_raw))
            else:
                raw_ids = _chat_ids(tokenizer, confirmatory_first_round(t_obj, "base", supervision_raw))
            sid = engine.add_request(
                raw_ids,
                SamplingParams(temperature=0.0, max_tokens=256, lora_slot_id=exp_slot),
            )
            queued_seq = engine.scheduler.waiting[-1]
            single_Rewrite_meta[tid] = {
                "mode": mode_tag,
                "original_prompt_tokens": len(raw_ids),
                "rewritten_prompt_tokens": queued_seq.num_prompt_tokens,
                "resolved_lora_slot_id": queued_seq.lora_slot_id,
                "paged_kv_blocks_needed": queued_seq.num_blocks,
            }
            drained = _drain_engine_to_completion(engine, [sid], tokenizer)
            single_outputs[tid] = drained["by_seq_id"][sid]["text"]

        mixed_sids: list[int] = []
        sid_to_tid: dict[int, str] = {}
        for tid, mode_tag, exp_slot in test_specs:
            t_obj = prompts_by_id[tid]
            if mode_tag == "text_auto_promoted":
                raw_ids = _chat_ids(tokenizer, confirmatory_first_round(t_obj, "text", supervision_raw))
            else:
                raw_ids = _chat_ids(tokenizer, confirmatory_first_round(t_obj, "base", supervision_raw))
            sid = engine.add_request(
                raw_ids,
                SamplingParams(temperature=0.0, max_tokens=256, lora_slot_id=exp_slot),
            )
            mixed_sids.append(sid)
            sid_to_tid[sid] = tid

        mixed_drained = _drain_engine_to_completion(engine, mixed_sids, tokenizer)
        mixed_verification: dict[str, dict] = {}
        for sid in mixed_sids:
            tid = sid_to_tid[sid]
            cand_txt = mixed_drained["by_seq_id"][sid]["text"]
            eval_res = _evaluate_python_candidate_locally_safe(
                cand_txt, oracles[tid]["entry_point"], oracles[tid]["cases"]
            )
            mixed_verification[tid] = {
                **single_Rewrite_meta[tid],
                "matches_single_request_cudagraph_run": cand_txt == single_outputs[tid],
                "oracle_eval": eval_res,
            }

        bench_tasks = [prompts_by_id[f"WT{idx:02d}"] for idx in range(1, 13)] + [
            prompts_by_id[f"WR{idx:02d}"] for idx in range(1, 5)
        ]

        def bench_promoted_batch16() -> dict:
            times_ms = []
            total_out_toks = 0
            peak_used_blocks = 0
            bm = engine.scheduler.block_manager
            total_blocks = len(bm.blocks)
            for trial in range(3):
                sids = []
                for t_obj in bench_tasks:
                    raw_long = _chat_ids(
                        tokenizer, confirmatory_first_round(t_obj, "text", supervision_raw)
                    )
                    sids.append(
                        engine.add_request(
                            raw_long, SamplingParams(temperature=0.0, max_tokens=128)
                        )
                    )
                t_start = time.perf_counter_ns()
                out_toks_trial = 0
                max_used = 0
                while not engine.is_finished():
                    fin, _ = engine.step(completion_only=True)
                    used_now = total_blocks - len(bm.free_block_ids)
                    if used_now > max_used:
                        max_used = used_now
                    for _s, c_ids in fin:
                        out_toks_trial += len(c_ids)
                el_ms = (time.perf_counter_ns() - t_start) / 1_000_000.0
                if trial >= 1:
                    times_ms.append(el_ms)
                    total_out_toks = out_toks_trial
                    peak_used_blocks = max(peak_used_blocks, max_used)
            med_ms = statistics.median(times_ms)
            return {
                "batch_requests": len(bench_tasks),
                "median_elapsed_ms": med_ms,
                "output_tokens": total_out_toks,
                "output_tokens_per_sec": total_out_toks / (med_ms / 1000.0),
                "peak_allocated_kv_blocks": peak_used_blocks,
                "peak_allocated_kv_mib": (peak_used_blocks * 256 * 36 * 2 * 8 * 128 * 2) / (1024.0 * 1024.0),
            }

        promoted_bench = bench_promoted_batch16()
        engine.clear_promoted_prefix_lora(1)

        def bench_native_apc_long_batch16() -> dict:
            bm = engine.scheduler.block_manager
            total_blocks = len(bm.blocks)
            warm_raw = _chat_ids(
                tokenizer, confirmatory_first_round(bench_tasks[0], "text", supervision_raw)
            )
            w_sid = engine.add_request(warm_raw, SamplingParams(temperature=0.0, max_tokens=4))
            _drain_engine_to_completion(engine, [w_sid], tokenizer)

            times_ms = []
            total_out_toks = 0
            peak_used_blocks = 0
            for trial in range(3):
                sids = []
                for t_obj in bench_tasks:
                    raw_long = _chat_ids(
                        tokenizer, confirmatory_first_round(t_obj, "text", supervision_raw)
                    )
                    sids.append(
                        engine.add_request(
                            raw_long, SamplingParams(temperature=0.0, max_tokens=128)
                        )
                    )
                t_start = time.perf_counter_ns()
                out_toks_trial = 0
                max_used = 0
                while not engine.is_finished():
                    fin, _ = engine.step(completion_only=True)
                    used_now = total_blocks - len(bm.free_block_ids)
                    if used_now > max_used:
                        max_used = used_now
                    for _s, c_ids in fin:
                        out_toks_trial += len(c_ids)
                el_ms = (time.perf_counter_ns() - t_start) / 1_000_000.0
                if trial >= 1:
                    times_ms.append(el_ms)
                    total_out_toks = out_toks_trial
                    peak_used_blocks = max(peak_used_blocks, max_used)
            med_ms = statistics.median(times_ms)
            return {
                "batch_requests": len(bench_tasks),
                "median_elapsed_ms": med_ms,
                "output_tokens": total_out_toks,
                "output_tokens_per_sec": total_out_toks / (med_ms / 1000.0),
                "peak_allocated_kv_blocks": peak_used_blocks,
                "peak_allocated_kv_mib": (peak_used_blocks * 256 * 36 * 2 * 8 * 128 * 2) / (1024.0 * 1024.0),
            }

        native_apc_bench = bench_native_apc_long_batch16()

        report = {
            "kind": "TINYVLLM_NATIVE_PREFIX_LORA_E2E_REPORT",
            "gpu_summary": {
                "name": selected_gpu["name"],
                "free_mib": selected_gpu["free_mib"],
                "used_mib": selected_gpu["used_mib"],
            },
            "cuda_graphs_captured_batch_sizes": cuda_graphs_captured,
            "cuda_graph_has_lora_slot_buffer": has_lora_graph_var,
            "long_prefix_tokens_registered": len(long_prefix_ids),
            "short_prefix_tokens_replacement": len(short_prefix_ids),
            "slot_receipts": {"slot_1": slot1_receipt, "slot_2": slot2_receipt},
            "mixed_batch_6_requests_verification": mixed_verification,
            "all_6_mixed_tasks_passed_all_8_cases": all(
                v["oracle_eval"]["all_passed"] for v in mixed_verification.values()
            ),
            "all_6_mixed_tasks_matched_cudagraph_single_run": all(
                v["matches_single_request_cudagraph_run"] for v in mixed_verification.values()
            ),
            "tinyvllm_paged_kv_batch16_comparison": {
                "native_warm_prefix_cache_apc": native_apc_bench,
                "promoted_prefix_to_bf16_lora": promoted_bench,
                "throughput_speedup": promoted_bench["output_tokens_per_sec"]
                / native_apc_bench["output_tokens_per_sec"],
                "kv_blocks_saved_ratio": 1.0
                - (
                    promoted_bench["peak_allocated_kv_blocks"]
                    / float(max(native_apc_bench["peak_allocated_kv_blocks"], 1))
                ),
            },
        }
        Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
