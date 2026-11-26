from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _select_idle_a100() -> dict:
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
    busy = {ln.split(",")[0].strip() for ln in apps_csv.splitlines() if ln.strip()}
    for ln in gpu_csv.splitlines():
        cols = [seg.strip() for seg in ln.split(",")]
        if len(cols) != 5:
            continue
        idx_s, uuid_s, name_s, free_s, used_s = cols
        if "A100" in name_s and uuid_s not in busy and int(free_s) >= 65536 and int(used_s) <= 512:
            return {
                "index": int(idx_s),
                "name": name_s,
                "free_mib": int(free_s),
                "used_mib": int(used_s),
            }
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


def _extract_prefix_pair(long_ids: list[int], short_ids: list[int]) -> tuple[list[int], list[int]]:
    suffix_len = 0
    while (
        suffix_len < min(len(long_ids), len(short_ids))
        and long_ids[len(long_ids) - 1 - suffix_len] == short_ids[len(short_ids) - 1 - suffix_len]
    ):
        suffix_len += 1
    if suffix_len < 40:
        raise RuntimeError(f"Common suffix too short ({suffix_len} tokens)")
    return long_ids[: len(long_ids) - suffix_len], short_ids[: len(short_ids) - suffix_len]


def _drain_engine(engine, seq_ids: list[int], tokenizer) -> dict:
    eos_id = getattr(engine.scheduler, "eos", getattr(tokenizer, "eos_token_id", None))
    by_sid: dict[int, dict] = {}
    t0 = time.perf_counter_ns()
    while not engine.is_finished():
        fin, _ = engine.step(completion_only=True)
        for sid, comp_ids in fin:
            dec_ids = list(comp_ids[:-1]) if (comp_ids and comp_ids[-1] == eos_id) else list(comp_ids)
            txt = tokenizer.decode(
                dec_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            by_sid[int(sid)] = {
                "completion_tokens": len(comp_ids),
                "text": txt,
            }
    elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
    for sid in seq_ids:
        if sid not in by_sid:
            raise RuntimeError(f"Missing completed sequence {sid}")
    return {"by_seq_id": by_sid, "elapsed_ms": elapsed_ms}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-adapter-path", required=True)
    parser.add_argument("--code-expected-sha256", required=True)
    parser.add_argument("--med-adapter-path", required=True)
    parser.add_argument("--med-expected-sha256", required=True)
    parser.add_argument("--legal-adapter-path", required=True)
    parser.add_argument("--legal-expected-sha256", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    selected_gpu = _select_idle_a100()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu["index"])

    import torch

    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "experiments/lora_necessity"))

    def _load_verified(p_str: str, exp_sha: str) -> dict:
        raw = Path(p_str).read_bytes()
        if _sha256_bytes(raw) != exp_sha:
            raise ValueError(f"Adapter SHA256 mismatch for {p_str}")
        return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)["adapter"]

    code_tensors = _load_verified(args.code_adapter_path, args.code_expected_sha256)
    med_tensors = _load_verified(args.med_adapter_path, args.med_expected_sha256)
    legal_tensors = _load_verified(args.legal_adapter_path, args.legal_expected_sha256)

    from confirmatory_v2_contract import TRANSFER_IDS, confirmatory_first_round
    from experiments.cross_domain_prefix_lora.domain_sop_contracts import (
        render_domain_long_messages,
        render_domain_short_messages,
    )
    from tinyvllm.engine.llm_engine import LLMEngine
    from tinyvllm.sampling_params import SamplingParams

    pub_bundle = json.loads(
        (repo_root / "experiments/cross_domain_prefix_lora/data/cross_domain_v1/public_bundle.json").read_text("utf-8")
    )
    conf_dir = repo_root / "experiments/lora_necessity/data/confirmatory_v2_r1"
    code_prompts = [
        x
        for x in json.loads((conf_dir / "prompts.json").read_text("utf-8"))
        if x["id"] in TRANSFER_IDS
    ]
    code_sup_raw = (conf_dir / "supervision.json").read_bytes()

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
        graphs_captured = sorted(int(bs) for bs in getattr(engine.model_runner, "graphs", {}).keys())

        code_long_p, code_short_p = _extract_prefix_pair(
            _chat_ids(tokenizer, confirmatory_first_round(code_prompts[0], "text", code_sup_raw)),
            _chat_ids(tokenizer, confirmatory_first_round(code_prompts[0], "lora", code_sup_raw)),
        )
        med_sample = pub_bundle["med_triage"]["blind_test_inputs"][0]
        med_long_p, med_short_p = _extract_prefix_pair(
            _chat_ids(tokenizer, render_domain_long_messages("med_triage", med_sample, pub_bundle)),
            _chat_ids(tokenizer, render_domain_short_messages("med_triage", med_sample)),
        )
        leg_sample = pub_bundle["legal_compliance"]["blind_test_inputs"][0]
        leg_long_p, leg_short_p = _extract_prefix_pair(
            _chat_ids(tokenizer, render_domain_long_messages("legal_compliance", leg_sample, pub_bundle)),
            _chat_ids(tokenizer, render_domain_short_messages("legal_compliance", leg_sample)),
        )

        r_slot1 = engine.register_promoted_prefix_lora(
            1,
            code_tensors,
            long_prefix_ids=code_long_p,
            short_prefix_ids=code_short_p,
            alpha=16.0,
            rank=8,
            name="code_hybrid_csd_g12",
        )
        r_slot2 = engine.register_promoted_prefix_lora(
            2,
            med_tensors,
            long_prefix_ids=med_long_p,
            short_prefix_ids=med_short_p,
            alpha=16.0,
            rank=8,
            name="med_sop_v4",
        )
        r_slot3 = engine.register_promoted_prefix_lora(
            3,
            legal_tensors,
            long_prefix_ids=leg_long_p,
            short_prefix_ids=leg_short_p,
            alpha=16.0,
            rank=8,
            name="legal_sop_v3",
        )

        mixed12_items = []
        for g_it in pub_bundle["general_retention"]["blind_test_inputs"][:3]:
            mixed12_items.append((g_it["id"], "general_retention", _chat_ids(tokenizer, render_domain_short_messages("general_retention", g_it))))
        for c_it in code_prompts[:3]:
            mixed12_items.append((c_it["id"], "code_bugfix", _chat_ids(tokenizer, confirmatory_first_round(c_it, "text", code_sup_raw))))
        for m_it in pub_bundle["med_triage"]["blind_test_inputs"][:3]:
            mixed12_items.append((m_it["id"], "med_triage", _chat_ids(tokenizer, render_domain_long_messages("med_triage", m_it, pub_bundle))))
        for l_it in pub_bundle["legal_compliance"]["blind_test_inputs"][:3]:
            mixed12_items.append((l_it["id"], "legal_compliance", _chat_ids(tokenizer, render_domain_long_messages("legal_compliance", l_it, pub_bundle))))

        single_texts: dict[str, str] = {}
        rewrite_meta: dict[str, dict] = {}
        for tid, dom_tag, raw_ids in mixed12_items:
            sid = engine.add_request(raw_ids, SamplingParams(temperature=0.0, max_tokens=192))
            q_seq = engine.scheduler.waiting[-1]
            rewrite_meta[tid] = {
                "domain": dom_tag,
                "original_tokens": len(raw_ids),
                "rewritten_tokens": q_seq.num_prompt_tokens,
                "resolved_lora_slot_id": q_seq.lora_slot_id,
                "initial_kv_blocks": q_seq.num_blocks,
            }
            dr = _drain_engine(engine, [sid], tokenizer)
            single_texts[tid] = dr["by_seq_id"][sid]["text"]

        batch_sids = []
        sid_to_tid = {}
        for tid, _dom_tag, raw_ids in mixed12_items:
            sid = engine.add_request(raw_ids, SamplingParams(temperature=0.0, max_tokens=192))
            batch_sids.append(sid)
            sid_to_tid[sid] = tid
        batch_dr = _drain_engine(engine, batch_sids, tokenizer)

        mixed12_verification = {}
        for sid in batch_sids:
            tid = sid_to_tid[sid]
            b_txt = batch_dr["by_seq_id"][sid]["text"]
            mixed12_verification[tid] = {
                **rewrite_meta[tid],
                "matches_single_request_cudagraph_run": b_txt == single_texts[tid],
            }

        burst32_raw_prompts: list[list[int]] = []
        for idx_b in range(8):
            burst32_raw_prompts.append(
                _chat_ids(tokenizer, confirmatory_first_round(code_prompts[idx_b], "text", code_sup_raw))
            )
            burst32_raw_prompts.append(
                _chat_ids(
                    tokenizer,
                    render_domain_long_messages(
                        "med_triage", pub_bundle["med_triage"]["blind_test_inputs"][idx_b], pub_bundle
                    ),
                )
            )
            burst32_raw_prompts.append(
                _chat_ids(
                    tokenizer,
                    render_domain_long_messages(
                        "legal_compliance", pub_bundle["legal_compliance"]["blind_test_inputs"][idx_b], pub_bundle
                    ),
                )
            )
            burst32_raw_prompts.append(
                _chat_ids(
                    tokenizer,
                    render_domain_short_messages(
                        "general_retention", pub_bundle["general_retention"]["blind_test_inputs"][idx_b]
                    ),
                )
            )

        def _run_burst32_bench() -> dict:
            bm = engine.scheduler.block_manager
            total_blocks = len(bm.blocks)
            times_ms = []
            out_toks = 0
            peak_blocks = 0
            for trial in range(3):
                for r_ids in burst32_raw_prompts:
                    engine.add_request(r_ids, SamplingParams(temperature=0.0, max_tokens=128))
                t_s = time.perf_counter_ns()
                trial_toks = 0
                max_used = 0
                while not engine.is_finished():
                    fin, _ = engine.step(completion_only=True)
                    used_now = total_blocks - len(bm.free_block_ids)
                    if used_now > max_used:
                        max_used = used_now
                    for _s, c_ids in fin:
                        trial_toks += len(c_ids)
                el_ms = (time.perf_counter_ns() - t_s) / 1_000_000.0
                if trial >= 1:
                    times_ms.append(el_ms)
                    out_toks = trial_toks
                    peak_blocks = max(peak_blocks, max_used)
            med_ms = statistics.median(times_ms)
            return {
                "batch_requests": len(burst32_raw_prompts),
                "median_elapsed_ms": round(med_ms, 2),
                "output_tokens": out_toks,
                "output_tokens_per_sec": round(out_toks / (med_ms / 1000.0), 2),
                "peak_allocated_kv_blocks": peak_blocks,
                "peak_allocated_kv_mib": round((peak_blocks * 256 * 36 * 2 * 8 * 128 * 2) / (1024.0 * 1024.0), 1),
            }

        promoted_32_bench = _run_burst32_bench()

        engine.clear_promoted_prefix_lora(1)
        engine.clear_promoted_prefix_lora(2)
        engine.clear_promoted_prefix_lora(3)

        for warm_ids in (burst32_raw_prompts[0], burst32_raw_prompts[1], burst32_raw_prompts[2]):
            ws = engine.add_request(warm_ids, SamplingParams(temperature=0.0, max_tokens=4))
            _drain_engine(engine, [ws], tokenizer)

        native_apc_32_bench = _run_burst32_bench()

        total_lora_bytes = (
            r_slot1["slot_resident_bytes"]
            + r_slot2["slot_resident_bytes"]
            + r_slot3["slot_resident_bytes"]
        )
        gpu_redacted = {
            "name": selected_gpu["name"],
            "free_mib": selected_gpu["free_mib"],
            "used_mib": selected_gpu["used_mib"],
        }
        report = {
            "kind": "TINYVLLM_4DOMAIN_MULTISLOT_COSERVING_REPORT",
            "version": "cross_domain_v1",
            "gpu_summary": gpu_redacted,
            "cuda_graphs_captured_batch_sizes": graphs_captured,
            "registered_domains": {
                "slot_1_code_bugfix": {
                    **r_slot1,
                    "long_prefix_tokens": len(code_long_p),
                    "short_prefix_tokens": len(code_short_p),
                },
                "slot_2_med_triage": {
                    **r_slot2,
                    "long_prefix_tokens": len(med_long_p),
                    "short_prefix_tokens": len(med_short_p),
                },
                "slot_3_legal_compliance": {
                    **r_slot3,
                    "long_prefix_tokens": len(leg_long_p),
                    "short_prefix_tokens": len(leg_short_p),
                },
            },
            "total_3domain_bf16_lora_mib": round(total_lora_bytes / (1024.0 * 1024.0), 4),
            "mixed_12req_4domain_verification": mixed12_verification,
            "all_12_mixed_4domain_matched_single_cudagraph": all(
                v["matches_single_request_cudagraph_run"] for v in mixed12_verification.values()
            ),
            "burst32_4domain_paged_kv_comparison": {
                "native_3domain_warm_prefix_cache_apc": native_apc_32_bench,
                "promoted_3domain_bf16_lora_slots": promoted_32_bench,
                "throughput_speedup": round(
                    promoted_32_bench["output_tokens_per_sec"]
                    / max(native_apc_32_bench["output_tokens_per_sec"], 1e-6),
                    4,
                ),
                "kv_blocks_saved": native_apc_32_bench["peak_allocated_kv_blocks"]
                - promoted_32_bench["peak_allocated_kv_blocks"],
                "kv_blocks_saved_ratio": round(
                    1.0
                    - (
                        promoted_32_bench["peak_allocated_kv_blocks"]
                        / float(max(native_apc_32_bench["peak_allocated_kv_blocks"], 1))
                    ),
                    4,
                ),
                "kv_mib_saved": round(
                    native_apc_32_bench["peak_allocated_kv_mib"]
                    - promoted_32_bench["peak_allocated_kv_mib"],
                    1,
                ),
            },
        }
        Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
