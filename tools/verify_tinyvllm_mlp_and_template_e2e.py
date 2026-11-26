from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
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


def _drain(engine, sids: list[int], tokenizer) -> dict[int, str]:
    eos_id = getattr(engine.scheduler, "eos", getattr(tokenizer, "eos_token_id", None))
    out: dict[int, str] = {}
    while not engine.is_finished():
        fin, _ = engine.step(completion_only=True)
        for sid, comp_ids in fin:
            dec = list(comp_ids[:-1]) if (comp_ids and comp_ids[-1] == eos_id) else list(comp_ids)
            out[int(sid)] = tokenizer.decode(dec, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    for s in sids:
        if s not in out:
            raise RuntimeError(f"Missing sid {s}")
    return out


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

    def _load(p: str, sha: str) -> dict:
        raw = Path(p).read_bytes()
        if _sha256_bytes(raw) != sha:
            raise ValueError(f"SHA256 mismatch for {p}")
        return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)["adapter"]

    code_t = _load(args.code_adapter_path, args.code_expected_sha256)
    med_t = _load(args.med_adapter_path, args.med_expected_sha256)
    leg_t = _load(args.legal_adapter_path, args.legal_expected_sha256)

    from confirmatory_v2_contract import TRANSFER_IDS, confirmatory_first_round
    from experiments.cross_domain_prefix_lora.dense_compilation_and_residual_study import (
        LEGAL_MICRO_TABLE_RESIDUAL,
        MED_MICRO_TABLE_RESIDUAL,
    )
    from experiments.cross_domain_prefix_lora.domain_sop_contracts import (
        LEGAL_TASK_SUFFIX_SPEC,
        MED_TASK_SUFFIX_SPEC,
    )
    from tinyvllm.engine.llm_engine import LLMEngine
    from tinyvllm.sampling_params import SamplingParams

    pub_bundle = json.loads(
        (repo_root / "experiments/cross_domain_prefix_lora/data/cross_domain_v1/public_bundle.json").read_text("utf-8")
    )
    conf_dir = repo_root / "experiments/lora_necessity/data/confirmatory_v2_r1"
    code_prompts = [
        x for x in json.loads((conf_dir / "prompts.json").read_text("utf-8")) if x["id"] in TRANSFER_IDS
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
        prefix_lora_rank=16,
        prefix_lora_enable_mlp=True,
    )
    try:
        tokenizer = engine.tokenizer
        graphs = sorted(int(bs) for bs in getattr(engine.model_runner, "graphs", {}).keys())

        code_long = _chat_ids(tokenizer, confirmatory_first_round(code_prompts[0], "text", code_sup_raw))
        code_short = _chat_ids(tokenizer, confirmatory_first_round(code_prompts[0], "lora", code_sup_raw))
        suf_c = 0
        while code_long[-1 - suf_c] == code_short[-1 - suf_c]:
            suf_c += 1
        r_slot1 = engine.register_promoted_prefix_lora(
            1,
            code_t,
            long_prefix_ids=code_long[: len(code_long) - suf_c],
            short_prefix_ids=code_short[: len(code_short) - suf_c],
            alpha=16.0,
            rank=8,
            name="code_r8_qv",
        )

        def _build_dynamic_long_and_template(dom: str):
            dom_d = pub_bundle[dom]
            cot_spec = (
                "First output one concise `TRACE: ...` line computing the exact intermediate tier/cap and first triggered rule, "
                "followed on the next line by ONLY the JSON object.\n"
                + (MED_TASK_SUFFIX_SPEC.split("\n", 1)[1] if dom == "med_triage" else LEGAL_TASK_SUFFIX_SPEC.split("\n", 1)[1])
            )
            micro_res = MED_MICRO_TABLE_RESIDUAL if dom == "med_triage" else LEGAL_MICRO_TABLE_RESIDUAL
            sop_title = "MED-SOP-2026-V4" if dom == "med_triage" else "LEGAL-SOP-2026-V3"

            def _make_long_msg(c_obj: dict, dyn_meta: str) -> list[dict]:
                content = (
                    f"[{sop_title} | SessionMeta: {dyn_meta}]\n"
                    + dom_d["sop_manual_text"]
                    + f"\n\n=== CURRENT CASE FOR REVIEW ===\n{c_obj['vignette']}\n\n{cot_spec}"
                )
                return [{"role": "user", "content": content}]

            def _make_short_target_msg(c_obj: dict, dyn_meta: str) -> list[dict]:
                content = (
                    f"[{sop_title} | SessionMeta: {dyn_meta}]\n"
                    + micro_res
                    + f"\n\n=== CURRENT CASE FOR REVIEW ===\n{c_obj['vignette']}\n\n{cot_spec}"
                )
                return [{"role": "user", "content": content}]

            s_case = dom_d["blind_test_inputs"][0]
            l_a = _chat_ids(tokenizer, _make_long_msg(s_case, "date=2026-11-26 user=alice_01"))
            l_b = _chat_ids(tokenizer, _make_long_msg(s_case, "date=2026-12-19 user=robert_chen_992"))
            s_a = _chat_ids(tokenizer, _make_short_target_msg(s_case, "date=2026-11-26 user=alice_01"))

            head_len = 0
            while head_len < min(len(l_a), len(l_b)) and l_a[head_len] == l_b[head_len]:
                head_len += 1
            common_tail_case = 0
            while l_a[-1 - common_tail_case] == s_a[-1 - common_tail_case]:
                common_tail_case += 1

            common_long_suffix = 0
            while l_a[-1 - common_long_suffix] == l_b[-1 - common_long_suffix]:
                common_long_suffix += 1

            tail_anchor = l_a[len(l_a) - common_long_suffix : len(l_a) - common_tail_case]
            var_end_in_s = len(l_a) - common_long_suffix
            short_repl = s_a[var_end_in_s : len(s_a) - common_tail_case]
            head_anchor = l_a[:head_len]
            return head_anchor, tail_anchor, short_repl, _make_long_msg

        med_head, med_tail, med_repl, make_med_long = _build_dynamic_long_and_template("med_triage")
        leg_head, leg_tail, leg_repl, make_leg_long = _build_dynamic_long_and_template("legal_compliance")

        r_slot2 = engine.register_promoted_template_prefix_lora(
            2,
            med_t,
            head_anchor_ids=med_head,
            tail_anchor_ids=med_tail,
            max_var_tokens=32,
            short_prefix_ids=med_repl,
            preserve_var_span=True,
            alpha=32.0,
            rank=16,
            name="med_dense_r16_mlp_tpl",
        )
        r_slot3 = engine.register_promoted_template_prefix_lora(
            3,
            leg_t,
            head_anchor_ids=leg_head,
            tail_anchor_ids=leg_tail,
            max_var_tokens=32,
            short_prefix_ids=leg_repl,
            preserve_var_span=True,
            alpha=32.0,
            rank=16,
            name="legal_dense_r16_mlp_tpl",
        )

        mixed_check_reqs = [
            ("WT05", _chat_ids(tokenizer, confirmatory_first_round(code_prompts[4], "text", code_sup_raw))),
            ("M_T01_dynA", _chat_ids(tokenizer, make_med_long(pub_bundle["med_triage"]["blind_test_inputs"][0], "date=2026-11-26 ward=ICU_3"))),
            ("M_T05_dynB", _chat_ids(tokenizer, make_med_long(pub_bundle["med_triage"]["blind_test_inputs"][4], "date=2026-12-08 ward=NEPHRO_99"))),
            ("L_T01_dynA", _chat_ids(tokenizer, make_leg_long(pub_bundle["legal_compliance"]["blind_test_inputs"][0], "date=2026-11-26 region=APAC"))),
            ("L_T05_dynB", _chat_ids(tokenizer, make_leg_long(pub_bundle["legal_compliance"]["blind_test_inputs"][4], "date=2027-01-15 region=EMEA_HQ"))),
        ]

        single_out = {}
        meta_out = {}
        for lbl, r_ids in mixed_check_reqs:
            sid = engine.add_request(r_ids, SamplingParams(temperature=0.0, max_tokens=220))
            q_seq = engine.scheduler.waiting[-1]
            meta_out[lbl] = {
                "original_tokens": len(r_ids),
                "rewritten_tokens": q_seq.num_prompt_tokens,
                "resolved_lora_slot_id": q_seq.lora_slot_id,
                "initial_kv_blocks": q_seq.num_blocks,
            }
            single_out[lbl] = _drain(engine, [sid], tokenizer)[sid]

        b_sids = []
        sid_lbl = {}
        for lbl, r_ids in mixed_check_reqs:
            sid = engine.add_request(r_ids, SamplingParams(temperature=0.0, max_tokens=220))
            b_sids.append(sid)
            sid_lbl[sid] = lbl
        b_out = _drain(engine, b_sids, tokenizer)
        for sid in b_sids:
            lbl = sid_lbl[sid]
            meta_out[lbl]["matches_single_cudagraph_run"] = b_out[sid] == single_out[lbl]

        med_16_sids = []
        med_sid_id = {}
        for idx_m, c_it in enumerate(pub_bundle["med_triage"]["blind_test_inputs"]):
            r_ids = _chat_ids(tokenizer, make_med_long(c_it, f"date=2026-11-{10 + idx_m:02d} req={1000 + idx_m}"))
            sid = engine.add_request(r_ids, SamplingParams(temperature=0.0, max_tokens=220))
            med_16_sids.append(sid)
            med_sid_id[sid] = c_it["id"]

        leg_16_sids = []
        leg_sid_id = {}
        for idx_l, c_it in enumerate(pub_bundle["legal_compliance"]["blind_test_inputs"]):
            r_ids = _chat_ids(tokenizer, make_leg_long(c_it, f"date=2026-12-{10 + idx_l:02d} deal={2000 + idx_l}"))
            sid = engine.add_request(r_ids, SamplingParams(temperature=0.0, max_tokens=220))
            leg_16_sids.append(sid)
            leg_sid_id[sid] = c_it["id"]

        t_b32 = time.perf_counter_ns()
        all_32_out = _drain(engine, med_16_sids + leg_16_sids, tokenizer)
        b32_ms = (time.perf_counter_ns() - t_b32) / 1_000_000.0

        med_engine_recs = [{"id": med_sid_id[s], "text": all_32_out[s]} for s in med_16_sids]
        leg_engine_recs = [{"id": leg_sid_id[s], "text": all_32_out[s]} for s in leg_16_sids]

        rep = {
            "kind": "TINYVLLM_MLP_AND_DYNAMIC_TEMPLATE_E2E_REPORT",
            "gpu_summary": {k: v for k, v in selected_gpu.items() if k != "index"},
            "cuda_graphs_captured": graphs,
            "slot_receipts": {"slot_1_code_r8_qv": r_slot1, "slot_2_med_r16_mlp": r_slot2, "slot_3_legal_r16_mlp": r_slot3},
            "mixed_rank_and_dynamic_template_verification": meta_out,
            "all_5_heterogeneous_rank_mlp_matched_single_run": all(
                v["matches_single_cudagraph_run"] for v in meta_out.values()
            ),
            "batch32_dynamic_template_elapsed_ms": round(b32_ms, 2),
            "med_16_engine_records": med_engine_recs,
            "legal_16_engine_records": leg_engine_recs,
        }
        Path(args.out_json).write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
