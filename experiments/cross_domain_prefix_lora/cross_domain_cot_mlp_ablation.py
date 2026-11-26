from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
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
    raise RuntimeError("No idle A100 80GB GPU available")


def render_seed_micro_cot(domain: str, seed_obj: dict) -> str:
    from experiments.cross_domain_prefix_lora.domain_sop_contracts import canonical_json

    inp = seed_obj["input"]
    exp = seed_obj["expected"]
    if domain == "med_triage":
        if inp["aki_stage"] >= 2:
            tier_trace = f"aki_stage={inp['aki_stage']}>=2 -> renal_tier=R4_SEVERE"
        elif inp["age"] >= 75 and inp["albumin_g_dl"] < 3.2:
            eff = inp["egfr"] - 12
            tier_trace = f"age={inp['age']}>=75 & alb={inp['albumin_g_dl']:.1f}<3.2 -> eff_egfr={eff} -> renal_tier={exp['renal_tier']}"
        else:
            tier_trace = f"eff_egfr={inp['egfr']} -> renal_tier={exp['renal_tier']}"
        rule_trace = (
            f"qtc={inp['qtc_ms']}, bleed={str(inp['active_bleed']).lower()}, plt={inp['platelets_k']}, "
            f"maoi={str(inp['on_maoi']).lower()}, k={inp['potassium_mmol']:.1f}, req={inp['requested_regimen']} "
            f"-> rule={exp['rule_id']}, regimen={exp['regimen']}, dose_mg={exp['dose_mg']}, action={exp['action']}"
        )
        return f"TRACE: {tier_trace} | {rule_trace}\n{canonical_json(exp)}"

    if domain == "legal_compliance":
        cap_trace = f"{inp['vendor_tier']}x{inp['data_class']} -> min_cap_x={exp['min_cap_x']}"
        red_trace = (
            f"cb={inp['cross_border']}, sec={str(inp['has_sec_assessment']).lower()}, "
            f"law={inp['governing_law']}, ip_uncap={str(inp['ip_indemnity_uncapped']).lower()}, "
            f"renew={inp['auto_renew_months']}m(conv={str(inp['terminate_for_conv']).lower()}), "
            f"prop_cap={inp['proposed_cap_x']}x -> rule={exp['primary_rule']}, grade={exp['risk_grade']}, "
            f"routing={exp['routing']}, cure={exp['cure_days']}"
        )
        return f"TRACE: {cap_trace} | {red_trace}\n{canonical_json(exp)}"
    raise ValueError(domain)


def render_cot_messages(
    domain: str,
    case_obj: dict,
    public_bundle: dict,
    *,
    is_long: bool,
    exclude_seed_id: str | None = None,
) -> list[dict]:
    from experiments.cross_domain_prefix_lora.domain_sop_contracts import (
        LEGAL_TASK_SUFFIX_SPEC,
        MED_TASK_SUFFIX_SPEC,
    )

    dom_data = public_bundle[domain]
    cot_spec = (
        "First output one concise `TRACE: ...` line computing the exact intermediate tier/cap and first triggered rule, "
        "followed on the next line by ONLY the JSON object.\n"
        + (MED_TASK_SUFFIX_SPEC.split("\n", 1)[1] if domain == "med_triage" else LEGAL_TASK_SUFFIX_SPEC.split("\n", 1)[1])
    )
    if not is_long:
        hdr = (
            "Apply institutional clinical pharmacy protocol MED-SOP-2026-V4."
            if domain == "med_triage"
            else "Apply enterprise legal procurement policy LEGAL-SOP-2026-V3."
        )
        return [{"role": "user", "content": f"{hdr}\n\n=== CURRENT CASE FOR REVIEW ===\n{case_obj['vignette']}\n\n{cot_spec}"}]

    ex_blocks = ["SECTION 4. VERIFIED INSTITUTIONAL WORKED CASE EXEMPLARS (WITH TRACE)"]
    for s_it in dom_data["seeds"]:
        if exclude_seed_id is not None and s_it["id"] == exclude_seed_id:
            continue
        ex_blocks.append(
            f"\n--- Worked Case {s_it['id']} ---\n{s_it['vignette']}\nVerified Output:\n{render_seed_micro_cot(domain, s_it)}"
        )
    long_hdr = (
        f"You are an institutional verification engine for {domain}.\n\n"
        + dom_data["sop_manual_text"]
        + "\n"
        + "\n".join(ex_blocks)
    )
    return [{"role": "user", "content": f"{long_hdr}\n\n=== CURRENT CASE FOR REVIEW ===\n{case_obj['vignette']}\n\n{cot_spec}"}]


def _collect_qkv_and_mlp_targets(model) -> list[tuple[str, object]]:
    targets = []
    for idx, layer in enumerate(model.model.layers):
        targets.append((f"model.layers.{idx}.self_attn.q_proj", layer.self_attn.q_proj))
        targets.append((f"model.layers.{idx}.self_attn.v_proj", layer.self_attn.v_proj))
        targets.append((f"model.layers.{idx}.mlp.down_proj", layer.mlp.down_proj))
    return targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    selected_gpu = _select_idle_a100()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu["index"])

    repo_root = Path(args.repo_root).resolve()
    if (repo_root / "experiments/cross_domain_prefix_lora/data/cross_domain_v1/blind_oracles.json").exists():
        raise RuntimeError("blind_oracles.json must not exist on GPU stage")

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "experiments/lora_necessity"))

    import torch
    from torch import nn
    from torch.nn import functional
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kl_anchored_lora_ablation import _compute_trajectory_kl_loss
    from training_worker import LoRAProjection, _state, _tensor_digest
    from experiments.cross_domain_prefix_lora.surgical_json_csd_trainer import (
        cached_greedy_generate,
        chat_prompt_and_target_ids,
        prepare_multimodal_background_trajectories,
        validate_unlabeled_sop_candidate,
    )

    public_bundle = json.loads(
        (repo_root / "experiments/cross_domain_prefix_lora/data/cross_domain_v1/public_bundle.json").read_text("utf-8")
    )

    torch.cuda.init()
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    init_sha = _tensor_digest(_state(model))

    bg_trajs = prepare_multimodal_background_trajectories(
        model, tokenizer, public_bundle["background_anchors"], max_new_tokens=128
    )

    results = {}
    for dom in ("med_triage", "legal_compliance"):
        dom_data = public_bundle[dom]

        def _slice_pair(t_msgs: list[dict], s_msgs: list[dict], ans_str: str) -> dict:
            t_p, t_t = chat_prompt_and_target_ids(tokenizer, t_msgs, answer_text=ans_str)
            s_p, s_t = chat_prompt_and_target_ids(tokenizer, s_msgs, answer_text=ans_str)
            if t_t != s_t or t_t is None or s_t is None:
                raise ValueError("target mismatch")
            alen = len(t_t)
            t_in = torch.tensor([(t_p + t_t)[:-1]], dtype=torch.long, device=device)
            with torch.no_grad():
                t_out = model(input_ids=t_in, attention_mask=torch.ones_like(t_in), use_cache=False, return_dict=True)
                t_sl = t_out.logits[0, len(t_p) - 1 : len(t_p) - 1 + alen].detach().to(torch.bfloat16).cpu().clone()
            del t_out, t_in
            return {
                "student_full_ids": s_p + s_t,
                "student_prompt_len": len(s_p),
                "answer_len": alen,
                "target_ids": s_t,
                "teacher_logits_bf16": t_sl,
            }

        seed_trajs = []
        for s_it in dom_data["seeds"]:
            ans_cot = render_seed_micro_cot(dom, s_it)
            t_m = render_cot_messages(dom, s_it, public_bundle, is_long=True, exclude_seed_id=s_it["id"])
            s_m = render_cot_messages(dom, s_it, public_bundle, is_long=False)
            seed_trajs.append({"source_pool": "seed", **_slice_pair(t_m, s_m, ans_cot)})

        extra_trajs = []
        for u_it in dom_data["unlabeled"]:
            t_m = render_cot_messages(dom, u_it, public_bundle, is_long=True)
            gen_u = cached_greedy_generate(model, tokenizer, t_m, max_new_tokens=220, device=device)
            if not gen_u["stop_eos"] or "TRACE:" not in gen_u["text"]:
                continue
            canon_j = validate_unlabeled_sop_candidate(dom, gen_u["text"])
            if canon_j is None:
                continue
            trace_ln = gen_u["text"].strip().splitlines()[0].strip()
            clean_ans = f"{trace_ln}\n{canon_j}"
            s_m = render_cot_messages(dom, u_it, public_bundle, is_long=False)
            extra_trajs.append({"source_pool": "extra", **_slice_pair(t_m, s_m, clean_ans)})

        base_cot_recs = []
        text_cot_recs = []
        for c_it in dom_data["blind_test_inputs"]:
            g_b = cached_greedy_generate(
                model, tokenizer, render_cot_messages(dom, c_it, public_bundle, is_long=False), max_new_tokens=220, device=device
            )
            base_cot_recs.append({"id": c_it["id"], "text": g_b["text"], "wall_ms": round(g_b["wall_ms"], 2), "input_tokens": g_b["input_tokens"]})
            g_t = cached_greedy_generate(
                model, tokenizer, render_cot_messages(dom, c_it, public_bundle, is_long=True), max_new_tokens=220, device=device
            )
            text_cot_recs.append({"id": c_it["id"], "text": g_t["text"], "wall_ms": round(g_t["wall_ms"], 2), "input_tokens": g_t["input_tokens"]})

        targets = _collect_qkv_and_mlp_targets(model)
        originals = []
        wrappers = []
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        try:
            for p_str, b_lin in targets:
                par_p, attr_n = p_str.rsplit(".", 1)
                par_m = model.get_submodule(par_p)
                w_m = LoRAProjection(b_lin)
                originals.append((par_m, attr_n, b_lin))
                setattr(par_m, attr_n, w_m)
                wrappers.append(w_m)

            t_params = [p for w in wrappers for p in (w.adapter_a, w.adapter_b)]
            opt = torch.optim.AdamW(t_params, lr=0.0003, weight_decay=0.01)
            steps = 112
            s_cur = 0
            e_cur = 0
            for st_i in range(steps):
                lr_now = 0.0003 * (1.0 - 0.85 * (st_i / float(steps - 1)))
                for pg in opt.param_groups:
                    pg["lr"] = lr_now
                if extra_trajs and (st_i % 3 == 2):
                    item = extra_trajs[e_cur % len(extra_trajs)]
                    e_cur += 1
                else:
                    item = seed_trajs[s_cur % len(seed_trajs)]
                    s_cur += 1
                opt.zero_grad(set_to_none=True)
                inp = torch.tensor([item["student_full_ids"][:-1]], dtype=torch.long, device=device)
                out = model(input_ids=inp, attention_mask=torch.ones_like(inp), use_cache=False, return_dict=True)
                plen = item["student_prompt_len"]
                alen = item["answer_len"]
                s_sl = out.logits[0, plen - 1 : plen - 1 + alen].float()
                t_sl = item["teacher_logits_bf16"].to(device=device, dtype=torch.float32)
                kl_l = functional.kl_div(
                    functional.log_softmax(s_sl, dim=-1),
                    functional.log_softmax(t_sl, dim=-1),
                    log_target=True,
                    reduction="batchmean",
                )
                loss = 1.5 * kl_l
                if item["source_pool"] == "seed":
                    tgt = torch.tensor(item["target_ids"], dtype=torch.long, device=device)
                    ce_l = functional.cross_entropy(s_sl, tgt)
                    loss = loss + 0.65 * ce_l
                loss.backward()
                del out, inp, s_sl, t_sl, kl_l, loss

                bg_it = bg_trajs[st_i % len(bg_trajs)]
                bg_kl, _, _ = _compute_trajectory_kl_loss(model, bg_it, device)
                (2.0 * bg_kl).backward()
                del bg_kl

                torch.nn.utils.clip_grad_norm_(t_params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            for w in wrappers:
                w.requires_grad_(False)
                w.eval()

            lora_cot_recs = []
            for c_it in dom_data["blind_test_inputs"]:
                g_l = cached_greedy_generate(
                    model, tokenizer, render_cot_messages(dom, c_it, public_bundle, is_long=False), max_new_tokens=220, device=device
                )
                lora_cot_recs.append({"id": c_it["id"], "text": g_l["text"], "wall_ms": round(g_l["wall_ms"], 2), "input_tokens": g_l["input_tokens"]})

            gen_ret_recs = []
            for g_it in public_bundle["general_retention"]["blind_test_inputs"][:8]:
                g_r = cached_greedy_generate(
                    model, tokenizer, [{"role": "user", "content": g_it["prompt"]}], max_new_tokens=220, device=device
                )
                gen_ret_recs.append({"id": g_it["id"], "text": g_r["text"]})

            lora_bf16_bytes = sum(p.numel() * 2 for p in t_params)
        finally:
            for par_m, attr_n, b_lin in reversed(originals):
                setattr(par_m, attr_n, b_lin)

        results[dom] = {
            "accepted_unlabeled_cot": len(extra_trajs),
            "lora_bf16_bytes": lora_bf16_bytes,
            "lora_bf16_mib": round(lora_bf16_bytes / (1024.0 * 1024.0), 3),
            "base_cot_records": base_cot_recs,
            "long_text_cot_records": text_cot_recs,
            "lora_cot_mlp_records": lora_cot_recs,
            "general_retention_8_records": gen_ret_recs,
        }

    if _tensor_digest(_state(model)) != init_sha:
        raise RuntimeError("Base model mutated")

    Path(args.out_json).write_text(
        json.dumps({"kind": "CROSS_DOMAIN_COT_MLP_DIAGNOSTIC_RAW", "domains": results}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
