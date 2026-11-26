from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import time

from experiments.cross_domain_prefix_lora.domain_sop_contracts import (
    LEGAL_CAPS,
    LEGAL_CURES,
    LEGAL_GRADES,
    LEGAL_KEYS,
    LEGAL_ROUTINGS,
    LEGAL_RULES,
    MED_ACTIONS,
    MED_DOSE_TABLE,
    MED_KEYS,
    MED_REGIMENS,
    MED_RENAL_TIERS,
    MED_RULES,
    canonical_json,
    extract_first_json_object,
    render_domain_long_messages,
    render_domain_short_messages,
)


LOCKED_CROSS_DOMAIN_HYPERPARAMS = {
    "rank": 8,
    "alpha": 16,
    "targets": ["q_proj", "v_proj"],
    "optimizer": "AdamW",
    "learning_rate": 0.00022,
    "lr_schedule": "linear_decay_to_10pct",
    "weight_decay": 0.01,
    "lambda_pos_kl": 1.5,
    "lambda_seed_surgical_ce": 0.55,
    "lambda_bg_kl": 2.5,
    "diff_token_weight": 4.0,
    "scaffold_token_weight": 1.0,
    "background_anchors_per_step": 2,
    "max_anchor_tokens": 128,
    "max_teacher_tokens": 192,
    "gradient_clip": 1.0,
    "seed": 42,
    "steps": 96,
}


def _ensure_lora_necessity_path() -> None:
    repo_lora = Path(__file__).resolve().parents[1] / "lora_necessity"
    if str(repo_lora) not in sys.path:
        sys.path.insert(0, str(repo_lora))


def find_json_value_char_spans(canonical_json_str: str) -> list[tuple[int, int]]:
    parsed = json.loads(canonical_json_str)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object string")
    spans: list[tuple[int, int]] = []
    idx = 0
    length = len(canonical_json_str)
    while idx < length:
        if canonical_json_str[idx] == '"':
            end_q = idx + 1
            while end_q < length:
                if canonical_json_str[end_q] == "\\":
                    end_q += 2
                    continue
                if canonical_json_str[end_q] == '"':
                    break
                end_q += 1
            after_key = end_q + 1
            while after_key < length and canonical_json_str[after_key].isspace():
                after_key += 1
            if after_key < length and canonical_json_str[after_key] == ":":
                val_start = after_key + 1
                while val_start < length and canonical_json_str[val_start].isspace():
                    val_start += 1
                if canonical_json_str[val_start] == '"':
                    val_end = val_start + 1
                    while val_end < length:
                        if canonical_json_str[val_end] == "\\":
                            val_end += 2
                            continue
                        if canonical_json_str[val_end] == '"':
                            val_end += 1
                            break
                        val_end += 1
                    spans.append((val_start, val_end))
                    idx = val_end
                    continue
                val_end = val_start
                while val_end < length and canonical_json_str[val_end] not in (",", "}"):
                    val_end += 1
                trimmed_end = val_end
                while trimmed_end > val_start and canonical_json_str[trimmed_end - 1].isspace():
                    trimmed_end -= 1
                spans.append((val_start, trimmed_end))
                idx = val_end
                continue
        idx += 1
    if len(spans) != len(parsed):
        raise ValueError("Value span count mismatch against parsed JSON keys")
    return spans


def build_json_value_surgical_weights(
    tokenizer,
    canonical_json_str: str,
    target_ids: list[int],
    *,
    diff_kl_weight: float = 4.0,
    scaffold_kl_weight: float = 1.0,
) -> tuple[list[float], list[float], int]:
    if len(target_ids) < 2:
        raise ValueError("target_ids must include JSON tokens and EOS")
    code_ids = list(target_ids[:-1])
    decoded_full = tokenizer.decode(
        code_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if not isinstance(decoded_full, str) or not decoded_full.startswith(canonical_json_str):
        raise ValueError("Decoded target_ids prefix mismatch with canonical_json_str")

    val_spans = find_json_value_char_spans(canonical_json_str)
    raw_kl: list[float] = []
    raw_surg: list[float] = []
    prev_len = 0
    val_tok_cnt = 0

    for tok_idx in range(len(code_ids)):
        prefix_str = tokenizer.decode(
            code_ids[: tok_idx + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        cur_len = len(prefix_str)
        char_lo = prev_len
        char_hi = max(cur_len, char_lo + 1)
        prev_len = cur_len

        in_value = any(sp_hi > char_lo and sp_lo < char_hi for sp_lo, sp_hi in val_spans)
        if in_value:
            raw_kl.append(float(diff_kl_weight))
            raw_surg.append(1.0)
            val_tok_cnt += 1
        else:
            raw_kl.append(float(scaffold_kl_weight))
            raw_surg.append(0.0)

    raw_kl.append(float(scaffold_kl_weight))
    raw_surg.append(0.0)

    if val_tok_cnt < 1:
        raise ValueError("Zero JSON value tokens mapped")
    kl_mean = sum(raw_kl) / float(len(raw_kl))
    surg_mean = sum(raw_surg) / float(len(raw_surg))
    if kl_mean <= 1e-9 or surg_mean <= 1e-9 or not math.isfinite(kl_mean) or not math.isfinite(surg_mean):
        raise ValueError("Invalid weight normalization mean")

    norm_kl = [wt / kl_mean for wt in raw_kl]
    norm_surg = [wt / surg_mean for wt in raw_surg]
    return norm_kl, norm_surg, val_tok_cnt


def validate_unlabeled_sop_candidate(domain: str, raw_text: str) -> str | None:
    parsed = extract_first_json_object(raw_text)
    if parsed is None:
        return None
    if domain == "med_triage":
        if set(parsed.keys()) != set(MED_KEYS):
            return None
        rt = parsed.get("renal_tier")
        act = parsed.get("action")
        reg = parsed.get("regimen")
        dose = parsed.get("dose_mg")
        rid = parsed.get("rule_id")
        if (
            rt not in MED_RENAL_TIERS
            or act not in MED_ACTIONS
            or reg not in MED_REGIMENS
            or rid not in MED_RULES
            or type(dose) is not int
        ):
            return None
        if act == "HOLD_STAT":
            if reg != "NONE" or dose != 0 or rid not in ("MED-R01", "MED-R04"):
                return None
        else:
            if reg == "NONE" or reg not in MED_DOSE_TABLE:
                return None
            if dose != MED_DOSE_TABLE[reg][rt]:
                return None
        ordered = {k: parsed[k] for k in MED_KEYS}
        return canonical_json(ordered)

    if domain == "legal_compliance":
        if set(parsed.keys()) != set(LEGAL_KEYS):
            return None
        grd = parsed.get("risk_grade")
        rou = parsed.get("routing")
        cap = parsed.get("min_cap_x")
        cur = parsed.get("cure_days")
        pr = parsed.get("primary_rule")
        if (
            grd not in LEGAL_GRADES
            or rou not in LEGAL_ROUTINGS
            or type(cap) is not int
            or cap not in LEGAL_CAPS
            or type(cur) is not int
            or cur not in LEGAL_CURES
            or pr not in LEGAL_RULES
        ):
            return None
        valid_pairs = {
            ("LEG-R01", "CRITICAL", "GC_BOARD_REJECT", 0),
            ("LEG-R02", "CRITICAL", "GC_BOARD_REJECT", 15),
            ("LEG-R03", "HIGH", "VP_LEGAL_CISO", 15),
            ("LEG-R04", "HIGH", "VP_LEGAL_CISO", 30),
            ("LEG-R05", "ELEVATED", "SENIOR_COUNSEL", 30),
            ("LEG-R06", "LOW", "AUTO_APPROVE", 45),
        }
        if (pr, grd, rou, cur) not in valid_pairs:
            return None
        ordered = {k: parsed[k] for k in LEGAL_KEYS}
        return canonical_json(ordered)

    raise ValueError(f"Unsupported SOP domain: {domain}")


def chat_prompt_and_target_ids(tokenizer, messages: list[dict], answer_text: str | None = None) -> tuple[list[int], list[int] | None]:
    p_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        truncation=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(p_ids, "get") and "input_ids" in p_ids:
        p_ids = p_ids["input_ids"]
    prompt_ids = [int(x) for x in p_ids]
    if answer_text is None:
        return prompt_ids, None
    full_msgs = list(messages) + [{"role": "assistant", "content": answer_text}]
    f_ids = tokenizer.apply_chat_template(
        full_msgs,
        tokenize=True,
        truncation=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if hasattr(f_ids, "get") and "input_ids" in f_ids:
        f_ids = f_ids["input_ids"]
    full_list = [int(x) for x in f_ids]
    if full_list[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template prefix mismatch when encoding assistant answer")
    eos_id = int(tokenizer.eos_token_id)
    tail = full_list[len(prompt_ids) :]
    if eos_id in tail:
        eos_pos = tail.index(eos_id)
        target_ids = tail[: eos_pos + 1]
    else:
        target_ids = tail + [eos_id]
    return prompt_ids, target_ids


def cached_greedy_generate(
    model,
    tokenizer,
    messages: list[dict],
    *,
    max_new_tokens: int,
    device,
) -> dict:
    import torch

    prompt_ids, _ = chat_prompt_and_target_ids(tokenizer, messages, answer_text=None)
    eos_id = int(tokenizer.eos_token_id)
    rng_devs = [device.index or 0] if device.type == "cuda" else []
    orig_cache = getattr(model.config, "use_cache", False)
    gen_ids: list[int] = []

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter_ns()
    with torch.random.fork_rng(devices=rng_devs), torch.no_grad():
        try:
            model.config.use_cache = True
            inp_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            out_obj = model(
                input_ids=inp_t,
                attention_mask=torch.ones_like(inp_t),
                use_cache=True,
                return_dict=True,
            )
            past_kv = getattr(out_obj, "past_key_values", None)
            nxt = int(torch.argmax(out_obj.logits[0, -1]).item())
            gen_ids.append(nxt)
            del out_obj, inp_t

            while nxt != eos_id and len(gen_ids) < max_new_tokens:
                step_in = torch.tensor([[nxt]], dtype=torch.long, device=device)
                step_out = model(
                    input_ids=step_in,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )
                past_kv = getattr(step_out, "past_key_values", None)
                nxt = int(torch.argmax(step_out.logits[0, -1]).item())
                gen_ids.append(nxt)
                del step_out, step_in
            del past_kv
        finally:
            model.config.use_cache = orig_cache

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
    stop_eos = bool(gen_ids and gen_ids[-1] == eos_id)
    dec_ids = gen_ids[:-1] if stop_eos else list(gen_ids)
    text_out = tokenizer.decode(dec_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {
        "prompt_ids": prompt_ids,
        "gen_ids": gen_ids,
        "stop_eos": stop_eos,
        "text": text_out,
        "wall_ms": wall_ms,
        "input_tokens": len(prompt_ids),
        "output_tokens": len(gen_ids),
    }


def prepare_multimodal_background_trajectories(
    model,
    tokenizer,
    background_anchors: list[dict],
    *,
    max_new_tokens: int = 128,
) -> list[dict]:
    import torch

    device = next(model.parameters()).device
    rng_devs = [device.index or 0] if device.type == "cuda" else []
    trajs: list[dict] = []
    with torch.random.fork_rng(devices=rng_devs), torch.no_grad():
        for anc in background_anchors:
            msgs = [{"role": "user", "content": anc["prompt"]}]
            dec = cached_greedy_generate(
                model, tokenizer, msgs, max_new_tokens=max_new_tokens, device=device
            )
            p_ids = dec["prompt_ids"]
            g_ids = dec["gen_ids"]
            full_ids = p_ids + g_ids
            inp_t = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            out_t = model(
                input_ids=inp_t,
                attention_mask=torch.ones_like(inp_t),
                use_cache=False,
                return_dict=True,
            )
            sl_cpu = (
                out_t.logits[0, len(p_ids) - 1 : len(p_ids) - 1 + len(g_ids)]
                .detach()
                .to(dtype=torch.bfloat16, device="cpu")
                .clone()
            )
            del out_t, inp_t
            trajs.append({
                "task_id": anc["id"],
                "full_ids": full_ids,
                "prompt_len": len(p_ids),
                "answer_len": len(g_ids),
                "base_answer_logits_bf16": sl_cpu,
            })
    return trajs


def prepare_domain_distillation_trajectories(
    model,
    tokenizer,
    domain: str,
    public_bundle: dict,
    *,
    hyperparams: dict | None = None,
) -> tuple[list[dict], dict]:
    import torch

    hp = copy.deepcopy(LOCKED_CROSS_DOMAIN_HYPERPARAMS if hyperparams is None else hyperparams)
    device = next(model.parameters()).device
    rng_devs = [device.index or 0] if device.type == "cuda" else []
    dom_data = public_bundle[domain]

    def _teacher_slice_for_canonical(t_msgs: list[dict], s_msgs: list[dict], canon_str: str) -> dict:
        t_pids, t_tids = chat_prompt_and_target_ids(tokenizer, t_msgs, answer_text=canon_str)
        s_pids, s_tids = chat_prompt_and_target_ids(tokenizer, s_msgs, answer_text=canon_str)
        if t_tids != s_tids or t_tids is None or s_tids is None:
            raise ValueError("Teacher/Student target token mismatch")
        ans_len = len(t_tids)
        t_full = t_pids + t_tids
        t_in = torch.tensor([t_full[:-1]], dtype=torch.long, device=device)
        t_out = model(
            input_ids=t_in,
            attention_mask=torch.ones_like(t_in),
            use_cache=False,
            return_dict=True,
        )
        t_slice = (
            t_out.logits[0, len(t_pids) - 1 : len(t_pids) - 1 + ans_len]
            .detach()
            .to(dtype=torch.bfloat16, device="cpu")
            .clone()
        )
        del t_out, t_in
        kl_wts, surg_wts, v_cnt = build_json_value_surgical_weights(
            tokenizer,
            canon_str,
            s_tids,
            diff_kl_weight=float(hp["diff_token_weight"]),
            scaffold_kl_weight=float(hp["scaffold_token_weight"]),
        )
        return {
            "student_full_ids": s_pids + s_tids,
            "student_prompt_len": len(s_pids),
            "answer_len": ans_len,
            "target_ids": s_tids,
            "target_weights": kl_wts,
            "surgical_ce_weights": surg_wts,
            "value_tokens": v_cnt,
            "teacher_answer_logits_bf16": t_slice,
        }

    seed_trajs: list[dict] = []
    mining_stats = {
        "domain": domain,
        "unlabeled_attempted": len(dom_data["unlabeled"]),
        "schema_consistent_accepted": 0,
        "accepted_ids": [],
    }
    verified_extra: list[dict] = []

    with torch.random.fork_rng(devices=rng_devs), torch.no_grad():
        for seed_item in dom_data["seeds"]:
            sid = seed_item["id"]
            canon_ans = canonical_json(seed_item["expected"])
            t_msgs = render_domain_long_messages(domain, seed_item, public_bundle, exclude_seed_id=sid)
            s_msgs = render_domain_short_messages(domain, seed_item)
            built = _teacher_slice_for_canonical(t_msgs, s_msgs, canon_ans)
            seed_trajs.append({
                "task_id": sid,
                "source_pool": "seed",
                **built,
            })

        for u_item in dom_data["unlabeled"]:
            uid = u_item["id"]
            t_msgs = render_domain_long_messages(domain, u_item, public_bundle, exclude_seed_id=None)
            gen = cached_greedy_generate(
                model,
                tokenizer,
                t_msgs,
                max_new_tokens=int(hp["max_teacher_tokens"]),
                device=device,
            )
            if not gen["stop_eos"]:
                continue
            canon_u = validate_unlabeled_sop_candidate(domain, gen["text"])
            if canon_u is None:
                continue
            s_msgs = render_domain_short_messages(domain, u_item)
            built_u = _teacher_slice_for_canonical(t_msgs, s_msgs, canon_u)
            built_u["surgical_ce_weights"] = [0.0] * int(built_u["answer_len"])
            mining_stats["schema_consistent_accepted"] += 1
            mining_stats["accepted_ids"].append(uid)
            verified_extra.append({
                "task_id": uid,
                "source_pool": "unlabeled_verified",
                **built_u,
            })

    return seed_trajs + verified_extra, mining_stats


def train_domain_sop_lora_adapter(
    model,
    positive_trajectories: list[dict],
    background_trajectories: list[dict],
    *,
    hyperparams: dict | None = None,
) -> dict:
    _ensure_lora_necessity_path()
    from hybrid_diff_csd_lora_ablation import train_hybrid_diff_csd_adapter

    hp = copy.deepcopy(LOCKED_CROSS_DOMAIN_HYPERPARAMS if hyperparams is None else hyperparams)
    return train_hybrid_diff_csd_adapter(
        model,
        positive_trajectories,
        background_trajectories,
        hyperparams=hp,
    )
