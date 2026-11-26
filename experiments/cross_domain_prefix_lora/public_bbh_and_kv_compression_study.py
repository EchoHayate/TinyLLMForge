from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time


def exact_mcnemar_pvalue(w2r: int, r2w: int) -> float:
    n_disc = int(w2r + r2w)
    if n_disc == 0:
        return 1.0
    lo = min(int(w2r), int(r2w))
    tail = sum(math.comb(n_disc, k) * (0.5 ** n_disc) for k in range(lo + 1))
    return min(1.0, 2.0 * tail)


def build_public_bbh_suite(seed: int = 2026) -> dict:
    rng = random.Random(seed)

    def _gen_bool_item(idx: int) -> dict:
        ops = ["and", "or"]
        a_v, b_v, c_v, d_v = [rng.choice([True, False]) for _ in range(4)]
        n1, n2 = rng.choice([True, False]), rng.choice([True, False])
        op1, op2, op3 = rng.choice(ops), rng.choice(ops), rng.choice(ops)
        t_a = f"not {a_v}" if n1 else f"{a_v}"
        t_c = f"not {c_v}" if n2 else f"{c_v}"
        expr = f"not ( ( {t_a} {op1} {b_v} ) {op2} ( {t_c} {op3} {d_v} ) )"
        val = eval(expr)
        v1 = (not a_v) if n1 else a_v
        left = (v1 and b_v) if op1 == "and" else (v1 or b_v)
        v3 = (not c_v) if n2 else c_v
        right = (v3 and d_v) if op3 == "and" else (v3 or d_v)
        mid = (left and right) if op2 == "and" else (left or right)
        cot = (
            f"TRACE: left=({t_a} {op1} {b_v})={left}; right=({t_c} {op3} {d_v})={right}; "
            f"inner=({left} {op2} {right})={mid}; outer=not {mid}={val}\nFINAL_ANSWER: {val}"
        )
        return {
            "id": f"BBH_BOOL_{idx:03d}",
            "subtask": "boolean_expressions",
            "question": f"Evaluate the boolean expression: `{expr}`. End with `FINAL_ANSWER: True` or `FINAL_ANSWER: False`.",
            "answer": str(val),
            "cot_target": cot,
        }

    def _gen_web_of_lies(idx: int) -> dict:
        names_pool = ["Alice", "Bob", "Claire", "Dave", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy"]
        chosen = rng.sample(names_pool, 5)
        truths = [rng.choice([True, False]) for _ in range(5)]
        stmts = [f"{chosen[0]} tells the truth." if truths[0] else f"{chosen[0]} lies."]
        trace_parts = [f"{chosen[0]}={truths[0]}"]
        for k in range(1, 5):
            says_truth = truths[k] == truths[k - 1]
            verb = "tells the truth" if says_truth else "lies"
            stmts.append(f"{chosen[k]} says {chosen[k - 1]} {verb}.")
            trace_parts.append(f"{chosen[k]}={truths[k]}")
        ans = "Yes" if truths[-1] else "No"
        q_txt = " ".join(stmts) + f" Does {chosen[-1]} tell the truth? End with `FINAL_ANSWER: Yes` or `FINAL_ANSWER: No`."
        cot = f"TRACE: {' -> '.join(trace_parts)}\nFINAL_ANSWER: {ans}"
        return {
            "id": f"BBH_LIES_{idx:03d}",
            "subtask": "web_of_lies",
            "question": q_txt,
            "answer": ans,
            "cot_target": cot,
        }

    def _gen_navigate(idx: int) -> dict:
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        d_idx = 0
        x, y = 0, 0
        steps_txt = ["Always face north at the start."]
        trace_xy = ["(0,0,N)"]
        dir_names = ["N", "E", "S", "W"]
        if idx % 2 == 0:
            n1 = rng.randint(2, 6)
            n2 = rng.randint(2, 5)
            moves = [
                ("forward", n1),
                ("right", 0),
                ("forward", n2),
                ("around", 0),
                ("forward", n2),
                ("left", 0),
                ("forward", n1),
            ]
        else:
            moves = []
            for _ in range(5):
                act = rng.choice(["forward", "left", "right", "backward"])
                dist = rng.randint(1, 5) if act in ("forward", "backward") else 0
                moves.append((act, dist))

        for act, dist in moves:
            if act == "left":
                d_idx = (d_idx - 1) % 4
                steps_txt.append("Turn left.")
            elif act == "right":
                d_idx = (d_idx + 1) % 4
                steps_txt.append("Turn right.")
            elif act == "around":
                d_idx = (d_idx + 2) % 4
                steps_txt.append("Turn around.")
            elif act == "forward":
                dx, dy = dirs[d_idx]
                x += dx * dist
                y += dy * dist
                steps_txt.append(f"Take {dist} steps forward.")
            elif act == "backward":
                dx, dy = dirs[d_idx]
                x -= dx * dist
                y -= dy * dist
                steps_txt.append(f"Take {dist} steps backward.")
            trace_xy.append(f"({x},{y},{dir_names[d_idx]})")

        ans = "Yes" if (x == 0 and y == 0) else "No"
        q_txt = (
            "If you follow these instructions, do you return to the starting point? "
            + " ".join(steps_txt)
            + " End with `FINAL_ANSWER: Yes` or `FINAL_ANSWER: No`."
        )
        cot = f"TRACE: {' -> '.join(trace_xy)} -> origin={(x == 0 and y == 0)}\nFINAL_ANSWER: {ans}"
        return {
            "id": f"BBH_NAV_{idx:03d}",
            "subtask": "navigate",
            "question": q_txt,
            "answer": ans,
            "cot_target": cot,
        }

    def _gen_dyck(idx: int) -> dict:
        pairs = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
        stack = []
        seq = []
        total_opens = rng.randint(4, 7)
        opened = 0
        while opened < total_opens:
            if stack and len(stack) >= 2 and rng.random() < 0.45:
                _op_c, cl_c = stack.pop()
                seq.append(cl_c)
            else:
                pr = rng.choice(pairs)
                stack.append(pr)
                seq.append(pr[0])
                opened += 1
        while len(stack) > rng.randint(2, 4):
            _, cl_c = stack.pop()
            seq.append(cl_c)
        if not stack:
            pr = rng.choice(pairs)
            stack.append(pr)
            seq.append(pr[0])
        closing = [cl_c for _, cl_c in reversed(stack)]
        ans = " ".join(closing)
        q_txt = (
            f"Complete the rest of the sequence, making sure that the parentheses are closed properly: "
            f"`{' '.join(seq)}`. Output space-separated closing brackets after `FINAL_ANSWER: `."
        )
        cot = f"TRACE: unclosed_stack=[{' '.join(op_c for op_c, _ in stack)}] -> reverse_close={ans}\nFINAL_ANSWER: {ans}"
        return {
            "id": f"BBH_DYCK_{idx:03d}",
            "subtask": "dyck_languages",
            "question": q_txt,
            "answer": ans,
            "cot_target": cot,
        }

    builders = [_gen_bool_item, _gen_web_of_lies, _gen_navigate, _gen_dyck]
    exemplars = []
    train_unlabeled = []
    test_items = []
    seen_q = set()

    for b_fn in builders:
        cnt_ex, cnt_tr, cnt_te = 0, 0, 0
        seq_i = 1
        while cnt_ex < 3 or cnt_tr < 16 or cnt_te < 30:
            item = b_fn(seq_i)
            seq_i += 1
            if item["question"] in seen_q:
                continue
            seen_q.add(item["question"])
            if cnt_ex < 3:
                exemplars.append(item)
                cnt_ex += 1
            elif cnt_tr < 16:
                train_unlabeled.append({"id": item["id"], "subtask": item["subtask"], "question": item["question"]})
                cnt_tr += 1
            elif cnt_te < 30:
                test_items.append(item)
                cnt_te += 1

    prefix_lines = [
        "================================================================================",
        "BIG-BENCH HARD (BBH) MULTI-TASK SYMBOLIC REASONING PROTOCOL & COT EXEMPLARS",
        "Always emit a single `TRACE: ...` step-by-step state trace line first, followed",
        "on the next line by `FINAL_ANSWER: <exact_answer>`.",
        "Rules:",
        "1. boolean_expressions: evaluate inner parentheses first (`not` binds tighter than `and`/`or`).",
        "2. web_of_lies: propagate truth values sequentially (`X says Y lies` => `X != Y`; `X says Y tells the truth` => `X == Y`).",
        "3. navigate: track `(x, y, heading)` starting at `(0, 0, N)` where N=(0,+1), E=(+1,0), S=(0,-1), W=(-1,0); `Yes` iff final `(x,y)==(0,0)`.",
        "4. dyck_languages: push opening brackets `( [ { <` onto stack, pop on matching close, then emit remaining stack top-to-bottom as closing brackets.",
        "================================================================================",
    ]
    for ex in exemplars:
        prefix_lines.append(f"\n--- Exemplar {ex['id']} ({ex['subtask']}) ---\nQ: {ex['question']}\nA:\n{ex['cot_target']}")
    long_prefix_text = "\n".join(prefix_lines)

    return {
        "long_prefix_text": long_prefix_text,
        "exemplars": exemplars,
        "train_unlabeled": train_unlabeled,
        "test_items": test_items,
    }


def score_bbh_prediction(raw_text: str, expected_answer: str) -> bool:
    if not isinstance(raw_text, str):
        return False
    matches = re.findall(r"FINAL_ANSWER:\s*([^\n\r`]+)", raw_text)
    if matches:
        got = matches[-1].strip().rstrip(".")
    else:
        lines = [ln.strip() for ln in raw_text.strip().splitlines() if ln.strip()]
        got = lines[-1] if lines else ""
    norm_got = " ".join(got.replace("`", "").strip().split())
    norm_exp = " ".join(expected_answer.strip().split())
    return norm_got.lower() == norm_exp.lower()


def quantize_dequantize_tensor_int8_per_token(tensor):
    import torch

    orig_dtype = tensor.dtype
    fl = tensor.float()
    amax = fl.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    scale = amax / 127.0
    q_int8 = torch.clamp(torch.round(fl / scale), -127, 127).to(torch.int8)
    deq = (q_int8.float() * scale).to(orig_dtype)
    bytes_used = q_int8.numel() * 1 + scale.numel() * 2
    return deq, bytes_used


def quantize_dequantize_tensor_int4_group32(tensor):
    import torch

    orig_dtype = tensor.dtype
    orig_shape = tensor.shape
    head_dim = orig_shape[-1]
    group_sz = 32
    fl = tensor.float().reshape(*orig_shape[:-1], head_dim // group_sz, group_sz)
    mn = fl.amin(dim=-1, keepdim=True)
    mx = fl.amax(dim=-1, keepdim=True)
    scale = ((mx - mn) / 15.0).clamp(min=1e-6)
    q_u4 = torch.clamp(torch.round((fl - mn) / scale), 0, 15)
    deq = (q_u4 * scale + mn).reshape(orig_shape).to(orig_dtype)
    bytes_used = (tensor.numel() // 2) + scale.numel() * 2 + mn.numel() * 2
    return deq, bytes_used


def select_snapkv_sparse_prefix_indices(attn_scores_bh_w_p, prefix_len: int, keep_ratio: float, sink_tokens: int = 16):
    import torch

    keep_total = max(sink_tokens + 8, int(round(prefix_len * keep_ratio)))
    keep_total = min(prefix_len, keep_total)
    if keep_total >= prefix_len:
        return torch.arange(prefix_len, device=attn_scores_bh_w_p.device)

    sink_idx = torch.arange(sink_tokens, device=attn_scores_bh_w_p.device)
    rem_scores = attn_scores_bh_w_p[..., sink_tokens:prefix_len].mean(dim=(0, 1, 2))
    heavy_k = keep_total - sink_tokens
    top_rel = torch.topk(rem_scores, k=heavy_k, dim=-1).indices + sink_tokens
    merged = torch.cat([sink_idx, top_rel], dim=0)
    return torch.sort(merged).values


def batched_cached_greedy_generate(
    model,
    tokenizer,
    messages_list: list[list[dict]],
    *,
    batch_size: int = 30,
    max_new_tokens: int = 128,
    device,
) -> list[dict]:
    import torch
    from experiments.cross_domain_prefix_lora.surgical_json_csd_trainer import (
        chat_prompt_and_target_ids,
    )

    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id)
    orig_cache = getattr(model.config, "use_cache", False)
    results: list[dict] = []

    with torch.no_grad():
        try:
            model.config.use_cache = True
            for off in range(0, len(messages_list), batch_size):
                chunk = messages_list[off : off + batch_size]
                p_lists = [chat_prompt_and_target_ids(tokenizer, m)[0] for m in chunk]
                max_p = max(len(p) for p in p_lists)
                bsz = len(chunk)

                inp = torch.full((bsz, max_p), pad_id, dtype=torch.long, device=device)
                mask = torch.zeros((bsz, max_p), dtype=torch.long, device=device)
                for row_i, p_ids in enumerate(p_lists):
                    plen = len(p_ids)
                    inp[row_i, max_p - plen :] = torch.tensor(p_ids, dtype=torch.long, device=device)
                    mask[row_i, max_p - plen :] = 1

                pos = (mask.cumsum(dim=-1) - 1).clamp(min=0)
                out = model(
                    input_ids=inp,
                    attention_mask=mask,
                    position_ids=pos,
                    use_cache=True,
                    return_dict=True,
                )
                pkv = out.past_key_values
                nxt = torch.argmax(out.logits[:, -1], dim=-1)
                del out

                gen_cols = [nxt.tolist()]
                finished = (nxt == eos_id)
                cur_mask = mask

                for _step in range(max_new_tokens - 1):
                    if bool(finished.all()):
                        break
                    cur_mask = torch.cat(
                        [cur_mask, torch.ones((bsz, 1), dtype=torch.long, device=device)], dim=-1
                    )
                    step_pos = (cur_mask.sum(dim=-1, keepdim=True) - 1).clamp(min=0)
                    st_out = model(
                        input_ids=nxt.unsqueeze(-1),
                        attention_mask=cur_mask,
                        position_ids=step_pos,
                        past_key_values=pkv,
                        use_cache=True,
                        return_dict=True,
                    )
                    pkv = st_out.past_key_values
                    nxt = torch.argmax(st_out.logits[:, -1], dim=-1)
                    del st_out
                    gen_cols.append(nxt.tolist())
                    finished = finished | (nxt == eos_id)

                del pkv
                n_steps = len(gen_cols)
                for row_i in range(bsz):
                    seq_g = [gen_cols[s_i][row_i] for s_i in range(n_steps)]
                    if eos_id in seq_g:
                        epos = seq_g.index(eos_id)
                        stop_eos = True
                        dec_ids = seq_g[:epos]
                    else:
                        stop_eos = False
                        dec_ids = seq_g
                    txt = tokenizer.decode(
                        dec_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                    )
                    results.append({"text": txt, "stop_eos": stop_eos, "input_tokens": len(p_lists[row_i])})
        finally:
            model.config.use_cache = orig_cache

    return results


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen8b-path", required=True)
    parser.add_argument("--qwen06b-path", default="")
    parser.add_argument("--legal-dense-adapter-path", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    selected_gpu = _select_idle_a100()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu["index"])

    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "experiments/lora_necessity"))

    import torch
    from torch import nn
    from torch.nn import functional
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    from kl_anchored_lora_ablation import _compute_trajectory_kl_loss
    from experiments.cross_domain_prefix_lora.cross_domain_cot_mlp_ablation import (
        _collect_qkv_and_mlp_targets,
        render_cot_messages,
    )
    from experiments.cross_domain_prefix_lora.dense_compilation_and_residual_study import (
        Rank16LoRAWrapper,
        render_dense_student_messages,
    )
    from experiments.cross_domain_prefix_lora.domain_sop_contracts import (
        _format_legal_vignette,
        build_and_verify_cross_domain_dataset,
        legal_sop_evaluate,
        score_structured_sop_prediction,
    )
    from experiments.cross_domain_prefix_lora.surgical_json_csd_trainer import (
        chat_prompt_and_target_ids,
        prepare_multimodal_background_trajectories,
    )

    torch.cuda.init()
    device = torch.device("cuda:0")

    bbh = build_public_bbh_suite(seed=2026)
    public_bundle = build_and_verify_cross_domain_dataset()["public_bundle"]

    def _run_bbh_multi_seed_on_model(model_path: str, model_tag: str, lora_rank: int) -> dict:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, local_files_only=True)
        mod = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(device)
        mod.eval()
        mod.requires_grad_(False)

        def _long_msgs(q_str: str) -> list[dict]:
            return [{"role": "user", "content": f"{bbh['long_prefix_text']}\n\n--- Current Task ---\nQ: {q_str}\nA:"}]

        def _short_msgs(q_str: str) -> list[dict]:
            return [{"role": "user", "content": f"Solve this symbolic reasoning task step-by-step with a `TRACE:` line first.\nQ: {q_str}\nA:"}]

        short_test_batch = [_short_msgs(it["question"]) for it in bbh["test_items"]]
        long_test_batch = [_long_msgs(it["question"]) for it in bbh["test_items"]]

        base_outs = batched_cached_greedy_generate(mod, tok, short_test_batch, batch_size=30, max_new_tokens=128, device=device)
        text_outs = batched_cached_greedy_generate(mod, tok, long_test_batch, batch_size=20, max_new_tokens=128, device=device)
        base_flags = [score_bbh_prediction(o["text"], it["answer"]) for o, it in zip(base_outs, bbh["test_items"])]
        text_flags = [score_bbh_prediction(o["text"], it["answer"]) for o, it in zip(text_outs, bbh["test_items"])]

        bg_t = prepare_multimodal_background_trajectories(mod, tok, public_bundle["background_anchors"][:8], max_new_tokens=96)

        distill_pool = []
        for ex in bbh["exemplars"]:
            other_ex = [e for e in bbh["exemplars"] if e["id"] != ex["id"]]
            loo_hdr = "\n".join(
                [bbh["long_prefix_text"].split("\n--- Exemplar")[0]]
                + [f"\n--- Exemplar {e['id']} ({e['subtask']}) ---\nQ: {e['question']}\nA:\n{e['cot_target']}" for e in other_ex]
            )
            tm = [{"role": "user", "content": f"{loo_hdr}\n\n--- Current Task ---\nQ: {ex['question']}\nA:"}]
            sm = _short_msgs(ex["question"])
            tp, tt = chat_prompt_and_target_ids(tok, tm, answer_text=ex["cot_target"])
            sp, st = chat_prompt_and_target_ids(tok, sm, answer_text=ex["cot_target"])
            if tt == st and tt is not None:
                tin = torch.tensor([(tp + tt)[:-1]], dtype=torch.long, device=device)
                with torch.no_grad():
                    tout = mod(input_ids=tin, attention_mask=torch.ones_like(tin), use_cache=False, return_dict=True)
                    tsl = tout.logits[0, len(tp) - 1 : len(tp) - 1 + len(tt)].detach().to(torch.bfloat16).cpu().clone()
                del tout, tin
                distill_pool.append({"pool": "seed", "s_full": sp + st, "s_plen": len(sp), "alen": len(st), "t_ids": st, "t_sl": tsl})

        unl_outs = batched_cached_greedy_generate(
            mod,
            tok,
            [_long_msgs(u_it["question"]) for u_it in bbh["train_unlabeled"]],
            batch_size=16,
            max_new_tokens=128,
            device=device,
        )
        for u_it, gu in zip(bbh["train_unlabeled"], unl_outs):
            if not gu["stop_eos"] or "FINAL_ANSWER:" not in gu["text"]:
                continue
            tm = _long_msgs(u_it["question"])
            sm = _short_msgs(u_it["question"])
            tp, tt = chat_prompt_and_target_ids(tok, tm, answer_text=gu["text"])
            sp, st = chat_prompt_and_target_ids(tok, sm, answer_text=gu["text"])
            if tt == st and tt is not None:
                tin = torch.tensor([(tp + tt)[:-1]], dtype=torch.long, device=device)
                with torch.no_grad():
                    tout = mod(input_ids=tin, attention_mask=torch.ones_like(tin), use_cache=False, return_dict=True)
                    tsl = tout.logits[0, len(tp) - 1 : len(tp) - 1 + len(tt)].detach().to(torch.bfloat16).cpu().clone()
                del tout, tin
                distill_pool.append({"pool": "extra", "s_full": sp + st, "s_plen": len(sp), "alen": len(st), "t_ids": st, "t_sl": tsl})

        seed_runs = []
        for run_seed in (42, 137, 2026):
            targets = [
                (f"model.layers.{i}.self_attn.{nm}", getattr(mod.model.layers[i].self_attn, nm))
                for i in range(len(mod.model.layers))
                for nm in ("q_proj", "v_proj")
            ]
            origs = []
            wraps = []
            torch.manual_seed(run_seed)
            torch.cuda.manual_seed(run_seed)
            rng_py = random.Random(run_seed)
            shuffled_pool = list(distill_pool)
            rng_py.shuffle(shuffled_pool)
            try:
                for p_s, b_l in targets:
                    pp, an = p_s.rsplit(".", 1)
                    pm = mod.get_submodule(pp)
                    wm = Rank16LoRAWrapper.wrap_module(torch, nn, b_l, rank=lora_rank, alpha=float(lora_rank * 2))
                    origs.append((pm, an, b_l))
                    setattr(pm, an, wm)
                    wraps.append(wm)
                t_pars = [p for w in wraps for p in (w.adapter_a, w.adapter_b)]
                opt = torch.optim.AdamW(t_pars, lr=0.00028, weight_decay=0.01)
                steps = 96
                for st_i in range(steps):
                    lr_c = 0.00028 * (1.0 - 0.85 * (st_i / float(steps - 1)))
                    for pg in opt.param_groups:
                        pg["lr"] = lr_c
                    opt.zero_grad(set_to_none=True)
                    it = shuffled_pool[st_i % len(shuffled_pool)]
                    inp = torch.tensor([it["s_full"][:-1]], dtype=torch.long, device=device)
                    out = mod(input_ids=inp, attention_mask=torch.ones_like(inp), use_cache=False, return_dict=True)
                    ssl = out.logits[0, it["s_plen"] - 1 : it["s_plen"] - 1 + it["alen"]].float()
                    tsl = it["t_sl"].to(device=device, dtype=torch.float32)
                    kl = functional.kl_div(
                        functional.log_softmax(ssl, dim=-1),
                        functional.log_softmax(tsl, dim=-1),
                        log_target=True,
                        reduction="batchmean",
                    )
                    tgt = torch.tensor(it["t_ids"], dtype=torch.long, device=device)
                    ce = functional.cross_entropy(ssl, tgt)
                    (1.5 * kl + (0.60 if it["pool"] == "seed" else 0.25) * ce).backward()
                    del out, inp, ssl, tsl, kl, ce
                    b_it = bg_t[st_i % len(bg_t)]
                    b_kl, _, _ = _compute_trajectory_kl_loss(mod, b_it, device)
                    (2.0 * b_kl).backward()
                    del b_kl
                    torch.nn.utils.clip_grad_norm_(t_pars, 1.0)
                    opt.step()
                for w in wraps:
                    w.requires_grad_(False)
                    w.eval()

                lora_outs = batched_cached_greedy_generate(
                    mod, tok, short_test_batch, batch_size=30, max_new_tokens=128, device=device
                )
                lora_flags = [score_bbh_prediction(o["text"], it["answer"]) for o, it in zip(lora_outs, bbh["test_items"])]
                bf16_mib = round(sum(p.numel() * 2 for p in t_pars) / (1024.0 * 1024.0), 3)
            finally:
                for pm, an, bl in reversed(origs):
                    setattr(pm, an, bl)

            w2r_base = sum(1 for b, l in zip(base_flags, lora_flags) if (not b) and l)
            r2w_base = sum(1 for b, l in zip(base_flags, lora_flags) if b and (not l))
            seed_runs.append({
                "seed": run_seed,
                "lora_pass_count": sum(lora_flags),
                "lora_accuracy": round(sum(lora_flags) / float(len(lora_flags)), 4),
                "vs_base_w2r": w2r_base,
                "vs_base_r2w": r2w_base,
                "mcnemar_p_vs_base": exact_mcnemar_pvalue(w2r_base, r2w_base),
                "bf16_lora_mib": bf16_mib,
            })

        accs = [r["lora_accuracy"] for r in seed_runs]
        del mod, bg_t
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "model": model_tag,
            "n_test_items": len(bbh["test_items"]),
            "distill_trajectories": len(distill_pool),
            "base_0shot_count": sum(base_flags),
            "base_0shot_accuracy": round(sum(base_flags) / float(len(base_flags)), 4),
            "long_cot_count": sum(text_flags),
            "long_cot_accuracy": round(sum(text_flags) / float(len(text_flags)), 4),
            "lora_3seed_mean_accuracy": round(statistics.mean(accs), 4),
            "lora_3seed_std_accuracy": round(statistics.stdev(accs), 4),
            "seed_runs": seed_runs,
        }

    multi_model_bbh = []
    if args.qwen06b_path and Path(args.qwen06b_path).exists():
        multi_model_bbh.append(_run_bbh_multi_seed_on_model(args.qwen06b_path, "Qwen3-0.6B", lora_rank=8))
    multi_model_bbh.append(_run_bbh_multi_seed_on_model(args.qwen8b_path, "Qwen3-8B", lora_rank=8))

    rng_leg = random.Random(888888)
    eval40_legal = []
    seen_l = set()
    for vt in ("TIER_STRATEGIC", "TIER_CORE", "TIER_STD"):
        for dc in ("D0_BIO_PII", "D1_CUST_PII", "D2_CONFID", "D3_PUBLIC"):
            for idx_sub in range(4):
                cand = {
                    "vendor_tier": vt,
                    "data_class": dc,
                    "cross_border": rng_leg.choice(["NONE", "EU_FRANKFURT", "SG_SINGAPORE"]),
                    "has_sec_assessment": rng_leg.choice([True, False]),
                    "governing_law": rng_leg.choice(["DOMESTIC_COURT", "NEUTRAL_ARBITRATION", "FOREIGN_LITIGATION"]) if idx_sub == 0 else "NEUTRAL_ARBITRATION",
                    "ip_indemnity_uncapped": rng_leg.choice([True, False]) if idx_sub == 1 else True,
                    "auto_renew_months": rng_leg.choice([12, 24, 36]),
                    "terminate_for_conv": rng_leg.choice([True, False]),
                    "proposed_cap_x": rng_leg.choice([1, 2, 3, 5]),
                }
                sig = json.dumps(cand, sort_keys=True)
                if sig in seen_l:
                    continue
                seen_l.add(sig)
                cid = f"L_BENCH_{len(eval40_legal) + 1:03d}"
                obj = {"id": cid, "input": cand, "expected": legal_sop_evaluate(cand)}
                obj["vignette"] = _format_legal_vignette(obj)
                eval40_legal.append(obj)
                if len(eval40_legal) >= 40:
                    break
            if len(eval40_legal) >= 40:
                break
        if len(eval40_legal) >= 40:
            break

    tok8 = AutoTokenizer.from_pretrained(args.qwen8b_path, trust_remote_code=False, local_files_only=True)
    sample_long_msgs = render_cot_messages("legal_compliance", eval40_legal[0], public_bundle, is_long=True)
    sample_short_msgs = render_dense_student_messages("legal_compliance", eval40_legal[0], include_micro_residual=False)
    l_ids, _ = chat_prompt_and_target_ids(tok8, sample_long_msgs)
    s_ids, _ = chat_prompt_and_target_ids(tok8, sample_short_msgs)
    suf = 0
    while l_ids[-1 - suf] == s_ids[-1 - suf]:
        suf += 1
    prefix_only_ids = l_ids[: len(l_ids) - suf]
    prefix_len = len(prefix_only_ids)

    mod8_eager = AutoModelForCausalLM.from_pretrained(
        args.qwen8b_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    mod8_eager.eval()
    mod8_eager.requires_grad_(False)
    with torch.no_grad():
        p_inp = torch.tensor([prefix_only_ids], dtype=torch.long, device=device)
        p_out = mod8_eager(
            input_ids=p_inp,
            attention_mask=torch.ones_like(p_inp),
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
        obs_win = min(32, prefix_len)
        keep20_per_layer = []
        keep10_per_layer = []
        for attn_t in p_out.attentions:
            obs_scores = attn_t[:, :, -obs_win:, :prefix_len].detach()
            keep20_per_layer.append(select_snapkv_sparse_prefix_indices(obs_scores, prefix_len, keep_ratio=0.20))
            keep10_per_layer.append(select_snapkv_sparse_prefix_indices(obs_scores, prefix_len, keep_ratio=0.10))
        del p_out, p_inp, mod8_eager
    gc.collect()
    torch.cuda.empty_cache()

    mod8_sdpa = AutoModelForCausalLM.from_pretrained(
        args.qwen8b_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    mod8_sdpa.eval()
    mod8_sdpa.requires_grad_(False)

    def _extract_kv_list(pkv_obj) -> list[tuple]:
        if hasattr(pkv_obj, "key_cache") and hasattr(pkv_obj, "value_cache") and len(pkv_obj.key_cache) > 0:
            return list(zip(pkv_obj.key_cache, pkv_obj.value_cache))
        if hasattr(pkv_obj, "layers") and len(pkv_obj.layers) > 0:
            return [(lyr.keys, lyr.values) for lyr in pkv_obj.layers]
        if hasattr(pkv_obj, "to_legacy_cache"):
            return list(pkv_obj.to_legacy_cache())
        return [pkv_obj[i] for i in range(len(pkv_obj))]

    with torch.no_grad():
        p_inp_s = torch.tensor([prefix_only_ids], dtype=torch.long, device=device)
        p_out_s = mod8_sdpa(
            input_ids=p_inp_s,
            attention_mask=torch.ones_like(p_inp_s),
            use_cache=True,
            return_dict=True,
        )
        raw_kv_pairs = [(k.detach().clone(), v.detach().clone()) for k, v in _extract_kv_list(p_out_s.past_key_values)]
        del p_out_s, p_inp_s

    def _build_batch_cache(mode: str, bsz: int):
        cache = DynamicCache()
        single_bytes = 0
        for l_idx, (k_t, v_t) in enumerate(raw_kv_pairs):
            k_c = k_t.detach().clone()
            v_c = v_t.detach().clone()
            if mode == "bf16_full":
                single_bytes += (k_c.numel() + v_c.numel()) * 2
            elif mode == "int8_kv":
                k_c, kb = quantize_dequantize_tensor_int8_per_token(k_c)
                v_c, vb = quantize_dequantize_tensor_int8_per_token(v_c)
                single_bytes += kb + vb
            elif mode == "int4_kv":
                k_c, kb = quantize_dequantize_tensor_int4_group32(k_c)
                v_c, vb = quantize_dequantize_tensor_int4_group32(v_c)
                single_bytes += kb + vb
            elif mode == "snapkv_20pct":
                idx_sel = keep20_per_layer[l_idx]
                k_c = k_c.index_select(2, idx_sel).contiguous()
                v_c = v_c.index_select(2, idx_sel).contiguous()
                single_bytes += (k_c.numel() + v_c.numel()) * 2
            elif mode == "snapkv_10pct":
                idx_sel = keep10_per_layer[l_idx]
                k_c = k_c.index_select(2, idx_sel).contiguous()
                v_c = v_c.index_select(2, idx_sel).contiguous()
                single_bytes += (k_c.numel() + v_c.numel()) * 2
            else:
                raise ValueError(mode)
            cache.update(k_c.expand(bsz, -1, -1, -1).contiguous(), v_c.expand(bsz, -1, -1, -1).contiguous(), l_idx)
        return cache, single_bytes

    def _eval_with_prefix_cache_mode(mode: str) -> dict:
        _, prefix_bytes = _build_batch_cache(mode, 1)
        eos_id = int(tok8.eos_token_id)
        pad_id = int(tok8.pad_token_id if tok8.pad_token_id is not None else eos_id)
        pass_cnt = 0
        f_ok = 0
        t0 = time.perf_counter_ns()
        chunk_sz = 10
        with torch.no_grad():
            for off in range(0, len(eval40_legal), chunk_sz):
                sub_items = eval40_legal[off : off + chunk_sz]
                bsz = len(sub_items)
                q_lists = [
                    chat_prompt_and_target_ids(
                        tok8, render_cot_messages("legal_compliance", c_it, public_bundle, is_long=True)
                    )[0][prefix_len:]
                    for c_it in sub_items
                ]
                max_q = max(len(q) for q in q_lists)
                kv_c, _ = _build_batch_cache(mode, bsz)
                cached_len = _extract_kv_list(kv_c)[0][0].shape[2]

                q_in = torch.full((bsz, max_q), pad_id, dtype=torch.long, device=device)
                q_mask = torch.zeros((bsz, max_q), dtype=torch.long, device=device)
                for r_i, ql in enumerate(q_lists):
                    q_in[r_i, max_q - len(ql) :] = torch.tensor(ql, dtype=torch.long, device=device)
                    q_mask[r_i, max_q - len(ql) :] = 1

                full_mask = torch.cat(
                    [torch.ones((bsz, cached_len), dtype=torch.long, device=device), q_mask], dim=-1
                )
                pos_ids = (q_mask.cumsum(dim=-1) - 1 + prefix_len).clamp(min=0)
                c_pos = torch.arange(cached_len, cached_len + max_q, dtype=torch.long, device=device)

                out = mod8_sdpa(
                    input_ids=q_in,
                    attention_mask=full_mask,
                    position_ids=pos_ids,
                    cache_position=c_pos,
                    past_key_values=kv_c,
                    use_cache=True,
                    return_dict=True,
                )
                kv_c = out.past_key_values
                nxt = torch.argmax(out.logits[:, -1], dim=-1)
                del out
                gen_cols = [nxt.tolist()]
                fin = nxt == eos_id
                cur_cpos = cached_len + max_q
                cur_pos = pos_ids[:, -1:] + 1

                for _st in range(160 - 1):
                    if bool(fin.all()):
                        break
                    full_mask = torch.cat(
                        [full_mask, torch.ones((bsz, 1), dtype=torch.long, device=device)], dim=-1
                    )
                    st_out = mod8_sdpa(
                        input_ids=nxt.unsqueeze(-1),
                        attention_mask=full_mask,
                        position_ids=cur_pos,
                        cache_position=torch.tensor([cur_cpos], dtype=torch.long, device=device),
                        past_key_values=kv_c,
                        use_cache=True,
                        return_dict=True,
                    )
                    kv_c = st_out.past_key_values
                    nxt = torch.argmax(st_out.logits[:, -1], dim=-1)
                    del st_out
                    gen_cols.append(nxt.tolist())
                    fin = fin | (nxt == eos_id)
                    cur_cpos += 1
                    cur_pos = cur_pos + 1

                del kv_c
                for r_i, c_it in enumerate(sub_items):
                    seq_g = [gen_cols[s_i][r_i] for s_i in range(len(gen_cols))]
                    dec = seq_g[: seq_g.index(eos_id)] if eos_id in seq_g else seq_g
                    txt = tok8.decode(dec, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    sc = score_structured_sop_prediction("legal_compliance", txt, c_it["expected"])
                    if sc["strict_pass"]:
                        pass_cnt += 1
                    f_ok += sc["matched_fields"]

        el_s = (time.perf_counter_ns() - t0) / 1e9
        return {
            "mode": mode,
            "prefix_resident_mib": round(prefix_bytes / (1024.0 * 1024.0), 2),
            "compression_ratio_vs_bf16_kv": round(((prefix_len * 36 * 2 * 8 * 128 * 2) / max(prefix_bytes, 1)), 2),
            "strict_pass": f"{pass_cnt}/40",
            "strict_pass_rate": round(pass_cnt / 40.0, 4),
            "field_accuracy": round(f_ok / 200.0, 4),
            "eval_40_seconds": round(el_s, 2),
        }

    kv_compression_comparison = [
        _eval_with_prefix_cache_mode("bf16_full"),
        _eval_with_prefix_cache_mode("int8_kv"),
        _eval_with_prefix_cache_mode("int4_kv"),
        _eval_with_prefix_cache_mode("snapkv_20pct"),
        _eval_with_prefix_cache_mode("snapkv_10pct"),
    ]
    del raw_kv_pairs
    gc.collect()
    torch.cuda.empty_cache()

    leg_loaded = torch.load(args.legal_dense_adapter_path, map_location="cpu", weights_only=True)
    leg_ad = leg_loaded["adapter"]
    leg_r = int(leg_loaded.get("rank", 16))
    leg_alpha = float(leg_loaded.get("alpha", 32.0))

    targets8 = _collect_qkv_and_mlp_targets(mod8_sdpa)
    origs8 = []
    wraps8 = []
    try:
        for p_s, b_l in targets8:
            pp, an = p_s.rsplit(".", 1)
            pm = mod8_sdpa.get_submodule(pp)
            wm = Rank16LoRAWrapper.wrap_module(torch, nn, b_l, rank=leg_r, alpha=leg_alpha)
            with torch.no_grad():
                wm.adapter_a.copy_(leg_ad[f"{p_s}.adapter_a"].to(device=device, dtype=torch.float32))
                wm.adapter_b.copy_(leg_ad[f"{p_s}.adapter_b"].to(device=device, dtype=torch.float32))
            wm.requires_grad_(False)
            wm.eval()
            origs8.append((pm, an, b_l))
            setattr(pm, an, wm)
            wraps8.append(wm)

        lora_bf16_bytes = sum((w.adapter_a.numel() + w.adapter_b.numel()) * 2 for w in wraps8)
        for use_res, tag_m in ((False, "lora_dense_r16_pure"), (True, "lora_dense_r16_plus_micro_table")):
            msgs_all = [render_dense_student_messages("legal_compliance", c_it, include_micro_residual=use_res) for c_it in eval40_legal]
            t0 = time.perf_counter_ns()
            outs_all = batched_cached_greedy_generate(mod8_sdpa, tok8, msgs_all, batch_size=20, max_new_tokens=160, device=device)
            el_s = (time.perf_counter_ns() - t0) / 1e9
            p_cnt = 0
            f_ok = 0
            for g, c_it in zip(outs_all, eval40_legal):
                sc = score_structured_sop_prediction("legal_compliance", g["text"], c_it["expected"])
                if sc["strict_pass"]:
                    p_cnt += 1
                f_ok += sc["matched_fields"]
            res_extra_kv_bytes = 0
            if use_res:
                pure_len = len(chat_prompt_and_target_ids(tok8, render_dense_student_messages("legal_compliance", eval40_legal[0], include_micro_residual=False))[0])
                res_extra_kv_bytes = max(0, outs_all[0]["input_tokens"] - pure_len) * 36 * 2 * 8 * 128 * 2
            tot_res_bytes = lora_bf16_bytes + res_extra_kv_bytes
            kv_compression_comparison.append({
                "mode": tag_m,
                "prefix_resident_mib": round(tot_res_bytes / (1024.0 * 1024.0), 2),
                "compression_ratio_vs_bf16_kv": round(((prefix_len * 36 * 2 * 8 * 128 * 2) / max(tot_res_bytes, 1)), 2),
                "strict_pass": f"{p_cnt}/40",
                "strict_pass_rate": round(p_cnt / 40.0, 4),
                "field_accuracy": round(f_ok / 200.0, 4),
                "eval_40_seconds": round(el_s, 2),
            })
    finally:
        for pm, an, bl in reversed(origs8):
            setattr(pm, an, bl)

    report = {
        "kind": "PUBLIC_BBH_MULTISEED_AND_KV_COMPRESSION_REPORT",
        "gpu_summary": {k: v for k, v in selected_gpu.items() if k != "index"},
        "public_bbh_multimodel_3seed": multi_model_bbh,
        "kv_compression_pareto_n40": kv_compression_comparison,
    }
    Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
