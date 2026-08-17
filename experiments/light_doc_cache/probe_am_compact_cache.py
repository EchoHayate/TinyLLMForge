from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--text-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--max-sample-tokens", type=int, default=512)
    p.add_argument("--skip-prefix-tokens", type=int, default=128)
    p.add_argument("--budgets", default="16,32,64,128")
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--accept-r2", type=float, default=0.80)
    p.add_argument("--borderline-r2", type=float, default=0.50)
    return p.parse_args()


def parse_ints(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def pick_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def pick_dtype(name, device):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32


def sample_indices(seq_len, max_samples, skip):
    idx = torch.arange(min(max(skip, 0), seq_len), seq_len, dtype=torch.long)
    if max_samples > 0 and idx.numel() > max_samples:
        take = torch.linspace(0, idx.numel() - 1, max_samples).round().long()
        idx = idx[take]
    if idx.numel() < 32:
        raise RuntimeError(f"need >=32 sampled positions, got {idx.numel()}")
    return idx


def normalize_cache(x, expected_heads=None):
    if expected_heads is not None:
        if x.shape[1] == expected_heads:
            return x
        if x.shape[2] == expected_heads:
            return x.permute(0, 2, 1, 3).contiguous()
    if x.shape[1] <= 128 and x.shape[2] >= x.shape[1]:
        return x
    return x.permute(0, 2, 1, 3).contiguous()


def legacy_pkv(pkv):
    if hasattr(pkv, "to_legacy_cache"):
        pkv = pkv.to_legacy_cache()
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        return list(zip(pkv.key_cache, pkv.value_cache))
    return [(x[0], x[1]) for x in pkv]


@torch.inference_mode()
def collect_qkv(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    kwargs = dict(trust_remote_code=True, local_files_only=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, **kwargs)
    model.eval().to(device)

    text = Path(args.text_file).read_text(encoding="utf-8")
    enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_tokens, add_special_tokens=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    seq_len = int(enc["input_ids"].shape[1])
    full_pos = torch.arange(seq_len, device=device).unsqueeze(0)

    t0 = time.perf_counter()
    out = model(**enc, use_cache=True, output_hidden_states=True, return_dict=True)
    forward_s = time.perf_counter() - t0

    idx = sample_indices(seq_len, args.max_sample_tokens, args.skip_prefix_tokens)
    idx_dev = idx.to(device)
    expected_kv = getattr(model.config, "num_key_value_heads", None) or getattr(model.config, "num_attention_heads", None)

    key_layers = []
    value_layers = []
    for k, v in legacy_pkv(out.past_key_values):
        k = normalize_cache(k, expected_kv)
        v = normalize_cache(v, expected_kv)
        key_layers.append(k[0, :, idx.to(k.device), :].detach().cpu().float().contiguous())
        value_layers.append(v[0, :, idx.to(v.device), :].detach().cpu().float().contiguous())
    keys = torch.stack(key_layers, 0)
    values = torch.stack(value_layers, 0)

    query_layers = []
    for layer_idx, layer in enumerate(model.model.layers):
        h = out.hidden_states[layer_idx]
        attn = layer.self_attn
        hidden_shape = (*h.shape[:-1], -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(h).view(hidden_shape)).transpose(1, 2)
        pos_emb = model.model.rotary_emb(h, full_pos)
        dummy_k = torch.zeros((h.shape[0], expected_kv, h.shape[1], attn.head_dim), device=h.device, dtype=q.dtype)
        q, _ = apply_rotary_pos_emb(q, dummy_k, *pos_emb)
        query_layers.append(q[0, :, idx_dev, :].detach().cpu().float().contiguous())
    queries = torch.stack(query_layers, 0)

    meta = dict(model=args.model, text_file=args.text_file, device=str(device), dtype=str(dtype),
                seq_len=seq_len, sampled_tokens=int(idx.numel()), sample_start=int(idx[0]), sample_end=int(idx[-1]),
                num_layers=int(keys.shape[0]), num_query_heads=int(queries.shape[1]),
                num_kv_heads=int(keys.shape[1]), head_dim=int(keys.shape[3]), forward_s=forward_s)
    return queries, keys, values, idx, meta


def attention_weights(q_group, k, q_pos, k_pos):
    scores = torch.einsum("gqd,kd->gqk", q_group.float(), k.float()) / math.sqrt(q_group.shape[-1])
    mask = k_pos.view(1, 1, -1) <= q_pos.view(1, -1, 1)
    scores = scores.masked_fill(~mask.to(scores.device), float("-inf"))
    return torch.softmax(scores, dim=-1)


def attention_output(q_group, k, v, q_pos, k_pos):
    probs = attention_weights(q_group, k, q_pos, k_pos)
    return torch.einsum("gqk,kd->gqd", probs, v.float())


def r2_score(y, pred):
    y = y.float(); pred = pred.float()
    sse = (pred - y).pow(2).sum()
    sst = (y - y.mean(dim=(0, 1), keepdim=True)).pow(2).sum()
    return float(1.0 - sse / sst.clamp_min(1e-8))


def select_uniform(eligible, budget):
    if eligible.numel() <= budget:
        return eligible
    take = torch.linspace(0, eligible.numel() - 1, budget).round().long()
    return eligible[take].unique(sorted=True)


def select_highest(probs_train, eligible, budget):
    # probs_train [G,Q,T]. Select by total probability mass over train queries.
    scores = probs_train.sum(dim=(0, 1))
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask[eligible] = True
    scores = scores.masked_fill(~mask, float("-inf"))
    k = min(int(budget), int(eligible.numel()))
    top = torch.topk(scores, k=k).indices
    # Always keep the first eligible token as an attention-sink candidate.
    top = torch.unique(torch.cat([top, eligible[:1]]), sorted=True)
    if top.numel() > budget:
        # Drop the lowest-score non-first item if preserving sink overflowed.
        keep_scores = scores[top]
        order = torch.argsort(keep_scores, descending=True)
        top = torch.unique(torch.cat([eligible[:1], top[order[: budget - 1]]]), sorted=True)
    return top


def fit_values(attn_train, target_train, ridge):
    # attn_train [G,Q,B], target_train [G,Q,D]
    a = attn_train.reshape(-1, attn_train.shape[-1]).float()
    y = target_train.reshape(-1, target_train.shape[-1]).float()
    eye = torch.eye(a.shape[1], dtype=a.dtype, device=a.device)
    return torch.linalg.solve(a.T @ a + ridge * eye, a.T @ y)


def eval_selected(q_group, k, v, positions, train_idx, val_idx, selected, true_train, true_val, ridge):
    k_sel = k[selected]
    v_sel = v[selected]
    pos_sel = positions[selected]

    direct_train = attention_output(q_group[:, train_idx, :], k_sel, v_sel, positions[train_idx], pos_sel)
    direct_val = attention_output(q_group[:, val_idx, :], k_sel, v_sel, positions[val_idx], pos_sel)
    direct_train_r2 = r2_score(true_train, direct_train)
    direct_val_r2 = r2_score(true_val, direct_val)

    attn_train = attention_weights(q_group[:, train_idx, :], k_sel, positions[train_idx], pos_sel)
    compact_v = fit_values(attn_train, true_train, ridge)
    fit_train = torch.einsum("gqb,bd->gqd", attn_train.float(), compact_v.float())
    attn_val = attention_weights(q_group[:, val_idx, :], k_sel, positions[val_idx], pos_sel)
    fit_val = torch.einsum("gqb,bd->gqd", attn_val.float(), compact_v.float())
    fit_train_r2 = r2_score(true_train, fit_train)
    fit_val_r2 = r2_score(true_val, fit_val)
    return direct_train_r2, direct_val_r2, fit_train_r2, fit_val_r2


def probe_head(q_group, k, v, positions, budgets, train_frac, ridge):
    t = k.shape[0]
    split = min(max(16, int(round(t * train_frac))), t - 16)
    all_idx = torch.arange(t, device=k.device)
    train_idx = torch.arange(0, split, device=k.device)
    holdout_val_idx = torch.arange(split, t, device=k.device)
    all_val_idx = all_idx

    rows = []
    regimes = [
        ("in_sample", all_idx, all_val_idx, all_idx),
        ("holdout", train_idx, holdout_val_idx, train_idx),
    ]
    true_all = attention_output(q_group, k, v, positions, positions)
    for regime, tr_idx, va_idx, eligible in regimes:
        true_train = true_all[:, tr_idx, :]
        true_val = true_all[:, va_idx, :]
        probs_train = attention_weights(q_group[:, tr_idx, :], k, positions[tr_idx], positions)
        for budget in budgets:
            for selector_name, selected in [
                ("uniform", select_uniform(eligible, budget)),
                ("highest", select_highest(probs_train, eligible, budget)),
            ]:
                if selected.numel() < 2:
                    continue
                dtr, dva, ftr, fva = eval_selected(
                    q_group, k, v, positions, tr_idx, va_idx, selected, true_train, true_val, ridge
                )
                rows.append(dict(
                    regime=regime,
                    selector=selector_name,
                    budget=int(selected.numel()),
                    direct_train_r2=dtr,
                    direct_val_r2=dva,
                    fitv_train_r2=ftr,
                    fitv_val_r2=fva,
                ))
    return rows


def summarize(rows, sampled_tokens):
    groups = {}
    for row in rows:
        key = (row["regime"], row["selector"], row["budget"])
        groups.setdefault(key, []).append(row)
    summary_rows = []
    for (regime, selector, budget), vals in sorted(groups.items()):
        fit_vals = [v["fitv_val_r2"] for v in vals]
        direct_vals = [v["direct_val_r2"] for v in vals]
        summary_rows.append(dict(
            regime=regime,
            selector=selector,
            budget=budget,
            budget_fraction=budget / sampled_tokens,
            heads=len(vals),
            direct_val_mean=sum(direct_vals) / len(direct_vals),
            fitv_val_mean=sum(fit_vals) / len(fit_vals),
            fitv_val_p50=percentile(fit_vals, 0.50),
            fitv_val_p90=percentile(fit_vals, 0.90),
            fitv_val_ge_050=sum(x >= 0.50 for x in fit_vals) / len(fit_vals),
            fitv_val_ge_080=sum(x >= 0.80 for x in fit_vals) / len(fit_vals),
        ))
    return summary_rows


def percentile(vals, p):
    vals = sorted(float(x) for x in vals)
    if not vals:
        return float("nan")
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * p))))
    return vals[idx]


def write_rows(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) and math.isfinite(v) else v) for k, v in row.items()})


def decide(summary_rows):
    holdout = [r for r in summary_rows if r["regime"] == "holdout" and r["selector"] == "highest"]
    best_holdout = max(holdout, key=lambda r: r["fitv_val_mean"], default=None)
    in_sample = [r for r in summary_rows if r["regime"] == "in_sample" and r["selector"] == "highest"]
    best_in = max(in_sample, key=lambda r: r["fitv_val_mean"], default=None)
    if best_holdout and best_holdout["fitv_val_mean"] >= 0.50 and best_holdout["fitv_val_ge_050"] >= 0.50:
        return "AM_PROMISING", f"holdout highest fitV mean R2={best_holdout['fitv_val_mean']:.3f}, coverage@0.5={best_holdout['fitv_val_ge_050']:.2%}"
    if best_in and best_in["fitv_val_mean"] >= 0.80 and (not best_holdout or best_holdout["fitv_val_mean"] < 0.50):
        return "AM_UPPER_BOUND_ONLY", f"in-sample AM is high but holdout is weak; best holdout={best_holdout['fitv_val_mean']:.3f}"
    if best_holdout:
        return "AM_WEAK", f"best holdout highest fitV mean R2={best_holdout['fitv_val_mean']:.3f}, coverage@0.5={best_holdout['fitv_val_ge_050']:.2%}"
    return "AM_NO_DATA", "no holdout AM rows"


def main():
    args = parse_args()
    budgets = parse_ints(args.budgets)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    queries, keys, values, positions_cpu, meta = collect_qkv(args)
    positions = positions_cpu.to(device)
    group = queries.shape[1] // keys.shape[1]
    rows = []
    t0 = time.perf_counter()
    for layer in range(keys.shape[0]):
        for kv_head in range(keys.shape[1]):
            q_group = queries[layer, kv_head * group:(kv_head + 1) * group].to(device)
            k = keys[layer, kv_head].to(device)
            v = values[layer, kv_head].to(device)
            head_rows = probe_head(q_group, k, v, positions, budgets, args.train_frac, args.ridge)
            for row in head_rows:
                row.update(layer=int(layer), kv_head=int(kv_head), query_group=int(group))
                rows.append(row)
    metric_s = time.perf_counter() - t0
    summary_rows = summarize(rows, meta["sampled_tokens"])
    dec, reason = decide(summary_rows)

    write_rows(out / "am_head_rows.csv", rows)
    write_rows(out / "am_summary_by_budget.csv", summary_rows)
    summary = dict(
        decision=dec,
        decision_reason=reason,
        metadata=meta,
        settings=vars(args),
        metric_s=metric_s,
        summary_by_budget=summary_rows,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report = ["# AM-Style Compact Cache Probe", "", f"Decision: **{dec}**", "", reason, "",
              "| Regime | Selector | Budget | Budget Frac | FitV Val Mean R2 | FitV Val >=0.5 | FitV Val >=0.8 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary_rows:
        report.append(
            f"| {r['regime']} | {r['selector']} | {r['budget']} | {r['budget_fraction']:.2%} | "
            f"{r['fitv_val_mean']:.4f} | {r['fitv_val_ge_050']:.2%} | {r['fitv_val_ge_080']:.2%} |"
        )
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("decision", dec, reason)
    print("output_dir", out)


if __name__ == "__main__":
    main()
