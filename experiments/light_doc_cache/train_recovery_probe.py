from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from probe_am_compact_cache import (
    attention_output,
    attention_weights,
    collect_qkv,
    fit_values,
    parse_ints,
    pick_device,
    r2_score,
    select_highest,
    select_uniform,
)


class ResidualRecoveryMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, compact_output: torch.Tensor) -> torch.Tensor:
        return compact_output + self.net(compact_output)


class FusedResidualRecoveryMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, direct_output: torch.Tensor, fitv_output: torch.Tensor) -> torch.Tensor:
        features = torch.cat([direct_output, fitv_output, direct_output - fitv_output], dim=-1)
        return fitv_output + self.net(features)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument(
        "--text-file",
        action="append",
        required=True,
        help="Document file. Can be passed multiple times; each doc is probed independently.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--max-sample-tokens", type=int, default=512)
    p.add_argument("--skip-prefix-tokens", type=int, default=128)
    p.add_argument("--budgets", default="32,64,128")
    p.add_argument("--selector", default="highest", choices=["highest", "uniform"])
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-heads", type=int, default=0, help="Debug cap over layer/kv-head pairs; 0 means all.")
    p.add_argument("--start-layer", type=int, default=0, help="First layer index to probe.")
    p.add_argument("--end-layer", type=int, default=0, help="Exclusive end layer index; 0 means all remaining layers.")
    p.add_argument("--accept-r2", type=float, default=0.50)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    return p.parse_args()


def split_indices(token_count: int, train_frac: float, device: torch.device):
    if token_count < 32:
        raise RuntimeError(f"need >=32 sampled positions, got {token_count}")
    split = min(max(16, int(round(token_count * train_frac))), token_count - 16)
    train_idx = torch.arange(0, split, device=device)
    val_idx = torch.arange(split, token_count, device=device)
    return train_idx, val_idx


def select_tokens(selector: str, q_group, keys, positions, train_idx, budget):
    eligible = train_idx
    if selector == "uniform":
        return select_uniform(eligible, budget)
    probs_train = attention_weights(q_group[:, train_idx, :], keys, positions[train_idx], positions)
    return select_highest(probs_train, eligible, budget)


def compact_attention_output(q_group, keys, values, positions, selected, query_idx):
    return attention_output(
        q_group[:, query_idx, :],
        keys[selected],
        values[selected],
        positions[query_idx],
        positions[selected],
    )


def fitv_attention_output(q_group, keys, positions, selected, train_idx, query_idx, compact_values):
    attn = attention_weights(
        q_group[:, query_idx, :],
        keys[selected],
        positions[query_idx],
        positions[selected],
    )
    return torch.einsum("gqb,bd->gqd", attn.float(), compact_values.float())


def train_compact_values(attn_train, attn_val, y_train, y_val, init_values, epochs, lr, weight_decay):
    learned_values = nn.Parameter(init_values.detach().clone().float())
    opt = torch.optim.AdamW([learned_values], lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    initial_pred = torch.einsum("gqb,bd->gqd", attn_train.float(), learned_values.float())
    initial_loss = float(loss_fn(initial_pred, y_train).detach().cpu())
    best_values = learned_values.detach().clone()
    best_loss = initial_loss

    for _ in range(max(0, int(epochs))):
        opt.zero_grad(set_to_none=True)
        pred = torch.einsum("gqb,bd->gqd", attn_train.float(), learned_values.float())
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt.step()
        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_values = learned_values.detach().clone()

    train_pred = torch.einsum("gqb,bd->gqd", attn_train.float(), best_values.float())
    val_pred = torch.einsum("gqb,bd->gqd", attn_val.float(), best_values.float())
    final_loss = float(loss_fn(train_pred, y_train).detach().cpu())
    return {
        "values": best_values,
        "train_pred": train_pred,
        "val_pred": val_pred,
        "train_loss_initial": initial_loss,
        "train_loss_final": final_loss,
    }


def train_head_recovery(
    *,
    q_group,
    keys,
    values,
    positions,
    train_idx,
    val_idx,
    selected,
    teacher_all,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    ridge: float = 1.0,
):
    q_group = q_group.to(device).float()
    keys = keys.to(device).float()
    values = values.to(device).float()
    positions = positions.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    selected = selected.to(device)
    teacher_all = teacher_all.to(device).float()

    x_train = compact_attention_output(q_group, keys, values, positions, selected, train_idx)
    x_val = compact_attention_output(q_group, keys, values, positions, selected, val_idx)
    y_train = teacher_all[:, train_idx, :]
    y_val = teacher_all[:, val_idx, :]

    attn_train = attention_weights(q_group[:, train_idx, :], keys[selected], positions[train_idx], positions[selected])
    attn_val = attention_weights(q_group[:, val_idx, :], keys[selected], positions[val_idx], positions[selected])
    compact_values = fit_values(attn_train, y_train, ridge=ridge)
    fitv_train = torch.einsum("gqb,bd->gqd", attn_train.float(), compact_values.float())
    fitv_val = torch.einsum("gqb,bd->gqd", attn_val.float(), compact_values.float())
    learned = train_compact_values(
        attn_train=attn_train,
        attn_val=attn_val,
        y_train=y_train,
        y_val=y_val,
        init_values=compact_values,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
    )

    model = ResidualRecoveryMLP(dim=x_train.shape[-1], hidden_dim=hidden_dim).to(device)
    fused_model = FusedResidualRecoveryMLP(dim=x_train.shape[-1], hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    fused_opt = torch.optim.AdamW(fused_model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    initial_loss = float(loss_fn(model(x_train), y_train).detach().cpu())
    fused_initial_loss = float(loss_fn(fused_model(x_train, fitv_train), y_train).detach().cpu())
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    fused_best_state = {k: v.detach().cpu().clone() for k, v in fused_model.state_dict().items()}
    best_loss = initial_loss
    fused_best_loss = fused_initial_loss

    for _ in range(max(0, int(epochs))):
        opt.zero_grad(set_to_none=True)
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt.step()
        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        fused_opt.zero_grad(set_to_none=True)
        fused_pred = fused_model(x_train, fitv_train)
        fused_loss = loss_fn(fused_pred, y_train)
        fused_loss.backward()
        fused_opt.step()
        fused_loss_value = float(fused_loss.detach().cpu())
        if fused_loss_value < fused_best_loss:
            fused_best_loss = fused_loss_value
            fused_best_state = {k: v.detach().cpu().clone() for k, v in fused_model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    fused_model.load_state_dict({k: v.to(device) for k, v in fused_best_state.items()})
    with torch.no_grad():
        mlp_train = model(x_train)
        mlp_val = model(x_val)
        fused_train = fused_model(x_train, fitv_train)
        fused_val = fused_model(x_val, fitv_val)
        final_loss = float(loss_fn(mlp_train, y_train).detach().cpu())
        fused_final_loss = float(loss_fn(fused_train, y_train).detach().cpu())

    direct_train_r2 = r2_score(y_train, x_train)
    direct_val_r2 = r2_score(y_val, x_val)
    fitv_train_r2 = r2_score(y_train, fitv_train)
    fitv_val_r2 = r2_score(y_val, fitv_val)
    learned_train_r2 = r2_score(y_train, learned["train_pred"])
    learned_val_r2 = r2_score(y_val, learned["val_pred"])
    mlp_train_r2 = r2_score(y_train, mlp_train)
    mlp_val_r2 = r2_score(y_val, mlp_val)
    fused_train_r2 = r2_score(y_train, fused_train)
    fused_val_r2 = r2_score(y_val, fused_val)

    recovery_candidates = [
        ("direct_compact", direct_train_r2, direct_val_r2),
        ("ridge_value_recovery", fitv_train_r2, fitv_val_r2),
        ("learned_compact_values", learned_train_r2, learned_val_r2),
        ("mlp_residual", mlp_train_r2, mlp_val_r2),
        ("fused_residual", fused_train_r2, fused_val_r2),
    ]
    recovery_variant, recovery_train_r2, recovery_val_r2 = max(recovery_candidates, key=lambda item: item[2])

    return {
        "method": "mlp_residual",
        "recovery_variant": recovery_variant,
        "budget": int(selected.numel()),
        "budget_fraction": float(selected.numel() / keys.shape[0]),
        "direct_train_r2": direct_train_r2,
        "direct_val_r2": direct_val_r2,
        "fitv_train_r2": fitv_train_r2,
        "fitv_val_r2": fitv_val_r2,
        "learned_value_train_r2": learned_train_r2,
        "learned_value_val_r2": learned_val_r2,
        "mlp_train_r2": mlp_train_r2,
        "mlp_val_r2": mlp_val_r2,
        "fused_train_r2": fused_train_r2,
        "fused_val_r2": fused_val_r2,
        "recovery_train_r2": recovery_train_r2,
        "recovery_val_r2": recovery_val_r2,
        "recovery_gain_vs_direct": recovery_val_r2 - direct_val_r2,
        "recovery_gain_vs_fitv": recovery_val_r2 - fitv_val_r2,
        "train_loss_initial": initial_loss,
        "train_loss_final": final_loss,
        "fused_train_loss_initial": fused_initial_loss,
        "fused_train_loss_final": fused_final_loss,
    }


def percentile(vals, p):
    vals = sorted(float(x) for x in vals)
    if not vals:
        return float("nan")
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * p))))
    return vals[idx]


def summarize_recovery(rows, sampled_tokens, accept_r2=0.50):
    recovery_vals = [float(r["recovery_val_r2"]) for r in rows]
    direct_vals = [float(r["direct_val_r2"]) for r in rows]
    fitv_vals = [float(r["fitv_val_r2"]) for r in rows]
    budget_fracs = [float(r["budget_fraction"]) for r in rows]
    gains_direct = [float(r["recovery_gain_vs_direct"]) for r in rows]
    gains_fitv = [float(r["recovery_gain_vs_fitv"]) for r in rows]
    mean_recovery = sum(recovery_vals) / len(recovery_vals) if recovery_vals else float("nan")
    mean_fitv = sum(fitv_vals) / len(fitv_vals) if fitv_vals else float("nan")
    mean_gain = sum(gains_direct) / len(gains_direct) if gains_direct else float("nan")
    coverage = sum(v >= accept_r2 for v in recovery_vals) / len(recovery_vals) if recovery_vals else 0.0

    if recovery_vals and mean_recovery >= accept_r2 and mean_recovery >= mean_fitv and mean_gain >= -1e-6:
        decision = "RECOVERY_PROMISING"
    elif recovery_vals and mean_recovery > mean_fitv:
        decision = "RECOVERY_NEEDS_TASK_SMOKE"
    else:
        decision = "RECOVERY_WEAK"

    return {
        "decision": decision,
        "heads": len(rows),
        "sampled_tokens": int(sampled_tokens),
        "mean_budget_fraction": sum(budget_fracs) / len(budget_fracs) if budget_fracs else float("nan"),
        "mean_direct_val_r2": sum(direct_vals) / len(direct_vals) if direct_vals else float("nan"),
        "mean_fitv_val_r2": mean_fitv,
        "mean_recovery_val_r2": mean_recovery,
        "p50_recovery_val_r2": percentile(recovery_vals, 0.50),
        "p90_recovery_val_r2": percentile(recovery_vals, 0.90),
        "recovery_val_ge_accept": coverage,
        "mean_recovery_gain_vs_direct": mean_gain,
        "mean_recovery_gain_vs_fitv": sum(gains_fitv) / len(gains_fitv) if gains_fitv else float("nan"),
    }


def write_rows(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: (f"{v:.6g}" if isinstance(v, float) and math.isfinite(v) else v)
                for k, v in row.items()
            })


def write_outputs(output_dir, rows, summary, metadata, settings, train_seconds):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / "recovery_head_rows.csv", rows)
    payload = {
        **summary,
        "metadata": metadata,
        "settings": settings,
        "train_seconds": train_seconds,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report = [
        "# Trainable Light Doc Cache Recovery Probe",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        f"- Heads: {summary['heads']}",
        f"- Mean budget fraction: {summary['mean_budget_fraction']:.2%}",
        f"- Mean direct val R2: {summary['mean_direct_val_r2']:.4f}",
        f"- Mean ridge-value val R2: {summary['mean_fitv_val_r2']:.4f}",
        f"- Mean recovery val R2: {summary['mean_recovery_val_r2']:.4f}",
        f"- Recovery coverage above accept R2: {summary['recovery_val_ge_accept']:.2%}",
        f"- Mean recovery gain vs direct: {summary['mean_recovery_gain_vs_direct']:.4f}",
        "",
        "| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        report.append(
            f"| {row['layer']} | {row['kv_head']} | {row['budget']} | {row['budget_fraction']:.2%} | "
            f"{row['direct_val_r2']:.4f} | {row['fitv_val_r2']:.4f} | {row['learned_value_val_r2']:.4f} | {row['mlp_val_r2']:.4f} | "
            f"{row['fused_val_r2']:.4f} | "
            f"{row['recovery_val_r2']:.4f} | {row['recovery_variant']} |"
        )
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run_doc(args, text_file, doc_index, output_dir):
    doc_args = argparse.Namespace(**{**vars(args), "text_file": text_file})
    queries, keys, values, positions_cpu, meta = collect_qkv(doc_args)
    device = pick_device(args.device)
    positions = positions_cpu.to(device)
    budgets = parse_ints(args.budgets)
    group = queries.shape[1] // keys.shape[1]
    rows = []
    train_idx, val_idx = split_indices(keys.shape[2], args.train_frac, device)
    max_heads = args.max_heads if args.max_heads > 0 else keys.shape[0] * keys.shape[1]
    seen_heads = 0

    start_layer = min(max(0, int(args.start_layer)), int(keys.shape[0]))
    end_layer = int(args.end_layer) if args.end_layer > 0 else int(keys.shape[0])
    end_layer = min(max(start_layer, end_layer), int(keys.shape[0]))
    for layer in range(start_layer, end_layer):
        for kv_head in range(keys.shape[1]):
            if seen_heads >= max_heads:
                break
            q_group = queries[layer, kv_head * group:(kv_head + 1) * group].to(device)
            k = keys[layer, kv_head].to(device)
            v = values[layer, kv_head].to(device)
            teacher = attention_output(q_group, k, v, positions, positions)
            for budget in budgets:
                selected = select_tokens(args.selector, q_group, k, positions, train_idx, budget)
                if selected.numel() < 2:
                    continue
                row = train_head_recovery(
                    q_group=q_group,
                    keys=k,
                    values=v,
                    positions=positions,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    selected=selected,
                    teacher_all=teacher,
                    hidden_dim=args.hidden_dim,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    ridge=args.ridge,
                    device=device,
                )
                row.update(
                    doc_index=doc_index,
                    text_file=str(text_file),
                    layer=int(layer),
                    kv_head=int(kv_head),
                    query_group=int(group),
                    selector=args.selector,
                )
                rows.append(row)
            seen_heads += 1
        if seen_heads >= max_heads:
            break

    summary = summarize_recovery(rows, meta["sampled_tokens"], accept_r2=args.accept_r2)
    meta = {**meta, "text_file": str(text_file), "doc_index": doc_index}
    write_outputs(output_dir, rows, summary, meta, vars(args), train_seconds=0.0)
    return rows, meta


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    all_rows = []
    metadata = {"documents": []}

    for doc_index, text_file in enumerate(args.text_file):
        doc_out = out / f"doc_{doc_index:02d}"
        rows, meta = run_doc(args, text_file, doc_index, doc_out)
        all_rows.extend(rows)
        metadata["documents"].append(meta)

    train_seconds = time.perf_counter() - t0
    sampled = metadata["documents"][0]["sampled_tokens"] if metadata["documents"] else 0
    summary = summarize_recovery(all_rows, sampled, accept_r2=args.accept_r2)
    write_outputs(out, all_rows, summary, metadata, vars(args), train_seconds=train_seconds)
    print("decision", summary["decision"])
    print("mean_recovery_val_r2", f"{summary['mean_recovery_val_r2']:.4f}")
    print("output_dir", out)


if __name__ == "__main__":
    main()
