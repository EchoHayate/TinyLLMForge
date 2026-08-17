"""Probe whether dropped KV can be predicted from retained layer/head KV.

This is the first executable artifact for Light Doc Cache T1.3.  It measures
low-cost recoverability on real HuggingFace `past_key_values` without modifying
TinyLLMForge's serving path.

Examples:

  python3 tools/probe_kv_recovery.py --synthetic \
      --output-dir light-doc-cache-plan/t1_3_probe/runs/synthetic

  python3 tools/probe_kv_recovery.py \
      --model ~/Qwen3-0.6B \
      --text-file docs/kv-sparse-attention.md \
      --max-tokens 2048 \
      --output-dir light-doc-cache-plan/t1_3_probe/runs/qwen3_0_6b
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - lets --help work on bare Python.
    torch = None  # type: ignore[assignment]


DEFAULT_TEXT = """
Light Doc Cache is a research prototype for compressing key-value caches in
long-context language model inference. The central question is whether a subset
of retained KV tensors contains enough information to reconstruct dropped KV
tensors through a low-cost online operator. This probe intentionally starts with
simple affine predictors and correlation heatmaps before adding trainable
recovery modules.
"""


def inference_mode():
    if torch is None:
        def decorator(func):
            return func
        return decorator
    return torch.inference_mode()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default=None, help="Local HF model path or model id.")
    parser.add_argument("--text-file", type=str, default=None, help="UTF-8 text file used as the probe document.")
    parser.add_argument("--prompt", type=str, default=None, help="Inline probe text. Ignored when --text-file is set.")
    parser.add_argument("--output-dir", type=str, default="light-doc-cache-plan/t1_3_probe/runs/latest")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum prompt tokens for the real model probe.")
    parser.add_argument("--max-sample-tokens", type=int, default=1024, help="Maximum token positions used for metrics.")
    parser.add_argument("--token-sample-stride", type=int, default=1, help="Use every Nth token before max sampling.")
    parser.add_argument("--skip-prefix-tokens", type=int, default=0, help="Drop the first N positions from metric samples.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--allow-download", action="store_true", default=False, help="Allow HF to download model files.")
    parser.add_argument("--attn-implementation", type=str, default=None, help="Optional HF attention implementation.")
    parser.add_argument("--train-frac", type=float, default=0.6, help="Token split fraction for affine predictor fitting.")
    parser.add_argument("--ridge", type=float, default=1e-6, help="Ridge term for diagonal affine fits.")
    parser.add_argument("--go-r2-threshold", type=float, default=0.50, help="Mean best R2 threshold for a tentative GO.")
    parser.add_argument("--borderline-r2-threshold", type=float, default=0.35)
    parser.add_argument("--no-png", action="store_true", default=False, help="Skip matplotlib heatmaps.")

    parser.add_argument("--synthetic", action="store_true", default=False, help="Run on synthetic correlated KV tensors.")
    parser.add_argument("--synthetic-layers", type=int, default=6)
    parser.add_argument("--synthetic-heads", type=int, default=4)
    parser.add_argument("--synthetic-tokens", type=int, default=256)
    parser.add_argument("--synthetic-head-dim", type=int, default=32)
    parser.add_argument("--synthetic-noise", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def require_torch() -> None:
    if torch is None:
        raise SystemExit(
            "PyTorch is required to run the KV recovery probe. "
            "Install/use a Python environment with torch before running this script."
        )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested --device cuda but CUDA is not available")
    if name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("requested --device mps but MPS is not available")
    return torch.device(name)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def build_probe_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8")
    if args.prompt:
        return args.prompt
    return DEFAULT_TEXT


def token_sample_indices(
    seq_len: int,
    max_sample_tokens: int,
    stride: int,
    skip_prefix_tokens: int,
) -> torch.Tensor:
    start = min(max(0, int(skip_prefix_tokens)), seq_len)
    stride = max(1, int(stride))
    idx = torch.arange(start, seq_len, stride, dtype=torch.long)
    if max_sample_tokens > 0 and idx.numel() > max_sample_tokens:
        select = torch.linspace(0, idx.numel() - 1, max_sample_tokens).round().to(torch.long)
        idx = idx[select]
    if idx.numel() < 4:
        raise RuntimeError(
            f"need at least 4 sampled token positions for the probe, got {idx.numel()} "
            f"(seq_len={seq_len}, skip_prefix_tokens={skip_prefix_tokens})"
        )
    return idx


def normalize_cache_layout(tensor: torch.Tensor, expected_heads: int | None) -> torch.Tensor:
    """Return cache tensor as [batch, heads, tokens, head_dim]."""
    if tensor.ndim != 4:
        raise RuntimeError(f"expected a 4D KV cache tensor, got shape={tuple(tensor.shape)}")
    if expected_heads is not None:
        if tensor.shape[1] == expected_heads:
            return tensor
        if tensor.shape[2] == expected_heads:
            return tensor.permute(0, 2, 1, 3).contiguous()
    # Common layouts are [B, H, T, D] and [B, T, H, D].  The token axis is
    # normally much larger than head count; this fallback also works for short
    # contexts as long as head count is not larger than 128.
    if tensor.shape[1] <= 128 and tensor.shape[2] >= tensor.shape[1]:
        return tensor
    if tensor.shape[2] <= 128:
        return tensor.permute(0, 2, 1, 3).contiguous()
    raise RuntimeError(f"cannot infer KV cache layout for shape={tuple(tensor.shape)}")


def legacy_past_key_values(past_key_values: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))
    pairs = []
    for item in past_key_values:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise RuntimeError(f"unexpected past_key_values entry: {type(item)}")
        pairs.append((item[0], item[1]))
    return pairs


@inference_mode()
def collect_hf_kv(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not args.model:
        raise RuntimeError("--model is required unless --synthetic is used")
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: WPS433

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": not args.allow_download,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=not args.allow_download,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval().to(device)

    text = build_probe_text(args)
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_tokens,
        add_special_tokens=True,
    )
    if enc["input_ids"].shape[1] < min(args.max_tokens, 64):
        # Repeat the default/small text enough to exercise position variation.
        repeated = text
        while True:
            repeated = repeated + "\n\n" + text
            enc = tokenizer(
                repeated,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_tokens,
                add_special_tokens=True,
            )
            if enc["input_ids"].shape[1] >= min(args.max_tokens, 64):
                break
    enc = {key: value.to(device) for key, value in enc.items()}

    started = time.perf_counter()
    outputs = model(**enc, use_cache=True, return_dict=True)
    elapsed_s = time.perf_counter() - started
    pairs = legacy_past_key_values(outputs.past_key_values)

    expected_heads = getattr(model.config, "num_key_value_heads", None)
    if expected_heads is None:
        expected_heads = getattr(model.config, "num_attention_heads", None)

    seq_len = int(enc["input_ids"].shape[1])
    idx = token_sample_indices(
        seq_len,
        args.max_sample_tokens,
        args.token_sample_stride,
        args.skip_prefix_tokens,
    )
    key_layers = []
    value_layers = []
    for key, value in pairs:
        key = normalize_cache_layout(key, expected_heads)
        value = normalize_cache_layout(value, expected_heads)
        key_layers.append(key[0, :, idx.to(key.device), :].detach().cpu().float().contiguous())
        value_layers.append(value[0, :, idx.to(value.device), :].detach().cpu().float().contiguous())

    keys = torch.stack(key_layers, dim=0)
    values = torch.stack(value_layers, dim=0)
    metadata = {
        "mode": "hf",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "seq_len": seq_len,
        "sampled_tokens": int(idx.numel()),
        "sample_start": int(idx[0].item()),
        "sample_end": int(idx[-1].item()),
        "num_layers": int(keys.shape[0]),
        "num_kv_heads": int(keys.shape[1]),
        "head_dim": int(keys.shape[3]),
        "forward_s": elapsed_s,
    }
    return keys, values, metadata


def collect_synthetic_kv(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    shape = (
        args.synthetic_layers,
        args.synthetic_heads,
        args.synthetic_tokens,
        args.synthetic_head_dim,
    )
    keys = torch.randn(shape, generator=generator)
    values = torch.randn(shape, generator=generator)
    for layer in range(1, shape[0]):
        layer_noise_k = args.synthetic_noise * torch.randn(shape[1:], generator=generator)
        layer_noise_v = args.synthetic_noise * torch.randn(shape[1:], generator=generator)
        keys[layer] = 0.82 * keys[layer - 1] + 0.18 * keys[layer] + layer_noise_k
        values[layer] = 0.74 * values[layer - 1] + 0.26 * values[layer] + layer_noise_v
    if shape[1] > 1:
        keys[:, 1:] = 0.35 * keys[:, :1] + 0.65 * keys[:, 1:]
        values[:, 1:] = 0.25 * values[:, :1] + 0.75 * values[:, 1:]
    metadata = {
        "mode": "synthetic",
        "seed": args.seed,
        "seq_len": shape[2],
        "sampled_tokens": shape[2],
        "num_layers": shape[0],
        "num_kv_heads": shape[1],
        "head_dim": shape[3],
        "synthetic_noise": args.synthetic_noise,
    }
    return keys.float(), values.float(), metadata


def mean_token_cosine(source: torch.Tensor, target: torch.Tensor) -> float:
    source = source.float()
    target = target.float()
    numerator = (source * target).sum(dim=-1)
    denominator = source.norm(dim=-1) * target.norm(dim=-1)
    score = numerator / denominator.clamp_min(1e-8)
    return float(score.mean().item())


def diag_affine_r2(
    source: torch.Tensor,
    target: torch.Tensor,
    train_frac: float,
    ridge: float,
) -> float:
    """Fit target ~= source * scale + bias per channel and evaluate token holdout R2."""
    if source.shape != target.shape:
        return float("nan")
    n_tokens = source.shape[0]
    if n_tokens < 4:
        return float("nan")
    split = int(round(n_tokens * train_frac))
    split = min(max(2, split), n_tokens - 2)
    x_train = source[:split].float()
    y_train = target[:split].float()
    x_val = source[split:].float()
    y_val = target[split:].float()

    x_mean = x_train.mean(dim=0, keepdim=True)
    y_mean = y_train.mean(dim=0, keepdim=True)
    x_centered = x_train - x_mean
    y_centered = y_train - y_mean
    covariance = (x_centered * y_centered).mean(dim=0)
    variance = (x_centered * x_centered).mean(dim=0)
    scale = covariance / (variance + ridge)
    bias = y_mean.squeeze(0) - scale * x_mean.squeeze(0)
    pred = x_val * scale + bias

    sse = (pred - y_val).pow(2).sum()
    baseline = y_val.mean(dim=0, keepdim=True)
    sst = (y_val - baseline).pow(2).sum()
    return float((1.0 - sse / sst.clamp_min(1e-8)).item())


def layer_metric_matrix(
    tensor: torch.Tensor,
    metric: str,
    train_frac: float,
    ridge: float,
) -> list[list[float]]:
    num_layers, num_heads = int(tensor.shape[0]), int(tensor.shape[1])
    matrix: list[list[float]] = []
    for target_layer in range(num_layers):
        row = []
        for source_layer in range(num_layers):
            scores = []
            for head in range(num_heads):
                source = tensor[source_layer, head]
                target = tensor[target_layer, head]
                if metric == "cosine":
                    scores.append(mean_token_cosine(source, target))
                elif metric == "diag_r2":
                    scores.append(diag_affine_r2(source, target, train_frac, ridge))
                else:
                    raise ValueError(f"unknown metric: {metric}")
            row.append(nanmean(scores))
        matrix.append(row)
    return matrix


def best_cross_head_rows(
    keys: torch.Tensor,
    values: torch.Tensor,
    train_frac: float,
    ridge: float,
) -> list[dict[str, Any]]:
    rows = []
    num_layers, num_heads = int(keys.shape[0]), int(keys.shape[1])
    if num_heads <= 1:
        return rows
    for layer in range(num_layers):
        for target_head in range(num_heads):
            best: dict[str, Any] | None = None
            best_joint = -float("inf")
            for source_head in range(num_heads):
                if source_head == target_head:
                    continue
                k_cos = mean_token_cosine(keys[layer, source_head], keys[layer, target_head])
                v_cos = mean_token_cosine(values[layer, source_head], values[layer, target_head])
                k_r2 = diag_affine_r2(keys[layer, source_head], keys[layer, target_head], train_frac, ridge)
                v_r2 = diag_affine_r2(values[layer, source_head], values[layer, target_head], train_frac, ridge)
                joint = min(k_r2, v_r2)
                if joint > best_joint:
                    best_joint = joint
                    best = {
                        "layer": layer,
                        "target_head": target_head,
                        "source_head": source_head,
                        "k_cosine": k_cos,
                        "v_cosine": v_cos,
                        "k_diag_r2": k_r2,
                        "v_diag_r2": v_r2,
                        "joint_diag_r2": joint,
                    }
            if best is not None:
                rows.append(best)
    return rows


def best_cross_layer_rows(
    keys: torch.Tensor,
    values: torch.Tensor,
    train_frac: float,
    ridge: float,
) -> list[dict[str, Any]]:
    rows = []
    num_layers, num_heads = int(keys.shape[0]), int(keys.shape[1])
    for target_layer in range(num_layers):
        for head in range(num_heads):
            best: dict[str, Any] | None = None
            best_joint = -float("inf")
            for source_layer in range(num_layers):
                if source_layer == target_layer:
                    continue
                k_cos = mean_token_cosine(keys[source_layer, head], keys[target_layer, head])
                v_cos = mean_token_cosine(values[source_layer, head], values[target_layer, head])
                k_r2 = diag_affine_r2(keys[source_layer, head], keys[target_layer, head], train_frac, ridge)
                v_r2 = diag_affine_r2(values[source_layer, head], values[target_layer, head], train_frac, ridge)
                joint = min(k_r2, v_r2)
                if joint > best_joint:
                    best_joint = joint
                    best = {
                        "target_layer": target_layer,
                        "source_layer": source_layer,
                        "head": head,
                        "k_cosine": k_cos,
                        "v_cosine": v_cos,
                        "k_diag_r2": k_r2,
                        "v_diag_r2": v_r2,
                        "joint_diag_r2": joint,
                    }
            if best is not None:
                rows.append(best)
    return rows


def nanmean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def max_except_diagonal_mean(matrix: list[list[float]]) -> float:
    scores = []
    for row_idx, row in enumerate(matrix):
        candidates = [value for col_idx, value in enumerate(row) if col_idx != row_idx]
        if candidates:
            scores.append(max(candidates))
    return nanmean(scores)


def adjacent_prev_mean(matrix: list[list[float]]) -> float:
    return nanmean([matrix[layer][layer - 1] for layer in range(1, len(matrix))])


def write_matrix_csv(path: Path, matrix: list[list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_layer"] + [f"source_layer_{idx}" for idx in range(len(matrix))])
        for idx, row in enumerate(matrix):
            writer.writerow([idx] + [format_float(value) for value in row])


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_float(value) if isinstance(value, float) else value for key, value in row.items()})


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def maybe_write_heatmap(path: Path, matrix: list[list[float]], title: str, vmin: float, vmax: float) -> bool:
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    image = ax.imshow(matrix, interpolation="nearest", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("source layer")
    ax.set_ylabel("target layer")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def summarize_decision(summary: dict[str, Any], go_threshold: float, borderline_threshold: float) -> tuple[str, str]:
    cross_layer_joint = min(
        summary["mean_best_cross_layer_k_diag_r2"],
        summary["mean_best_cross_layer_v_diag_r2"],
    )
    cross_head_joint = min(
        summary["mean_best_cross_head_k_diag_r2"],
        summary["mean_best_cross_head_v_diag_r2"],
    )
    best_joint = max(cross_layer_joint, cross_head_joint)
    if best_joint >= go_threshold:
        return "GO", (
            f"best mean joint K/V diagonal-affine R2 is {best_joint:.3f}, "
            f"above threshold {go_threshold:.3f}"
        )
    if best_joint >= borderline_threshold:
        return "BORDERLINE", (
            f"best mean joint K/V diagonal-affine R2 is {best_joint:.3f}, "
            f"between {borderline_threshold:.3f} and {go_threshold:.3f}; run larger probes"
        )
    return "NO-GO", (
        f"best mean joint K/V diagonal-affine R2 is {best_joint:.3f}, "
        f"below borderline threshold {borderline_threshold:.3f}"
    )


def write_report(path: Path, metadata: dict[str, Any], summary: dict[str, Any], files: list[str]) -> None:
    decision = summary["decision"]
    reason = summary["decision_reason"]
    lines = [
        "# T1.3 KV Recovery Probe Report",
        "",
        f"- Mode: `{metadata.get('mode')}`",
        f"- Model: `{metadata.get('model', 'synthetic')}`",
        f"- Layers: {metadata['num_layers']}",
        f"- KV heads: {metadata['num_kv_heads']}",
        f"- Head dim: {metadata['head_dim']}",
        f"- Sequence tokens: {metadata['seq_len']}",
        f"- Sampled tokens: {metadata['sampled_tokens']}",
        "",
        "## Decision",
        "",
        f"**{decision}**: {reason}.",
        "",
        "This is a probe result, not a final compression result. It only tests cheap",
        "per-channel affine predictability from retained layer/head KV.",
        "",
        "## Key Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mean previous-layer K diag R2 | {summary['mean_prev_layer_k_diag_r2']:.4f} |",
        f"| Mean previous-layer V diag R2 | {summary['mean_prev_layer_v_diag_r2']:.4f} |",
        f"| Mean best cross-layer K diag R2 | {summary['mean_best_cross_layer_k_diag_r2']:.4f} |",
        f"| Mean best cross-layer V diag R2 | {summary['mean_best_cross_layer_v_diag_r2']:.4f} |",
        f"| Mean best cross-head K diag R2 | {summary['mean_best_cross_head_k_diag_r2']:.4f} |",
        f"| Mean best cross-head V diag R2 | {summary['mean_best_cross_head_v_diag_r2']:.4f} |",
        "",
        "## Outputs",
        "",
    ]
    lines.extend([f"- `{name}`" for name in files])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_probe(keys: torch.Tensor, values: torch.Tensor, args: argparse.Namespace, metadata: dict[str, Any]) -> None:
    if keys.shape != values.shape:
        raise RuntimeError(f"key/value shape mismatch: {tuple(keys.shape)} vs {tuple(values.shape)}")
    if keys.ndim != 4:
        raise RuntimeError(f"expected [layers, heads, tokens, head_dim], got {tuple(keys.shape)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrices = {
        "layer_k_cosine": layer_metric_matrix(keys, "cosine", args.train_frac, args.ridge),
        "layer_v_cosine": layer_metric_matrix(values, "cosine", args.train_frac, args.ridge),
        "layer_k_diag_r2": layer_metric_matrix(keys, "diag_r2", args.train_frac, args.ridge),
        "layer_v_diag_r2": layer_metric_matrix(values, "diag_r2", args.train_frac, args.ridge),
    }
    cross_head = best_cross_head_rows(keys, values, args.train_frac, args.ridge)
    cross_layer = best_cross_layer_rows(keys, values, args.train_frac, args.ridge)

    files = []
    for name, matrix in matrices.items():
        csv_name = f"{name}.csv"
        write_matrix_csv(output_dir / csv_name, matrix)
        files.append(csv_name)

    write_rows_csv(output_dir / "best_cross_head.csv", cross_head)
    write_rows_csv(output_dir / "best_cross_layer.csv", cross_layer)
    files.extend(["best_cross_head.csv", "best_cross_layer.csv"])

    if not args.no_png:
        for name, matrix in matrices.items():
            png_name = f"{name}.png"
            if "cosine" in name:
                ok = maybe_write_heatmap(output_dir / png_name, matrix, name, vmin=-1.0, vmax=1.0)
            else:
                ok = maybe_write_heatmap(output_dir / png_name, matrix, name, vmin=0.0, vmax=1.0)
            if ok:
                files.append(png_name)

    summary = {
        "mean_prev_layer_k_diag_r2": adjacent_prev_mean(matrices["layer_k_diag_r2"]),
        "mean_prev_layer_v_diag_r2": adjacent_prev_mean(matrices["layer_v_diag_r2"]),
        "mean_best_cross_layer_k_diag_r2": max_except_diagonal_mean(matrices["layer_k_diag_r2"]),
        "mean_best_cross_layer_v_diag_r2": max_except_diagonal_mean(matrices["layer_v_diag_r2"]),
        "mean_best_cross_head_k_diag_r2": nanmean([row["k_diag_r2"] for row in cross_head]),
        "mean_best_cross_head_v_diag_r2": nanmean([row["v_diag_r2"] for row in cross_head]),
    }
    decision, reason = summarize_decision(summary, args.go_r2_threshold, args.borderline_r2_threshold)
    summary["decision"] = decision
    summary["decision_reason"] = reason
    summary["thresholds"] = {
        "go_r2": args.go_r2_threshold,
        "borderline_r2": args.borderline_r2_threshold,
    }
    summary["metadata"] = metadata
    summary["outputs"] = files

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    files.append("summary.json")
    write_report(output_dir / "report.md", metadata, summary, files)
    print(f"decision={decision} reason={reason}")
    print(f"wrote {output_dir}")


def main() -> None:
    args = parse_args()
    require_torch()
    if args.synthetic:
        keys, values, metadata = collect_synthetic_kv(args)
    else:
        keys, values, metadata = collect_hf_kv(args)
    run_probe(keys, values, args, metadata)


if __name__ == "__main__":
    main()
