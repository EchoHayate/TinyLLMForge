"""Dump real post-RoPE Q/K from a real model on a real needle prompt.

The selector fidelity gate must not be fed synthetic attention: the whole question is
whether a query-aware selector survives the actual geometry of a trained model's keys.
So this captures the exact tensors the attention kernel would see, by intercepting
`apply_rotary_pos_emb` inside the HF Qwen3 implementation - after q_norm/k_norm and
after RoPE, which is the same point at which tinyvllm's Quest selector reads them.

It also generates a short greedy continuation and records whether the model got the
answer right. A fidelity number measured on a prompt the model cannot answer anyway
would be meaningless, so validity of the prompt is recorded alongside the tensors
rather than assumed.

Usage (on a GPU box):
    python tools/dump_needle_qk.py \
        --model /path/to/Qwen3-8B --seq-len 8192 \
        --variants repetitive natural distractor \
        --out-dir /tmp/qkdump
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

from tools.needle_haystack_variants import build_variant


def _select_layers(n_layers: int, spec: str) -> list[int]:
    if spec == "all":
        return list(range(n_layers))
    if spec.startswith("auto"):
        count = int(spec[4:] or 6)
        count = min(count, n_layers)
        return sorted(set(int(round(i * (n_layers - 1) / max(count - 1, 1))) for i in range(count)))
    return [int(x) for x in spec.split(",")]


class RopeCapture:
    """Records the (q, k) returned by apply_rotary_pos_emb, one call per layer."""

    def __init__(self, module):
        self.module = module
        self.original = module.apply_rotary_pos_emb
        self.records: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.enabled = False

    def __enter__(self):
        capture = self

        def patched(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            q_out, k_out = capture.original(q, k, cos, sin, position_ids, unsqueeze_dim)
            if capture.enabled:
                capture.records.append((q_out.detach(), k_out.detach()))
            return q_out, k_out

        self.module.apply_rotary_pos_emb = patched
        return self

    def __exit__(self, *exc):
        self.module.apply_rotary_pos_emb = self.original
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--variants", nargs="+", default=["repetitive", "natural", "distractor"])
    p.add_argument("--layers", default="auto6", help="'all', 'auto<N>', or a comma list")
    p.add_argument("--depth", type=float, default=0.5)
    p.add_argument("--num-decoys", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    # 12 was too few: the distractor prompt's answer got cut mid-number and the run
    # reported model_answered_correctly=False, which is a broken instrument, not a
    # finding. 32 leaves room for the model to finish the sentence.
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3 import modeling_qwen3

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device).eval()

    n_layers = model.config.num_hidden_layers
    layers = _select_layers(n_layers, args.layers)
    print(f"model layers={n_layers} capturing={layers}")

    written = []
    for variant in args.variants:
        spec = build_variant(
            tokenizer, variant, args.seq_len,
            depth=args.depth, num_decoys=args.num_decoys, seed=args.seed,
        )
        ids = torch.tensor([spec["ids"]], dtype=torch.long, device=args.device)
        print(f"[{variant}] seq_len={spec['seq_len']} answer={spec['answer']} "
              f"needle_pos={spec['answer_needle_positions'][:3]}... decoys={spec['num_decoys']}")

        with RopeCapture(modeling_qwen3) as cap, torch.inference_mode():
            cap.enabled = True
            model(input_ids=ids, use_cache=False)
            cap.enabled = False
            records = cap.records
            if len(records) != n_layers:
                raise RuntimeError(f"expected {n_layers} rope calls, captured {len(records)}")

            q_sel, k_sel = [], []
            for li in layers:
                q_out, k_out = records[li]
                # [B, heads, seq, dim] -> last position for q, all positions for k
                q_sel.append(q_out[0, :, -1, :].to(torch.float16).cpu().numpy())
                k_sel.append(k_out[0].permute(1, 0, 2).to(torch.float16).cpu().numpy())
            del records, cap.records[:]

            gen = model.generate(
                input_ids=ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        correct = spec["answer"] in completion

        dump_path = os.path.join(args.out_dir, f"qk-{variant}.npz")
        np.savez_compressed(
            dump_path,
            q=np.stack(q_sel),                 # [n_sel, n_q_heads, dim]
            k=np.stack(k_sel),                 # [n_sel, seq_len, n_kv_heads, dim]
            layers=np.array(layers, dtype=np.int32),
        )
        meta = {
            "variant": variant,
            "model": args.model,
            "seq_len": spec["seq_len"],
            "layers": layers,
            "answer": spec["answer"],
            "answer_needle_positions": spec["answer_needle_positions"],
            "decoy_positions": spec["decoy_positions"],
            "num_decoys": spec["num_decoys"],
            "depth": spec["depth"],
            "seed": spec["seed"],
            "n_q_heads": int(q_sel[0].shape[0]),
            "n_kv_heads": int(k_sel[0].shape[1]),
            "head_dim": int(k_sel[0].shape[2]),
            "greedy_completion": completion,
            "model_answered_correctly": bool(correct),
        }
        meta_path = os.path.join(args.out_dir, f"qk-{variant}.meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[{variant}] correct={correct} completion={completion!r} -> {dump_path}")
        written.append(dump_path)
        del q_sel, k_sel
        torch.cuda.empty_cache()

    print("wrote:\n  " + "\n  ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
