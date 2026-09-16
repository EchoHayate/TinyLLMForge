"""End-to-end sparse decode: does the selector still answer the question?

Why this file exists
--------------------
`tools/kv_selector_fidelity.py` measured the selector in isolation and produced a
sharp result: at 32-token granularity the shipped shared-head Quest ranking keeps the
answer tokens at k/L ~ 5.1%. But coverage of the answer tokens is a *necessary*
condition, not a sufficient one. The model can still derail because

  - coverage was measured at the last prompt position only, while decode moves the
    query every step and the selected set is re-chosen every step;
  - it was measured on 6 sampled layers, while a real run sparsifies all 36;
  - keeping the answer token says nothing about keeping the tokens that carry the
    instruction, the format, or the syntactic frame around the answer.

So this tool closes the loop: prefill dense, decode with query-aware selection in
*every* layer at *every* step, and compare the greedy continuation against the dense
continuation of the same prompt. Bad arms run at the identical budget, because a gate
whose arms all pass is not a gate.

The selection math is the engine's, not the paper's: `quest_shared_heads` reproduces
`quest_score_kernel` in tinyvllm/layers/attention.py, which sums the per-channel
min/max bound over every kv head into one shared ranking, and force-keeps the first
and last unit. The last unit is force-kept by all arms including the bad ones, because
it holds the current token's own key and dropping it would be a bug rather than a
selection policy.

Granularity note: the engine cannot currently run gran=32 as-is
(`config.py` asserts `kvcache_block_size % 256 == 0`, a flash-attn paged-KV
constraint), so this harness selects at 32-token granularity by gathering the chosen
units into a contiguous buffer and attending over that. That is exactly the shape the
CPU-offload path has to take anyway - the CPU gathers selected units into a staging
buffer - so measuring it here is measuring the thing we intend to build, not a
convenient stand-in.

Usage (on a GPU box):
    python tools/e2e_sparse_attention.py \
        --model /path/to/Qwen3-8B --seq-len 8192 --variant distractor \
        --granularities 32 256 --k-fracs 0.02 0.051 0.11 0.25 \
        --out-json /tmp/e2e-sparse.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Iterable, Sequence

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F

ARMS = ("quest_shared_heads", "quest_per_head", "recency", "sink_recency", "random")
BAD_ARMS = ("recency", "sink_recency", "random")
PRIMARY_ARM = "quest_shared_heads"


# ---------------------------------------------------------------------------
# selection math (pure torch, no model needed - unit tested on CPU)
# ---------------------------------------------------------------------------

def unit_minmax(key: torch.Tensor, granularity: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-unit per-channel min/max of K.

    key: [B, KVH, S, D] -> (kmin, kmax) each [B, U, KVH, D], U = ceil(S / granularity).

    The engine maintains these incrementally with atomic min/max as tokens are written;
    recomputing them here is the same set reduced the same way, so the selector sees the
    same summary it would see in the engine.
    """
    if granularity <= 0:
        raise ValueError("granularity must be positive")
    b, kvh, s, d = key.shape
    n_units = (s + granularity - 1) // granularity
    pad = n_units * granularity - s
    if pad:
        # Pad with +inf for the min reduction and -inf for the max reduction so the
        # padding cannot widen the bound of the (short) last unit. Widening it would
        # make the last unit look more attractive than it is, which is exactly the kind
        # of silent optimism that makes a selector look better than it is.
        kmin_src = F.pad(key.to(torch.float32), (0, 0, 0, pad), value=float("inf"))
        kmax_src = F.pad(key.to(torch.float32), (0, 0, 0, pad), value=float("-inf"))
    else:
        kmin_src = kmax_src = key
    kmin = kmin_src.view(b, kvh, n_units, granularity, d).amin(dim=3)
    kmax = kmax_src.view(b, kvh, n_units, granularity, d).amax(dim=3)
    return kmin.transpose(1, 2).contiguous(), kmax.transpose(1, 2).contiguous()


def query_repr(query: torch.Tensor, n_kv_heads: int) -> torch.Tensor:
    """Group-wise amax of the decode query: [B, H, 1, D] -> [B, KVH, D].

    Engine semantics (`quest_select_blocks`): within a GQA group the estimate is made
    aggressive by taking the max over q heads, so a unit that matters to any head in the
    group survives.
    """
    b, h, q_len, d = query.shape
    if q_len != 1:
        raise ValueError("query_repr is a decode-only path (q_len must be 1)")
    if h % n_kv_heads:
        raise ValueError(f"q heads {h} not divisible by kv heads {n_kv_heads}")
    group = h // n_kv_heads
    return query.view(b, n_kv_heads, group, d).amax(dim=2)


def quest_scores_shared(q_repr: torch.Tensor, kmin: torch.Tensor, kmax: torch.Tensor) -> torch.Tensor:
    """One ranking shared by all kv heads: [B, U]. Mirrors `quest_score_kernel`."""
    q = q_repr.unsqueeze(1).to(torch.float32)                # [B, 1, KVH, D]
    bound = torch.maximum(q * kmin.to(torch.float32), q * kmax.to(torch.float32))
    return bound.sum(dim=(2, 3))


def quest_scores_per_head(q_repr: torch.Tensor, kmin: torch.Tensor, kmax: torch.Tensor) -> torch.Tensor:
    """One ranking per kv head: [B, U, KVH]. What the Quest paper describes."""
    q = q_repr.unsqueeze(1).to(torch.float32)
    bound = torch.maximum(q * kmin.to(torch.float32), q * kmax.to(torch.float32))
    return bound.sum(dim=3)


def _topk_with_forced(scores: torch.Tensor, k_units: int, n_units: int,
                      forced: Sequence[int]) -> torch.Tensor:
    """Top-k over `scores` [U] with `forced` units pinned in. Returns sorted unit ids."""
    k_units = max(1, min(k_units, n_units))
    scores = scores.clone()
    forced = [u for u in forced if 0 <= u < n_units]
    if forced:
        scores[torch.tensor(forced, device=scores.device, dtype=torch.long)] = float("inf")
    idx = torch.topk(scores, k_units).indices
    return torch.sort(idx).values


def select_units(arm: str, *, scores_shared: torch.Tensor | None,
                 scores_per_head: torch.Tensor | None, n_units: int, k_units: int,
                 generator: torch.Generator | None, device) -> torch.Tensor | list[torch.Tensor]:
    """Choose units for one batch row. Returns unit ids, or a list of them per kv head.

    Every arm force-keeps the last unit: it contains the current token's own key, and an
    arm that drops it is measuring a broken kernel rather than a selection policy.
    """
    k_units = max(1, min(k_units, n_units))
    last = n_units - 1

    if arm == "recency":
        return torch.arange(max(n_units - k_units, 0), n_units, device=device)
    if arm == "sink_recency":
        tail = torch.arange(max(n_units - (k_units - 1), 1), n_units, device=device) \
            if k_units > 1 else torch.empty(0, dtype=torch.long, device=device)
        return torch.unique(torch.cat([torch.zeros(1, dtype=torch.long, device=device), tail]))
    if arm == "random":
        pool = torch.arange(1, max(last, 1), device=device)
        take = min(max(k_units - 2, 0), int(pool.numel()))
        if take:
            perm = torch.randperm(int(pool.numel()), generator=generator, device=device)[:take]
            chosen = pool[perm]
        else:
            chosen = torch.empty(0, dtype=torch.long, device=device)
        pinned = torch.tensor([0, last], device=device, dtype=torch.long)
        return torch.unique(torch.cat([pinned.clamp_max(last), chosen]))
    if arm == "quest_shared_heads":
        assert scores_shared is not None
        return _topk_with_forced(scores_shared, k_units, n_units, (0, last))
    if arm == "quest_per_head":
        assert scores_per_head is not None
        return [_topk_with_forced(scores_per_head[:, h], k_units, n_units, (0, last))
                for h in range(scores_per_head.shape[1])]
    raise ValueError(f"unknown arm {arm!r}")


def units_to_token_index(units: torch.Tensor, granularity: int, seq_len: int) -> torch.Tensor:
    """Expand unit ids into token positions, dropping positions past the sequence end."""
    offsets = torch.arange(granularity, device=units.device)
    idx = (units.unsqueeze(1) * granularity + offsets.unsqueeze(0)).reshape(-1)
    return idx[idx < seq_len]


# ---------------------------------------------------------------------------
# the patched attention
# ---------------------------------------------------------------------------

def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, kvh, s, d = x.shape
    return x[:, :, None].expand(b, kvh, n_rep, s, d).reshape(b, kvh * n_rep, s, d)


class SparseAttentionController:
    """Holds one (arm, granularity, k_frac) configuration and the counters it produces.

    Installed by replacing `modeling_qwen3.eager_attention_forward`, which the model's
    attention module resolves from module globals at call time. Prefill (q_len > 1) and
    any layer listed in `dense_layers` fall through to dense SDPA, because layer 0 was
    measured to have no usable sparse structure and because sparsifying prefill is a
    different experiment.
    """

    def __init__(self, arm: str, granularity: int, k_frac: float, *,
                 dense_layers: Iterable[int] = (0,), seed: int = 0, dense: bool = False):
        self.arm = arm
        self.granularity = int(granularity)
        self.k_frac = float(k_frac)
        self.dense_layers = set(int(x) for x in dense_layers)
        self.dense = bool(dense)
        self.seed = int(seed)
        self.generator: torch.Generator | None = None
        self.selected_tokens = 0.0
        self.total_tokens = 0
        self.sparse_calls = 0
        self.dense_calls = 0

    def reset(self, device) -> None:
        self.selected_tokens = 0.0
        self.total_tokens = 0
        self.sparse_calls = 0
        self.dense_calls = 0
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(self.seed)

    # -- the hook -----------------------------------------------------------
    def __call__(self, module, query, key, value, attention_mask, scaling,
                 dropout: float = 0.0, **kwargs):
        q_len = query.shape[2]
        layer_idx = int(getattr(module, "layer_idx", -1))
        if self.dense or q_len > 1 or layer_idx in self.dense_layers:
            self.dense_calls += 1
            return self._dense(module, query, key, value, attention_mask, scaling)
        self.sparse_calls += 1
        return self._sparse(module, query, key, value, scaling)

    def _dense(self, module, query, key, value, attention_mask, scaling):
        k = _repeat_kv(key, module.num_key_value_groups)
        v = _repeat_kv(value, module.num_key_value_groups)
        mask = None
        if attention_mask is not None:
            mask = attention_mask[:, :, :, : k.shape[-2]]
        out = F.scaled_dot_product_attention(query, k, v, attn_mask=mask, scale=scaling)
        return out.transpose(1, 2).contiguous(), None

    def _sparse(self, module, query, key, value, scaling):
        b, kvh, s, d = key.shape
        if b != 1:
            raise NotImplementedError("this gate runs one sequence at a time on purpose")
        n_units = (s + self.granularity - 1) // self.granularity
        k_units = max(1, int(math.ceil(self.k_frac * n_units)))

        kmin, kmax = unit_minmax(key, self.granularity)
        q_repr = query_repr(query, kvh)

        needs_shared = self.arm == "quest_shared_heads"
        needs_per_head = self.arm == "quest_per_head"
        scores_shared = quest_scores_shared(q_repr, kmin, kmax)[0] if needs_shared else None
        scores_per_head = quest_scores_per_head(q_repr, kmin, kmax)[0] if needs_per_head else None

        picked = select_units(
            self.arm, scores_shared=scores_shared, scores_per_head=scores_per_head,
            n_units=n_units, k_units=k_units, generator=self.generator, device=key.device,
        )
        self.total_tokens += s

        if isinstance(picked, list):
            # per-head selection: each kv head attends over its own token set, so the
            # cost charged is the mean over heads (the CPU would gather each head's set)
            outs = []
            group = module.num_key_value_groups
            per_head_tokens = 0
            for h, units in enumerate(picked):
                idx = units_to_token_index(units, self.granularity, s)
                per_head_tokens += int(idx.numel())
                k_sel = key[:, h : h + 1, idx, :]
                v_sel = value[:, h : h + 1, idx, :]
                q_h = query[:, h * group : (h + 1) * group]
                out_h = F.scaled_dot_product_attention(
                    q_h, k_sel.expand(-1, group, -1, -1), v_sel.expand(-1, group, -1, -1),
                    attn_mask=None, scale=scaling,
                )
                outs.append(out_h)
            self.selected_tokens += per_head_tokens / max(kvh, 1)
            out = torch.cat(outs, dim=1)
            return out.transpose(1, 2).contiguous(), None

        idx = units_to_token_index(picked, self.granularity, s)
        self.selected_tokens += float(idx.numel())
        k_sel = _repeat_kv(key[:, :, idx, :], module.num_key_value_groups)
        v_sel = _repeat_kv(value[:, :, idx, :], module.num_key_value_groups)
        out = F.scaled_dot_product_attention(query, k_sel, v_sel, attn_mask=None, scale=scaling)
        return out.transpose(1, 2).contiguous(), None

    # -- reporting ----------------------------------------------------------
    @property
    def touched_frac(self) -> float:
        """Mean fraction of the context actually read per sparse attention call.

        Reported rather than assumed equal to k/L: forced units and the partial last unit
        make the real cost larger than the nominal budget at small k.
        """
        if not self.total_tokens:
            return 0.0
        return self.selected_tokens / float(self.total_tokens)


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

def token_agreement(dense_ids: Sequence[int], arm_ids: Sequence[int]) -> tuple[float, int]:
    """Fraction of matching tokens and the index of the first divergence (-1 if none)."""
    n = min(len(dense_ids), len(arm_ids))
    first = -1
    match = 0
    for i in range(n):
        if dense_ids[i] == arm_ids[i]:
            match += 1
        elif first < 0:
            first = i
    if len(dense_ids) != len(arm_ids) and first < 0:
        first = n
    return (match / n if n else 0.0), first


def build_verdict(rows: list[dict], dense_correct: bool) -> dict:
    """A configuration is evidence only if the real selector passes where bad arms fail."""
    if not dense_correct:
        return {
            "verdict": "INVALID_PROMPT",
            "reason": "the dense baseline did not answer the question, so nothing measured "
                      "downstream is about sparsity",
            "discriminative_cells": [],
        }
    by_cfg: dict[tuple[int, float], dict[str, dict]] = {}
    for r in rows:
        by_cfg.setdefault((r["granularity"], r["k_frac"]), {})[r["arm"]] = r

    discriminative = []
    for (gran, kf), arms in sorted(by_cfg.items()):
        primary = arms.get(PRIMARY_ARM)
        if primary is None or not primary["answer_correct"]:
            continue
        failing_bad = [a for a in BAD_ARMS if a in arms and not arms[a]["answer_correct"]]
        if failing_bad:
            discriminative.append({
                "granularity": gran, "k_frac": kf,
                "primary_touched_frac": primary["touched_frac"],
                "primary_agreement": primary["token_agreement"],
                "failing_bad_arms": failing_bad,
            })

    if not discriminative:
        all_pass = all(r["answer_correct"] for r in rows if r["arm"] in BAD_ARMS)
        return {
            "verdict": "NON_DISCRIMINATIVE",
            "reason": ("every bad arm answered correctly too, so this prompt cannot license "
                       "a sparsity decision" if all_pass else
                       "the shipped selector did not survive any configuration where a bad arm failed"),
            "discriminative_cells": [],
        }

    cheapest = min(discriminative, key=lambda c: c["primary_touched_frac"])
    return {
        "verdict": "DISCRIMINATIVE",
        "discriminative_cells": discriminative,
        "cheapest_passing": cheapest,
    }


# ---------------------------------------------------------------------------
# model driver
# ---------------------------------------------------------------------------

def _iter_cache_layers(cache):
    if getattr(cache, "key_cache", None):
        return list(zip(cache.key_cache, cache.value_cache))
    if getattr(cache, "layers", None):
        return [(l.keys, l.values) for l in cache.layers]
    raise RuntimeError("unrecognised cache layout; cannot clone the prefill state")


def clone_cache(cache):
    from transformers.cache_utils import DynamicCache
    new = DynamicCache()
    for li, (k, v) in enumerate(_iter_cache_layers(cache)):
        new.update(k.clone(), v.clone(), li)
    return new


def top2_margin(logits: torch.Tensor) -> float:
    """Gap between the best and second-best logit.

    Recorded for the dense run so a token divergence can be attributed instead of
    guessed: if the dense model was itself nearly indifferent at that step, a sparse run
    picking the other branch is not evidence that the selector dropped something needed.
    """
    top2 = torch.topk(logits.float(), 2).values
    return float(top2[0] - top2[1])


def greedy_decode(model, cache, first_token: torch.Tensor, prompt_len: int,
                  max_new_tokens: int, eos_ids: set) -> tuple:
    out_ids: list = []
    margins: list = []
    cur = first_token.view(1, 1)
    past = cache
    pos = prompt_len
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            res = model(
                input_ids=cur,
                past_key_values=past,
                use_cache=True,
                cache_position=torch.tensor([pos], device=cur.device),
            )
            past = res.past_key_values
            logits = res.logits[0, -1]
            nxt = int(logits.argmax())
            out_ids.append(nxt)
            margins.append(round(top2_margin(logits), 4))
            if nxt in eos_ids:
                break
            cur = torch.tensor([[nxt]], device=cur.device)
            pos += 1
    return out_ids, margins


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--variant", default="distractor")
    p.add_argument("--depth", type=float, default=0.5)
    p.add_argument("--num-decoys", type=int, default=4)
    p.add_argument("--granularities", nargs="+", type=int, default=[32, 256])
    p.add_argument("--k-fracs", nargs="+", type=float, default=[0.02, 0.051, 0.11, 0.25])
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--dense-layers", default="0",
                   help="comma list of layers kept dense; layer 0 had no sparse structure")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3 import modeling_qwen3

    from tools.needle_haystack_variants import build_variant

    if not hasattr(modeling_qwen3, "eager_attention_forward"):
        raise RuntimeError("this transformers build has no eager_attention_forward to patch")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(args.device).eval()
    n_layers = model.config.num_hidden_layers
    dense_layers = [int(x) for x in args.dense_layers.split(",") if x.strip() != ""]

    spec = build_variant(tokenizer, args.variant, args.seq_len,
                         depth=args.depth, num_decoys=args.num_decoys, seed=args.seed)
    ids = torch.tensor([spec["ids"]], dtype=torch.long, device=args.device)
    prompt_len = ids.shape[1]
    eos_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()

    original_eager = modeling_qwen3.eager_attention_forward
    dense_ctl = SparseAttentionController("dense", 256, 1.0, dense=True)
    modeling_qwen3.eager_attention_forward = dense_ctl

    print(f"[prefill] variant={args.variant} prompt_len={prompt_len} answer={spec['answer']}",
          flush=True)
    dense_ctl.reset(args.device)
    t0 = time.time()
    with torch.inference_mode():
        prefill = model(input_ids=ids, use_cache=True)
    prefill_cache = prefill.past_key_values
    first_token = prefill.logits[0, -1].argmax().view(1, 1)
    first_margin = top2_margin(prefill.logits[0, -1])
    print(f"[prefill] done in {time.time() - t0:.1f}s", flush=True)

    def run_once(ctl: SparseAttentionController) -> dict:
        modeling_qwen3.eager_attention_forward = ctl
        ctl.reset(args.device)
        cache = clone_cache(prefill_cache)
        t = time.time()
        gen, margins = greedy_decode(model, cache, first_token, prompt_len,
                                     args.max_new_tokens, eos_ids)
        ids_out = [int(first_token)] + gen
        text = tokenizer.decode(ids_out, skip_special_tokens=True)
        del cache
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return {
            "ids": ids_out,
            "text": text,
            "answer_correct": spec["answer"] in text,
            "top2_margins": [round(first_margin, 4)] + margins,
            "touched_frac": ctl.touched_frac,
            "sparse_calls": ctl.sparse_calls,
            "dense_calls": ctl.dense_calls,
            "seconds": round(time.time() - t, 2),
        }

    baseline = run_once(SparseAttentionController("dense", 256, 1.0, dense=True))
    print(f"[dense] correct={baseline['answer_correct']} "
          f"{baseline['seconds']}s text={baseline['text']!r}", flush=True)

    rows: list = []
    for gran in args.granularities:
        for kf in args.k_fracs:
            for arm in args.arms:
                ctl = SparseAttentionController(arm, gran, kf,
                                                dense_layers=dense_layers, seed=args.seed)
                res = run_once(ctl)
                agree, first_div = token_agreement(baseline["ids"], res["ids"])
                dense_margins = baseline["top2_margins"]
                row = {
                    "arm": arm, "granularity": gran, "k_frac": kf,
                    "answer_correct": res["answer_correct"],
                    "token_agreement": round(agree, 4),
                    "first_divergence": first_div,
                    "dense_margin_at_divergence": (
                        dense_margins[first_div]
                        if 0 <= first_div < len(dense_margins) else None
                    ),
                    "touched_frac": round(res["touched_frac"], 5),
                    "text": res["text"],
                    "seconds": res["seconds"],
                }
                rows.append(row)
                print(f"[gran={gran:4d} k/L={kf:.3f} {arm:>18s}] "
                      f"correct={str(row['answer_correct']):5s} "
                      f"agree={row['token_agreement']:.2f} "
                      f"touched={row['touched_frac']:.4f} {res['seconds']}s", flush=True)

    modeling_qwen3.eager_attention_forward = original_eager

    verdict = build_verdict(rows, baseline["answer_correct"])
    payload = {
        "model": args.model,
        "variant": args.variant,
        "prompt_len": prompt_len,
        "answer": spec["answer"],
        "n_layers": n_layers,
        "dense_layers": dense_layers,
        "max_new_tokens": args.max_new_tokens,
        "granularities": args.granularities,
        "k_fracs": args.k_fracs,
        "arms": args.arms,
        "dense_baseline": baseline,
        "rows": rows,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(verdict, indent=2))
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
