"""Does a token actually come out of the CPU path, and is it the *right* token?

Why this file exists
--------------------
Everything measured so far is performance: the selector keeps the answer (fidelity gate),
the CPU can compute sparse attention in 2.05 ms (compute gate), a CUDA-graph engine only
inflates 1.5 ms under CPU load (overlap gate), and cross-microbatch pipelining hides 97% of
the CPU time provided the host sleeps instead of spinning (mechanism gates). Not one of those
runs produced a single token through the offload path. They measured stand-ins.

This harness closes that hole. The KV cache lives in CPU DRAM, the attention arithmetic
happens on the CPU, and the tokens that come out are compared against a GPU reference that
uses the *identical* selected indices. Three things can be wrong and only one of them is
interesting:

  1. the selection differs (bug in incremental summary maintenance)
  2. the arithmetic differs beyond bf16 noise (bug in the CPU gather/attention)
  3. the selection and arithmetic agree, and the generation still diverges (that is the
     fidelity story already measured, not a correctness failure)

So the harness reports all three separately, and the summary maintenance is checked against
a from-scratch recompute rather than trusted.

Design, following the gates' conclusions
----------------------------------------
  - K/V bytes live in **CPU** memory (pinned), written one token per step, which is the real
    KV write path rather than a bulk copy.
  - min/max unit summaries live on the **GPU** and are maintained **incrementally**. Gate 5
    measured a CPU-side selector at gran=32 as 7.9-17.5 ms versus 0.26 ms on GPU, so the
    selector belongs on the GPU. That means the GPU keeps the summaries and only sends
    *indices* down, not Q.
  - the CPU gathers the selected units out of its mirror and computes attention in fp32.

The selection math is imported from `tools/e2e_sparse_attention.py` rather than reimplemented,
because a correctness harness that quietly disagrees with the fidelity harness proves nothing.

Numerics
--------
The CPU accumulates in fp32 while the GPU reference runs bf16 like the model. fp32 CPU output
is therefore *more* accurate than the bf16 GPU path, so the honest comparison needs both:

  - vs a GPU **fp32** reference with the same indices: this is the real correctness bound and
    should sit at ~1e-6.
  - vs the GPU **bf16** reference: this shows how much of the difference is just bf16, i.e.
    the noise the model already tolerates.

Usage (on a GPU box):
    python tools/cpu_offload_correctness.py \
        --model /path/to/Qwen3-8B --seq-len 8192 --granularity 32 --k-frac 0.051 \
        --max-new-tokens 16 --out-json /tmp/cpu-offload-correctness.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.e2e_sparse_attention import (  # noqa: E402
    SparseAttentionController,
    _repeat_kv,
    _topk_with_forced,
    clone_cache,
    greedy_decode,
    quest_scores_shared,
    query_repr,
    token_agreement,
    top2_margin,
    unit_minmax,
    units_to_token_index,
)


class CpuKvMirror:
    """KV in CPU DRAM, min/max summaries on the GPU, both maintained incrementally.

    One instance holds every layer. Allocation is lazy per layer because the capacity is only
    known once the prefill length is seen, and pinning 36 layers of a wrong guess is a slow
    way to run out of memory.

    The summaries are the part most likely to be silently wrong, for one specific reason: a
    unit that has only been partially filled must not be widened by the tokens it does not yet
    contain, and a unit that is starting must not inherit the previous occupant's bounds. Both
    cases are handled by tracking how many tokens each unit has actually seen, and both are
    checked against a from-scratch recompute by `drift_vs_recompute`.
    """

    def __init__(self, granularity: int, device: str, pin: bool = True) -> None:
        self.granularity = int(granularity)
        self.device = device
        self.pin = bool(pin)
        self.k: dict[int, torch.Tensor] = {}
        self.v: dict[int, torch.Tensor] = {}
        self.kmin: dict[int, torch.Tensor] = {}
        self.kmax: dict[int, torch.Tensor] = {}
        self.length: dict[int, int] = {}
        self.d2h_bytes = 0
        self.h2d_bytes = 0

    # -- allocation ---------------------------------------------------------
    def _alloc(self, layer: int, kv_heads: int, dim: int, capacity: int,
               dtype: torch.dtype) -> None:
        n_units = (capacity + self.granularity - 1) // self.granularity
        self.k[layer] = torch.empty((1, kv_heads, capacity, dim), dtype=dtype,
                                    device="cpu", pin_memory=self.pin)
        self.v[layer] = torch.empty((1, kv_heads, capacity, dim), dtype=dtype,
                                    device="cpu", pin_memory=self.pin)
        # summaries on the GPU in fp32: they are tiny (36 MiB bf16 / 72 MiB fp32 for a whole
        # 8k model, per gate 5) and fp32 keeps the bound exact rather than merely close
        self.kmin[layer] = torch.full((1, n_units, kv_heads, dim), float("inf"),
                                      dtype=torch.float32, device=self.device)
        self.kmax[layer] = torch.full((1, n_units, kv_heads, dim), float("-inf"),
                                      dtype=torch.float32, device=self.device)
        self.length[layer] = 0

    # -- writes -------------------------------------------------------------
    def append(self, layer: int, key: torch.Tensor, value: torch.Tensor,
               capacity_hint: int | None = None) -> None:
        """Append the tokens of `key`/`value` that this mirror has not seen yet.

        `key`/`value` are [1, KVH, S, D] on the GPU and hold the whole history, because that
        is what the HF cache hands to the attention hook. Only the tail is copied, so the
        traffic is one token per step after the first call - the real write path.
        """
        b, kvh, s, d = key.shape
        if b != 1:
            raise NotImplementedError("one sequence at a time, on purpose")
        if layer not in self.k:
            cap = int(capacity_hint or s)
            if cap < s:
                raise ValueError(f"capacity_hint {cap} < prefill length {s}")
            self._alloc(layer, kvh, d, cap, key.dtype)
        have = self.length[layer]
        if s == have:
            return
        if s < have:
            raise ValueError(f"layer {layer} shrank: {have} -> {s}")
        if s > self.k[layer].shape[2]:
            raise ValueError(f"layer {layer} overflowed capacity {self.k[layer].shape[2]}")

        k_new = key[:, :, have:s, :]
        v_new = value[:, :, have:s, :]
        self.k[layer][:, :, have:s, :].copy_(k_new)          # D2H
        self.v[layer][:, :, have:s, :].copy_(v_new)          # D2H
        self.d2h_bytes += k_new.numel() * k_new.element_size() * 2
        self._fold(layer, k_new.to(torch.float32), have)
        self.length[layer] = s

    def _fold(self, layer: int, k_new_f32: torch.Tensor, start: int) -> None:
        """Fold new keys into the per-unit min/max, one unit at a time.

        Done on the GPU where the summaries live. The loop is over units touched by this
        write, which is one unit per step during decode.
        """
        gran = self.granularity
        n_new = k_new_f32.shape[2]
        kmin, kmax = self.kmin[layer], self.kmax[layer]
        pos = start
        while pos < start + n_new:
            u = pos // gran
            unit_end = min((u + 1) * gran, start + n_new)
            chunk = k_new_f32[:, :, pos - start:unit_end - start, :]     # [1,KVH,n,D]
            # summaries are [1, U, KVH, D]; the chunk reductions are [KVH, D]
            cmin = chunk.amin(dim=2)[0]     # [KVH, D]
            cmax = chunk.amax(dim=2)[0]
            torch.minimum(kmin[0, u], cmin, out=kmin[0, u])
            torch.maximum(kmax[0, u], cmax, out=kmax[0, u])
            pos = unit_end

    # -- reads --------------------------------------------------------------
    def summaries(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Live summaries, trimmed to the units that actually hold tokens."""
        n_units = (self.length[layer] + self.granularity - 1) // self.granularity
        return self.kmin[layer][:, :n_units], self.kmax[layer][:, :n_units]

    def gather(self, layer: int, idx_cpu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather selected tokens out of CPU DRAM. This is the CPU-side staging buffer."""
        k = self.k[layer][:, :, : self.length[layer], :]
        v = self.v[layer][:, :, : self.length[layer], :]
        return k[:, :, idx_cpu, :], v[:, :, idx_cpu, :]

    def drift_vs_recompute(self, layer: int, key_full: torch.Tensor) -> float:
        """Max abs difference between the maintained summaries and a from-scratch recompute.

        Should be exactly 0.0. Anything else is an incremental-maintenance bug, which would
        show up as a *different selection* rather than as numerical noise.
        """
        ref_min, ref_max = unit_minmax(key_full.to(torch.float32), self.granularity)
        cur_min, cur_max = self.summaries(layer)
        return float(max((cur_min - ref_min.to(cur_min)).abs().max().item(),
                         (cur_max - ref_max.to(cur_max)).abs().max().item()))


def cpu_sparse_attention(query_cpu: torch.Tensor, k_sel: torch.Tensor, v_sel: torch.Tensor,
                         scaling: float, n_rep: int) -> torch.Tensor:
    """Attention over the gathered units, on the CPU, accumulating in fp32.

    query_cpu: [1, H, 1, D]; k_sel/v_sel: [1, KVH, T, D]. Returns [1, H, 1, D] fp32.
    """
    q = query_cpu.to(torch.float32)
    k = _repeat_kv(k_sel.to(torch.float32), n_rep)
    v = _repeat_kv(v_sel.to(torch.float32), n_rep)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, scale=scaling)


class CpuOffloadController:
    """The offload path, installed in place of `modeling_qwen3.eager_attention_forward`.

    Per sparse call: write the new token's K/V to CPU DRAM, fold the summaries on the GPU,
    rank units on the GPU, ship *indices* down, gather and attend on the CPU, ship the output
    back up. Prefill and the dense layers fall through to GPU SDPA, exactly as in the fidelity
    harness - offloading prefill is a different experiment and layer 0 has no sparse structure.
    """

    def __init__(self, granularity: int, k_frac: float, *, dense_layers=(0,),
                 capacity: int, device: str, pin: bool = True, verify_every: int = 1,
                 verify: bool = True) -> None:
        self.granularity = int(granularity)
        self.k_frac = float(k_frac)
        self.dense_layers = {int(x) for x in dense_layers}
        self.capacity = int(capacity)
        self.device = device
        self.verify_every = max(1, int(verify_every))
        self.verify = bool(verify)
        self.mirror = CpuKvMirror(granularity, device, pin=pin)
        self.reset(device)

    def reset(self, device) -> None:
        self.device = str(device)
        self.mirror = CpuKvMirror(self.granularity, self.device, pin=self.mirror.pin)
        self.sparse_calls = 0
        self.dense_calls = 0
        self.selected_tokens = 0.0
        self.total_tokens = 0
        self.err_fp32: list[float] = []
        self.err_bf16: list[float] = []
        self.summary_drift: list[float] = []
        self.idx_mismatches = 0
        self.idx_checks = 0
        self.cpu_seconds = 0.0

    # -- the hook -----------------------------------------------------------
    def __call__(self, module, query, key, value, attention_mask, scaling,
                 dropout: float = 0.0, **kwargs):
        q_len = query.shape[2]
        layer_idx = int(getattr(module, "layer_idx", -1))
        if q_len > 1 or layer_idx in self.dense_layers:
            # still mirror the KV, otherwise the dense layers' history would be missing if
            # they were ever sparsified, and the write cost would be understated
            self.mirror.append(layer_idx, key, value, capacity_hint=self.capacity)
            self.dense_calls += 1
            k = _repeat_kv(key, module.num_key_value_groups)
            v = _repeat_kv(value, module.num_key_value_groups)
            mask = attention_mask[:, :, :, : k.shape[-2]] if attention_mask is not None else None
            out = F.scaled_dot_product_attention(query, k, v, attn_mask=mask, scale=scaling)
            return out.transpose(1, 2).contiguous(), None

        self.sparse_calls += 1
        return self._offload(module, query, key, value, scaling, layer_idx)

    def _offload(self, module, query, key, value, scaling, layer_idx: int):
        b, kvh, s, d = key.shape
        group = module.num_key_value_groups

        # 1. KV write: one token per step, GPU -> CPU DRAM
        self.mirror.append(layer_idx, key, value, capacity_hint=self.capacity)

        # 2. selection, on the GPU, from the incrementally maintained summaries
        n_units = (s + self.granularity - 1) // self.granularity
        k_units = max(1, int(math.ceil(self.k_frac * n_units)))
        kmin, kmax = self.mirror.summaries(layer_idx)
        q_repr = query_repr(query, kvh)
        scores = quest_scores_shared(q_repr, kmin, kmax)[0]
        picked = _topk_with_forced(scores, k_units, n_units, forced=(0, n_units - 1))
        idx = units_to_token_index(picked, self.granularity, s)
        self.total_tokens += s
        self.selected_tokens += float(idx.numel())

        # 3. indices down to the CPU (not Q: gate 5 put the selector on the GPU)
        idx_cpu = idx.to("cpu", non_blocking=False)
        q_cpu = query.to("cpu", non_blocking=False)

        # 4. gather and attend on the CPU
        t0 = time.perf_counter()
        k_sel, v_sel = self.mirror.gather(layer_idx, idx_cpu)
        out_cpu = cpu_sparse_attention(q_cpu, k_sel, v_sel, scaling, group)
        self.cpu_seconds += time.perf_counter() - t0

        # 5. output back up to the GPU
        out = out_cpu.to(self.device, dtype=query.dtype)
        self.mirror.h2d_bytes += out.numel() * out.element_size()

        if self.verify and (self.sparse_calls % self.verify_every == 0):
            self._verify(key, value, query, scaling, group, idx, layer_idx, scores,
                         k_units, n_units, out_cpu)

        return out.transpose(1, 2).contiguous(), None

    # -- verification -------------------------------------------------------
    def _verify(self, key, value, query, scaling, group, idx, layer_idx, scores,
                k_units, n_units, out_cpu) -> None:
        """Three independent checks, because they fail for different reasons."""
        # (a) did the incremental summaries stay identical to a recompute?
        drift = self.mirror.drift_vs_recompute(layer_idx, key[:, :, : self.mirror.length[layer_idx], :])
        self.summary_drift.append(drift)

        # (b) would a recomputed summary have selected the same units?
        ref_min, ref_max = unit_minmax(key.to(torch.float32), self.granularity)
        ref_scores = quest_scores_shared(query_repr(query, key.shape[1]), ref_min, ref_max)[0]
        ref_picked = _topk_with_forced(ref_scores, k_units, n_units, forced=(0, n_units - 1))
        cur_picked = _topk_with_forced(scores, k_units, n_units, forced=(0, n_units - 1))
        self.idx_checks += 1
        if not torch.equal(ref_picked, cur_picked):
            self.idx_mismatches += 1

        # (c) is the CPU arithmetic right? Compare against the GPU with the SAME indices,
        # in fp32 (the real bound) and in bf16 (how much of it is just the model's dtype).
        k_sel_g = key[:, :, idx, :]
        v_sel_g = value[:, :, idx, :]
        ref32 = F.scaled_dot_product_attention(
            query.to(torch.float32), _repeat_kv(k_sel_g.to(torch.float32), group),
            _repeat_kv(v_sel_g.to(torch.float32), group), attn_mask=None, scale=scaling)
        refbf = F.scaled_dot_product_attention(
            query, _repeat_kv(k_sel_g, group), _repeat_kv(v_sel_g, group),
            attn_mask=None, scale=scaling)
        cpu32 = out_cpu.to(ref32.device)
        scale = ref32.abs().max().clamp(min=1e-6)
        self.err_fp32.append(float(((cpu32 - ref32).abs().max() / scale).item()))
        self.err_bf16.append(float(((cpu32 - refbf.to(torch.float32)).abs().max() / scale).item()))

    # -- reporting ----------------------------------------------------------
    @property
    def touched_frac(self) -> float:
        return self.selected_tokens / float(self.total_tokens) if self.total_tokens else 0.0

    def stats(self) -> dict:
        def summarise(xs: list[float]) -> dict:
            if not xs:
                return {"n": 0}
            return {"n": len(xs), "max": max(xs),
                    "mean": sum(xs) / len(xs), "min": min(xs)}
        return {
            "sparse_calls": self.sparse_calls,
            "dense_calls": self.dense_calls,
            "touched_frac": round(self.touched_frac, 5),
            "summary_drift": summarise(self.summary_drift),
            "index_checks": self.idx_checks,
            "index_mismatches": self.idx_mismatches,
            "rel_err_vs_gpu_fp32": summarise(self.err_fp32),
            "rel_err_vs_gpu_bf16": summarise(self.err_bf16),
            "cpu_seconds": round(self.cpu_seconds, 3),
            "d2h_mib": round(self.mirror.d2h_bytes / (1 << 20), 2),
            "h2d_mib": round(self.mirror.h2d_bytes / (1 << 20), 2),
        }


def verdict(stats: dict, agree_vs_gpu: float, agree_vs_dense: float,
            answer_correct: bool) -> dict:
    """Correctness is a conjunction, and each clause names its own failure."""
    checks = {
        "summaries_exact": stats["summary_drift"].get("max", 1.0) == 0.0,
        "selection_identical": stats["index_mismatches"] == 0,
        "arithmetic_matches_fp32": stats["rel_err_vs_gpu_fp32"].get("max", 1.0) < 1e-4,
        "tokens_match_gpu_sparse": agree_vs_gpu >= 0.999,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "note": ("token agreement against the *dense* arm is a fidelity number, not a "
                 "correctness one: the selector is allowed to change the continuation, "
                 "the CPU is not."),
        "agreement_vs_gpu_sparse": round(agree_vs_gpu, 4),
        "agreement_vs_dense": round(agree_vs_dense, 4),
        "answer_correct": answer_correct,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None,
                   help="path to real weights; omit and pass --random-model instead when the "
                        "checkpoint is not on the box")
    p.add_argument("--random-model", action="store_true",
                   help="build a randomly initialised Qwen3 with the *same per-layer shape* "
                        "as Qwen3-8B (32 q heads, 8 kv heads, head_dim 128) but few layers. "
                        "Legitimate for this gate and not for the fidelity gate: every check "
                        "here compares two paths through the SAME weights, so what the weights "
                        "mean is irrelevant. Only answer-level fidelity needs a real "
                        "checkpoint, and that was measured separately.")
    p.add_argument("--random-layers", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--variant", default="distractor")
    p.add_argument("--depth", type=float, default=0.5)
    p.add_argument("--num-decoys", type=int, default=4)
    p.add_argument("--granularity", type=int, default=32)
    p.add_argument("--k-frac", type=float, default=0.051)
    p.add_argument("--dense-layers", default="0")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--verify-every", type=int, default=1,
                   help="verify every Nth sparse call; the fp32 reference is not free")
    p.add_argument("--no-pin", action="store_true", help="skip pinned memory for the mirror")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3 import modeling_qwen3

    if not hasattr(modeling_qwen3, "eager_attention_forward"):
        raise RuntimeError("this transformers build has no eager_attention_forward to patch")
    if not args.model and not args.random_model:
        raise SystemExit("pass --model <path> or --random-model")

    dense_layers = [int(x) for x in args.dense_layers.split(",") if x.strip() != ""]

    if args.random_model:
        from transformers.models.qwen3 import Qwen3Config
        torch.manual_seed(args.seed)
        cfg = Qwen3Config(
            vocab_size=1024, hidden_size=4096, intermediate_size=1024,
            num_hidden_layers=args.random_layers, num_attention_heads=32,
            num_key_value_heads=8, head_dim=128, max_position_embeddings=args.seq_len + 64,
            attn_implementation="eager",
        )
        model = AutoModelForCausalLM.from_config(cfg)
        model = model.to(device=args.device, dtype=torch.bfloat16).eval()
        ids = torch.randint(0, cfg.vocab_size, (1, args.seq_len), device=args.device)
        spec = {"answer": None, "ids": ids[0].tolist()}
        tokenizer = None
        eos_ids = set()
        print(f"[random model] layers={args.random_layers} kv_heads=8 head_dim=128 "
              f"(shape of Qwen3-8B, weights meaningless on purpose)", flush=True)
    else:
        from tools.needle_haystack_variants import build_variant
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="eager"
        ).to(args.device).eval()
        spec = build_variant(tokenizer, args.variant, args.seq_len,
                             depth=args.depth, num_decoys=args.num_decoys, seed=args.seed)
        ids = torch.tensor([spec["ids"]], dtype=torch.long, device=args.device)
        eos_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    prompt_len = ids.shape[1]
    capacity = prompt_len + args.max_new_tokens + 2

    original_eager = modeling_qwen3.eager_attention_forward
    dense_ctl = SparseAttentionController("dense", 256, 1.0, dense=True)
    modeling_qwen3.eager_attention_forward = dense_ctl
    dense_ctl.reset(args.device)
    print(f"[prefill] prompt_len={prompt_len} answer={spec['answer']!r}", flush=True)
    t0 = time.time()
    with torch.inference_mode():
        prefill = model(input_ids=ids, use_cache=True)
    prefill_cache = prefill.past_key_values
    first_token = prefill.logits[0, -1].argmax().view(1, 1)
    first_margin = top2_margin(prefill.logits[0, -1])
    print(f"[prefill] {time.time() - t0:.1f}s", flush=True)

    def run(ctl) -> dict:
        modeling_qwen3.eager_attention_forward = ctl
        ctl.reset(args.device)
        cache = clone_cache(prefill_cache)
        t = time.time()
        gen, margins = greedy_decode(model, cache, first_token, prompt_len,
                                     args.max_new_tokens, eos_ids)
        out_ids = [int(first_token)] + gen
        text = tokenizer.decode(out_ids, skip_special_tokens=True) if tokenizer else ""
        del cache
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return {"ids": out_ids, "text": text, "seconds": round(time.time() - t, 2),
                "answer_correct": (spec["answer"] in text) if spec["answer"] else None,
                "margins": [round(first_margin, 4)] + margins}

    dense = run(SparseAttentionController("dense", 256, 1.0, dense=True))
    print(f"[dense]      {dense['seconds']}s correct={dense['answer_correct']} "
          f"{dense['text']!r}", flush=True)

    gpu_sparse = run(SparseAttentionController(
        "quest_shared_heads", args.granularity, args.k_frac,
        dense_layers=dense_layers, seed=args.seed))
    print(f"[gpu sparse] {gpu_sparse['seconds']}s correct={gpu_sparse['answer_correct']} "
          f"{gpu_sparse['text']!r}", flush=True)

    cpu_ctl = CpuOffloadController(args.granularity, args.k_frac,
                                   dense_layers=dense_layers, capacity=capacity,
                                   device=args.device, pin=not args.no_pin,
                                   verify_every=args.verify_every)
    cpu = run(cpu_ctl)
    stats = cpu_ctl.stats()
    print(f"[cpu offload] {cpu['seconds']}s correct={cpu['answer_correct']} "
          f"{cpu['text']!r}", flush=True)

    agree_gpu, first_div_gpu = token_agreement(gpu_sparse["ids"], cpu["ids"])
    agree_dense, first_div_dense = token_agreement(dense["ids"], cpu["ids"])
    v = verdict(stats, agree_gpu, agree_dense, cpu["answer_correct"])

    print("\n--- correctness ---", flush=True)
    print(f"summary drift max      {stats['summary_drift'].get('max')}  (must be 0.0)")
    print(f"selection mismatches   {stats['index_mismatches']} / {stats['index_checks']}")
    print(f"rel err vs GPU fp32    max={stats['rel_err_vs_gpu_fp32'].get('max'):.3e} "
          f"mean={stats['rel_err_vs_gpu_fp32'].get('mean'):.3e}")
    print(f"rel err vs GPU bf16    max={stats['rel_err_vs_gpu_bf16'].get('max'):.3e} "
          f"mean={stats['rel_err_vs_gpu_bf16'].get('mean'):.3e}")
    print(f"tokens vs gpu sparse   agree={agree_gpu:.4f} first_div={first_div_gpu}")
    print(f"tokens vs dense        agree={agree_dense:.4f} first_div={first_div_dense}")
    print(f"touched frac           {stats['touched_frac']}")
    print(f"traffic                D2H {stats['d2h_mib']} MiB, H2D {stats['h2d_mib']} MiB")
    print(f"VERDICT                {'PASS' if v['passed'] else 'FAIL'}  {v['checks']}",
          flush=True)

    modeling_qwen3.eager_attention_forward = original_eager
    payload = {"config": dict(vars(args)), "prompt_len": prompt_len,
               "answer": spec["answer"], "dense": dense, "gpu_sparse": gpu_sparse,
               "cpu_offload": cpu, "cpu_stats": stats, "verdict": v}
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out_json}", flush=True)
    return 0 if v["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
