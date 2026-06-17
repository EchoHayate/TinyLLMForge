"""Hidden-state reinjection smoke test for latent scratchpad experiments.

This script uses TinyLLMForge's own model runner so it exercises the real KV-cache
path. It performs:

  prompt prefill -> take final hidden state -> K latent input_embeds steps
  -> greedy token decode.

The latent projector is intentionally untrained; this is only a smoke test for
whether hidden-state reinjection immediately collapses.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoTokenizer

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tinyvllm import LLM, SamplingParams  # noqa: E402
from tinyvllm.engine.sequence import Sequence, SequenceStatus  # noqa: E402
from tinyvllm.utils.context import get_context, reset_context  # noqa: E402


HAYSTACK_SENTENCE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again. "
)
QUESTION = "\n\nWhat is the magic number? Answer with only the digits, nothing else."
NEEDLE_TEMPLATE = "The magic number is {num}. Remember it. "


class IdentityProjector(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class RMSNormProjector(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.weight).to(dtype=x.dtype)


class LinearIdentityProjector(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.linear.weight)
        self.linear.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SmallMLPProjector(nn.Module):
    """Identity-initialized residual MLP, still untrained."""

    def __init__(self, hidden_size: int):
        super().__init__()
        inner = max(16, hidden_size // 4)
        self.norm = RMSNormProjector(hidden_size)
        self.up = nn.Linear(hidden_size, inner, bias=False)
        self.down = nn.Linear(inner, hidden_size, bias=False)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.down.weight)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.down(torch.nn.functional.silu(self.up(self.norm(x))))


def make_projector(name: str, hidden_size: int, dtype: torch.dtype, device: torch.device) -> nn.Module:
    if name == "identity":
        module = IdentityProjector()
    elif name == "rmsnorm":
        module = RMSNormProjector(hidden_size)
    elif name == "linear":
        module = LinearIdentityProjector(hidden_size)
    elif name == "mlp":
        module = SmallMLPProjector(hidden_size)
    else:
        raise ValueError(f"unknown projector: {name}")
    return module.to(device=device, dtype=dtype).eval()


def _pad_prompt_to_context(tokenizer, core_text: str, ctx_len_tok: int, depth: float) -> str:
    question_ids = tokenizer.encode(core_text, add_special_tokens=False)
    budget = max(0, ctx_len_tok - len(question_ids))
    if budget <= 0:
        return tokenizer.decode(question_ids[:ctx_len_tok])
    repeat_n = max(1, budget // 4)
    haystack_ids = tokenizer.encode(HAYSTACK_SENTENCE * repeat_n, add_special_tokens=False)[:budget]
    insert_pos = int(depth * len(haystack_ids))
    return tokenizer.decode(haystack_ids[:insert_pos] + question_ids + haystack_ids[insert_pos:])


def build_needle_prompt(tokenizer, ctx_len_tok: int, depth: float, magic_num: int) -> str:
    needle = NEEDLE_TEMPLATE.format(num=magic_num)
    target_filler_tok = max(64, ctx_len_tok - 64)
    repeat_n = max(1, target_filler_tok // 4)
    haystack = HAYSTACK_SENTENCE * repeat_n

    haystack_ids = tokenizer.encode(haystack, add_special_tokens=False)
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    question_ids = tokenizer.encode(QUESTION, add_special_tokens=False)

    budget = max(64, ctx_len_tok - len(needle_ids) - len(question_ids))
    haystack_ids = haystack_ids[:budget]
    insert_pos = int(depth * len(haystack_ids))
    full_ids = haystack_ids[:insert_pos] + needle_ids + haystack_ids[insert_pos:] + question_ids
    return tokenizer.decode(full_ids)


def _first_int(text: str) -> str | None:
    m = re.search(r"-?\d+", text)
    return m.group(0) if m else None


def extract_answer(task: str, text: str) -> str | None:
    if task in ("needle", "arithmetic"):
        return _first_int(text)
    if task == "tool_action":
        upper = text.upper()
        for tool in ("READ", "GREP", "LS"):
            if re.search(rf"\b{tool}\b", upper):
                return tool
        return None
    raise ValueError(f"unknown task: {task}")


@dataclass
class EvalCase:
    task: str
    ctx_len: int
    depth: float
    trial: int
    expected: str
    prompt: str
    meta: dict


def build_needle_cases(tokenizer, context_lens: list[int], depths: list[float], num_trials: int, seed: int) -> list[EvalCase]:
    rng = random.Random(seed)
    cases = []
    for ctx_len in context_lens:
        for depth in depths:
            for trial in range(num_trials):
                magic = rng.randint(10000, 99999)
                prompt = build_needle_prompt(tokenizer, ctx_len, depth, magic)
                cases.append(EvalCase("needle", ctx_len, depth, trial, str(magic), prompt, {"magic": magic}))
    return cases


def build_arithmetic_cases(tokenizer, context_lens: list[int], depths: list[float], num_trials: int, seed: int) -> list[EvalCase]:
    rng = random.Random(seed)
    cases = []
    for ctx_len in context_lens:
        for depth in depths:
            for trial in range(num_trials):
                a = rng.randint(20, 90)
                b = rng.randint(10, 80)
                c = rng.randint(2, 9)
                d = rng.randint(10, 120)
                expected = (a + b) * c - d
                core = (
                    "\n\nSolve this arithmetic task. Answer with only the final integer, nothing else.\n"
                    f"Problem: Start with {a}. Add {b}. Multiply the result by {c}. Subtract {d}.\n"
                    "Final integer:"
                )
                prompt = _pad_prompt_to_context(tokenizer, core, ctx_len, depth)
                cases.append(EvalCase(
                    "arithmetic", ctx_len, depth, trial, str(expected), prompt,
                    {"a": a, "b": b, "c": c, "d": d},
                ))
    return cases


def build_tool_action_cases(tokenizer, context_lens: list[int], depths: list[float], num_trials: int, seed: int) -> list[EvalCase]:
    scenarios = [
        ("Need to inspect a specific file at /tmp/config.json.", "READ"),
        ("Need to search all Python files for the function name compute_score.", "GREP"),
        ("Need to see what files are inside the current directory.", "LS"),
        ("Need to find occurrences of the word ERROR in logs.", "GREP"),
        ("Need to open README.md and read its content.", "READ"),
        ("Need to list the entries under /data/project.", "LS"),
    ]
    rng = random.Random(seed)
    cases = []
    for ctx_len in context_lens:
        for depth in depths:
            for trial in range(num_trials):
                observation, expected = scenarios[rng.randrange(len(scenarios))]
                core = (
                    "\n\nYou are choosing the next tool for an agent.\n"
                    "Available tool names: LS, READ, GREP.\n"
                    f"Observation: {observation}\n"
                    "Answer with only one tool name.\n"
                    "Tool:"
                )
                prompt = _pad_prompt_to_context(tokenizer, core, ctx_len, depth)
                cases.append(EvalCase(
                    "tool_action", ctx_len, depth, trial, expected, prompt,
                    {"observation": observation},
                ))
    return cases


def build_cases(task: str, tokenizer, context_lens: list[int], depths: list[float], num_trials: int, seed: int) -> list[EvalCase]:
    if task == "needle":
        return build_needle_cases(tokenizer, context_lens, depths, num_trials, seed)
    if task == "arithmetic":
        return build_arithmetic_cases(tokenizer, context_lens, depths, num_trials, seed)
    if task == "tool_action":
        return build_tool_action_cases(tokenizer, context_lens, depths, num_trials, seed)
    raise ValueError(f"unknown task: {task}")


def _last_prefill_hidden(hidden_states: torch.Tensor) -> torch.Tensor:
    context = get_context()
    last_indices = context.logits_indices
    if last_indices is None:
        last_indices = context.cu_seqlens_q[1:] - 1
    return hidden_states[last_indices].contiguous()


def _append_input_token(seq: Sequence, block_manager, token_id: int) -> None:
    seq.append_token(int(token_id))
    if not block_manager.can_append(seq):
        raise RuntimeError("KV block manager cannot append token for latent smoke test")
    block_manager.may_append(seq)


@torch.inference_mode()
def generate_with_latent_steps(
    llm: LLM,
    tokenizer,
    prompt: str,
    projector: nn.Module,
    latent_steps: int,
    max_new_tokens: int,
    dummy_token_id: int,
) -> tuple[str, list[int], float]:
    runner = llm.model_runner
    block_manager = llm.scheduler.block_manager
    # Prefix-cache reuse can turn repeated fixed prompts into zero-token prefill,
    # which is not useful for this smoke test because we need fresh last hidden states.
    block_manager.hash_to_block_id.clear()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    seq = Sequence(prompt_ids, SamplingParams(temperature=0.0, max_tokens=max_new_tokens, ignore_eos=False))
    block_manager.allocate(seq)
    seq.status = SequenceStatus.RUNNING
    seq.num_computed_tokens = len(seq)
    seq.prefill_chunk_start = seq.num_cached_tokens
    seq.prefill_chunk_end = len(seq)
    seq.prefill_chunk_final = True

    t0 = time.time()
    generated: list[int] = []
    try:
        input_ids, positions = runner.prepare_prefill([seq])
        logits, hidden_states = runner.run_model(input_ids, positions, True, return_hidden=True)
        hidden = _last_prefill_hidden(hidden_states)
        reset_context()

        for _ in range(int(latent_steps)):
            latent_embed = projector(hidden).contiguous()
            _append_input_token(seq, block_manager, dummy_token_id)
            input_ids, positions = runner.prepare_decode([seq])
            logits, hidden_states = runner.run_model(
                input_ids,
                positions,
                False,
                input_embeds=latent_embed,
                return_hidden=True,
            )
            hidden = hidden_states[-1:].contiguous()
            reset_context()

        for _ in range(max_new_tokens):
            next_id = int(torch.argmax(logits[0], dim=-1).item())
            generated.append(next_id)
            if next_id == tokenizer.eos_token_id:
                break
            _append_input_token(seq, block_manager, next_id)
            input_ids, positions = runner.prepare_decode([seq])
            logits, hidden_states = runner.run_model(input_ids, positions, False, return_hidden=True)
            hidden = hidden_states[-1:].contiguous()
            reset_context()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
    finally:
        try:
            reset_context()
        except Exception:
            pass
        block_manager.deallocate(seq)
    return tokenizer.decode(generated, skip_special_tokens=True), generated, dt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", type=str, default="needle", choices=["needle", "arithmetic", "tool_action"])
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    p.add_argument("--context-lens", type=int, nargs="+", default=[512, 1024])
    p.add_argument("--depths", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--latent-steps-list", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    p.add_argument("--projectors", type=str, nargs="+", default=["identity", "rmsnorm", "linear", "mlp"],
                   choices=["identity", "rmsnorm", "linear", "mlp"])
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", type=str, default="latent_reinjection_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    llm = LLM(
        args.model,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=1,
    )
    hidden_size = int(llm.model_runner.config.hf_config.hidden_size)
    model_dtype = next(llm.model_runner.model.parameters()).dtype
    model_device = next(llm.model_runner.model.parameters()).device
    dummy_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    cases = build_cases(args.task, tokenizer, args.context_lens, args.depths, args.num_trials, args.seed)

    all_results = []
    for projector_name in args.projectors:
        projector = make_projector(projector_name, hidden_size, model_dtype, model_device)
        for latent_steps in args.latent_steps_list:
            print(f"\n=== projector={projector_name} latent_steps={latent_steps} ===", flush=True)
            details = []
            total_tokens = 0
            total_time = 0.0
            for case in cases:
                text, token_ids, dt = generate_with_latent_steps(
                    llm,
                    tokenizer,
                    case.prompt,
                    projector,
                    latent_steps=latent_steps,
                    max_new_tokens=args.max_new_tokens,
                    dummy_token_id=dummy_token_id,
                )
                ans = extract_answer(case.task, text)
                hit = ans == case.expected
                total_tokens += len(token_ids)
                total_time += dt
                details.append(dict(
                    ctx_len=case.ctx_len,
                    depth=case.depth,
                    trial=case.trial,
                    expected=case.expected,
                    answer=ans,
                    hit=hit,
                    raw=text[:128],
                    output_tokens=len(token_ids),
                    time_s=dt,
                    meta=case.meta,
                ))
                print(
                    f"  ctx={case.ctx_len:>5} depth={case.depth:.2f} "
                    f"expected={case.expected} answer={ans} hit={hit} raw={text[:40]!r}",
                    flush=True,
                )
            acc = sum(int(x["hit"]) for x in details) / max(1, len(details))
            tps = total_tokens / total_time if total_time > 0 else 0.0
            print(f"  overall_acc={acc*100:.1f}%  throughput={tps:.2f} tok/s", flush=True)
            all_results.append(dict(
                task=args.task,
                projector=projector_name,
                latent_steps=latent_steps,
                overall_accuracy=acc,
                throughput_tok_s=tps,
                total_time_s=total_time,
                total_output_tokens=total_tokens,
                details=details,
            ))

    print("\n========== SUMMARY ==========")
    print(f"{'projector':>10} | {'K':>3} | {'acc':>7} | {'tok/s':>9}")
    print("-" * 40)
    for r in all_results:
        print(f"{r['projector']:>10} | {r['latent_steps']:>3} | {r['overall_accuracy']*100:6.1f}% | {r['throughput_tok_s']:9.2f}")

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(dict(args=vars(args), results=all_results), f, indent=2)
    print(f"\nDetails saved to {args.out_json}")


if __name__ == "__main__":
    main()
