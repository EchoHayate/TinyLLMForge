"""Teacher-CoT distillation prototype for latent scratchpad projectors.

This is a deliberately small B1 prototype:

  prompt hidden  -> trainable projector -> teacher-CoT hidden

The LLM stays frozen and is used only to collect hidden-state targets.  The
projector is then evaluated by reinjecting its output as one or more latent
`input_embeds` steps before normal token decoding.
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
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from eval_latent_reinjection import (  # noqa: E402
    HAYSTACK_SENTENCE,
    _append_input_token,
    _last_prefill_hidden,
    generate_with_latent_steps,
)
from tinyvllm import LLM, SamplingParams  # noqa: E402
from tinyvllm.engine.sequence import Sequence, SequenceStatus  # noqa: E402
from tinyvllm.utils.context import reset_context  # noqa: E402


@dataclass
class DistillCase:
    task: str
    expected: str
    prompt: str
    teacher_prefix: str
    answer_token_id: int
    meta: dict


class TrainableRMSLinearProjector(nn.Module):
    """Smallest useful trainable projector: RMSNorm + full linear map."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.linear.weight)
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.weight).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self._norm(x))


def _pad_to_context(tokenizer, core_text: str, ctx_len_tok: int, depth: float) -> str:
    del depth  # B1 tasks must keep the answer prompt at the end of the context.
    core_ids = tokenizer.encode(core_text, add_special_tokens=False)
    budget = max(0, ctx_len_tok - len(core_ids))
    if budget <= 0:
        return tokenizer.decode(core_ids[-ctx_len_tok:])
    repeat_n = max(1, budget // 4)
    haystack_ids = tokenizer.encode(HAYSTACK_SENTENCE * repeat_n, add_special_tokens=False)[:budget]
    return tokenizer.decode(haystack_ids + core_ids)


def build_arithmetic_distill_cases(
    tokenizer,
    num_cases: int,
    ctx_len: int,
    depth: float,
    seed: int,
) -> list[DistillCase]:
    rng = random.Random(seed)
    cases: list[DistillCase] = []
    for _ in range(num_cases):
        a = rng.randint(20, 90)
        b = rng.randint(10, 80)
        c = rng.randint(2, 9)
        d = rng.randint(10, 120)
        expected = (a + b) * c - d
        prompt_core = (
            "\n\nSolve this arithmetic task. You may reason internally, but the visible answer "
            "must be only the final integer.\n"
            f"Problem: Start with {a}. Add {b}. Multiply the result by {c}. Subtract {d}.\n"
            "Reasoning:"
        )
        teacher_prefix = (
            f" ({a} + {b}) = {a + b}; "
            f"{a + b} * {c} = {(a + b) * c}; "
            f"{(a + b) * c} - {d} = {expected}.\n"
            "Final integer:"
        )
        prompt = _pad_to_context(tokenizer, prompt_core, ctx_len, depth)
        answer_token_id = tokenizer.encode(str(expected), add_special_tokens=False)[0]
        cases.append(DistillCase(
            task="arithmetic",
            expected=str(expected),
            prompt=prompt,
            teacher_prefix=teacher_prefix,
            answer_token_id=answer_token_id,
            meta={"a": a, "b": b, "c": c, "d": d},
        ))
    return cases


def build_tool_action_distill_cases(
    tokenizer,
    num_cases: int,
    ctx_len: int,
    depth: float,
    seed: int,
) -> list[DistillCase]:
    scenarios = [
        ("Need to inspect a specific file at /tmp/config.json.", "READ"),
        ("Need to search all Python files for the function name compute_score.", "GREP"),
        ("Need to see what files are inside the current directory.", "LS"),
        ("Need to find occurrences of the word ERROR in logs.", "GREP"),
        ("Need to open README.md and read its content.", "READ"),
        ("Need to list the entries under /data/project.", "LS"),
    ]
    rng = random.Random(seed)
    cases: list[DistillCase] = []
    for _ in range(num_cases):
        observation, expected = scenarios[rng.randrange(len(scenarios))]
        prompt_core = (
            "\n\nYou are choosing the next tool for an agent.\n"
            "Available tool names: LS, READ, GREP.\n"
            f"Observation: {observation}\n"
            "Reasoning:"
        )
        teacher_prefix = f" The best next tool is {expected}.\nTool:"
        prompt = _pad_to_context(tokenizer, prompt_core, ctx_len, depth)
        answer_token_id = tokenizer.encode(expected, add_special_tokens=False)[0]
        cases.append(DistillCase(
            task="tool_action",
            expected=expected,
            prompt=prompt,
            teacher_prefix=teacher_prefix,
            answer_token_id=answer_token_id,
            meta={"observation": observation},
        ))
    return cases


def build_distill_cases(
    task: str,
    tokenizer,
    num_cases: int,
    ctx_len: int,
    depth: float,
    seed: int,
) -> list[DistillCase]:
    if task == "arithmetic":
        return build_arithmetic_distill_cases(tokenizer, num_cases, ctx_len, depth, seed)
    if task == "tool_action":
        return build_tool_action_distill_cases(tokenizer, num_cases, ctx_len, depth, seed)
    raise ValueError(f"unknown task: {task}")


@torch.inference_mode()
def collect_last_hidden(llm: LLM, tokenizer, text: str) -> torch.Tensor:
    runner = llm.model_runner
    block_manager = llm.scheduler.block_manager
    block_manager.hash_to_block_id.clear()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    seq = Sequence(token_ids, SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
    block_manager.allocate(seq)
    seq.status = SequenceStatus.RUNNING
    seq.num_computed_tokens = len(seq)
    seq.prefill_chunk_start = seq.num_cached_tokens
    seq.prefill_chunk_end = len(seq)
    seq.prefill_chunk_final = True
    try:
        input_ids, positions = runner.prepare_prefill([seq])
        _, hidden_states = runner.run_model(input_ids, positions, True, return_hidden=True)
        hidden = _last_prefill_hidden(hidden_states)
        reset_context()
        return hidden.detach().float().cpu()[0]
    finally:
        try:
            reset_context()
        except Exception:
            pass
        block_manager.deallocate(seq)


def collect_pairs(llm: LLM, tokenizer, cases: list[DistillCase]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for i, case in enumerate(cases, 1):
        src = collect_last_hidden(llm, tokenizer, case.prompt)
        tgt = collect_last_hidden(llm, tokenizer, case.prompt + case.teacher_prefix)
        xs.append(src)
        ys.append(tgt)
        print(f"  collected {i}/{len(cases)} expected={case.expected}", flush=True)
    target_ids = torch.tensor([case.answer_token_id for case in cases], dtype=torch.long)
    return torch.stack(xs), torch.stack(ys), target_ids


def _rms_norm_cpu(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)


def train_projector(
    projector: nn.Module,
    src: torch.Tensor,
    tgt: torch.Tensor,
    target_ids: torch.Tensor,
    lm_head_weight: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    train_device: str,
    hidden_loss_weight: float,
    logit_loss_weight: float,
) -> list[dict]:
    device = torch.device(train_device)
    projector.to(device=device, dtype=torch.float32).train()
    src = src.to(device=device, dtype=torch.float32)
    tgt = _rms_norm_cpu(tgt).to(device=device, dtype=torch.float32)
    target_ids = target_ids.to(device=device, dtype=torch.long)
    lm_head_weight = lm_head_weight.detach().to(device=device)
    opt = torch.optim.AdamW((p for p in projector.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)
    history = []
    n = src.size(0)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            pred = projector(src[idx])
            pred_n = _rms_norm_cpu(pred)
            tgt_n = tgt[idx]
            hidden_mse = torch.nn.functional.mse_loss(pred_n, tgt_n)
            hidden_cos = 1.0 - torch.nn.functional.cosine_similarity(pred_n, tgt_n, dim=-1).mean()
            hidden_loss = hidden_mse + 0.1 * hidden_cos
            logits = torch.nn.functional.linear(pred_n.to(dtype=lm_head_weight.dtype), lm_head_weight).float()
            logit_loss = torch.nn.functional.cross_entropy(logits, target_ids[idx])
            loss = hidden_loss_weight * hidden_loss + logit_loss_weight * logit_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * idx.numel()
        avg = total / max(1, n)
        history.append({"epoch": epoch, "loss": avg})
        print(f"  epoch={epoch:03d} loss={avg:.6f}", flush=True)
    return history


def train_projector_soft_step(
    llm: LLM,
    tokenizer,
    projector: nn.Module,
    cases: list[DistillCase],
    src: torch.Tensor,
    epochs: int,
    lr: float,
) -> list[dict]:
    """End-to-end first-answer-token tuning through one latent decode step.

    The model is frozen, but gradients flow through the one-token decode forward
    into `input_embeds` and then into the projector.  This is slow but useful as
    a B1.2 smoke because the optimized object matches deployment better than a
    pure LM-head loss on final hidden states.
    """

    runner = llm.model_runner
    block_manager = llm.scheduler.block_manager
    model = runner.model
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    projector.to(device=model_device, dtype=torch.float32).train()
    src = src.to(device=model_device, dtype=torch.float32)
    target_ids = torch.tensor([case.answer_token_id for case in cases], device=model_device, dtype=torch.long)
    dummy_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    opt = torch.optim.AdamW((p for p in projector.parameters() if p.requires_grad), lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        total = 0.0
        order = torch.randperm(len(cases)).tolist()
        for case_idx in order:
            case = cases[case_idx]
            block_manager.hash_to_block_id.clear()
            prompt_ids = tokenizer.encode(case.prompt, add_special_tokens=False)
            seq = Sequence(prompt_ids, SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
            block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            seq.num_computed_tokens = len(seq)
            seq.prefill_chunk_start = seq.num_cached_tokens
            seq.prefill_chunk_end = len(seq)
            seq.prefill_chunk_final = True
            try:
                with torch.inference_mode():
                    input_ids, positions = runner.prepare_prefill([seq])
                    runner.run_model(input_ids, positions, True)
                    reset_context()

                latent_embed = projector(src[case_idx:case_idx + 1])
                # Decoder layers may reuse/mutate the input hidden buffer internally.
                # Feed a clone so autograd does not see the projector output itself
                # modified in-place by the frozen model forward.
                latent_embed_for_model = latent_embed.to(dtype=model_dtype).clone().contiguous()
                _append_input_token(seq, block_manager, dummy_token_id)
                with torch.inference_mode():
                    input_ids, positions = runner.prepare_decode([seq])
                hidden_states = model(input_ids, positions, input_embeds=latent_embed_for_model)
                logits = model.compute_logits(hidden_states).float()
                loss = torch.nn.functional.cross_entropy(logits, target_ids[case_idx:case_idx + 1])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.item())
            finally:
                try:
                    reset_context()
                except Exception:
                    pass
                block_manager.deallocate(seq)
        avg = total / max(1, len(cases))
        history.append({"epoch": epoch, "soft_step_loss": avg})
        print(f"  soft_epoch={epoch:03d} loss={avg:.6f}", flush=True)
    return history


def extract_distill_answer(task: str, text: str) -> str | None:
    if task == "arithmetic":
        nums = re.findall(r"-?\d+", text)
        return nums[-1] if nums else None
    if task == "tool_action":
        upper = text.upper()
        for tool in ("READ", "GREP", "LS"):
            if re.search(rf"\b{tool}\b", upper):
                return tool
        return None
    raise ValueError(f"unknown task: {task}")


def contains_expected_answer(task: str, text: str, expected: str) -> bool:
    if task == "arithmetic":
        return expected in re.findall(r"-?\d+", text)
    return extract_distill_answer(task, text) == expected


def extract_answer_only(task: str, text: str) -> str | None:
    if task == "arithmetic":
        m = re.match(r"\s*(-?\d+)\b", text)
        return m.group(1) if m else None
    if task == "tool_action":
        m = re.match(r"\s*(READ|GREP|LS)\b", text.upper())
        return m.group(1) if m else None
    raise ValueError(f"unknown task: {task}")


@torch.inference_mode()
def evaluate_projector(
    llm: LLM,
    tokenizer,
    projector: nn.Module,
    cases: list[DistillCase],
    latent_steps: int,
    max_new_tokens: int,
) -> dict:
    model_dtype = next(llm.model_runner.model.parameters()).dtype
    model_device = next(llm.model_runner.model.parameters()).device
    projector.to(device=model_device, dtype=model_dtype).eval()
    dummy_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
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
            max_new_tokens=max_new_tokens,
            dummy_token_id=dummy_token_id,
        )
        ans = extract_distill_answer(case.task, text)
        contains_expected = contains_expected_answer(case.task, text, case.expected)
        answer_only = extract_answer_only(case.task, text)
        answer_only_hit = answer_only == case.expected
        hit = answer_only_hit
        total_tokens += len(token_ids)
        total_time += dt
        details.append({
            "expected": case.expected,
            "answer": ans,
            "contains_expected": contains_expected,
            "answer_only": answer_only,
            "answer_only_hit": answer_only_hit,
            "hit": hit,
            "raw": text[:160],
            "tokens": len(token_ids),
            "time_s": dt,
            "meta": case.meta,
        })
        print(
            f"  eval expected={case.expected} answer_only={answer_only} "
            f"contains={contains_expected} hit={hit} raw={text[:48]!r}",
            flush=True,
        )
    acc = sum(int(x["hit"]) for x in details) / max(1, len(details))
    contains_acc = sum(int(x["contains_expected"]) for x in details) / max(1, len(details))
    tps = total_tokens / total_time if total_time > 0 else 0.0
    return {"accuracy": acc, "contains_accuracy": contains_acc, "throughput_tok_s": tps, "details": details}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", choices=["arithmetic", "tool_action"], default="arithmetic")
    p.add_argument("--train-cases", type=int, default=64)
    p.add_argument("--eval-cases", type=int, default=16)
    p.add_argument("--context-len", type=int, default=512)
    p.add_argument("--depth", type=float, default=0.5)
    p.add_argument("--latent-steps", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-loss-weight", type=float, default=1.0)
    p.add_argument("--logit-loss-weight", type=float, default=0.0)
    p.add_argument("--soft-step-epochs", type=int, default=0)
    p.add_argument("--soft-step-lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-device", type=str, default="cpu")
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--checkpoint", type=str, default="latent_projector.pt")
    p.add_argument("--out-json", type=str, default="latent_projector_train.json")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    llm = LLM(
        args.model,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=1,
    )
    hidden_size = int(llm.model_runner.config.hf_config.hidden_size)
    train_cases = build_distill_cases(args.task, tokenizer, args.train_cases, args.context_len, args.depth, args.seed)
    eval_cases = build_distill_cases(args.task, tokenizer, args.eval_cases, args.context_len, args.depth, args.seed + 10_000)

    t0 = time.time()
    print("Collecting teacher hidden pairs...", flush=True)
    src, tgt, target_ids = collect_pairs(llm, tokenizer, train_cases)
    projector = TrainableRMSLinearProjector(hidden_size)
    print("Training projector...", flush=True)
    history = train_projector(
        projector,
        src,
        tgt,
        target_ids,
        llm.model_runner.model.lm_head.weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_device=args.train_device,
        hidden_loss_weight=args.hidden_loss_weight,
        logit_loss_weight=args.logit_loss_weight,
    )
    soft_history = []
    if args.soft_step_epochs > 0:
        print("Fine-tuning projector with one-step logits loss...", flush=True)
        soft_history = train_projector_soft_step(
            llm,
            tokenizer,
            projector,
            train_cases,
            src,
            epochs=args.soft_step_epochs,
            lr=args.soft_step_lr,
        )
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save({
        "projector": projector.state_dict(),
        "hidden_size": hidden_size,
        "args": vars(args),
        "history": history,
        "soft_history": soft_history,
    }, args.checkpoint)

    eval_result = None
    if not args.skip_eval:
        print("Evaluating latent projector...", flush=True)
        eval_result = evaluate_projector(
            llm,
            tokenizer,
            projector,
            eval_cases,
            latent_steps=args.latent_steps,
            max_new_tokens=args.max_new_tokens,
        )
        print(
            f"eval_acc={eval_result['accuracy'] * 100:.1f}% "
            f"contains_acc={eval_result['contains_accuracy'] * 100:.1f}% "
            f"throughput={eval_result['throughput_tok_s']:.2f} tok/s",
            flush=True,
        )

    result = {
        "args": vars(args),
        "hidden_size": hidden_size,
        "history": history,
        "soft_history": soft_history,
        "eval": eval_result,
        "elapsed_s": time.time() - t0,
        "checkpoint": args.checkpoint,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved results to {args.out_json}")


if __name__ == "__main__":
    main()
