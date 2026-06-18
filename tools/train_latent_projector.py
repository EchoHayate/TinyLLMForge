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
import typing
from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

try:  # noqa: E402
    from eval_latent_reinjection import (
        HAYSTACK_SENTENCE,
        _append_input_token,
        _last_prefill_hidden,
        generate_with_latent_steps,
    )
    from tinyvllm import LLM, SamplingParams
    from tinyvllm.engine.sequence import Sequence, SequenceStatus
    from tinyvllm.utils.context import reset_context
except Exception:  # pragma: no cover - HF teacher-forcing mode does not need TinyLLMForge.
    HAYSTACK_SENTENCE = (
        "The grass is green. The sky is blue. The sun is yellow. "
        "Here we go. There and back again. "
    )
    LLM = object
    SamplingParams = Sequence = SequenceStatus = None
    _append_input_token = _last_prefill_hidden = generate_with_latent_steps = reset_context = None


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


class StepwiseRMSLinearProjector(nn.Module):
    """Per-step latent transition with a learned latent step embedding.

    This is the B1.7 variant: instead of recurrently applying one shared
    projector for every latent token, each latent position has its own
    RMSNorm+Linear transition and a small learned step embedding.
    """

    def __init__(self, hidden_size: int, max_steps: int, eps: float = 1e-6):
        super().__init__()
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        self.max_steps = max_steps
        self.step_embed = nn.Parameter(torch.zeros(max_steps, hidden_size))
        self.transitions = nn.ModuleList([
            TrainableRMSLinearProjector(hidden_size, eps=eps)
            for _ in range(max_steps)
        ])

    def project_step(self, hidden: torch.Tensor, step_idx: int) -> torch.Tensor:
        if step_idx < 0 or step_idx >= self.max_steps:
            raise ValueError(f"step_idx={step_idx} outside max_steps={self.max_steps}")
        step = self.step_embed[step_idx].to(device=hidden.device, dtype=hidden.dtype)
        return self.transitions[step_idx](hidden + step)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project_step(x, 0)


def build_trainable_projector(hidden_size: int, kind: str, max_steps: int) -> nn.Module:
    if kind == "shared":
        return TrainableRMSLinearProjector(hidden_size)
    if kind == "stepwise":
        return StepwiseRMSLinearProjector(hidden_size, max_steps=max_steps)
    raise ValueError(f"unknown projector kind: {kind}")


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
            "".join([
                f" ({a} + {b}) = {a + b};",
                f" {a + b} * {c} = {(a + b) * c};",
                f" {(a + b) * c} - {d} = {expected}.",
                "\nFinal integer:",
            ])
        )
        teacher_steps = [
            f" ({a} + {b}) = {a + b};",
            f" {a + b} * {c} = {(a + b) * c};",
            f" {(a + b) * c} - {d} = {expected}.",
            "\nFinal integer:",
        ]
        prompt = _pad_to_context(tokenizer, prompt_core, ctx_len, depth)
        answer_token_id = tokenizer.encode(str(expected), add_special_tokens=False)[0]
        cases.append(DistillCase(
            task="arithmetic",
            expected=str(expected),
            prompt=prompt,
            teacher_prefix=teacher_prefix,
            answer_token_id=answer_token_id,
            meta={"a": a, "b": b, "c": c, "d": d, "teacher_steps": teacher_steps},
        ))
    return cases


def build_tool_action_distill_cases(
    tokenizer,
    num_cases: int,
    ctx_len: int,
    depth: float,
    seed: int,
) -> list[DistillCase]:
    rng = random.Random(seed)
    files = [
        "README.md",
        "pyproject.toml",
        "src/app.py",
        "configs/prod.yaml",
        "package.json",
        "docs/usage.md",
        "logs/server.log",
        "tests/test_api.py",
    ]
    dirs = [
        ".",
        "src",
        "configs",
        "docs",
        "tests",
        "/data/project",
        "/tmp/workspace",
        "services/api",
    ]
    needles = [
        "compute_score",
        "TODO",
        "ERROR",
        "AuthToken",
        "UserService",
        "timeout_ms",
        "def handler",
        "raise ValueError",
    ]
    read_templates = [
        "Need to inspect the exact contents of file {file} before editing.",
        "The next step is to open {file} and read what it says.",
        "A user asks for the content inside {file}, not for a directory listing.",
        "We already know the path {file}; choose the tool that reads one file.",
    ]
    ls_templates = [
        "Need to see what files are inside directory {dir}.",
        "Before choosing a file, list the entries under {dir}.",
        "We need an overview of the current directory tree at {dir}.",
        "The path {dir} is a directory; choose the tool that lists its children.",
    ]
    grep_templates = [
        "Need to search the repository for occurrences of {needle}.",
        "The next step is to find all files containing the pattern {needle}.",
        "We do not know the file path; search for symbol {needle} across files.",
        "Need to locate log lines or code references matching {needle}.",
    ]
    cases: list[DistillCase] = []
    for _ in range(num_cases):
        expected = rng.choice(["LS", "READ", "GREP"])
        if expected == "READ":
            observation = rng.choice(read_templates).format(file=rng.choice(files))
        elif expected == "LS":
            observation = rng.choice(ls_templates).format(dir=rng.choice(dirs))
        else:
            observation = rng.choice(grep_templates).format(needle=rng.choice(needles))
        if rng.random() < 0.35:
            observation += " Ignore unrelated tool names mentioned in previous notes."
        prompt_core = (
            "\n\nYou are choosing the next tool for an agent.\n"
            "Available tool names: LS, READ, GREP.\n"
            "Return only the next tool name.\n"
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


def build_tool_action_structured_distill_cases(
    tokenizer,
    num_cases: int,
    ctx_len: int,
    depth: float,
    seed: int,
) -> list[DistillCase]:
    rng = random.Random(seed)
    files = [
        "README.md",
        "pyproject.toml",
        "src/app.py",
        "configs/prod.yaml",
        "package.json",
        "docs/usage.md",
        "logs/server.log",
        "tests/test_api.py",
        "services/api/router.go",
        "internal/auth/token.py",
    ]
    dirs = [
        ".",
        "src",
        "configs",
        "docs",
        "tests",
        "/data/project",
        "/tmp/workspace",
        "services/api",
        "internal/auth",
        "needle_sq_results",
    ]
    needles = [
        "compute_score",
        "TODO",
        "ERROR",
        "AuthToken",
        "UserService",
        "timeout_ms",
        "handler_fn",
        "ValueError",
        "ModelRunner",
        "latent_steps",
    ]
    read_templates = [
        "Need to inspect the exact contents of file {path} before editing it.",
        "Open {path}; the user needs the file content, not a directory listing.",
        "We already know the target file path is {path}; read that single file.",
        "The next operation should load source text from {path}.",
    ]
    ls_templates = [
        "Need to see what entries are inside directory {path}.",
        "List the children under {path} before choosing a file.",
        "We need an overview of directory {path}; do not read a specific file yet.",
        "The known path {path} is a folder; list it.",
    ]
    grep_templates = [
        "Need to search across files for pattern {pattern}.",
        "The file path is unknown; locate occurrences of {pattern}.",
        "Find every reference matching {pattern} in the repository.",
        "Search logs or code for the string {pattern}.",
    ]
    cases: list[DistillCase] = []
    for _ in range(num_cases):
        action = rng.choice(["READ", "LS", "GREP"])
        if action == "READ":
            path = rng.choice(files)
            expected = f"READ path={path}"
            observation = rng.choice(read_templates).format(path=path)
            meta = {"action": action, "path": path}
        elif action == "LS":
            path = rng.choice(dirs)
            expected = f"LS path={path}"
            observation = rng.choice(ls_templates).format(path=path)
            meta = {"action": action, "path": path}
        else:
            pattern = rng.choice(needles)
            expected = f"GREP pattern={pattern}"
            observation = rng.choice(grep_templates).format(pattern=pattern)
            meta = {"action": action, "pattern": pattern}
        if rng.random() < 0.35:
            observation += " Previous notes may mention LS, READ, or GREP; ignore those distractors."
        prompt_core = (
            "\n\nYou are choosing the next structured tool action for an agent.\n"
            "Available action schemas:\n"
            "- LS path=<directory>\n"
            "- READ path=<file>\n"
            "- GREP pattern=<query>\n"
            "Return only one structured action in exactly one of those schemas.\n"
            f"Observation: {observation}\n"
            "Reasoning:"
        )
        teacher_prefix = f" The correct structured action is {expected}.\nAction:"
        prompt = _pad_to_context(tokenizer, prompt_core, ctx_len, depth)
        answer_token_id = tokenizer.encode(expected, add_special_tokens=False)[0]
        cases.append(DistillCase(
            task="tool_action_structured",
            expected=expected,
            prompt=prompt,
            teacher_prefix=teacher_prefix,
            answer_token_id=answer_token_id,
            meta={"observation": observation, **meta},
        ))
    return cases


def build_tool_action_structured2_distill_cases(
    tokenizer,
    num_cases: int,
    ctx_len: int,
    depth: float,
    seed: int,
) -> list[DistillCase]:
    rng = random.Random(seed)
    files = [
        "README.md",
        "pyproject.toml",
        "src/app.py",
        "configs/prod.yaml",
        "package.json",
        "docs/usage.md",
        "logs/server.log",
        "tests/test_api.py",
        "services/api/router.go",
        "internal/auth/token.py",
    ]
    dirs = [
        ".",
        "src",
        "configs",
        "docs",
        "tests",
        "/data/project",
        "/tmp/workspace",
        "services/api",
        "internal/auth",
        "needle_sq_results",
    ]
    needles = [
        "compute_score",
        "TODO",
        "ERROR",
        "AuthToken",
        "UserService",
        "timeout_ms",
        "handler_fn",
        "ValueError",
        "ModelRunner",
        "latent_steps",
    ]
    cases: list[DistillCase] = []
    for _ in range(num_cases):
        action = rng.choice(["READ", "LS", "GREP"])
        if action == "READ":
            path = rng.choice(files)
            expected = f"READ path={path}"
            observation = rng.choice([
                "Need to inspect file {path}; return a structured read action.",
                "Open file {path} and load its contents.",
                "The target is a known file path {path}, not a directory.",
            ]).format(path=path)
            meta = {"action": action, "path": path}
        elif action == "LS":
            path = rng.choice(dirs)
            expected = f"LS path={path}"
            observation = rng.choice([
                "Need to list entries under directory {path}.",
                "Inspect the children of folder {path} before choosing a file.",
                "The target is a known directory {path}, so list it.",
            ]).format(path=path)
            meta = {"action": action, "path": path}
        else:
            pattern = rng.choice(needles)
            path = rng.choice(dirs)
            expected = f"GREP pattern={pattern} path={path}"
            observation = rng.choice([
                "Search for pattern {pattern} under {path}.",
                "The query is {pattern}; restrict the search scope to {path}.",
                "Find references matching {pattern} inside directory {path}.",
            ]).format(pattern=pattern, path=path)
            meta = {"action": action, "pattern": pattern, "path": path}
        if rng.random() < 0.35:
            observation += " Distractor: previous notes may mention other tools or paths."
        prompt_core = (
            "\n\nYou are choosing the next structured tool action for an agent.\n"
            "Available action schemas:\n"
            "- LS path=<directory>\n"
            "- READ path=<file>\n"
            "- GREP pattern=<query> path=<directory>\n"
            "Return only one structured action in exactly one schema.\n"
            f"Observation: {observation}\n"
            "Reasoning:"
        )
        teacher_prefix = f" The correct structured action is {expected}.\nAction:"
        prompt = _pad_to_context(tokenizer, prompt_core, ctx_len, depth)
        answer_token_id = tokenizer.encode(expected, add_special_tokens=False)[0]
        cases.append(DistillCase(
            task="tool_action_structured2",
            expected=expected,
            prompt=prompt,
            teacher_prefix=teacher_prefix,
            answer_token_id=answer_token_id,
            meta={"observation": observation, **meta},
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
    if task == "tool_action_structured":
        return build_tool_action_structured_distill_cases(tokenizer, num_cases, ctx_len, depth, seed)
    if task == "tool_action_structured2":
        return build_tool_action_structured2_distill_cases(tokenizer, num_cases, ctx_len, depth, seed)
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


def _hf_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unknown dtype: {name}")


def _patch_torch_custom_op_string_annotations() -> None:
    """Compat for newer Transformers custom ops on older torch releases.

    Transformers may define custom-op functions under ``from __future__ import
    annotations``.  Older torch versions inspect ``fn.__annotations__`` directly
    and reject string values like ``"torch.Tensor"`` while inferring schemas.
    Resolve those strings just before registration so HF model import remains
    usable without changing the environment.
    """

    original_custom_op = torch.library.custom_op
    if getattr(original_custom_op, "_latent_string_annotation_patch", False):
        return

    def resolve_annotation(value):
        if not isinstance(value, str):
            return value
        simple = {
            "torch.Tensor": torch.Tensor,
            "Tensor": torch.Tensor,
            "int": int,
            "float": float,
            "bool": bool,
            "str": str,
            "torch.dtype": torch.dtype,
            "torch.device": torch.device,
        }
        if value in simple:
            return simple[value]
        wrappers = (
            ("Optional[", typing.Optional),
            ("typing.Optional[", typing.Optional),
            ("Sequence[", typing.Sequence),
            ("typing.Sequence[", typing.Sequence),
            ("List[", typing.List),
            ("typing.List[", typing.List),
        )
        for prefix, wrapper in wrappers:
            if value.startswith(prefix) and value.endswith("]"):
                inner = value[len(prefix):-1]
                resolved_inner = resolve_annotation(inner)
                if not isinstance(resolved_inner, str):
                    return wrapper[resolved_inner]
        return value

    def patch_fn(fn):
        annotations = getattr(fn, "__annotations__", None)
        if annotations and any(isinstance(v, str) for v in annotations.values()):
            fn.__annotations__ = {k: resolve_annotation(v) for k, v in annotations.items()}
        return fn

    def custom_op_compat(name, fn=None, /, **kwargs):
        if fn is None:
            decorator = original_custom_op(name, **kwargs)

            def wrapped_decorator(inner_fn):
                return decorator(patch_fn(inner_fn))

            return wrapped_decorator
        return original_custom_op(name, patch_fn(fn), **kwargs)

    custom_op_compat._latent_string_annotation_patch = True
    torch.library.custom_op = custom_op_compat


def _answer_token_ids(tokenizer, case: DistillCase, append_eos: bool = True) -> list[int]:
    ids = tokenizer.encode(case.expected, add_special_tokens=False)
    if not ids:
        raise ValueError(f"empty answer tokens for expected={case.expected!r}")
    if append_eos and tokenizer.eos_token_id is not None:
        ids = ids + [int(tokenizer.eos_token_id)]
    return ids


def _last_nonpad_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return the rightmost non-padding index for either left or right padding."""

    if attention_mask.dim() != 2:
        raise ValueError(f"expected 2D attention_mask, got shape={tuple(attention_mask.shape)}")
    flipped = attention_mask.flip(dims=[1])
    return attention_mask.size(1) - 1 - flipped.argmax(dim=1)


@torch.no_grad()
def collect_source_hidden_hf(model, tokenizer, prompts: list[str], device: torch.device) -> torch.Tensor:
    batch = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    last_idx = _last_nonpad_indices(attention_mask)
    rows = torch.arange(input_ids.size(0), device=device)
    return out.hidden_states[-1][rows, last_idx].detach().float()


@torch.no_grad()
def collect_last_hidden_texts_hf(model, tokenizer, texts: list[str], device: torch.device) -> torch.Tensor:
    return collect_source_hidden_hf(model, tokenizer, texts, device)


def _teacher_steps(case: DistillCase) -> list[str]:
    steps = case.meta.get("teacher_steps")
    if isinstance(steps, list) and steps:
        return [str(x) for x in steps]
    return [case.teacher_prefix]


@torch.no_grad()
def collect_step_teacher_hiddens_hf(
    model,
    tokenizer,
    cases: list[DistillCase],
    device: torch.device,
    latent_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    texts = []
    slots = []
    for case_idx, case in enumerate(cases):
        prefix = ""
        for step_idx, step_text in enumerate(_teacher_steps(case)[:latent_steps]):
            prefix += step_text
            texts.append(case.prompt + prefix)
            slots.append((case_idx, step_idx))

    hidden_size = int(model.config.hidden_size)
    targets = torch.zeros(len(cases), latent_steps, hidden_size, device=device, dtype=torch.float32)
    mask = torch.zeros(len(cases), latent_steps, device=device, dtype=torch.bool)
    if not texts:
        return targets, mask

    hiddens = collect_last_hidden_texts_hf(model, tokenizer, texts, device)
    for hidden, (case_idx, step_idx) in zip(hiddens, slots):
        targets[case_idx, step_idx] = hidden
        mask[case_idx, step_idx] = True
    return targets, mask


def _project_latent_sequence(
    projector: nn.Module,
    src: torch.Tensor,
    latent_steps: int,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    if latent_steps < 1:
        raise ValueError(f"latent_steps must be >= 1, got {latent_steps}")
    latents = []
    hidden = src.to(dtype=torch.float32)
    for step_idx in range(latent_steps):
        if hasattr(projector, "project_step"):
            hidden = projector.project_step(hidden, step_idx)
        else:
            hidden = projector(hidden)
        latents.append(hidden.to(dtype=model_dtype))
    return torch.stack(latents, dim=1)


def _build_hf_teacher_batch(
    model,
    tokenizer,
    projector,
    cases: list[DistillCase],
    src: torch.Tensor,
    device: torch.device,
    latent_steps: int,
):
    embed = model.get_input_embeddings()
    model_dtype = next(model.parameters()).dtype
    seq_embeds = []
    answer_targets = []
    answer_positions = []
    latent_positions = []
    max_len = 0
    for i, case in enumerate(cases):
        prompt_ids = torch.tensor(tokenizer.encode(case.prompt, add_special_tokens=False), device=device, dtype=torch.long)
        answer_ids = torch.tensor(_answer_token_ids(tokenizer, case), device=device, dtype=torch.long)
        prompt_embeds = embed(prompt_ids).detach()
        latents = _project_latent_sequence(
            projector,
            src[i:i + 1].to(device=device, dtype=torch.float32),
            latent_steps,
            model_dtype,
        ).squeeze(0)
        if answer_ids.numel() > 1:
            answer_prefix = embed(answer_ids[:-1]).detach()
            parts = [prompt_embeds, latents, answer_prefix]
        else:
            parts = [prompt_embeds, latents]
        sample_embeds = torch.cat(parts, dim=0)
        first_answer_position = prompt_embeds.size(0) + latent_steps - 1
        sample_latent_positions = torch.arange(
            prompt_embeds.size(0),
            prompt_embeds.size(0) + latent_steps,
            device=device,
            dtype=torch.long,
        )
        positions = torch.arange(
            first_answer_position,
            first_answer_position + answer_ids.numel(),
            device=device,
            dtype=torch.long,
        )
        seq_embeds.append(sample_embeds)
        answer_targets.append(answer_ids)
        answer_positions.append(positions)
        latent_positions.append(sample_latent_positions)
        max_len = max(max_len, sample_embeds.size(0))

    hidden_size = seq_embeds[0].size(-1)
    inputs_embeds = torch.zeros(len(cases), max_len, hidden_size, device=device, dtype=model_dtype)
    attention_mask = torch.zeros(len(cases), max_len, device=device, dtype=torch.long)
    flat_positions = []
    flat_targets = []
    flat_latent_positions = []
    latent_batch = []
    for i, sample_embeds in enumerate(seq_embeds):
        n = sample_embeds.size(0)
        inputs_embeds[i, :n] = sample_embeds
        attention_mask[i, :n] = 1
        flat_positions.append(answer_positions[i] + i * max_len)
        flat_latent_positions.append(latent_positions[i] + i * max_len)
        flat_targets.append(answer_targets[i])
        latent_start = int(latent_positions[i][0].item())
        latent_batch.append(sample_embeds[latent_start:latent_start + latent_steps])
    return (
        inputs_embeds,
        attention_mask,
        torch.cat(flat_positions),
        torch.cat(flat_targets),
        torch.stack(latent_batch),
        torch.cat(flat_latent_positions),
    )


def train_projector_hf_teacher_forcing(
    model,
    tokenizer,
    projector: nn.Module,
    cases: list[DistillCase],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    latent_steps: int,
    step_hidden_loss_weight: float,
    step_hidden_mode: str,
    latent_step_curriculum: str,
) -> list[dict]:
    if step_hidden_mode not in ("input", "output"):
        raise ValueError(f"unknown step_hidden_mode={step_hidden_mode!r}")
    if latent_step_curriculum not in ("none", "linear"):
        raise ValueError(f"unknown latent_step_curriculum={latent_step_curriculum!r}")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    projector.to(device=device, dtype=torch.float32).train()
    opt = torch.optim.AdamW((p for p in projector.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(1, epochs + 1):
        effective_latent_steps = latent_steps
        if latent_step_curriculum == "linear":
            effective_latent_steps = max(1, min(latent_steps, (latent_steps * epoch + epochs - 1) // epochs))
        rng = random.Random(epoch)
        order = list(range(len(cases)))
        rng.shuffle(order)
        total = 0.0
        total_ce = 0.0
        total_step_hidden = 0.0
        total_tokens = 0
        total_step_slots = 0
        for start in range(0, len(order), batch_size):
            idxs = order[start:start + batch_size]
            batch_cases = [cases[i] for i in idxs]
            prompts = [case.prompt for case in batch_cases]
            with torch.no_grad():
                src = collect_source_hidden_hf(model, tokenizer, prompts, device)
                if step_hidden_loss_weight > 0:
                    step_targets, step_mask = collect_step_teacher_hiddens_hf(
                        model, tokenizer, batch_cases, device, effective_latent_steps
                    )
                else:
                    step_targets = step_mask = None
            (
                inputs_embeds,
                attention_mask,
                flat_positions,
                targets,
                latent_embeds,
                flat_latent_positions,
            ) = _build_hf_teacher_batch(
                model, tokenizer, projector, batch_cases, src, device, effective_latent_steps
            )
            out = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=step_hidden_loss_weight > 0 and step_hidden_mode == "output",
                use_cache=False,
                return_dict=True,
            )
            logits = out.logits.reshape(-1, out.logits.size(-1))[flat_positions].float()
            ce_loss = torch.nn.functional.cross_entropy(logits, targets)
            loss = ce_loss
            step_hidden_loss_value = None
            if step_hidden_loss_weight > 0 and step_mask is not None and step_mask.any():
                if step_hidden_mode == "output":
                    output_hiddens = out.hidden_states[-1].reshape(-1, out.hidden_states[-1].size(-1))[
                        flat_latent_positions
                    ].view(len(batch_cases), effective_latent_steps, -1)
                    pred = _rms_norm_cpu(output_hiddens.float())
                else:
                    pred = _rms_norm_cpu(latent_embeds.float())
                tgt = _rms_norm_cpu(step_targets.float())
                mask = step_mask.unsqueeze(-1)
                hidden_mse = ((pred - tgt).pow(2) * mask).sum() / (mask.sum().clamp_min(1) * pred.size(-1))
                hidden_cos_each = 1.0 - torch.nn.functional.cosine_similarity(pred, tgt, dim=-1)
                hidden_cos = (hidden_cos_each * step_mask.float()).sum() / step_mask.float().sum().clamp_min(1)
                step_hidden_loss_value = hidden_mse + 0.1 * hidden_cos
                loss = loss + step_hidden_loss_weight * step_hidden_loss_value
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * targets.numel()
            total_ce += float(ce_loss.item()) * targets.numel()
            if step_hidden_loss_value is not None:
                slots = int(step_mask.sum().item())
                total_step_hidden += float(step_hidden_loss_value.item()) * slots
                total_step_slots += slots
            total_tokens += int(targets.numel())
        avg = total / max(1, total_tokens)
        avg_ce = total_ce / max(1, total_tokens)
        avg_step_hidden = total_step_hidden / max(1, total_step_slots) if total_step_slots else 0.0
        history.append({
            "epoch": epoch,
            "teacher_forcing_loss": avg,
            "ce_loss": avg_ce,
            "step_hidden_loss": avg_step_hidden,
            "step_hidden_mode": step_hidden_mode,
            "latent_steps": effective_latent_steps,
            "latent_step_curriculum": latent_step_curriculum,
        })
        print(
            f"  hf_epoch={epoch:03d} latent_steps={effective_latent_steps} "
            f"loss={avg:.6f} ce={avg_ce:.6f} step_hidden={avg_step_hidden:.6f}",
            flush=True,
        )
    return history


@torch.no_grad()
def generate_hf_with_latent(
    model,
    tokenizer,
    projector,
    case: DistillCase,
    max_new_tokens: int,
    device: torch.device,
    latent_steps: int,
) -> str:
    model.eval()
    projector.eval()
    prompt_ids = torch.tensor(tokenizer.encode(case.prompt, add_special_tokens=False), device=device, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(prompt_ids)
    out = model(
        input_ids=prompt_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    src = out.hidden_states[-1][:, -1].float()
    embed = model.get_input_embeddings()
    prompt_embeds = embed(prompt_ids).detach()
    latents = _project_latent_sequence(projector, src, latent_steps, prompt_embeds.dtype)
    inputs_embeds = torch.cat([prompt_embeds, latents], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)
    generated = []
    past = None
    for step in range(max_new_tokens):
        if step == 0:
            out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=True, return_dict=True)
        else:
            out = model(input_ids=next_id, past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        token = int(next_id.item())
        generated.append(token)
        if tokenizer.eos_token_id is not None and token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def evaluate_hf_teacher_forcing(
    model,
    tokenizer,
    projector,
    cases: list[DistillCase],
    max_new_tokens: int,
    device: torch.device,
    latent_steps: int,
) -> dict:
    details = []
    for case in cases:
        text = generate_hf_with_latent(model, tokenizer, projector, case, max_new_tokens, device, latent_steps)
        answer_only = extract_answer_only(case.task, text)
        contains_expected = contains_expected_answer(case.task, text, case.expected)
        hit = answer_only == case.expected
        details.append({
            "expected": case.expected,
            "answer_only": answer_only,
            "contains_expected": contains_expected,
            "hit": hit,
            "raw": text[:160],
            "meta": case.meta,
        })
        print(
            f"  hf_eval expected={case.expected} answer_only={answer_only} "
            f"contains={contains_expected} hit={hit} raw={text[:48]!r}",
            flush=True,
        )
    acc = sum(int(x["hit"]) for x in details) / max(1, len(details))
    contains_acc = sum(int(x["contains_expected"]) for x in details) / max(1, len(details))
    return {"accuracy": acc, "contains_accuracy": contains_acc, "details": details}


def run_hf_teacher_forcing(args) -> None:
    device = torch.device(args.hf_device)
    dtype = _hf_dtype(args.hf_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Keep HF source-hidden collection aligned with the manually right-padded
    # inputs_embeds teacher-forcing batch below.  Some decoder-only tokenizers
    # default to left padding for generation, which otherwise makes batched
    # source hidden states differ from the unpadded eval path.
    tokenizer.padding_side = "right"
    _patch_torch_custom_op_string_annotations()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    hidden_size = int(model.config.hidden_size)
    train_cases = build_distill_cases(args.task, tokenizer, args.train_cases, args.context_len, args.depth, args.seed)
    eval_cases = build_distill_cases(args.task, tokenizer, args.eval_cases, args.context_len, args.depth, args.seed + 10_000)
    projector = build_trainable_projector(
        hidden_size,
        args.projector_kind,
        max_steps=max(args.max_latent_steps, args.latent_steps),
    )
    t0 = time.time()
    print("Training HF teacher-forcing latent projector...", flush=True)
    history = train_projector_hf_teacher_forcing(
        model,
        tokenizer,
        projector,
        train_cases,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        latent_steps=args.latent_steps,
        step_hidden_loss_weight=args.step_hidden_loss_weight,
        step_hidden_mode=args.step_hidden_mode,
        latent_step_curriculum=args.latent_step_curriculum,
    )
    eval_result = None
    if not args.skip_eval:
        print("Evaluating HF latent projector...", flush=True)
        eval_result = evaluate_hf_teacher_forcing(
            model,
            tokenizer,
            projector,
            eval_cases,
            args.max_new_tokens,
            device,
            args.latent_steps,
        )
        print(
            f"hf_eval_acc={eval_result['accuracy'] * 100:.1f}% "
            f"contains_acc={eval_result['contains_accuracy'] * 100:.1f}%",
            flush=True,
        )
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save({
        "projector": projector.state_dict(),
        "hidden_size": hidden_size,
        "args": vars(args),
        "history": history,
    }, args.checkpoint)
    result = {
        "args": vars(args),
        "hidden_size": hidden_size,
        "history": history,
        "eval": eval_result,
        "elapsed_s": time.time() - t0,
        "checkpoint": args.checkpoint,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved results to {args.out_json}")


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
    if task in ("tool_action_structured", "tool_action_structured2"):
        return _extract_structured_action(text)
    raise ValueError(f"unknown task: {task}")


def _normalize_structured_action(text: str | None) -> str | None:
    if text is None:
        return None
    text = " ".join(text.strip().split())
    m = re.match(r"^(READ|LS)\s+path=([^\s]+)$", text, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()} path={m.group(2)}"
    m = re.match(r"^GREP\s+pattern=([^\s]+)$", text, flags=re.IGNORECASE)
    if m:
        return f"GREP pattern={m.group(1)}"
    m = re.match(r"^GREP\s+pattern=([^\s]+)\s+path=([^\s]+)$", text, flags=re.IGNORECASE)
    if m:
        return f"GREP pattern={m.group(1)} path={m.group(2)}"
    return None


def _extract_structured_action(text: str) -> str | None:
    m = re.search(r"\b(READ|LS)\s+path=([^\s]+)", text, flags=re.IGNORECASE)
    if m:
        return _normalize_structured_action(f"{m.group(1)} path={m.group(2)}")
    m = re.search(r"\bGREP\s+pattern=([^\s]+)", text, flags=re.IGNORECASE)
    if m:
        maybe_two_slot = re.search(r"\bGREP\s+pattern=([^\s]+)\s+path=([^\s]+)", text, flags=re.IGNORECASE)
        if maybe_two_slot:
            return _normalize_structured_action(
                f"GREP pattern={maybe_two_slot.group(1)} path={maybe_two_slot.group(2)}"
            )
        return _normalize_structured_action(f"GREP pattern={m.group(1)}")
    return None


def contains_expected_answer(task: str, text: str, expected: str) -> bool:
    if task == "arithmetic":
        return expected in re.findall(r"-?\d+", text)
    if task in ("tool_action_structured", "tool_action_structured2"):
        return _extract_structured_action(text) == _normalize_structured_action(expected)
    return extract_distill_answer(task, text) == expected


def extract_answer_only(task: str, text: str) -> str | None:
    if task == "arithmetic":
        m = re.match(r"\s*(-?\d+)\b", text)
        return m.group(1) if m else None
    if task == "tool_action":
        m = re.match(r"\s*(READ|GREP|LS)\b", text.upper())
        return m.group(1) if m else None
    if task in ("tool_action_structured", "tool_action_structured2"):
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        return _normalize_structured_action(first_line)
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
    p.add_argument(
        "--task",
        choices=["arithmetic", "tool_action", "tool_action_structured", "tool_action_structured2"],
        default="arithmetic",
    )
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
    p.add_argument("--hf-teacher-forcing", action="store_true")
    p.add_argument("--hf-device", type=str, default="cuda")
    p.add_argument("--hf-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--step-hidden-loss-weight", type=float, default=0.0)
    p.add_argument("--step-hidden-mode", choices=["input", "output"], default="input")
    p.add_argument("--projector-kind", choices=["shared", "stepwise"], default="shared")
    p.add_argument("--max-latent-steps", type=int, default=16)
    p.add_argument("--latent-step-curriculum", choices=["none", "linear"], default="none")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.hf_teacher_forcing:
        run_hf_teacher_forcing(args)
        return
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
    projector = build_trainable_projector(
        hidden_size,
        args.projector_kind,
        max_steps=max(args.max_latent_steps, args.latent_steps),
    )
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
