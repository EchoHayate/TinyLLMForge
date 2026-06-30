"""Teacher-CoT distillation prototype for latent scratchpad projectors.

This is a deliberately small B1 prototype:

  prompt hidden  -> trainable projector -> teacher-CoT hidden

The LLM stays frozen and is used only to collect hidden-state targets.  The
projector is then evaluated by reinjecting its output as one or more latent
`input_embeds` steps before normal token decoding.
"""

from __future__ import annotations

import argparse
import hashlib
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


class ArithmeticStateMachineProjector(nn.Module):
    """Stepwise arithmetic state machine with built-in transition heads.

    This is the B2 variant: each latent position is an explicit transition
    phase (sum -> product -> final), and the projector itself owns the value
    and phase heads used by the state-machine trainer.  The heads are not used
    at inference time; they make transition correctness measurable during
    training without introducing a separate side probe module.
    """

    PHASE_SUM = 0
    PHASE_PRODUCT = 1
    PHASE_FINAL = 2

    def __init__(
        self,
        hidden_size: int,
        max_steps: int,
        min_value: int,
        max_value: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        if max_value < min_value:
            raise ValueError(f"max_value={max_value} must be >= min_value={min_value}")
        self.max_steps = max_steps
        self.min_value = min_value
        self.max_value = max_value
        self.num_values = max_value - min_value + 1
        self.step_embed = nn.Parameter(torch.zeros(max_steps, hidden_size))
        self.phase_embed = nn.Parameter(torch.zeros(3, hidden_size))
        self.transitions = nn.ModuleList([
            TrainableRMSLinearProjector(hidden_size, eps=eps)
            for _ in range(max_steps)
        ])
        self.value_heads = nn.ModuleList([
            nn.Linear(hidden_size, self.num_values)
            for _ in range(max_steps)
        ])
        self.phase_heads = nn.ModuleList([
            nn.Linear(hidden_size, 3)
            for _ in range(max_steps)
        ])

    def _phase_for_step(self, step_idx: int) -> int:
        if step_idx == 0:
            return self.PHASE_SUM
        if step_idx == 1:
            return self.PHASE_PRODUCT
        return self.PHASE_FINAL

    def project_step(self, hidden: torch.Tensor, step_idx: int) -> torch.Tensor:
        if step_idx < 0 or step_idx >= self.max_steps:
            raise ValueError(f"step_idx={step_idx} outside max_steps={self.max_steps}")
        step = self.step_embed[step_idx].to(device=hidden.device, dtype=hidden.dtype)
        phase = self.phase_embed[self._phase_for_step(step_idx)].to(device=hidden.device, dtype=hidden.dtype)
        return self.transitions[step_idx](hidden + step + phase)

    def state_machine_logits(self, hidden: torch.Tensor, step_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if step_idx < 0 or step_idx >= self.max_steps:
            raise ValueError(f"step_idx={step_idx} outside max_steps={self.max_steps}")
        x = hidden.float()
        return self.value_heads[step_idx](x), self.phase_heads[step_idx](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project_step(x, 0)


def build_trainable_projector(
    hidden_size: int,
    kind: str,
    max_steps: int,
    state_min_value: int = -256,
    state_max_value: int = 2048,
) -> nn.Module:
    if kind == "shared":
        return TrainableRMSLinearProjector(hidden_size)
    if kind == "stepwise":
        return StepwiseRMSLinearProjector(hidden_size, max_steps=max_steps)
    if kind == "arithmetic_state_machine":
        return ArithmeticStateMachineProjector(
            hidden_size,
            max_steps=max_steps,
            min_value=state_min_value,
            max_value=state_max_value,
        )
    raise ValueError(f"unknown projector kind: {kind}")


class ArithmeticStateProbe(nn.Module):
    """Training-only probes for explicit arithmetic intermediate-state supervision."""

    def __init__(self, hidden_size: int, max_steps: int, min_value: int, max_value: int):
        super().__init__()
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        if max_value < min_value:
            raise ValueError(f"max_value={max_value} must be >= min_value={min_value}")
        self.max_steps = max_steps
        self.min_value = min_value
        self.max_value = max_value
        self.num_values = max_value - min_value + 1
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, self.num_values)
            for _ in range(max_steps)
        ])

    def forward_step(self, hidden: torch.Tensor, step_idx: int) -> torch.Tensor:
        if step_idx < 0 or step_idx >= self.max_steps:
            raise ValueError(f"step_idx={step_idx} outside max_steps={self.max_steps}")
        return self.heads[step_idx](hidden.float())


class NumericStateEmbedding(nn.Module):
    """Trainable numeric state table for directly shaping latent states."""

    def __init__(
        self,
        hidden_size: int,
        min_value: int,
        max_value: int,
        temperature: float = 0.1,
    ):
        super().__init__()
        if max_value < min_value:
            raise ValueError(f"max_value={max_value} must be >= min_value={min_value}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.min_value = min_value
        self.max_value = max_value
        self.num_values = max_value - min_value + 1
        self.temperature = temperature
        self.embedding = nn.Embedding(self.num_values, hidden_size)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=hidden_size ** -0.5)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden_n = torch.nn.functional.normalize(hidden.float(), dim=-1)
        table_n = torch.nn.functional.normalize(self.embedding.weight.float(), dim=-1)
        return torch.nn.functional.linear(hidden_n, table_n) / self.temperature

    def target_embeddings(self, target_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(target_ids)


def _numeric_state_embedding_loss(
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
    numeric_embedding: NumericStateEmbedding,
    mse_weight: float,
) -> tuple[torch.Tensor | None, float, float, int, int]:
    if not mask.any():
        return None, 0.0, 0.0, 0, 0
    flat_hidden = hidden[mask].float()
    flat_targets = target_ids[mask]
    logits = numeric_embedding.logits(flat_hidden)
    ce_loss = torch.nn.functional.cross_entropy(logits, flat_targets)
    target_embeds = numeric_embedding.target_embeddings(flat_targets)
    mse_loss = torch.nn.functional.mse_loss(
        _rms_norm_cpu(flat_hidden),
        _rms_norm_cpu(target_embeds.float()),
    )
    loss = ce_loss + mse_weight * mse_loss
    correct = int((logits.argmax(dim=-1) == flat_targets).sum().item())
    slots = int(flat_targets.numel())
    return loss, float(ce_loss.item()), float(mse_loss.item()), correct, slots


def _build_arithmetic_state_targets(
    cases: list[DistillCase],
    latent_steps: int,
    min_value: int,
    max_value: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.zeros(len(cases), latent_steps, device=device, dtype=torch.long)
    mask = torch.zeros(len(cases), latent_steps, device=device, dtype=torch.bool)
    for case_idx, case in enumerate(cases):
        if case.task != "arithmetic":
            continue
        a = int(case.meta["a"])
        b = int(case.meta["b"])
        c = int(case.meta["c"])
        d = int(case.meta["d"])
        sum_value = a + b
        product_value = sum_value * c
        final_value = product_value - d
        step_values = [sum_value, product_value, final_value, final_value]
        for step_idx in range(latent_steps):
            value = step_values[min(step_idx, len(step_values) - 1)]
            if min_value <= value <= max_value:
                targets[case_idx, step_idx] = value - min_value
                mask[case_idx, step_idx] = True
    return targets, mask


@dataclass
class ArithmeticTransitionTargets:
    value_targets: torch.Tensor
    phase_targets: torch.Tensor
    mask: torch.Tensor


def _build_arithmetic_transition_targets(
    cases: list[DistillCase],
    latent_steps: int,
    min_value: int,
    max_value: int,
    device: torch.device,
    repeat_final: bool,
) -> ArithmeticTransitionTargets:
    value_targets = torch.zeros(len(cases), latent_steps, device=device, dtype=torch.long)
    phase_targets = torch.zeros(len(cases), latent_steps, device=device, dtype=torch.long)
    mask = torch.zeros(len(cases), latent_steps, device=device, dtype=torch.bool)
    for case_idx, case in enumerate(cases):
        if case.task != "arithmetic":
            continue
        a = int(case.meta["a"])
        b = int(case.meta["b"])
        c = int(case.meta["c"])
        d = int(case.meta["d"])
        sum_value = a + b
        product_value = sum_value * c
        final_value = product_value - d
        step_values = [sum_value, product_value, final_value]
        step_phases = [
            ArithmeticStateMachineProjector.PHASE_SUM,
            ArithmeticStateMachineProjector.PHASE_PRODUCT,
            ArithmeticStateMachineProjector.PHASE_FINAL,
        ]
        for step_idx in range(latent_steps):
            if step_idx >= len(step_values) and not repeat_final:
                continue
            value = step_values[min(step_idx, len(step_values) - 1)]
            phase = step_phases[min(step_idx, len(step_phases) - 1)]
            if min_value <= value <= max_value:
                value_targets[case_idx, step_idx] = value - min_value
                phase_targets[case_idx, step_idx] = phase
                mask[case_idx, step_idx] = True
    return ArithmeticTransitionTargets(value_targets=value_targets, phase_targets=phase_targets, mask=mask)


def _safe_div(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def _arithmetic_state_machine_loss_and_metrics(
    hidden: torch.Tensor,
    transition_targets: ArithmeticTransitionTargets,
    projector: nn.Module,
    value_loss_weight: float,
    phase_loss_weight: float,
) -> tuple[torch.Tensor | None, dict]:
    mask = transition_targets.mask
    if not mask.any():
        return None, {
            "loss": 0.0,
            "value_ce": 0.0,
            "phase_ce": 0.0,
            "slots": 0,
            "value_correct": 0,
            "phase_correct": 0,
            "transition_correct": 0,
            "sum_correct": 0,
            "sum_slots": 0,
            "product_correct": 0,
            "product_slots": 0,
            "final_correct": 0,
            "final_slots": 0,
            "sequence_correct": 0,
            "sequence_slots": 0,
        }
    if not hasattr(projector, "state_machine_logits"):
        raise RuntimeError("arithmetic state-machine training requires --projector-kind arithmetic_state_machine")
    value_logits = []
    phase_logits = []
    for step_idx in range(hidden.size(1)):
        step_value_logits, step_phase_logits = projector.state_machine_logits(hidden[:, step_idx], step_idx)
        value_logits.append(step_value_logits)
        phase_logits.append(step_phase_logits)
    value_logits = torch.stack(value_logits, dim=1)
    phase_logits = torch.stack(phase_logits, dim=1)
    flat_value_logits = value_logits[mask]
    flat_phase_logits = phase_logits[mask]
    flat_value_targets = transition_targets.value_targets[mask]
    flat_phase_targets = transition_targets.phase_targets[mask]
    value_ce = torch.nn.functional.cross_entropy(flat_value_logits, flat_value_targets)
    phase_ce = torch.nn.functional.cross_entropy(flat_phase_logits, flat_phase_targets)
    loss = value_loss_weight * value_ce + phase_loss_weight * phase_ce

    value_pred = value_logits.argmax(dim=-1)
    phase_pred = phase_logits.argmax(dim=-1)
    value_ok = (value_pred == transition_targets.value_targets) & mask
    phase_ok = (phase_pred == transition_targets.phase_targets) & mask
    transition_ok = value_ok & phase_ok
    sum_mask = mask & (transition_targets.phase_targets == ArithmeticStateMachineProjector.PHASE_SUM)
    product_mask = mask & (transition_targets.phase_targets == ArithmeticStateMachineProjector.PHASE_PRODUCT)
    final_mask = mask & (transition_targets.phase_targets == ArithmeticStateMachineProjector.PHASE_FINAL)
    sample_has_targets = mask.any(dim=1)
    sequence_ok = ((transition_ok | ~mask).all(dim=1) & sample_has_targets)
    metrics = {
        "loss": float(loss.item()),
        "value_ce": float(value_ce.item()),
        "phase_ce": float(phase_ce.item()),
        "slots": int(mask.sum().item()),
        "value_correct": int(value_ok.sum().item()),
        "phase_correct": int(phase_ok.sum().item()),
        "transition_correct": int(transition_ok.sum().item()),
        "sum_correct": int((transition_ok & sum_mask).sum().item()),
        "sum_slots": int(sum_mask.sum().item()),
        "product_correct": int((transition_ok & product_mask).sum().item()),
        "product_slots": int(product_mask.sum().item()),
        "final_correct": int((transition_ok & final_mask).sum().item()),
        "final_slots": int(final_mask.sum().item()),
        "sequence_correct": int(sequence_ok.sum().item()),
        "sequence_slots": int(sample_has_targets.sum().item()),
    }
    return loss, metrics


@dataclass(frozen=True)
class PureArithmeticTuple:
    a: int
    b: int
    c: int
    d: int


@dataclass
class PureDigitArithmeticTrace:
    sum_digits: torch.Tensor
    product_digits: torch.Tensor
    final_digits: torch.Tensor
    final_sign: torch.Tensor
    add_carry: torch.Tensor
    mul_carry: torch.Tensor
    sub_borrow: torch.Tensor
    overflow_mask: torch.Tensor


def _positive_values_to_lsd_digits(values: torch.Tensor, num_digits: int) -> torch.Tensor:
    digits = []
    x = values.long().clamp_min(0)
    for pos in range(num_digits):
        digits.append((x // (10 ** pos)) % 10)
    return torch.stack(digits, dim=-1)


def _lsd_digits_to_positive_values(digits: torch.Tensor) -> torch.Tensor:
    values = torch.zeros(digits.shape[:-1], device=digits.device, dtype=torch.long)
    for pos in range(digits.size(-1)):
        values = values + digits[..., pos].long() * (10 ** pos)
    return values


def _infer_pure_digit_num_digits(
    a_max: int,
    b_max: int,
    c_max: int,
    d_max: int,
) -> int:
    max_sum = a_max + b_max
    max_product = max_sum * c_max
    max_final_abs = max(max_product, d_max)
    return max(1, len(str(max_final_abs)))


def _pure_digit_items_to_operands(items: list[PureArithmeticTuple], device: torch.device) -> tuple[torch.Tensor, ...]:
    a = torch.tensor([x.a for x in items], device=device, dtype=torch.long)
    b = torch.tensor([x.b for x in items], device=device, dtype=torch.long)
    c = torch.tensor([x.c for x in items], device=device, dtype=torch.long)
    d = torch.tensor([x.d for x in items], device=device, dtype=torch.long)
    return a, b, c, d


def _build_direct_digit_arithmetic_trace(
    items: list[PureArithmeticTuple],
    num_digits: int,
    device: torch.device,
) -> PureDigitArithmeticTrace:
    a, b, c, d = _pure_digit_items_to_operands(items, device)
    sum_value = a + b
    product_value = sum_value * c
    final_value = product_value - d
    final_sign = (final_value < 0).long()
    final_abs = final_value.abs()
    limit = 10 ** num_digits
    overflow_mask = (sum_value >= limit) | (product_value >= limit) | (final_abs >= limit)
    a_digits = _positive_values_to_lsd_digits(a, num_digits)
    b_digits = _positive_values_to_lsd_digits(b, num_digits)
    d_digits = _positive_values_to_lsd_digits(d, num_digits)
    sum_digits = _positive_values_to_lsd_digits(sum_value, num_digits)
    product_digits = _positive_values_to_lsd_digits(product_value, num_digits)
    add_carry = torch.zeros(len(items), num_digits + 1, device=device, dtype=torch.long)
    mul_carry = torch.zeros(len(items), num_digits + 1, device=device, dtype=torch.long)
    sub_borrow = torch.zeros(len(items), num_digits + 1, device=device, dtype=torch.long)
    for pos in range(num_digits):
        add_raw = a_digits[:, pos] + b_digits[:, pos] + add_carry[:, pos]
        add_carry[:, pos + 1] = add_raw // 10
        mul_raw = sum_digits[:, pos] * c + mul_carry[:, pos]
        mul_carry[:, pos + 1] = mul_raw // 10
    subtract_positive = product_value >= d
    lhs_digits = torch.where(subtract_positive.unsqueeze(-1), product_digits, d_digits)
    rhs_digits = torch.where(subtract_positive.unsqueeze(-1), d_digits, product_digits)
    for pos in range(num_digits):
        sub_raw = lhs_digits[:, pos] - rhs_digits[:, pos] - sub_borrow[:, pos]
        sub_borrow[:, pos + 1] = (sub_raw < 0).long()
    return PureDigitArithmeticTrace(
        sum_digits=sum_digits,
        product_digits=product_digits,
        final_digits=_positive_values_to_lsd_digits(final_abs, num_digits),
        final_sign=final_sign,
        add_carry=add_carry,
        mul_carry=mul_carry,
        sub_borrow=sub_borrow,
        overflow_mask=overflow_mask,
    )


def _build_algorithmic_digit_arithmetic_trace(
    items: list[PureArithmeticTuple],
    num_digits: int,
    device: torch.device,
) -> PureDigitArithmeticTrace:
    a, b, c, d = _pure_digit_items_to_operands(items, device)
    a_digits = _positive_values_to_lsd_digits(a, num_digits)
    b_digits = _positive_values_to_lsd_digits(b, num_digits)
    d_digits = _positive_values_to_lsd_digits(d, num_digits)
    batch = len(items)

    add_carry = torch.zeros(batch, num_digits + 1, device=device, dtype=torch.long)
    sum_digit_list = []
    for pos in range(num_digits):
        raw = a_digits[:, pos] + b_digits[:, pos] + add_carry[:, pos]
        sum_digit_list.append(raw % 10)
        add_carry[:, pos + 1] = raw // 10
    sum_digits = torch.stack(sum_digit_list, dim=-1)

    mul_carry = torch.zeros(batch, num_digits + 1, device=device, dtype=torch.long)
    product_digit_list = []
    for pos in range(num_digits):
        raw = sum_digits[:, pos] * c + mul_carry[:, pos]
        product_digit_list.append(raw % 10)
        mul_carry[:, pos + 1] = raw // 10
    product_digits = torch.stack(product_digit_list, dim=-1)
    product_value = _lsd_digits_to_positive_values(product_digits)

    direct_product = (a + b) * c
    subtract_positive = direct_product >= d
    lhs_digits = torch.where(subtract_positive.unsqueeze(-1), product_digits, d_digits)
    rhs_digits = torch.where(subtract_positive.unsqueeze(-1), d_digits, product_digits)
    sub_borrow = torch.zeros(batch, num_digits + 1, device=device, dtype=torch.long)
    final_digit_list = []
    for pos in range(num_digits):
        raw = lhs_digits[:, pos] - rhs_digits[:, pos] - sub_borrow[:, pos]
        borrow = raw < 0
        final_digit_list.append(torch.where(borrow, raw + 10, raw))
        sub_borrow[:, pos + 1] = borrow.long()
    final_digits = torch.stack(final_digit_list, dim=-1)
    final_sign = (~subtract_positive).long()

    limit = 10 ** num_digits
    final_abs = (direct_product - d).abs()
    overflow_mask = (
        (add_carry[:, -1] > 0)
        | (mul_carry[:, -1] > 0)
        | (sub_borrow[:, -1] > 0)
        | ((a + b) >= limit)
        | (direct_product >= limit)
        | (final_abs >= limit)
        | (product_value != direct_product.clamp_max(limit - 1))
    )
    return PureDigitArithmeticTrace(
        sum_digits=sum_digits,
        product_digits=product_digits,
        final_digits=final_digits,
        final_sign=final_sign,
        add_carry=add_carry,
        mul_carry=mul_carry,
        sub_borrow=sub_borrow,
        overflow_mask=overflow_mask,
    )


def _pure_digit_trace_metrics(
    pred: PureDigitArithmeticTrace,
    target: PureDigitArithmeticTrace,
) -> dict:
    valid_mask = ~(pred.overflow_mask | target.overflow_mask)
    valid = int(valid_mask.sum().item())
    if valid == 0:
        return {
            "valid_cases": 0,
            "overflow_cases": int((pred.overflow_mask | target.overflow_mask).sum().item()),
            "sum_value_accuracy": 0.0,
            "product_value_accuracy": 0.0,
            "final_value_accuracy": 0.0,
            "sequence_value_accuracy": 0.0,
            "sum_digit_accuracy": 0.0,
            "product_digit_accuracy": 0.0,
            "final_digit_accuracy": 0.0,
            "add_carry_accuracy": 0.0,
            "mul_carry_accuracy": 0.0,
            "sub_borrow_accuracy": 0.0,
            "final_sign_accuracy": 0.0,
            "sum_mae": 0.0,
            "product_mae": 0.0,
            "final_mae": 0.0,
        }
    sum_digit_ok = pred.sum_digits == target.sum_digits
    product_digit_ok = pred.product_digits == target.product_digits
    final_digit_ok = pred.final_digits == target.final_digits
    sum_value_ok = sum_digit_ok.all(dim=-1) & valid_mask
    product_value_ok = product_digit_ok.all(dim=-1) & valid_mask
    final_value_ok = final_digit_ok.all(dim=-1) & (pred.final_sign == target.final_sign) & valid_mask
    sequence_ok = sum_value_ok & product_value_ok & final_value_ok
    pred_sum = _lsd_digits_to_positive_values(pred.sum_digits)
    target_sum = _lsd_digits_to_positive_values(target.sum_digits)
    pred_product = _lsd_digits_to_positive_values(pred.product_digits)
    target_product = _lsd_digits_to_positive_values(target.product_digits)
    pred_final_abs = _lsd_digits_to_positive_values(pred.final_digits)
    target_final_abs = _lsd_digits_to_positive_values(target.final_digits)
    pred_final = torch.where(pred.final_sign.bool(), -pred_final_abs, pred_final_abs)
    target_final = torch.where(target.final_sign.bool(), -target_final_abs, target_final_abs)
    return {
        "valid_cases": valid,
        "overflow_cases": int((pred.overflow_mask | target.overflow_mask).sum().item()),
        "sum_value_accuracy": _safe_div(int(sum_value_ok.sum().item()), valid),
        "product_value_accuracy": _safe_div(int(product_value_ok.sum().item()), valid),
        "final_value_accuracy": _safe_div(int(final_value_ok.sum().item()), valid),
        "sequence_value_accuracy": _safe_div(int(sequence_ok.sum().item()), valid),
        "sum_digit_accuracy": _safe_div(int((sum_digit_ok & valid_mask.unsqueeze(-1)).sum().item()), valid * pred.sum_digits.size(-1)),
        "product_digit_accuracy": _safe_div(int((product_digit_ok & valid_mask.unsqueeze(-1)).sum().item()), valid * pred.product_digits.size(-1)),
        "final_digit_accuracy": _safe_div(int((final_digit_ok & valid_mask.unsqueeze(-1)).sum().item()), valid * pred.final_digits.size(-1)),
        "add_carry_accuracy": _safe_div(int(((pred.add_carry == target.add_carry) & valid_mask.unsqueeze(-1)).sum().item()), valid * pred.add_carry.size(-1)),
        "mul_carry_accuracy": _safe_div(int(((pred.mul_carry == target.mul_carry) & valid_mask.unsqueeze(-1)).sum().item()), valid * pred.mul_carry.size(-1)),
        "sub_borrow_accuracy": _safe_div(int(((pred.sub_borrow == target.sub_borrow) & valid_mask.unsqueeze(-1)).sum().item()), valid * pred.sub_borrow.size(-1)),
        "final_sign_accuracy": _safe_div(int(((pred.final_sign == target.final_sign) & valid_mask).sum().item()), valid),
        "sum_mae": float((pred_sum[valid_mask] - target_sum[valid_mask]).abs().float().mean().item()),
        "product_mae": float((pred_product[valid_mask] - target_product[valid_mask]).abs().float().mean().item()),
        "final_mae": float((pred_final[valid_mask] - target_final[valid_mask]).abs().float().mean().item()),
    }


@dataclass(frozen=True)
class DigitOperatorExample:
    op: int
    x: int
    y: int
    carry_in: int
    digit_out: int
    carry_out: int


class TrainableDigitLocalOperator(nn.Module):
    """Small local digit operator: x, y, carry_in -> digit_out, carry_out."""

    def __init__(self, y_classes: int, carry_classes: int, hidden_size: int, depth: int):
        super().__init__()
        if y_classes < 1 or carry_classes < 1:
            raise ValueError("y_classes and carry_classes must be positive")
        self.y_classes = y_classes
        self.carry_classes = carry_classes
        in_dim = 10 + y_classes + carry_classes + 3
        layers: list[nn.Module] = []
        cur = in_dim
        for _ in range(max(1, depth)):
            layers.append(nn.Linear(cur, hidden_size))
            layers.append(nn.GELU())
            cur = hidden_size
        self.net = nn.Sequential(*layers)
        self.digit_head = nn.Linear(cur, 10)
        self.carry_head = nn.Linear(cur, carry_classes)

    def _features(self, x: torch.Tensor, y: torch.Tensor, carry_in: torch.Tensor) -> torch.Tensor:
        x = x.long().clamp(0, 9)
        y = y.long().clamp(0, self.y_classes - 1)
        carry_in = carry_in.long().clamp(0, self.carry_classes - 1)
        x_oh = torch.nn.functional.one_hot(x, num_classes=10).float()
        y_oh = torch.nn.functional.one_hot(y, num_classes=self.y_classes).float()
        carry_oh = torch.nn.functional.one_hot(carry_in, num_classes=self.carry_classes).float()
        scalar = torch.stack([
            x.float() / 9.0,
            y.float() / max(1.0, float(self.y_classes - 1)),
            carry_in.float() / max(1.0, float(self.carry_classes - 1)),
        ], dim=-1)
        return torch.cat([x_oh, y_oh, carry_oh, scalar], dim=-1)

    def forward(self, x: torch.Tensor, y: torch.Tensor, carry_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(self._features(x, y, carry_in))
        return self.digit_head(h), self.carry_head(h)


class TrainableDigitWiseOperator(nn.Module):
    def __init__(self, c_max: int, hidden_size: int, depth: int):
        super().__init__()
        self.c_max = c_max
        self.mul_carry_classes = max(2, c_max + 1)
        self.add_op = TrainableDigitLocalOperator(y_classes=10, carry_classes=2, hidden_size=hidden_size, depth=depth)
        self.mul_op = TrainableDigitLocalOperator(
            y_classes=max(10, c_max + 1),
            carry_classes=self.mul_carry_classes,
            hidden_size=hidden_size,
            depth=depth,
        )
        self.sub_op = TrainableDigitLocalOperator(y_classes=10, carry_classes=2, hidden_size=hidden_size, depth=depth)

    def forward_op(self, op: int, x: torch.Tensor, y: torch.Tensor, carry_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if op == 0:
            return self.add_op(x, y, carry_in)
        if op == 1:
            return self.mul_op(x, y, carry_in)
        if op == 2:
            return self.sub_op(x, y, carry_in)
        raise ValueError(f"unknown digit op id: {op}")


class PrimitiveLookupDigitWiseOperator(nn.Module):
    """Exact local digit primitive exposed through the same logits API as the trainable operator."""

    def __init__(self, c_min: int, c_max: int, logit_scale: float = 20.0):
        super().__init__()
        if c_max < c_min:
            raise ValueError("c_max must be >= c_min")
        self.c_min = c_min
        self.c_max = c_max
        self.mul_carry_classes = max(2, c_max + 1)
        self.mul_y_classes = max(10, c_max + 1)
        self.logit_scale = float(logit_scale)

        add_digit = torch.zeros(10, 10, 2, dtype=torch.long)
        add_carry = torch.zeros(10, 10, 2, dtype=torch.long)
        for x in range(10):
            for y in range(10):
                for carry in range(2):
                    raw = x + y + carry
                    add_digit[x, y, carry] = raw % 10
                    add_carry[x, y, carry] = raw // 10
        self.register_buffer("add_digit_table", add_digit, persistent=False)
        self.register_buffer("add_carry_table", add_carry, persistent=False)

        mul_digit = torch.zeros(10, self.mul_y_classes, self.mul_carry_classes, dtype=torch.long)
        mul_carry = torch.zeros(10, self.mul_y_classes, self.mul_carry_classes, dtype=torch.long)
        for x in range(10):
            for y in range(self.mul_y_classes):
                for carry in range(self.mul_carry_classes):
                    raw = x * y + carry
                    mul_digit[x, y, carry] = raw % 10
                    mul_carry[x, y, carry] = min(raw // 10, self.mul_carry_classes - 1)
        self.register_buffer("mul_digit_table", mul_digit, persistent=False)
        self.register_buffer("mul_carry_table", mul_carry, persistent=False)

        sub_digit = torch.zeros(10, 10, 2, dtype=torch.long)
        sub_borrow = torch.zeros(10, 10, 2, dtype=torch.long)
        for x in range(10):
            for y in range(10):
                for borrow in range(2):
                    raw = x - y - borrow
                    sub_digit[x, y, borrow] = raw + 10 if raw < 0 else raw
                    sub_borrow[x, y, borrow] = int(raw < 0)
        self.register_buffer("sub_digit_table", sub_digit, persistent=False)
        self.register_buffer("sub_borrow_table", sub_borrow, persistent=False)

    def _table_logits(self, ids: torch.Tensor, num_classes: int) -> torch.Tensor:
        logits = torch.zeros(*ids.shape, num_classes, device=ids.device, dtype=torch.float32)
        return logits.scatter_(-1, ids.long().unsqueeze(-1), self.logit_scale)

    def forward_op(self, op: int, x: torch.Tensor, y: torch.Tensor, carry_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.long().clamp(0, 9)
        if op == 0:
            y = y.long().clamp(0, 9)
            carry_in = carry_in.long().clamp(0, 1)
            digit = self.add_digit_table[x, y, carry_in]
            carry = self.add_carry_table[x, y, carry_in]
            return self._table_logits(digit, 10), self._table_logits(carry, 2)
        if op == 1:
            y = y.long().clamp(0, self.mul_y_classes - 1)
            carry_in = carry_in.long().clamp(0, self.mul_carry_classes - 1)
            digit = self.mul_digit_table[x, y, carry_in]
            carry = self.mul_carry_table[x, y, carry_in]
            return self._table_logits(digit, 10), self._table_logits(carry, self.mul_carry_classes)
        if op == 2:
            y = y.long().clamp(0, 9)
            carry_in = carry_in.long().clamp(0, 1)
            digit = self.sub_digit_table[x, y, carry_in]
            borrow = self.sub_borrow_table[x, y, carry_in]
            return self._table_logits(digit, 10), self._table_logits(borrow, 2)
        raise ValueError(f"unknown digit op id: {op}")


class OracleStructuredDecodeBridge(nn.Module):
    """Map oracle structured arithmetic state features into latent LLM input embeddings."""

    def __init__(self, feature_dim: int, hidden_size: int, latent_steps: int, bridge_hidden_size: int, depth: int):
        super().__init__()
        if feature_dim < 1 or hidden_size < 1 or latent_steps < 1:
            raise ValueError("feature_dim, hidden_size, and latent_steps must be positive")
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.latent_steps = latent_steps
        layers: list[nn.Module] = []
        cur = feature_dim
        for _ in range(max(1, depth)):
            layers.append(nn.Linear(cur, bridge_hidden_size))
            layers.append(nn.GELU())
            cur = bridge_hidden_size
        layers.append(nn.Linear(cur, latent_steps * hidden_size))
        self.net = nn.Sequential(*layers)
        self.step_embed = nn.Parameter(torch.zeros(latent_steps, hidden_size))
        nn.init.normal_(self.step_embed, mean=0.0, std=hidden_size ** -0.5)

    def forward(self, features: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        latents = self.net(features.float()).view(features.size(0), self.latent_steps, self.hidden_size)
        latents = latents + self.step_embed.to(device=features.device, dtype=latents.dtype).unsqueeze(0)
        if dtype is not None:
            latents = latents.to(dtype=dtype)
        return latents


def _stable_digit_combo_is_heldout(example: DigitOperatorExample, heldout_frac: float, seed: int) -> bool:
    key = f"{example.op}:{example.x}:{example.y}:{example.carry_in}:{seed}".encode()
    bucket = int(hashlib.sha1(key).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < heldout_frac


def enumerate_digit_operator_examples(c_min: int, c_max: int) -> list[DigitOperatorExample]:
    examples: list[DigitOperatorExample] = []
    for x in range(10):
        for y in range(10):
            for carry in range(2):
                raw = x + y + carry
                examples.append(DigitOperatorExample(0, x, y, carry, raw % 10, raw // 10))
    for x in range(10):
        for c in range(c_min, c_max + 1):
            for carry in range(max(1, c_max)):
                raw = x * c + carry
                examples.append(DigitOperatorExample(1, x, c, carry, raw % 10, raw // 10))
    for x in range(10):
        for y in range(10):
            for borrow in range(2):
                raw = x - y - borrow
                examples.append(DigitOperatorExample(2, x, y, borrow, raw + 10 if raw < 0 else raw, int(raw < 0)))
    return examples


def split_digit_operator_examples(
    examples: list[DigitOperatorExample],
    heldout_frac: float,
    seed: int,
) -> tuple[list[DigitOperatorExample], list[DigitOperatorExample]]:
    train = []
    heldout = []
    for example in examples:
        if _stable_digit_combo_is_heldout(example, heldout_frac, seed):
            heldout.append(example)
        else:
            train.append(example)
    if not train or not heldout:
        raise ValueError("digit operator split produced empty train or heldout set")
    return train, heldout


def _digit_operator_batch_tensors(
    examples: list[DigitOperatorExample],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([x.op for x in examples], device=device, dtype=torch.long),
        torch.tensor([x.x for x in examples], device=device, dtype=torch.long),
        torch.tensor([x.y for x in examples], device=device, dtype=torch.long),
        torch.tensor([x.carry_in for x in examples], device=device, dtype=torch.long),
        torch.tensor([x.digit_out for x in examples], device=device, dtype=torch.long),
        torch.tensor([x.carry_out for x in examples], device=device, dtype=torch.long),
    )


def _digit_operator_loss_and_metrics(
    model: TrainableDigitWiseOperator,
    examples: list[DigitOperatorExample],
    device: torch.device,
    carry_weight: float,
) -> tuple[torch.Tensor | None, dict]:
    if not examples:
        return None, {"examples": 0, "digit_accuracy": 0.0, "carry_accuracy": 0.0, "transition_accuracy": 0.0}
    ops, xs, ys, carries, digit_targets, carry_targets = _digit_operator_batch_tensors(examples, device)
    losses = []
    digit_correct = 0
    carry_correct = 0
    transition_correct = 0
    total = 0
    per_op = {}
    for op_id, name in ((0, "add"), (1, "mul"), (2, "sub")):
        mask = ops == op_id
        if not mask.any():
            continue
        digit_logits, carry_logits = model.forward_op(op_id, xs[mask], ys[mask], carries[mask])
        digit_loss = torch.nn.functional.cross_entropy(digit_logits, digit_targets[mask])
        carry_loss = torch.nn.functional.cross_entropy(carry_logits, carry_targets[mask])
        losses.append(digit_loss + carry_weight * carry_loss)
        digit_pred = digit_logits.argmax(dim=-1)
        carry_pred = carry_logits.argmax(dim=-1)
        digit_ok = digit_pred == digit_targets[mask]
        carry_ok = carry_pred == carry_targets[mask]
        transition_ok = digit_ok & carry_ok
        count = int(mask.sum().item())
        total += count
        digit_correct += int(digit_ok.sum().item())
        carry_correct += int(carry_ok.sum().item())
        transition_correct += int(transition_ok.sum().item())
        per_op[f"{name}_digit_accuracy"] = _safe_div(int(digit_ok.sum().item()), count)
        per_op[f"{name}_carry_accuracy"] = _safe_div(int(carry_ok.sum().item()), count)
        per_op[f"{name}_transition_accuracy"] = _safe_div(int(transition_ok.sum().item()), count)
    loss = torch.stack(losses).mean() if losses else None
    metrics = {
        "examples": total,
        "digit_accuracy": _safe_div(digit_correct, total),
        "carry_accuracy": _safe_div(carry_correct, total),
        "transition_accuracy": _safe_div(transition_correct, total),
    }
    metrics.update(per_op)
    return loss, metrics


@torch.no_grad()
def _build_trainable_digit_arithmetic_trace(
    model: TrainableDigitWiseOperator,
    items: list[PureArithmeticTuple],
    num_digits: int,
    device: torch.device,
) -> PureDigitArithmeticTrace:
    a, b, c, d = _pure_digit_items_to_operands(items, device)
    a_digits = _positive_values_to_lsd_digits(a, num_digits)
    b_digits = _positive_values_to_lsd_digits(b, num_digits)
    d_digits = _positive_values_to_lsd_digits(d, num_digits)
    batch = len(items)
    add_carry = torch.zeros(batch, num_digits + 1, device=device, dtype=torch.long)
    sum_digit_list = []
    for pos in range(num_digits):
        digit_logits, carry_logits = model.forward_op(0, a_digits[:, pos], b_digits[:, pos], add_carry[:, pos])
        sum_digit_list.append(digit_logits.argmax(dim=-1))
        add_carry[:, pos + 1] = carry_logits.argmax(dim=-1)
    sum_digits = torch.stack(sum_digit_list, dim=-1)
    mul_carry = torch.zeros(batch, num_digits + 1, device=device, dtype=torch.long)
    product_digit_list = []
    for pos in range(num_digits):
        digit_logits, carry_logits = model.forward_op(1, sum_digits[:, pos], c, mul_carry[:, pos])
        product_digit_list.append(digit_logits.argmax(dim=-1))
        mul_carry[:, pos + 1] = carry_logits.argmax(dim=-1).clamp(0, model.mul_carry_classes - 1)
    product_digits = torch.stack(product_digit_list, dim=-1)
    product_value = _lsd_digits_to_positive_values(product_digits)
    subtract_positive = product_value >= d
    lhs_digits = torch.where(subtract_positive.unsqueeze(-1), product_digits, d_digits)
    rhs_digits = torch.where(subtract_positive.unsqueeze(-1), d_digits, product_digits)
    sub_borrow = torch.zeros(batch, num_digits + 1, device=device, dtype=torch.long)
    final_digit_list = []
    for pos in range(num_digits):
        digit_logits, carry_logits = model.forward_op(2, lhs_digits[:, pos], rhs_digits[:, pos], sub_borrow[:, pos])
        final_digit_list.append(digit_logits.argmax(dim=-1))
        sub_borrow[:, pos + 1] = carry_logits.argmax(dim=-1)
    final_digits = torch.stack(final_digit_list, dim=-1)
    overflow_mask = (add_carry[:, -1] > 0) | (mul_carry[:, -1] > 0) | (sub_borrow[:, -1] > 0)
    return PureDigitArithmeticTrace(
        sum_digits=sum_digits,
        product_digits=product_digits,
        final_digits=final_digits,
        final_sign=(~subtract_positive).long(),
        add_carry=add_carry,
        mul_carry=mul_carry,
        sub_borrow=sub_borrow,
        overflow_mask=overflow_mask,
    )


class PureArithmeticTupleEncoder(nn.Module):
    """Encode explicit arithmetic operands into an initial latent state."""

    def __init__(self, hidden_size: int, min_value: int, max_value: int):
        super().__init__()
        if max_value < min_value:
            raise ValueError(f"max_value={max_value} must be >= min_value={min_value}")
        self.min_value = min_value
        self.max_value = max_value
        self.num_values = max_value - min_value + 1
        self.value_embed = nn.Embedding(self.num_values, hidden_size)
        self.slot_embed = nn.Parameter(torch.zeros(4, hidden_size))
        self.net = nn.Sequential(
            nn.Linear(4 * hidden_size, 2 * hidden_size),
            nn.GELU(),
            nn.Linear(2 * hidden_size, hidden_size),
        )
        nn.init.normal_(self.value_embed.weight, mean=0.0, std=hidden_size ** -0.5)

    def forward(self, operands: torch.Tensor) -> torch.Tensor:
        if operands.size(-1) != 4:
            raise ValueError(f"operands must have shape [B, 4], got {tuple(operands.shape)}")
        ids = operands.long() - self.min_value
        if bool((ids < 0).any().item()) or bool((ids >= self.num_values).any().item()):
            raise ValueError("operand outside state-min/state-max range")
        embeds = self.value_embed(ids) + self.slot_embed.to(device=operands.device, dtype=self.value_embed.weight.dtype)
        return self.net(embeds.reshape(operands.size(0), -1))


def _num_decimal_digits(min_value: int, max_value: int) -> int:
    return max(1, len(str(max_value - min_value)))


def _normalise_numeric_values(values: torch.Tensor, min_value: int, max_value: int) -> torch.Tensor:
    scale = max(1.0, float(max_value - min_value))
    return 2.0 * ((values.float() - float(min_value)) / scale) - 1.0


def _denormalise_numeric_values(values: torch.Tensor, min_value: int, max_value: int) -> torch.Tensor:
    scale = max(1.0, float(max_value - min_value))
    return ((values.float() + 1.0) * 0.5 * scale) + float(min_value)


def _value_ids_to_digits(value_ids: torch.Tensor, num_digits: int) -> torch.Tensor:
    digits = []
    x = value_ids.long().clamp_min(0)
    for pos in range(num_digits):
        divisor = 10 ** (num_digits - pos - 1)
        digits.append((x // divisor) % 10)
    return torch.stack(digits, dim=-1)


def _digits_to_value_ids(digits: torch.Tensor) -> torch.Tensor:
    value_ids = torch.zeros(digits.shape[:-1], device=digits.device, dtype=torch.long)
    num_digits = digits.size(-1)
    for pos in range(num_digits):
        value_ids = value_ids + digits[..., pos].long() * (10 ** (num_digits - pos - 1))
    return value_ids


class PureStructuredArithmeticTupleEncoder(nn.Module):
    """Encode operands with normalized scalar + decimal digit representation."""

    def __init__(self, hidden_size: int, min_value: int, max_value: int):
        super().__init__()
        if max_value < min_value:
            raise ValueError(f"max_value={max_value} must be >= min_value={min_value}")
        self.min_value = min_value
        self.max_value = max_value
        self.num_digits = _num_decimal_digits(min_value, max_value)
        self.scalar_proj = nn.Linear(1, hidden_size)
        self.digit_proj = nn.Linear(self.num_digits * 10, hidden_size)
        self.slot_embed = nn.Parameter(torch.zeros(4, hidden_size))
        self.net = nn.Sequential(
            nn.Linear(4 * hidden_size, 2 * hidden_size),
            nn.GELU(),
            nn.Linear(2 * hidden_size, hidden_size),
        )

    def forward(self, operands: torch.Tensor) -> torch.Tensor:
        if operands.size(-1) != 4:
            raise ValueError(f"operands must have shape [B, 4], got {tuple(operands.shape)}")
        value_ids = operands.long() - self.min_value
        max_id = self.max_value - self.min_value
        if bool((value_ids < 0).any().item()) or bool((value_ids > max_id).any().item()):
            raise ValueError("operand outside state-min/state-max range")
        scalars = _normalise_numeric_values(operands.float(), self.min_value, self.max_value).unsqueeze(-1)
        digits = _value_ids_to_digits(value_ids, self.num_digits)
        digit_onehot = torch.nn.functional.one_hot(digits, num_classes=10).float().reshape(operands.size(0), 4, -1)
        embeds = self.scalar_proj(scalars) + self.digit_proj(digit_onehot)
        embeds = embeds + self.slot_embed.to(device=operands.device, dtype=embeds.dtype)
        return self.net(embeds.reshape(operands.size(0), -1))


class PureLatentArithmeticTransitionModel(nn.Module):
    """Pure latent arithmetic state machine without HF/LLM decode."""

    def __init__(
        self,
        hidden_size: int,
        latent_steps: int,
        min_value: int,
        max_value: int,
        numeric_representation: str = "id_embedding",
    ):
        super().__init__()
        if latent_steps < 3:
            raise ValueError(f"pure latent arithmetic requires latent_steps >= 3, got {latent_steps}")
        if numeric_representation not in ("id_embedding", "structured"):
            raise ValueError(f"unknown pure numeric representation: {numeric_representation}")
        self.latent_steps = latent_steps
        self.min_value = min_value
        self.max_value = max_value
        self.num_digits = _num_decimal_digits(min_value, max_value)
        self.numeric_representation = numeric_representation
        if numeric_representation == "structured":
            self.encoder = PureStructuredArithmeticTupleEncoder(hidden_size, min_value=min_value, max_value=max_value)
        else:
            self.encoder = PureArithmeticTupleEncoder(hidden_size, min_value=min_value, max_value=max_value)
        self.projector = ArithmeticStateMachineProjector(
            hidden_size,
            max_steps=latent_steps,
            min_value=min_value,
            max_value=max_value,
        )
        self.regression_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1)
            for _ in range(latent_steps)
        ])
        self.digit_heads = nn.ModuleList([
            nn.Linear(hidden_size, self.num_digits * 10)
            for _ in range(latent_steps)
        ])

    def forward(self, operands: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(operands).float()
        latents = []
        for step_idx in range(self.latent_steps):
            hidden = self.projector.project_step(hidden, step_idx)
            latents.append(hidden)
        return torch.stack(latents, dim=1)

    def structured_numeric_logits(self, hidden: torch.Tensor, step_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        regression = self.regression_heads[step_idx](hidden.float()).squeeze(-1)
        digit_logits = self.digit_heads[step_idx](hidden.float()).view(hidden.size(0), self.num_digits, 10)
        return regression, digit_logits


def _pure_arithmetic_tuple_values(item: PureArithmeticTuple) -> tuple[int, int, int]:
    sum_value = item.a + item.b
    product_value = sum_value * item.c
    final_value = product_value - item.d
    return sum_value, product_value, final_value


def _build_pure_arithmetic_transition_targets(
    items: list[PureArithmeticTuple],
    latent_steps: int,
    min_value: int,
    max_value: int,
    device: torch.device,
) -> ArithmeticTransitionTargets:
    value_targets = torch.zeros(len(items), latent_steps, device=device, dtype=torch.long)
    phase_targets = torch.zeros(len(items), latent_steps, device=device, dtype=torch.long)
    mask = torch.zeros(len(items), latent_steps, device=device, dtype=torch.bool)
    for item_idx, item in enumerate(items):
        sum_value, product_value, final_value = _pure_arithmetic_tuple_values(item)
        step_values = [sum_value, product_value, final_value]
        step_phases = [
            ArithmeticStateMachineProjector.PHASE_SUM,
            ArithmeticStateMachineProjector.PHASE_PRODUCT,
            ArithmeticStateMachineProjector.PHASE_FINAL,
        ]
        for step_idx in range(min(latent_steps, 3)):
            value = step_values[step_idx]
            if min_value <= value <= max_value:
                value_targets[item_idx, step_idx] = value - min_value
                phase_targets[item_idx, step_idx] = step_phases[step_idx]
                mask[item_idx, step_idx] = True
    return ArithmeticTransitionTargets(value_targets=value_targets, phase_targets=phase_targets, mask=mask)


def _pure_arithmetic_operands_tensor(items: list[PureArithmeticTuple], device: torch.device) -> torch.Tensor:
    return torch.tensor([[x.a, x.b, x.c, x.d] for x in items], device=device, dtype=torch.long)


def build_pure_arithmetic_tuples(
    total_cases: int,
    seed: int,
    a_min: int,
    a_max: int,
    b_min: int,
    b_max: int,
    c_min: int,
    c_max: int,
    d_min: int,
    d_max: int,
) -> list[PureArithmeticTuple]:
    if min(a_max - a_min, b_max - b_min, c_max - c_min, d_max - d_min) < 0:
        raise ValueError("invalid pure arithmetic operand range")
    rng = random.Random(seed)
    items: list[PureArithmeticTuple] = []
    seen = set()
    max_unique = (a_max - a_min + 1) * (b_max - b_min + 1) * (c_max - c_min + 1) * (d_max - d_min + 1)
    if total_cases > max_unique:
        raise ValueError(f"requested {total_cases} cases but only {max_unique} unique tuples are available")
    while len(items) < total_cases:
        item = PureArithmeticTuple(
            a=rng.randint(a_min, a_max),
            b=rng.randint(b_min, b_max),
            c=rng.randint(c_min, c_max),
            d=rng.randint(d_min, d_max),
        )
        key = (item.a, item.b, item.c, item.d)
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
    return items


def _pure_value_mae(
    hidden: torch.Tensor,
    targets: ArithmeticTransitionTargets,
    projector: ArithmeticStateMachineProjector,
    min_value: int,
) -> float:
    if not targets.mask.any():
        return 0.0
    preds = []
    for step_idx in range(hidden.size(1)):
        value_logits, _phase_logits = projector.state_machine_logits(hidden[:, step_idx], step_idx)
        preds.append(value_logits.argmax(dim=-1))
    pred_ids = torch.stack(preds, dim=1)
    pred_values = pred_ids[targets.mask].float() + float(min_value)
    target_values = targets.value_targets[targets.mask].float() + float(min_value)
    return float((pred_values - target_values).abs().mean().item())


def _pure_structured_numeric_loss_and_metrics(
    hidden: torch.Tensor,
    targets: ArithmeticTransitionTargets,
    model: PureLatentArithmeticTransitionModel,
    regression_weight: float,
    digit_weight: float,
) -> tuple[torch.Tensor | None, dict]:
    mask = targets.mask
    if not mask.any():
        return None, {
            "loss": 0.0,
            "regression_loss": 0.0,
            "digit_ce": 0.0,
            "slots": 0,
            "digit_correct": 0,
            "digit_slots": 0,
            "digit_value_correct": 0,
            "regression_rounded_correct": 0,
            "regression_mae_sum": 0.0,
            "sequence_correct": 0,
            "sequence_slots": 0,
        }
    regressions = []
    digit_logits = []
    for step_idx in range(hidden.size(1)):
        step_regression, step_digit_logits = model.structured_numeric_logits(hidden[:, step_idx], step_idx)
        regressions.append(step_regression)
        digit_logits.append(step_digit_logits)
    regressions = torch.stack(regressions, dim=1)
    digit_logits = torch.stack(digit_logits, dim=1)
    target_values = targets.value_targets.float() + float(model.min_value)
    regression_targets = _normalise_numeric_values(target_values, model.min_value, model.max_value)
    regression_loss = torch.nn.functional.smooth_l1_loss(regressions[mask], regression_targets[mask])
    target_digits = _value_ids_to_digits(targets.value_targets, model.num_digits)
    flat_digit_logits = digit_logits[mask].reshape(-1, 10)
    flat_digit_targets = target_digits[mask].reshape(-1)
    digit_ce = torch.nn.functional.cross_entropy(flat_digit_logits, flat_digit_targets)
    loss = regression_weight * regression_loss + digit_weight * digit_ce

    digit_pred = digit_logits.argmax(dim=-1)
    digit_ok = digit_pred == target_digits
    digit_value_ids = _digits_to_value_ids(digit_pred).clamp(0, model.max_value - model.min_value)
    digit_value_ok = (digit_value_ids == targets.value_targets) & mask
    regression_values = _denormalise_numeric_values(regressions, model.min_value, model.max_value)
    regression_rounded = regression_values.round().long().clamp(model.min_value, model.max_value)
    regression_rounded_ok = (regression_rounded == (targets.value_targets + model.min_value)) & mask
    regression_mae = (regression_values[mask] - target_values[mask]).abs()
    sample_has_targets = mask.any(dim=1)
    sequence_ok = ((digit_value_ok | ~mask).all(dim=1) & sample_has_targets)
    metrics = {
        "loss": float(loss.item()),
        "regression_loss": float(regression_loss.item()),
        "digit_ce": float(digit_ce.item()),
        "slots": int(mask.sum().item()),
        "digit_correct": int((digit_ok & mask.unsqueeze(-1)).sum().item()),
        "digit_slots": int(mask.sum().item()) * model.num_digits,
        "digit_value_correct": int(digit_value_ok.sum().item()),
        "regression_rounded_correct": int(regression_rounded_ok.sum().item()),
        "regression_mae_sum": float(regression_mae.sum().item()),
        "sequence_correct": int(sequence_ok.sum().item()),
        "sequence_slots": int(sample_has_targets.sum().item()),
    }
    return loss, metrics


def _empty_structured_totals() -> dict:
    return {
        "loss": 0.0,
        "regression_loss": 0.0,
        "digit_ce": 0.0,
        "slots": 0,
        "digit_correct": 0,
        "digit_slots": 0,
        "digit_value_correct": 0,
        "regression_rounded_correct": 0,
        "regression_mae_sum": 0.0,
        "sequence_correct": 0,
        "sequence_slots": 0,
    }


def _accumulate_structured_totals(totals: dict, metrics: dict) -> None:
    slots = int(metrics["slots"])
    totals["loss"] += metrics["loss"] * slots
    totals["regression_loss"] += metrics["regression_loss"] * slots
    totals["digit_ce"] += metrics["digit_ce"] * slots
    totals["regression_mae_sum"] += metrics["regression_mae_sum"]
    for key in (
        "slots",
        "digit_correct",
        "digit_slots",
        "digit_value_correct",
        "regression_rounded_correct",
        "sequence_correct",
        "sequence_slots",
    ):
        totals[key] += int(metrics[key])


def _structured_result_from_totals(totals: dict) -> dict:
    return {
        "structured_loss": totals["loss"] / max(1, totals["slots"]),
        "regression_loss": totals["regression_loss"] / max(1, totals["slots"]),
        "digit_ce": totals["digit_ce"] / max(1, totals["slots"]),
        "digit_accuracy": _safe_div(totals["digit_correct"], totals["digit_slots"]),
        "digit_value_accuracy": _safe_div(totals["digit_value_correct"], totals["slots"]),
        "regression_rounded_accuracy": _safe_div(totals["regression_rounded_correct"], totals["slots"]),
        "regression_mae": totals["regression_mae_sum"] / max(1, totals["slots"]),
        "digit_sequence_accuracy": _safe_div(totals["sequence_correct"], totals["sequence_slots"]),
    }


def evaluate_pure_latent_arithmetic_transition(
    model: PureLatentArithmeticTransitionModel,
    items: list[PureArithmeticTuple],
    batch_size: int,
    device: torch.device,
    state_min_value: int,
    state_max_value: int,
    value_weight: float,
    phase_weight: float,
    regression_weight: float,
    digit_weight: float,
) -> dict:
    model.eval()
    totals = {
        "loss": 0.0,
        "value_ce": 0.0,
        "phase_ce": 0.0,
        "slots": 0,
        "value_correct": 0,
        "phase_correct": 0,
        "transition_correct": 0,
        "sum_correct": 0,
        "sum_slots": 0,
        "product_correct": 0,
        "product_slots": 0,
        "final_correct": 0,
        "final_slots": 0,
        "sequence_correct": 0,
        "sequence_slots": 0,
        "mae_sum": 0.0,
        "mae_slots": 0,
    }
    structured_totals = _empty_structured_totals()
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            operands = _pure_arithmetic_operands_tensor(batch_items, device)
            hidden = model(operands)
            transition_targets = _build_pure_arithmetic_transition_targets(
                batch_items,
                model.latent_steps,
                state_min_value,
                state_max_value,
                device,
            )
            _loss, metrics = _arithmetic_state_machine_loss_and_metrics(
                hidden,
                transition_targets,
                model.projector,
                value_weight,
                phase_weight,
            )
            slots = int(metrics["slots"])
            totals["loss"] += metrics["loss"] * slots
            totals["value_ce"] += metrics["value_ce"] * slots
            totals["phase_ce"] += metrics["phase_ce"] * slots
            totals["mae_sum"] += _pure_value_mae(hidden, transition_targets, model.projector, state_min_value) * slots
            totals["mae_slots"] += slots
            structured_loss, structured_metrics = _pure_structured_numeric_loss_and_metrics(
                hidden,
                transition_targets,
                model,
                regression_weight,
                digit_weight,
            )
            if structured_loss is not None:
                _accumulate_structured_totals(structured_totals, structured_metrics)
            for key in (
                "slots",
                "value_correct",
                "phase_correct",
                "transition_correct",
                "sum_correct",
                "sum_slots",
                "product_correct",
                "product_slots",
                "final_correct",
                "final_slots",
                "sequence_correct",
                "sequence_slots",
            ):
                totals[key] += int(metrics[key])
    result = {
        "loss": totals["loss"] / max(1, totals["slots"]),
        "value_ce": totals["value_ce"] / max(1, totals["slots"]),
        "phase_ce": totals["phase_ce"] / max(1, totals["slots"]),
        "value_accuracy": _safe_div(totals["value_correct"], totals["slots"]),
        "phase_accuracy": _safe_div(totals["phase_correct"], totals["slots"]),
        "transition_accuracy": _safe_div(totals["transition_correct"], totals["slots"]),
        "sum_accuracy": _safe_div(totals["sum_correct"], totals["sum_slots"]),
        "product_accuracy": _safe_div(totals["product_correct"], totals["product_slots"]),
        "final_accuracy": _safe_div(totals["final_correct"], totals["final_slots"]),
        "sequence_accuracy": _safe_div(totals["sequence_correct"], totals["sequence_slots"]),
        "value_mae": totals["mae_sum"] / max(1, totals["mae_slots"]),
        "slots": totals["slots"],
        "sequence_slots": totals["sequence_slots"],
    }
    result.update(_structured_result_from_totals(structured_totals))
    return result


def train_pure_latent_arithmetic_transition(
    model: PureLatentArithmeticTransitionModel,
    train_items: list[PureArithmeticTuple],
    eval_items: list[PureArithmeticTuple],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    state_min_value: int,
    state_max_value: int,
    value_weight: float,
    phase_weight: float,
    regression_weight: float,
    digit_weight: float,
) -> tuple[list[dict], dict]:
    model.to(device=device, dtype=torch.float32).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(1, epochs + 1):
        rng = random.Random(epoch)
        order = list(range(len(train_items)))
        rng.shuffle(order)
        totals = {
            "loss": 0.0,
            "value_ce": 0.0,
            "phase_ce": 0.0,
            "slots": 0,
            "value_correct": 0,
            "phase_correct": 0,
            "transition_correct": 0,
            "sum_correct": 0,
            "sum_slots": 0,
            "product_correct": 0,
            "product_slots": 0,
            "final_correct": 0,
            "final_slots": 0,
            "sequence_correct": 0,
            "sequence_slots": 0,
        }
        structured_totals = _empty_structured_totals()
        for start in range(0, len(order), batch_size):
            idxs = order[start:start + batch_size]
            batch_items = [train_items[i] for i in idxs]
            operands = _pure_arithmetic_operands_tensor(batch_items, device)
            hidden = model(operands)
            transition_targets = _build_pure_arithmetic_transition_targets(
                batch_items,
                model.latent_steps,
                state_min_value,
                state_max_value,
                device,
            )
            loss_value, metrics = _arithmetic_state_machine_loss_and_metrics(
                hidden,
                transition_targets,
                model.projector,
                value_weight,
                phase_weight,
            )
            structured_loss, structured_metrics = _pure_structured_numeric_loss_and_metrics(
                hidden,
                transition_targets,
                model,
                regression_weight,
                digit_weight,
            )
            if loss_value is None and structured_loss is None:
                continue
            if loss_value is None:
                total_loss_value = structured_loss
            elif structured_loss is None:
                total_loss_value = loss_value
            else:
                total_loss_value = loss_value + structured_loss
            opt.zero_grad(set_to_none=True)
            total_loss_value.backward()
            opt.step()
            slots = int(metrics["slots"])
            totals["loss"] += metrics["loss"] * slots
            totals["value_ce"] += metrics["value_ce"] * slots
            totals["phase_ce"] += metrics["phase_ce"] * slots
            if structured_loss is not None:
                _accumulate_structured_totals(structured_totals, structured_metrics)
            for key in (
                "slots",
                "value_correct",
                "phase_correct",
                "transition_correct",
                "sum_correct",
                "sum_slots",
                "product_correct",
                "product_slots",
                "final_correct",
                "final_slots",
                "sequence_correct",
                "sequence_slots",
            ):
                totals[key] += int(metrics[key])
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / max(1, totals["slots"]),
            "value_ce": totals["value_ce"] / max(1, totals["slots"]),
            "phase_ce": totals["phase_ce"] / max(1, totals["slots"]),
            "value_accuracy": _safe_div(totals["value_correct"], totals["slots"]),
            "phase_accuracy": _safe_div(totals["phase_correct"], totals["slots"]),
            "transition_accuracy": _safe_div(totals["transition_correct"], totals["slots"]),
            "sum_accuracy": _safe_div(totals["sum_correct"], totals["sum_slots"]),
            "product_accuracy": _safe_div(totals["product_correct"], totals["product_slots"]),
            "final_accuracy": _safe_div(totals["final_correct"], totals["final_slots"]),
            "sequence_accuracy": _safe_div(totals["sequence_correct"], totals["sequence_slots"]),
        }
        row.update(_structured_result_from_totals(structured_totals))
        row["training_loss"] = row["loss"] + row["structured_loss"]
        history.append(row)
        print(
            f"  pure_epoch={epoch:03d} loss={row['training_loss']:.6f} value_ce={row['value_ce']:.6f} "
            f"trans_acc={row['transition_accuracy'] * 100:.1f}% seq_acc={row['sequence_accuracy'] * 100:.1f}% "
            f"sum={row['sum_accuracy'] * 100:.1f}% product={row['product_accuracy'] * 100:.1f}% "
            f"final={row['final_accuracy'] * 100:.1f}% "
            f"digit_value_acc={row['digit_value_accuracy'] * 100:.1f}% "
            f"reg_mae={row['regression_mae']:.2f}",
            flush=True,
        )
    eval_result = evaluate_pure_latent_arithmetic_transition(
        model,
        eval_items,
        batch_size,
        device,
        state_min_value,
        state_max_value,
        value_weight,
        phase_weight,
        regression_weight,
        digit_weight,
    )
    print(
        f"pure_eval transition_acc={eval_result['transition_accuracy'] * 100:.1f}% "
        f"sequence_acc={eval_result['sequence_accuracy'] * 100:.1f}% "
        f"sum={eval_result['sum_accuracy'] * 100:.1f}% product={eval_result['product_accuracy'] * 100:.1f}% "
        f"final={eval_result['final_accuracy'] * 100:.1f}% mae={eval_result['value_mae']:.2f} "
        f"digit_value_acc={eval_result['digit_value_accuracy'] * 100:.1f}% "
        f"reg_mae={eval_result['regression_mae']:.2f}",
        flush=True,
    )
    return history, eval_result


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


def _arithmetic_cases_to_tuples(cases: list[DistillCase]) -> list[PureArithmeticTuple]:
    items: list[PureArithmeticTuple] = []
    for case in cases:
        if case.task != "arithmetic":
            raise ValueError("oracle decode bridge only supports arithmetic cases")
        items.append(PureArithmeticTuple(
            a=int(case.meta["a"]),
            b=int(case.meta["b"]),
            c=int(case.meta["c"]),
            d=int(case.meta["d"]),
        ))
    return items


def _oracle_structured_decode_features(
    cases: list[DistillCase],
    num_digits: int,
    c_max: int,
    device: torch.device,
) -> torch.Tensor:
    """Build oracle structured arithmetic state features for B2.6 decode bridge.

    The feature vector is deliberately explicit: operands, intermediate digit
    slots, final sign, and carry/borrow traces.  The arithmetic itself is not
    learned here; this isolates the bridge from structured state to LLM decode.
    """
    items = _arithmetic_cases_to_tuples(cases)
    trace = _build_direct_digit_arithmetic_trace(items, num_digits, device)
    a, b, c, d = _pure_digit_items_to_operands(items, device)
    sum_value = a + b
    product_value = sum_value * c
    final_value = product_value - d
    scalar = torch.stack([
        a.float() / 100.0,
        b.float() / 100.0,
        c.float() / 10.0,
        d.float() / 200.0,
        sum_value.float() / 200.0,
        product_value.float() / 2000.0,
        final_value.float() / 2000.0,
    ], dim=-1)
    digit_features = torch.cat([
        torch.nn.functional.one_hot(trace.sum_digits.long().clamp(0, 9), num_classes=10).float().reshape(len(cases), -1),
        torch.nn.functional.one_hot(trace.product_digits.long().clamp(0, 9), num_classes=10).float().reshape(len(cases), -1),
        torch.nn.functional.one_hot(trace.final_digits.long().clamp(0, 9), num_classes=10).float().reshape(len(cases), -1),
    ], dim=-1)
    sign_features = torch.nn.functional.one_hot(trace.final_sign.long().clamp(0, 1), num_classes=2).float()
    add_carry = torch.nn.functional.one_hot(trace.add_carry.long().clamp(0, 1), num_classes=2).float().reshape(len(cases), -1)
    mul_classes = max(2, c_max + 1)
    mul_carry = torch.nn.functional.one_hot(trace.mul_carry.long().clamp(0, mul_classes - 1), num_classes=mul_classes).float().reshape(len(cases), -1)
    sub_borrow = torch.nn.functional.one_hot(trace.sub_borrow.long().clamp(0, 1), num_classes=2).float().reshape(len(cases), -1)
    overflow = trace.overflow_mask.float().unsqueeze(-1)
    return torch.cat([scalar, digit_features, sign_features, add_carry, mul_carry, sub_borrow, overflow], dim=-1)


def _build_oracle_decode_bridge_batch(
    model,
    tokenizer,
    bridge: OracleStructuredDecodeBridge,
    cases: list[DistillCase],
    num_digits: int,
    c_max: int,
    device: torch.device,
):
    embed = model.get_input_embeddings()
    model_dtype = next(model.parameters()).dtype
    features = _oracle_structured_decode_features(cases, num_digits, c_max, device)
    latents = bridge(features, dtype=model_dtype)
    seq_embeds = []
    answer_targets = []
    answer_positions = []
    max_len = 0
    for i, case in enumerate(cases):
        prompt_ids = torch.tensor(tokenizer.encode(case.prompt, add_special_tokens=False), device=device, dtype=torch.long)
        answer_ids = torch.tensor(_answer_token_ids(tokenizer, case), device=device, dtype=torch.long)
        prompt_embeds = embed(prompt_ids).detach()
        if answer_ids.numel() > 1:
            answer_prefix = embed(answer_ids[:-1]).detach()
            parts = [prompt_embeds, latents[i], answer_prefix]
        else:
            parts = [prompt_embeds, latents[i]]
        sample_embeds = torch.cat(parts, dim=0)
        first_answer_position = prompt_embeds.size(0) + bridge.latent_steps - 1
        positions = torch.arange(
            first_answer_position,
            first_answer_position + answer_ids.numel(),
            device=device,
            dtype=torch.long,
        )
        seq_embeds.append(sample_embeds)
        answer_targets.append(answer_ids)
        answer_positions.append(positions)
        max_len = max(max_len, sample_embeds.size(0))
    hidden_size = seq_embeds[0].size(-1)
    inputs_embeds = torch.zeros(len(cases), max_len, hidden_size, device=device, dtype=model_dtype)
    attention_mask = torch.zeros(len(cases), max_len, device=device, dtype=torch.long)
    flat_positions = []
    flat_targets = []
    for i, sample_embeds in enumerate(seq_embeds):
        n = sample_embeds.size(0)
        inputs_embeds[i, :n] = sample_embeds
        attention_mask[i, :n] = 1
        flat_positions.append(answer_positions[i] + i * max_len)
        flat_targets.append(answer_targets[i])
    return inputs_embeds, attention_mask, torch.cat(flat_positions), torch.cat(flat_targets)


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
    state_probe: ArithmeticStateProbe | None,
    numeric_state_embedding: NumericStateEmbedding | None,
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
    state_supervision_weight: float,
    state_min_value: int,
    state_max_value: int,
    state_embedding_loss_weight: float,
    state_embedding_mse_weight: float,
    arithmetic_state_machine_weight: float,
    arithmetic_state_machine_value_weight: float,
    arithmetic_state_machine_phase_weight: float,
    arithmetic_state_machine_source: str,
    arithmetic_state_machine_repeat_final: bool,
) -> list[dict]:
    if step_hidden_mode not in ("input", "output"):
        raise ValueError(f"unknown step_hidden_mode={step_hidden_mode!r}")
    if latent_step_curriculum not in ("none", "linear"):
        raise ValueError(f"unknown latent_step_curriculum={latent_step_curriculum!r}")
    if arithmetic_state_machine_source not in ("input", "output"):
        raise ValueError(f"unknown arithmetic_state_machine_source={arithmetic_state_machine_source!r}")
    if arithmetic_state_machine_weight > 0:
        if not hasattr(projector, "state_machine_logits"):
            raise ValueError("--arithmetic-state-machine-weight requires --projector-kind arithmetic_state_machine")
        if any(case.task != "arithmetic" for case in cases):
            raise ValueError("arithmetic state-machine trainer only supports --task arithmetic")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    projector.to(device=device, dtype=torch.float32).train()
    trainable_params = list(p for p in projector.parameters() if p.requires_grad)
    if state_probe is not None:
        state_probe.to(device=device, dtype=torch.float32).train()
        trainable_params.extend(p for p in state_probe.parameters() if p.requires_grad)
    if numeric_state_embedding is not None:
        numeric_state_embedding.to(device=device, dtype=torch.float32).train()
        trainable_params.extend(p for p in numeric_state_embedding.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
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
        total_state = 0.0
        total_state_correct = 0
        total_state_slots = 0
        total_state_embedding = 0.0
        total_state_embedding_ce = 0.0
        total_state_embedding_mse = 0.0
        total_state_embedding_correct = 0
        total_state_embedding_slots = 0
        total_state_machine = 0.0
        total_state_machine_value_ce = 0.0
        total_state_machine_phase_ce = 0.0
        total_state_machine_slots = 0
        total_state_machine_value_correct = 0
        total_state_machine_phase_correct = 0
        total_state_machine_transition_correct = 0
        total_state_machine_sum_correct = 0
        total_state_machine_sum_slots = 0
        total_state_machine_product_correct = 0
        total_state_machine_product_slots = 0
        total_state_machine_final_correct = 0
        total_state_machine_final_slots = 0
        total_state_machine_sequence_correct = 0
        total_state_machine_sequence_slots = 0
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
                output_hidden_states=(
                    (step_hidden_loss_weight > 0 and step_hidden_mode == "output")
                    or (state_supervision_weight > 0 and state_probe is not None)
                    or (state_embedding_loss_weight > 0 and numeric_state_embedding is not None)
                    or (arithmetic_state_machine_weight > 0 and arithmetic_state_machine_source == "output")
                ),
                use_cache=False,
                return_dict=True,
            )
            logits = out.logits.reshape(-1, out.logits.size(-1))[flat_positions].float()
            ce_loss = torch.nn.functional.cross_entropy(logits, targets)
            loss = ce_loss
            step_hidden_loss_value = None
            output_hiddens = None
            if out.hidden_states is not None:
                output_hiddens = out.hidden_states[-1].reshape(-1, out.hidden_states[-1].size(-1))[
                    flat_latent_positions
                ].view(len(batch_cases), effective_latent_steps, -1)
            if step_hidden_loss_weight > 0 and step_mask is not None and step_mask.any():
                if step_hidden_mode == "output":
                    if output_hiddens is None:
                        raise RuntimeError("output hidden states are required for output hidden supervision")
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
            state_loss_value = None
            if state_supervision_weight > 0 and state_probe is not None:
                if output_hiddens is None:
                    raise RuntimeError("output hidden states are required for arithmetic state supervision")
                state_targets, state_mask = _build_arithmetic_state_targets(
                    batch_cases,
                    effective_latent_steps,
                    state_min_value,
                    state_max_value,
                    device,
                )
                if state_mask.any():
                    state_logits = torch.stack([
                        state_probe.forward_step(output_hiddens[:, step_idx], step_idx)
                        for step_idx in range(effective_latent_steps)
                    ], dim=1)
                    flat_state_logits = state_logits[state_mask]
                    flat_state_targets = state_targets[state_mask]
                    state_loss_value = torch.nn.functional.cross_entropy(flat_state_logits, flat_state_targets)
                    loss = loss + state_supervision_weight * state_loss_value
            state_embedding_loss_value = None
            state_embedding_ce_value = 0.0
            state_embedding_mse_value = 0.0
            if state_embedding_loss_weight > 0 and numeric_state_embedding is not None:
                if output_hiddens is None:
                    raise RuntimeError("output hidden states are required for numeric state embedding supervision")
                state_embed_targets, state_embed_mask = _build_arithmetic_state_targets(
                    batch_cases,
                    effective_latent_steps,
                    numeric_state_embedding.min_value,
                    numeric_state_embedding.max_value,
                    device,
                )
                (
                    state_embedding_loss_value,
                    state_embedding_ce_value,
                    state_embedding_mse_value,
                    state_embedding_correct,
                    state_embedding_slots,
                ) = _numeric_state_embedding_loss(
                    output_hiddens,
                    state_embed_targets,
                    state_embed_mask,
                    numeric_state_embedding,
                    state_embedding_mse_weight,
                )
                if state_embedding_loss_value is not None:
                    loss = loss + state_embedding_loss_weight * state_embedding_loss_value
            state_machine_loss_value = None
            state_machine_metrics = None
            if arithmetic_state_machine_weight > 0:
                if arithmetic_state_machine_source == "output":
                    if output_hiddens is None:
                        raise RuntimeError("output hidden states are required for output state-machine supervision")
                    state_machine_hidden = output_hiddens
                else:
                    state_machine_hidden = latent_embeds.float()
                transition_targets = _build_arithmetic_transition_targets(
                    batch_cases,
                    effective_latent_steps,
                    state_min_value,
                    state_max_value,
                    device,
                    repeat_final=arithmetic_state_machine_repeat_final,
                )
                state_machine_loss_value, state_machine_metrics = _arithmetic_state_machine_loss_and_metrics(
                    state_machine_hidden,
                    transition_targets,
                    projector,
                    arithmetic_state_machine_value_weight,
                    arithmetic_state_machine_phase_weight,
                )
                if state_machine_loss_value is not None:
                    loss = loss + arithmetic_state_machine_weight * state_machine_loss_value
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * targets.numel()
            total_ce += float(ce_loss.item()) * targets.numel()
            if step_hidden_loss_value is not None:
                slots = int(step_mask.sum().item())
                total_step_hidden += float(step_hidden_loss_value.item()) * slots
                total_step_slots += slots
            if state_loss_value is not None:
                state_slots = int(state_mask.sum().item())
                total_state += float(state_loss_value.item()) * state_slots
                total_state_correct += int((flat_state_logits.argmax(dim=-1) == flat_state_targets).sum().item())
                total_state_slots += state_slots
            if state_embedding_loss_value is not None:
                total_state_embedding += float(state_embedding_loss_value.item()) * state_embedding_slots
                total_state_embedding_ce += state_embedding_ce_value * state_embedding_slots
                total_state_embedding_mse += state_embedding_mse_value * state_embedding_slots
                total_state_embedding_correct += state_embedding_correct
                total_state_embedding_slots += state_embedding_slots
            if state_machine_loss_value is not None and state_machine_metrics is not None:
                sm_slots = int(state_machine_metrics["slots"])
                total_state_machine += state_machine_metrics["loss"] * sm_slots
                total_state_machine_value_ce += state_machine_metrics["value_ce"] * sm_slots
                total_state_machine_phase_ce += state_machine_metrics["phase_ce"] * sm_slots
                total_state_machine_slots += sm_slots
                total_state_machine_value_correct += int(state_machine_metrics["value_correct"])
                total_state_machine_phase_correct += int(state_machine_metrics["phase_correct"])
                total_state_machine_transition_correct += int(state_machine_metrics["transition_correct"])
                total_state_machine_sum_correct += int(state_machine_metrics["sum_correct"])
                total_state_machine_sum_slots += int(state_machine_metrics["sum_slots"])
                total_state_machine_product_correct += int(state_machine_metrics["product_correct"])
                total_state_machine_product_slots += int(state_machine_metrics["product_slots"])
                total_state_machine_final_correct += int(state_machine_metrics["final_correct"])
                total_state_machine_final_slots += int(state_machine_metrics["final_slots"])
                total_state_machine_sequence_correct += int(state_machine_metrics["sequence_correct"])
                total_state_machine_sequence_slots += int(state_machine_metrics["sequence_slots"])
            total_tokens += int(targets.numel())
        avg = total / max(1, total_tokens)
        avg_ce = total_ce / max(1, total_tokens)
        avg_step_hidden = total_step_hidden / max(1, total_step_slots) if total_step_slots else 0.0
        avg_state_loss = total_state / max(1, total_state_slots) if total_state_slots else 0.0
        state_acc = total_state_correct / max(1, total_state_slots) if total_state_slots else 0.0
        avg_state_embedding_loss = total_state_embedding / max(1, total_state_embedding_slots) if total_state_embedding_slots else 0.0
        avg_state_embedding_ce = total_state_embedding_ce / max(1, total_state_embedding_slots) if total_state_embedding_slots else 0.0
        avg_state_embedding_mse = total_state_embedding_mse / max(1, total_state_embedding_slots) if total_state_embedding_slots else 0.0
        state_embedding_acc = total_state_embedding_correct / max(1, total_state_embedding_slots) if total_state_embedding_slots else 0.0
        avg_state_machine_loss = total_state_machine / max(1, total_state_machine_slots) if total_state_machine_slots else 0.0
        avg_state_machine_value_ce = total_state_machine_value_ce / max(1, total_state_machine_slots) if total_state_machine_slots else 0.0
        avg_state_machine_phase_ce = total_state_machine_phase_ce / max(1, total_state_machine_slots) if total_state_machine_slots else 0.0
        state_machine_value_acc = _safe_div(total_state_machine_value_correct, total_state_machine_slots)
        state_machine_phase_acc = _safe_div(total_state_machine_phase_correct, total_state_machine_slots)
        state_machine_transition_acc = _safe_div(total_state_machine_transition_correct, total_state_machine_slots)
        state_machine_sum_acc = _safe_div(total_state_machine_sum_correct, total_state_machine_sum_slots)
        state_machine_product_acc = _safe_div(total_state_machine_product_correct, total_state_machine_product_slots)
        state_machine_final_acc = _safe_div(total_state_machine_final_correct, total_state_machine_final_slots)
        state_machine_sequence_acc = _safe_div(total_state_machine_sequence_correct, total_state_machine_sequence_slots)
        history.append({
            "epoch": epoch,
            "teacher_forcing_loss": avg,
            "ce_loss": avg_ce,
            "step_hidden_loss": avg_step_hidden,
            "step_hidden_mode": step_hidden_mode,
            "latent_steps": effective_latent_steps,
            "latent_step_curriculum": latent_step_curriculum,
            "state_loss": avg_state_loss,
            "state_accuracy": state_acc,
            "state_embedding_loss": avg_state_embedding_loss,
            "state_embedding_ce": avg_state_embedding_ce,
            "state_embedding_mse": avg_state_embedding_mse,
            "state_embedding_accuracy": state_embedding_acc,
            "state_machine_loss": avg_state_machine_loss,
            "state_machine_value_ce": avg_state_machine_value_ce,
            "state_machine_phase_ce": avg_state_machine_phase_ce,
            "state_machine_value_accuracy": state_machine_value_acc,
            "state_machine_phase_accuracy": state_machine_phase_acc,
            "state_machine_transition_accuracy": state_machine_transition_acc,
            "state_machine_sum_accuracy": state_machine_sum_acc,
            "state_machine_product_accuracy": state_machine_product_acc,
            "state_machine_final_accuracy": state_machine_final_acc,
            "state_machine_sequence_accuracy": state_machine_sequence_acc,
        })
        print(
            f"  hf_epoch={epoch:03d} latent_steps={effective_latent_steps} "
            f"loss={avg:.6f} ce={avg_ce:.6f} step_hidden={avg_step_hidden:.6f} "
            f"state={avg_state_loss:.6f} state_acc={state_acc * 100:.1f}% "
            f"state_embed={avg_state_embedding_loss:.6f} state_embed_acc={state_embedding_acc * 100:.1f}% "
            f"sm={avg_state_machine_loss:.6f} sm_trans_acc={state_machine_transition_acc * 100:.1f}% "
            f"sm_seq_acc={state_machine_sequence_acc * 100:.1f}%",
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


def train_oracle_decode_bridge(
    model,
    tokenizer,
    bridge: OracleStructuredDecodeBridge,
    cases: list[DistillCase],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    num_digits: int,
    c_max: int,
) -> list[dict]:
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    bridge.to(device=device, dtype=torch.float32).train()
    opt = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(1, epochs + 1):
        rng = random.Random(epoch)
        order = list(range(len(cases)))
        rng.shuffle(order)
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        for start in range(0, len(order), batch_size):
            idxs = order[start:start + batch_size]
            batch_cases = [cases[i] for i in idxs]
            inputs_embeds, attention_mask, flat_positions, targets = _build_oracle_decode_bridge_batch(
                model, tokenizer, bridge, batch_cases, num_digits, c_max, device
            )
            out = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            logits = out.logits.reshape(-1, out.logits.size(-1))[flat_positions].float()
            loss = torch.nn.functional.cross_entropy(logits, targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tokens = int(targets.numel())
            total_loss += float(loss.item()) * tokens
            total_tokens += tokens
            total_correct += int((logits.argmax(dim=-1) == targets).sum().item())
        row = {
            "epoch": epoch,
            "loss": total_loss / max(1, total_tokens),
            "token_accuracy": _safe_div(total_correct, total_tokens),
            "tokens": total_tokens,
        }
        history.append(row)
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            print(
                f"  oracle_bridge_epoch={epoch:03d} loss={row['loss']:.4f} "
                f"tok_acc={row['token_accuracy'] * 100:.1f}%",
                flush=True,
            )
    return history


@torch.no_grad()
def generate_hf_with_oracle_decode_bridge(
    model,
    tokenizer,
    bridge: OracleStructuredDecodeBridge,
    case: DistillCase,
    max_new_tokens: int,
    device: torch.device,
    num_digits: int,
    c_max: int,
) -> str:
    model.eval()
    bridge.eval()
    prompt_ids = torch.tensor(tokenizer.encode(case.prompt, add_special_tokens=False), device=device, dtype=torch.long).unsqueeze(0)
    embed = model.get_input_embeddings()
    prompt_embeds = embed(prompt_ids).detach()
    features = _oracle_structured_decode_features([case], num_digits, c_max, device)
    latents = bridge(features, dtype=prompt_embeds.dtype)
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
def evaluate_oracle_decode_bridge(
    model,
    tokenizer,
    bridge: OracleStructuredDecodeBridge,
    cases: list[DistillCase],
    max_new_tokens: int,
    device: torch.device,
    num_digits: int,
    c_max: int,
) -> dict:
    details = []
    for case in cases:
        text = generate_hf_with_oracle_decode_bridge(
            model, tokenizer, bridge, case, max_new_tokens, device, num_digits, c_max
        )
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
            f"  oracle_bridge_eval expected={case.expected} answer_only={answer_only} "
            f"contains={contains_expected} hit={hit} raw={text[:48]!r}",
            flush=True,
        )
    acc = sum(int(x["hit"]) for x in details) / max(1, len(details))
    contains_acc = sum(int(x["contains_expected"]) for x in details) / max(1, len(details))
    return {"accuracy": acc, "contains_accuracy": contains_acc, "details": details}


@torch.no_grad()
def evaluate_arithmetic_state_machine_hf(
    model,
    tokenizer,
    projector,
    cases: list[DistillCase],
    batch_size: int,
    device: torch.device,
    latent_steps: int,
    source: str,
    repeat_final: bool,
    state_min_value: int,
    state_max_value: int,
    value_weight: float,
    phase_weight: float,
) -> dict | None:
    if not hasattr(projector, "state_machine_logits"):
        return None
    if source not in ("input", "output"):
        raise ValueError(f"unknown arithmetic state-machine source={source!r}")
    model.eval()
    projector.eval()
    totals = {
        "loss": 0.0,
        "value_ce": 0.0,
        "phase_ce": 0.0,
        "slots": 0,
        "value_correct": 0,
        "phase_correct": 0,
        "transition_correct": 0,
        "sum_correct": 0,
        "sum_slots": 0,
        "product_correct": 0,
        "product_slots": 0,
        "final_correct": 0,
        "final_slots": 0,
        "sequence_correct": 0,
        "sequence_slots": 0,
    }
    for start in range(0, len(cases), batch_size):
        batch_cases = cases[start:start + batch_size]
        prompts = [case.prompt for case in batch_cases]
        src = collect_source_hidden_hf(model, tokenizer, prompts, device)
        (
            inputs_embeds,
            attention_mask,
            _flat_positions,
            _answer_targets,
            latent_embeds,
            flat_latent_positions,
        ) = _build_hf_teacher_batch(
            model, tokenizer, projector, batch_cases, src, device, latent_steps
        )
        if source == "output":
            out = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden = out.hidden_states[-1].reshape(-1, out.hidden_states[-1].size(-1))[
                flat_latent_positions
            ].view(len(batch_cases), latent_steps, -1)
        else:
            hidden = latent_embeds.float()
        transition_targets = _build_arithmetic_transition_targets(
            batch_cases,
            latent_steps,
            state_min_value,
            state_max_value,
            device,
            repeat_final=repeat_final,
        )
        _loss, metrics = _arithmetic_state_machine_loss_and_metrics(
            hidden,
            transition_targets,
            projector,
            value_weight,
            phase_weight,
        )
        slots = int(metrics["slots"])
        totals["loss"] += metrics["loss"] * slots
        totals["value_ce"] += metrics["value_ce"] * slots
        totals["phase_ce"] += metrics["phase_ce"] * slots
        for key in (
            "slots",
            "value_correct",
            "phase_correct",
            "transition_correct",
            "sum_correct",
            "sum_slots",
            "product_correct",
            "product_slots",
            "final_correct",
            "final_slots",
            "sequence_correct",
            "sequence_slots",
        ):
            totals[key] += int(metrics[key])
    result = {
        "loss": totals["loss"] / max(1, totals["slots"]),
        "value_ce": totals["value_ce"] / max(1, totals["slots"]),
        "phase_ce": totals["phase_ce"] / max(1, totals["slots"]),
        "value_accuracy": _safe_div(totals["value_correct"], totals["slots"]),
        "phase_accuracy": _safe_div(totals["phase_correct"], totals["slots"]),
        "transition_accuracy": _safe_div(totals["transition_correct"], totals["slots"]),
        "sum_accuracy": _safe_div(totals["sum_correct"], totals["sum_slots"]),
        "product_accuracy": _safe_div(totals["product_correct"], totals["product_slots"]),
        "final_accuracy": _safe_div(totals["final_correct"], totals["final_slots"]),
        "sequence_accuracy": _safe_div(totals["sequence_correct"], totals["sequence_slots"]),
        "slots": totals["slots"],
        "sequence_slots": totals["sequence_slots"],
        "source": source,
    }
    print(
        f"state_machine_eval transition_acc={result['transition_accuracy'] * 100:.1f}% "
        f"sequence_acc={result['sequence_accuracy'] * 100:.1f}% "
        f"sum={result['sum_accuracy'] * 100:.1f}% product={result['product_accuracy'] * 100:.1f}% "
        f"final={result['final_accuracy'] * 100:.1f}%",
        flush=True,
    )
    return result


def run_pure_latent_arithmetic_benchmark(args) -> None:
    if args.latent_steps < 3:
        raise ValueError("--pure-latent-arithmetic requires --latent-steps >= 3")
    device = torch.device(args.train_device)
    total_cases = args.train_cases + args.eval_cases
    all_items = build_pure_arithmetic_tuples(
        total_cases,
        seed=args.seed,
        a_min=args.pure_arith_a_min,
        a_max=args.pure_arith_a_max,
        b_min=args.pure_arith_b_min,
        b_max=args.pure_arith_b_max,
        c_min=args.pure_arith_c_min,
        c_max=args.pure_arith_c_max,
        d_min=args.pure_arith_d_min,
        d_max=args.pure_arith_d_max,
    )
    train_items = all_items[:args.train_cases]
    eval_items = all_items[args.train_cases:]
    model = PureLatentArithmeticTransitionModel(
        args.pure_hidden_size,
        latent_steps=args.latent_steps,
        min_value=args.state_min_value,
        max_value=args.state_max_value,
        numeric_representation=args.pure_numeric_representation,
    )
    t0 = time.time()
    print("Training pure latent arithmetic transition micro-benchmark...", flush=True)
    history, eval_result = train_pure_latent_arithmetic_transition(
        model,
        train_items,
        eval_items,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        state_min_value=args.state_min_value,
        state_max_value=args.state_max_value,
        value_weight=args.pure_arith_value_weight,
        phase_weight=args.pure_arith_phase_weight,
        regression_weight=args.pure_arith_regression_weight,
        digit_weight=args.pure_arith_digit_weight,
    )
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "hidden_size": args.pure_hidden_size,
        "args": vars(args),
        "history": history,
        "eval": eval_result,
    }, args.checkpoint)
    result = {
        "args": vars(args),
        "hidden_size": args.pure_hidden_size,
        "train_cases": len(train_items),
        "eval_cases": len(eval_items),
        "numeric_representation": args.pure_numeric_representation,
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


def evaluate_pure_digit_arithmetic(
    items: list[PureArithmeticTuple],
    batch_size: int,
    device: torch.device,
    num_digits: int,
) -> dict:
    weighted = {
        "valid_cases": 0,
        "overflow_cases": 0,
        "sum_value_accuracy": 0.0,
        "product_value_accuracy": 0.0,
        "final_value_accuracy": 0.0,
        "sequence_value_accuracy": 0.0,
        "sum_digit_accuracy": 0.0,
        "product_digit_accuracy": 0.0,
        "final_digit_accuracy": 0.0,
        "add_carry_accuracy": 0.0,
        "mul_carry_accuracy": 0.0,
        "sub_borrow_accuracy": 0.0,
        "final_sign_accuracy": 0.0,
        "sum_mae": 0.0,
        "product_mae": 0.0,
        "final_mae": 0.0,
    }
    total_valid = 0
    total_overflow = 0
    for start in range(0, len(items), batch_size):
        batch_items = items[start:start + batch_size]
        target = _build_direct_digit_arithmetic_trace(batch_items, num_digits, device)
        pred = _build_algorithmic_digit_arithmetic_trace(batch_items, num_digits, device)
        metrics = _pure_digit_trace_metrics(pred, target)
        valid = int(metrics["valid_cases"])
        total_valid += valid
        total_overflow += int(metrics["overflow_cases"])
        for key in weighted:
            if key in ("valid_cases", "overflow_cases"):
                continue
            weighted[key] += float(metrics[key]) * valid
    result = {
        key: (weighted[key] / max(1, total_valid))
        for key in weighted
        if key not in ("valid_cases", "overflow_cases")
    }
    result["valid_cases"] = total_valid
    result["overflow_cases"] = total_overflow
    result["num_digits"] = num_digits
    return result


@torch.no_grad()
def evaluate_trainable_digit_operator(
    model: TrainableDigitWiseOperator,
    examples: list[DigitOperatorExample],
    batch_size: int,
    device: torch.device,
    carry_weight: float,
) -> dict:
    del batch_size  # the digit-combination table is small; one pass keeps per-op metrics exact.
    model.eval()
    _loss, metrics = _digit_operator_loss_and_metrics(model, examples, device, carry_weight)
    return metrics


@torch.no_grad()
def evaluate_trainable_digit_tuple_arithmetic(
    model: TrainableDigitWiseOperator,
    items: list[PureArithmeticTuple],
    batch_size: int,
    device: torch.device,
    num_digits: int,
) -> dict:
    model.eval()
    weighted = {
        "valid_cases": 0,
        "overflow_cases": 0,
        "sum_value_accuracy": 0.0,
        "product_value_accuracy": 0.0,
        "final_value_accuracy": 0.0,
        "sequence_value_accuracy": 0.0,
        "sum_digit_accuracy": 0.0,
        "product_digit_accuracy": 0.0,
        "final_digit_accuracy": 0.0,
        "add_carry_accuracy": 0.0,
        "mul_carry_accuracy": 0.0,
        "sub_borrow_accuracy": 0.0,
        "final_sign_accuracy": 0.0,
        "sum_mae": 0.0,
        "product_mae": 0.0,
        "final_mae": 0.0,
    }
    total_valid = 0
    total_overflow = 0
    for start in range(0, len(items), batch_size):
        batch_items = items[start:start + batch_size]
        target = _build_direct_digit_arithmetic_trace(batch_items, num_digits, device)
        pred = _build_trainable_digit_arithmetic_trace(model, batch_items, num_digits, device)
        metrics = _pure_digit_trace_metrics(pred, target)
        valid = int(metrics["valid_cases"])
        total_valid += valid
        total_overflow += int(metrics["overflow_cases"])
        for key in weighted:
            if key in ("valid_cases", "overflow_cases"):
                continue
            weighted[key] += float(metrics[key]) * valid
    result = {
        key: (weighted[key] / max(1, total_valid))
        for key in weighted
        if key not in ("valid_cases", "overflow_cases")
    }
    result["valid_cases"] = total_valid
    result["overflow_cases"] = total_overflow
    result["num_digits"] = num_digits
    return result


def train_pure_digit_operator(
    model: TrainableDigitWiseOperator,
    train_examples: list[DigitOperatorExample],
    heldout_examples: list[DigitOperatorExample],
    train_items: list[PureArithmeticTuple],
    eval_items: list[PureArithmeticTuple],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    carry_weight: float,
    num_digits: int,
) -> tuple[list[dict], dict, dict]:
    model.to(device=device, dtype=torch.float32).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(1, epochs + 1):
        rng = random.Random(epoch)
        order = list(range(len(train_examples)))
        rng.shuffle(order)
        weighted: dict[str, float] = {}
        total = 0
        for start in range(0, len(order), batch_size):
            batch = [train_examples[i] for i in order[start:start + batch_size]]
            loss, metrics = _digit_operator_loss_and_metrics(model, batch, device, carry_weight)
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            count = int(metrics["examples"])
            total += count
            for key, value in metrics.items():
                if key == "examples":
                    continue
                weighted[key] = weighted.get(key, 0.0) + float(value) * count
        row = {key: value / max(1, total) for key, value in weighted.items()}
        row["epoch"] = epoch
        row["examples"] = total
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            heldout = evaluate_trainable_digit_operator(model, heldout_examples, batch_size, device, carry_weight)
            tuple_eval = evaluate_trainable_digit_tuple_arithmetic(model, eval_items, batch_size, device, num_digits)
            row["heldout_transition_accuracy"] = heldout.get("transition_accuracy", 0.0)
            row["tuple_sequence_accuracy"] = tuple_eval.get("sequence_value_accuracy", 0.0)
            print(
                f"  digit_op_epoch={epoch:03d} train_trans={row.get('transition_accuracy', 0.0) * 100:.1f}% "
                f"heldout_trans={row['heldout_transition_accuracy'] * 100:.1f}% "
                f"tuple_seq={row['tuple_sequence_accuracy'] * 100:.1f}%",
                flush=True,
            )
        history.append(row)
    train_operator_eval = evaluate_trainable_digit_operator(model, train_examples, batch_size, device, carry_weight)
    heldout_operator_eval = evaluate_trainable_digit_operator(model, heldout_examples, batch_size, device, carry_weight)
    train_tuple_eval = evaluate_trainable_digit_tuple_arithmetic(model, train_items, batch_size, device, num_digits)
    eval_tuple_eval = evaluate_trainable_digit_tuple_arithmetic(model, eval_items, batch_size, device, num_digits)
    eval_result = {
        "operator_train": train_operator_eval,
        "operator_heldout": heldout_operator_eval,
        "tuple_train": train_tuple_eval,
        "tuple_eval": eval_tuple_eval,
    }
    print(
        f"digit_operator_eval train_trans={train_operator_eval['transition_accuracy'] * 100:.1f}% "
        f"heldout_trans={heldout_operator_eval['transition_accuracy'] * 100:.1f}% "
        f"tuple_seq={eval_tuple_eval['sequence_value_accuracy'] * 100:.1f}% "
        f"tuple_sum={eval_tuple_eval['sum_value_accuracy'] * 100:.1f}% "
        f"tuple_product={eval_tuple_eval['product_value_accuracy'] * 100:.1f}% "
        f"tuple_final={eval_tuple_eval['final_value_accuracy'] * 100:.1f}%",
        flush=True,
    )
    return history, eval_result, {
        "train_examples": len(train_examples),
        "heldout_examples": len(heldout_examples),
    }


def run_pure_digit_arithmetic_benchmark(args) -> None:
    device = torch.device(args.train_device)
    total_cases = args.train_cases + args.eval_cases
    all_items = build_pure_arithmetic_tuples(
        total_cases,
        seed=args.seed,
        a_min=args.pure_arith_a_min,
        a_max=args.pure_arith_a_max,
        b_min=args.pure_arith_b_min,
        b_max=args.pure_arith_b_max,
        c_min=args.pure_arith_c_min,
        c_max=args.pure_arith_c_max,
        d_min=args.pure_arith_d_min,
        d_max=args.pure_arith_d_max,
    )
    train_items = all_items[:args.train_cases]
    eval_items = all_items[args.train_cases:]
    num_digits = args.pure_digit_num_digits
    if num_digits <= 0:
        num_digits = _infer_pure_digit_num_digits(
            args.pure_arith_a_max,
            args.pure_arith_b_max,
            args.pure_arith_c_max,
            args.pure_arith_d_max,
        )
    t0 = time.time()
    if args.pure_digit_mode == "deterministic":
        print("Evaluating pure digit-wise arithmetic state machine...", flush=True)
        train_result = evaluate_pure_digit_arithmetic(train_items, args.batch_size, device, num_digits)
        eval_result = evaluate_pure_digit_arithmetic(eval_items, args.batch_size, device, num_digits)
        print(
            f"pure_digit_eval sequence={eval_result['sequence_value_accuracy'] * 100:.1f}% "
            f"sum={eval_result['sum_value_accuracy'] * 100:.1f}% "
            f"product={eval_result['product_value_accuracy'] * 100:.1f}% "
            f"final={eval_result['final_value_accuracy'] * 100:.1f}% "
            f"add_carry={eval_result['add_carry_accuracy'] * 100:.1f}% "
            f"mul_carry={eval_result['mul_carry_accuracy'] * 100:.1f}% "
            f"borrow={eval_result['sub_borrow_accuracy'] * 100:.1f}% "
            f"overflow={eval_result['overflow_cases']}",
            flush=True,
        )
        payload = {"train": train_result, "eval": eval_result, "history": []}
        checkpoint_extra = {}
    elif args.pure_digit_mode == "lookup":
        print("Evaluating primitive lookup local digit operators...", flush=True)
        examples = enumerate_digit_operator_examples(args.pure_arith_c_min, args.pure_arith_c_max)
        train_examples, heldout_examples = split_digit_operator_examples(
            examples,
            args.pure_digit_combo_heldout_frac,
            args.pure_digit_combo_split_seed,
        )
        model = PrimitiveLookupDigitWiseOperator(
            c_min=args.pure_arith_c_min,
            c_max=args.pure_arith_c_max,
        ).to(device)
        eval_result = {
            "operator_train": evaluate_trainable_digit_operator(model, train_examples, args.batch_size, device, args.pure_digit_carry_loss_weight),
            "operator_heldout": evaluate_trainable_digit_operator(model, heldout_examples, args.batch_size, device, args.pure_digit_carry_loss_weight),
            "tuple_train": evaluate_trainable_digit_tuple_arithmetic(model, train_items, args.batch_size, device, num_digits),
            "tuple_eval": evaluate_trainable_digit_tuple_arithmetic(model, eval_items, args.batch_size, device, num_digits),
        }
        print(
            f"primitive_lookup_eval train_trans={eval_result['operator_train']['transition_accuracy'] * 100:.1f}% "
            f"heldout_trans={eval_result['operator_heldout']['transition_accuracy'] * 100:.1f}% "
            f"tuple_seq={eval_result['tuple_eval']['sequence_value_accuracy'] * 100:.1f}% "
            f"tuple_sum={eval_result['tuple_eval']['sum_value_accuracy'] * 100:.1f}% "
            f"tuple_product={eval_result['tuple_eval']['product_value_accuracy'] * 100:.1f}% "
            f"tuple_final={eval_result['tuple_eval']['final_value_accuracy'] * 100:.1f}%",
            flush=True,
        )
        payload = {
            "history": [],
            "eval": eval_result,
            "split": {
                "train_examples": len(train_examples),
                "heldout_examples": len(heldout_examples),
            },
        }
        checkpoint_extra = {"model_state_dict": model.state_dict()}
    else:
        print("Training trainable local digit operators...", flush=True)
        examples = enumerate_digit_operator_examples(args.pure_arith_c_min, args.pure_arith_c_max)
        train_examples, heldout_examples = split_digit_operator_examples(
            examples,
            args.pure_digit_combo_heldout_frac,
            args.pure_digit_combo_split_seed,
        )
        model = TrainableDigitWiseOperator(
            c_max=args.pure_arith_c_max,
            hidden_size=args.pure_digit_operator_hidden_size,
            depth=args.pure_digit_operator_depth,
        )
        history, eval_result, split_info = train_pure_digit_operator(
            model,
            train_examples,
            heldout_examples,
            train_items,
            eval_items,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            carry_weight=args.pure_digit_carry_loss_weight,
            num_digits=num_digits,
        )
        payload = {
            "history": history,
            "eval": eval_result,
            "split": split_info,
        }
        checkpoint_extra = {"model_state_dict": model.state_dict()}
    result = {
        "args": vars(args),
        "train_cases": len(train_items),
        "eval_cases": len(eval_items),
        "num_digits": num_digits,
        "mode": args.pure_digit_mode,
        "elapsed_s": time.time() - t0,
        "checkpoint": args.checkpoint,
    }
    result.update(payload)
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    checkpoint_result = dict(result)
    checkpoint_result.update(checkpoint_extra)
    torch.save(checkpoint_result, args.checkpoint)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved results to {args.out_json}")


def run_oracle_decode_bridge(args) -> None:
    if args.task != "arithmetic":
        raise ValueError("--oracle-decode-bridge only supports --task arithmetic")
    if not args.model:
        raise ValueError("--model is required for --oracle-decode-bridge")
    device = torch.device(args.hf_device)
    dtype = _hf_dtype(args.hf_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    num_digits = args.pure_digit_num_digits
    if num_digits <= 0:
        num_digits = _infer_pure_digit_num_digits(
            args.pure_arith_a_max,
            args.pure_arith_b_max,
            args.pure_arith_c_max,
            args.pure_arith_d_max,
        )
    sample_features = _oracle_structured_decode_features(train_cases[:1], num_digits, args.pure_arith_c_max, device)
    bridge = OracleStructuredDecodeBridge(
        feature_dim=sample_features.size(-1),
        hidden_size=hidden_size,
        latent_steps=args.latent_steps,
        bridge_hidden_size=args.decode_bridge_hidden_size,
        depth=args.decode_bridge_depth,
    )
    t0 = time.time()
    print("Training oracle structured-state decode bridge...", flush=True)
    history = train_oracle_decode_bridge(
        model,
        tokenizer,
        bridge,
        train_cases,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        num_digits=num_digits,
        c_max=args.pure_arith_c_max,
    )
    eval_result = None
    if not args.skip_eval:
        print("Evaluating oracle structured-state decode bridge...", flush=True)
        eval_result = evaluate_oracle_decode_bridge(
            model,
            tokenizer,
            bridge,
            eval_cases,
            max_new_tokens=args.max_new_tokens,
            device=device,
            num_digits=num_digits,
            c_max=args.pure_arith_c_max,
        )
        print(
            f"oracle_bridge_eval_acc={eval_result['accuracy'] * 100:.1f}% "
            f"contains_acc={eval_result['contains_accuracy'] * 100:.1f}%",
            flush=True,
        )
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save({
        "bridge": bridge.state_dict(),
        "hidden_size": hidden_size,
        "feature_dim": sample_features.size(-1),
        "num_digits": num_digits,
        "args": vars(args),
        "history": history,
        "eval": eval_result,
    }, args.checkpoint)
    result = {
        "args": vars(args),
        "hidden_size": hidden_size,
        "feature_dim": sample_features.size(-1),
        "num_digits": num_digits,
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


def _textual_oracle_suffix(case: DistillCase, mode: str, num_digits: int) -> str:
    if case.task != "arithmetic":
        raise ValueError("textual oracle eval only supports arithmetic cases")
    a = int(case.meta["a"])
    b = int(case.meta["b"])
    c = int(case.meta["c"])
    d = int(case.meta["d"])
    sum_value = a + b
    product_value = sum_value * c
    final_value = product_value - d
    if mode == "final_answer":
        return f"\nOracle computed final integer: {final_value}\nFinal integer:"
    if mode == "msd_digits":
        final_abs = abs(final_value)
        final_digits_msd = list(str(final_abs))
        sign = "negative" if final_value < 0 else "positive"
        return (
            "\nOracle structured state:\n"
            f"final_sign={sign}\n"
            f"final_digits_msd={','.join(final_digits_msd)}\n"
            "The final_digits_msd list is most-significant digit first; concatenate the digits.\n"
            "Final integer:"
        )
    if mode == "final_abs":
        final_abs = abs(final_value)
        sign = "negative" if final_value < 0 else "positive"
        return (
            "\nOracle structured state:\n"
            f"final_sign={sign}\n"
            f"final_abs={final_abs}\n"
            "Use final_abs as the magnitude and apply final_sign.\n"
            "Final integer:"
        )
    if mode == "natural_summary":
        return f"\nOracle summary: The computed final integer is {final_value}.\nFinal integer:"
    if mode == "digit_state":
        final_abs = abs(final_value)
        final_digits_lsd = [(final_abs // (10 ** pos)) % 10 for pos in range(num_digits)]
        sign = "negative" if final_value < 0 else "positive"
        return (
            "\nOracle structured state:\n"
            f"sum_digits_lsd={','.join(str((sum_value // (10 ** pos)) % 10) for pos in range(num_digits))}\n"
            f"product_digits_lsd={','.join(str((product_value // (10 ** pos)) % 10) for pos in range(num_digits))}\n"
            f"final_sign={sign}\n"
            f"final_digits_lsd={','.join(str(x) for x in final_digits_lsd)}\n"
            "The final_digits_lsd list is least-significant digit first; reverse it and drop leading zeros.\n"
            "Final integer:"
        )
    raise ValueError(f"unknown textual oracle mode: {mode}")


def _final_digit_slots_from_case(case: DistillCase, num_digits: int) -> dict:
    if case.task != "arithmetic":
        raise ValueError("digit slot renderer only supports arithmetic cases")
    a = int(case.meta["a"])
    b = int(case.meta["b"])
    c = int(case.meta["c"])
    d = int(case.meta["d"])
    final_value = (a + b) * c - d
    final_abs = abs(final_value)
    digits_lsd = [(final_abs // (10 ** pos)) % 10 for pos in range(num_digits)]
    digits_msd = list(reversed(digits_lsd))
    rendered_abs = "".join(str(x) for x in digits_msd).lstrip("0") or "0"
    sign = "negative" if final_value < 0 else "positive"
    return {
        "final_value": final_value,
        "final_sign": sign,
        "final_digits_lsd": digits_lsd,
        "final_digits_msd": digits_msd,
        "rendered_abs": rendered_abs,
    }


def _render_digit_slot_suffix(final_sign: str, final_digits_msd: list[int]) -> tuple[str, str]:
    rendered_abs = "".join(str(int(x)) for x in final_digits_msd).lstrip("0") or "0"
    suffix = (
        "\nRendered structured digit slots:\n"
        f"final_sign={final_sign}\n"
        f"final_abs={rendered_abs}\n"
        "The final_abs field was deterministically rendered from explicit MSD digit slots.\n"
        "Use final_abs as the magnitude and apply final_sign.\n"
        "Final integer:"
    )
    return suffix, rendered_abs


def _digit_slot_renderer_suffix(case: DistillCase, num_digits: int) -> tuple[str, dict]:
    slots = _final_digit_slots_from_case(case, num_digits)
    suffix, _ = _render_digit_slot_suffix(slots["final_sign"], slots["final_digits_msd"])
    return suffix, slots


class DigitSlotPredictor(nn.Module):
    """Predict explicit final sign and fixed-width MSD digit slots from structured features."""

    def __init__(self, feature_dim: int, num_digits: int, hidden_size: int, depth: int):
        super().__init__()
        if feature_dim < 1 or num_digits < 1 or hidden_size < 1:
            raise ValueError("feature_dim, num_digits, and hidden_size must be positive")
        self.feature_dim = feature_dim
        self.num_digits = num_digits
        layers: list[nn.Module] = []
        cur = feature_dim
        for _ in range(max(1, depth)):
            layers.append(nn.Linear(cur, hidden_size))
            layers.append(nn.GELU())
            cur = hidden_size
        self.net = nn.Sequential(*layers)
        self.sign_head = nn.Linear(cur, 2)
        self.digit_head = nn.Linear(cur, num_digits * 10)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(features.float())
        sign_logits = self.sign_head(h)
        digit_logits = self.digit_head(h).view(features.size(0), self.num_digits, 10)
        return sign_logits, digit_logits


def _digit_slot_targets(cases: list[DistillCase], num_digits: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sign_targets = []
    digit_targets = []
    for case in cases:
        slots = _final_digit_slots_from_case(case, num_digits)
        sign_targets.append(1 if slots["final_sign"] == "negative" else 0)
        digit_targets.append([int(x) for x in slots["final_digits_msd"]])
    return (
        torch.tensor(sign_targets, device=device, dtype=torch.long),
        torch.tensor(digit_targets, device=device, dtype=torch.long),
    )


def _digit_slot_loss_and_metrics(
    sign_logits: torch.Tensor,
    digit_logits: torch.Tensor,
    sign_targets: torch.Tensor,
    digit_targets: torch.Tensor,
    digit_loss_weight: float,
) -> tuple[torch.Tensor, dict]:
    sign_loss = torch.nn.functional.cross_entropy(sign_logits, sign_targets)
    digit_loss = torch.nn.functional.cross_entropy(digit_logits.reshape(-1, 10), digit_targets.reshape(-1))
    loss = sign_loss + digit_loss_weight * digit_loss
    sign_pred = sign_logits.argmax(dim=-1)
    digit_pred = digit_logits.argmax(dim=-1)
    sign_ok = sign_pred == sign_targets
    digit_ok = digit_pred == digit_targets
    exact_ok = sign_ok & digit_ok.all(dim=-1)
    return loss, {
        "loss": float(loss.item()),
        "sign_loss": float(sign_loss.item()),
        "digit_loss": float(digit_loss.item()),
        "sign_accuracy": _safe_div(int(sign_ok.sum().item()), sign_targets.numel()),
        "per_digit_accuracy": _safe_div(int(digit_ok.sum().item()), digit_targets.numel()),
        "exact_slot_accuracy": _safe_div(int(exact_ok.sum().item()), sign_targets.numel()),
        "examples": int(sign_targets.numel()),
    }


def train_digit_slot_predictor(
    predictor: DigitSlotPredictor,
    cases: list[DistillCase],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    num_digits: int,
    c_max: int,
    digit_loss_weight: float,
) -> list[dict]:
    predictor.to(device=device, dtype=torch.float32).train()
    opt = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(1, epochs + 1):
        rng = random.Random(epoch)
        order = list(range(len(cases)))
        rng.shuffle(order)
        totals = {"loss": 0.0, "sign_loss": 0.0, "digit_loss": 0.0, "sign_correct": 0, "digit_correct": 0, "exact_correct": 0, "examples": 0, "digits": 0}
        for start in range(0, len(order), batch_size):
            idxs = order[start:start + batch_size]
            batch_cases = [cases[i] for i in idxs]
            features = _oracle_structured_decode_features(batch_cases, num_digits, c_max, device)
            sign_targets, digit_targets = _digit_slot_targets(batch_cases, num_digits, device)
            sign_logits, digit_logits = predictor(features)
            loss, _ = _digit_slot_loss_and_metrics(sign_logits, digit_logits, sign_targets, digit_targets, digit_loss_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                sign_loss = torch.nn.functional.cross_entropy(sign_logits, sign_targets)
                digit_loss = torch.nn.functional.cross_entropy(digit_logits.reshape(-1, 10), digit_targets.reshape(-1))
                sign_pred = sign_logits.argmax(dim=-1)
                digit_pred = digit_logits.argmax(dim=-1)
                sign_ok = sign_pred == sign_targets
                digit_ok = digit_pred == digit_targets
                exact_ok = sign_ok & digit_ok.all(dim=-1)
                count = int(sign_targets.numel())
                totals["loss"] += float(loss.item()) * count
                totals["sign_loss"] += float(sign_loss.item()) * count
                totals["digit_loss"] += float(digit_loss.item()) * count
                totals["sign_correct"] += int(sign_ok.sum().item())
                totals["digit_correct"] += int(digit_ok.sum().item())
                totals["exact_correct"] += int(exact_ok.sum().item())
                totals["examples"] += count
                totals["digits"] += int(digit_targets.numel())
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / max(1, totals["examples"]),
            "sign_loss": totals["sign_loss"] / max(1, totals["examples"]),
            "digit_loss": totals["digit_loss"] / max(1, totals["examples"]),
            "sign_accuracy": _safe_div(totals["sign_correct"], totals["examples"]),
            "per_digit_accuracy": _safe_div(totals["digit_correct"], totals["digits"]),
            "exact_slot_accuracy": _safe_div(totals["exact_correct"], totals["examples"]),
            "examples": totals["examples"],
        }
        history.append(row)
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            print(
                f"  digit_slot_epoch={epoch:03d} loss={row['loss']:.6f} "
                f"sign={row['sign_accuracy'] * 100:.1f}% digit={row['per_digit_accuracy'] * 100:.1f}% "
                f"exact={row['exact_slot_accuracy'] * 100:.1f}%",
                flush=True,
            )
    return history


@torch.no_grad()
def _predict_digit_slots_for_case(
    predictor: DigitSlotPredictor,
    case: DistillCase,
    num_digits: int,
    c_max: int,
    device: torch.device,
) -> dict:
    features = _oracle_structured_decode_features([case], num_digits, c_max, device)
    sign_logits, digit_logits = predictor(features)
    sign_id = int(sign_logits.argmax(dim=-1).item())
    digits_msd = [int(x) for x in digit_logits.argmax(dim=-1).squeeze(0).tolist()]
    final_sign = "negative" if sign_id == 1 else "positive"
    suffix, rendered_abs = _render_digit_slot_suffix(final_sign, digits_msd)
    return {"final_sign": final_sign, "final_digits_msd": digits_msd, "rendered_abs": rendered_abs, "suffix": suffix}


@torch.no_grad()
def generate_hf_with_digit_slot_predictor(
    model,
    tokenizer,
    predictor: DigitSlotPredictor,
    case: DistillCase,
    max_new_tokens: int,
    device: torch.device,
    num_digits: int,
    c_max: int,
    input_mode: str,
) -> tuple[str, dict]:
    model.eval()
    predictor.eval()
    predicted = _predict_digit_slots_for_case(predictor, case, num_digits, c_max, device)
    prompt_ids = torch.tensor(tokenizer.encode(case.prompt, add_special_tokens=False), device=device, dtype=torch.long).unsqueeze(0)
    suffix_ids = torch.tensor(tokenizer.encode(predicted["suffix"], add_special_tokens=False), device=device, dtype=torch.long).unsqueeze(0)
    generated = []
    past = None
    if input_mode == "input_ids":
        input_ids = torch.cat([prompt_ids, suffix_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)
        first_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
    elif input_mode == "inputs_embeds":
        embed = model.get_input_embeddings()
        inputs_embeds = torch.cat([embed(prompt_ids).detach(), embed(suffix_ids).detach()], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)
        first_kwargs = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
    else:
        raise ValueError(f"unknown digit slot predictor decode input mode: {input_mode}")
    for step in range(max_new_tokens):
        if step == 0:
            out = model(**first_kwargs, use_cache=True, return_dict=True)
        else:
            out = model(input_ids=next_id, past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        token = int(next_id.item())
        generated.append(token)
        if tokenizer.eos_token_id is not None and token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated, skip_special_tokens=True), predicted


@torch.no_grad()
def evaluate_digit_slot_predictor(
    predictor: DigitSlotPredictor,
    cases: list[DistillCase],
    batch_size: int,
    device: torch.device,
    num_digits: int,
    c_max: int,
    digit_loss_weight: float,
) -> dict:
    predictor.eval()
    totals = {"loss": 0.0, "sign_loss": 0.0, "digit_loss": 0.0, "sign_correct": 0, "digit_correct": 0, "exact_correct": 0, "examples": 0, "digits": 0}
    details = []
    for start in range(0, len(cases), batch_size):
        batch_cases = cases[start:start + batch_size]
        features = _oracle_structured_decode_features(batch_cases, num_digits, c_max, device)
        sign_targets, digit_targets = _digit_slot_targets(batch_cases, num_digits, device)
        sign_logits, digit_logits = predictor(features)
        loss, _ = _digit_slot_loss_and_metrics(sign_logits, digit_logits, sign_targets, digit_targets, digit_loss_weight)
        sign_loss = torch.nn.functional.cross_entropy(sign_logits, sign_targets)
        digit_loss = torch.nn.functional.cross_entropy(digit_logits.reshape(-1, 10), digit_targets.reshape(-1))
        sign_pred = sign_logits.argmax(dim=-1)
        digit_pred = digit_logits.argmax(dim=-1)
        sign_ok = sign_pred == sign_targets
        digit_ok = digit_pred == digit_targets
        exact_ok = sign_ok & digit_ok.all(dim=-1)
        count = int(sign_targets.numel())
        totals["loss"] += float(loss.item()) * count
        totals["sign_loss"] += float(sign_loss.item()) * count
        totals["digit_loss"] += float(digit_loss.item()) * count
        totals["sign_correct"] += int(sign_ok.sum().item())
        totals["digit_correct"] += int(digit_ok.sum().item())
        totals["exact_correct"] += int(exact_ok.sum().item())
        totals["examples"] += count
        totals["digits"] += int(digit_targets.numel())
        for i, case in enumerate(batch_cases):
            target_slots = _final_digit_slots_from_case(case, num_digits)
            pred_sign = "negative" if int(sign_pred[i].item()) == 1 else "positive"
            pred_digits = [int(x) for x in digit_pred[i].tolist()]
            _, pred_abs = _render_digit_slot_suffix(pred_sign, pred_digits)
            details.append({
                "expected": case.expected,
                "target_sign": target_slots["final_sign"],
                "pred_sign": pred_sign,
                "target_digits_msd": target_slots["final_digits_msd"],
                "pred_digits_msd": pred_digits,
                "target_abs": target_slots["rendered_abs"],
                "pred_abs": pred_abs,
                "sign_hit": bool(sign_ok[i].item()),
                "digits_hit": bool(digit_ok[i].all().item()),
                "exact_hit": bool(exact_ok[i].item()),
                "meta": case.meta,
            })
    return {
        "loss": totals["loss"] / max(1, totals["examples"]),
        "sign_loss": totals["sign_loss"] / max(1, totals["examples"]),
        "digit_loss": totals["digit_loss"] / max(1, totals["examples"]),
        "sign_accuracy": _safe_div(totals["sign_correct"], totals["examples"]),
        "per_digit_accuracy": _safe_div(totals["digit_correct"], totals["digits"]),
        "exact_slot_accuracy": _safe_div(totals["exact_correct"], totals["examples"]),
        "examples": totals["examples"],
        "details": details,
    }


@torch.no_grad()
def evaluate_digit_slot_predictor_decode(
    model,
    tokenizer,
    predictor: DigitSlotPredictor,
    cases: list[DistillCase],
    max_new_tokens: int,
    device: torch.device,
    num_digits: int,
    c_max: int,
    input_mode: str,
) -> dict:
    details = []
    for case in cases:
        text, predicted = generate_hf_with_digit_slot_predictor(
            model, tokenizer, predictor, case, max_new_tokens, device, num_digits, c_max, input_mode
        )
        answer_only = extract_answer_only(case.task, text)
        contains_expected = contains_expected_answer(case.task, text, case.expected)
        hit = answer_only == case.expected
        target_slots = _final_digit_slots_from_case(case, num_digits)
        slot_hit = predicted["final_sign"] == target_slots["final_sign"] and predicted["final_digits_msd"] == target_slots["final_digits_msd"]
        details.append({
            "expected": case.expected,
            "answer_only": answer_only,
            "contains_expected": contains_expected,
            "decode_hit": hit,
            "slot_hit": slot_hit,
            "raw": text[:160],
            "input_mode": input_mode,
            "predicted": {k: v for k, v in predicted.items() if k != "suffix"},
            "target_slots": target_slots,
            "meta": case.meta,
        })
        print(
            f"  digit_slot_predictor_decode mode={input_mode} expected={case.expected} answer_only={answer_only} "
            f"slot_hit={slot_hit} decode_hit={hit} pred_abs={predicted['rendered_abs']} raw={text[:48]!r}",
            flush=True,
        )
    acc = sum(int(x["decode_hit"]) for x in details) / max(1, len(details))
    contains_acc = sum(int(x["contains_expected"]) for x in details) / max(1, len(details))
    slot_acc = sum(int(x["slot_hit"]) for x in details) / max(1, len(details))
    return {"accuracy": acc, "contains_accuracy": contains_acc, "slot_exact_accuracy": slot_acc, "input_mode": input_mode, "details": details}


@torch.no_grad()
def generate_hf_digit_slot_renderer(
    model,
    tokenizer,
    case: DistillCase,
    max_new_tokens: int,
    device: torch.device,
    num_digits: int,
    input_mode: str,
) -> str:
    model.eval()
    suffix, _ = _digit_slot_renderer_suffix(case, num_digits)
    prompt_ids = torch.tensor(tokenizer.encode(case.prompt, add_special_tokens=False), device=device, dtype=torch.long).unsqueeze(0)
    suffix_ids = torch.tensor(tokenizer.encode(suffix, add_special_tokens=False), device=device, dtype=torch.long).unsqueeze(0)
    generated = []
    past = None
    if input_mode == "input_ids":
        input_ids = torch.cat([prompt_ids, suffix_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)
        first_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
    elif input_mode == "inputs_embeds":
        embed = model.get_input_embeddings()
        inputs_embeds = torch.cat([embed(prompt_ids).detach(), embed(suffix_ids).detach()], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)
        first_kwargs = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
    else:
        raise ValueError(f"unknown digit slot renderer input mode: {input_mode}")
    for step in range(max_new_tokens):
        if step == 0:
            out = model(**first_kwargs, use_cache=True, return_dict=True)
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
def evaluate_digit_slot_renderer(
    model,
    tokenizer,
    cases: list[DistillCase],
    max_new_tokens: int,
    device: torch.device,
    num_digits: int,
    input_mode: str,
) -> dict:
    details = []
    for case in cases:
        text = generate_hf_digit_slot_renderer(model, tokenizer, case, max_new_tokens, device, num_digits, input_mode)
        answer_only = extract_answer_only(case.task, text)
        contains_expected = contains_expected_answer(case.task, text, case.expected)
        hit = answer_only == case.expected
        _, slots = _digit_slot_renderer_suffix(case, num_digits)
        details.append({
            "expected": case.expected,
            "answer_only": answer_only,
            "contains_expected": contains_expected,
            "hit": hit,
            "raw": text[:160],
            "input_mode": input_mode,
            "slots": slots,
            "meta": case.meta,
        })
        print(
            f"  digit_slot_renderer_eval mode={input_mode} expected={case.expected} answer_only={answer_only} "
            f"contains={contains_expected} hit={hit} rendered_abs={slots['rendered_abs']} raw={text[:48]!r}",
            flush=True,
        )
    acc = sum(int(x["hit"]) for x in details) / max(1, len(details))
    contains_acc = sum(int(x["contains_expected"]) for x in details) / max(1, len(details))
    return {"accuracy": acc, "contains_accuracy": contains_acc, "input_mode": input_mode, "details": details}


def run_digit_slot_renderer_eval(args) -> None:
    if args.task != "arithmetic":
        raise ValueError("--digit-slot-renderer-eval only supports --task arithmetic")
    if not args.model:
        raise ValueError("--model is required for --digit-slot-renderer-eval")
    device = torch.device(args.hf_device)
    dtype = _hf_dtype(args.hf_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    _patch_torch_custom_op_string_annotations()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    eval_cases = build_distill_cases(args.task, tokenizer, args.eval_cases, args.context_len, args.depth, args.seed + 10_000)
    num_digits = args.pure_digit_num_digits
    if num_digits <= 0:
        num_digits = _infer_pure_digit_num_digits(
            args.pure_arith_a_max,
            args.pure_arith_b_max,
            args.pure_arith_c_max,
            args.pure_arith_d_max,
        )
    t0 = time.time()
    input_modes = ["input_ids", "inputs_embeds"] if args.digit_slot_renderer_input_mode == "both" else [args.digit_slot_renderer_input_mode]
    eval_result = {}
    for input_mode in input_modes:
        print(f"Evaluating digit-slot deterministic renderer input_mode={input_mode}...", flush=True)
        mode_result = evaluate_digit_slot_renderer(
            model,
            tokenizer,
            eval_cases,
            max_new_tokens=args.max_new_tokens,
            device=device,
            num_digits=num_digits,
            input_mode=input_mode,
        )
        eval_result[input_mode] = mode_result
        print(
            f"digit_slot_renderer_mode={input_mode} acc={mode_result['accuracy'] * 100:.1f}% "
            f"contains_acc={mode_result['contains_accuracy'] * 100:.1f}%",
            flush=True,
        )
    result = {
        "args": vars(args),
        "num_digits": num_digits,
        "eval_cases": len(eval_cases),
        "eval": eval_result if args.digit_slot_renderer_input_mode == "both" else eval_result[input_modes[0]],
        "elapsed_s": time.time() - t0,
    }
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save(result, args.checkpoint)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved results to {args.out_json}")


def run_digit_slot_predictor(args) -> None:
    if args.task != "arithmetic":
        raise ValueError("--digit-slot-predictor only supports --task arithmetic")
    if not args.model:
        raise ValueError("--model is required for --digit-slot-predictor")
    device = torch.device(args.hf_device)
    dtype = _hf_dtype(args.hf_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    _patch_torch_custom_op_string_annotations()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    train_cases = build_distill_cases(args.task, tokenizer, args.train_cases, args.context_len, args.depth, args.seed)
    eval_cases = build_distill_cases(args.task, tokenizer, args.eval_cases, args.context_len, args.depth, args.seed + 10_000)
    num_digits = args.pure_digit_num_digits
    if num_digits <= 0:
        num_digits = _infer_pure_digit_num_digits(
            args.pure_arith_a_max,
            args.pure_arith_b_max,
            args.pure_arith_c_max,
            args.pure_arith_d_max,
        )
    sample_features = _oracle_structured_decode_features(train_cases[:1], num_digits, args.pure_arith_c_max, device)
    predictor = DigitSlotPredictor(
        feature_dim=sample_features.size(-1),
        num_digits=num_digits,
        hidden_size=args.digit_slot_predictor_hidden_size,
        depth=args.digit_slot_predictor_depth,
    )
    t0 = time.time()
    print(
        f"Training digit-slot predictor num_digits={num_digits} feature_dim={sample_features.size(-1)}...",
        flush=True,
    )
    history = train_digit_slot_predictor(
        predictor,
        train_cases,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        num_digits=num_digits,
        c_max=args.pure_arith_c_max,
        digit_loss_weight=args.digit_slot_digit_loss_weight,
    )
    train_slots = evaluate_digit_slot_predictor(
        predictor, train_cases, args.batch_size, device, num_digits, args.pure_arith_c_max, args.digit_slot_digit_loss_weight
    )
    eval_slots = evaluate_digit_slot_predictor(
        predictor, eval_cases, args.batch_size, device, num_digits, args.pure_arith_c_max, args.digit_slot_digit_loss_weight
    )
    eval_decode = None
    if not args.skip_eval:
        decode_modes = ["input_ids", "inputs_embeds"] if args.digit_slot_decode_input_mode == "both" else [args.digit_slot_decode_input_mode]
        eval_decode = {}
        for input_mode in decode_modes:
            print(f"Evaluating digit-slot predictor decode input_mode={input_mode}...", flush=True)
            mode_result = evaluate_digit_slot_predictor_decode(
                model,
                tokenizer,
                predictor,
                eval_cases,
                max_new_tokens=args.max_new_tokens,
                device=device,
                num_digits=num_digits,
                c_max=args.pure_arith_c_max,
                input_mode=input_mode,
            )
            eval_decode[input_mode] = mode_result
            print(
                f"digit_slot_predictor_decode_mode={input_mode} slot={mode_result['slot_exact_accuracy'] * 100:.1f}% "
                f"decode={mode_result['accuracy'] * 100:.1f}% contains={mode_result['contains_accuracy'] * 100:.1f}%",
                flush=True,
            )
        if args.digit_slot_decode_input_mode != "both":
            eval_decode = eval_decode[decode_modes[0]]
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save({
        "predictor": predictor.state_dict(),
        "feature_dim": sample_features.size(-1),
        "num_digits": num_digits,
        "args": vars(args),
        "history": history,
        "train_slots": train_slots,
        "eval_slots": eval_slots,
        "eval_decode": eval_decode,
    }, args.checkpoint)
    result = {
        "args": vars(args),
        "feature_dim": sample_features.size(-1),
        "num_digits": num_digits,
        "history": history,
        "train_slots": train_slots,
        "eval_slots": eval_slots,
        "eval_decode": eval_decode,
        "elapsed_s": time.time() - t0,
        "checkpoint": args.checkpoint,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved results to {args.out_json}")


@torch.no_grad()
def _primitive_digit_operator_slots_from_case(
    operator,
    case: DistillCase,
    num_digits: int,
    device: torch.device,
) -> dict:
    if case.task != "arithmetic":
        raise ValueError("primitive digit renderer only supports arithmetic cases")
    items = _arithmetic_cases_to_tuples([case])
    trace = _build_trainable_digit_arithmetic_trace(operator, items, num_digits, device)
    digits_lsd = [int(x) for x in trace.final_digits[0].tolist()]
    digits_msd = list(reversed(digits_lsd))
    final_sign = "negative" if int(trace.final_sign[0].item()) == 1 else "positive"
    suffix, rendered_abs = _render_digit_slot_suffix(final_sign, digits_msd)
    return {
        "final_sign": final_sign,
        "final_digits_lsd": digits_lsd,
        "final_digits_msd": digits_msd,
        "rendered_abs": rendered_abs,
        "overflow": bool(trace.overflow_mask[0].item()),
        "suffix": suffix,
    }


@torch.no_grad()
def generate_hf_with_primitive_digit_renderer(
    model,
    tokenizer,
    operator,
    case: DistillCase,
    max_new_tokens: int,
    device: torch.device,
    num_digits: int,
    input_mode: str,
) -> tuple[str, dict]:
    model.eval()
    operator.eval()
    predicted = _primitive_digit_operator_slots_from_case(operator, case, num_digits, device)
    prompt_ids = torch.tensor(tokenizer.encode(case.prompt, add_special_tokens=False), device=device, dtype=torch.long).unsqueeze(0)
    suffix_ids = torch.tensor(tokenizer.encode(predicted["suffix"], add_special_tokens=False), device=device, dtype=torch.long).unsqueeze(0)
    generated = []
    past = None
    if input_mode == "input_ids":
        input_ids = torch.cat([prompt_ids, suffix_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)
        first_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
    elif input_mode == "inputs_embeds":
        embed = model.get_input_embeddings()
        inputs_embeds = torch.cat([embed(prompt_ids).detach(), embed(suffix_ids).detach()], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)
        first_kwargs = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
    else:
        raise ValueError(f"unknown primitive digit renderer input mode: {input_mode}")
    for step in range(max_new_tokens):
        if step == 0:
            out = model(**first_kwargs, use_cache=True, return_dict=True)
        else:
            out = model(input_ids=next_id, past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        token = int(next_id.item())
        generated.append(token)
        if tokenizer.eos_token_id is not None and token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated, skip_special_tokens=True), predicted


def _sync_device_for_timing(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@torch.no_grad()
def _generate_hf_greedy_timed(
    model,
    tokenizer,
    first_kwargs: dict,
    max_new_tokens: int,
    device: torch.device,
    expected: str | None = None,
) -> tuple[str, list[int], float]:
    model.eval()
    generated: list[int] = []
    past = None
    next_id = None
    _sync_device_for_timing(device)
    t0 = time.time()
    for step in range(max_new_tokens):
        if step == 0:
            out = model(**first_kwargs, use_cache=True, return_dict=True)
        else:
            out = model(input_ids=next_id, past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        token = int(next_id.item())
        generated.append(token)
        if tokenizer.eos_token_id is not None and token == tokenizer.eos_token_id:
            break
        if expected is not None:
            partial = tokenizer.decode(generated, skip_special_tokens=True)
            answer_only = extract_answer_only("arithmetic", partial)
            if answer_only == expected:
                break
            final_answer = extract_distill_answer("arithmetic", partial)
            if final_answer == expected:
                break
            m = re.match(r"\s*(-?\d+)(.*)", partial, flags=re.DOTALL)
            if m and m.group(2) and not m.group(2)[0].isdigit():
                break
    _sync_device_for_timing(device)
    return tokenizer.decode(generated, skip_special_tokens=True), generated, time.time() - t0


def _hf_input_kwargs_from_token_ids(input_ids: torch.Tensor) -> dict:
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }


def _hf_input_kwargs_from_embeds(model, input_ids: torch.Tensor, device: torch.device) -> dict:
    embed = model.get_input_embeddings()
    inputs_embeds = embed(input_ids).detach()
    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long),
    }


@torch.no_grad()
def evaluate_operator_vm_benchmark(
    model,
    tokenizer,
    operator,
    cases: list[DistillCase],
    baseline_max_new_tokens: int,
    final_max_new_tokens: int,
    device: torch.device,
    num_digits: int,
    input_mode: str,
) -> dict:
    details = []
    baseline_generated_tokens = 0
    baseline_latency_s = 0.0
    baseline_prefill_tokens = 0
    operator_generated_tokens = 0
    operator_latency_s = 0.0
    operator_llm_latency_s = 0.0
    operator_overhead_s = 0.0
    operator_prefill_tokens = 0
    operator_suffix_tokens = 0
    for case in cases:
        prompt_ids = torch.tensor(
            tokenizer.encode(case.prompt, add_special_tokens=False),
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        baseline_kwargs = _hf_input_kwargs_from_token_ids(prompt_ids)
        baseline_text, baseline_ids, baseline_dt = _generate_hf_greedy_timed(
            model,
            tokenizer,
            baseline_kwargs,
            baseline_max_new_tokens,
            device,
            expected=case.expected,
        )
        baseline_answer_only = extract_answer_only(case.task, baseline_text)
        baseline_answer = extract_distill_answer(case.task, baseline_text)
        baseline_hit = baseline_answer == case.expected
        baseline_contains = contains_expected_answer(case.task, baseline_text, case.expected)

        _sync_device_for_timing(device)
        op_t0 = time.time()
        predicted = _primitive_digit_operator_slots_from_case(operator, case, num_digits, device)
        _sync_device_for_timing(device)
        op_overhead = time.time() - op_t0

        suffix_ids = torch.tensor(
            tokenizer.encode(predicted["suffix"], add_special_tokens=False),
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        op_input_ids = torch.cat([prompt_ids, suffix_ids], dim=1)
        if input_mode == "input_ids":
            op_kwargs = _hf_input_kwargs_from_token_ids(op_input_ids)
        elif input_mode == "inputs_embeds":
            op_kwargs = _hf_input_kwargs_from_embeds(model, op_input_ids, device)
        else:
            raise ValueError(f"unknown operator VM input mode: {input_mode}")
        op_text, op_ids, op_llm_dt = _generate_hf_greedy_timed(
            model,
            tokenizer,
            op_kwargs,
            final_max_new_tokens,
            device,
            expected=case.expected,
        )
        op_answer_only = extract_answer_only(case.task, op_text)
        op_answer = extract_distill_answer(case.task, op_text)
        op_hit = op_answer == case.expected
        op_contains = contains_expected_answer(case.task, op_text, case.expected)
        op_dt = op_overhead + op_llm_dt

        baseline_generated_tokens += len(baseline_ids)
        baseline_latency_s += baseline_dt
        baseline_prefill_tokens += int(prompt_ids.numel())
        operator_generated_tokens += len(op_ids)
        operator_latency_s += op_dt
        operator_llm_latency_s += op_llm_dt
        operator_overhead_s += op_overhead
        operator_prefill_tokens += int(op_input_ids.numel())
        operator_suffix_tokens += int(suffix_ids.numel())

        details.append({
            "expected": case.expected,
            "baseline": {
                "answer_only": baseline_answer_only,
                "answer": baseline_answer,
                "hit": baseline_hit,
                "contains_expected": baseline_contains,
                "generated_tokens": len(baseline_ids),
                "prefill_tokens": int(prompt_ids.numel()),
                "total_kv_tokens": int(prompt_ids.numel()) + len(baseline_ids),
                "latency_s": baseline_dt,
                "raw": baseline_text[:160],
            },
            "operator_vm": {
                "answer_only": op_answer_only,
                "answer": op_answer,
                "hit": op_hit,
                "contains_expected": op_contains,
                "generated_tokens": len(op_ids),
                "prefill_tokens": int(op_input_ids.numel()),
                "rendered_suffix_tokens": int(suffix_ids.numel()),
                "total_kv_tokens": int(op_input_ids.numel()) + len(op_ids),
                "latency_s": op_dt,
                "llm_latency_s": op_llm_dt,
                "operator_overhead_s": op_overhead,
                "predicted": {k: v for k, v in predicted.items() if k != "suffix"},
                "raw": op_text[:160],
            },
            "meta": case.meta,
        })
        print(
            f"  c1 expected={case.expected} baseline={baseline_answer} b_tok={len(baseline_ids)} "
            f"op={op_answer} op_tok={len(op_ids)} op_overhead={op_overhead:.4f}s",
            flush=True,
        )

    n = max(1, len(details))
    baseline = {
        "accuracy": sum(int(x["baseline"]["hit"]) for x in details) / n,
        "contains_accuracy": sum(int(x["baseline"]["contains_expected"]) for x in details) / n,
        "avg_generated_tokens": baseline_generated_tokens / n,
        "avg_latency_s": baseline_latency_s / n,
        "avg_prefill_tokens": baseline_prefill_tokens / n,
        "avg_total_kv_tokens": (baseline_prefill_tokens + baseline_generated_tokens) / n,
    }
    operator_vm = {
        "accuracy": sum(int(x["operator_vm"]["hit"]) for x in details) / n,
        "contains_accuracy": sum(int(x["operator_vm"]["contains_expected"]) for x in details) / n,
        "avg_generated_tokens": operator_generated_tokens / n,
        "avg_latency_s": operator_latency_s / n,
        "avg_llm_latency_s": operator_llm_latency_s / n,
        "operator_overhead_s": operator_overhead_s / n,
        "avg_prefill_tokens": operator_prefill_tokens / n,
        "avg_rendered_suffix_tokens": operator_suffix_tokens / n,
        "avg_total_kv_tokens": (operator_prefill_tokens + operator_generated_tokens) / n,
    }
    speedup = {
        "latency": baseline["avg_latency_s"] / operator_vm["avg_latency_s"] if operator_vm["avg_latency_s"] > 0 else 0.0,
        "llm_latency": baseline["avg_latency_s"] / operator_vm["avg_llm_latency_s"] if operator_vm["avg_llm_latency_s"] > 0 else 0.0,
        "generated_tokens": baseline["avg_generated_tokens"] / operator_vm["avg_generated_tokens"] if operator_vm["avg_generated_tokens"] > 0 else 0.0,
        "total_kv_tokens": baseline["avg_total_kv_tokens"] / operator_vm["avg_total_kv_tokens"] if operator_vm["avg_total_kv_tokens"] > 0 else 0.0,
    }
    token_saving = {
        "generated_token_reduction": 1.0 - (operator_vm["avg_generated_tokens"] / baseline["avg_generated_tokens"]) if baseline["avg_generated_tokens"] > 0 else 0.0,
        "total_kv_token_reduction": 1.0 - (operator_vm["avg_total_kv_tokens"] / baseline["avg_total_kv_tokens"]) if baseline["avg_total_kv_tokens"] > 0 else 0.0,
    }
    return {
        "baseline": baseline,
        "operator_vm": operator_vm,
        "speedup": speedup,
        "token_saving": token_saving,
        "details": details,
    }


def run_operator_vm_benchmark(args) -> None:
    if args.task != "arithmetic":
        raise ValueError("--operator-vm-benchmark only supports --task arithmetic")
    if not args.model:
        raise ValueError("--model is required for --operator-vm-benchmark")
    device = torch.device(args.hf_device)
    dtype = _hf_dtype(args.hf_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    _patch_torch_custom_op_string_annotations()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    operator = PrimitiveLookupDigitWiseOperator(
        c_min=args.pure_arith_c_min,
        c_max=args.pure_arith_c_max,
    ).to(device)
    eval_cases = build_distill_cases(args.task, tokenizer, args.eval_cases, args.context_len, args.depth, args.seed + 10_000)
    num_digits = args.pure_digit_num_digits
    if num_digits <= 0:
        num_digits = _infer_pure_digit_num_digits(
            args.pure_arith_a_max,
            args.pure_arith_b_max,
            args.pure_arith_c_max,
            args.pure_arith_d_max,
        )
    baseline_max_new_tokens = args.operator_vm_baseline_max_new_tokens or args.max_new_tokens
    final_max_new_tokens = args.operator_vm_final_max_new_tokens or args.max_new_tokens
    t0 = time.time()
    print("Running C1 operator VM benchmark...", flush=True)
    eval_result = evaluate_operator_vm_benchmark(
        model,
        tokenizer,
        operator,
        eval_cases,
        baseline_max_new_tokens=baseline_max_new_tokens,
        final_max_new_tokens=final_max_new_tokens,
        device=device,
        num_digits=num_digits,
        input_mode=args.operator_vm_input_mode,
    )
    result = {
        "args": vars(args),
        "benchmark": "C1 arithmetic operator VM token-saving benchmark",
        "eval_cases": len(eval_cases),
        "num_digits": num_digits,
        "baseline_max_new_tokens": baseline_max_new_tokens,
        "operator_vm_final_max_new_tokens": final_max_new_tokens,
        "input_mode": args.operator_vm_input_mode,
        "elapsed_s": time.time() - t0,
    }
    result.update(eval_result)
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save(result, args.checkpoint)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"c1 baseline_acc={result['baseline']['accuracy'] * 100:.1f}% "
        f"op_acc={result['operator_vm']['accuracy'] * 100:.1f}% "
        f"latency_speedup={result['speedup']['latency']:.2f}x "
        f"generated_token_speedup={result['speedup']['generated_tokens']:.2f}x",
        flush=True,
    )
    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved results to {args.out_json}")


def _signed_rendered_answer(predicted: dict) -> str:
    rendered_abs = str(predicted["rendered_abs"])
    if predicted["final_sign"] == "negative" and rendered_abs != "0":
        return f"-{rendered_abs}"
    return rendered_abs


def _minimal_embedding_suffix(predicted: dict, mode: str) -> str:
    signed_answer = _signed_rendered_answer(predicted)
    if mode == "final_abs":
        return f"\nfinal_abs={predicted['rendered_abs']}\nFinal integer:"
    if mode == "sign_final_abs":
        return f"\nfinal_sign={predicted['final_sign']}\nfinal_abs={predicted['rendered_abs']}\nFinal integer:"
    if mode == "signed_answer":
        return f"\nanswer={signed_answer}\nFinal integer:"
    if mode == "digits_msd":
        digits = "".join(str(int(x)) for x in predicted["final_digits_msd"])
        return f"\nsign={predicted['final_sign']}\ndigits_msd={digits}\nFinal integer:"
    raise ValueError(f"unknown minimal embedding suffix mode: {mode}")


def _empty_path_totals() -> dict:
    return {
        "generated_tokens": 0,
        "latency_s": 0.0,
        "llm_latency_s": 0.0,
        "operator_overhead_s": 0.0,
        "prefill_tokens": 0,
        "injected_tokens": 0,
    }


def _summarize_generation_path(details: list[dict], key: str, totals: dict) -> dict:
    n = max(1, len(details))
    result = {
        "accuracy": sum(int(x[key]["hit"]) for x in details) / n,
        "contains_accuracy": sum(int(x[key]["contains_expected"]) for x in details) / n,
        "avg_generated_tokens": totals["generated_tokens"] / n,
        "avg_latency_s": totals["latency_s"] / n,
        "avg_prefill_tokens": totals["prefill_tokens"] / n,
        "avg_total_kv_tokens": (totals["prefill_tokens"] + totals["generated_tokens"]) / n,
    }
    if totals["llm_latency_s"] > 0:
        result["avg_llm_latency_s"] = totals["llm_latency_s"] / n
    if totals["operator_overhead_s"] > 0:
        result["operator_overhead_s"] = totals["operator_overhead_s"] / n
    if totals["injected_tokens"] > 0:
        result["avg_injected_tokens"] = totals["injected_tokens"] / n
    return result


def _speedup_and_saving(reference: dict, candidate: dict) -> dict:
    return {
        "latency": reference["avg_latency_s"] / candidate["avg_latency_s"] if candidate["avg_latency_s"] > 0 else 0.0,
        "generated_tokens": reference["avg_generated_tokens"] / candidate["avg_generated_tokens"] if candidate["avg_generated_tokens"] > 0 else 0.0,
        "total_kv_tokens": reference["avg_total_kv_tokens"] / candidate["avg_total_kv_tokens"] if candidate["avg_total_kv_tokens"] > 0 else 0.0,
        "generated_token_reduction": 1.0 - (candidate["avg_generated_tokens"] / reference["avg_generated_tokens"]) if reference["avg_generated_tokens"] > 0 else 0.0,
        "total_kv_token_reduction": 1.0 - (candidate["avg_total_kv_tokens"] / reference["avg_total_kv_tokens"]) if reference["avg_total_kv_tokens"] > 0 else 0.0,
    }


@torch.no_grad()
def evaluate_minimal_embedding_suffix_benchmark(
    model,
    tokenizer,
    operator,
    cases: list[DistillCase],
    baseline_max_new_tokens: int,
    final_max_new_tokens: int,
    device: torch.device,
    num_digits: int,
    text_input_mode: str,
    minimal_modes: list[str],
) -> dict:
    details = []
    baseline_totals = _empty_path_totals()
    text_totals = _empty_path_totals()
    minimal_totals = {mode: _empty_path_totals() for mode in minimal_modes}
    for case in cases:
        prompt_ids = torch.tensor(
            tokenizer.encode(case.prompt, add_special_tokens=False),
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)

        baseline_text, baseline_ids, baseline_dt = _generate_hf_greedy_timed(
            model,
            tokenizer,
            _hf_input_kwargs_from_token_ids(prompt_ids),
            baseline_max_new_tokens,
            device,
            expected=case.expected,
        )
        baseline_answer_only = extract_answer_only(case.task, baseline_text)
        baseline_answer = extract_distill_answer(case.task, baseline_text)
        baseline_record = {
            "answer_only": baseline_answer_only,
            "answer": baseline_answer,
            "hit": baseline_answer == case.expected,
            "contains_expected": contains_expected_answer(case.task, baseline_text, case.expected),
            "generated_tokens": len(baseline_ids),
            "prefill_tokens": int(prompt_ids.numel()),
            "total_kv_tokens": int(prompt_ids.numel()) + len(baseline_ids),
            "latency_s": baseline_dt,
            "raw": baseline_text[:160],
        }
        baseline_totals["generated_tokens"] += len(baseline_ids)
        baseline_totals["latency_s"] += baseline_dt
        baseline_totals["prefill_tokens"] += int(prompt_ids.numel())

        _sync_device_for_timing(device)
        op_t0 = time.time()
        predicted = _primitive_digit_operator_slots_from_case(operator, case, num_digits, device)
        _sync_device_for_timing(device)
        operator_overhead = time.time() - op_t0

        text_suffix_ids = torch.tensor(
            tokenizer.encode(predicted["suffix"], add_special_tokens=False),
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        text_input_ids = torch.cat([prompt_ids, text_suffix_ids], dim=1)
        if text_input_mode == "input_ids":
            text_kwargs = _hf_input_kwargs_from_token_ids(text_input_ids)
        elif text_input_mode == "inputs_embeds":
            text_kwargs = _hf_input_kwargs_from_embeds(model, text_input_ids, device)
        else:
            raise ValueError(f"unknown operator VM text input mode: {text_input_mode}")
        text_out, text_ids, text_llm_dt = _generate_hf_greedy_timed(
            model,
            tokenizer,
            text_kwargs,
            final_max_new_tokens,
            device,
            expected=case.expected,
        )
        text_answer_only = extract_answer_only(case.task, text_out)
        text_answer = extract_distill_answer(case.task, text_out)
        text_dt = operator_overhead + text_llm_dt
        text_record = {
            "answer_only": text_answer_only,
            "answer": text_answer,
            "hit": text_answer == case.expected,
            "contains_expected": contains_expected_answer(case.task, text_out, case.expected),
            "generated_tokens": len(text_ids),
            "prefill_tokens": int(text_input_ids.numel()),
            "injected_text_tokens": int(text_suffix_ids.numel()),
            "total_kv_tokens": int(text_input_ids.numel()) + len(text_ids),
            "latency_s": text_dt,
            "llm_latency_s": text_llm_dt,
            "operator_overhead_s": operator_overhead,
            "raw": text_out[:160],
        }
        text_totals["generated_tokens"] += len(text_ids)
        text_totals["latency_s"] += text_dt
        text_totals["llm_latency_s"] += text_llm_dt
        text_totals["operator_overhead_s"] += operator_overhead
        text_totals["prefill_tokens"] += int(text_input_ids.numel())
        text_totals["injected_tokens"] += int(text_suffix_ids.numel())

        minimal_records = {}
        for mode in minimal_modes:
            minimal_suffix = _minimal_embedding_suffix(predicted, mode)
            minimal_suffix_ids = torch.tensor(
                tokenizer.encode(minimal_suffix, add_special_tokens=False),
                device=device,
                dtype=torch.long,
            ).unsqueeze(0)
            minimal_input_ids = torch.cat([prompt_ids, minimal_suffix_ids], dim=1)
            minimal_kwargs = _hf_input_kwargs_from_embeds(model, minimal_input_ids, device)
            minimal_out, minimal_ids, minimal_llm_dt = _generate_hf_greedy_timed(
                model,
                tokenizer,
                minimal_kwargs,
                final_max_new_tokens,
                device,
                expected=case.expected,
            )
            minimal_answer_only = extract_answer_only(case.task, minimal_out)
            minimal_answer = extract_distill_answer(case.task, minimal_out)
            minimal_dt = operator_overhead + minimal_llm_dt
            minimal_records[mode] = {
                "answer_only": minimal_answer_only,
                "answer": minimal_answer,
                "hit": minimal_answer == case.expected,
                "contains_expected": contains_expected_answer(case.task, minimal_out, case.expected),
                "generated_tokens": len(minimal_ids),
                "prefill_tokens": int(minimal_input_ids.numel()),
                "injected_embedding_tokens": int(minimal_suffix_ids.numel()),
                "total_kv_tokens": int(minimal_input_ids.numel()) + len(minimal_ids),
                "latency_s": minimal_dt,
                "llm_latency_s": minimal_llm_dt,
                "operator_overhead_s": operator_overhead,
                "suffix": minimal_suffix,
                "raw": minimal_out[:160],
            }
            totals = minimal_totals[mode]
            totals["generated_tokens"] += len(minimal_ids)
            totals["latency_s"] += minimal_dt
            totals["llm_latency_s"] += minimal_llm_dt
            totals["operator_overhead_s"] += operator_overhead
            totals["prefill_tokens"] += int(minimal_input_ids.numel())
            totals["injected_tokens"] += int(minimal_suffix_ids.numel())

        details.append({
            "expected": case.expected,
            "baseline": baseline_record,
            "operator_vm_text": text_record,
            "minimal_embedding_suffix": minimal_records,
            "predicted": {k: v for k, v in predicted.items() if k != "suffix"},
            "meta": case.meta,
        })
        mode_summary = " ".join(
            f"{mode}={minimal_records[mode]['answer']} tok={minimal_records[mode]['generated_tokens']} inj={minimal_records[mode]['injected_embedding_tokens']}"
            for mode in minimal_modes
        )
        print(
            f"  c2.1 expected={case.expected} baseline={baseline_answer} b_tok={len(baseline_ids)} "
            f"text={text_answer} text_inj={int(text_suffix_ids.numel())} {mode_summary}",
            flush=True,
        )

    baseline = _summarize_generation_path(details, "baseline", baseline_totals)
    operator_vm_text = _summarize_generation_path(details, "operator_vm_text", text_totals)
    minimal_embedding_suffix = {
        mode: _summarize_generation_path(
            [{"minimal": x["minimal_embedding_suffix"][mode]} for x in details],
            "minimal",
            minimal_totals[mode],
        )
        for mode in minimal_modes
    }
    return {
        "baseline": baseline,
        "operator_vm_text": operator_vm_text,
        "minimal_embedding_suffix": minimal_embedding_suffix,
        "speedup": {
            "text_vm_vs_baseline": _speedup_and_saving(baseline, operator_vm_text),
            "minimal_vs_baseline": {
                mode: _speedup_and_saving(baseline, minimal_embedding_suffix[mode])
                for mode in minimal_modes
            },
            "minimal_vs_text_vm": {
                mode: _speedup_and_saving(operator_vm_text, minimal_embedding_suffix[mode])
                for mode in minimal_modes
            },
        },
        "details": details,
    }


def run_minimal_embedding_suffix_benchmark(args) -> None:
    if args.task != "arithmetic":
        raise ValueError("--minimal-embedding-suffix-benchmark only supports --task arithmetic")
    if not args.model:
        raise ValueError("--model is required for --minimal-embedding-suffix-benchmark")
    device = torch.device(args.hf_device)
    dtype = _hf_dtype(args.hf_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    _patch_torch_custom_op_string_annotations()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    operator = PrimitiveLookupDigitWiseOperator(
        c_min=args.pure_arith_c_min,
        c_max=args.pure_arith_c_max,
    ).to(device)
    eval_cases = build_distill_cases(args.task, tokenizer, args.eval_cases, args.context_len, args.depth, args.seed + 10_000)
    num_digits = args.pure_digit_num_digits
    if num_digits <= 0:
        num_digits = _infer_pure_digit_num_digits(
            args.pure_arith_a_max,
            args.pure_arith_b_max,
            args.pure_arith_c_max,
            args.pure_arith_d_max,
        )
    baseline_max_new_tokens = args.operator_vm_baseline_max_new_tokens or args.max_new_tokens
    final_max_new_tokens = args.operator_vm_final_max_new_tokens or args.max_new_tokens
    t0 = time.time()
    print("Running C2.1 minimal embedding suffix benchmark...", flush=True)
    eval_result = evaluate_minimal_embedding_suffix_benchmark(
        model,
        tokenizer,
        operator,
        eval_cases,
        baseline_max_new_tokens=baseline_max_new_tokens,
        final_max_new_tokens=final_max_new_tokens,
        device=device,
        num_digits=num_digits,
        text_input_mode=args.operator_vm_input_mode,
        minimal_modes=args.minimal_embedding_suffix_modes,
    )
    result = {
        "args": vars(args),
        "benchmark": "C2.1 minimal embedding suffix upper bound",
        "eval_cases": len(eval_cases),
        "num_digits": num_digits,
        "baseline_max_new_tokens": baseline_max_new_tokens,
        "operator_vm_final_max_new_tokens": final_max_new_tokens,
        "text_input_mode": args.operator_vm_input_mode,
        "minimal_embedding_suffix_modes": args.minimal_embedding_suffix_modes,
        "elapsed_s": time.time() - t0,
    }
    result.update(eval_result)
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save(result, args.checkpoint)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"c2.1 baseline_acc={result['baseline']['accuracy'] * 100:.1f}% "
        f"text_acc={result['operator_vm_text']['accuracy'] * 100:.1f}% "
        + " ".join(
            f"{mode}_acc={result['minimal_embedding_suffix'][mode]['accuracy'] * 100:.1f}%"
            for mode in args.minimal_embedding_suffix_modes
        ),
        flush=True,
    )
    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved results to {args.out_json}")


class CompactRendererBridge(nn.Module):
    """Map explicit final sign/digit slots into compact LLM-readable latent embeddings."""

    def __init__(self, feature_dim: int, hidden_size: int, latent_steps: int, bridge_hidden_size: int, depth: int):
        super().__init__()
        if feature_dim < 1 or hidden_size < 1 or latent_steps < 1:
            raise ValueError("feature_dim, hidden_size, and latent_steps must be positive")
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.latent_steps = latent_steps
        layers: list[nn.Module] = []
        cur = feature_dim
        for _ in range(max(1, depth)):
            layers.append(nn.Linear(cur, bridge_hidden_size))
            layers.append(nn.GELU())
            cur = bridge_hidden_size
        layers.append(nn.Linear(cur, latent_steps * hidden_size))
        self.net = nn.Sequential(*layers)
        self.step_embed = nn.Parameter(torch.zeros(latent_steps, hidden_size))
        nn.init.normal_(self.step_embed, mean=0.0, std=hidden_size ** -0.5)

    def forward(self, features: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        latents = self.net(features.float()).view(features.size(0), self.latent_steps, self.hidden_size)
        latents = latents + self.step_embed.to(device=features.device, dtype=latents.dtype).unsqueeze(0)
        if dtype is not None:
            latents = latents.to(dtype=dtype)
        return latents


def _compact_renderer_slot_features(cases: list[DistillCase], num_digits: int, device: torch.device) -> torch.Tensor:
    return _compact_renderer_slot_features_from_slots(
        [_final_digit_slots_from_case(case, num_digits) for case in cases],
        num_digits,
        device,
    )


def _compact_renderer_slot_features_from_slots(slots_list: list[dict], num_digits: int, device: torch.device) -> torch.Tensor:
    sign_ids = []
    digits = []
    for slots in slots_list:
        sign_ids.append(1 if slots["final_sign"] == "negative" else 0)
        digits.append([int(x) for x in slots["final_digits_msd"]])
    sign = torch.nn.functional.one_hot(torch.tensor(sign_ids, device=device, dtype=torch.long), num_classes=2).float()
    digit_tensor = torch.tensor(digits, device=device, dtype=torch.long)
    digit_features = torch.nn.functional.one_hot(digit_tensor.clamp(0, 9), num_classes=10).float().reshape(len(slots_list), -1)
    return torch.cat([sign, digit_features], dim=-1)


def _signed_answer_suffix_from_slots(slots: dict) -> str:
    return f"\nanswer={_signed_rendered_answer(slots)}\nFinal integer:"


def _signed_answer_suffix_ids_from_case(tokenizer, case: DistillCase, num_digits: int, device: torch.device) -> torch.Tensor:
    slots = _final_digit_slots_from_case(case, num_digits)
    return torch.tensor(
        tokenizer.encode(_signed_answer_suffix_from_slots(slots), add_special_tokens=False),
        device=device,
        dtype=torch.long,
    )


class DiscreteLookupRenderer:
    """Deterministic slot -> canonical token IDs -> embedding lookup renderer."""

    def __init__(self, tokenizer, embed):
        self.tokenizer = tokenizer
        self.embed = embed

    def render_suffix(self, slots: dict) -> str:
        return _signed_answer_suffix_from_slots(slots)

    def render_token_ids(self, slots: dict, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            self.tokenizer.encode(self.render_suffix(slots), add_special_tokens=False),
            device=device,
            dtype=torch.long,
        )

    def render_embeddings(self, slots: dict, device: torch.device, dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor, str]:
        suffix = self.render_suffix(slots)
        token_ids = torch.tensor(
            self.tokenizer.encode(suffix, add_special_tokens=False),
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        embeddings = self.embed(token_ids).detach()
        if dtype is not None:
            embeddings = embeddings.to(dtype=dtype)
        return token_ids, embeddings, suffix


@torch.no_grad()
def evaluate_discrete_lookup_renderer(
    model,
    tokenizer,
    operator,
    cases: list[DistillCase],
    final_max_new_tokens: int,
    device: torch.device,
    num_digits: int,
) -> dict:
    model.eval()
    embed = model.get_input_embeddings()
    model_dtype = next(model.parameters()).dtype
    renderer = DiscreteLookupRenderer(tokenizer, embed)
    details = []
    totals = _empty_path_totals()
    renderer_overhead_s = 0.0
    token_identity_hits = 0
    slot_hits = 0
    for case in cases:
        prompt_ids = torch.tensor(
            tokenizer.encode(case.prompt, add_special_tokens=False),
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)

        _sync_device_for_timing(device)
        op_t0 = time.time()
        predicted = _primitive_digit_operator_slots_from_case(operator, case, num_digits, device)
        _sync_device_for_timing(device)
        operator_overhead = time.time() - op_t0

        _sync_device_for_timing(device)
        render_t0 = time.time()
        suffix_ids, suffix_embeds, suffix = renderer.render_embeddings(predicted, device, dtype=model_dtype)
        _sync_device_for_timing(device)
        renderer_overhead = time.time() - render_t0

        inputs_embeds = torch.cat([embed(prompt_ids).detach(), suffix_embeds], dim=1)
        first_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long),
        }
        out_text, generated_ids, llm_dt = _generate_hf_greedy_timed(
            model,
            tokenizer,
            first_kwargs,
            final_max_new_tokens,
            device,
            expected=case.expected,
        )
        answer_only = extract_answer_only(case.task, out_text)
        answer = extract_distill_answer(case.task, out_text)
        target_slots = _final_digit_slots_from_case(case, num_digits)
        target_suffix_ids = _signed_answer_suffix_ids_from_case(tokenizer, case, num_digits, device).unsqueeze(0)
        slot_hit = (
            predicted["final_sign"] == target_slots["final_sign"]
            and predicted["final_digits_msd"] == target_slots["final_digits_msd"]
        )
        token_identity_hit = bool(torch.equal(suffix_ids, target_suffix_ids))
        total_dt = operator_overhead + renderer_overhead + llm_dt
        record = {
            "answer_only": answer_only,
            "answer": answer,
            "hit": answer == case.expected,
            "contains_expected": contains_expected_answer(case.task, out_text, case.expected),
            "slot_hit": slot_hit,
            "token_identity_hit": token_identity_hit,
            "generated_tokens": len(generated_ids),
            "prefill_tokens": int(prompt_ids.numel()) + int(suffix_ids.numel()),
            "injected_embedding_tokens": int(suffix_ids.numel()),
            "total_kv_tokens": int(prompt_ids.numel()) + int(suffix_ids.numel()) + len(generated_ids),
            "latency_s": total_dt,
            "llm_latency_s": llm_dt,
            "operator_overhead_s": operator_overhead,
            "renderer_overhead_s": renderer_overhead,
            "non_llm_overhead_s": operator_overhead + renderer_overhead,
            "suffix": suffix,
            "suffix_token_ids": [int(x) for x in suffix_ids.squeeze(0).tolist()],
            "raw": out_text[:160],
        }
        totals["generated_tokens"] += len(generated_ids)
        totals["latency_s"] += total_dt
        totals["llm_latency_s"] += llm_dt
        totals["operator_overhead_s"] += operator_overhead
        totals["prefill_tokens"] += int(prompt_ids.numel()) + int(suffix_ids.numel())
        totals["injected_tokens"] += int(suffix_ids.numel())
        renderer_overhead_s += renderer_overhead
        token_identity_hits += int(token_identity_hit)
        slot_hits += int(slot_hit)
        details.append({
            "expected": case.expected,
            "discrete_lookup_renderer": record,
            "predicted": {k: v for k, v in predicted.items() if k != "suffix"},
            "target_slots": target_slots,
            "meta": case.meta,
        })
        print(
            f"  c2.3 expected={case.expected} answer={answer} hit={record['hit']} "
            f"slot_hit={slot_hit} token_identity={token_identity_hit} inj={int(suffix_ids.numel())} gen_tok={len(generated_ids)}",
            flush=True,
        )

    summary = _summarize_generation_path(
        [{"discrete_lookup_renderer": x["discrete_lookup_renderer"]} for x in details],
        "discrete_lookup_renderer",
        totals,
    )
    n = max(1, len(details))
    summary["avg_renderer_overhead_s"] = renderer_overhead_s / n
    summary["avg_non_llm_overhead_s"] = (totals["operator_overhead_s"] + renderer_overhead_s) / n
    summary["slot_exact_accuracy"] = slot_hits / n
    summary["token_identity_accuracy"] = token_identity_hits / n
    return {
        "discrete_lookup_renderer": summary,
        "c21_signed_answer_equivalence": {
            "canonical_suffix_mode": "signed_answer",
            "uses_tokenizer_encode_plus_embedding_lookup": True,
            "token_identity_accuracy": token_identity_hits / n,
            "avg_injected_tokens": summary.get("avg_injected_tokens", 0.0),
        },
        "details": details,
    }


def run_discrete_lookup_renderer(args) -> None:
    if args.task != "arithmetic":
        raise ValueError("--discrete-lookup-renderer only supports --task arithmetic")
    if not args.model:
        raise ValueError("--model is required for --discrete-lookup-renderer")
    device = torch.device(args.hf_device)
    dtype = _hf_dtype(args.hf_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    _patch_torch_custom_op_string_annotations()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    operator = PrimitiveLookupDigitWiseOperator(
        c_min=args.pure_arith_c_min,
        c_max=args.pure_arith_c_max,
    ).to(device)
    eval_cases = build_distill_cases(args.task, tokenizer, args.eval_cases, args.context_len, args.depth, args.seed + 10_000)
    num_digits = args.pure_digit_num_digits
    if num_digits <= 0:
        num_digits = _infer_pure_digit_num_digits(
            args.pure_arith_a_max,
            args.pure_arith_b_max,
            args.pure_arith_c_max,
            args.pure_arith_d_max,
        )
    final_max_new_tokens = args.operator_vm_final_max_new_tokens or args.max_new_tokens
    t0 = time.time()
    print("Running C2.3 deterministic discrete lookup renderer...", flush=True)
    eval_result = evaluate_discrete_lookup_renderer(
        model,
        tokenizer,
        operator,
        eval_cases,
        final_max_new_tokens=final_max_new_tokens,
        device=device,
        num_digits=num_digits,
    )
    result = {
        "args": vars(args),
        "benchmark": "C2.3 deterministic discrete lookup renderer",
        "eval_cases": len(eval_cases),
        "num_digits": num_digits,
        "operator_vm_final_max_new_tokens": final_max_new_tokens,
        "renderer": "slot -> signed_answer suffix -> tokenizer IDs -> embedding lookup",
        "text_input_mode": "inputs_embeds",
        "elapsed_s": time.time() - t0,
    }
    result.update(eval_result)
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save(result, args.checkpoint)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    summary = result["discrete_lookup_renderer"]
    print(
        f"c2.3 acc={summary['accuracy'] * 100:.1f}% "
        f"token_identity={summary['token_identity_accuracy'] * 100:.1f}% "
        f"avg_injected_tokens={summary.get('avg_injected_tokens', 0.0):.2f} "
        f"avg_total_kv_tokens={summary['avg_total_kv_tokens']:.2f}",
        flush=True,
    )
    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved results to {args.out_json}")


class VocabBottleneckRendererBridge(nn.Module):
    """Map final slots to canonical suffix token IDs through a small discrete vocabulary bottleneck."""

    def __init__(self, feature_dim: int, hidden_size: int, max_suffix_tokens: int, num_candidates: int, depth: int):
        super().__init__()
        if feature_dim < 1 or hidden_size < 1 or max_suffix_tokens < 1 or num_candidates < 1:
            raise ValueError("feature_dim, hidden_size, max_suffix_tokens, and num_candidates must be positive")
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.max_suffix_tokens = max_suffix_tokens
        self.num_candidates = num_candidates
        layers: list[nn.Module] = []
        cur = feature_dim
        for _ in range(max(1, depth)):
            layers.append(nn.Linear(cur, hidden_size))
            layers.append(nn.GELU())
            cur = hidden_size
        self.net = nn.Sequential(*layers)
        self.token_head = nn.Linear(cur, max_suffix_tokens * num_candidates)
        self.length_head = nn.Linear(cur, max_suffix_tokens + 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(features.float())
        token_logits = self.token_head(hidden).view(features.size(0), self.max_suffix_tokens, self.num_candidates)
        length_logits = self.length_head(hidden)
        return token_logits, length_logits


def _vocab_renderer_candidate_token_ids(tokenizer, device: torch.device) -> tuple[torch.Tensor, list[int]]:
    ids: set[int] = set()
    examples = ["0", "-1"] + [str(x) for x in range(10)]
    for answer in examples:
        suffix = f"\nanswer={answer}\nFinal integer:"
        ids.update(int(x) for x in tokenizer.encode(suffix, add_special_tokens=False))
    ordered = sorted(ids)
    return torch.tensor(ordered, device=device, dtype=torch.long), ordered


def _vocab_renderer_target_classes(target_ids: torch.Tensor, candidate_index: dict[int, int], max_suffix_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    if target_ids.numel() > max_suffix_tokens:
        raise ValueError(f"target suffix length {target_ids.numel()} exceeds max_suffix_tokens={max_suffix_tokens}")
    target = torch.zeros(max_suffix_tokens, device=target_ids.device, dtype=torch.long)
    mask = torch.zeros(max_suffix_tokens, device=target_ids.device, dtype=torch.bool)
    for pos, token_id in enumerate(target_ids.tolist()):
        token_id = int(token_id)
        if token_id not in candidate_index:
            raise ValueError(f"target token id {token_id} is missing from C2.4 candidate vocabulary")
        target[pos] = int(candidate_index[token_id])
        mask[pos] = True
    return target, mask


def train_vocab_bottleneck_renderer_bridge(
    tokenizer,
    bridge: VocabBottleneckRendererBridge,
    cases: list[DistillCase],
    num_digits: int,
    candidate_token_ids: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    length_ce_weight: float,
) -> list[dict]:
    bridge.to(device=device, dtype=torch.float32).train()
    opt = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=weight_decay)
    candidate_index = {int(tok): idx for idx, tok in enumerate(candidate_token_ids.tolist())}
    history = []
    for epoch in range(1, epochs + 1):
        rng = random.Random(epoch)
        order = list(range(len(cases)))
        rng.shuffle(order)
        totals = {
            "loss": 0.0,
            "token_ce": 0.0,
            "length_ce": 0.0,
            "tokens": 0,
            "token_correct": 0,
            "length_correct": 0,
            "seq_correct": 0,
            "seqs": 0,
        }
        for start in range(0, len(order), batch_size):
            batch_cases = [cases[i] for i in order[start:start + batch_size]]
            features = _compact_renderer_slot_features(batch_cases, num_digits, device)
            token_logits, length_logits = bridge(features)
            row_losses = []
            token_ce_sum = 0.0
            length_targets = []
            seq_correct = 0
            token_correct = 0
            token_count = 0
            for row, case in enumerate(batch_cases):
                target_ids = _signed_answer_suffix_ids_from_case(tokenizer, case, num_digits, device)
                target_classes, mask = _vocab_renderer_target_classes(target_ids, candidate_index, bridge.max_suffix_tokens)
                row_logits = token_logits[row][mask]
                row_targets = target_classes[mask]
                row_token_loss = torch.nn.functional.cross_entropy(row_logits, row_targets)
                row_losses.append(row_token_loss * int(row_targets.numel()))
                nearest = row_logits.argmax(dim=-1)
                row_correct = int((nearest == row_targets).sum().item())
                token_correct += row_correct
                token_count += int(row_targets.numel())
                token_ce_sum += float(row_token_loss.item()) * int(row_targets.numel())
                length_targets.append(int(target_ids.numel()))
                seq_correct += int(row_correct == int(row_targets.numel()))
            length_target_tensor = torch.tensor(length_targets, device=device, dtype=torch.long)
            length_loss = torch.nn.functional.cross_entropy(length_logits, length_target_tensor)
            token_loss = torch.stack(row_losses).sum() / max(1, token_count)
            loss = token_loss + length_ce_weight * length_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            length_pred = length_logits.argmax(dim=-1)
            totals["loss"] += float(loss.item()) * max(1, token_count)
            totals["token_ce"] += token_ce_sum
            totals["length_ce"] += float(length_loss.item()) * len(batch_cases)
            totals["tokens"] += token_count
            totals["token_correct"] += token_correct
            totals["length_correct"] += int((length_pred == length_target_tensor).sum().item())
            totals["seq_correct"] += seq_correct
            totals["seqs"] += len(batch_cases)
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / max(1, totals["tokens"]),
            "token_ce": totals["token_ce"] / max(1, totals["tokens"]),
            "length_ce": totals["length_ce"] / max(1, totals["seqs"]),
            "token_accuracy": _safe_div(totals["token_correct"], totals["tokens"]),
            "length_accuracy": _safe_div(totals["length_correct"], totals["seqs"]),
            "sequence_token_accuracy": _safe_div(totals["seq_correct"], totals["seqs"]),
            "tokens": totals["tokens"],
            "seqs": totals["seqs"],
        }
        history.append(row)
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            print(
                f"  c24_epoch={epoch:03d} loss={row['loss']:.6f} token_acc={row['token_accuracy'] * 100:.1f}% "
                f"len_acc={row['length_accuracy'] * 100:.1f}% seq_tok_acc={row['sequence_token_accuracy'] * 100:.1f}%",
                flush=True,
            )
    return history


@torch.no_grad()
def evaluate_vocab_bottleneck_renderer_decode(
    model,
    tokenizer,
    operator,
    bridge: VocabBottleneckRendererBridge,
    cases: list[DistillCase],
    num_digits: int,
    candidate_token_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
) -> dict:
    model.eval()
    bridge.eval()
    embed = model.get_input_embeddings()
    details = []
    totals = _empty_path_totals()
    bridge_overhead_s = 0.0
    token_identity_hits = 0
    length_hits = 0
    token_correct = 0
    token_total = 0
    slot_hits = 0
    for case in cases:
        prompt_ids = torch.tensor(tokenizer.encode(case.prompt, add_special_tokens=False), device=device, dtype=torch.long).unsqueeze(0)
        _sync_device_for_timing(device)
        op_t0 = time.time()
        predicted_slots = _primitive_digit_operator_slots_from_case(operator, case, num_digits, device)
        _sync_device_for_timing(device)
        operator_overhead = time.time() - op_t0

        _sync_device_for_timing(device)
        bridge_t0 = time.time()
        features = _compact_renderer_slot_features_from_slots([predicted_slots], num_digits, device)
        token_logits, length_logits = bridge(features)
        pred_len = int(length_logits.argmax(dim=-1).item())
        pred_len = max(1, min(pred_len, bridge.max_suffix_tokens))
        pred_candidate_ids = token_logits.argmax(dim=-1)[0, :pred_len]
        pred_token_ids = candidate_token_ids[pred_candidate_ids].unsqueeze(0)
        suffix_embeds = embed(pred_token_ids).detach()
        _sync_device_for_timing(device)
        bridge_overhead = time.time() - bridge_t0

        inputs_embeds = torch.cat([embed(prompt_ids).detach(), suffix_embeds], dim=1)
        first_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long),
        }
        text, generated_ids, llm_dt = _generate_hf_greedy_timed(
            model,
            tokenizer,
            first_kwargs,
            max_new_tokens,
            device,
            expected=case.expected,
        )
        answer_only = extract_answer_only(case.task, text)
        answer = extract_distill_answer(case.task, text)
        target_ids = _signed_answer_suffix_ids_from_case(tokenizer, case, num_digits, device).unsqueeze(0)
        compare_len = min(pred_token_ids.size(1), target_ids.size(1))
        row_token_correct = int((pred_token_ids[:, :compare_len] == target_ids[:, :compare_len]).sum().item())
        token_correct += row_token_correct
        token_total += int(target_ids.numel())
        length_hit = pred_token_ids.size(1) == target_ids.size(1)
        token_identity_hit = length_hit and bool(torch.equal(pred_token_ids, target_ids))
        target_slots = _final_digit_slots_from_case(case, num_digits)
        slot_hit = (
            predicted_slots["final_sign"] == target_slots["final_sign"]
            and predicted_slots["final_digits_msd"] == target_slots["final_digits_msd"]
        )
        total_dt = operator_overhead + bridge_overhead + llm_dt
        record = {
            "answer_only": answer_only,
            "answer": answer,
            "hit": answer == case.expected,
            "contains_expected": contains_expected_answer(case.task, text, case.expected),
            "slot_hit": slot_hit,
            "length_hit": length_hit,
            "token_identity_hit": token_identity_hit,
            "token_accuracy": row_token_correct / max(1, int(target_ids.numel())),
            "generated_tokens": len(generated_ids),
            "prefill_tokens": int(prompt_ids.numel()) + int(pred_token_ids.numel()),
            "injected_embedding_tokens": int(pred_token_ids.numel()),
            "total_kv_tokens": int(prompt_ids.numel()) + int(pred_token_ids.numel()) + len(generated_ids),
            "latency_s": total_dt,
            "llm_latency_s": llm_dt,
            "operator_overhead_s": operator_overhead,
            "bridge_overhead_s": bridge_overhead,
            "predicted_suffix": tokenizer.decode(pred_token_ids.squeeze(0).tolist(), skip_special_tokens=True),
            "target_suffix": tokenizer.decode(target_ids.squeeze(0).tolist(), skip_special_tokens=True),
            "predicted_token_ids": [int(x) for x in pred_token_ids.squeeze(0).tolist()],
            "target_token_ids": [int(x) for x in target_ids.squeeze(0).tolist()],
            "raw": text[:160],
        }
        totals["generated_tokens"] += len(generated_ids)
        totals["latency_s"] += total_dt
        totals["llm_latency_s"] += llm_dt
        totals["operator_overhead_s"] += operator_overhead
        totals["prefill_tokens"] += int(prompt_ids.numel()) + int(pred_token_ids.numel())
        totals["injected_tokens"] += int(pred_token_ids.numel())
        bridge_overhead_s += bridge_overhead
        token_identity_hits += int(token_identity_hit)
        length_hits += int(length_hit)
        slot_hits += int(slot_hit)
        details.append({
            "expected": case.expected,
            "vocab_bottleneck_renderer": record,
            "predicted_slots": {k: v for k, v in predicted_slots.items() if k != "suffix"},
            "target_slots": target_slots,
            "meta": case.meta,
        })
        print(
            f"  c2.4 expected={case.expected} answer={answer} hit={record['hit']} "
            f"token_identity={token_identity_hit} len_hit={length_hit} inj={int(pred_token_ids.numel())} raw={text[:48]!r}",
            flush=True,
        )
    summary = _summarize_generation_path(
        [{"vocab_bottleneck_renderer": x["vocab_bottleneck_renderer"]} for x in details],
        "vocab_bottleneck_renderer",
        totals,
    )
    n = max(1, len(details))
    summary["avg_bridge_overhead_s"] = bridge_overhead_s / n
    summary["avg_non_llm_overhead_s"] = (totals["operator_overhead_s"] + bridge_overhead_s) / n
    summary["slot_exact_accuracy"] = slot_hits / n
    summary["length_accuracy"] = length_hits / n
    summary["token_identity_accuracy"] = token_identity_hits / n
    summary["token_accuracy"] = token_correct / max(1, token_total)
    return {"summary": summary, "details": details}


def run_vocab_bottleneck_renderer_bridge(args) -> None:
    if args.task != "arithmetic":
        raise ValueError("--vocab-bottleneck-renderer-bridge only supports --task arithmetic")
    if not args.model:
        raise ValueError("--model is required for --vocab-bottleneck-renderer-bridge")
    device = torch.device(args.hf_device)
    dtype = _hf_dtype(args.hf_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    _patch_torch_custom_op_string_annotations()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    train_cases = build_distill_cases(args.task, tokenizer, args.train_cases, args.context_len, args.depth, args.seed)
    eval_cases = build_distill_cases(args.task, tokenizer, args.eval_cases, args.context_len, args.depth, args.seed + 10_000)
    num_digits = args.pure_digit_num_digits
    if num_digits <= 0:
        num_digits = _infer_pure_digit_num_digits(
            args.pure_arith_a_max,
            args.pure_arith_b_max,
            args.pure_arith_c_max,
            args.pure_arith_d_max,
        )
    candidate_token_ids, candidate_token_id_list = _vocab_renderer_candidate_token_ids(tokenizer, device)
    sample_features = _compact_renderer_slot_features(train_cases[:1], num_digits, device)
    max_suffix_tokens = max(
        int(_signed_answer_suffix_ids_from_case(tokenizer, case, num_digits, device).numel())
        for case in train_cases + eval_cases
    )
    bridge = VocabBottleneckRendererBridge(
        feature_dim=int(sample_features.size(-1)),
        hidden_size=args.vocab_renderer_hidden_size,
        max_suffix_tokens=max_suffix_tokens,
        num_candidates=int(candidate_token_ids.numel()),
        depth=args.vocab_renderer_depth,
    )
    operator = PrimitiveLookupDigitWiseOperator(
        c_min=args.pure_arith_c_min,
        c_max=args.pure_arith_c_max,
    ).to(device)
    t0 = time.time()
    print(
        f"Training C2.4 vocabulary-bottleneck renderer bridge candidates={int(candidate_token_ids.numel())} "
        f"max_suffix_tokens={max_suffix_tokens}...",
        flush=True,
    )
    history = train_vocab_bottleneck_renderer_bridge(
        tokenizer,
        bridge,
        train_cases,
        num_digits,
        candidate_token_ids,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        length_ce_weight=args.vocab_renderer_length_ce_weight,
    )
    eval_result = evaluate_vocab_bottleneck_renderer_decode(
        model,
        tokenizer,
        operator,
        bridge,
        eval_cases,
        num_digits,
        candidate_token_ids,
        max_new_tokens=args.operator_vm_final_max_new_tokens or args.max_new_tokens,
        device=device,
    )
    result = {
        "args": vars(args),
        "benchmark": "C2.4 vocabulary-bottleneck renderer bridge",
        "train_cases": len(train_cases),
        "eval_cases": len(eval_cases),
        "num_digits": num_digits,
        "feature_dim": int(sample_features.size(-1)),
        "max_suffix_tokens": max_suffix_tokens,
        "candidate_token_ids": candidate_token_id_list,
        "candidate_tokens": [tokenizer.decode([int(x)], skip_special_tokens=True) for x in candidate_token_id_list],
        "history": history,
        "vocab_bottleneck_renderer": eval_result["summary"],
        "details": eval_result["details"],
        "elapsed_s": time.time() - t0,
    }
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    checkpoint_result = dict(result)
    checkpoint_result["bridge"] = bridge.state_dict()
    torch.save(checkpoint_result, args.checkpoint)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    summary = result["vocab_bottleneck_renderer"]
    print(
        f"c2.4 acc={summary['accuracy'] * 100:.1f}% token_identity={summary['token_identity_accuracy'] * 100:.1f}% "
        f"token_acc={summary['token_accuracy'] * 100:.1f}% avg_injected={summary.get('avg_injected_tokens', 0.0):.2f}",
        flush=True,
    )
    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved results to {args.out_json}")


def _final_answer_ids(tokenizer, expected: str, device: torch.device) -> torch.Tensor:
    return torch.tensor(tokenizer.encode(f" {expected}", add_special_tokens=False), device=device, dtype=torch.long)


def _bridge_alignment_loss_for_batch(
    bridge: CompactRendererBridge,
    embed,
    tokenizer,
    batch_cases: list[DistillCase],
    num_digits: int,
    device: torch.device,
    cosine_weight: float,
    token_ce_weight: float,
    token_ce_temperature: float,
    normalized_embedding_table: torch.Tensor | None,
) -> tuple[torch.Tensor, dict]:
    features = _compact_renderer_slot_features(batch_cases, num_digits, device)
    pred = bridge(features, dtype=torch.float32)
    losses = []
    total_mse = 0.0
    total_cos = 0.0
    total_ce = 0.0
    total_ce_correct = 0
    total_tokens = 0
    for row, case in enumerate(batch_cases):
        target_ids = _signed_answer_suffix_ids_from_case(tokenizer, case, num_digits, device)
        if target_ids.numel() > bridge.latent_steps:
            raise ValueError(f"target suffix length {target_ids.numel()} exceeds bridge latent_steps={bridge.latent_steps}")
        target = embed(target_ids).detach().float()
        row_loss, row_mse, row_cos = _canonical_embedding_alignment_loss(
            pred[row, :target_ids.numel()], target, cosine_weight
        )
        if token_ce_weight > 0:
            if normalized_embedding_table is None:
                raise ValueError("normalized_embedding_table is required when token_ce_weight > 0")
            row_ce_loss, row_ce, row_ce_correct = _canonical_embedding_token_ce_loss(
                pred[row, :target_ids.numel()],
                target_ids,
                normalized_embedding_table,
                token_ce_temperature,
            )
            row_loss = row_loss + token_ce_weight * row_ce_loss
            total_ce += row_ce * int(target_ids.numel())
            total_ce_correct += row_ce_correct
        losses.append(row_loss * target_ids.numel())
        total_mse += row_mse * int(target_ids.numel())
        total_cos += row_cos * int(target_ids.numel())
        total_tokens += int(target_ids.numel())
    loss = torch.stack(losses).sum() / max(1, total_tokens)
    return loss, {
        "tokens": total_tokens,
        "mse": total_mse,
        "cosine": total_cos,
        "token_ce": total_ce,
        "token_ce_correct": total_ce_correct,
    }


def _bridge_answer_ce_loss_for_batch(
    model,
    tokenizer,
    bridge: CompactRendererBridge,
    batch_cases: list[DistillCase],
    num_digits: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    embed = model.get_input_embeddings()
    model_dtype = next(model.parameters()).dtype
    features = _compact_renderer_slot_features(batch_cases, num_digits, device)
    latents = bridge(features, dtype=model_dtype)
    seq_embeds = []
    seq_lens = []
    target_positions = []
    target_ids_all = []
    for row, case in enumerate(batch_cases):
        prompt_ids = torch.tensor(tokenizer.encode(case.prompt, add_special_tokens=False), device=device, dtype=torch.long)
        answer_ids = _final_answer_ids(tokenizer, case.expected, device)
        parts = [embed(prompt_ids).detach(), latents[row]]
        if answer_ids.numel() > 1:
            parts.append(embed(answer_ids[:-1]).detach())
        seq = torch.cat(parts, dim=0)
        seq_embeds.append(seq)
        seq_lens.append(seq.size(0))
        base = int(prompt_ids.numel()) + bridge.latent_steps - 1
        target_positions.extend((row, base + int(i)) for i in range(answer_ids.numel()))
        target_ids_all.extend(int(x) for x in answer_ids.tolist())
    max_len = max(seq_lens)
    hidden_size = seq_embeds[0].size(-1)
    inputs_embeds = torch.zeros(len(batch_cases), max_len, hidden_size, device=device, dtype=model_dtype)
    attention_mask = torch.zeros(len(batch_cases), max_len, device=device, dtype=torch.long)
    for row, seq in enumerate(seq_embeds):
        inputs_embeds[row, :seq.size(0)] = seq
        attention_mask[row, :seq.size(0)] = 1
    out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False, return_dict=True)
    rows = torch.tensor([x[0] for x in target_positions], device=device, dtype=torch.long)
    cols = torch.tensor([x[1] for x in target_positions], device=device, dtype=torch.long)
    targets = torch.tensor(target_ids_all, device=device, dtype=torch.long)
    logits = out.logits[rows, cols].float()
    loss = torch.nn.functional.cross_entropy(logits, targets)
    correct = int((logits.argmax(dim=-1) == targets).sum().item())
    return loss, {"tokens": int(targets.numel()), "answer_ce_correct": correct, "answer_ce": float(loss.item())}


def train_compact_renderer_bridge(
    model,
    tokenizer,
    bridge: CompactRendererBridge,
    cases: list[DistillCase],
    num_digits: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    objective: str,
    cosine_weight: float,
    token_ce_weight: float,
    token_ce_temperature: float,
) -> list[dict]:
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    embed = model.get_input_embeddings()
    normalized_embedding_table = None
    if token_ce_weight > 0:
        normalized_embedding_table = torch.nn.functional.normalize(embed.weight.detach().float(), dim=-1)
    bridge.to(device=device, dtype=torch.float32).train()
    opt = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(1, epochs + 1):
        rng = random.Random(epoch)
        order = list(range(len(cases)))
        rng.shuffle(order)
        totals = {
            "loss": 0.0,
            "tokens": 0,
            "mse": 0.0,
            "cosine": 0.0,
            "token_ce": 0.0,
            "token_ce_correct": 0,
            "answer_ce": 0.0,
            "answer_ce_correct": 0,
        }
        for start in range(0, len(order), batch_size):
            idxs = order[start:start + batch_size]
            batch_cases = [cases[i] for i in idxs]
            if objective == "alignment":
                loss, metrics = _bridge_alignment_loss_for_batch(
                    bridge,
                    embed,
                    tokenizer,
                    batch_cases,
                    num_digits,
                    device,
                    cosine_weight,
                    token_ce_weight,
                    token_ce_temperature,
                    normalized_embedding_table,
                )
            elif objective == "answer_ce":
                loss, metrics = _bridge_answer_ce_loss_for_batch(model, tokenizer, bridge, batch_cases, num_digits, device)
            else:
                raise ValueError(f"unknown compact renderer bridge objective: {objective}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tokens = int(metrics.get("tokens", len(batch_cases)))
            totals["loss"] += float(loss.item()) * max(1, tokens)
            totals["tokens"] += max(1, tokens)
            for key in ("mse", "cosine", "token_ce", "answer_ce"):
                if key in metrics:
                    totals[key] += float(metrics[key]) * max(1, tokens)
            for key in ("token_ce_correct", "answer_ce_correct"):
                totals[key] += int(metrics.get(key, 0))
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / max(1, totals["tokens"]),
            "tokens": totals["tokens"],
        }
        if objective == "alignment":
            row.update({
                "mse": totals["mse"] / max(1, totals["tokens"]),
                "cosine": totals["cosine"] / max(1, totals["tokens"]),
                "token_ce": totals["token_ce"] / max(1, totals["tokens"]),
                "token_ce_accuracy": _safe_div(totals["token_ce_correct"], totals["tokens"]),
            })
        else:
            row.update({
                "answer_ce": totals["answer_ce"] / max(1, totals["tokens"]),
                "answer_ce_accuracy": _safe_div(totals["answer_ce_correct"], totals["tokens"]),
            })
        history.append(row)
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            if objective == "alignment":
                print(
                    f"  c22_{objective}_epoch={epoch:03d} loss={row['loss']:.6f} "
                    f"mse={row['mse']:.6f} cos={row['cosine']:.6f} token_ce_acc={row['token_ce_accuracy'] * 100:.1f}%",
                    flush=True,
                )
            else:
                print(
                    f"  c22_{objective}_epoch={epoch:03d} loss={row['loss']:.6f} "
                    f"answer_ce_acc={row['answer_ce_accuracy'] * 100:.1f}%",
                    flush=True,
                )
    return history


@torch.no_grad()
def evaluate_compact_renderer_bridge_decode(
    model,
    tokenizer,
    bridge: CompactRendererBridge,
    cases: list[DistillCase],
    num_digits: int,
    max_new_tokens: int,
    device: torch.device,
    mode: str,
) -> dict:
    model.eval()
    bridge.eval()
    embed = model.get_input_embeddings()
    model_dtype = next(model.parameters()).dtype
    details = []
    totals = _empty_path_totals()
    for case in cases:
        prompt_ids = torch.tensor(tokenizer.encode(case.prompt, add_special_tokens=False), device=device, dtype=torch.long).unsqueeze(0)
        features = _compact_renderer_slot_features([case], num_digits, device)
        latents = bridge(features, dtype=model_dtype)
        if mode == "variable":
            target_len = int(_signed_answer_suffix_ids_from_case(tokenizer, case, num_digits, device).numel())
            injected = latents[:, :target_len]
        else:
            target_len = bridge.latent_steps
            injected = latents
        inputs_embeds = torch.cat([embed(prompt_ids).detach(), injected], dim=1)
        first_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long),
        }
        text, generated_ids, dt = _generate_hf_greedy_timed(
            model,
            tokenizer,
            first_kwargs,
            max_new_tokens,
            device,
            expected=case.expected,
        )
        answer_only = extract_answer_only(case.task, text)
        answer = extract_distill_answer(case.task, text)
        hit = answer == case.expected
        contains = contains_expected_answer(case.task, text, case.expected)
        totals["generated_tokens"] += len(generated_ids)
        totals["latency_s"] += dt
        totals["llm_latency_s"] += dt
        totals["prefill_tokens"] += int(prompt_ids.numel()) + target_len
        totals["injected_tokens"] += target_len
        details.append({
            "expected": case.expected,
            "answer_only": answer_only,
            "answer": answer,
            "hit": hit,
            "contains_expected": contains,
            "generated_tokens": len(generated_ids),
            "prefill_tokens": int(prompt_ids.numel()) + target_len,
            "injected_latent_slots": target_len,
            "total_kv_units": int(prompt_ids.numel()) + target_len + len(generated_ids),
            "latency_s": dt,
            "raw": text[:160],
            "meta": case.meta,
        })
        print(
            f"  c22_eval mode={mode} K={target_len} expected={case.expected} answer={answer} "
            f"hit={hit} gen_tok={len(generated_ids)} raw={text[:48]!r}",
            flush=True,
        )
    summary = _summarize_generation_path([{"bridge": x} for x in details], "bridge", totals)
    summary["avg_injected_latent_slots"] = summary.pop("avg_injected_tokens", totals["injected_tokens"] / max(1, len(details)))
    summary["avg_total_kv_units"] = summary.pop("avg_total_kv_tokens")
    summary["details"] = details
    return summary


def run_compact_renderer_bridge(args) -> None:
    if args.task != "arithmetic":
        raise ValueError("--compact-renderer-bridge only supports --task arithmetic")
    if not args.model:
        raise ValueError("--model is required for --compact-renderer-bridge")
    device = torch.device(args.hf_device)
    dtype = _hf_dtype(args.hf_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    num_digits = args.pure_digit_num_digits
    if num_digits <= 0:
        num_digits = _infer_pure_digit_num_digits(
            args.pure_arith_a_max,
            args.pure_arith_b_max,
            args.pure_arith_c_max,
            args.pure_arith_d_max,
        )
    sample_features = _compact_renderer_slot_features(train_cases[:1], num_digits, device)
    max_suffix_tokens = max(
        int(_signed_answer_suffix_ids_from_case(tokenizer, case, num_digits, device).numel())
        for case in train_cases + eval_cases
    )
    t0 = time.time()
    results = {}
    histories = {}
    checkpoints = {}
    if args.compact_renderer_run_variable:
        variable_bridge = CompactRendererBridge(
            feature_dim=sample_features.size(-1),
            hidden_size=hidden_size,
            latent_steps=max_suffix_tokens,
            bridge_hidden_size=args.compact_renderer_hidden_size,
            depth=args.compact_renderer_depth,
        )
        print(f"Training C2.2-a variable bridge max_suffix_tokens={max_suffix_tokens}...", flush=True)
        histories["variable"] = train_compact_renderer_bridge(
            model,
            tokenizer,
            variable_bridge,
            train_cases,
            num_digits,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            objective="alignment",
            cosine_weight=args.compact_renderer_cosine_weight,
            token_ce_weight=args.compact_renderer_token_ce_weight,
            token_ce_temperature=args.compact_renderer_token_ce_temperature,
        )
        results["variable"] = evaluate_compact_renderer_bridge_decode(
            model,
            tokenizer,
            variable_bridge,
            eval_cases,
            num_digits,
            args.operator_vm_final_max_new_tokens or args.max_new_tokens,
            device,
            mode="variable",
        )
        checkpoints["variable"] = variable_bridge.state_dict()
    for k in args.compact_renderer_fixed_ks:
        if int(k) <= 0:
            continue
        fixed_bridge = CompactRendererBridge(
            feature_dim=sample_features.size(-1),
            hidden_size=hidden_size,
            latent_steps=int(k),
            bridge_hidden_size=args.compact_renderer_hidden_size,
            depth=args.compact_renderer_depth,
        )
        print(f"Training C2.2-b fixed-K bridge K={k}...", flush=True)
        histories[f"K{k}"] = train_compact_renderer_bridge(
            model,
            tokenizer,
            fixed_bridge,
            train_cases,
            num_digits,
            epochs=args.compact_renderer_fixed_epochs or args.epochs,
            batch_size=args.batch_size,
            lr=args.compact_renderer_fixed_lr or args.lr,
            weight_decay=args.weight_decay,
            device=device,
            objective="answer_ce",
            cosine_weight=args.compact_renderer_cosine_weight,
            token_ce_weight=0.0,
            token_ce_temperature=args.compact_renderer_token_ce_temperature,
        )
        results[f"K{k}"] = evaluate_compact_renderer_bridge_decode(
            model,
            tokenizer,
            fixed_bridge,
            eval_cases,
            num_digits,
            args.operator_vm_final_max_new_tokens or args.max_new_tokens,
            device,
            mode="fixed",
        )
        checkpoints[f"K{k}"] = fixed_bridge.state_dict()
    c21_signed_answer = None
    if args.compact_renderer_include_c21_reference:
        c21_eval = evaluate_minimal_embedding_suffix_benchmark(
            model,
            tokenizer,
            PrimitiveLookupDigitWiseOperator(args.pure_arith_c_min, args.pure_arith_c_max).to(device),
            eval_cases,
            baseline_max_new_tokens=args.operator_vm_baseline_max_new_tokens or args.max_new_tokens,
            final_max_new_tokens=args.operator_vm_final_max_new_tokens or args.max_new_tokens,
            device=device,
            num_digits=num_digits,
            text_input_mode=args.operator_vm_input_mode,
            minimal_modes=["signed_answer"],
        )
        c21_signed_answer = c21_eval["minimal_embedding_suffix"]["signed_answer"]
    result = {
        "args": vars(args),
        "benchmark": "C2.2 learned compact renderer bridge",
        "train_cases": len(train_cases),
        "eval_cases": len(eval_cases),
        "num_digits": num_digits,
        "feature_dim": int(sample_features.size(-1)),
        "hidden_size": hidden_size,
        "max_suffix_tokens": max_suffix_tokens,
        "c21_signed_answer": c21_signed_answer,
        "history": histories,
        "c22_bridge": results,
        "elapsed_s": time.time() - t0,
    }
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    checkpoint_result = dict(result)
    checkpoint_result["bridges"] = checkpoints
    torch.save(checkpoint_result, args.checkpoint)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(
        "c22 " + " ".join(f"{name}_acc={res['accuracy'] * 100:.1f}%" for name, res in results.items()),
        flush=True,
    )
    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved results to {args.out_json}")


@torch.no_grad()
def evaluate_primitive_digit_renderer(
    model,
    tokenizer,
    operator,
    cases: list[DistillCase],
    max_new_tokens: int,
    device: torch.device,
    num_digits: int,
    input_mode: str,
) -> dict:
    details = []
    for case in cases:
        text, predicted = generate_hf_with_primitive_digit_renderer(
            model, tokenizer, operator, case, max_new_tokens, device, num_digits, input_mode
        )
        answer_only = extract_answer_only(case.task, text)
        contains_expected = contains_expected_answer(case.task, text, case.expected)
        decode_hit = answer_only == case.expected
        target_slots = _final_digit_slots_from_case(case, num_digits)
        slot_hit = (
            predicted["final_sign"] == target_slots["final_sign"]
            and predicted["final_digits_msd"] == target_slots["final_digits_msd"]
        )
        details.append({
            "expected": case.expected,
            "answer_only": answer_only,
            "contains_expected": contains_expected,
            "decode_hit": decode_hit,
            "slot_hit": slot_hit,
            "raw": text[:160],
            "input_mode": input_mode,
            "predicted": {k: v for k, v in predicted.items() if k != "suffix"},
            "target_slots": target_slots,
            "meta": case.meta,
        })
        print(
            f"  primitive_digit_renderer_eval mode={input_mode} expected={case.expected} answer_only={answer_only} "
            f"slot_hit={slot_hit} decode_hit={decode_hit} pred_abs={predicted['rendered_abs']} raw={text[:48]!r}",
            flush=True,
        )
    slot_acc = sum(int(x["slot_hit"]) for x in details) / max(1, len(details))
    decode_acc = sum(int(x["decode_hit"]) for x in details) / max(1, len(details))
    contains_acc = sum(int(x["contains_expected"]) for x in details) / max(1, len(details))
    overflow_cases = sum(int(x["predicted"].get("overflow", False)) for x in details)
    return {
        "slot_exact_accuracy": slot_acc,
        "accuracy": decode_acc,
        "contains_accuracy": contains_acc,
        "overflow_cases": overflow_cases,
        "input_mode": input_mode,
        "details": details,
    }


def run_primitive_digit_renderer_eval(args) -> None:
    if args.task != "arithmetic":
        raise ValueError("--primitive-digit-renderer-eval only supports --task arithmetic")
    if not args.model:
        raise ValueError("--model is required for --primitive-digit-renderer-eval")
    device = torch.device(args.hf_device)
    dtype = _hf_dtype(args.hf_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    _patch_torch_custom_op_string_annotations()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    operator = PrimitiveLookupDigitWiseOperator(
        c_min=args.pure_arith_c_min,
        c_max=args.pure_arith_c_max,
    ).to(device)
    eval_cases = build_distill_cases(args.task, tokenizer, args.eval_cases, args.context_len, args.depth, args.seed + 10_000)
    num_digits = args.pure_digit_num_digits
    if num_digits <= 0:
        num_digits = _infer_pure_digit_num_digits(
            args.pure_arith_a_max,
            args.pure_arith_b_max,
            args.pure_arith_c_max,
            args.pure_arith_d_max,
        )
    t0 = time.time()
    input_modes = ["input_ids", "inputs_embeds"] if args.primitive_digit_renderer_input_mode == "both" else [args.primitive_digit_renderer_input_mode]
    eval_result = {}
    for input_mode in input_modes:
        print(f"Evaluating primitive digit-operator renderer input_mode={input_mode}...", flush=True)
        mode_result = evaluate_primitive_digit_renderer(
            model,
            tokenizer,
            operator,
            eval_cases,
            max_new_tokens=args.max_new_tokens,
            device=device,
            num_digits=num_digits,
            input_mode=input_mode,
        )
        eval_result[input_mode] = mode_result
        print(
            f"primitive_digit_renderer_mode={input_mode} slot={mode_result['slot_exact_accuracy'] * 100:.1f}% "
            f"decode={mode_result['accuracy'] * 100:.1f}% contains={mode_result['contains_accuracy'] * 100:.1f}% "
            f"overflow={mode_result['overflow_cases']}",
            flush=True,
        )
    result = {
        "args": vars(args),
        "num_digits": num_digits,
        "eval_cases": len(eval_cases),
        "eval": eval_result if args.primitive_digit_renderer_input_mode == "both" else eval_result[input_modes[0]],
        "elapsed_s": time.time() - t0,
    }
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save(result, args.checkpoint)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved results to {args.out_json}")


@torch.no_grad()
def generate_hf_textual_oracle(
    model,
    tokenizer,
    case: DistillCase,
    mode: str,
    max_new_tokens: int,
    device: torch.device,
    num_digits: int,
) -> str:
    model.eval()
    text = case.prompt + _textual_oracle_suffix(case, mode, num_digits)
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False), device=device, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    generated = []
    past = None
    for step in range(max_new_tokens):
        if step == 0:
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
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
def evaluate_textual_oracle(
    model,
    tokenizer,
    cases: list[DistillCase],
    mode: str,
    max_new_tokens: int,
    device: torch.device,
    num_digits: int,
) -> dict:
    details = []
    for case in cases:
        text = generate_hf_textual_oracle(model, tokenizer, case, mode, max_new_tokens, device, num_digits)
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
            f"  textual_oracle_eval mode={mode} expected={case.expected} answer_only={answer_only} "
            f"contains={contains_expected} hit={hit} raw={text[:48]!r}",
            flush=True,
        )
    acc = sum(int(x["hit"]) for x in details) / max(1, len(details))
    contains_acc = sum(int(x["contains_expected"]) for x in details) / max(1, len(details))
    return {"accuracy": acc, "contains_accuracy": contains_acc, "details": details}


def run_textual_oracle_eval(args) -> None:
    if args.task != "arithmetic":
        raise ValueError("--textual-oracle-eval only supports --task arithmetic")
    if not args.model:
        raise ValueError("--model is required for --textual-oracle-eval")
    device = torch.device(args.hf_device)
    dtype = _hf_dtype(args.hf_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    _patch_torch_custom_op_string_annotations()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    eval_cases = build_distill_cases(args.task, tokenizer, args.eval_cases, args.context_len, args.depth, args.seed + 10_000)
    num_digits = args.pure_digit_num_digits
    if num_digits <= 0:
        num_digits = _infer_pure_digit_num_digits(
            args.pure_arith_a_max,
            args.pure_arith_b_max,
            args.pure_arith_c_max,
            args.pure_arith_d_max,
        )
    t0 = time.time()
    print(f"Evaluating textual oracle mode={args.textual_oracle_mode}...", flush=True)
    eval_result = evaluate_textual_oracle(
        model,
        tokenizer,
        eval_cases,
        mode=args.textual_oracle_mode,
        max_new_tokens=args.max_new_tokens,
        device=device,
        num_digits=num_digits,
    )
    print(
        f"textual_oracle_eval_acc={eval_result['accuracy'] * 100:.1f}% "
        f"contains_acc={eval_result['contains_accuracy'] * 100:.1f}%",
        flush=True,
    )
    result = {
        "args": vars(args),
        "mode": args.textual_oracle_mode,
        "num_digits": num_digits,
        "eval_cases": len(eval_cases),
        "eval": eval_result,
        "elapsed_s": time.time() - t0,
    }
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save(result, args.checkpoint)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved results to {args.out_json}")


@torch.no_grad()
def generate_hf_embedding_oracle(
    model,
    tokenizer,
    case: DistillCase,
    mode: str,
    max_new_tokens: int,
    device: torch.device,
    num_digits: int,
) -> str:
    """Decode after appending canonical oracle text through input embeddings, not input_ids."""
    model.eval()
    embed = model.get_input_embeddings()
    prompt_ids = torch.tensor(tokenizer.encode(case.prompt, add_special_tokens=False), device=device, dtype=torch.long).unsqueeze(0)
    suffix = _textual_oracle_suffix(case, mode, num_digits)
    suffix_ids = torch.tensor(tokenizer.encode(suffix, add_special_tokens=False), device=device, dtype=torch.long).unsqueeze(0)
    prompt_embeds = embed(prompt_ids).detach()
    suffix_embeds = embed(suffix_ids).detach()
    inputs_embeds = torch.cat([prompt_embeds, suffix_embeds], dim=1)
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
def evaluate_embedding_oracle(
    model,
    tokenizer,
    cases: list[DistillCase],
    mode: str,
    max_new_tokens: int,
    device: torch.device,
    num_digits: int,
) -> dict:
    details = []
    for case in cases:
        text = generate_hf_embedding_oracle(model, tokenizer, case, mode, max_new_tokens, device, num_digits)
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
            f"  embedding_oracle_eval mode={mode} expected={case.expected} answer_only={answer_only} "
            f"contains={contains_expected} hit={hit} raw={text[:48]!r}",
            flush=True,
        )
    acc = sum(int(x["hit"]) for x in details) / max(1, len(details))
    contains_acc = sum(int(x["contains_expected"]) for x in details) / max(1, len(details))
    return {"accuracy": acc, "contains_accuracy": contains_acc, "details": details}


def run_embedding_oracle_eval(args) -> None:
    if args.task != "arithmetic":
        raise ValueError("--embedding-oracle-eval only supports --task arithmetic")
    if not args.model:
        raise ValueError("--model is required for --embedding-oracle-eval")
    device = torch.device(args.hf_device)
    dtype = _hf_dtype(args.hf_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    _patch_torch_custom_op_string_annotations()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    eval_cases = build_distill_cases(args.task, tokenizer, args.eval_cases, args.context_len, args.depth, args.seed + 10_000)
    num_digits = args.pure_digit_num_digits
    if num_digits <= 0:
        num_digits = _infer_pure_digit_num_digits(
            args.pure_arith_a_max,
            args.pure_arith_b_max,
            args.pure_arith_c_max,
            args.pure_arith_d_max,
        )
    t0 = time.time()
    print(f"Evaluating embedding oracle mode={args.embedding_oracle_mode}...", flush=True)
    eval_result = evaluate_embedding_oracle(
        model,
        tokenizer,
        eval_cases,
        mode=args.embedding_oracle_mode,
        max_new_tokens=args.max_new_tokens,
        device=device,
        num_digits=num_digits,
    )
    print(
        f"embedding_oracle_eval_acc={eval_result['accuracy'] * 100:.1f}% "
        f"contains_acc={eval_result['contains_accuracy'] * 100:.1f}%",
        flush=True,
    )
    result = {
        "args": vars(args),
        "mode": args.embedding_oracle_mode,
        "num_digits": num_digits,
        "eval_cases": len(eval_cases),
        "eval": eval_result,
        "elapsed_s": time.time() - t0,
    }
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save(result, args.checkpoint)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved results to {args.out_json}")


def _canonical_suffix_ids(tokenizer, case: DistillCase, mode: str, num_digits: int, device: torch.device) -> torch.Tensor:
    suffix = _textual_oracle_suffix(case, mode, num_digits)
    return torch.tensor(tokenizer.encode(suffix, add_special_tokens=False), device=device, dtype=torch.long)


def _max_canonical_suffix_len(tokenizer, cases: list[DistillCase], mode: str, num_digits: int) -> int:
    max_len = 1
    for case in cases:
        max_len = max(max_len, len(tokenizer.encode(_textual_oracle_suffix(case, mode, num_digits), add_special_tokens=False)))
    return max_len


def _canonical_embedding_alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cosine_weight: float,
) -> tuple[torch.Tensor, float, float]:
    mse = torch.nn.functional.mse_loss(pred.float(), target.float())
    cosine = 1.0 - torch.nn.functional.cosine_similarity(pred.float(), target.float(), dim=-1).mean()
    loss = mse + cosine_weight * cosine
    return loss, float(mse.item()), float(cosine.item())


def _canonical_embedding_token_ce_loss(
    pred: torch.Tensor,
    target_ids: torch.Tensor,
    normalized_embedding_table: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, float, int]:
    if temperature <= 0:
        raise ValueError("canonical embedding CE temperature must be > 0")
    logits = torch.nn.functional.linear(
        torch.nn.functional.normalize(pred.float(), dim=-1),
        normalized_embedding_table,
    ) / temperature
    loss = torch.nn.functional.cross_entropy(logits, target_ids)
    nearest = logits.argmax(dim=-1)
    correct = int((nearest == target_ids).sum().item())
    return loss, float(loss.item()), correct


def train_canonical_embedding_bridge(
    model,
    tokenizer,
    bridge: OracleStructuredDecodeBridge,
    cases: list[DistillCase],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    num_digits: int,
    c_max: int,
    mode: str,
    cosine_weight: float,
    supervision_weight: float,
    token_ce_weight: float,
    token_ce_temperature: float,
) -> list[dict]:
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    embed = model.get_input_embeddings()
    normalized_embedding_table = None
    if token_ce_weight > 0:
        normalized_embedding_table = torch.nn.functional.normalize(embed.weight.detach().float(), dim=-1)
    bridge.to(device=device, dtype=torch.float32).train()
    opt = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(1, epochs + 1):
        rng = random.Random(epoch)
        order = list(range(len(cases)))
        rng.shuffle(order)
        total_loss = 0.0
        total_mse = 0.0
        total_cos = 0.0
        total_ce = 0.0
        total_ce_correct = 0
        total_tokens = 0
        for start in range(0, len(order), batch_size):
            idxs = order[start:start + batch_size]
            batch_cases = [cases[i] for i in idxs]
            features = _oracle_structured_decode_features(batch_cases, num_digits, c_max, device)
            pred = bridge(features, dtype=torch.float32)
            losses = []
            batch_mse = 0.0
            batch_cos = 0.0
            batch_ce = 0.0
            batch_ce_correct = 0
            batch_tokens = 0
            for row, case in enumerate(batch_cases):
                target_ids = _canonical_suffix_ids(tokenizer, case, mode, num_digits, device)
                if target_ids.numel() > bridge.latent_steps:
                    raise ValueError(
                        f"canonical suffix length {target_ids.numel()} exceeds bridge slots {bridge.latent_steps}"
                    )
                target = embed(target_ids).detach().float()
                row_loss, row_mse, row_cos = _canonical_embedding_alignment_loss(
                    pred[row, :target_ids.numel()], target, cosine_weight
                )
                if token_ce_weight > 0:
                    assert normalized_embedding_table is not None
                    row_ce_loss, row_ce, row_ce_correct = _canonical_embedding_token_ce_loss(
                        pred[row, :target_ids.numel()],
                        target_ids,
                        normalized_embedding_table,
                        token_ce_temperature,
                    )
                    row_loss = row_loss + token_ce_weight * row_ce_loss
                    batch_ce += row_ce * int(target_ids.numel())
                    batch_ce_correct += row_ce_correct
                losses.append(row_loss * target_ids.numel())
                batch_mse += row_mse * int(target_ids.numel())
                batch_cos += row_cos * int(target_ids.numel())
                batch_tokens += int(target_ids.numel())
            loss = supervision_weight * torch.stack(losses).sum() / max(1, batch_tokens)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * batch_tokens
            total_mse += batch_mse
            total_cos += batch_cos
            total_ce += batch_ce
            total_ce_correct += batch_ce_correct
            total_tokens += batch_tokens
        row = {
            "epoch": epoch,
            "loss": total_loss / max(1, total_tokens),
            "mse": total_mse / max(1, total_tokens),
            "cosine_distance": total_cos / max(1, total_tokens),
            "token_ce": total_ce / max(1, total_tokens),
            "token_ce_accuracy": _safe_div(total_ce_correct, total_tokens),
            "tokens": total_tokens,
        }
        history.append(row)
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            print(
                f"  canonical_bridge_epoch={epoch:03d} loss={row['loss']:.6f} "
                f"mse={row['mse']:.6f} cos={row['cosine_distance']:.6f} "
                f"ce={row['token_ce']:.6f} ce_acc={row['token_ce_accuracy'] * 100:.1f}%",
                flush=True,
            )
    return history


@torch.no_grad()
def evaluate_canonical_embedding_retrieval(
    model,
    tokenizer,
    bridge: OracleStructuredDecodeBridge,
    cases: list[DistillCase],
    batch_size: int,
    device: torch.device,
    num_digits: int,
    c_max: int,
    mode: str,
) -> dict:
    model.eval()
    bridge.eval()
    embed = model.get_input_embeddings()
    table = torch.nn.functional.normalize(embed.weight.detach().float(), dim=-1)
    total = 0
    correct = 0
    total_mse = 0.0
    total_cos = 0.0
    for start in range(0, len(cases), batch_size):
        batch_cases = cases[start:start + batch_size]
        features = _oracle_structured_decode_features(batch_cases, num_digits, c_max, device)
        pred = bridge(features, dtype=torch.float32)
        for row, case in enumerate(batch_cases):
            target_ids = _canonical_suffix_ids(tokenizer, case, mode, num_digits, device)
            target = embed(target_ids).detach().float()
            slot_pred = pred[row, :target_ids.numel()].float()
            total_mse += float(torch.nn.functional.mse_loss(slot_pred, target, reduction="sum").item()) / target.size(-1)
            total_cos += float((1.0 - torch.nn.functional.cosine_similarity(slot_pred, target, dim=-1)).sum().item())
            logits = torch.nn.functional.linear(torch.nn.functional.normalize(slot_pred, dim=-1), table)
            nearest = logits.argmax(dim=-1)
            correct += int((nearest == target_ids).sum().item())
            total += int(target_ids.numel())
    return {
        "tokens": total,
        "nearest_token_accuracy": _safe_div(correct, total),
        "mse": total_mse / max(1, total),
        "cosine_distance": total_cos / max(1, total),
    }


@torch.no_grad()
def generate_hf_with_canonical_embedding_bridge(
    model,
    tokenizer,
    bridge: OracleStructuredDecodeBridge,
    case: DistillCase,
    mode: str,
    max_new_tokens: int,
    device: torch.device,
    num_digits: int,
    c_max: int,
    decode_mode: str,
) -> str:
    model.eval()
    bridge.eval()
    prompt_ids = torch.tensor(tokenizer.encode(case.prompt, add_special_tokens=False), device=device, dtype=torch.long).unsqueeze(0)
    embed = model.get_input_embeddings()
    prompt_embeds = embed(prompt_ids).detach()
    features = _oracle_structured_decode_features([case], num_digits, c_max, device)
    target_ids = _canonical_suffix_ids(tokenizer, case, mode, num_digits, device)
    latents = bridge(features, dtype=prompt_embeds.dtype)[:, :target_ids.numel()]
    if decode_mode == "nearest":
        table = torch.nn.functional.normalize(embed.weight.detach().float(), dim=-1)
        nearest_logits = torch.nn.functional.linear(torch.nn.functional.normalize(latents.float(), dim=-1), table)
        nearest_ids = nearest_logits.argmax(dim=-1)
        latents = embed(nearest_ids).detach()
    elif decode_mode != "continuous":
        raise ValueError(f"unknown canonical embedding decode mode: {decode_mode}")
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
def evaluate_canonical_embedding_bridge_decode(
    model,
    tokenizer,
    bridge: OracleStructuredDecodeBridge,
    cases: list[DistillCase],
    mode: str,
    max_new_tokens: int,
    device: torch.device,
    num_digits: int,
    c_max: int,
    decode_mode: str,
) -> dict:
    details = []
    for case in cases:
        text = generate_hf_with_canonical_embedding_bridge(
            model, tokenizer, bridge, case, mode, max_new_tokens, device, num_digits, c_max, decode_mode
        )
        answer_only = extract_answer_only(case.task, text)
        contains_expected = contains_expected_answer(case.task, text, case.expected)
        hit = answer_only == case.expected
        details.append({
            "expected": case.expected,
            "answer_only": answer_only,
            "contains_expected": contains_expected,
            "hit": hit,
            "raw": text[:160],
            "decode_mode": decode_mode,
            "meta": case.meta,
        })
        print(
            f"  canonical_bridge_eval mode={decode_mode} expected={case.expected} answer_only={answer_only} "
            f"contains={contains_expected} hit={hit} raw={text[:48]!r}",
            flush=True,
        )
    acc = sum(int(x["hit"]) for x in details) / max(1, len(details))
    contains_acc = sum(int(x["contains_expected"]) for x in details) / max(1, len(details))
    return {"accuracy": acc, "contains_accuracy": contains_acc, "decode_mode": decode_mode, "details": details}


def run_canonical_embedding_bridge(args) -> None:
    if args.task != "arithmetic":
        raise ValueError("--canonical-embedding-bridge only supports --task arithmetic")
    if not args.model:
        raise ValueError("--model is required for --canonical-embedding-bridge")
    device = torch.device(args.hf_device)
    dtype = _hf_dtype(args.hf_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    num_digits = args.pure_digit_num_digits
    if num_digits <= 0:
        num_digits = _infer_pure_digit_num_digits(
            args.pure_arith_a_max,
            args.pure_arith_b_max,
            args.pure_arith_c_max,
            args.pure_arith_d_max,
        )
    sample_features = _oracle_structured_decode_features(train_cases[:1], num_digits, args.pure_arith_c_max, device)
    target_tokens = args.canonical_embedding_target_tokens
    if target_tokens <= 0:
        target_tokens = max(
            _max_canonical_suffix_len(tokenizer, train_cases, args.canonical_embedding_mode, num_digits),
            _max_canonical_suffix_len(tokenizer, eval_cases, args.canonical_embedding_mode, num_digits),
        )
    bridge = OracleStructuredDecodeBridge(
        feature_dim=sample_features.size(-1),
        hidden_size=hidden_size,
        latent_steps=target_tokens,
        bridge_hidden_size=args.decode_bridge_hidden_size,
        depth=args.decode_bridge_depth,
    )
    t0 = time.time()
    print(
        f"Training canonical embedding bridge mode={args.canonical_embedding_mode} target_tokens={target_tokens}...",
        flush=True,
    )
    history = train_canonical_embedding_bridge(
        model,
        tokenizer,
        bridge,
        train_cases,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        num_digits=num_digits,
        c_max=args.pure_arith_c_max,
        mode=args.canonical_embedding_mode,
        cosine_weight=args.canonical_embedding_cosine_weight,
        supervision_weight=args.bridge_supervision_weight,
        token_ce_weight=args.canonical_embedding_token_ce_weight,
        token_ce_temperature=args.canonical_embedding_token_ce_temperature,
    )
    train_retrieval = evaluate_canonical_embedding_retrieval(
        model, tokenizer, bridge, train_cases, args.batch_size, device, num_digits, args.pure_arith_c_max, args.canonical_embedding_mode
    )
    eval_retrieval = evaluate_canonical_embedding_retrieval(
        model, tokenizer, bridge, eval_cases, args.batch_size, device, num_digits, args.pure_arith_c_max, args.canonical_embedding_mode
    )
    eval_result = None
    if not args.skip_eval:
        eval_modes = ["continuous", "nearest"] if args.canonical_embedding_decode_mode == "both" else [args.canonical_embedding_decode_mode]
        eval_results = {}
        for decode_mode in eval_modes:
            print(f"Evaluating canonical embedding bridge decode mode={decode_mode}...", flush=True)
            mode_result = evaluate_canonical_embedding_bridge_decode(
                model,
                tokenizer,
                bridge,
                eval_cases,
                mode=args.canonical_embedding_mode,
                max_new_tokens=args.max_new_tokens,
                device=device,
                num_digits=num_digits,
                c_max=args.pure_arith_c_max,
                decode_mode=decode_mode,
            )
            eval_results[decode_mode] = mode_result
            print(
                f"canonical_bridge_eval_mode={decode_mode} acc={mode_result['accuracy'] * 100:.1f}% "
                f"contains_acc={mode_result['contains_accuracy'] * 100:.1f}%",
                flush=True,
            )
        eval_result = eval_results if args.canonical_embedding_decode_mode == "both" else eval_results[eval_modes[0]]
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save({
        "bridge": bridge.state_dict(),
        "hidden_size": hidden_size,
        "feature_dim": sample_features.size(-1),
        "num_digits": num_digits,
        "target_tokens": target_tokens,
        "args": vars(args),
        "history": history,
        "train_retrieval": train_retrieval,
        "eval_retrieval": eval_retrieval,
        "eval": eval_result,
    }, args.checkpoint)
    result = {
        "args": vars(args),
        "hidden_size": hidden_size,
        "feature_dim": sample_features.size(-1),
        "num_digits": num_digits,
        "target_tokens": target_tokens,
        "history": history,
        "train_retrieval": train_retrieval,
        "eval_retrieval": eval_retrieval,
        "eval": eval_result,
        "elapsed_s": time.time() - t0,
        "checkpoint": args.checkpoint,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved results to {args.out_json}")


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
    if args.arithmetic_state_machine_weight > 0 and args.task != "arithmetic":
        raise ValueError("--arithmetic-state-machine-weight only supports --task arithmetic")
    if args.arithmetic_state_machine_weight > 0 and args.projector_kind != "arithmetic_state_machine":
        raise ValueError("--arithmetic-state-machine-weight requires --projector-kind arithmetic_state_machine")
    projector = build_trainable_projector(
        hidden_size,
        args.projector_kind,
        max_steps=max(args.max_latent_steps, args.latent_steps),
        state_min_value=args.state_min_value,
        state_max_value=args.state_max_value,
    )
    state_probe = None
    if args.state_supervision_weight > 0:
        state_probe = ArithmeticStateProbe(
            hidden_size,
            max_steps=max(args.max_latent_steps, args.latent_steps),
            min_value=args.state_min_value,
            max_value=args.state_max_value,
        )
    numeric_state_embedding = None
    if args.state_embedding_loss_weight > 0:
        numeric_state_embedding = NumericStateEmbedding(
            hidden_size,
            min_value=args.state_min_value,
            max_value=args.state_max_value,
        )
    t0 = time.time()
    print("Training HF teacher-forcing latent projector...", flush=True)
    history = train_projector_hf_teacher_forcing(
        model,
        tokenizer,
        projector,
        state_probe,
        numeric_state_embedding,
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
        state_supervision_weight=args.state_supervision_weight,
        state_min_value=args.state_min_value,
        state_max_value=args.state_max_value,
        state_embedding_loss_weight=args.state_embedding_loss_weight,
        state_embedding_mse_weight=args.state_embedding_mse_weight,
        arithmetic_state_machine_weight=args.arithmetic_state_machine_weight,
        arithmetic_state_machine_value_weight=args.arithmetic_state_machine_value_weight,
        arithmetic_state_machine_phase_weight=args.arithmetic_state_machine_phase_weight,
        arithmetic_state_machine_source=args.arithmetic_state_machine_source,
        arithmetic_state_machine_repeat_final=not args.arithmetic_state_machine_no_repeat_final,
    )
    eval_result = None
    state_machine_eval = None
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
        if args.arithmetic_state_machine_weight > 0:
            state_machine_eval = evaluate_arithmetic_state_machine_hf(
                model,
                tokenizer,
                projector,
                eval_cases,
                args.batch_size,
                device,
                args.latent_steps,
                args.arithmetic_state_machine_source,
                not args.arithmetic_state_machine_no_repeat_final,
                args.state_min_value,
                args.state_max_value,
                args.arithmetic_state_machine_value_weight,
                args.arithmetic_state_machine_phase_weight,
            )
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save({
        "projector": projector.state_dict(),
        "state_probe": state_probe.state_dict() if state_probe is not None else None,
        "numeric_state_embedding": numeric_state_embedding.state_dict() if numeric_state_embedding is not None else None,
        "hidden_size": hidden_size,
        "args": vars(args),
        "history": history,
        "state_machine_eval": state_machine_eval,
    }, args.checkpoint)
    result = {
        "args": vars(args),
        "hidden_size": hidden_size,
        "history": history,
        "eval": eval_result,
        "state_machine_eval": state_machine_eval,
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
    p.add_argument("--model", type=str, default="")
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
    p.add_argument("--projector-kind", choices=["shared", "stepwise", "arithmetic_state_machine"], default="shared")
    p.add_argument("--max-latent-steps", type=int, default=16)
    p.add_argument("--latent-step-curriculum", choices=["none", "linear"], default="none")
    p.add_argument("--state-supervision-weight", type=float, default=0.0)
    p.add_argument("--state-min-value", type=int, default=-256)
    p.add_argument("--state-max-value", type=int, default=2048)
    p.add_argument("--state-embedding-loss-weight", type=float, default=0.0)
    p.add_argument("--state-embedding-mse-weight", type=float, default=0.1)
    p.add_argument("--arithmetic-state-machine-weight", type=float, default=0.0)
    p.add_argument("--arithmetic-state-machine-value-weight", type=float, default=1.0)
    p.add_argument("--arithmetic-state-machine-phase-weight", type=float, default=0.2)
    p.add_argument("--arithmetic-state-machine-source", choices=["input", "output"], default="output")
    p.add_argument("--arithmetic-state-machine-no-repeat-final", action="store_true")
    p.add_argument("--pure-latent-arithmetic", action="store_true")
    p.add_argument("--pure-hidden-size", type=int, default=128)
    p.add_argument("--pure-numeric-representation", choices=["id_embedding", "structured"], default="id_embedding")
    p.add_argument("--pure-arith-a-min", type=int, default=20)
    p.add_argument("--pure-arith-a-max", type=int, default=90)
    p.add_argument("--pure-arith-b-min", type=int, default=10)
    p.add_argument("--pure-arith-b-max", type=int, default=80)
    p.add_argument("--pure-arith-c-min", type=int, default=2)
    p.add_argument("--pure-arith-c-max", type=int, default=9)
    p.add_argument("--pure-arith-d-min", type=int, default=10)
    p.add_argument("--pure-arith-d-max", type=int, default=120)
    p.add_argument("--pure-arith-value-weight", type=float, default=1.0)
    p.add_argument("--pure-arith-phase-weight", type=float, default=0.0)
    p.add_argument("--pure-arith-regression-weight", type=float, default=0.0)
    p.add_argument("--pure-arith-digit-weight", type=float, default=0.0)
    p.add_argument("--pure-digit-arithmetic", action="store_true")
    p.add_argument("--pure-digit-mode", choices=["deterministic", "trainable", "lookup"], default="deterministic")
    p.add_argument("--pure-digit-num-digits", type=int, default=0)
    p.add_argument("--pure-digit-operator-hidden-size", type=int, default=64)
    p.add_argument("--pure-digit-operator-depth", type=int, default=2)
    p.add_argument("--pure-digit-combo-heldout-frac", type=float, default=0.2)
    p.add_argument("--pure-digit-combo-split-seed", type=int, default=0)
    p.add_argument("--pure-digit-carry-loss-weight", type=float, default=1.0)
    p.add_argument("--oracle-decode-bridge", action="store_true")
    p.add_argument("--decode-bridge-hidden-size", type=int, default=1024)
    p.add_argument("--decode-bridge-depth", type=int, default=2)
    p.add_argument("--textual-oracle-eval", action="store_true")
    p.add_argument("--textual-oracle-mode", choices=["final_answer", "digit_state", "msd_digits", "final_abs", "natural_summary"], default="final_answer")
    p.add_argument("--embedding-oracle-eval", action="store_true")
    p.add_argument("--embedding-oracle-mode", choices=["final_abs", "natural_summary", "msd_digits"], default="final_abs")
    p.add_argument("--digit-slot-renderer-eval", action="store_true")
    p.add_argument("--digit-slot-renderer-input-mode", choices=["input_ids", "inputs_embeds", "both"], default="inputs_embeds")
    p.add_argument("--digit-slot-predictor", action="store_true")
    p.add_argument("--digit-slot-predictor-hidden-size", type=int, default=256)
    p.add_argument("--digit-slot-predictor-depth", type=int, default=2)
    p.add_argument("--digit-slot-digit-loss-weight", type=float, default=1.0)
    p.add_argument("--digit-slot-decode-input-mode", choices=["input_ids", "inputs_embeds", "both"], default="inputs_embeds")
    p.add_argument("--primitive-digit-renderer-eval", action="store_true")
    p.add_argument("--primitive-digit-renderer-input-mode", choices=["input_ids", "inputs_embeds", "both"], default="inputs_embeds")
    p.add_argument("--operator-vm-benchmark", action="store_true")
    p.add_argument("--operator-vm-input-mode", choices=["input_ids", "inputs_embeds"], default="inputs_embeds")
    p.add_argument("--operator-vm-baseline-max-new-tokens", type=int, default=0)
    p.add_argument("--operator-vm-final-max-new-tokens", type=int, default=0)
    p.add_argument("--minimal-embedding-suffix-benchmark", action="store_true")
    p.add_argument(
        "--minimal-embedding-suffix-modes",
        nargs="+",
        choices=["final_abs", "sign_final_abs", "signed_answer", "digits_msd"],
        default=["final_abs", "sign_final_abs", "signed_answer"],
    )
    p.add_argument("--discrete-lookup-renderer", action="store_true")
    p.add_argument("--vocab-bottleneck-renderer-bridge", action="store_true")
    p.add_argument("--vocab-renderer-hidden-size", type=int, default=256)
    p.add_argument("--vocab-renderer-depth", type=int, default=2)
    p.add_argument("--vocab-renderer-length-ce-weight", type=float, default=1.0)
    p.add_argument("--compact-renderer-bridge", action="store_true")
    p.add_argument("--compact-renderer-hidden-size", type=int, default=1024)
    p.add_argument("--compact-renderer-depth", type=int, default=2)
    p.add_argument("--compact-renderer-cosine-weight", type=float, default=1.0)
    p.add_argument("--compact-renderer-token-ce-weight", type=float, default=0.0)
    p.add_argument("--compact-renderer-token-ce-temperature", type=float, default=0.05)
    p.add_argument("--compact-renderer-run-variable", action="store_true")
    p.add_argument("--compact-renderer-fixed-ks", nargs="+", type=int, default=[8, 4, 2, 1])
    p.add_argument("--compact-renderer-fixed-epochs", type=int, default=0)
    p.add_argument("--compact-renderer-fixed-lr", type=float, default=0.0)
    p.add_argument("--compact-renderer-include-c21-reference", action="store_true")
    p.add_argument("--canonical-embedding-bridge", action="store_true")
    p.add_argument("--canonical-embedding-mode", choices=["final_abs", "natural_summary", "msd_digits"], default="final_abs")
    p.add_argument("--canonical-embedding-cosine-weight", type=float, default=1.0)
    p.add_argument("--canonical-embedding-token-ce-weight", type=float, default=0.0)
    p.add_argument("--canonical-embedding-token-ce-temperature", type=float, default=0.05)
    p.add_argument("--canonical-embedding-target-tokens", type=int, default=0)
    p.add_argument("--canonical-embedding-decode-mode", choices=["continuous", "nearest", "both"], default="continuous")
    p.add_argument("--bridge-supervision-weight", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.pure_digit_arithmetic and args.pure_latent_arithmetic:
        raise ValueError("choose only one of --pure-digit-arithmetic or --pure-latent-arithmetic")
    if args.pure_digit_arithmetic:
        run_pure_digit_arithmetic_benchmark(args)
        return
    if args.pure_latent_arithmetic:
        run_pure_latent_arithmetic_benchmark(args)
        return
    if args.oracle_decode_bridge:
        run_oracle_decode_bridge(args)
        return
    if args.textual_oracle_eval:
        run_textual_oracle_eval(args)
        return
    if args.embedding_oracle_eval:
        run_embedding_oracle_eval(args)
        return
    if args.digit_slot_renderer_eval:
        run_digit_slot_renderer_eval(args)
        return
    if args.digit_slot_predictor:
        run_digit_slot_predictor(args)
        return
    if args.primitive_digit_renderer_eval:
        run_primitive_digit_renderer_eval(args)
        return
    if args.operator_vm_benchmark:
        run_operator_vm_benchmark(args)
        return
    if args.minimal_embedding_suffix_benchmark:
        run_minimal_embedding_suffix_benchmark(args)
        return
    if args.discrete_lookup_renderer:
        run_discrete_lookup_renderer(args)
        return
    if args.vocab_bottleneck_renderer_bridge:
        run_vocab_bottleneck_renderer_bridge(args)
        return
    if args.compact_renderer_bridge:
        run_compact_renderer_bridge(args)
        return
    if args.canonical_embedding_bridge:
        run_canonical_embedding_bridge(args)
        return
    if not args.model:
        raise ValueError("--model is required unless a pure arithmetic mode is set")
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
        state_min_value=args.state_min_value,
        state_max_value=args.state_max_value,
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
