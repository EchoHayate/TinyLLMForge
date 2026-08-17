from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
import types
from pathlib import Path

import torch
import torch.nn as nn

from probe_am_compact_cache import (
    attention_output,
    attention_weights,
    fit_values,
    legacy_pkv,
    normalize_cache,
    pick_device,
    pick_dtype,
    select_highest,
)
from train_recovery_probe import train_compact_values


TASKS = [
    {
        "id": "phase1_method",
        "question": "Which technique is listed as Phase 1 in the route?",
        "choices": {
            "A": "Quest dynamic page top-k",
            "B": "SnapKV prompt compression",
            "C": "Speculative decoding",
            "D": "CPU offload",
        },
        "answer": "A",
    },
    {
        "id": "quest_selects",
        "question": "During decode, what does Quest select?",
        "choices": {
            "A": "Top-k KV pages or blocks",
            "B": "Tokenizer merge rules",
            "C": "Optimizer states",
            "D": "Vocabulary shards",
        },
        "answer": "A",
    },
    {
        "id": "phase2_eval",
        "question": "Which phase is the needle-in-haystack evaluation?",
        "choices": {
            "A": "Phase 1",
            "B": "Phase 2",
            "C": "Phase 3",
            "D": "Phase 4",
        },
        "answer": "B",
    },
    {
        "id": "phase3_method",
        "question": "Which method is planned for Phase 3?",
        "choices": {
            "A": "SnapKV",
            "B": "Quest",
            "C": "Beam search",
            "D": "W4A8 weight-only quantization",
        },
        "answer": "A",
    },
    {
        "id": "decode_bottleneck",
        "question": "What bottleneck is highlighted for long-context decode?",
        "choices": {
            "A": "KV cache load bandwidth",
            "B": "Disk IO",
            "C": "Vocabulary sorting",
            "D": "Embedding table lookup only",
        },
        "answer": "A",
    },
]


class RuntimeState:
    def __init__(self):
        self.banks = {}
        self.prefix_len = 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--text-file", required=True)
    p.add_argument("--task-file", default=None, help="Optional JSON task list overriding the built-in smoke tasks.")
    p.add_argument("--policy-dir", required=True)
    p.add_argument("--adaptive-policy-file", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--thresholds", default="0.35,0.50")
    p.add_argument("--max-prompt-tokens", type=int, default=1536)
    p.add_argument("--max-doc-tokens", type=int, default=1200)
    p.add_argument("--bank-train-tokens", type=int, default=512)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--bank-method", default="ridge", choices=["ridge", "learned_values"])
    p.add_argument("--bank-train-epochs", type=int, default=100)
    p.add_argument("--bank-train-lr", type=float, default=1e-3)
    p.add_argument("--bank-weight-decay", type=float, default=1e-4)
    p.add_argument("--task-limit", type=int, default=0)
    p.add_argument("--min-baseline-accuracy", type=float, default=0.80)
    p.add_argument(
        "--choice-scoring",
        default="text_only",
        choices=["letter", "space_letter", "letter_dot_text", "text_only", "space_text"],
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    return p.parse_args()


def parse_floats(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


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


def load_policy(policy_dir, threshold):
    rows = []
    with (Path(policy_dir) / "policy_rows.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if abs(float(row["threshold"]) - threshold) < 1e-9 and row["action"] == "compact":
                rows.append({
                    "layer": int(row["layer"]),
                    "kv_head": int(row["kv_head"]),
                    "budget": int(float(row["selected_budget"])),
                    "quality": float(row["quality"]),
                })
    return rows


def parse_head_key(text):
    layer_text, head_text = str(text).split(":", 1)
    return int(layer_text), int(head_text)


def load_adaptive_policy(path):
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    overrides = payload.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("adaptive policy overrides must be an object keyed by task id")
    normalized = {**payload, "overrides": {}}
    for task_id, override in overrides.items():
        if not isinstance(override, dict):
            raise ValueError(f"adaptive override for {task_id!r} must be an object")
        normalized["overrides"][str(task_id)] = {
            **override,
            "drop_heads": {parse_head_key(item) for item in override.get("drop_heads", [])},
            "full_heads": {parse_head_key(item) for item in override.get("full_heads", [])},
        }
    return normalized


def apply_adaptive_policy(policy_heads, adaptive_policy, task_id):
    if not adaptive_policy:
        return list(policy_heads)
    override = adaptive_policy.get("overrides", {}).get(str(task_id))
    if not override:
        return list(policy_heads)
    drop_heads = set(override.get("drop_heads", set())) | set(override.get("full_heads", set()))
    if not drop_heads:
        return list(policy_heads)
    return [
        row for row in policy_heads
        if (int(row["layer"]), int(row["kv_head"])) not in drop_heads
    ]


def effective_entry_saving_fraction(policy_heads, seq_len=1536, num_layers=28, num_kv_heads=8):
    original_entries = num_layers * num_kv_heads * seq_len
    compact_entries = sum(min(int(row["budget"]), seq_len) for row in policy_heads)
    compact_entries += (num_layers * num_kv_heads - len(policy_heads)) * seq_len
    return 1.0 - compact_entries / original_entries


def format_task_prompt(doc_text, task):
    choices = "\n".join(f"{k}. {v}" for k, v in task["choices"].items())
    return (
        "Read the document and answer the multiple-choice question. "
        "Answer with the letter only.\n\n"
        f"Document:\n{doc_text}\n\n"
        f"Question: {task['question']}\n"
        f"{choices}\n"
        "Answer:"
    )


def format_candidate(choice_key, choice_text, scoring):
    if scoring == "letter":
        return choice_key
    if scoring == "space_letter":
        return " " + choice_key
    if scoring == "letter_dot_text":
        return f"{choice_key}. {choice_text}"
    if scoring == "text_only":
        return choice_text
    if scoring == "space_text":
        return " " + choice_text
    raise ValueError(f"unknown choice scoring mode: {scoring}")


def validate_task(task, index):
    if not isinstance(task, dict):
        raise ValueError(f"task #{index} must be an object")
    for key in ("id", "question", "choices", "answer"):
        if key not in task:
            raise ValueError(f"task #{index} missing required field: {key}")
    if not isinstance(task["choices"], dict) or len(task["choices"]) < 2:
        raise ValueError(f"task #{index} choices must be an object with at least two choices")
    answer = task["answer"]
    if answer not in task["choices"]:
        raise ValueError(f"task #{index} answer {answer!r} is not in choices")
    normalized_choices = {}
    for choice_key, choice_text in task["choices"].items():
        normalized_choices[str(choice_key)] = str(choice_text)
    return {
        "id": str(task["id"]),
        "question": str(task["question"]),
        "choices": normalized_choices,
        "answer": str(answer),
    }


def load_tasks(task_file):
    if task_file is None:
        return TASKS
    payload = json.loads(Path(task_file).read_text(encoding="utf-8"))
    tasks = payload["tasks"] if isinstance(payload, dict) and "tasks" in payload else payload
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("--task-file must contain a non-empty task list or an object with a non-empty 'tasks' list")
    return [validate_task(task, index) for index, task in enumerate(tasks)]


def build_prompts(tokenizer, text, tasks, max_doc_tokens, max_prompt_tokens):
    doc_ids = tokenizer(text, add_special_tokens=False)["input_ids"][:max_doc_tokens]
    doc_text = tokenizer.decode(doc_ids)
    prompts = []
    for task in tasks:
        prompt = format_task_prompt(doc_text, task)
        ids = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_prompt_tokens)["input_ids"]
        prompt = tokenizer.decode(ids)
        prompts.append({**task, "prompt": prompt, "prompt_tokens": len(ids)})
    return prompts


def install_attention_patch(model, runtime):
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ):
        original_forward = self._light_doc_cache_original_forward
        layer_banks = runtime.banks.get(int(self.layer_idx), {})
        if not layer_banks:
            return original_forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        key_rep = repeat_kv(key_states, self.num_key_value_groups)
        value_rep = repeat_kv(value_states, self.num_key_value_groups)

        value_rep = value_rep.clone()
        num_q_heads = query_states.shape[1]
        key_len = key_rep.shape[-2]
        allowed_by_q = torch.ones((num_q_heads, key_len), dtype=torch.bool, device=query_states.device)
        for kv_head, bank in layer_banks.items():
            selected = bank["selected"].to(query_states.device)
            compact_v = bank["compact_v"].to(device=query_states.device, dtype=value_rep.dtype)
            allowed = torch.zeros((key_len,), dtype=torch.bool, device=query_states.device)
            selected = selected[selected < key_len]
            allowed[selected] = True
            if runtime.prefix_len < key_len:
                allowed[runtime.prefix_len:key_len] = True
            q_start = int(kv_head) * self.num_key_value_groups
            q_end = q_start + self.num_key_value_groups
            allowed_by_q[q_start:q_end] = allowed.unsqueeze(0).expand(q_end - q_start, key_len)
            if selected.numel() > 0:
                compact_v = compact_v[: selected.numel()]
                value_rep[:, q_start:q_end, selected, :] = compact_v.view(1, 1, selected.numel(), -1)

        attn_weights = torch.matmul(query_states, key_rep.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = attn_weights.masked_fill(~allowed_by_q.view(1, allowed_by_q.shape[0], 1, -1), float("-inf"))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_rep)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    for layer in model.model.layers:
        if not hasattr(layer.self_attn, "_light_doc_cache_original_forward"):
            layer.self_attn._light_doc_cache_original_forward = layer.self_attn.forward
        layer.self_attn.forward = types.MethodType(patched_forward, layer.self_attn)


def choose_train_indices(seq_len, max_train, device):
    idx = torch.arange(seq_len, device=device)
    if max_train > 0 and idx.numel() > max_train:
        take = torch.linspace(0, idx.numel() - 1, max_train, device=device).round().long()
        idx = idx[take].unique(sorted=True)
    return idx


@torch.inference_mode()
def collect_layer_qkv(model, out, layer_idx, expected_kv):
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    h = out.hidden_states[layer_idx]
    device = h.device
    seq_len = h.shape[1]
    attn = model.model.layers[layer_idx].self_attn
    hidden_shape = (*h.shape[:-1], -1, attn.head_dim)
    q = attn.q_norm(attn.q_proj(h).view(hidden_shape)).transpose(1, 2)
    full_pos = torch.arange(seq_len, device=device).unsqueeze(0)
    pos_emb = model.model.rotary_emb(h, full_pos)
    dummy_k = torch.zeros((h.shape[0], expected_kv, seq_len, attn.head_dim), device=device, dtype=q.dtype)
    q, _ = apply_rotary_pos_emb(q, dummy_k, *pos_emb)

    k, v = legacy_pkv(out.past_key_values)[layer_idx]
    k = normalize_cache(k, expected_kv)
    v = normalize_cache(v, expected_kv)
    return q[0].float(), k[0].float(), v[0].float()


def build_banks_for_prompt(model, input_ids, policy_heads, args):
    device = input_ids.device
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True, output_hidden_states=True, return_dict=True)
    cache = out.past_key_values
    prefix_len = int(input_ids.shape[1])
    expected_kv = getattr(model.config, "num_key_value_heads", None) or getattr(model.config, "num_attention_heads", None)
    group = int(getattr(model.config, "num_attention_heads") // expected_kv)
    positions = torch.arange(prefix_len, device=device)
    train_idx = choose_train_indices(prefix_len, args.bank_train_tokens, device)
    val_idx = train_idx

    heads_by_layer = {}
    for row in policy_heads:
        heads_by_layer.setdefault(row["layer"], []).append(row)

    banks = {}
    t0 = time.perf_counter()
    for layer_idx, head_rows in heads_by_layer.items():
        q_all, k_all, v_all = collect_layer_qkv(model, out, layer_idx, expected_kv)
        banks[layer_idx] = {}
        for row in head_rows:
            kv_head = row["kv_head"]
            budget = min(int(row["budget"]), prefix_len)
            q_group = q_all[kv_head * group:(kv_head + 1) * group]
            k = k_all[kv_head]
            v = v_all[kv_head]
            probs_train = attention_weights(q_group[:, train_idx, :], k, positions[train_idx], positions)
            selected = select_highest(probs_train, positions, budget).to(device)
            k_sel = k[selected]
            pos_sel = positions[selected]
            true_train = attention_output(q_group[:, train_idx, :], k, v, positions[train_idx], positions)
            attn_train = attention_weights(q_group[:, train_idx, :], k_sel, positions[train_idx], pos_sel)
            compact_v = fit_values(attn_train, true_train, args.ridge).to(device)
            bank_method = args.bank_method
            if args.bank_method == "learned_values":
                learned = train_compact_values(
                    attn_train=attn_train,
                    attn_val=attn_train,
                    y_train=true_train,
                    y_val=true_train,
                    init_values=compact_v,
                    epochs=args.bank_train_epochs,
                    lr=args.bank_train_lr,
                    weight_decay=args.bank_weight_decay,
                )
                compact_v = learned["values"].to(device)
            banks[layer_idx][kv_head] = {
                "selected": selected.detach(),
                "compact_v": compact_v.detach(),
                "budget": int(selected.numel()),
                "method": bank_method,
            }
    return cache, banks, prefix_len, time.perf_counter() - t0


@torch.inference_mode()
def prepare_cache(model, input_ids):
    return model(input_ids=input_ids, use_cache=True, return_dict=True).past_key_values


def clone_cache(cache):
    return copy.deepcopy(cache)


@torch.inference_mode()
def score_candidate(model, cache, last_token_id, candidate_ids, runtime, banks, prefix_len):
    runtime.banks = banks
    runtime.prefix_len = prefix_len
    past = clone_cache(cache)
    out = model(input_ids=last_token_id, past_key_values=past, use_cache=True, return_dict=True)
    logits = out.logits[:, -1, :]
    past = out.past_key_values
    score = 0.0
    for i, token_id in enumerate(candidate_ids):
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        score += float(log_probs[0, int(token_id)])
        if i + 1 < len(candidate_ids):
            next_id = torch.tensor([[int(token_id)]], dtype=torch.long, device=last_token_id.device)
            out = model(input_ids=next_id, past_key_values=past, use_cache=True, return_dict=True)
            logits = out.logits[:, -1, :]
            past = out.past_key_values
    runtime.banks = {}
    return score


def rank_scores(scores):
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def evaluate_task(model, tokenizer, prompt, choices, answer, runtime, policy_heads, threshold, args):
    device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    if ids.shape[1] < 2:
        raise RuntimeError("prompt must contain at least two tokens")
    prefix_ids = ids[:, :-1]
    last_token_id = ids[:, -1:]
    candidate_ids = {
        key: tokenizer(format_candidate(key, value, args.choice_scoring), add_special_tokens=False)["input_ids"]
        for key, value in choices.items()
    }

    baseline_cache = prepare_cache(model, prefix_ids)
    baseline_scores = {
        key: score_candidate(model, baseline_cache, last_token_id, cand, runtime, {}, int(prefix_ids.shape[1]))
        for key, cand in candidate_ids.items()
    }
    baseline_ranked = rank_scores(baseline_scores)

    if policy_heads:
        compact_cache, banks, prefix_len, bank_build_s = build_banks_for_prompt(
            model, prefix_ids, policy_heads, args
        )
    else:
        compact_cache, banks, prefix_len, bank_build_s = baseline_cache, {}, int(prefix_ids.shape[1]), 0.0

    compact_scores = {
        key: score_candidate(model, compact_cache, last_token_id, cand, runtime, banks, prefix_len)
        for key, cand in candidate_ids.items()
    }
    compact_ranked = rank_scores(compact_scores)
    return {
        "threshold": threshold,
        "prompt_tokens": int(ids.shape[1]),
        "compressed_heads": len(policy_heads),
        "bank_build_s": bank_build_s,
        "baseline_pred": baseline_ranked[0][0],
        "compact_pred": compact_ranked[0][0],
        "answer": answer,
        "baseline_correct": baseline_ranked[0][0] == answer,
        "compact_correct": compact_ranked[0][0] == answer,
        "agreement": baseline_ranked[0][0] == compact_ranked[0][0],
        "baseline_answer_score": baseline_scores[answer],
        "compact_answer_score": compact_scores[answer],
        "answer_score_delta": compact_scores[answer] - baseline_scores[answer],
        "baseline_margin": baseline_scores[answer] - max(v for k, v in baseline_scores.items() if k != answer),
        "compact_margin": compact_scores[answer] - max(v for k, v in compact_scores.items() if k != answer),
        "baseline_scores": json.dumps(baseline_scores, sort_keys=True),
        "compact_scores": json.dumps(compact_scores, sort_keys=True),
        "choice_scoring": args.choice_scoring,
    }


def summarize(rows, min_baseline_accuracy):
    by_threshold = {}
    for row in rows:
        by_threshold.setdefault(row["threshold"], []).append(row)
    summary_rows = []
    for threshold, vals in sorted(by_threshold.items()):
        baseline_correct_vals = [v for v in vals if v["baseline_correct"]]
        baseline_accuracy = sum(v["baseline_correct"] for v in vals) / len(vals)
        summary_rows.append({
            "threshold": threshold,
            "tasks": len(vals),
            "compressed_heads": vals[0]["compressed_heads"] if vals else 0,
            "baseline_accuracy": baseline_accuracy,
            "min_baseline_accuracy": min_baseline_accuracy,
            "baseline_gate_pass": baseline_accuracy >= min_baseline_accuracy,
            "compact_accuracy": sum(v["compact_correct"] for v in vals) / len(vals),
            "agreement": sum(v["agreement"] for v in vals) / len(vals),
            "baseline_correct_tasks": len(baseline_correct_vals),
            "compact_accuracy_on_baseline_correct": (
                sum(v["compact_correct"] for v in baseline_correct_vals) / len(baseline_correct_vals)
                if baseline_correct_vals else float("nan")
            ),
            "agreement_on_baseline_correct": (
                sum(v["agreement"] for v in baseline_correct_vals) / len(baseline_correct_vals)
                if baseline_correct_vals else float("nan")
            ),
            "mean_answer_score_delta": sum(v["answer_score_delta"] for v in vals) / len(vals),
            "mean_baseline_margin": sum(v["baseline_margin"] for v in vals) / len(vals),
            "mean_compact_margin": sum(v["compact_margin"] for v in vals) / len(vals),
            "mean_bank_build_s": sum(v["bank_build_s"] for v in vals) / len(vals),
            "effective_entry_saving_fraction": (
                sum(v.get("effective_entry_saving_fraction", 0.0) for v in vals) / len(vals)
            ),
        })
    return summary_rows


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    kwargs = dict(trust_remote_code=True, local_files_only=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, **kwargs)
    model.eval().to(device)

    runtime = RuntimeState()
    install_attention_patch(model, runtime)

    text = Path(args.text_file).read_text(encoding="utf-8")
    tasks = load_tasks(args.task_file)
    prompts = build_prompts(tokenizer, text, tasks, args.max_doc_tokens, args.max_prompt_tokens)
    if args.task_limit > 0:
        prompts = prompts[: args.task_limit]
    (out_dir / "tasks.json").write_text(json.dumps(prompts, indent=2, ensure_ascii=False), encoding="utf-8")
    adaptive_policy = load_adaptive_policy(args.adaptive_policy_file)
    if adaptive_policy is not None:
        serializable_adaptive_policy = {
            **adaptive_policy,
            "overrides": {
                task_id: {
                    **override,
                    "drop_heads": sorted(f"{layer}:{head}" for layer, head in override.get("drop_heads", set())),
                    "full_heads": sorted(f"{layer}:{head}" for layer, head in override.get("full_heads", set())),
                }
                for task_id, override in adaptive_policy.get("overrides", {}).items()
            },
        }
        (out_dir / "adaptive_policy.json").write_text(
            json.dumps(serializable_adaptive_policy, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    thresholds = parse_floats(args.thresholds)
    rows = []
    t0 = time.perf_counter()
    for threshold in thresholds:
        policy_heads = load_policy(args.policy_dir, threshold)
        for task in prompts:
            task_policy_heads = apply_adaptive_policy(policy_heads, adaptive_policy, task["id"])
            row = evaluate_task(
                model,
                tokenizer,
                task["prompt"],
                task["choices"],
                task["answer"],
                runtime,
                task_policy_heads,
                threshold,
                args,
            )
            row.update(task_id=task["id"], question=task["question"])
            row["default_compressed_heads"] = len(policy_heads)
            row["adaptive_dropped_heads"] = len(policy_heads) - len(task_policy_heads)
            row["effective_entry_saving_fraction"] = effective_entry_saving_fraction(
                task_policy_heads,
                seq_len=args.max_prompt_tokens,
            )
            rows.append(row)
            print(
                f"thr={threshold:.2f} task={task['id']} baseline={row['baseline_pred']} "
                f"compact={row['compact_pred']} answer={row['answer']} "
                f"heads={len(task_policy_heads)} delta={row['answer_score_delta']:.4f}"
            )

    summary_rows = summarize(rows, args.min_baseline_accuracy)
    write_rows(out_dir / "task_rows.csv", rows)
    write_rows(out_dir / "summary.csv", summary_rows)

    summary = {
        "settings": vars(args),
        "elapsed_s": time.perf_counter() - t0,
        "summary": summary_rows,
        "note": (
            "This is a quality-only simulation. It keeps the full KV tensor shape, "
            "but selected heads are forced to attend only to their compact bank during decode scoring."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report = [
        "# Task-Level Quality Smoke Test",
        "",
        summary["note"],
        "",
        f"Policy dir: `{args.policy_dir}`",
        f"Adaptive policy file: `{args.adaptive_policy_file or ''}`",
        f"Minimum baseline accuracy for a reliable quality smoke: `{args.min_baseline_accuracy:.2%}`.",
        "The baseline gate marks rows as `weak-baseline` when the baseline is below that threshold.",
        "",
        "| Threshold | Tasks | Heads | Avg Entry Saving | Baseline Gate | Baseline Acc | Compact Acc | Agreement | Mean Answer LogP Delta | Baseline Margin | Compact Margin |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        gate = "pass" if row["baseline_gate_pass"] else "weak-baseline"
        report.append(
            f"| {row['threshold']:.2f} | {row['tasks']} | {row['compressed_heads']} | "
            f"{row['effective_entry_saving_fraction']:.2%} | {gate} | "
            f"{row['baseline_accuracy']:.2%} | {row['compact_accuracy']:.2%} | {row['agreement']:.2%} | "
            f"{row['mean_answer_score_delta']:.4f} | {row['mean_baseline_margin']:.4f} | {row['mean_compact_margin']:.4f} |"
        )
    report.extend([
        "",
        "| Threshold | Baseline-Correct Tasks | Compact Acc on Baseline-Correct | Agreement on Baseline-Correct |",
        "|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        report.append(
            f"| {row['threshold']:.2f} | {row['baseline_correct_tasks']} / {row['tasks']} | "
            f"{row['compact_accuracy_on_baseline_correct']:.2%} | "
            f"{row['agreement_on_baseline_correct']:.2%} |"
        )
    report.extend([
        "",
        "Per-task rows are in `task_rows.csv`.",
    ])
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out_dir)
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
