import copy
import math

from confirmatory_v2_contract import _render_user_prompt
from pilot_protocol import FLAGS, fields, require
from prompt_rendering import INSTRUCTION
from validation_contract import digest


ROUTER_KIND = "LORA_WHITENED_PROTOTYPE_ROUTER_V1"
ROUTER_LAYER_INDICES = tuple(range(18, 31))
ROUTER_ALLOWED_PUBLIC_KEYS = ("entry_point", "spec", "initial_source", "public_cases")
TASK_HEADER_MARKER = "\n\nCurrent task:\n"


def _validate_unit_vector(vector_values, expected_dim=None):
    require(
        isinstance(vector_values, list) and len(vector_values) >= 4,
        "router vector must be list with dim >= 4",
    )
    if expected_dim is not None:
        require(len(vector_values) == expected_dim, "router vector dimension mismatch")
    float_list = []
    sq_sum = 0.0
    for elem in vector_values:
        require(type(elem) in (int, float) and math.isfinite(float(elem)), "nonfinite vector element")
        val_f = float(elem)
        float_list.append(val_f)
        sq_sum += val_f * val_f
    norm_val = math.sqrt(sq_sum)
    require(abs(norm_val - 1.0) <= 1e-4, "unit vector required")
    return float_list


def _normalize_raw_vector(raw_values):
    sq_sum = 0.0
    for elem in raw_values:
        require(math.isfinite(elem), "nonfinite raw vector component")
        sq_sum += elem * elem
    norm_val = math.sqrt(sq_sum)
    require(norm_val > 1e-12, "zero norm vector in router")
    return [elem / norm_val for elem in raw_values]


def _mean_vector(vectors_list, dim_size):
    count_num = len(vectors_list)
    require(count_num > 0, "nonempty vector list required")
    accum = [0.0] * dim_size
    for vec_vals in vectors_list:
        for axis_idx in range(dim_size):
            accum[axis_idx] += vec_vals[axis_idx]
    inv_count = 1.0 / float(count_num)
    return [val_sum * inv_count for val_sum in accum]


def _center_and_normalize(vec_vals, mean_vals):
    diff_vals = [val_a - val_b for val_a, val_b in zip(vec_vals, mean_vals)]
    return _normalize_raw_vector(diff_vals)


def _dot_product(first_vec, second_vec):
    total_dot = 0.0
    for val_a, val_b in zip(first_vec, second_vec):
        total_dot += val_a * val_b
    return total_dot


def sanitize_task_for_router(task_record):
    require(isinstance(task_record, dict), "task_record must be dict")
    for req_key in ROUTER_ALLOWED_PUBLIC_KEYS:
        require(req_key in task_record, "missing public router key: " + req_key)
    sanitized = {"id": "TASK"}
    for req_key in ROUTER_ALLOWED_PUBLIC_KEYS:
        sanitized[req_key] = copy.deepcopy(task_record[req_key])
    return sanitized


def render_sanitized_router_prompt(task_record):
    sanitized = sanitize_task_for_router(task_record)
    return _render_user_prompt(sanitized, examples_list=None, witness_obj=None)


def _chat_template_ids(tokenizer, messages, generation_flag):
    raw_ids = tokenizer.apply_chat_template(
        copy.deepcopy(messages),
        tokenize=True,
        truncation=False,
        add_generation_prompt=generation_flag,
        enable_thinking=False,
    )
    if hasattr(raw_ids, "get") and "input_ids" in raw_ids:
        raw_ids = raw_ids["input_ids"]
    id_list = list(raw_ids)
    require(
        bool(id_list) and all(type(tok_id) is int and tok_id >= 0 for tok_id in id_list),
        "valid integer token IDs required",
    )
    return id_list


def locate_task_token_span(tokenizer, messages):
    require(isinstance(messages, list) and len(messages) == 1, "single-turn router message required")
    fields(messages[0], ("role", "content"))
    require(messages[0]["role"] == "user", "user role required")
    body_text = messages[0]["content"]
    require(
        isinstance(body_text, str) and (INSTRUCTION + TASK_HEADER_MARKER) in body_text,
        "router prompt missing task header marker",
    )

    full_ids = _chat_template_ids(tokenizer, messages, True)
    no_gen_ids = _chat_template_ids(tokenizer, messages, False)
    probe_alpha = _chat_template_ids(
        tokenizer,
        [{"role": "user", "content": INSTRUCTION + TASK_HEADER_MARKER + "Alpha11111"}],
        False,
    )
    probe_bravo = _chat_template_ids(
        tokenizer,
        [{"role": "user", "content": INSTRUCTION + TASK_HEADER_MARKER + "Bravo99999"}],
        False,
    )

    prefix_len = 0
    max_prefix = min(len(probe_alpha), len(probe_bravo))
    while prefix_len < max_prefix and probe_alpha[prefix_len] == probe_bravo[prefix_len]:
        prefix_len += 1

    suffix_len = 0
    rem_alpha = len(probe_alpha) - prefix_len
    rem_bravo = len(probe_bravo) - prefix_len
    max_suffix = min(rem_alpha, rem_bravo)
    while (
        suffix_len < max_suffix
        and probe_alpha[len(probe_alpha) - 1 - suffix_len]
        == probe_bravo[len(probe_bravo) - 1 - suffix_len]
    ):
        suffix_len += 1

    span_start = prefix_len
    span_end = len(no_gen_ids) - suffix_len
    require(span_start > 0 and span_end > span_start, "empty router task token span")
    require(
        len(full_ids) >= span_end and full_ids[:span_end] == no_gen_ids[:span_end],
        "chat template generation suffix mutated task prefix",
    )
    require(
        no_gen_ids[:span_start] == probe_alpha[:span_start],
        "router task prefix mismatch",
    )
    if suffix_len > 0:
        require(
            no_gen_ids[span_end:] == probe_alpha[len(probe_alpha) - suffix_len:],
            "router user-turn suffix mismatch",
        )
    return full_ids, span_start, span_end


def extract_router_embedding(model, tokenizer, task_record, *, layer_indices=ROUTER_LAYER_INDICES):
    import torch

    require(isinstance(model, torch.nn.Module), "torch Module required")
    require(all(not mod_item.training for mod_item in model.modules()), "eval mode required")
    params_list = list(model.parameters())
    require(bool(params_list), "nonempty model parameters required")
    require(
        all(not par_item.requires_grad and par_item.grad is None for par_item in params_list),
        "frozen unadapted model without gradients required",
    )
    devices = {par_item.device for par_item in params_list}
    require(len(devices) == 1, "single device required")
    device = next(iter(devices))

    layer_tuple = tuple(int(l_idx) for l_idx in layer_indices)
    require(
        bool(layer_tuple) and all(l_idx >= 1 for l_idx in layer_tuple),
        "positive transformer layer indices required",
    )

    messages = render_sanitized_router_prompt(task_record)
    full_ids, span_start, span_end = locate_task_token_span(tokenizer, messages)
    require(len(full_ids) <= 4096, "router prompt exceeds 4096 token limit")

    rng_devices = [device.index or 0] if device.type == "cuda" else []
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
        input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
        mask_tensor = torch.ones_like(input_tensor)
        out_obj = model(
            input_ids=input_tensor,
            attention_mask=mask_tensor,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_tuple = getattr(out_obj, "hidden_states", None)
        require(
            isinstance(hidden_tuple, (tuple, list)) and len(hidden_tuple) > max(layer_tuple),
            "insufficient hidden_states layers",
        )
        accum_vec = None
        for l_idx in layer_tuple:
            layer_hid = hidden_tuple[l_idx]
            require(
                isinstance(layer_hid, torch.Tensor)
                and layer_hid.ndim == 3
                and layer_hid.shape[0] == 1
                and layer_hid.shape[1] == len(full_ids),
                "unexpected hidden_states tensor shape",
            )
            span_slice = layer_hid[0, span_start:span_end].to(dtype=torch.float64)
            require(bool(torch.isfinite(span_slice).all()), "nonfinite hidden state")
            pooled = span_slice.mean(dim=0)
            layer_norm = torch.linalg.vector_norm(pooled)
            require(bool(torch.isfinite(layer_norm)) and float(layer_norm.item()) > 1e-12, "zero hidden norm")
            unit_layer = pooled / layer_norm
            accum_vec = unit_layer if accum_vec is None else (accum_vec + unit_layer)

        total_norm = torch.linalg.vector_norm(accum_vec)
        require(bool(torch.isfinite(total_norm)) and float(total_norm.item()) > 1e-12, "zero pooled norm")
        final_unit = (accum_vec / total_norm).detach().cpu().tolist()
        del out_obj, hidden_tuple, input_tensor, mask_tensor

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return _validate_unit_vector([float(comp_v) for comp_v in final_unit])


def _solve_loo_threshold(loo_pos_scores, loo_bg_scores):
    pos_vals = list(loo_pos_scores.values())
    bg_vals = list(loo_bg_scores.values())
    distinct_sorted = sorted(set(pos_vals + bg_vals))
    if len(distinct_sorted) == 1:
        candidates = [(distinct_sorted[0] - 0.01, 0.0), (distinct_sorted[0] + 0.01, 0.0)]
    else:
        candidates = [(distinct_sorted[0] - 0.01, 0.0)]
        for idx_pos in range(len(distinct_sorted) - 1):
            low_v = distinct_sorted[idx_pos]
            high_v = distinct_sorted[idx_pos + 1]
            candidates.append((0.5 * (low_v + high_v), high_v - low_v))
        candidates.append((distinct_sorted[-1] + 0.01, 0.0))

    best_choice = None
    best_tau = candidates[0][0]
    best_stats = None
    for cand_tau, gap_width in candidates:
        pos_hits = sum(1 for sc_v in pos_vals if sc_v > cand_tau)
        bg_false = sum(1 for sc_v in bg_vals if sc_v > cand_tau)
        tpr_val = pos_hits / float(len(pos_vals))
        tnr_val = (len(bg_vals) - bg_false) / float(len(bg_vals))
        bal_acc = 0.5 * (tpr_val + tnr_val)
        rank_key = (
            round(bal_acc, 12),
            round(gap_width, 12),
            round(tnr_val, 12),
            round(cand_tau, 12),
        )
        if best_choice is None or rank_key > best_choice:
            best_choice = rank_key
            best_tau = cand_tau
            best_stats = {
                "positive_recall_count": pos_hits,
                "background_false_open_count": bg_false,
                "true_positive_rate": tpr_val,
                "true_negative_rate": tnr_val,
                "balanced_accuracy": bal_acc,
                "selected_interval_width": gap_width,
            }
    return best_tau, best_stats


def fit_whitened_prototype_router(
    positive_vectors,
    background_vectors,
    *,
    layer_indices=ROUTER_LAYER_INDICES,
):
    require(
        isinstance(positive_vectors, dict) and len(positive_vectors) == 8,
        "exactly 8 positive supervision vectors required",
    )
    require(
        isinstance(background_vectors, dict) and len(background_vectors) == 16,
        "exactly 16 background reference vectors required",
    )
    pos_ids = sorted(positive_vectors.keys())
    bg_ids = sorted(background_vectors.keys())
    require(all(isinstance(key_s, str) and bool(key_s) for key_s in pos_ids + bg_ids), "string IDs required")
    require(set(pos_ids).isdisjoint(set(bg_ids)), "positive and background IDs must be disjoint")

    first_vec = _validate_unit_vector(positive_vectors[pos_ids[0]])
    dim_size = len(first_vec)
    clean_pos = {pos_id: _validate_unit_vector(positive_vectors[pos_id], dim_size) for pos_id in pos_ids}
    clean_bg = {bg_id: _validate_unit_vector(background_vectors[bg_id], dim_size) for bg_id in bg_ids}

    bg_mean = _mean_vector([clean_bg[bg_id] for bg_id in bg_ids], dim_size)
    centered_pos = {
        pos_id: _center_and_normalize(clean_pos[pos_id], bg_mean)
        for pos_id in pos_ids
    }

    loo_pos_scores = {}
    for held_pos_id in pos_ids:
        held_vec = centered_pos[held_pos_id]
        other_scores = [
            _dot_product(held_vec, centered_pos[other_id])
            for other_id in pos_ids
            if other_id != held_pos_id
        ]
        loo_pos_scores[held_pos_id] = max(other_scores)

    loo_bg_scores = {}
    for held_bg_id in bg_ids:
        rem_bg_vecs = [clean_bg[other_bg] for other_bg in bg_ids if other_bg != held_bg_id]
        loo_mean = _mean_vector(rem_bg_vecs, dim_size)
        held_bg_centered = _center_and_normalize(clean_bg[held_bg_id], loo_mean)
        pos_under_loo = [
            _center_and_normalize(clean_pos[pos_id], loo_mean)
            for pos_id in pos_ids
        ]
        loo_bg_scores[held_bg_id] = max(
            _dot_product(held_bg_centered, proto_vec) for proto_vec in pos_under_loo
        )

    tau_val, loo_stats = _solve_loo_threshold(loo_pos_scores, loo_bg_scores)
    separating_margin = min(loo_pos_scores.values()) - max(loo_bg_scores.values())
    serialized_bytes = (len(pos_ids) + 1) * dim_size * 4

    core_bundle = {
        "kind": ROUTER_KIND,
        "version": 1,
        "dimension": dim_size,
        "layer_indices": [int(l_idx) for l_idx in layer_indices],
        "positive_ids": pos_ids,
        "background_ids": bg_ids,
        "background_mean": bg_mean,
        "centered_positive_prototypes": centered_pos,
        "threshold_tau": tau_val,
        "estimated_fp32_fingerprint_bytes": serialized_bytes,
        "loo_summary": {
            "positive_scores": loo_pos_scores,
            "background_scores": loo_bg_scores,
            "min_positive_score": min(loo_pos_scores.values()),
            "max_background_score": max(loo_bg_scores.values()),
            "separating_margin": separating_margin,
            **loo_stats,
        },
        **{flag_name: False for flag_name in FLAGS},
    }
    return {**core_bundle, "bundle_sha256": digest(core_bundle)}


def validate_router_bundle(router_bundle):
    require(isinstance(router_bundle, dict), "router_bundle must be dict")
    require(router_bundle.get("kind") == ROUTER_KIND, "router bundle kind mismatch")
    require(router_bundle.get("version") == 1, "router bundle version mismatch")
    require(
        all(router_bundle.get(flag_name) is False for flag_name in FLAGS),
        "router governance flags must be false",
    )
    dim_size = router_bundle.get("dimension")
    require(type(dim_size) is int and dim_size >= 4, "invalid router dimension")
    pos_ids = router_bundle.get("positive_ids")
    bg_ids = router_bundle.get("background_ids")
    require(
        isinstance(pos_ids, list) and len(pos_ids) == 8 and pos_ids == sorted(set(pos_ids)),
        "invalid router positive_ids",
    )
    require(
        isinstance(bg_ids, list) and len(bg_ids) == 16 and bg_ids == sorted(set(bg_ids)),
        "invalid router background_ids",
    )
    bg_mean = router_bundle.get("background_mean")
    require(
        isinstance(bg_mean, list)
        and len(bg_mean) == dim_size
        and all(type(val_x) in (int, float) and math.isfinite(float(val_x)) for val_x in bg_mean),
        "invalid background_mean",
    )
    protos = router_bundle.get("centered_positive_prototypes")
    require(isinstance(protos, dict) and sorted(protos.keys()) == pos_ids, "prototype keys mismatch")
    for pos_id in pos_ids:
        _validate_unit_vector(protos[pos_id], dim_size)
    tau_val = router_bundle.get("threshold_tau")
    require(type(tau_val) in (int, float) and math.isfinite(float(tau_val)), "invalid threshold_tau")
    expected_sha = router_bundle.get("bundle_sha256")
    core_copy = {key_n: val_v for key_n, val_v in router_bundle.items() if key_n != "bundle_sha256"}
    computed_sha = digest(core_copy)
    require(expected_sha == computed_sha, "router bundle_sha256 mismatch")
    return computed_sha


def score_router_embedding(router_bundle, query_vector):
    validate_router_bundle(router_bundle)
    dim_size = router_bundle["dimension"]
    clean_query = _validate_unit_vector(query_vector, dim_size)
    centered_query = _center_and_normalize(clean_query, router_bundle["background_mean"])
    protos = router_bundle["centered_positive_prototypes"]
    per_anchor = {
        pos_id: _dot_product(centered_query, protos[pos_id])
        for pos_id in router_bundle["positive_ids"]
    }
    nearest_id = max(
        router_bundle["positive_ids"],
        key=lambda item_id: (per_anchor[item_id], item_id),
    )
    max_sim = per_anchor[nearest_id]
    tau_val = float(router_bundle["threshold_tau"])
    return {
        "score": max_sim,
        "threshold_tau": tau_val,
        "margin_over_tau": max_sim - tau_val,
        "gate_open": bool(max_sim > tau_val),
        "nearest_positive_id": nearest_id,
        "anchor_similarities": per_anchor,
    }


def evaluate_gate_g6a_probe(router_bundle, dev_positive_vectors, dev_negative_vectors):
    bundle_sha = validate_router_bundle(router_bundle)
    dim_size = router_bundle["dimension"]
    require(
        isinstance(dev_positive_vectors, dict) and len(dev_positive_vectors) == 12,
        "12 independent dev positive tasks required",
    )
    require(
        isinstance(dev_negative_vectors, dict) and len(dev_negative_vectors) == 4,
        "4 independent dev negative tasks required",
    )
    dev_pos_ids = sorted(dev_positive_vectors.keys())
    dev_neg_ids = sorted(dev_negative_vectors.keys())
    used_cal_ids = set(router_bundle["positive_ids"]) | set(router_bundle["background_ids"])
    require(used_cal_ids.isdisjoint(set(dev_pos_ids)), "dev positive overlaps calibration set")
    require(used_cal_ids.isdisjoint(set(dev_neg_ids)), "dev negative overlaps calibration set")
    require(set(dev_pos_ids).isdisjoint(set(dev_neg_ids)), "dev positive/negative overlap")

    dev_pos_decisions = {
        task_id: score_router_embedding(
            router_bundle, _validate_unit_vector(dev_positive_vectors[task_id], dim_size)
        )
        for task_id in dev_pos_ids
    }
    dev_neg_decisions = {
        task_id: score_router_embedding(
            router_bundle, _validate_unit_vector(dev_negative_vectors[task_id], dim_size)
        )
        for task_id in dev_neg_ids
    }

    dev_pos_open = sum(1 for dec in dev_pos_decisions.values() if dec["gate_open"])
    dev_neg_open = sum(1 for dec in dev_neg_decisions.values() if dec["gate_open"])
    loo_bal = float(router_bundle["loo_summary"]["balanced_accuracy"])
    loo_bg_open = int(router_bundle["loo_summary"]["background_false_open_count"])

    failed_gates = []
    if loo_bal < 0.80:
        failed_gates.append("loo_balanced_accuracy_below_0_80")
    if dev_pos_open < 9:
        failed_gates.append("dev_positive_recall_below_9_of_12")
    if dev_neg_open > 1:
        failed_gates.append("dev_negative_false_open_above_1_of_4")
    if loo_bg_open > 4:
        failed_gates.append("loo_background_false_open_above_4_of_16")

    return {
        "kind": "LORA_GATE_G6A_ROUTER_PROBE_EVALUATION",
        "version": 1,
        "router_bundle_sha256": bundle_sha,
        "threshold_tau": router_bundle["threshold_tau"],
        "loo_balanced_accuracy": loo_bal,
        "loo_background_false_open_count": loo_bg_open,
        "dev_positive_recall_count": dev_pos_open,
        "dev_positive_total": 12,
        "dev_positive_recall_rate": dev_pos_open / 12.0,
        "dev_negative_false_open_count": dev_neg_open,
        "dev_negative_total": 4,
        "dev_negative_false_open_rate": dev_neg_open / 4.0,
        "dev_positive_decisions": dev_pos_decisions,
        "dev_negative_decisions": dev_neg_decisions,
        "failed_gates": failed_gates,
        "gate_g6a_passed": len(failed_gates) == 0,
        **{flag_name: False for flag_name in FLAGS},
    }
