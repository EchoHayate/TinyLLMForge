from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time


MED_MICRO_TABLE_RESIDUAL = (
    "Reference Dose Matrix (R1_NORMAL / R2_MILD / R3_MOD / R4_SEVERE in mg): "
    "CEFEPIME_IV=2000/1000/500/250; MEROPENEM_IV=1000/1000/500/250; "
    "VANCO_IV=1500/1000/750/500; APIXABAN_PO=10/10/5/2; "
    "LINEZOLID_PO=600/600/600/600; PIPTAZO_IV=4500/3375/2250/1125; "
    "DAPTO_IV=700/500/350/250."
)

LEGAL_MICRO_TABLE_RESIDUAL = (
    "Reference Liability Cap Matrix (D0_BIO_PII / D1_CUST_PII / D2_CONFID / D3_PUBLIC): "
    "TIER_STRATEGIC=5x/5x/3x/2x; TIER_CORE=5x/3x/2x/1x; TIER_STD=5x/3x/2x/1x. "
    "Rule Cure Days: LEG-R01=0, LEG-R02=15, LEG-R03=15, LEG-R04=30, LEG-R05=30, LEG-R06=45."
)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _select_idle_a100() -> dict:
    gpu_csv = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    apps_csv = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    busy = {ln.split(",")[0].strip() for ln in apps_csv.splitlines() if ln.strip()}
    for ln in gpu_csv.splitlines():
        cols = [seg.strip() for seg in ln.split(",")]
        if len(cols) != 5:
            continue
        idx_s, uuid_s, name_s, free_s, used_s = cols
        if "A100" in name_s and uuid_s not in busy and int(free_s) >= 65536 and int(used_s) <= 512:
            return {"index": int(idx_s), "name": name_s, "free_mib": int(free_s), "used_mib": int(used_s)}
    raise RuntimeError("No idle A100 80GB GPU available")


def generate_synthetic_coverage_probes(domain: str, forbidden_signatures: set[str]) -> list[dict]:
    from experiments.cross_domain_prefix_lora.domain_sop_contracts import (
        _format_legal_vignette,
        _format_med_vignette,
    )

    rng = random.Random(20261126 if domain == "med_triage" else 20261127)
    probes: list[dict] = []
    seen = set(forbidden_signatures)

    if domain == "med_triage":
        regimens = [
            "CEFEPIME_IV",
            "MEROPENEM_IV",
            "VANCO_IV",
            "APIXABAN_PO",
            "LINEZOLID_PO",
            "PIPTAZO_IV",
            "DAPTO_IV",
        ]
        egfr_bands = [78, 64, 48, 34, 23, 12]
        candidates = []

        for reg in regimens:
            for eg in egfr_bands:
                candidates.append({
                    "age": rng.choice([51, 63, 72]),
                    "albumin_g_dl": 3.8,
                    "egfr": eg,
                    "aki_stage": rng.choice([0, 1]),
                    "qtc_ms": 442,
                    "potassium_mmol": 4.3,
                    "platelets_k": 210,
                    "active_bleed": False,
                    "on_maoi": False,
                    "requested_regimen": reg,
                })

        for reg in regimens:
            for alb in (2.8, 3.5):
                for eg in (66, 38, 24):
                    candidates.append({
                        "age": rng.choice([76, 81, 85]),
                        "albumin_g_dl": alb,
                        "egfr": eg,
                        "aki_stage": 0,
                        "qtc_ms": 446,
                        "potassium_mmol": 4.2,
                        "platelets_k": 195,
                        "active_bleed": False,
                        "on_maoi": False,
                        "requested_regimen": reg,
                    })

        for aki_v in (1, 2, 3):
            for reg in ("CEFEPIME_IV", "PIPTAZO_IV", "VANCO_IV", "MEROPENEM_IV"):
                candidates.append({
                    "age": 58,
                    "albumin_g_dl": 3.6,
                    "egfr": rng.choice([68, 45]),
                    "aki_stage": aki_v,
                    "qtc_ms": 450,
                    "potassium_mmol": 4.4,
                    "platelets_k": 185,
                    "active_bleed": False,
                    "on_maoi": False,
                    "requested_regimen": reg,
                })

        special_overrides = [
            {"age": 60, "albumin_g_dl": 3.7, "egfr": eg, "aki_stage": 0, "qtc_ms": qtc, "potassium_mmol": 4.3, "platelets_k": 200, "active_bleed": False, "on_maoi": False, "requested_regimen": reg}
            for qtc in (498, 514, 528)
            for eg in (72, 24)
            for reg in ("VANCO_IV", "LINEZOLID_PO")
        ] + [
            {"age": 66, "albumin_g_dl": 3.6, "egfr": eg, "aki_stage": 0, "qtc_ms": 440, "potassium_mmol": 4.2, "platelets_k": plt, "active_bleed": bld, "on_maoi": False, "requested_regimen": reg}
            for plt, bld in ((36, False), (180, True), (35, True))
            for eg in (65, 28)
            for reg in ("APIXABAN_PO", "VANCO_IV")
        ] + [
            {"age": 77, "albumin_g_dl": 2.9, "egfr": eg, "aki_stage": 0, "qtc_ms": 444, "potassium_mmol": 4.3, "platelets_k": 190, "active_bleed": False, "on_maoi": True, "requested_regimen": reg}
            for eg in (74, 50, 36, 14)
            for reg in ("LINEZOLID_PO", "CEFEPIME_IV")
        ] + [
            {"age": 59, "albumin_g_dl": 3.7, "egfr": eg, "aki_stage": 0, "qtc_ms": 448, "potassium_mmol": 5.9, "platelets_k": 205, "active_bleed": False, "on_maoi": False, "requested_regimen": reg}
            for eg in (70, 44, 22, 13)
            for reg in ("PIPTAZO_IV", "VANCO_IV")
        ]
        candidates.extend(special_overrides)

        for cand in candidates:
            sig = json.dumps(cand, sort_keys=True)
            if sig in seen:
                continue
            seen.add(sig)
            pid = f"M_G{len(probes) + 1:03d}"
            obj = {"id": pid, "input": cand}
            obj["vignette"] = _format_med_vignette(obj)
            probes.append(obj)
            if len(probes) >= 112:
                break
        return probes

    if domain == "legal_compliance":
        v_tiers = ["TIER_STRATEGIC", "TIER_CORE", "TIER_STD"]
        d_classes = ["D0_BIO_PII", "D1_CUST_PII", "D2_CONFID", "D3_PUBLIC"]
        candidates = []

        for vt in v_tiers:
            for dc in d_classes:
                for pcap in (1, 2, 3, 5):
                    candidates.append({
                        "vendor_tier": vt,
                        "data_class": dc,
                        "cross_border": "NONE",
                        "has_sec_assessment": True,
                        "governing_law": rng.choice(["DOMESTIC_COURT", "NEUTRAL_ARBITRATION"]),
                        "ip_indemnity_uncapped": True,
                        "auto_renew_months": 12,
                        "terminate_for_conv": True,
                        "proposed_cap_x": pcap,
                    })

        for vt in v_tiers:
            for dc in ("D0_BIO_PII", "D1_CUST_PII", "D3_PUBLIC"):
                for cb in ("NONE", "EU_FRANKFURT", "SG_SINGAPORE"):
                    for sec in (False, True):
                        if cb == "NONE" and sec:
                            continue
                        candidates.append({
                            "vendor_tier": vt,
                            "data_class": dc,
                            "cross_border": cb,
                            "has_sec_assessment": sec,
                            "governing_law": "NEUTRAL_ARBITRATION",
                            "ip_indemnity_uncapped": True,
                            "auto_renew_months": 12,
                            "terminate_for_conv": True,
                            "proposed_cap_x": 5,
                        })

        for vt in v_tiers:
            for dc in d_classes:
                candidates.append({
                    "vendor_tier": vt,
                    "data_class": dc,
                    "cross_border": "NONE",
                    "has_sec_assessment": True,
                    "governing_law": "FOREIGN_LITIGATION",
                    "ip_indemnity_uncapped": rng.choice([True, False]),
                    "auto_renew_months": rng.choice([12, 24]),
                    "terminate_for_conv": False,
                    "proposed_cap_x": rng.choice([1, 3, 5]),
                })
                candidates.append({
                    "vendor_tier": vt,
                    "data_class": dc,
                    "cross_border": "NONE",
                    "has_sec_assessment": True,
                    "governing_law": "NEUTRAL_ARBITRATION",
                    "ip_indemnity_uncapped": False,
                    "auto_renew_months": 12,
                    "terminate_for_conv": True,
                    "proposed_cap_x": 5,
                })
                for ren, conv in ((24, False), (24, True), (36, False)):
                    candidates.append({
                        "vendor_tier": vt,
                        "data_class": dc,
                        "cross_border": "NONE",
                        "has_sec_assessment": True,
                        "governing_law": "DOMESTIC_COURT",
                        "ip_indemnity_uncapped": True,
                        "auto_renew_months": ren,
                        "terminate_for_conv": conv,
                        "proposed_cap_x": 5,
                    })

        for cand in candidates:
            sig = json.dumps(cand, sort_keys=True)
            if sig in seen:
                continue
            seen.add(sig)
            pid = f"L_G{len(probes) + 1:03d}"
            obj = {"id": pid, "input": cand}
            obj["vignette"] = _format_legal_vignette(obj)
            probes.append(obj)
            if len(probes) >= 112:
                break
        return probes

    raise ValueError(domain)


def render_dense_student_messages(
    domain: str,
    case_obj: dict,
    *,
    include_micro_residual: bool,
) -> list[dict]:
    from experiments.cross_domain_prefix_lora.domain_sop_contracts import (
        LEGAL_TASK_SUFFIX_SPEC,
        MED_TASK_SUFFIX_SPEC,
    )

    cot_spec = (
        "First output one concise `TRACE: ...` line computing the exact intermediate tier/cap and first triggered rule, "
        "followed on the next line by ONLY the JSON object.\n"
        + (MED_TASK_SUFFIX_SPEC.split("\n", 1)[1] if domain == "med_triage" else LEGAL_TASK_SUFFIX_SPEC.split("\n", 1)[1])
    )
    if domain == "med_triage":
        hdr = "Apply institutional clinical pharmacy protocol MED-SOP-2026-V4."
        if include_micro_residual:
            hdr = hdr + "\n" + MED_MICRO_TABLE_RESIDUAL
    else:
        hdr = "Apply enterprise legal procurement policy LEGAL-SOP-2026-V3."
        if include_micro_residual:
            hdr = hdr + "\n" + LEGAL_MICRO_TABLE_RESIDUAL

    return [{"role": "user", "content": f"{hdr}\n\n=== CURRENT CASE FOR REVIEW ===\n{case_obj['vignette']}\n\n{cot_spec}"}]


def validate_trace_and_json_self_consistency(domain: str, raw_text: str) -> str | None:
    from experiments.cross_domain_prefix_lora.domain_sop_contracts import extract_first_json_object
    from experiments.cross_domain_prefix_lora.surgical_json_csd_trainer import (
        validate_unlabeled_sop_candidate,
    )

    if "TRACE:" not in raw_text:
        return None
    canon_j = validate_unlabeled_sop_candidate(domain, raw_text)
    if canon_j is None:
        return None
    parsed = extract_first_json_object(canon_j)
    if parsed is None:
        return None
    trace_ln = raw_text.strip().splitlines()[0].strip()
    if domain == "med_triage":
        if f"renal_tier={parsed['renal_tier']}" not in trace_ln:
            return None
        if f"rule={parsed['rule_id']}" not in trace_ln:
            return None
    else:
        if f"min_cap_x={parsed['min_cap_x']}" not in trace_ln:
            return None
        if f"rule={parsed['primary_rule']}" not in trace_ln:
            return None
    return f"{trace_ln}\n{canon_j}"


class Rank16LoRAWrapper:
    @staticmethod
    def wrap_module(torch_mod, nn_mod, base_lin, rank: int = 16, alpha: float = 32.0):
        class _Wrap(nn_mod.Module):
            def __init__(self):
                super().__init__()
                self.base = base_lin
                self.scale = float(alpha) / float(rank)
                in_f = int(base_lin.in_features)
                out_f = int(base_lin.out_features)
                dev = base_lin.weight.device
                self.adapter_a = nn_mod.Parameter(torch_mod.empty((rank, in_f), device=dev, dtype=torch_mod.float32))
                self.adapter_b = nn_mod.Parameter(torch_mod.zeros((out_f, rank), device=dev, dtype=torch_mod.float32))
                nn_mod.init.kaiming_uniform_(self.adapter_a, a=math.sqrt(5))

            def forward(self, x):
                base_y = self.base(x)
                upd = nn_mod.functional.linear(
                    nn_mod.functional.linear(x.float(), self.adapter_a),
                    self.adapter_b,
                )
                return base_y + (self.scale * upd).to(base_y.dtype)

        return _Wrap()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-save-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    selected_gpu = _select_idle_a100()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu["index"])

    repo_root = Path(args.repo_root).resolve()
    if (repo_root / "experiments/cross_domain_prefix_lora/data/cross_domain_v1/blind_oracles.json").exists():
        raise RuntimeError("blind_oracles.json must never exist on GPU stage")

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "experiments/lora_necessity"))

    import torch
    from torch import nn
    from torch.nn import functional
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kl_anchored_lora_ablation import _compute_trajectory_kl_loss
    from training_worker import _state, _tensor_digest
    from experiments.cross_domain_prefix_lora.cross_domain_cot_mlp_ablation import (
        _collect_qkv_and_mlp_targets,
        render_cot_messages,
        render_seed_micro_cot,
    )
    from experiments.cross_domain_prefix_lora.surgical_json_csd_trainer import (
        cached_greedy_generate,
        chat_prompt_and_target_ids,
        prepare_multimodal_background_trajectories,
    )

    public_bundle = json.loads(
        (repo_root / "experiments/cross_domain_prefix_lora/data/cross_domain_v1/public_bundle.json").read_text("utf-8")
    )

    torch.cuda.init()
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    init_sha = _tensor_digest(_state(model))

    bg_trajs = prepare_multimodal_background_trajectories(
        model, tokenizer, public_bundle["background_anchors"], max_new_tokens=128
    )

    save_dir = Path(args.adapter_save_dir).resolve()
    save_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    out_domains = {}
    for dom, pt_name in (
        ("med_triage", "med_dense_r16_mlp.pt"),
        ("legal_compliance", "legal_dense_r16_mlp.pt"),
    ):
        dom_data = public_bundle[dom]
        forb_sigs = {
            json.dumps(x["input"], sort_keys=True)
            for grp in (dom_data["seeds"], dom_data["unlabeled"], dom_data["blind_test_inputs"])
            for x in grp
        }
        grid_probes = generate_synthetic_coverage_probes(dom, forb_sigs)
        all_unlabeled = list(dom_data["unlabeled"]) + grid_probes

        def _build_pair(t_msgs: list[dict], s_pure_msgs: list[dict], s_res_msgs: list[dict], ans_str: str) -> dict:
            t_p, t_t = chat_prompt_and_target_ids(tokenizer, t_msgs, answer_text=ans_str)
            sp_p, sp_t = chat_prompt_and_target_ids(tokenizer, s_pure_msgs, answer_text=ans_str)
            sr_p, sr_t = chat_prompt_and_target_ids(tokenizer, s_res_msgs, answer_text=ans_str)
            if t_t != sp_t or t_t != sr_t or t_t is None:
                raise ValueError("target token mismatch")
            alen = len(t_t)
            t_in = torch.tensor([(t_p + t_t)[:-1]], dtype=torch.long, device=device)
            with torch.no_grad():
                t_out = model(input_ids=t_in, attention_mask=torch.ones_like(t_in), use_cache=False, return_dict=True)
                t_sl = t_out.logits[0, len(t_p) - 1 : len(t_p) - 1 + alen].detach().to(torch.bfloat16).cpu().clone()
            del t_out, t_in
            return {
                "pure_full_ids": sp_p + sp_t,
                "pure_prompt_len": len(sp_p),
                "res_full_ids": sr_p + sr_t,
                "res_prompt_len": len(sr_p),
                "answer_len": alen,
                "target_ids": t_t,
                "teacher_logits_bf16": t_sl,
            }

        t_mine0 = time.monotonic()
        seed_trajs = []
        for s_it in dom_data["seeds"]:
            ans_cot = render_seed_micro_cot(dom, s_it)
            t_m = render_cot_messages(dom, s_it, public_bundle, is_long=True, exclude_seed_id=s_it["id"])
            sp_m = render_dense_student_messages(dom, s_it, include_micro_residual=False)
            sr_m = render_dense_student_messages(dom, s_it, include_micro_residual=True)
            seed_trajs.append({"source_pool": "seed", **_build_pair(t_m, sp_m, sr_m, ans_cot)})

        extra_trajs = []
        for u_it in all_unlabeled:
            t_m = render_cot_messages(dom, u_it, public_bundle, is_long=True)
            gen_u = cached_greedy_generate(model, tokenizer, t_m, max_new_tokens=200, device=device)
            if not gen_u["stop_eos"]:
                continue
            clean_ans = validate_trace_and_json_self_consistency(dom, gen_u["text"])
            if clean_ans is None:
                continue
            sp_m = render_dense_student_messages(dom, u_it, include_micro_residual=False)
            sr_m = render_dense_student_messages(dom, u_it, include_micro_residual=True)
            extra_trajs.append({"source_pool": "extra", **_build_pair(t_m, sp_m, sr_m, clean_ans)})
        mine_s = time.monotonic() - t_mine0

        base_micro_table_recs = []
        long_text_cot_recs = []
        for c_it in dom_data["blind_test_inputs"]:
            gb_m = cached_greedy_generate(
                model,
                tokenizer,
                render_dense_student_messages(dom, c_it, include_micro_residual=True),
                max_new_tokens=220,
                device=device,
            )
            base_micro_table_recs.append({
                "id": c_it["id"],
                "text": gb_m["text"],
                "input_tokens": gb_m["input_tokens"],
                "wall_ms": round(gb_m["wall_ms"], 2),
            })
            gt_m = cached_greedy_generate(
                model,
                tokenizer,
                render_cot_messages(dom, c_it, public_bundle, is_long=True),
                max_new_tokens=220,
                device=device,
            )
            long_text_cot_recs.append({
                "id": c_it["id"],
                "text": gt_m["text"],
                "input_tokens": gt_m["input_tokens"],
                "wall_ms": round(gt_m["wall_ms"], 2),
            })

        targets = _collect_qkv_and_mlp_targets(model)
        originals = []
        named_wrappers = []
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        t_tr0 = time.monotonic()
        try:
            for p_str, b_lin in targets:
                par_p, attr_n = p_str.rsplit(".", 1)
                par_m = model.get_submodule(par_p)
                w_m = Rank16LoRAWrapper.wrap_module(torch, nn, b_lin, rank=16, alpha=32.0)
                originals.append((par_m, attr_n, b_lin))
                setattr(par_m, attr_n, w_m)
                named_wrappers.append((p_str, w_m))

            t_params = [p for _n, w in named_wrappers for p in (w.adapter_a, w.adapter_b)]
            opt = torch.optim.AdamW(t_params, lr=0.00028, weight_decay=0.01)
            total_steps = 240
            s_cur = 0
            e_cur = 0
            for st_i in range(total_steps):
                lr_now = 0.00028 * (1.0 - 0.88 * (st_i / float(total_steps - 1)))
                for pg in opt.param_groups:
                    pg["lr"] = lr_now
                opt.zero_grad(set_to_none=True)

                for sub_b in range(2):
                    if extra_trajs and ((st_i * 2 + sub_b) % 4 != 0):
                        item = extra_trajs[e_cur % len(extra_trajs)]
                        e_cur += 1
                    else:
                        item = seed_trajs[s_cur % len(seed_trajs)]
                        s_cur += 1

                    use_res = ((st_i + sub_b) % 2 == 1)
                    f_ids = item["res_full_ids"] if use_res else item["pure_full_ids"]
                    plen = item["res_prompt_len"] if use_res else item["pure_prompt_len"]
                    alen = item["answer_len"]

                    inp = torch.tensor([f_ids[:-1]], dtype=torch.long, device=device)
                    out = model(input_ids=inp, attention_mask=torch.ones_like(inp), use_cache=False, return_dict=True)
                    s_sl = out.logits[0, plen - 1 : plen - 1 + alen].float()
                    t_sl = item["teacher_logits_bf16"].to(device=device, dtype=torch.float32)
                    kl_l = functional.kl_div(
                        functional.log_softmax(s_sl, dim=-1),
                        functional.log_softmax(t_sl, dim=-1),
                        log_target=True,
                        reduction="batchmean",
                    )
                    tgt = torch.tensor(item["target_ids"], dtype=torch.long, device=device)
                    ce_l = functional.cross_entropy(s_sl, tgt)
                    ce_wt = 0.70 if item["source_pool"] == "seed" else 0.25
                    loss = 0.5 * (1.5 * kl_l + ce_wt * ce_l)
                    loss.backward()
                    del out, inp, s_sl, t_sl, kl_l, ce_l, loss

                bg_it = bg_trajs[st_i % len(bg_trajs)]
                bg_kl, _, _ = _compute_trajectory_kl_loss(model, bg_it, device)
                (2.0 * bg_kl).backward()
                del bg_kl

                torch.nn.utils.clip_grad_norm_(t_params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            train_s = time.monotonic() - t_tr0
            for _n, w in named_wrappers:
                w.requires_grad_(False)
                w.eval()

            adapter_dict = {}
            for p_str, w in named_wrappers:
                adapter_dict[f"{p_str}.adapter_a"] = w.adapter_a.detach().cpu().clone()
                adapter_dict[f"{p_str}.adapter_b"] = w.adapter_b.detach().cpu().clone()

            buf = io.BytesIO()
            torch.save({"adapter": adapter_dict, "rank": 16, "alpha": 32.0}, buf)
            pt_bytes = buf.getvalue()
            pt_sha = _sha256_bytes(pt_bytes)
            pt_file = save_dir / pt_name
            fd = os.open(pt_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as st_f:
                os.fchmod(st_f.fileno(), 0o600)
                st_f.write(pt_bytes)
                st_f.flush()
                os.fsync(st_f.fileno())

            lora_pure_recs = []
            lora_residual_recs = []
            for c_it in dom_data["blind_test_inputs"]:
                gp = cached_greedy_generate(
                    model,
                    tokenizer,
                    render_dense_student_messages(dom, c_it, include_micro_residual=False),
                    max_new_tokens=220,
                    device=device,
                )
                lora_pure_recs.append({
                    "id": c_it["id"],
                    "text": gp["text"],
                    "input_tokens": gp["input_tokens"],
                    "wall_ms": round(gp["wall_ms"], 2),
                })
                gr = cached_greedy_generate(
                    model,
                    tokenizer,
                    render_dense_student_messages(dom, c_it, include_micro_residual=True),
                    max_new_tokens=220,
                    device=device,
                )
                lora_residual_recs.append({
                    "id": c_it["id"],
                    "text": gr["text"],
                    "input_tokens": gr["input_tokens"],
                    "wall_ms": round(gr["wall_ms"], 2),
                })

            gen_ret_recs = []
            for g_it in public_bundle["general_retention"]["blind_test_inputs"][:8]:
                g_r = cached_greedy_generate(
                    model, tokenizer, [{"role": "user", "content": g_it["prompt"]}], max_new_tokens=220, device=device
                )
                gen_ret_recs.append({"id": g_it["id"], "text": g_r["text"]})

            bf16_bytes = sum(p.numel() * 2 for p in t_params)
        finally:
            for par_m, attr_n, b_lin in reversed(originals):
                setattr(par_m, attr_n, b_lin)

        out_domains[dom] = {
            "unlabeled_probes_attempted": len(all_unlabeled),
            "self_consistent_accepted": len(extra_trajs),
            "total_distill_trajectories": len(seed_trajs) + len(extra_trajs),
            "mining_seconds": round(mine_s, 2),
            "training_seconds": round(train_s, 2),
            "rank": 16,
            "alpha": 32.0,
            "lora_bf16_mib": round(bf16_bytes / (1024.0 * 1024.0), 3),
            "artifact_name": pt_name,
            "artifact_sha256": pt_sha,
            "base_micro_table_only_records": base_micro_table_recs,
            "long_text_cot_records": long_text_cot_recs,
            "lora_dense_pure_records": lora_pure_recs,
            "lora_dense_plus_micro_table_records": lora_residual_recs,
            "general_retention_8_records": gen_ret_recs,
        }

    if _tensor_digest(_state(model)) != init_sha:
        raise RuntimeError("Base model weights drifted")

    Path(args.out_json).write_text(
        json.dumps(
            {
                "kind": "DENSE_COMPILATION_AND_RESIDUAL_STAGE_RAW",
                "gpu_summary": {k: v for k, v in selected_gpu.items() if k != "index"},
                "domains": out_domains,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
