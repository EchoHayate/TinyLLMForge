from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


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
            return {
                "index": int(idx_s),
                "name": name_s,
                "free_mib": int(free_s),
                "used_mib": int(used_s),
            }
    raise RuntimeError("No idle A100 80GB GPU available")


def _write_private_bytes(path_obj: Path, raw: bytes) -> None:
    path_obj.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path_obj, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-g12-adapter-path", required=True)
    parser.add_argument("--code-g12-expected-sha256", required=True)
    parser.add_argument("--adapter-save-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    selected_gpu = _select_idle_a100()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu["index"])

    repo_root = Path(args.repo_root).resolve()
    if (repo_root / "experiments/cross_domain_prefix_lora/data/cross_domain_v1/blind_oracles.json").exists():
        raise RuntimeError("BLIND_ORACLE_VIOLATION: blind_oracles.json must never exist on GPU stage")
    if (repo_root / "experiments/lora_necessity/data/confirmatory_v2_r1/oracles.json").exists():
        raise RuntimeError("CODE_ORACLE_VIOLATION: oracles.json must never exist on GPU stage")

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "experiments/lora_necessity"))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from confirmatory_v2_contract import TRANSFER_IDS, confirmatory_first_round
    from confirmatory_v2_converged_ablation import mounted_converged_adapter
    from training_worker import _state, _tensor_digest
    from experiments.cross_domain_prefix_lora.domain_sop_contracts import (
        render_domain_long_messages,
        render_domain_short_messages,
    )
    from experiments.cross_domain_prefix_lora.surgical_json_csd_trainer import (
        LOCKED_CROSS_DOMAIN_HYPERPARAMS,
        cached_greedy_generate,
        prepare_domain_distillation_trajectories,
        prepare_multimodal_background_trajectories,
        train_domain_sop_lora_adapter,
    )

    code_g12_raw = Path(args.code_g12_adapter_path).read_bytes()
    if _sha256_bytes(code_g12_raw) != args.code_g12_expected_sha256:
        raise ValueError("Gate G12 code adapter SHA256 mismatch")
    code_g12_loaded = torch.load(io.BytesIO(code_g12_raw), map_location="cpu", weights_only=True)
    code_g12_pack = {"adapter": code_g12_loaded["adapter"]}

    pub_path = repo_root / "experiments/cross_domain_prefix_lora/data/cross_domain_v1/public_bundle.json"
    manifest_path = repo_root / "experiments/cross_domain_prefix_lora/data/cross_domain_v1/manifest.json"
    pub_raw = pub_path.read_bytes()
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if _sha256_bytes(pub_raw) != manifest["public_bundle_sha256"]:
        raise ValueError("public_bundle.json SHA256 mismatch")
    public_bundle = json.loads(pub_raw.decode("utf-8"))

    conf_dir = repo_root / "experiments/lora_necessity/data/confirmatory_v2_r1"
    code_prompts_all = json.loads((conf_dir / "prompts.json").read_text("utf-8"))
    code_sup_raw = (conf_dir / "supervision.json").read_bytes()
    code_wt_tasks = [item for item in code_prompts_all if item["id"] in TRANSFER_IDS]
    if len(code_wt_tasks) != 12:
        raise RuntimeError("Expected 12 WT transfer tasks")

    torch.cuda.init()
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=False, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    init_base_sha = _tensor_digest(_state(model))

    t_bg0 = time.monotonic()
    bg_trajs = prepare_multimodal_background_trajectories(
        model,
        tokenizer,
        public_bundle["background_anchors"],
        max_new_tokens=int(LOCKED_CROSS_DOMAIN_HYPERPARAMS["max_anchor_tokens"]),
    )
    bg_prep_s = time.monotonic() - t_bg0

    save_dir = Path(args.adapter_save_dir).resolve()
    save_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    domain_training_summaries = {}
    trained_packs = {}

    for dom_name, fname in (
        ("med_triage", "med_sop_adapter.pt"),
        ("legal_compliance", "legal_sop_adapter.pt"),
    ):
        t_p0 = time.monotonic()
        pos_trajs, mining_stats = prepare_domain_distillation_trajectories(
            model,
            tokenizer,
            dom_name,
            public_bundle,
            hyperparams=LOCKED_CROSS_DOMAIN_HYPERPARAMS,
        )
        prep_s = time.monotonic() - t_p0

        train_out = train_domain_sop_lora_adapter(
            model,
            pos_trajs,
            bg_trajs,
            hyperparams=LOCKED_CROSS_DOMAIN_HYPERPARAMS,
        )
        del pos_trajs

        buf = io.BytesIO()
        torch.save({"adapter": train_out["adapter"]}, buf)
        art_bytes = buf.getvalue()
        art_sha = _sha256_bytes(art_bytes)
        out_pt = save_dir / fname
        _write_private_bytes(out_pt, art_bytes)

        reloaded = torch.load(io.BytesIO(out_pt.read_bytes()), map_location="cpu", weights_only=True)
        if _tensor_digest(reloaded["adapter"]) != train_out["adapter_sha256"]:
            raise RuntimeError(f"Reloaded adapter digest mismatch for {dom_name}")
        trained_packs[dom_name] = {"adapter": reloaded["adapter"]}

        seed_steps = [st for st in train_out["steps"] if st["source_pool"] == "seed"]
        domain_training_summaries[dom_name] = {
            "hyperparams": train_out["hyperparams"],
            "positive_trajectory_count": train_out["positive_trajectory_count"],
            "seed_trajectory_count": train_out["seed_trajectory_count"],
            "verified_unlabeled_trajectory_count": train_out["verified_unlabeled_trajectory_count"],
            "mining_stats": mining_stats,
            "adapter_bytes_fp32": train_out["adapter_bytes"],
            "adapter_bytes_bf16_equiv": train_out["adapter_bytes"] // 2,
            "artifact_sha256": art_sha,
            "artifact_path_name": fname,
            "trajectory_prep_seconds": round(prep_s, 3),
            "training_wall_seconds": round(train_out["core_wall_ns"] / 1e9, 3),
            "first_seed_surgical_ce": round(seed_steps[0]["surgical_ce_loss"], 5),
            "last_seed_surgical_ce": round(seed_steps[-1]["surgical_ce_loss"], 5),
            "last_pos_kl": round(train_out["steps"][-1]["pos_distill_kl"], 5),
            "last_bg_kl": round(train_out["steps"][-1]["bg_kl_loss"], 6),
        }

    del bg_trajs

    long_text_baselines: dict[str, dict] = {}
    for dom_name in ("code_bugfix", "med_triage", "legal_compliance"):
        recs = []
        lat_ms = []
        in_toks = []
        if dom_name == "code_bugfix":
            for t_obj in code_wt_tasks:
                msgs = confirmatory_first_round(t_obj, "text", code_sup_raw)
                gen = cached_greedy_generate(model, tokenizer, msgs, max_new_tokens=256, device=device)
                lat_ms.append(gen["wall_ms"])
                in_toks.append(gen["input_tokens"])
                recs.append({
                    "id": t_obj["id"],
                    "text": gen["text"],
                    "input_tokens": gen["input_tokens"],
                    "output_tokens": gen["output_tokens"],
                    "wall_ms": round(gen["wall_ms"], 3),
                })
        else:
            for c_obj in public_bundle[dom_name]["blind_test_inputs"]:
                msgs = render_domain_long_messages(dom_name, c_obj, public_bundle, exclude_seed_id=None)
                gen = cached_greedy_generate(model, tokenizer, msgs, max_new_tokens=192, device=device)
                lat_ms.append(gen["wall_ms"])
                in_toks.append(gen["input_tokens"])
                recs.append({
                    "id": c_obj["id"],
                    "text": gen["text"],
                    "input_tokens": gen["input_tokens"],
                    "output_tokens": gen["output_tokens"],
                    "wall_ms": round(gen["wall_ms"], 3),
                })
        long_text_baselines[dom_name] = {
            "mean_input_tokens": round(statistics.mean(in_toks), 1),
            "total_wall_ms": round(sum(lat_ms), 2),
            "median_wall_ms": round(statistics.median(lat_ms), 2),
            "records": recs,
        }

    slot_specs = [
        ("slot_0_base", None),
        ("slot_1_code_lora", code_g12_pack),
        ("slot_2_med_lora", trained_packs["med_triage"]),
        ("slot_3_legal_lora", trained_packs["legal_compliance"]),
    ]

    matrix_generations: dict[str, dict[str, dict]] = {}
    for slot_name, pack_obj in slot_specs:
        matrix_generations[slot_name] = {}
        ctx = mounted_converged_adapter(model, pack_obj) if pack_obj is not None else nullcontext(model)
        with ctx as active_mod:
            for eval_dom in ("code_bugfix", "med_triage", "legal_compliance", "general_retention"):
                recs = []
                lat_ms = []
                in_toks = []
                if eval_dom == "code_bugfix":
                    for t_obj in code_wt_tasks:
                        msgs = confirmatory_first_round(t_obj, "lora", code_sup_raw)
                        gen = cached_greedy_generate(
                            active_mod, tokenizer, msgs, max_new_tokens=256, device=device
                        )
                        lat_ms.append(gen["wall_ms"])
                        in_toks.append(gen["input_tokens"])
                        recs.append({
                            "id": t_obj["id"],
                            "text": gen["text"],
                            "input_tokens": gen["input_tokens"],
                            "output_tokens": gen["output_tokens"],
                            "wall_ms": round(gen["wall_ms"], 3),
                        })
                elif eval_dom in ("med_triage", "legal_compliance"):
                    for c_obj in public_bundle[eval_dom]["blind_test_inputs"]:
                        msgs = render_domain_short_messages(eval_dom, c_obj)
                        gen = cached_greedy_generate(
                            active_mod, tokenizer, msgs, max_new_tokens=192, device=device
                        )
                        lat_ms.append(gen["wall_ms"])
                        in_toks.append(gen["input_tokens"])
                        recs.append({
                            "id": c_obj["id"],
                            "text": gen["text"],
                            "input_tokens": gen["input_tokens"],
                            "output_tokens": gen["output_tokens"],
                            "wall_ms": round(gen["wall_ms"], 3),
                        })
                else:
                    for g_obj in public_bundle["general_retention"]["blind_test_inputs"]:
                        msgs = render_domain_short_messages("general_retention", g_obj)
                        gen = cached_greedy_generate(
                            active_mod, tokenizer, msgs, max_new_tokens=256, device=device
                        )
                        lat_ms.append(gen["wall_ms"])
                        in_toks.append(gen["input_tokens"])
                        recs.append({
                            "id": g_obj["id"],
                            "text": gen["text"],
                            "input_tokens": gen["input_tokens"],
                            "output_tokens": gen["output_tokens"],
                            "wall_ms": round(gen["wall_ms"], 3),
                        })

                matrix_generations[slot_name][eval_dom] = {
                    "mean_input_tokens": round(statistics.mean(in_toks), 1),
                    "total_wall_ms": round(sum(lat_ms), 2),
                    "median_wall_ms": round(statistics.median(lat_ms), 2),
                    "records": recs,
                }

    final_base_sha = _tensor_digest(_state(model))
    if final_base_sha != init_base_sha:
        raise RuntimeError("Base model weights drifted after cross-domain runs")

    raw_stage_out = {
        "kind": "CROSS_DOMAIN_BLIND_GPU_STAGE1_RAW",
        "version": "cross_domain_v1",
        "single_shot_blind_protocol": True,
        "zero_blind_oracle_in_gpu_payload": True,
        "gpu_summary": selected_gpu,
        "public_bundle_sha256": manifest["public_bundle_sha256"],
        "base_model_sha256_verified_unchanged": final_base_sha == init_base_sha,
        "background_anchor_prep_seconds": round(bg_prep_s, 3),
        "domain_training_summaries": domain_training_summaries,
        "long_text_baselines": long_text_baselines,
        "matrix_generations": matrix_generations,
    }
    Path(args.out_json).write_text(json.dumps(raw_stage_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
