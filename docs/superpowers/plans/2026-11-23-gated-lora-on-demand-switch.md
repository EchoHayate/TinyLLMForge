# Gate G6: Route A Gated LoRA On-Demand Switch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the lightweight inference-time background-denoised prototype switch (`gated_lora_router.py` and `confirmatory_v2_gated_ablation.py`) that attaches a `~148 KiB` task-routing fingerprint to the 96-step converged LoRA adapter, validates out-of-sample routing accuracy on the 40-task development probe (Gate G6-A, zero `WT/WR` exposure) with an explicit early-stop pivot rule, and—upon passing Gate G6-A—evaluates 24-task end-to-end gated accuracy and latency on `confirmatory_v2_r1` (Gate G6-B).

**Architecture:** `gated_lora_router.py` strips task IDs (`"id": "TASK"`), locates the `"Current task:\n"` token span without instruction boilerplate dilution, pools normalized mid-to-late (`layers 18..30`) unadapted `Qwen3-8B` hidden states into a 4096-D unit vector, subtracts the 16-task `v1` unrelated background mean (`R01..R16`), scores nearest-prototype cosine similarity against the 8 supervision anchors (`V01..V08`), and solves a parameter-free Leave-One-Out max-margin threshold `tau`. `confirmatory_v2_gated_ablation.py` orchestrates both the 40-task read-only Gate G6-A probe (`V01..V08`, `R01..R16`, `C01..C12`, `D01..D04`) and the two-phase Gate G6-B evaluation (running `gate_open=False` tasks directly on unadapted `base` and `gate_open=True` tasks inside a single `mounted_converged_adapter` context, followed by isolated `bwrap`+`seccomp` CPU scoring and bit-exact cross-checks against archived Gate G4-B and Gate G5 outputs).

**Tech Stack:** Python `3.12` stdlib (`math`, `json`, `hashlib`, `ast`, `tokenize`, `unittest`), PyTorch `2.5+` (`Qwen3ForCausalLM` BF16 `eager`, `use_cache=False`), remote `bwrap` + `seccomp` sandboxed task worker.

## Global Constraints

- Preserve all pre-existing committed files in `experiments/lora_necessity/` and `artifacts/lora_necessity_v2/20261122-confirmatory-r4/` byte-for-byte without modification.
- Enforce **zero `#` comments (`tokenize.COMMENT`)** and **zero single-letter AST identifiers (`Name`, `arg`, `ExceptHandler`, `alias`)** across every Python file created under `experiments/lora_necessity/`.
- Enforce strict data partitioning and SHA256 file pins from `docs/superpowers/specs/2026-11-23-gated-lora-on-demand-switch-design.md`: `V01..V08` (`1b2173eb...`), `prompts_v1.json` (`3668d996...`), `structure_v1.json` (`3905663b...`), `recalibration_v2_r1/prompts.json` (`0b8b0370...`), `confirmatory_v2_r1/prompts.json` (`f78ea75b...`), and `confirmatory_v2_r1/oracles.json` (`8d251cdc...`).
- Never read `family`, `split`, `mechanics`, `reference_source`, `initial_witness`, or `oracles` inside any router function; rewrite `"id"` to `"TASK"` before rendering router prompts.
- Do not run `git commit`, `git push`, or any live remote SSH command without explicit user authorization at its gate.

---

### Task 1: Sanitized Router Math, LOO Threshold Solver & Hidden-State Extractor (`gated_lora_router.py`)

**Files:**
- Create: `experiments/lora_necessity/gated_lora_router.py`
- Test: `experiments/lora_necessity/test_gated_lora_router.py`

**Interfaces:**
- Consumes: `confirmatory_v2_contract._render_user_prompt`, `prompt_rendering.PUBLIC_FIELDS`, `pilot_protocol.{FLAGS, fields, require}`, `validation_contract.{digest, encoded}`
- Produces:
  - `ROUTER_LAYER_INDICES = tuple(range(18, 31))`
  - `ROUTER_ALLOWED_PUBLIC_KEYS = ("entry_point", "spec", "initial_source", "public_cases")`
  - `sanitize_task_for_router(task_record: dict) -> dict`
  - `render_sanitized_router_prompt(task_record: dict) -> list[dict]`
  - `locate_task_token_span(tokenizer, messages: list[dict]) -> tuple[list[int], int, int]`
  - `extract_router_embedding(model, tokenizer, task_record: dict, *, layer_indices=ROUTER_LAYER_INDICES) -> list[float]`
  - `fit_whitened_prototype_router(positive_vectors: dict[str, list[float]], background_vectors: dict[str, list[float]]) -> dict`
  - `score_router_embedding(router_bundle: dict, query_vector: list[float]) -> dict`
  - `validate_router_bundle(router_bundle: dict) -> str`
  - `evaluate_gate_g6a_probe(router_bundle: dict, dev_positive_vectors: dict[str, list[float]], dev_negative_vectors: dict[str, list[float]]) -> dict`

- [ ] **Step 1: Write the failing unit test suite (`experiments/lora_necessity/test_gated_lora_router.py`)**

```python
import ast
import math
from pathlib import Path
import tokenize
import unittest

from gated_lora_router import (
    ROUTER_LAYER_INDICES,
    evaluate_gate_g6a_probe,
    extract_router_embedding,
    fit_whitened_prototype_router,
    locate_task_token_span,
    render_sanitized_router_prompt,
    sanitize_task_for_router,
    score_router_embedding,
    validate_router_bundle,
)


def _sample_task(task_id, entry_name="clip_window", extra_fields=None):
    record = {
        "id": task_id,
        "entry_point": entry_name,
        "spec": "Clip intervals to [low, high) and sum widths.",
        "initial_source": "def " + entry_name + "(low, high, spans):\n    return 0\n",
        "public_cases": [
            {"id": "P01", "args": [0, 5, [[1, 3]]], "expected": {"kind": "return", "value": 2}},
            {"id": "P02", "args": [0, 5, [[4, 8]]], "expected": {"kind": "return", "value": 1}},
        ],
    }
    if extra_fields:
        record.update(extra_fields)
    return record


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize=True, add_generation_prompt=False):
        assert tokenize is True
        text_body = messages[0]["content"]
        encoded_bytes = list(text_body.encode("utf-8"))
        if add_generation_prompt:
            return [901, 902] + encoded_bytes + [903, 904, 905]
        return [901, 902] + encoded_bytes + [903]


class GatedLoRARouterTest(unittest.TestCase):
    def test_sanitization_strips_task_id_and_ignores_metadata(self):
        first_task = _sample_task(
            "WT01",
            extra_fields={"family": "bounds", "split": "transfer", "reference_source": "secret_a"},
        )
        second_task = _sample_task(
            "WR99",
            extra_fields={"family": "dispatch", "split": "retention", "reference_source": "secret_b"},
        )
        sanitized_first = sanitize_task_for_router(first_task)
        sanitized_second = sanitize_task_for_router(second_task)
        self.assertEqual(sanitized_first, sanitized_second)
        self.assertEqual(sanitized_first["id"], "TASK")
        self.assertNotIn("family", sanitized_first)
        self.assertNotIn("split", sanitized_first)
        self.assertEqual(
            render_sanitized_router_prompt(first_task),
            render_sanitized_router_prompt(second_task),
        )

    def test_locate_task_token_span_excludes_instruction_and_gen_suffix(self):
        fake_tok = FakeTokenizer()
        msgs = render_sanitized_router_prompt(_sample_task("V01"))
        full_ids, span_start, span_end = locate_task_token_span(fake_tok, msgs)
        self.assertGreater(span_start, 20)
        self.assertLess(span_end, len(full_ids))
        extracted_bytes = bytes(full_ids[span_start:span_end]).decode("utf-8")
        self.assertTrue(extracted_bytes.startswith('Task: {"cases":'))
        self.assertNotIn("Current task:", extracted_bytes)

    def test_whitened_prototype_router_separates_clusters_and_passes_g6a(self):
        dim_size = 16
        common_bias = [3.0] + [0.0] * (dim_size - 1)

        def make_unit(axis_idx, scale_val=1.0, noise_axis=15, noise_val=0.0):
            raw_vec = list(common_bias)
            raw_vec[axis_idx] += scale_val
            raw_vec[noise_axis] += noise_val
            norm_val = math.sqrt(sum(val_x * val_x for val_x in raw_vec))
            return [val_x / norm_val for val_x in raw_vec]

        pos_map = {}
        for fam_idx, fam_axis in enumerate((1, 2, 3, 4)):
            pos_map["V%02d" % (fam_idx * 2 + 1)] = make_unit(fam_axis, 1.0, 14, 0.05)
            pos_map["V%02d" % (fam_idx * 2 + 2)] = make_unit(fam_axis, 0.95, 15, -0.05)

        bg_map = {}
        for bg_idx in range(16):
            bg_axis = 5 + (bg_idx % 4)
            bg_map["R%02d" % (bg_idx + 1)] = make_unit(bg_axis, 1.0, 13, 0.02 * (bg_idx - 8))

        bundle = fit_whitened_prototype_router(pos_map, bg_map)
        bundle_sha = validate_router_bundle(bundle)
        self.assertEqual(len(bundle_sha), 64)
        self.assertEqual(bundle["loo_summary"]["balanced_accuracy"], 1.0)

        dev_pos = {
            "C%02d" % (idx_num + 1): make_unit(1 + (idx_num % 4), 0.9, 14, 0.03)
            for idx_num in range(12)
        }
        dev_neg = {
            "D%02d" % (idx_num + 1): make_unit(9 + idx_num, 1.0, 15, 0.02)
            for idx_num in range(4)
        }
        probe_report = evaluate_gate_g6a_probe(bundle, dev_pos, dev_neg)
        self.assertTrue(probe_report["gate_g6a_passed"])
        self.assertEqual(probe_report["dev_positive_recall_count"], 12)
        self.assertEqual(probe_report["dev_negative_false_open_count"], 0)

        in_domain_decision = score_router_embedding(bundle, dev_pos["C01"])
        out_domain_decision = score_router_embedding(bundle, dev_neg["D01"])
        self.assertTrue(in_domain_decision["gate_open"])
        self.assertFalse(out_domain_decision["gate_open"])

    def test_extract_router_embedding_on_tiny_torch_stub(self):
        try:
            import torch
            from torch import nn
        except ImportError:
            self.skipTest("torch unavailable in isolated stdlib python")

        class TinyHiddenOutput:
            def __init__(self, hidden_states):
                self.hidden_states = hidden_states

        class TinyBaseStub(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy_param = nn.Parameter(torch.ones(1, 8, dtype=torch.bfloat16))

            def forward(self, input_ids, attention_mask=None, use_cache=False, output_hidden_states=False, return_dict=True):
                assert use_cache is False and output_hidden_states is True and return_dict is True
                seq_len = input_ids.shape[1]
                states = []
                for layer_num in range(37):
                    base_tensor = torch.zeros(1, seq_len, 8, dtype=torch.bfloat16)
                    base_tensor[0, :, layer_num % 8] = float(layer_num + 1)
                    states.append(base_tensor)
                return TinyHiddenOutput(tuple(states))

        vec = extract_router_embedding(
            TinyBaseStub(),
            FakeTokenizer(),
            _sample_task("WT05"),
            layer_indices=ROUTER_LAYER_INDICES,
        )
        self.assertEqual(len(vec), 8)
        vec_norm = math.sqrt(sum(comp * comp for comp in vec))
        self.assertAlmostEqual(vec_norm, 1.0, places=6)

    def test_zero_comments_and_zero_single_letter_identifiers(self):
        module_dir = Path(__file__).resolve().parent
        for file_name in ("gated_lora_router.py", "test_gated_lora_router.py"):
            target_path = module_dir / file_name
            with target_path.open("rb") as raw_stream:
                comment_tokens = [
                    tok_info
                    for tok_info in tokenize.tokenize(raw_stream.readline)
                    if tok_info.type == tokenize.COMMENT
                ]
            self.assertEqual(comment_tokens, [], file_name)
            parsed_tree = ast.parse(target_path.read_text(encoding="utf-8"), filename=file_name)
            bad_names = []
            for node_item in ast.walk(parsed_tree):
                if isinstance(node_item, ast.Name) and len(node_item.id) == 1:
                    bad_names.append(node_item.id)
                elif isinstance(node_item, ast.arg) and len(node_item.arg) == 1:
                    bad_names.append(node_item.arg)
                elif isinstance(node_item, ast.ExceptHandler) and node_item.name and len(node_item.name) == 1:
                    bad_names.append(node_item.name)
                elif isinstance(node_item, ast.alias):
                    bound_name = node_item.asname or node_item.name
                    if len(bound_name) == 1:
                        bad_names.append(bound_name)
            self.assertEqual(bad_names, [], file_name + ": " + repr(bad_names))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the unit tests to verify RED failure**

Run: `PYTHONPATH=/Users/bytedance/dev/TinyLLMForge/experiments/lora_necessity /usr/bin/python3 -I -S -B -m unittest test_gated_lora_router -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gated_lora_router'`

- [ ] **Step 3: Implement `experiments/lora_necessity/gated_lora_router.py`**

Implement `sanitize_task_for_router`, `render_sanitized_router_prompt`, `locate_task_token_span` (finding exact token prefix for `INSTRUCTION + "\n\nCurrent task:\n"` via two distinct dummy task blocks and exact suffix via `add_generation_prompt=True` vs `False`), `extract_router_embedding` (mean-pooling `hidden_states[layer_idx][0, span_start:span_end].float()`, L2-normalizing per layer across `18..30`, averaging, and L2-normalizing), `fit_whitened_prototype_router` (background mean $\mu_B$, LOO positive scores against $P \setminus \{p_{\text{idx}}\}$, LOO background scores centered on $\mu_{B \setminus \{b_{\text{idx}}\}}$ against $P$, max-margin midpoint threshold $\tau$), `score_router_embedding`, `validate_router_bundle`, and `evaluate_gate_g6a_probe` (enforcing `loo_bal_acc >= 0.80`, `dev_pos_recall >= 9/12`, `dev_neg_fpr <= 1/4`, `loo_bg_fpr <= 4/16`). Ensure zero `#` comments and zero single-letter AST identifiers.

- [ ] **Step 4: Run both isolated stdlib and PyTorch venv tests to verify GREEN**

Run:
1. `PYTHONPATH=/Users/bytedance/dev/TinyLLMForge/experiments/lora_necessity /usr/bin/python3 -I -S -B -m unittest test_gated_lora_router -v`
2. `PYTHONPATH=/Users/bytedance/dev/TinyLLMForge/experiments/lora_necessity /Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python -B -m unittest test_gated_lora_router -v`
Expected: All 5 tests PASS (`0` skipped under `.venv/bin/python`, `1` skipped under stdlib `-I -S`).

---

### Task 2: Gate G6-A Probe & Gate G6-B End-to-End Gated Ablation Orchestrator (`confirmatory_v2_gated_ablation.py`)

**Files:**
- Create: `experiments/lora_necessity/confirmatory_v2_gated_ablation.py`
- Test: `experiments/lora_necessity/test_confirmatory_v2_gated_ablation.py`

**Interfaces:**
- Consumes: `gated_lora_router.*`, `confirmatory_v2_converged_ablation.{ABLATION_TRAINING_HYPERPARAMS, G5_STDLIB_BOOTSTRAP, train_converged_ablation_adapter, mounted_converged_adapter}`, `confirmatory_v2_contract.*`, `confirmatory_v2_gate_a.{build_multi_bundle_ssh_argv, run_single_ssh_json_call}`, `recalibration_diagnostic_scoring.*`
- Produces:
  - `V1_PROMPTS_SHA256`, `V1_STRUCTURE_SHA256`, `RECAL_V2_R1_PROMPTS_SHA256`
  - `load_g6_calibration_and_probe_inputs(supervision_v2_raw: bytes, v1_prompts_raw: bytes, v1_structure_raw: bytes, recal_prompts_raw: bytes) -> dict`
  - `build_gate_g6a_probe_package(...) -> dict`
  - `execute_remote_g6a_probe_body(...) -> dict`
  - `run_gate_g6a_probe_stage(package: dict, *, local_run_root: str) -> dict`
  - `build_gated_ablation_schedule(prompts_raw: bytes, supervision_raw: bytes, *, run_id: str) -> dict`
  - `build_gate_g6b_ablation_package(...) -> dict`
  - `execute_remote_g6b_ablation_body(...) -> dict`
  - `score_gated_ablation_run(raw_files: dict, phase_one_result: dict, batch_envelopes: list, g4b_report: dict, g5_report: dict, *, run_id: str) -> dict`
  - `run_gate_g6b_ablation_stage(package: dict, g4b_report_raw: bytes, g5_report_raw: bytes, *, local_run_root: str) -> dict`

- [ ] **Step 1: Write failing unit tests (`experiments/lora_necessity/test_confirmatory_v2_gated_ablation.py`)**

Cover:
1. `load_g6_calibration_and_probe_inputs` against the real repository dataset files (`V01..V08` = 8, `R01..R16` = 16, `C01..C12` = 12, `D01..D04` = 4; total = 40 disjoint tasks) and verify zero `WT`/`WR` tasks exist in the G6-A input or `build_gate_g6a_probe_package` payload hex;
2. `build_gate_g6b_ablation_package` payload inspection verifying zero `oracles.json` and zero `H01..H06` private test cases in Phase 1;
3. Synthetic `score_gated_ablation_run` test verifying:
   - `gate_open == False` tasks cross-check byte-for-byte against `g4b_report` `base` round-1 outputs;
   - `gate_open == True` tasks cross-check byte-for-byte against `g5_report` `lora_conv` round-1 outputs;
   - Paired comparisons (`lora_gated` vs `base`, `lora_gated` vs `lora_conv`, `lora_gated` vs `text`) and break-even amortization math (`(train_fixed_ns + router_fit_ns) / per_task_saved_ns`) match exact arithmetic;
   - All governance flags (`benchmark_eligible`, `formal_evaluation_ready`, `isolated`, `run_authorized`, `diagnostic_dispatch_authorized`) remain `False`;
4. AST & comment hygiene check (`0` `tokenize.COMMENT`, `0` single-letter identifiers) across both `confirmatory_v2_gated_ablation.py` and `test_confirmatory_v2_gated_ablation.py`.

- [ ] **Step 2: Run unit tests to verify RED failure**

Run: `PYTHONPATH=/Users/bytedance/dev/TinyLLMForge/experiments/lora_necessity /usr/bin/python3 -I -S -B -m unittest test_confirmatory_v2_gated_ablation -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'confirmatory_v2_gated_ablation'`

- [ ] **Step 3: Implement `experiments/lora_necessity/confirmatory_v2_gated_ablation.py`**

Implement both Gate G6-A (`40` unadapted base Prefill calls -> `fit_whitened_prototype_router` -> `evaluate_gate_g6a_probe`, returning full per-task scores, LOO summary, `C01..C12` recall, `D01..D04` FPR, and `gate_g6a_passed`) and Gate G6-B (requires verified G6-A pass receipt hash; trains 96-step converged LoRA, fits router on unadapted `base` for `V01..V08` + `R01..R16`, computes router decisions for `WT01..WR12` + `WT01/WR01` canaries on unadapted `base`, executes `base_canary` and `gate_open == False` episodes on unadapted `base`, enters `mounted_converged_adapter` **exactly once** to execute all `gate_open == True` episodes, restores canonical episode ordering `0..25`, seals `phase1-wire.bin`, runs Phase 2 isolated `bwrap`+`seccomp` scoring via `run_single_ssh_json_call`, and cross-checks byte-exact parity against archived Gate G4-B and Gate G5 reports).

- [ ] **Step 4: Run full unit test regression suite to verify GREEN**

Run:
`PYTHONPATH=/Users/bytedance/dev/TinyLLMForge/experiments/lora_necessity /usr/bin/python3 -I -S -B -m unittest test_recalibration_diagnostic_scoring test_recalibration_diagnostic_controller test_confirmatory_v2_contract test_confirmatory_v2_gate_a test_confirmatory_v2_lora_stage test_confirmatory_v2_converged_ablation test_gated_lora_router test_confirmatory_v2_gated_ablation -v`
Expected: All existing 25 tests + all new Gate G6 tests PASS with zero errors and zero baseline file modifications.

---

### Task 3: Build & Audit Frozen Gate G6-A 40-Task Probe Launch Package (Local Only)

**Files:**
- Create (private `/private/tmp/`, mode `0700`/`0600`): `/private/tmp/lora-conf-v2-gate-g6a-package-20261123/gate-g6a-package-full.json`, `/private/tmp/lora-conf-v2-gate-g6a-package-20261123/gate-g6a-package-redacted.json`, `/private/tmp/lora-conf-v2-gate-g6a-package-20261123/execute_gate_g6a_once.py`

**Interfaces:**
- Consumes: `confirmatory_v2_gated_ablation.{build_gate_g6a_probe_package, run_gate_g6a_probe_stage}` and preserved preflight/SSH runtime binding from `/private/tmp/lora-conf-v2-gate-g5-runs-r2-20261122-hvwvhdd7`
- Produces: Byte-reproducible Gate G6-A launch package (`0` `WT/WR` tasks, `0` `oracles.json`, `0` sensitive strings in redacted JSON, unconsumed `0700` run root)

- [ ] **Step 1: Materialize and verify the Gate G6-A package offline**

Run `/usr/bin/python3 -I -S -B` offline builder and audit script verifying:
1. `phase_one_payload_hex` contains only `V01..V08`, `R01..R16`, `C01..C12`, `D01..D04` (`40` tasks total) and zero occurrences of `WT01..WT12`, `WR01..WR12`, or `oracles.json`;
2. `build_gate_g6a_probe_package` reconstructs `gate-g6a-package-full.json` byte-for-byte;
3. `gate-g6a-package-redacted.json` contains zero hostnames, usernames, UIDs, GPU UUIDs, or remote paths;
4. Local SSH config and `subprocess.py` closure digest match `check_local_ssh_and_popen_closure`.

- [ ] **Step 2: Pause and request explicit single-shot authorization for Gate G6-A (~1 min GPU Prefill probe)**

Present exact scope (`40` read-only `Qwen3-8B` Prefill passes, `0` generations, `0` `WT/WR` exposure, `< 60s` GPU wall time) and wait for explicit user confirmation before executing `execute_gate_g6a_once.py`.

---

### Task 4: Execute Gate G6-A Probe, Enforce Early-Stop Pivot Rule, & Run Gate G6-B End-to-End Evaluation

**Files:**
- Create (on G6-B completion, redacted): `artifacts/lora_necessity_v2/20261123-gated-ablation-r1/gate-g6a-router-probe-report.json`, `artifacts/lora_necessity_v2/20261123-gated-ablation-r1/gate-g6b-gated-ablation-report.json`, `experiments/lora_necessity/20261123-gated-lora-ablation-results.md`

**Interfaces:**
- Consumes: Approved `execute_gate_g6a_once.py`, and (iff `gate_g6a_passed is True`) `build_gate_g6b_ablation_package` / `run_gate_g6b_ablation_stage`
- Produces: Verified G6-A probe report and either (a) immediate clean pivot to Route B if any G6-A threshold fails, or (b) verified G6-B end-to-end 24-task report with bit-exact parity check against G4-B `base` and G5 `lora_conv`.

- [ ] **Step 1: Execute authorized single-shot Gate G6-A probe and verify 3 pass gates**

Verify `loo_summary.balanced_accuracy >= 0.80`, `dev_positive_recall_count >= 9` (`>= 75%` on `C01..C12`), `dev_negative_false_open_count <= 1` (`<= 25%` on `D01..D04`), and `loo_background_false_open_count <= 4` (`<= 25%` on `R01..R16`). If any check fails, stop Route A immediately and report exact score distributions to begin Route B spec.

- [ ] **Step 2: If Gate G6-A passes, materialize & audit Gate G6-B package and pause for explicit single-shot confirmation**

Build `/private/tmp/lora-conf-v2-gate-g6b-package-20261123/gate-g6b-package-full.json` bound to the verified G6-A receipt SHA256, audit zero `oracles.json` leakage in Phase 1 and zero sensitive strings in redacted metadata, and request user confirmation for single-shot Gate G6-B execution (`~15` min on 1× A100).

- [ ] **Step 3: Execute authorized single-shot Gate G6-B, cross-check bit-exact parity, and write redacted results report**

Verify `48/48` candidates scored (`0 UNKNOWN`), `100%` round-1 bit-exact match of `gate_open == False` tasks with G4-B `base` and `gate_open == True` tasks with G5 `lora_conv`, record `transfer` (`WT01..WT12`) and `retention` (`WR01..WR12`) pass counts plus speed/size amortization in `experiments/lora_necessity/20261123-gated-lora-ablation-results.md`.
