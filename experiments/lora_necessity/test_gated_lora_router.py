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
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=True,
        truncation=False,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        assert tokenize is True and truncation is False and enable_thinking is False
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
        self.assertEqual(bundle["loo_summary"]["background_false_open_count"], 0)

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

        bad_dev_pos = {
            "C%02d" % (idx_num + 1): make_unit(9 + (idx_num % 4), 1.0, 15, 0.01)
            for idx_num in range(12)
        }
        failed_probe = evaluate_gate_g6a_probe(bundle, bad_dev_pos, dev_neg)
        self.assertFalse(failed_probe["gate_g6a_passed"])
        self.assertIn("dev_positive_recall_below_9_of_12", failed_probe["failed_gates"])

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

            def forward(
                self,
                input_ids,
                attention_mask=None,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            ):
                assert use_cache is False and output_hidden_states is True and return_dict is True
                seq_len = input_ids.shape[1]
                states = []
                for layer_num in range(37):
                    base_tensor = torch.zeros(1, seq_len, 8, dtype=torch.bfloat16)
                    base_tensor[0, :, layer_num % 8] = float(layer_num + 1)
                    states.append(base_tensor)
                return TinyHiddenOutput(tuple(states))

        unfrozen_stub = TinyBaseStub()
        with self.assertRaises(ValueError):
            extract_router_embedding(unfrozen_stub, FakeTokenizer(), _sample_task("WT05"))

        frozen_stub = TinyBaseStub().eval().requires_grad_(False)
        vec = extract_router_embedding(
            frozen_stub,
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
