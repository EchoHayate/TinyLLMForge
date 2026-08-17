from pathlib import Path
import json
import tempfile
import unittest

import qwen35_decode_row_parallel_comparison as comparison


POLICIES = ("recompute", "exact_restore")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _summary(
    *,
    steady_wall_ns: int,
    steady_cuda_ns: int,
    collective_cuda_ns: int,
    steady_wall_p90_ns: int,
    steady_cuda_p90_ns: int,
) -> dict:
    by_policy = {}
    for policy in POLICIES:
        by_policy[policy] = {
            "median_steady_wall_ns": steady_wall_ns,
            "median_steady_cuda_ns": steady_cuda_ns,
            "median_collective_cuda_ns": collective_cuda_ns,
            "median_steady_wall_p90_ns": steady_wall_p90_ns,
            "median_steady_cuda_p90_ns": steady_cuda_p90_ns,
        }
    return {
        "schema_version": "qwen35.tp4-decode-internal-summary.v1",
        "generated_tokens": 8,
        "measured_pairs": 5,
        "by_policy": by_policy,
    }


def _write_attempt(
    root: Path,
    *,
    operation: str,
    output_token_ids: tuple[int, ...],
    steady_wall_ns: int,
    steady_cuda_ns: int,
    collective_cuda_ns: int,
    steady_wall_p90_ns: int,
    steady_cuda_p90_ns: int,
) -> None:
    _write_json(
        root / "decode_summary.json",
        _summary(
            steady_wall_ns=steady_wall_ns,
            steady_cuda_ns=steady_cuda_ns,
            collective_cuda_ns=collective_cuda_ns,
            steady_wall_p90_ns=steady_wall_p90_ns,
            steady_cuda_p90_ns=steady_cuda_p90_ns,
        ),
    )
    for repetition in range(5):
        for policy in POLICIES:
            case_id = (
                f"w2_long_reuse__measured__r{repetition}__{policy}"
            )
            case_root = root / "download" / "cases" / case_id
            ranks = []
            for rank in range(4):
                ranks.append({
                    "rank": rank,
                    "steps": [{
                        "rank": rank,
                        "step_index": 1,
                        "decode_ordinal": 0,
                        "is_decode": True,
                        "wall_ns": steady_wall_ns,
                        "cuda_ns": steady_cuda_ns,
                    }],
                    "collectives": [{
                        "rank": rank,
                        "step_index": 1,
                        "decode_ordinal": 0,
                        "operation": operation,
                        "wall_ns": collective_cuda_ns,
                        "cuda_ns": collective_cuda_ns,
                    }],
                })
            _write_json(
                case_root / "decode_profile.json",
                {
                    "schema_version": (
                        "qwen35.tp4-decode-internal-case.v1"
                    ),
                    "case_id": case_id,
                    "phase": "measured",
                    "policy": policy,
                    "repetition": repetition,
                    "generated_tokens": 8,
                    "ranks": ranks,
                },
            )
            row = {
                "case_id": case_id,
                "phase": "measured",
                "policy": policy,
                "repetition": repetition,
                "request_id": "request-0",
                "output_token_ids": list(output_token_ids),
            }
            (case_root / "case_rows.jsonl").write_text(
                json.dumps(row, sort_keys=True) + "\n",
                encoding="utf-8",
            )


class DecodeRowParallelComparisonTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.baseline = self.root / "baseline"
        self.candidate = self.root / "candidate"

    def tearDown(self):
        self.temporary.cleanup()

    def _write_baseline(self):
        _write_attempt(
            self.baseline,
            operation=(
                "replicated_weight_row_parallel_all_gather"
            ),
            output_token_ids=(1, 2, 3),
            steady_wall_ns=100,
            steady_cuda_ns=90,
            collective_cuda_ns=60,
            steady_wall_p90_ns=120,
            steady_cuda_p90_ns=110,
        )

    def test_classifies_repeated_speedup_as_performance_pass(self):
        self._write_baseline()
        _write_attempt(
            self.candidate,
            operation="row_parallel_all_reduce",
            output_token_ids=(1, 2, 3),
            steady_wall_ns=90,
            steady_cuda_ns=82,
            collective_cuda_ns=52,
            steady_wall_p90_ns=116,
            steady_cuda_p90_ns=106,
        )

        result = comparison.compare_decode_attempts(
            self.baseline,
            self.candidate,
        )

        self.assertEqual(result["classification"], "PERFORMANCE_PASS")
        self.assertEqual(
            result["legacy_all_gather_rows"]["candidate"],
            0,
        )
        self.assertGreater(
            result["row_parallel_all_reduce_rows"]["candidate"],
            0,
        )
        self.assertTrue(result["output_parity"])

    def test_classifies_small_speedup_as_structural_only(self):
        self._write_baseline()
        _write_attempt(
            self.candidate,
            operation="row_parallel_all_reduce",
            output_token_ids=(1, 2, 3),
            steady_wall_ns=98,
            steady_cuda_ns=89,
            collective_cuda_ns=58,
            steady_wall_p90_ns=120,
            steady_cuda_p90_ns=110,
        )

        result = comparison.compare_decode_attempts(
            self.baseline,
            self.candidate,
        )

        self.assertEqual(result["classification"], "STRUCTURAL_ONLY")

    def test_rejects_candidate_with_legacy_all_gather(self):
        self._write_baseline()
        _write_attempt(
            self.candidate,
            operation=(
                "replicated_weight_row_parallel_all_gather"
            ),
            output_token_ids=(1, 2, 3),
            steady_wall_ns=80,
            steady_cuda_ns=75,
            collective_cuda_ns=45,
            steady_wall_p90_ns=100,
            steady_cuda_p90_ns=95,
        )

        result = comparison.compare_decode_attempts(
            self.baseline,
            self.candidate,
        )

        self.assertEqual(result["classification"], "NO_GO")
        self.assertIn("legacy AllGather", " ".join(result["reasons"]))

    def test_rejects_output_token_mismatch(self):
        self._write_baseline()
        _write_attempt(
            self.candidate,
            operation="row_parallel_all_reduce",
            output_token_ids=(1, 2, 4),
            steady_wall_ns=80,
            steady_cuda_ns=75,
            collective_cuda_ns=45,
            steady_wall_p90_ns=100,
            steady_cuda_p90_ns=95,
        )

        result = comparison.compare_decode_attempts(
            self.baseline,
            self.candidate,
        )

        self.assertEqual(result["classification"], "NO_GO")
        self.assertFalse(result["output_parity"])


if __name__ == "__main__":
    unittest.main()
