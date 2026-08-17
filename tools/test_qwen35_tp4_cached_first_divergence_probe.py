from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name, filename):
    path = TOOLS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load(
    "qwen35_tp4_cached_contract_for_first_divergence_probe_test",
    "qwen35_tp4_cached_continuation_correctness_contract.py",
)
engine_executor = _load(
    "qwen35_tp4_engine_executor_for_first_divergence_probe_test",
    "qwen35_tp4_engine_correctness_executor.py",
)
probe = _load(
    "qwen35_tp4_cached_first_divergence_probe",
    "qwen35_tp4_cached_first_divergence_probe.py",
)


def _configuration():
    return engine_executor.ExecutorConfiguration(
        model_dir="/models/qwen35",
        model_manifest_path="/authority/model_manifest.json",
        model_manifest_sha256="a" * 64,
        source_tree_sha256="b" * 64,
        workload_manifest_path="/authority/workload_manifest.json",
        workload_manifest_sha256=contract.WORKLOAD_MANIFEST_SHA256,
        model_fingerprint="qwen35-m8-authority",
        gpu_indices=(0, 1, 2, 3),
        dist_port=31001,
        master_port=31002,
        max_cache_entries=32,
        max_cache_bytes=1 << 30,
        timeout_s=600.0,
    )


class FakeLogits:
    def __init__(self, value):
        self.value = float(value)

    def clone(self):
        return FakeLogits(self.value)


class FakeEngine:
    def __init__(self):
        self.model_runner = SimpleNamespace(world_size=4, rank=0)
        self.ps = [SimpleNamespace() for _ in range(3)]
        self.calls = []
        self.steps = []
        self.phase = None
        self.request_calls = 0
        self.last_logits = None
        self.last_step_observation = None
        self.snapshot = {
            "current_entries": 0,
            "hits": 0,
            "misses": 0,
            "publication_commits": 0,
            "invalidations": 0,
            "clears": 0,
            "last_publication_block_identities": [],
        }

    def configure_qwen35_hybrid_prefix_publication_runtime(self, **kwargs):
        self.calls.append(("configure", kwargs))

    def qwen35_hybrid_prefix_authority_snapshots(self, *, timeout_s):
        return tuple(
            {"rank": rank, **self.snapshot}
            for rank in range(4)
        )

    def clear_qwen35_hybrid_prefix_caches(self, *, timeout_s):
        self.calls.append(("clear",))
        self.snapshot["current_entries"] = 0
        self.snapshot["clears"] += 1
        return tuple(
            {"rank": rank, "cleared_entries": 1}
            for rank in range(4)
        )

    def add_request(self, prompt, sampling_params):
        self.calls.append(("add", list(prompt), sampling_params.max_tokens))
        self.request_calls += 1
        if self.request_calls == 2:
            self.phase = "source"
            self.steps = [(1, [17])]
            return
        if self.request_calls == 3:
            self.phase = "restore"
            self.steps = [(3, [197])]
            return
        self.phase = "recompute"
        self.steps = [(11, [197])]

    def is_finished(self):
        return not self.steps

    def step(self):
        num_tokens, output = self.steps.pop(0)
        if self.phase == "source":
            self.snapshot.update({
                "current_entries": 1,
                "publication_commits": (
                    self.snapshot["publication_commits"] + 1
                ),
                "last_publication_block_identities": [[7, 2, 99]],
            })
        elif self.phase == "restore":
            self.snapshot["hits"] += 1
            self.last_logits = FakeLogits(2.0)
        else:
            self.snapshot["misses"] += 1
            self.last_logits = FakeLogits(1.0)
        self.last_step_observation = {
            "do_sample": True,
            "new_completion_tokens_by_seq": {17: [output[-1]]},
        }
        return [(17, output)], num_tokens

    def enable_step_logits_authority_recording(
        self,
        enabled,
        *,
        timeout_s,
    ):
        self.calls.append(("record", bool(enabled)))
        if enabled:
            self.last_logits = None
        return {
            "enabled": bool(enabled),
            "rank_inventory": [0, 1, 2, 3],
        }

    def read_step_logits_authority(self):
        return self.last_logits.clone()

    def exit(self):
        self.calls.append(("exit",))
        return {
            "process_group_destroyed": True,
            "rank_exit_codes": [0, 0, 0, 0],
            "owned_children_remaining": [],
            "rank_cleanup_receipts": [
                {"rank": rank, "process_group_destroyed": True}
                for rank in range(4)
            ],
        }


def _reference_executor_factory():
    class Executor:
        def generate_reference_with_step_logits(self, **kwargs):
            return {
                "output_token_ids": [197],
                "step_logits": [FakeLogits(1.0)],
            }

        def close(self):
            pass

    return Executor()


def _compare(engine_logits, reference_logits, *, atol):
    difference = abs(
        engine_logits[0].value - reference_logits[0].value
    )
    return {
        "max_abs_diff": difference,
        "allclose": difference <= atol,
        "first_mismatch_step": 0 if difference > atol else None,
        "per_step_max_abs_diff": [difference],
        "first_mismatch_engine_argmax": None,
        "first_mismatch_reference_argmax": None,
    }


def test_probe_separates_engine_recompute_from_restore():
    engine = FakeEngine()
    result = probe.run_probe(
        configuration=_configuration(),
        engine_factory=lambda configuration: engine,
        reference_executor_factory=_reference_executor_factory,
        logits_comparator=_compare,
        workload="w1_medium_reuse",
        request_index=0,
        generated_tokens=1,
    )

    assert result["classification"] == "RESTORE_ONLY_DIVERGENCE"
    assert result["official_output_token_ids"] == [197]
    assert result["recompute"]["output_token_ids"] == [197]
    assert result["recompute"]["executed_prefill_tokens"] == 11
    assert result["recompute"]["restore_hits"] == 0
    assert result["recompute"]["restore_misses"] == 1
    assert result["recompute"]["comparison"]["allclose"] is True
    assert result["restore"]["output_token_ids"] == [197]
    assert result["restore"]["executed_prefill_tokens"] == 3
    assert result["restore"]["restore_hits"] == 1
    assert result["restore"]["restore_misses"] == 0
    assert result["restore"]["comparison"]["allclose"] is False
    assert result["restore"]["comparison"]["max_abs_diff"] == 1.0
    assert [call[0] for call in engine.calls] == [
        "configure",
        "clear",
        "record",
        "add",
        "record",
        "add",
        "record",
        "add",
        "record",
        "exit",
    ]


def _run():
    test_probe_separates_engine_recompute_from_restore()
    print("qwen35 TP4 cached first-divergence probe tests passed")


if __name__ == "__main__":
    _run()
