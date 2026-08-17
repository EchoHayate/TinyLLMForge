from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parent
    / "qwen35_tp4_correctness_resource_policy.py"
)
SPEC = importlib.util.spec_from_file_location(
    "qwen35_tp4_correctness_resource_policy_tested",
    MODULE_PATH,
)
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)


SSH_TARGET = "sitian@10.232.195.203"
GPU_INDICES = [2, 4, 5, 6]
GPU_UUIDS = [
    "GPU-63c05907-407b-8240-07a0-f38872840867",
    "GPU-56b882d2-6e6e-adb3-80e7-95f0a9e678f1",
    "GPU-687b7858-ca44-98ad-cfba-b6785eaf05e8",
    "GPU-c27f6fd6-8a66-7935-41fd-bd5ccdaced31",
]


def _process(pid, name, start_time_ticks, used_memory_mib=436):
    return {
        "pid": pid,
        "process_name": name,
        "used_memory_mib": used_memory_mib,
        "start_time_ticks": start_time_ticks,
    }


def _selected(*, include_second=True):
    rows = []
    for offset, (index, uuid) in enumerate(
        zip(GPU_INDICES, GPU_UUIDS)
    ):
        processes = [
            _process(
                330291 + offset * 2,
                "python3",
                100000 + offset,
            )
        ]
        if include_second and index == 2:
            processes.append(
                _process(
                    1097889,
                    "inferencer_worker_0_0",
                    200000,
                    11308,
                )
            )
        rows.append({
            "gpu_index": index,
            "gpu_uuid": uuid,
            "free_bytes": (64 + offset) * 1024**3,
            "compute_processes": processes,
        })
    return rows


def _baseline():
    return {
        "schema_version": policy.BASELINE_SCHEMA_VERSION,
        "classification": "READY",
        "ssh_target": SSH_TARGET,
        "captured_at": "2026-07-29T15:00:00+08:00",
        "gpu_indices": list(GPU_INDICES),
        "selected": _selected(),
        "minimum_free_bytes_per_gpu": policy.MIN_GPU_FREE_BYTES,
        "benchmark_execution_authorized": False,
    }


def _guard_payload(selected=None):
    return {
        "classification": "READY",
        "resource_policy": policy.CONTROLLED_SHARED,
        "baseline_sha256": "a" * 64,
        "selected": _selected() if selected is None else selected,
        "benchmark_execution_authorized": False,
    }


class CorrectnessResourcePolicyTest(unittest.TestCase):
    def test_validate_baseline_manifest_accepts_regular_canonical_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resource_baseline.json"
            path.write_text(
                json.dumps(
                    _baseline(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            result = policy.validate_baseline_manifest(
                path,
                ssh_target=SSH_TARGET,
                gpu_indices=GPU_INDICES,
            )
            self.assertEqual(result["gpu_indices"], GPU_INDICES)
            self.assertEqual(
                [row["gpu_uuid"] for row in result["selected"]],
                GPU_UUIDS,
            )

    def test_validate_baseline_manifest_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            target.write_text(
                json.dumps(_baseline()),
                encoding="utf-8",
            )
            link = root / "resource_baseline.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "regular file"):
                policy.validate_baseline_manifest(
                    link,
                    ssh_target=SSH_TARGET,
                    gpu_indices=GPU_INDICES,
                )

    def test_validate_guard_accepts_disappearing_baseline_process(self):
        baseline = _baseline()
        selected = _selected()
        selected[0]["compute_processes"] = selected[0][
            "compute_processes"
        ][:1]
        result = policy.validate_guard_payload(
            policy.CONTROLLED_SHARED,
            _guard_payload(selected),
            gpu_indices=GPU_INDICES,
            baseline=baseline,
            baseline_sha256="a" * 64,
        )
        self.assertEqual(result, GPU_INDICES)

    def test_validate_guard_rejects_new_process(self):
        selected = _selected()
        selected[1]["compute_processes"].append(
            _process(999999, "python", 300000)
        )
        with self.assertRaisesRegex(ValueError, "process drift"):
            policy.validate_guard_payload(
                policy.CONTROLLED_SHARED,
                _guard_payload(selected),
                gpu_indices=GPU_INDICES,
                baseline=_baseline(),
                baseline_sha256="a" * 64,
            )

    def test_validate_guard_rejects_pid_reuse(self):
        selected = _selected()
        selected[0]["compute_processes"][0][
            "start_time_ticks"
        ] += 1
        with self.assertRaisesRegex(ValueError, "process drift"):
            policy.validate_guard_payload(
                policy.CONTROLLED_SHARED,
                _guard_payload(selected),
                gpu_indices=GPU_INDICES,
                baseline=_baseline(),
                baseline_sha256="a" * 64,
            )

    def test_validate_guard_rejects_uuid_drift(self):
        selected = _selected()
        selected[2]["gpu_uuid"] = "GPU-drift"
        with self.assertRaisesRegex(ValueError, "GPU drift"):
            policy.validate_guard_payload(
                policy.CONTROLLED_SHARED,
                _guard_payload(selected),
                gpu_indices=GPU_INDICES,
                baseline=_baseline(),
                baseline_sha256="a" * 64,
            )

    def test_validate_guard_rejects_low_free_memory(self):
        selected = _selected()
        selected[3]["free_bytes"] = policy.MIN_GPU_FREE_BYTES - 1
        with self.assertRaisesRegex(ValueError, "free memory"):
            policy.validate_guard_payload(
                policy.CONTROLLED_SHARED,
                _guard_payload(selected),
                gpu_indices=GPU_INDICES,
                baseline=_baseline(),
                baseline_sha256="a" * 64,
            )

    def test_validate_guard_rejects_baseline_sha_drift(self):
        payload = _guard_payload()
        payload["baseline_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "baseline"):
            policy.validate_guard_payload(
                policy.CONTROLLED_SHARED,
                payload,
                gpu_indices=GPU_INDICES,
                baseline=_baseline(),
                baseline_sha256="a" * 64,
            )

    def test_validate_strict_guard_preserves_zero_process_rule(self):
        payload = {
            "classification": "READY",
            "selected": [
                {
                    **row,
                    "compute_processes": [],
                }
                for row in _selected()
            ],
        }
        result = policy.validate_guard_payload(
            policy.STRICT_EXCLUSIVE,
            payload,
            gpu_indices=GPU_INDICES,
        )
        self.assertEqual(result, GPU_INDICES)
        payload["selected"][0]["compute_processes"] = [
            _process(1, "python", 1)
        ]
        with self.assertRaisesRegex(ValueError, "strict"):
            policy.validate_guard_payload(
                policy.STRICT_EXCLUSIVE,
                payload,
                gpu_indices=GPU_INDICES,
            )

    def test_commands_are_read_only_and_policy_bound(self):
        capture = policy.capture_command(GPU_INDICES)
        guard = policy.guard_command(
            policy.CONTROLLED_SHARED,
            GPU_INDICES,
            baseline_path="/tmp/resource_baseline.json",
            baseline_sha256="a" * 64,
        )
        text = "\n".join(capture + guard)
        self.assertIn("nvidia-smi", text)
        self.assertIn("/proc", text)
        self.assertIn("controlled_shared", text)
        for forbidden in ("kill ", "pkill", "renice", "nvidia-smi -c"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
