from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TOOLS = Path(__file__).resolve().parent
gate = _load_module(
    "qwen35_native_mtp_tp4_32k_target_kv_offload_gate",
    _TOOLS
    / "qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py",
)
_frozen_worker = _load_module(
    "_qwen35_native_mtp_tp4_32k_frozen_worker",
    _TOOLS
    / "qwen35_native_mtp_tp4_16k_target_kv_offload_worker.py",
)
_frozen_worker.gate = gate
_frozen_worker._frozen_worker.gate = gate
_frozen_worker.tp1_worker.gate = gate

for _name, _value in vars(_frozen_worker).items():
    if not _name.startswith("__") and _name != "gate":
        globals()[_name] = _value


if __name__ == "__main__":
    sys.exit(main())
