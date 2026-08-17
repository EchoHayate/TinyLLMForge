from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_tools = Path(__file__).resolve().parent
gate = _load_module(
    "qwen35_generic_speculative_tp4_16k_gate",
    _tools / "qwen35_generic_speculative_tp4_16k_gate.py",
)
_frozen_verifier = _load_module(
    "_verify_qwen35_generic_speculative_tp4_frozen_gate",
    _tools / "verify_qwen35_generic_speculative_tp4_gate.py",
)
_frozen_verifier.gate = gate

for _name, _value in vars(_frozen_verifier).items():
    if not _name.startswith("__") and _name != "gate":
        globals()[_name] = _value


if __name__ == "__main__":
    sys.exit(main())
