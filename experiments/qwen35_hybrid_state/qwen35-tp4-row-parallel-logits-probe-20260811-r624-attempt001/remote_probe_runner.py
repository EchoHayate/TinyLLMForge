from __future__ import annotations

from pathlib import Path
import sys


def main():
    source_root = sys.argv[1]
    output_path = sys.argv[2]
    base_runner = Path(sys.argv[3])
    overlay_tools = Path(__file__).resolve().parent / "tools"
    sys.path.insert(0, str(overlay_tools))
    text = base_runner.read_text(encoding="utf-8")
    replacements = {
        "20260811-r622-attempt001": "20260811-r624-attempt001",
        "dist_port=16321": "dist_port=16341",
        "master_port=16322": "master_port=16342",
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise RuntimeError(f"base runner marker mismatch: {old}")
        text = text.replace(old, new)
    sys.argv = [str(base_runner), source_root, output_path]
    namespace = {
        "__file__": str(base_runner),
        "__name__": "__main__",
    }
    exec(compile(text, str(base_runner), "exec"), namespace)


if __name__ == "__main__":
    main()
