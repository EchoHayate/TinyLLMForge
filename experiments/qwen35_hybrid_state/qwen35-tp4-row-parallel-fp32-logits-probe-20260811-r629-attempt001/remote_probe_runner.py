from __future__ import annotations

from pathlib import Path
import re
import sys


def main():
    source_root = sys.argv[1]
    output_path = sys.argv[2]
    base_runner = Path(sys.argv[3])
    overlay_tools = Path(__file__).resolve().parent / "tools"
    sys.path.insert(0, str(overlay_tools))
    text = base_runner.read_text(encoding="utf-8")
    replacements = {
        "20260811-r622-attempt001": "20260811-r629-attempt001",
        "dist_port=16321": "dist_port=16351",
        "master_port=16322": "master_port=16352",
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise RuntimeError(f"base runner marker mismatch: {old}")
        text = text.replace(old, new)
    pattern = re.compile(
        r"source_tree_sha256=\(\n"
        r"\s+\"[0-9a-f]+\"\n"
        r"\s+\"[0-9a-f]+\"\n"
        r"\s+\),"
    )
    text, count = pattern.subn(
        'source_tree_sha256="'
        '7a80d62c2c9e71f7899dc397f8104272'
        '86b330d95da2b63f67839aa98d47b3b3",',
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("source hash marker mismatch")
    sys.argv = [str(base_runner), source_root, output_path]
    namespace = {
        "__file__": str(base_runner),
        "__name__": "__main__",
    }
    exec(compile(text, str(base_runner), "exec"), namespace)


if __name__ == "__main__":
    main()
