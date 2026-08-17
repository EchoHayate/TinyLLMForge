from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def _load_gate():
    path = (
        Path(__file__).resolve().parent
        / "qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_verify_focused_h2d_slot_reuse_gate",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load focused H2D gate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError(
            f"invalid JSON artifact: {path.name}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(
            f"JSON artifact must be an object: {path.name}"
        )
    return value


def checkpoint_manifest_sha256(model_path: str | Path) -> str:
    root = Path(model_path)
    if not root.is_dir():
        raise ValueError(
            "model path must be a checkpoint directory"
        )
    manifest = root.parent / "model_manifest.json"
    if not manifest.is_file():
        raise ValueError(
            "approved target model manifest is missing"
        )
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def verify_run(
    *,
    run_dir,
    repo_root,
    model_path,
    source_digest_fn=gate.source_tree_sha256,
    checkpoint_digest_fn=checkpoint_manifest_sha256,
) -> dict:
    try:
        artifact = gate.validate_artifact(
            _read_json(Path(run_dir) / "artifact.json")
        )
        source_digest = source_digest_fn(Path(repo_root))
        checkpoint_digest = checkpoint_digest_fn(model_path)
        for cell_key in gate.REQUIRED_CELL_KEYS:
            for repetition in artifact["cells"][cell_key]:
                if (
                    repetition["source_tree_sha256"]
                    != source_digest
                ):
                    raise ValueError(
                        "source tree identity mismatch"
                    )
                if (
                    repetition["checkpoint_sha256"]
                    != checkpoint_digest
                ):
                    raise ValueError(
                        "checkpoint identity mismatch"
                    )
        decision = gate.evaluate_campaign(artifact)
        return {
            "classification": "PASS",
            "failures": [],
            "decision": decision,
        }
    except Exception as error:
        return {
            "classification": "FAIL",
            "failures": [str(error)],
            "decision": None,
        }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--model", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = verify_run(
        run_dir=Path(args.run_dir),
        repo_root=Path(args.repo_root),
        model_path=Path(args.model),
    )
    sys.stdout.write(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return 0 if result["classification"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
