from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.assemble_quantized_draft_int4_microgate import assemble_bundle
from tools.test_assemble_quantized_draft_int4_microgate import (
    RUN_TAG,
    SOURCE_REVISION,
    make_raw_evidence,
)
from tools.verify_quantized_draft_int4_microgate import verify_bundle


def _bundle(tmp_path):
    output = tmp_path / "bundle"
    assemble_bundle(
        raw_dir=make_raw_evidence(tmp_path),
        output_dir=output,
        source_revision=SOURCE_REVISION,
        run_tag=RUN_TAG,
    )
    return output


def _rewrite_manifest(bundle: Path) -> None:
    lines = []
    for path in sorted(bundle.iterdir()):
        if path.name == "manifest.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    (bundle / "manifest.sha256").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def test_verifier_recomputes_metrics_and_classification(tmp_path):
    bundle = _bundle(tmp_path)

    result = verify_bundle(bundle)

    assert result["status"] == "PASS"
    assert result["classification"] == "GO_FUSED_INT4_DRAFT_KERNEL"
    assert result["source_revision"] == SOURCE_REVISION
    assert result["run_tag"] == RUN_TAG
    assert result["shape_summaries"][0]["pair_count"] == 200
    assert result["weighted_summary"][
        "candidate_to_bf16_p99_ratio"
    ] < 0.95


@pytest.mark.parametrize(
    "mutation",
    ("added", "removed", "symlink", "one_byte"),
)
def test_verifier_rejects_inventory_or_content_mutation(
    tmp_path,
    mutation,
):
    bundle = _bundle(tmp_path)
    if mutation == "added":
        (bundle / "extra.json").write_text("{}\n", encoding="utf-8")
    elif mutation == "removed":
        (bundle / "memory.json").unlink()
    elif mutation == "symlink":
        target = tmp_path / "outside.json"
        target.write_text("{}\n", encoding="utf-8")
        (bundle / "memory.json").unlink()
        (bundle / "memory.json").symlink_to(target)
    elif mutation == "one_byte":
        path = bundle / "environment.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        raise AssertionError(mutation)

    with pytest.raises(ValueError):
        verify_bundle(bundle)


def test_verifier_rejects_identity_drift_even_with_refreshed_manifest(
    tmp_path,
):
    bundle = _bundle(tmp_path)
    classification_path = bundle / "classification.json"
    payload = json.loads(classification_path.read_text(encoding="utf-8"))
    payload["source_revision"] = "d" * 40
    classification_path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rewrite_manifest(bundle)

    with pytest.raises(ValueError, match="identity"):
        verify_bundle(bundle)
