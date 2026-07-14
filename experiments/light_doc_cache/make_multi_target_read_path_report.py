"""Aggregate Light Doc Cache multi-target read-path artifacts."""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_CATEGORIES = {
    "short_factual",
    "long_document_qa",
    "source_code",
    "mathematical_reasoning",
    "structured_text",
    "repetitive_text",
    "cross_paragraph_dependency",
    "out_of_distribution",
}
LENGTH_BUCKETS = {"short", "medium", "long"}


def load_target_dataset(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    targets = validate_target_dataset(payload)
    return {"version": int(payload["version"]), "targets": targets}


def validate_target_dataset(payload: dict[str, object]) -> list[dict[str, str]]:
    if int(payload.get("version", 0)) != 1:
        raise ValueError("target dataset version must be 1")
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or len(raw_targets) != 8:
        raise ValueError("target dataset must contain exactly eight targets")
    targets: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    category_counts = {category: 0 for category in REQUIRED_CATEGORIES}
    bucket_counts = {bucket: 0 for bucket in LENGTH_BUCKETS}
    for index, raw in enumerate(raw_targets):
        if not isinstance(raw, dict):
            raise ValueError(f"target {index} must be an object")
        target = {
            "id": str(raw.get("id", "")).strip(),
            "category": str(raw.get("category", "")).strip(),
            "length_bucket": str(raw.get("length_bucket", "")).strip(),
            "prompt": str(raw.get("prompt", "")).strip(),
        }
        if not target["id"] or not target["prompt"]:
            raise ValueError(f"target {index} requires non-empty id and prompt")
        if target["id"] in seen_ids:
            raise ValueError(f"duplicate target id: {target['id']}")
        if target["category"] not in REQUIRED_CATEGORIES:
            raise ValueError(f"unknown target category: {target['category']}")
        if target["length_bucket"] not in LENGTH_BUCKETS:
            raise ValueError(f"unknown length bucket: {target['length_bucket']}")
        seen_ids.add(target["id"])
        category_counts[target["category"]] += 1
        bucket_counts[target["length_bucket"]] += 1
        targets.append(target)
    if {category for category, count in category_counts.items() if count} != REQUIRED_CATEGORIES:
        raise ValueError("target dataset must cover every required category exactly once")
    if any(count < 2 for count in bucket_counts.values()):
        raise ValueError("target dataset requires at least two targets in each length bucket")
    return targets
