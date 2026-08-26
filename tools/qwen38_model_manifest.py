from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import re


MODEL_SCHEMA = "tinyllmforge.qwen38-model-manifest.v1"
REPOSITORY = "Qwen/Qwen3.8-27B"
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
_TOKENIZER_NAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
}


def canonical_bytes(payload):
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )


def manifest_sha256(payload):
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path, label):
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _validate_identity(repository, revision):
    if repository != REPOSITORY:
        raise ValueError(
            f"Qwen3.8 repository must be {REPOSITORY}"
        )
    if (
        not isinstance(revision, str)
        or _IMMUTABLE_REVISION.fullmatch(revision) is None
    ):
        raise ValueError(
            "Qwen3.8 requires an immutable revision SHA"
        )


def _regular_file_inventory(model_root):
    rows = {}
    for path in sorted(model_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(
                f"model inventory contains symlink: "
                f"{path.relative_to(model_root)}"
            )
        if not path.is_file():
            continue
        relative = path.relative_to(model_root).as_posix()
        if relative.startswith("../") or relative.startswith("/"):
            raise ValueError("model inventory path escapes model root")
        rows[relative] = {
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }
    return rows


def _checkpoint_shards(model_root, index_payload):
    weight_map = index_payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint index weight_map is incomplete")
    if any(
        not isinstance(name, str) or not name
        for name in weight_map
    ):
        raise ValueError("checkpoint tensor name is invalid")
    if "lm_head.weight" not in weight_map:
        raise ValueError(
            "checkpoint index is missing lm_head.weight"
        )
    listed = sorted(set(weight_map.values()))
    if (
        any(
            not isinstance(name, str)
            or not name.endswith(".safetensors")
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            for name in listed
        )
    ):
        raise ValueError("checkpoint shard name is invalid")
    existing = sorted(
        path.relative_to(model_root).as_posix()
        for path in model_root.rglob("*.safetensors")
        if path.is_file() and not path.is_symlink()
    )
    missing = sorted(set(listed) - set(existing))
    if missing:
        raise ValueError(
            f"checkpoint shard is missing: {missing[0]}"
        )
    extra = sorted(set(existing) - set(listed))
    if extra:
        raise ValueError(
            f"unlisted checkpoint shard: {extra[0]}"
        )
    return listed


def _validate_config_identity(config):
    if config.get("model_type") != "qwen3_5":
        raise ValueError("config model_type mismatch")
    if config.get("architectures") != [
        "Qwen3_5ForConditionalGeneration"
    ]:
        raise ValueError("config architecture mismatch")
    if config.get("language_model_only") is not False:
        raise ValueError("config language_model_only mismatch")
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("config text_config is missing")
    expected_layers = [
        "full_attention" if (index + 1) % 4 == 0
        else "linear_attention"
        for index in range(64)
    ]
    expected = {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "dtype": "bfloat16",
        "vocab_size": 248320,
        "tie_word_embeddings": False,
        "layer_types": expected_layers,
    }
    for name, value in expected.items():
        if text_config.get(name) != value:
            raise ValueError(
                f"config text topology mismatch: {name}"
            )
    return text_config


def build_model_manifest(model_root, *, repository, revision):
    _validate_identity(repository, revision)
    model_root = Path(model_root).resolve()
    if not model_root.is_dir():
        raise ValueError("model root must be an existing directory")
    config_path = model_root / "config.json"
    index_path = model_root / "model.safetensors.index.json"
    if not config_path.is_file() or config_path.is_symlink():
        raise ValueError("config.json must be a regular file")
    if not index_path.is_file() or index_path.is_symlink():
        raise ValueError(
            "model.safetensors.index.json must be a regular file"
        )
    config = _load_json(config_path, "config.json")
    text_config = _validate_config_identity(config)
    index_payload = _load_json(
        index_path,
        "model.safetensors.index.json",
    )
    checkpoint_shards = _checkpoint_shards(
        model_root,
        index_payload,
    )
    files = _regular_file_inventory(model_root)
    tokenizer_files = sorted(
        name for name in files if Path(name).name in _TOKENIZER_NAMES
    )
    if not tokenizer_files:
        raise ValueError("tokenizer inventory is empty")
    return {
        "schema_version": MODEL_SCHEMA,
        "repository": repository,
        "resolved_revision": revision,
        "model_root": str(model_root),
        "config_sha256": files["config.json"]["sha256"],
        "text_config_sha256": hashlib.sha256(
            canonical_bytes(text_config)
        ).hexdigest(),
        "tokenizer_inventory_sha256": hashlib.sha256(
            canonical_bytes({
                name: files[name]
                for name in tokenizer_files
            })
        ).hexdigest(),
        "tokenizer_files": tokenizer_files,
        "checkpoint_index": "model.safetensors.index.json",
        "checkpoint_index_sha256": files[
            "model.safetensors.index.json"
        ]["sha256"],
        "checkpoint_tensor_count": len(
            index_payload["weight_map"]
        ),
        "checkpoint_shards": checkpoint_shards,
        "files": files,
    }


def verify_model_manifest(path, *, model_root=None):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("model manifest must be a regular file")
    payload = _load_json(path, "model manifest")
    if payload.get("schema_version") != MODEL_SCHEMA:
        raise ValueError("model manifest schema mismatch")
    repository = payload.get("repository")
    revision = payload.get("resolved_revision")
    _validate_identity(repository, revision)
    root = Path(model_root or payload.get("model_root", "")).resolve()
    manifest_root = payload.get("model_root")
    if (
        not isinstance(manifest_root, str)
        or manifest_root != str(root)
    ):
        raise ValueError("model manifest model root mismatch")
    rebuilt = build_model_manifest(
        root,
        repository=repository,
        revision=revision,
    )
    expected_files = payload.get("files")
    if not isinstance(expected_files, dict):
        raise ValueError("model manifest files are invalid")
    for name in expected_files:
        if not isinstance(name, str) or not name:
            raise ValueError("model manifest path must be relative")
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or name != str(relative)
        ):
            raise ValueError("model manifest path must be relative")
    checkpoint_shards = payload.get("checkpoint_shards")
    if (
        not isinstance(checkpoint_shards, list)
        or len(checkpoint_shards) != len(set(checkpoint_shards))
    ):
        raise ValueError(
            "checkpoint shard inventory contains duplicate paths"
        )
    for name, expected in expected_files.items():
        observed = rebuilt["files"].get(name)
        if observed is None:
            raise ValueError(f"model file is missing: {name}")
        if observed.get("size") != expected.get("size"):
            raise ValueError(f"model file size mismatch: {name}")
        if observed.get("sha256") != expected.get("sha256"):
            raise ValueError(f"model file hash mismatch: {name}")
    if set(rebuilt["files"]) != set(expected_files):
        raise ValueError("model file inventory mismatch")
    if rebuilt != payload:
        raise ValueError("model manifest semantic mismatch")
    return payload


def _atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise ValueError("temporary manifest path already exists")
    temporary.write_bytes(canonical_bytes(payload))
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--repository", default=REPOSITORY)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        payload = verify_model_manifest(
            args.output,
            model_root=args.model_root,
        )
    else:
        payload = build_model_manifest(
            args.model_root,
            repository=args.repository,
            revision=args.revision,
        )
        _atomic_write(args.output, payload)
        verify_model_manifest(
            args.output,
            model_root=args.model_root,
        )
    print(json.dumps({
        "classification": "PASS",
        "manifest_sha256": manifest_sha256(payload),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
