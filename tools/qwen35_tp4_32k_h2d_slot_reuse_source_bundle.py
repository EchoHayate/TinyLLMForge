from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


SCHEMA = "qwen35.tp4-32k-h2d-source-bundle.v1"
SOURCE_INVENTORY_NAME = "source_inventory.json"
SOURCE_TAR_NAME = "source.tar"
ENTRYPOINTS = (
    "tools/qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic_gate.py",
    "tools/qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic_worker.py",
    "tools/verify_qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic.py",
    "tools/qwen35_native_mtp_tp4_32k_target_kv_offload_worker.py",
    "tools/qwen35_tp4_32k_h2d_slot_reuse_source_bundle.py",
    "tools/qwen35_tp4_32k_h2d_slot_reuse_campaign.py",
    "tools/qwen35_tp4_32k_h2d_slot_reuse_campaign_authorization.py",
    "tools/qwen35_tp4_32k_h2d_slot_reuse_campaign_executor.py",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or path.as_posix() != value
    ):
        raise ValueError("source path is unsafe")
    return value


def _require_regular_source(repo_root: Path, relative: str) -> Path:
    _safe_relative_path(relative)
    path = repo_root / relative
    if path.is_symlink():
        raise ValueError(f"source file is a symlink: {relative}")
    if not path.is_file():
        raise ValueError(f"source file is missing: {relative}")
    return path


def _local_import_candidates(tree: ast.AST) -> set[str]:
    candidates: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        else:
            names = []
        for name in names:
            candidates.add(f"tools/{name.replace('.', '/')}.py")
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.endswith(".py")
        ):
            literal = node.value
            if "/" in literal:
                candidates.add(literal.removeprefix("./"))
            else:
                candidates.add(f"tools/{literal}")
    return candidates


def collect_source_files(repo_root: str | Path) -> tuple[str, ...]:
    root = Path(repo_root).resolve()
    tinyvllm = root / "tinyvllm"
    if not tinyvllm.is_dir():
        raise ValueError("tinyvllm source directory is missing")
    owned: set[str] = set()
    for path in sorted(tinyvllm.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"source file is a symlink: {relative}")
        if not path.is_file():
            raise ValueError(f"source file is not regular: {relative}")
        owned.add(relative)
    pending = list(ENTRYPOINTS)
    while pending:
        relative = _safe_relative_path(pending.pop())
        if relative in owned:
            continue
        path = _require_regular_source(root, relative)
        owned.add(relative)
        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8"),
                filename=relative,
            )
        except (OSError, UnicodeDecodeError, SyntaxError) as error:
            raise ValueError(
                f"source file cannot be parsed: {relative}"
            ) from error
        for candidate in sorted(_local_import_candidates(tree)):
            if candidate in owned:
                continue
            candidate_path = root / candidate
            if candidate_path.is_symlink():
                raise ValueError(
                    f"source file is a symlink: {candidate}"
                )
            if candidate_path.is_file():
                pending.append(candidate)
    return tuple(sorted(owned))


def _tree_sha256(repo_root: Path, owned_files: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for relative in owned_files:
        path = _require_regular_source(repo_root, relative)
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _write_deterministic_tar(
    *,
    repo_root: Path,
    owned_files: tuple[str, ...],
    output_path: Path,
) -> None:
    with tarfile.open(output_path, "w", format=tarfile.GNU_FORMAT) as archive:
        for relative in owned_files:
            source = _require_regular_source(repo_root, relative)
            data = source.read_bytes()
            info = tarfile.TarInfo(relative)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            from io import BytesIO

            archive.addfile(info, BytesIO(data))


def build_source_bundle(
    *,
    repo_root: str | Path,
    output_dir: str | Path,
) -> dict:
    root = Path(repo_root).resolve()
    output = Path(output_dir)
    if output.exists():
        raise ValueError("source bundle output already exists")
    owned_files = collect_source_files(root)
    files = [
        {
            "path": relative,
            "sha256": sha256_file(root / relative),
            "size_bytes": (root / relative).stat().st_size,
        }
        for relative in owned_files
    ]
    source_tree_sha256 = _tree_sha256(root, owned_files)
    inventory = {
        "schema": SCHEMA,
        "files": files,
        "source_tree_sha256": source_tree_sha256,
    }
    output.mkdir(parents=True)
    inventory_path = output / SOURCE_INVENTORY_NAME
    inventory_path.write_bytes(_canonical_bytes(inventory) + b"\n")
    tar_path = output / SOURCE_TAR_NAME
    _write_deterministic_tar(
        repo_root=root,
        owned_files=owned_files,
        output_path=tar_path,
    )
    result = {
        **inventory,
        "source_inventory": str(inventory_path),
        "source_inventory_sha256": sha256_file(inventory_path),
        "source_tar": str(tar_path),
        "source_tar_sha256": sha256_file(tar_path),
    }
    validate_source_bundle(
        inventory_path=inventory_path,
        tar_path=tar_path,
    )
    return result


def validate_source_bundle(
    *,
    inventory_path: str | Path,
    tar_path: str | Path,
) -> dict:
    inventory_file = Path(inventory_path)
    archive_path = Path(tar_path)
    if (
        not inventory_file.is_file()
        or inventory_file.is_symlink()
        or not archive_path.is_file()
        or archive_path.is_symlink()
    ):
        raise ValueError("source bundle input is missing")
    try:
        inventory = json.loads(
            inventory_file.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source inventory is invalid") from error
    if (
        not isinstance(inventory, dict)
        or set(inventory) != {"schema", "files", "source_tree_sha256"}
        or inventory["schema"] != SCHEMA
        or not isinstance(inventory["files"], list)
        or not inventory["files"]
    ):
        raise ValueError("source inventory schema mismatch")
    paths = []
    for row in inventory["files"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "sha256", "size_bytes"}
            or not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in row["sha256"]
            )
            or isinstance(row["size_bytes"], bool)
            or not isinstance(row["size_bytes"], int)
            or row["size_bytes"] < 0
        ):
            raise ValueError("source inventory row mismatch")
        paths.append(_safe_relative_path(row["path"]))
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("source inventory ordering mismatch")
    with tarfile.open(archive_path, "r:") as archive:
        members = archive.getmembers()
        if [member.name for member in members] != paths:
            raise ValueError("source tar inventory mismatch")
        digest = hashlib.sha256()
        for member, row in zip(members, inventory["files"]):
            if (
                not member.isfile()
                or member.issym()
                or member.islnk()
                or member.size != row["size_bytes"]
            ):
                raise ValueError("unsafe source tar member")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError("source tar member is unreadable")
            data = extracted.read()
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("source tar file SHA mismatch")
            encoded = member.name.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            digest.update(data)
    if digest.hexdigest() != inventory["source_tree_sha256"]:
        raise ValueError("source tree SHA mismatch")
    return {
        **inventory,
        "source_inventory": str(inventory_file),
        "source_inventory_sha256": sha256_file(inventory_file),
        "source_tar": str(archive_path),
        "source_tar_sha256": sha256_file(archive_path),
    }
