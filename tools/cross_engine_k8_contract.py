from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
import re


GIB = 1024**3
REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818/cross-engine-k8-qwen3-06b"
)
MODEL_PATH = "/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B"
WARNING_BYTES = 16 * GIB
HARD_STOP_BYTES = 20 * GIB
LOCAL_HARD_STOP_BYTES = 50 * 1024**2
REQUIRED_CACHE_VARIABLES = (
    "XDG_CACHE_HOME",
    "HF_HOME",
    "MODELSCOPE_CACHE",
    "PIP_CACHE_DIR",
    "UV_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "PYTHONPYCACHEPREFIX",
    "TMPDIR",
)
LOCAL_ALLOWLIST = (
    "controller_manifest.json",
    "environment_manifest.json",
    "workload_manifest.json",
    "case_rows.jsonl",
    "correctness_rows.jsonl",
    "comparison.json",
    "summary.json",
    "gate.json",
    "remote_verification.json",
    "local_verification.json",
    "manifest.sha256",
)
_ATTEMPT_TAG = re.compile(
    r"^20260829-cross-engine-k8-qwen3-06b-r[1-9][0-9]*$"
)
_KLIST_TGT = re.compile(
    r"^(?P<issued>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4})"
    r"\s+"
    r"(?P<expires>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4})"
    r"\s+krbtgt/",
    re.MULTILINE,
)


@dataclass(frozen=True)
class CampaignPaths:
    remote_root: PurePosixPath
    model_path: PurePosixPath

    @classmethod
    def create(
        cls,
        *,
        remote_root: str,
        model_path: str,
    ) -> "CampaignPaths":
        root = _canonical_absolute(remote_root, "remote_root")
        model = _canonical_absolute(model_path, "model_path")
        if root == model or root in model.parents or model in root.parents:
            raise ValueError("campaign and model paths must be disjoint")
        return cls(remote_root=root, model_path=model)

    def require_owned_remote(self, path: str) -> PurePosixPath:
        candidate = _canonical_absolute(path, "remote path")
        if candidate == self.remote_root:
            return candidate
        if self.remote_root not in candidate.parents:
            raise ValueError("remote path is outside the campaign root")
        if (
            candidate == self.model_path
            or self.model_path in candidate.parents
            or candidate.name.startswith("model-copy")
        ):
            raise ValueError("model path is not campaign-owned")
        return candidate


def _canonical_absolute(raw: str, name: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw.startswith("/"):
        raise ValueError(f"{name} must be absolute")
    path = PurePosixPath(raw)
    if any(part in (".", "..") for part in path.parts):
        raise ValueError(f"{name} must be canonical")
    if path.as_posix() != raw.rstrip("/") and raw != "/":
        raise ValueError(f"{name} must be canonical")
    return path


def cache_environment(paths: CampaignPaths) -> dict[str, str]:
    cache_root = paths.remote_root / "shared" / "package-cache"
    mapping = {
        "XDG_CACHE_HOME": cache_root / "xdg",
        "HF_HOME": cache_root / "huggingface",
        "MODELSCOPE_CACHE": cache_root / "modelscope",
        "PIP_CACHE_DIR": cache_root / "pip",
        "UV_CACHE_DIR": cache_root / "uv",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "TORCHINDUCTOR_CACHE_DIR": cache_root / "torchinductor",
        "CUDA_CACHE_PATH": cache_root / "cuda",
        "PYTHONPYCACHEPREFIX": cache_root / "pycache",
        "TMPDIR": paths.remote_root / "shared" / "tmp",
    }
    return {
        key: paths.require_owned_remote(value.as_posix()).as_posix()
        for key, value in mapping.items()
    }


def classify_allocated_bytes(bytes_used: int) -> str:
    if (
        isinstance(bytes_used, bool)
        or not isinstance(bytes_used, int)
        or bytes_used < 0
    ):
        raise ValueError("bytes_used must be a non-negative integer")
    if bytes_used >= HARD_STOP_BYTES:
        return "HARD_STOP"
    if bytes_used >= WARNING_BYTES:
        return "WARNING"
    return "OK"


def parse_klist_lifetime(text: str, now: datetime) -> timedelta:
    if not isinstance(text, str):
        raise TypeError("klist output must be text")
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    expiries = []
    for match in _KLIST_TGT.finditer(text):
        expiry = datetime.strptime(
            match.group("expires"),
            "%b %d %H:%M:%S %Y",
        ).replace(tzinfo=now.tzinfo)
        expiries.append(expiry)
    if not expiries:
        raise RuntimeError("KERBEROS_TGT_NOT_FOUND")
    return max(expiries) - now


def require_kerberos_coverage(
    lifetime: timedelta,
    estimated: timedelta,
    margin: timedelta,
) -> None:
    if lifetime < estimated + margin:
        raise RuntimeError("KERBEROS_TTL_INSUFFICIENT")


def validate_attempt_tag(tag: str) -> str:
    if not isinstance(tag, str) or _ATTEMPT_TAG.fullmatch(tag) is None:
        raise ValueError("attempt tag does not match the frozen format")
    return tag


def validate_local_allowlist(root: Path) -> dict:
    root = Path(root)
    if not root.is_dir():
        raise RuntimeError("LOCAL_BUNDLE_MISSING")
    entries = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    )
    if set(entries) != set(LOCAL_ALLOWLIST):
        raise RuntimeError("LOCAL_ALLOWLIST_VIOLATION")
    total_bytes = sum((root / name).stat().st_size for name in entries)
    if total_bytes > LOCAL_HARD_STOP_BYTES:
        raise RuntimeError("LOCAL_STORAGE_HARD_STOP")
    return {
        "files": list(LOCAL_ALLOWLIST),
        "total_bytes": total_bytes,
        "limit_bytes": LOCAL_HARD_STOP_BYTES,
    }
