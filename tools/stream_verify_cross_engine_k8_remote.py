from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Callable, Mapping, Optional, Sequence

from tools.cross_engine_k8_contract import (
    LOCAL_ALLOWLIST,
    LOCAL_HARD_STOP_BYTES,
    REMOTE_ROOT,
    validate_attempt_tag,
    validate_local_allowlist,
)
from tools.verify_cross_engine_k8 import verify_bundle


KRB5_CACHE = "FILE:/Users/bytedance/krb5cc_sitian"
REMOTE_HOST = "sitian@10.232.195.203"


@dataclass(frozen=True)
class StreamVerificationConfig:
    remote_run_tag: str
    local_root: Path
    expected_source: str

    def __post_init__(self):
        validate_attempt_tag(self.remote_run_tag)
        if (
            not isinstance(self.expected_source, str)
            or len(self.expected_source) != 40
            or any(
                character not in "0123456789abcdef"
                for character in self.expected_source
            )
        ):
            raise ValueError("expected source revision is invalid")

    @property
    def remote_root(self) -> str:
        return (
            f"{REMOTE_ROOT}/attempts/{self.remote_run_tag}/remote-final"
        )


class SSHRemoteReader:
    def __init__(
        self,
        host: str,
        *,
        command_runner: Callable = subprocess.run,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.host = host
        self._command_runner = command_runner
        self._sleep = sleep

    def _run(self, remote_command: str, *, text: bool):
        environment = dict(os.environ)
        environment["KRB5CCNAME"] = KRB5_CACHE
        argv = [
            "ssh",
            "-o",
            "ControlMaster=no",
            "-o",
            "ControlPath=none",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            self.host,
            remote_command,
        ]
        result = None
        for attempt in range(5):
            result = self._command_runner(
                argv,
                env=environment,
                text=text,
                capture_output=True,
                check=False,
            )
            if result.returncode != 255 or attempt == 4:
                break
            self._sleep(1.0)
        if result is None or result.returncode != 0:
            detail = "" if result is None else str(result.stderr or "")
            raise RuntimeError(f"remote stream failed: {detail.strip()}")
        return result

    def list_files(self, remote_root: str) -> dict[str, int]:
        script = "\n".join((
            "import json,os,sys",
            "root=sys.argv[1]",
            "rows={}",
            "for name in os.listdir(root):",
            " path=os.path.join(root,name)",
            " if os.path.isfile(path) and not os.path.islink(path):",
            "  rows[name]=os.stat(path).st_size",
            "print(json.dumps(rows,sort_keys=True))",
        ))
        result = self._run(
            shlex.join(["python3", "-c", script, remote_root]),
            text=True,
        )
        payload = json.loads(result.stdout)
        if (
            not isinstance(payload, dict)
            or any(
                not isinstance(name, str)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                for name, size in payload.items()
            )
        ):
            raise RuntimeError("remote file inventory is invalid")
        return payload

    def read_bytes(self, remote_root: str, name: str) -> bytes:
        if name not in LOCAL_ALLOWLIST:
            raise ValueError("remote read is outside the local allowlist")
        path = f"{remote_root}/{name}"
        result = self._run(
            shlex.join(["python3", "-c", (
                "import pathlib,sys;"
                "sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())"
            ), path]),
            text=False,
        )
        return bytes(result.stdout)


def stream_verify(
    config: StreamVerificationConfig,
    ssh_runner,
    verifier: Callable = verify_bundle,
) -> dict:
    destination = Path(config.local_root)
    destination.mkdir(parents=True, exist_ok=True)
    remote_inventory = ssh_runner.list_files(config.remote_root)
    required_remote = set(LOCAL_ALLOWLIST) - {"local_verification.json"}
    missing = sorted(required_remote - set(remote_inventory))
    if missing:
        raise RuntimeError(
            "REMOTE_ALLOWLIST_INCOMPLETE:" + ",".join(missing)
        )
    retained_bytes = sum(
        remote_inventory[name] for name in required_remote
    )
    if retained_bytes > LOCAL_HARD_STOP_BYTES:
        raise RuntimeError("LOCAL_STORAGE_HARD_STOP")
    created = []
    temporary_paths = []
    try:
        for name in LOCAL_ALLOWLIST:
            if name == "local_verification.json":
                continue
            payload = ssh_runner.read_bytes(config.remote_root, name)
            if len(payload) != remote_inventory[name]:
                raise RuntimeError("REMOTE_STREAM_SIZE_MISMATCH")
            target = destination / name
            temporary = destination / f".{name}.streaming-tmp"
            temporary_paths.append(temporary)
            temporary.write_bytes(payload)
            temporary.replace(target)
            created.append(target)
        local_result = verifier(
            destination,
            expected_source=config.expected_source,
        )
        remote_result = json.loads(
            (destination / "remote_verification.json").read_text(
                encoding="utf-8"
            )
        )
        agreement_fields = (
            "valid",
            "recomputed_classification",
        )
        if any(
            local_result.get(field) != remote_result.get(field)
            for field in agreement_fields
        ):
            raise RuntimeError("VERIFIER_DISAGREEMENT")
        local_path = destination / "local_verification.json"
        local_path.write_text(
            json.dumps(local_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        created.append(local_path)
        allowlist = validate_local_allowlist(destination)
        return {
            **dict(local_result),
            "remote_verifier_agrees": True,
            "local_retention": allowlist,
        }
    except Exception:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=REMOTE_HOST)
    parser.add_argument("--remote-run-tag", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--expected-source", required=True)
    args = parser.parse_args(argv)
    destination = args.local_root / args.remote_run_tag
    result = stream_verify(
        StreamVerificationConfig(
            remote_run_tag=args.remote_run_tag,
            local_root=destination,
            expected_source=args.expected_source,
        ),
        SSHRemoteReader(args.host),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
