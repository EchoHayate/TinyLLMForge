from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import tarfile
import time

import qwen35_tp4_hybrid_prefix_benchmark_contract as contract
import run_qwen35_tp4_hybrid_prefix_benchmark_remote as runner


TAG = os.environ.get(
    "QWEN35_W2_RUN_TAG",
    "qwen35-tp4-decode-phase-split-20260811-r631-attempt004",
)
OLD_TAG = "qwen35-tp4-decode-row-parallel-fp32-20260811-r630-attempt001"
OLD_SOURCE = (
    "7a80d62c2c9e71f7899dc397f810427286b330d95da2b63f67839aa98d47b3b3"
)
SOURCE = (
    "6f881fae7010cc5f048100b147a72fbf27ffba0f77bc34e2e2e68388a98a2837"
)
SOURCE_TAR_SHA = (
    "f791f27e807e602f889345d301b72035dcd4a93d55a32adf51fd5eb3eaefb79c"
)
PREREQUISITES_SHA = (
    "35b4bf092d5c4c84746b88ecd88b32bf14357a21d2923336d62653186cf352f8"
)
MINIMUM_FREE_BYTES = 25 * 1024**3
MAXIMUM_UTILIZATION = 10
ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "experiments/qwen35_hybrid_state" / TAG
REMOTE = f"{runner.REMOTE_ROOT}/{TAG}"
SOURCE_TAR = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-tp4-decode-phase-split-source-20260811-r631-prep001/"
    "first/benchmark_source.tar"
)
PREREQUISITES = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-tp4-decode-row-parallel-fp32-20260811-r630-attempt001/"
    "inputs/prerequisites_bundle/correctness_prerequisites.json"
)
COMMAND_TEMPLATE = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-tp4-decode-row-parallel-fp32-20260811-r630-attempt001/"
    "commands.json"
)
SSH = [
    "ssh",
    "-o",
    "BatchMode=yes",
    "-o",
    "ControlMaster=no",
    "-o",
    "ControlPath=none",
    "-o",
    "ConnectTimeout=20",
    runner.SSH_TARGET,
]
SCP = [
    "scp",
    "-o",
    "BatchMode=yes",
    "-o",
    "ControlMaster=no",
    "-o",
    "ControlPath=none",
    "-o",
    "ConnectTimeout=20",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(argv, text=True, capture_output=True)
    if check and result.returncode != 0:
        detail = result.stderr or result.stdout or repr(argv)
        raise RuntimeError(detail[-4096:])
    return result


def retry_idempotent(argv: list[str], attempts: int = 3) -> subprocess.CompletedProcess:
    result = None
    for attempt in range(1, attempts + 1):
        result = run(argv, check=False)
        if result.returncode == 0:
            return result
        if attempt < attempts:
            time.sleep(attempt)
    detail = result.stderr or result.stdout or repr(argv)
    raise RuntimeError(detail[-4096:])


def ssh(command: str, *, retry: bool = False) -> subprocess.CompletedProcess:
    argv = [*SSH, command]
    return retry_idempotent(argv) if retry else run(argv)


def remote_path_exists(path: str) -> bool:
    result = None
    for attempt in range(1, 4):
        result = run(
            [*SSH, f"test -e {shlex.quote(path)}"],
            check=False,
        )
        if result.returncode in {0, 1}:
            return result.returncode == 0
        if attempt < 3:
            time.sleep(attempt)
    detail = result.stderr or result.stdout or "remote existence check failed"
    raise RuntimeError(detail[-4096:])


def scp(source: Path | str, destination: Path | str) -> None:
    retry_idempotent([*SCP, str(source), str(destination)])


def replace_candidate(value):
    if isinstance(value, str):
        return value.replace(OLD_TAG, TAG).replace(OLD_SOURCE, SOURCE)
    if isinstance(value, list):
        return [replace_candidate(item) for item in value]
    if isinstance(value, dict):
        return {
            key: replace_candidate(item)
            for key, item in value.items()
        }
    return value


def build_commands() -> list[dict]:
    commands = json.loads(COMMAND_TEMPLATE.read_text())
    if (
        len(commands) != 12
        or any(
            not row.get("case_id", "").startswith("w2_long_reuse__")
            for row in commands
        )
    ):
        raise ValueError("r630 template is not the expected 12-case w2 matrix")
    return [replace_candidate(row) for row in commands]


def build_remote_guard() -> str:
    return f"""\
import json
from pathlib import Path
import subprocess
import sys
import time

INDICES = (2, 4, 5, 6)
MINIMUM_FREE_BYTES = {MINIMUM_FREE_BYTES}
MAXIMUM_UTILIZATION = {MAXIMUM_UTILIZATION}
MIB = 1024 * 1024
OUTPUT = Path({REMOTE!r}) / "guards"


def query():
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    rows = []
    by_uuid = {{}}
    for line in gpu.splitlines():
        parts = [value.strip() for value in line.split(",", 3)]
        row = {{
            "gpu_index": int(parts[0]),
            "gpu_uuid": parts[1],
            "free_bytes": int(parts[2]) * MIB,
            "utilization_percent": int(parts[3]),
            "compute_processes": [],
        }}
        rows.append(row)
        by_uuid[row["gpu_uuid"]] = row
    for line in processes.splitlines():
        if not line.strip() or line.strip() == "No running processes found":
            continue
        parts = [value.strip() for value in line.split(",", 3)]
        if parts[0] in by_uuid:
            by_uuid[parts[0]]["compute_processes"].append({{
                "pid": int(parts[1]),
                "process_name": parts[2],
                "used_bytes": int(parts[3]) * MIB,
            }})
    return rows


label = sys.argv[1]
started = time.monotonic()
samples = []
while True:
    rows = query()
    by_index = {{row["gpu_index"]: row for row in rows}}
    selected = [by_index[index] for index in INDICES if index in by_index]
    reasons = []
    if len(selected) != len(INDICES):
        reasons.append("required GPUs 2,4,5,6 missing")
    for row in selected:
        if row["free_bytes"] < MINIMUM_FREE_BYTES:
            reasons.append(
                f"GPU {{row['gpu_index']}} free memory below threshold"
            )
        if row["utilization_percent"] > MAXIMUM_UTILIZATION:
            reasons.append(
                f"GPU {{row['gpu_index']}} utilization above threshold"
            )
    samples.append({{
        "selected_gpus": selected,
        "reasons": reasons,
    }})
    if not reasons:
        payload = {{
            "classification": "READY",
            "exclusive": False,
            "label": label,
            "resource_policy": "shared-low-utilization",
            "minimum_gpu_free_bytes": MINIMUM_FREE_BYTES,
            "maximum_gpu_utilization_percent": MAXIMUM_UTILIZATION,
            "selected_gpus": selected,
            "wait_elapsed_s": time.monotonic() - started,
            "samples": samples,
        }}
        OUTPUT.joinpath(label + ".json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\\n"
        )
        raise SystemExit(0)
    if time.monotonic() - started >= 1800:
        payload = {{
            "classification": "BLOCKED",
            "label": label,
            "reasons": reasons,
            "samples": samples,
        }}
        OUTPUT.joinpath(label + ".json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\\n"
        )
        raise SystemExit(3)
    time.sleep(30)
"""


def build_remote_runner(commands: list[dict]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"run_root={shlex.quote(REMOTE)}",
        "failed=1",
        (
            "trap 'rc=$?; if [[ $failed -eq 1 ]]; then "
            "printf \"%s\\n\" \"$rc\" > \"$run_root/RUN_FAILED\"; fi' EXIT"
        ),
    ]
    for index, row in enumerate(commands, 1):
        case_id = row["case_id"]
        guard_label = f"worker-{case_id}"
        env = [
            f"{key}={value}"
            for key, value in sorted(row["env"].items())
        ]
        command = " ".join(
            [
                "env",
                *[shlex.quote(value) for value in env],
                *[shlex.quote(value) for value in row["argv"]],
            ]
        )
        lines.extend([
            (
                f"{shlex.quote(runner.REMOTE_PYTHON)} "
                f"{shlex.quote(REMOTE + '/remote_guard.py')} "
                f"{shlex.quote(guard_label)}"
            ),
            f"printf '[{index}/12] start %s\\n' {shlex.quote(case_id)}",
            f"cd {shlex.quote(REMOTE + '/source')}",
            (
                f"{command} > "
                f"{shlex.quote(REMOTE + '/logs/' + case_id + '.log')} 2>&1"
            ),
            f"printf '[{index}/12] done %s\\n' {shlex.quote(case_id)}",
        ])
    lines.extend([
        "failed=0",
        "touch \"$run_root/RUN_COMPLETE\"",
    ])
    return "\n".join(lines) + "\n"


def cleanup() -> dict:
    code = r"""
import os
import signal
import sys
import time

needle = sys.argv[1]


def snapshot():
    rows = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            stat = open(f"/proc/{pid}/stat").read()
            command = (
                open(f"/proc/{pid}/cmdline", "rb")
                .read()
                .replace(b"\0", b" ")
                .decode("utf-8", "replace")
                .strip()
            )
        except OSError:
            continue
        fields = stat[stat.rfind(")") + 2 :].split()
        rows[pid] = (int(fields[1]), command)
    return rows


for _ in range(10):
    rows = snapshot()
    excluded = set()
    pid = os.getpid()
    while pid > 1 and pid not in excluded:
        excluded.add(pid)
        pid = rows.get(pid, (0, ""))[0]
    roots = {
        pid
        for pid, (_, command) in rows.items()
        if pid not in excluded and needle in command
    }
    targets = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, (parent, _) in rows.items():
            if pid not in targets and parent in targets:
                targets.add(pid)
                changed = True
    if not targets:
        time.sleep(0.5)
        rows = snapshot()
        if not any(
            pid not in excluded and needle in command
            for pid, (_, command) in rows.items()
        ):
            print("CLEAN")
            raise SystemExit(0)
    for pid in sorted(targets, reverse=True):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(1)
    for pid in targets:
        if os.path.exists(f"/proc/{pid}"):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    time.sleep(1)
raise SystemExit("cleanup did not stabilize")
"""
    result = retry_idempotent([
        *SSH,
        f"python3 -c {shlex.quote(code)} {shlex.quote(TAG)}",
    ])
    return {
        "classification": (
            "CLEAN"
            if result.returncode == 0 and "CLEAN" in result.stdout
            else "FAILED"
        ),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:") as handle:
        members = handle.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or member.issym()
                or member.islnk()
            ):
                raise ValueError("unsafe downloaded member")
        handle.extractall(destination, members=members)


def main() -> int:
    if sha256(SOURCE_TAR) != SOURCE_TAR_SHA:
        raise ValueError("source tar SHA mismatch")
    if sha256(PREREQUISITES) != PREREQUISITES_SHA:
        raise ValueError("prerequisite SHA mismatch")
    if OUTPUT.exists() and OUTPUT != Path(__file__).resolve().parent:
        raise ValueError("fresh local tag already exists")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    script_copy = OUTPUT / "launch_w2.py"
    if script_copy != Path(__file__).resolve():
        shutil.copy2(Path(__file__).resolve(), script_copy)
    inputs = OUTPUT / "inputs"
    inputs.mkdir()
    (OUTPUT / "download").mkdir()
    shutil.copy2(SOURCE_TAR, inputs / "benchmark_source.tar")
    shutil.copy2(
        PREREQUISITES,
        inputs / "correctness_prerequisites.json",
    )
    workload = inputs / "workload_manifest.json"
    workload.write_text(
        json.dumps(
            contract.workload_manifest_payload(),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    commands = build_commands()
    (OUTPUT / "commands.json").write_text(
        json.dumps(commands, indent=2, sort_keys=True) + "\n"
    )
    guard = inputs / "remote_guard.py"
    guard.write_text(build_remote_guard())
    remote_runner = inputs / "remote_runner.sh"
    remote_runner.write_text(build_remote_runner(commands))
    receipt = {
        "classification": "RUNNING",
        "run_tag": TAG,
        "source_tree_sha256": SOURCE,
        "source_tar_sha256": SOURCE_TAR_SHA,
    }
    error = None
    try:
        if remote_path_exists(REMOTE):
            raise ValueError("fresh remote tag already exists")
        ssh(
            " && ".join([
                f"mkdir -p {shlex.quote(REMOTE + '/source')}",
                f"mkdir -p {shlex.quote(REMOTE + '/output/cases')}",
                f"mkdir -p {shlex.quote(REMOTE + '/logs')}",
                f"mkdir -p {shlex.quote(REMOTE + '/guards')}",
                f"mkdir -p {shlex.quote(REMOTE + '/prerequisites')}",
            ]),
            retry=True,
        )
        uploads = [
            (
                inputs / "benchmark_source.tar",
                f"{runner.SSH_TARGET}:{REMOTE}/benchmark_source.tar",
            ),
            (
                inputs / "correctness_prerequisites.json",
                (
                    f"{runner.SSH_TARGET}:{REMOTE}/prerequisites/"
                    "correctness_prerequisites.json"
                ),
            ),
            (
                workload,
                f"{runner.SSH_TARGET}:{REMOTE}/workload_manifest.json",
            ),
            (
                guard,
                f"{runner.SSH_TARGET}:{REMOTE}/remote_guard.py",
            ),
            (
                remote_runner,
                f"{runner.SSH_TARGET}:{REMOTE}/remote_runner.sh",
            ),
        ]
        for source, destination in uploads:
            scp(source, destination)
        workload_sha = sha256(workload)
        ssh(
            " && ".join([
                (
                    f"test $(sha256sum "
                    f"{shlex.quote(REMOTE + '/benchmark_source.tar')} "
                    f"| awk '{{print $1}}') = {SOURCE_TAR_SHA}"
                ),
                (
                    f"test $(sha256sum "
                    f"{shlex.quote(REMOTE + '/prerequisites/correctness_prerequisites.json')} "
                    f"| awk '{{print $1}}') = {PREREQUISITES_SHA}"
                ),
                (
                    f"test $(sha256sum "
                    f"{shlex.quote(REMOTE + '/workload_manifest.json')} "
                    f"| awk '{{print $1}}') = {workload_sha}"
                ),
                (
                    f"tar -xf {shlex.quote(REMOTE + '/benchmark_source.tar')} "
                    f"-C {shlex.quote(REMOTE + '/source')}"
                ),
            ]),
            retry=True,
        )
        print("remote runner starting", flush=True)
        result = run(
            [
                *SSH,
                f"bash {shlex.quote(REMOTE + '/remote_runner.sh')}",
            ],
            check=False,
        )
        print(result.stdout, end="", flush=True)
        if result.returncode != 0:
            detail = result.stderr or result.stdout
            raise RuntimeError(detail[-4096:])
        ssh(
            (
                f"tar -C {shlex.quote(REMOTE)} -cf "
                f"{shlex.quote(REMOTE + '/result.tar')} "
                "output/cases guards logs RUN_COMPLETE"
            ),
            retry=True,
        )
        archive = OUTPUT / "download/result.tar"
        scp(
            f"{runner.SSH_TARGET}:{REMOTE}/result.tar",
            archive,
        )
        safe_extract(archive, OUTPUT / "download")
        receipt["classification"] = "DOWNLOADED"
    except BaseException as caught:
        error = caught
        receipt["classification"] = "FAILED"
        receipt["error"] = repr(caught)
    finally:
        try:
            receipt["cleanup"] = cleanup()
        except BaseException as cleanup_error:
            receipt["cleanup"] = {
                "classification": "FAILED",
                "error": repr(cleanup_error),
            }
        (OUTPUT / "attempt_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
    if error is not None:
        raise error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
