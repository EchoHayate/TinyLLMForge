from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import secrets
import shlex
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
OUTPUT_ROOT = ROOT / "experiments/qwen35_hybrid_state"
SSH_TARGET = "sitian@10.232.195.203"
REMOTE_PYTHON = (
    "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
)
REMOTE_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-tp4-hybrid-prefix-benchmark-runs"
)
GPU_INDICES = (2, 4, 5, 6)
WORKERS_TIMEOUT_S = 12 * 60 * 60
MAXIMUM_GPU_UTILIZATION_PERCENT = 10
RESOURCE_SHARING_POLICY = "shared-low-utilization"
PERFORMANCE_CLAIM_BOUNDARY = (
    "non-exclusive shared-GPU observation; not an "
    "uncontended strict-P1 performance baseline"
)


def _load_module(name, filename):
    path = TOOLS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


adapter = _load_module(
    "qwen35_tp4_engine_remote_subprocess_adapter_for_monitor",
    "qwen35_tp4_engine_remote_subprocess_adapter.py",
)
transport = _load_module(
    "qwen35_tp4_controlmaster_transport_for_monitor",
    "qwen35_tp4_controlmaster_transport.py",
)
monitor = _load_module(
    "qwen35_tp4_strict_p1_monitor_for_live_runner",
    "qwen35_tp4_strict_p1_monitor.py",
)
benchmark = _load_module(
    "run_qwen35_tp4_hybrid_prefix_benchmark_remote_for_monitor",
    "run_qwen35_tp4_hybrid_prefix_benchmark_remote.py",
)
MINIMUM_FREE_BYTES = benchmark.MIN_GPU_FREE_BYTES


def _remote_shell_argv(script):
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ControlMaster=no",
        "-o",
        "ConnectTimeout=20",
        SSH_TARGET,
        "bash",
        "-lc",
        shlex.quote(script),
    ]


def _gpu_query_script():
    return shlex.join([
        REMOTE_PYTHON,
        "-c",
        "\n".join([
            "import importlib.util,json,subprocess,sys",
            "from pathlib import Path",
            (
                "path=Path('/data00/home/sitian/sitian-workspace01/"
                "tllm/qwen35-tp4-root-logit-tests/"
                "qwen35-tp4-source-prep-20260729-010400/source/"
                "tools/qwen35_tp4_real_root_logit_correctness_"
                "preflight.py')"
            ),
            (
                "spec=importlib.util.spec_from_file_location("
                "'qwen35_monitor_gpu_query',path)"
            ),
            "module=importlib.util.module_from_spec(spec)",
            "sys.modules[spec.name]=module",
            "spec.loader.exec_module(module)",
            "rows=[dict(row) for row in module._query_tp4_gpu_resources()]",
            (
                "query=subprocess.run(["
                "'nvidia-smi',"
                "'--query-gpu=index,utilization.gpu',"
                "'--format=csv,noheader,nounits'"
                "],check=False,text=True,capture_output=True)"
            ),
            (
                "query.returncode==0 or sys.exit("
                "'nvidia-smi utilization query failed: '+query.stderr)"
            ),
            "utilization={}",
            "for line in query.stdout.splitlines():",
            " fields=[field.strip() for field in line.split(',')]",
            (
                " len(fields)==2 or sys.exit("
                "'nvidia-smi utilization output is invalid')"
            ),
            " utilization[int(fields[0])]=int(fields[1])",
            "for row in rows:",
            (
                " row['utilization_percent']="
                "utilization[row['gpu_index']]"
            ),
            (
                "print(json.dumps({'gpus':rows},"
                "sort_keys=True,separators=(',',':')))"
            ),
        ]),
    ])


def _benchmark_remote_query(command_runner):
    result = command_runner(
        name="monitor_resource_query",
        argv=_remote_shell_argv(_gpu_query_script()),
        stdout_path=None,
        env=dict(adapter.REQUIRED_EXECUTION_ENV),
    )
    if result["returncode"] != 0:
        raise RuntimeError(
            result["stderr"].strip()
            or result["stdout"].strip()
            or "remote GPU resource query failed"
        )
    try:
        payload = json.loads(result["stdout"])
    except json.JSONDecodeError as error:
        raise ValueError("remote GPU resource JSON is invalid") from error
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("gpus"), list)
    ):
        raise ValueError("remote GPU resource payload is invalid")
    return {"gpus": payload["gpus"]}


def _select_shared_tp4_gpu_resources(rows):
    if not isinstance(rows, list):
        raise ValueError("GPU resource rows must be a list")
    by_index = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("GPU resource row is invalid")
        gpu_index = row.get("gpu_index")
        gpu_uuid = row.get("gpu_uuid")
        free_bytes = row.get("free_bytes")
        compute_processes = row.get("compute_processes")
        utilization_percent = row.get("utilization_percent")
        if (
            isinstance(gpu_index, bool)
            or not isinstance(gpu_index, int)
            or gpu_index < 0
            or not isinstance(gpu_uuid, str)
            or not gpu_uuid.startswith("GPU-")
            or isinstance(free_bytes, bool)
            or not isinstance(free_bytes, int)
            or not isinstance(compute_processes, list)
            or isinstance(utilization_percent, bool)
            or not isinstance(utilization_percent, int)
            or not 0 <= utilization_percent <= 100
        ):
            raise ValueError("GPU resource row is invalid")
        if gpu_index in by_index:
            raise ValueError("GPU resource indices must be unique")
        by_index[gpu_index] = dict(row)
    missing = [index for index in GPU_INDICES if index not in by_index]
    if missing:
        raise ValueError("fixed GPUs 2,4,5,6 are required")
    selected = [by_index[index] for index in GPU_INDICES]
    if any(
        row["free_bytes"] < MINIMUM_FREE_BYTES
        for row in selected
    ):
        raise ValueError("fixed GPU free memory is insufficient")
    if any(
        row["utilization_percent"]
        > MAXIMUM_GPU_UTILIZATION_PERCENT
        for row in selected
    ):
        raise ValueError("fixed GPU utilization exceeds shared limit")
    if len({row["gpu_uuid"] for row in selected}) != len(GPU_INDICES):
        raise ValueError("selected GPU identities must be unique")
    return selected


def _query_resources(command_runner, sample_id):
    payload = _benchmark_remote_query(command_runner)
    rows = payload.get("gpus") if isinstance(payload, dict) else None
    classification = "READY"
    reason = None
    try:
        selected = _select_shared_tp4_gpu_resources(rows)
    except ValueError as error:
        classification = "BLOCKED_RESOURCES"
        reason = str(error)
        selected = []
    return {
        "schema_version": monitor.SCHEMA_VERSION,
        "sample_id": sample_id,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "classification": classification,
        "reason": reason,
        "gpu_indices": list(GPU_INDICES),
        "minimum_free_bytes_per_gpu": MINIMUM_FREE_BYTES,
        "requires_no_active_compute_processes": False,
        "maximum_gpu_utilization_percent": (
            MAXIMUM_GPU_UTILIZATION_PERCENT
        ),
        "resource_sharing_policy": RESOURCE_SHARING_POLICY,
        "performance_claim_boundary": PERFORMANCE_CLAIM_BOUNDARY,
        "gpu_rows": rows,
        "selected_gpus": selected,
    }


def _cleanup_script(run_tag):
    remote_run = f"{REMOTE_ROOT}/{run_tag}"
    encoded_tag = json.dumps(run_tag)
    encoded_root = json.dumps(remote_run)
    code = "\n".join([
        "import json,os,signal,time",
        f"run_tag={encoded_tag}",
        f"remote_root={encoded_root}",
        "self_pid=os.getpid()",
        "ancestors=set()",
        "pid=self_pid",
        "while pid>1:",
        " ancestors.add(pid)",
        " try:",
        "  fields=open(f'/proc/{pid}/stat',encoding='utf-8').read().split()",
        "  pid=int(fields[3])",
        " except (OSError,ValueError,IndexError):",
        "  break",
        "def snapshot():",
        " rows={}",
        " for name in os.listdir('/proc'):",
        "  if not name.isdigit(): continue",
        "  pid=int(name)",
        "  if pid in ancestors: continue",
        "  try:",
        "   stat_text=open(f'/proc/{pid}/stat',encoding='utf-8').read()",
        "   close=stat_text.rfind(')')",
        "   fields=stat_text[close+2:].split()",
        "   ppid=int(fields[1])",
        "   start_time=int(fields[19])",
        "   raw=open(f'/proc/{pid}/cmdline','rb').read()",
        "  except (OSError,ValueError,IndexError):",
        "   continue",
        "  text=raw.replace(b'\\0',b' ').decode('utf-8','replace')",
        "  rows[pid]={'ppid':ppid,'start_time':start_time,'cmdline':text[:2048]}",
        " return rows",
        "initial=snapshot()",
        "root_pids=sorted(pid for pid,row in initial.items() if run_tag in row['cmdline'] and remote_root in row['cmdline'])",
        "target_pids=set(root_pids)",
        "changed=True",
        "while changed:",
        " changed=False",
        " for pid,row in initial.items():",
        "  if pid not in target_pids and row['ppid'] in target_pids:",
        "   target_pids.add(pid)",
        "   changed=True",
        "descendants=sorted(target_pids-set(root_pids))",
        "target_start_times={pid:initial[pid]['start_time'] for pid in target_pids}",
        "def live_targets():",
        " current=snapshot()",
        " return sorted(pid for pid,start_time in target_start_times.items() if pid in current and current[pid]['start_time']==start_time)",
        "before=live_targets()",
        "for pid in before:",
        " try: os.kill(pid,signal.SIGTERM)",
        " except ProcessLookupError: pass",
        "deadline=time.monotonic()+10.0",
        "while time.monotonic()<deadline and live_targets(): time.sleep(0.2)",
        "remaining=live_targets()",
        "for pid in remaining:",
        " try: os.kill(pid,signal.SIGKILL)",
        " except ProcessLookupError: pass",
        "time.sleep(0.5)",
        "after=live_targets()",
        "gpu_process_pids=[]",
        "try:",
        " import subprocess",
        " query=subprocess.run([",
        "  'nvidia-smi','--query-compute-apps=pid',",
        "  '--format=csv,noheader,nounits',",
        " ],check=False,text=True,capture_output=True)",
        " if query.returncode==0:",
        "  for line in query.stdout.splitlines():",
        "   value=line.strip()",
        "   if value.isdigit(): gpu_process_pids.append(int(value))",
        "except (OSError,ValueError):",
        " pass",
        "matched_gpu_pids_after_cleanup=sorted(",
        " set(gpu_process_pids)&set(after)",
        ")",
        "print(json.dumps({",
        " 'classification':'CLEAN' if not after and not matched_gpu_pids_after_cleanup else 'CLEANUP_INCOMPLETE',",
        " 'scope':{'run_tag':run_tag,'remote_root':remote_root},",
        " 'root_pids':root_pids,",
        " 'descendants':descendants,",
        " 'target_pids':sorted(target_pids),",
        " 'target_start_times':{str(pid):start for pid,start in sorted(target_start_times.items())},",
        " 'matched_pids':before,",
        " 'term_remaining_pids':remaining,",
        " 'remaining_pids':after,",
        " 'remaining_target_pids':after,",
        " 'gpu_process_pids':sorted(set(gpu_process_pids)),",
        " 'matched_gpu_pids_after_cleanup':matched_gpu_pids_after_cleanup,",
        "},sort_keys=True,separators=(',',':')))",
        "raise SystemExit(0 if not after and not matched_gpu_pids_after_cleanup else 2)",
    ])
    return shlex.join([REMOTE_PYTHON, "-c", code])


def _cleanup_run(command_runner, run_tag):
    result = command_runner(
        name="monitor_scoped_cleanup",
        argv=_remote_shell_argv(_cleanup_script(run_tag)),
        stdout_path=None,
        env=dict(adapter.REQUIRED_EXECUTION_ENV),
    )
    try:
        payload = json.loads(result["stdout"])
    except json.JSONDecodeError:
        payload = {
            "classification": "CLEANUP_FAILED",
            "matched_pids": [],
            "remaining_pids": [],
            "returncode": result["returncode"],
            "stderr": result["stderr"][-4096:],
        }
    if result["returncode"] != 0:
        payload["classification"] = "CLEANUP_FAILED"
        payload["returncode"] = result["returncode"]
        payload["stderr"] = result["stderr"][-4096:]
    return payload


def _authorization_nonce():
    return (
        "strict-p1-monitor-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + secrets.token_hex(16)
    )


def _raise_signal_exit(signal_number, frame):
    del frame
    raise SystemExit(128 + signal_number)


def _install_signal_handlers():
    signal.signal(signal.SIGTERM, _raise_signal_exit)
    signal.signal(signal.SIGINT, _raise_signal_exit)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor-tag", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--control-path", required=True, type=Path)
    parser.add_argument("--prerequisites", required=True, type=Path)
    parser.add_argument("--local-model-manifest", required=True, type=Path)
    parser.add_argument("--remote-model-dir", required=True)
    parser.add_argument("--remote-model-manifest", required=True)
    parser.add_argument("--interval-s", type=int, default=60)
    parser.add_argument("--required-ready-samples", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=1440)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--monitor-only",
        action="store_true",
        help="sample resources but never launch the benchmark",
    )
    args = parser.parse_args(argv)
    _install_signal_handlers()

    command_runner = transport.controlmaster_command_runner(
        base_runner=adapter.run_command,
        control_path=args.control_path,
        command_timeouts_s={
            "workers": WORKERS_TIMEOUT_S,
        },
    )
    monitor_output_dir = args.output_root / args.monitor_tag
    sample_counter = (
        monitor.resumable_sample_count(monitor_output_dir)
        if args.resume_existing and monitor_output_dir.exists()
        else 0
    )
    launch_counter = 0
    active_run_tag = args.run_tag

    def sample():
        nonlocal sample_counter
        sample_counter += 1
        return _query_resources(command_runner, sample_counter)

    def launch():
        nonlocal launch_counter, active_run_tag
        if args.monitor_only:
            return {"classification": "MONITOR_ONLY_READY"}
        launch_counter += 1
        active_run_tag = (
            f"{args.run_tag}-attempt{launch_counter:03d}"
        )
        launch_result = benchmark.execute_benchmark_launch(
            mode="canonical",
            run_tag=active_run_tag,
            prerequisites_path=args.prerequisites,
            local_model_manifest=args.local_model_manifest,
            remote_model_dir=args.remote_model_dir,
            remote_model_manifest=args.remote_model_manifest,
            authorization_nonce=_authorization_nonce(),
            output_root=args.output_root,
            preflight_runner=lambda **kwargs: benchmark.run_preflight(
                **kwargs,
                remote_query=lambda: _benchmark_remote_query(
                    command_runner
                ),
                resource_selector=_select_shared_tp4_gpu_resources,
            ),
            command_runner=command_runner,
            resource_policy=(
                benchmark.SHARED_LOW_UTILIZATION_RESOURCE_POLICY
            ),
            maximum_gpu_utilization_percent=(
                MAXIMUM_GPU_UTILIZATION_PERCENT
            ),
        )
        return {
            **launch_result,
            "resource_sharing_policy": RESOURCE_SHARING_POLICY,
            "performance_claim_boundary": PERFORMANCE_CLAIM_BOUNDARY,
        }

    result = monitor.monitor_until_launch(
        monitor_tag=args.monitor_tag,
        output_dir=monitor_output_dir,
        sample_fn=sample,
        launch_fn=launch,
        cleanup_fn=lambda: _cleanup_run(
            command_runner,
            active_run_tag,
        ),
        interval_s=args.interval_s,
        required_ready_samples=args.required_ready_samples,
        max_samples=args.max_samples,
        resume_existing=args.resume_existing,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["classification"] not in {
        "FAILED",
        "CLEANUP_FAILED",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
