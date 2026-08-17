from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def _load_module(module_name, filename):
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_module(
    "run_qwen35_tp4_real_root_logit_gate_remote_for_shared_stage",
    "run_qwen35_tp4_real_root_logit_gate_remote.py",
)
resource_policy = _load_module(
    "qwen35_tp4_correctness_resource_policy_for_shared_root_stage",
    "qwen35_tp4_correctness_resource_policy.py",
)


REQUIRED_EXECUTION_ENV = {
    "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian",
}


def _run(command, **kwargs):
    environment = dict(os.environ)
    environment.update(REQUIRED_EXECUTION_ENV)
    return subprocess.run(
        command,
        check=False,
        env=environment,
        **kwargs,
    )


def _require_success(result, label):
    if getattr(result, "returncode", None) != 0:
        detail = getattr(result, "stderr", "") or getattr(
            result,
            "stdout",
            "",
        )
        raise RuntimeError(f"{label} failed: {str(detail).strip()}")
    return result


def _remote_baseline_path(plan):
    return (
        f"{runner.REMOTE_GATE_ROOT}/."
        f"{plan['run_tag']}.resource-baseline.json"
    )


def _guard_command(plan):
    return runner.build_ssh_command(
        resource_policy.guard_command(
            resource_policy.CONTROLLED_SHARED,
            plan["gpu_indices"],
            baseline_path=_remote_baseline_path(plan),
            baseline_sha256=plan["resource_baseline_sha256"],
            ssh_target=runner.REMOTE_TARGET,
        )
    )


def _load_guard(plan, result):
    _require_success(result, "controlled-shared resource guard")
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("controlled-shared guard JSON is invalid") from error
    baseline = resource_policy.validate_baseline_manifest(
        plan["resource_baseline_path"],
        ssh_target=runner.REMOTE_TARGET,
        gpu_indices=plan["gpu_indices"],
    )
    resource_policy.validate_guard_payload(
        resource_policy.CONTROLLED_SHARED,
        payload,
        gpu_indices=plan["gpu_indices"],
        baseline=baseline,
        baseline_sha256=plan["resource_baseline_sha256"],
    )
    return payload


def _run_guard(plan, command_runner):
    for _ in range(3):
        result = command_runner(
            _guard_command(plan),
            text=True,
            capture_output=True,
        )
        if getattr(result, "returncode", None) == 0:
            break
    return _load_guard(plan, result)


def _selected_rows(plan, guard):
    rows = []
    for rank, row in enumerate(guard["selected"]):
        value = dict(row)
        value.update({
            "rank": rank,
            "world_size": 4,
            "minimum_free_bytes": resource_policy.MIN_GPU_FREE_BYTES,
        })
        rows.append(value)
    if [row["gpu_uuid"] for row in rows] != plan["gpu_uuids"]:
        raise ValueError("controlled-shared root GPU UUID drift")
    return rows


def _logical_query_rows(guard):
    return [
        {
            **row,
            "compute_processes": [],
        }
        for row in guard["selected"]
    ]


def _run_cases_with_float32_root_logits(
    *,
    module,
    candidate,
    rank,
    **kwargs,
):
    def canonicalize_results(results):
        canonical = []
        for result in results:
            state = getattr(result, "state_nonzero_after_commit", None)
            if not isinstance(state, dict):
                raise ValueError("native state evidence must be a mapping")
            roles_by_layer = {}
            for key, value in state.items():
                if (
                    not isinstance(key, str)
                    or ":" not in key
                    or value is not True
                ):
                    raise ValueError("native state evidence is invalid")
                layer_text, role = key.split(":", 1)
                try:
                    layer_index = int(layer_text)
                except ValueError as error:
                    raise ValueError(
                        "native state layer index is invalid"
                    ) from error
                roles_by_layer.setdefault(layer_index, set()).add(role)
            expected_roles = {
                "linear_convolution",
                "linear_recurrent",
            }
            physical_layers = tuple(sorted(roles_by_layer))
            if (
                len(physical_layers) != 18
                or any(
                    roles != expected_roles
                    for roles in roles_by_layer.values()
                )
            ):
                raise ValueError(
                    "native state layer inventory is invalid"
                )
            logical_state = {
                f"{logical_layer}:{role}": state[
                    f"{physical_layer}:{role}"
                ]
                for logical_layer, physical_layer in enumerate(
                    physical_layers
                )
                for role in sorted(expected_roles)
            }
            canonical.append(replace(
                result,
                state_nonzero_after_commit=logical_state,
            ))
        return tuple(canonical)

    if rank != 0:
        return canonicalize_results(module.run_tp4_native_cases(
            candidate=candidate,
            rank=rank,
            **kwargs,
        ))
    model = getattr(getattr(candidate, "owner", None), "model", None)
    lm_head = getattr(model, "lm_head", None)
    register_forward_hook = getattr(lm_head, "register_forward_hook", None)
    if not callable(register_forward_hook):
        raise ValueError("rank zero lm_head forward hook is unavailable")

    def canonicalize_logits(_lm_head, _inputs, logits):
        if logits is None:
            return None
        return logits.to(dtype=module.torch.float32)

    hook = register_forward_hook(canonicalize_logits)
    try:
        return canonicalize_results(module.run_tp4_native_cases(
            candidate=candidate,
            rank=rank,
            **kwargs,
        ))
    finally:
        hook.remove()


def _wrapper_script(plan, guard):
    query_rows = _logical_query_rows(guard)
    return "\n".join([
        "import importlib.util,json,os,subprocess,sys,traceback",
        "from dataclasses import replace",
        "from pathlib import Path",
        f"gate_path=Path({runner.FROZEN_PREFLIGHT!r})",
        f"source_root=Path({runner.FROZEN_SOURCE_ROOT!r})",
        f"manifest_path=Path({runner.FROZEN_MANIFEST!r})",
        f"run_dir=Path({plan['remote_run_dir']!r})",
        f"run_tag={plan['run_tag']!r}",
        f"query_rows={query_rows!r}",
        f"diagnostic_root=Path({runner.REMOTE_GATE_ROOT!r})",
        "diagnostic_paths=[diagnostic_root/f'.{run_tag}.native-rank-failure-{rank}.log' for rank in range(4)]",
        "comparison_diagnostic_path=diagnostic_root/f'.{run_tag}.comparison-diagnostics.json'",
        "spec=importlib.util.spec_from_file_location('controlled_shared_frozen_root',gate_path)",
        "module=importlib.util.module_from_spec(spec)",
        "sys.modules[spec.name]=module",
        "spec.loader.exec_module(module)",
        "comparison_rows=[]",
        "original_compare_logits=module._TP4_CONTRACT.compare_logits",
        "original_classify_rows=module._TP4_CONTRACT.classify_rows",
        "def observed_compare_logits(*args,**kwargs):",
        " row=original_compare_logits(*args,**kwargs)",
        " comparison_rows.append(dict(row))",
        " return row",
        "def observed_classify_rows(rows):",
        " classification=original_classify_rows(rows)",
        " comparison_diagnostic_path.write_text(json.dumps({'comparisons':comparison_rows,'classification':classification},sort_keys=True,separators=(',',':')),encoding='utf-8')",
        " return classification",
        "module._TP4_CONTRACT.compare_logits=observed_compare_logits",
        "module._TP4_CONTRACT.classify_rows=observed_classify_rows",
        "def query_gpus(): return tuple(dict(row) for row in query_rows)",
        "def process_factory_builder(**builder_kwargs):",
        " def process_factory(**kwargs):",
        "  rank=kwargs['rank']",
        "  command=[sys.executable,'-c',SCRIPT,'internal-native-rank','--rank-output',os.fspath(builder_kwargs['work_dir']/f'rank-{rank}.json.partial')]",
        "  logits=builder_kwargs['work_dir']/('native_rank0_logits.pt.partial')",
        "  if rank==0: command.extend(['--logits-output',os.fspath(logits)])",
        "  environment=dict(kwargs['environment'])",
        "  environment.update({'TINYVLLM_GATE_LOCAL_RANK':str(rank),'TINYVLLM_GATE_PHYSICAL_GPU_INDEX':str(kwargs['gpu_index']),'TINYVLLM_GATE_GPU_UUID':kwargs['gpu_uuid'],'TINYVLLM_GATE_PROCESS_GROUP_NONCE':kwargs['process_group_nonce'],'TINYVLLM_GATE_RENDEZVOUS':kwargs['rendezvous'],'TINYVLLM_GATE_DIAGNOSTIC_PATH':os.fspath(diagnostic_paths[rank])})",
        "  return module._DeferredSubprocess(command=command,work_dir=builder_kwargs['work_dir'],environment=environment,popen=subprocess.Popen)",
        " return process_factory",
        "def build_candidate(rank): return module.build_real_tp4_cpu_candidate(rank=rank)",
        "def canonicalize_results(results):",
        " canonical=[]",
        " for result in results:",
        "  state=result.state_nonzero_after_commit",
        "  roles_by_layer={}",
        "  for key,value in state.items():",
        "   if not isinstance(key,str) or ':' not in key or value is not True: raise ValueError('native state evidence is invalid')",
        "   layer_text,role=key.split(':',1)",
        "   try: layer_index=int(layer_text)",
        "   except ValueError as error: raise ValueError('native state layer index is invalid') from error",
        "   roles_by_layer.setdefault(layer_index,set()).add(role)",
        "  expected_roles={'linear_convolution','linear_recurrent'}",
        "  physical_layers=tuple(sorted(roles_by_layer))",
        "  if len(physical_layers)!=18 or any(roles!=expected_roles for roles in roles_by_layer.values()): raise ValueError('native state layer inventory is invalid')",
        "  logical_state={f'{logical_layer}:{role}':state[f'{physical_layer}:{role}'] for logical_layer,physical_layer in enumerate(physical_layers) for role in sorted(expected_roles)}",
        "  canonical.append(replace(result,state_nonzero_after_commit=logical_state))",
        " return tuple(canonical)",
        "def run_cases(**kwargs):",
        " if kwargs['rank']!=0: return canonicalize_results(module.run_tp4_native_cases(**kwargs))",
        " model=kwargs['candidate'].owner.model",
        " def canonicalize_logits(_lm_head,_inputs,logits):",
        "  if logits is None: return None",
        "  return logits.to(dtype=module.torch.float32)",
        " hook=model.lm_head.register_forward_hook(canonicalize_logits)",
        " try: return canonicalize_results(module.run_tp4_native_cases(**kwargs))",
        " finally: hook.remove()",
        "def native(**kwargs): return module.execute_native_rank_worker(**kwargs,build_candidate=build_candidate,run_cases=run_cases,query_gpus=query_gpus)",
        "mode=sys.argv[1]",
        "if mode=='run':",
        " for path in diagnostic_paths:",
        "  if path.exists(): path.unlink()",
        " if comparison_diagnostic_path.exists(): comparison_diagnostic_path.unlink()",
        " try:",
        "  result=module.execute_source_bound_run(run_dir=run_dir,run_tag=run_tag,source_manifest_path=manifest_path,source_root=source_root,query_gpus=query_gpus,process_factory_builder=process_factory_builder)",
        " except BaseException as error:",
        "  details='\\n'.join(path.read_text(encoding='utf-8',errors='replace') for path in diagnostic_paths if path.is_file())",
        "  comparison_details=comparison_diagnostic_path.read_text(encoding='utf-8',errors='replace') if comparison_diagnostic_path.is_file() else ''",
        "  raise RuntimeError('native rank diagnostics:\\n'+details+'\\ncomparison diagnostics:\\n'+comparison_details) from error",
        " finally:",
        "  for path in diagnostic_paths:",
        "   if path.exists(): path.unlink()",
        "  if comparison_diagnostic_path.exists(): comparison_diagnostic_path.unlink()",
        " print(json.dumps({'classification':result['classification'],'artifact_names':sorted(Path(value).name for value in result['paths'])},sort_keys=True,separators=(',',':')))",
        "elif mode=='internal-native-rank':",
        " try:",
        "  exit_code=module.main(sys.argv[1:],execute_native_rank=native)",
        " except BaseException:",
        "  Path(os.environ['TINYVLLM_GATE_DIAGNOSTIC_PATH']).write_text(traceback.format_exc(),encoding='utf-8')",
        "  raise",
        " raise SystemExit(exit_code)",
        "else: raise SystemExit('unsupported controlled-shared wrapper mode')",
    ])


def _run_preflight(plan, command_runner):
    local_run = Path(plan["local_run_dir"])
    remote_baseline = _remote_baseline_path(plan)
    reserve_command = runner.build_ssh_command([
        "bash",
        "-lc",
        " && ".join([
            "set -eu",
            f"test ! -e {shlex.quote(plan['remote_run_dir'])}",
            f"test ! -e {shlex.quote(remote_baseline)}",
        ]),
    ])
    for _ in range(3):
        reserve = command_runner(
            reserve_command,
            text=True,
            capture_output=True,
        )
        if getattr(reserve, "returncode", None) == 0:
            break
    _require_success(reserve, "controlled-shared root reservation")
    upload_command = [
        "scp",
        "-o",
        "BatchMode=yes",
        "-o",
        "ControlMaster=no",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        plan["resource_baseline_path"],
        f"{runner.REMOTE_TARGET}:{remote_baseline}",
    ]
    for _ in range(3):
        upload = command_runner(
            upload_command,
            text=True,
            capture_output=True,
        )
        if getattr(upload, "returncode", None) == 0:
            break
    _require_success(upload, "controlled-shared baseline upload")
    guard = _run_guard(plan, command_runner)
    selected = _selected_rows(plan, guard)
    evidence = {
        "run_tag": plan["run_tag"],
        "frozen_source_tag": plan["frozen_source_tag"],
        "frozen_source_tree_sha256": (
            plan["frozen_source_tree_sha256"]
        ),
        "status": "READY",
        "source_tree_sha256": plan["frozen_source_tree_sha256"],
        "selected": selected,
        "rows": selected,
        "resource_policy": resource_policy.CONTROLLED_SHARED,
        "baseline_sha256": plan["resource_baseline_sha256"],
        "benchmark_execution_authorized": False,
    }
    local_run.mkdir(parents=True)
    (local_run / "remote_resource_preflight.json").write_text(
        json.dumps(evidence, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence


def _run_authority(plan, command_runner):
    local_run = Path(plan["local_run_dir"])
    if not (local_run / "remote_resource_preflight.json").is_file():
        raise ValueError("controlled-shared root preflight is missing")
    guard = _run_guard(plan, command_runner)
    script = _wrapper_script(plan, guard)
    global_script = f"SCRIPT={script!r}\n{script}"
    result = command_runner(
        runner.build_ssh_command([
            "env",
            f"PYTHONPATH={runner.FROZEN_SOURCE_ROOT}",
            runner.REMOTE_PYTHON,
            "-c",
            global_script,
            "run",
        ]),
        text=True,
        capture_output=True,
    )
    _require_success(result, "controlled-shared root authority")
    try:
        authority = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            "controlled-shared root authority JSON is invalid"
        ) from error
    final_resource = _run_guard(plan, command_runner)
    if (
        authority.get("classification") != "PASS"
        or authority.get("artifact_names")
        != sorted(plan["exact_artifact_names"])
    ):
        raise ValueError(
            "controlled-shared root authority inventory is invalid"
        )
    evidence = {
        "status": "REMOTE_PASS",
        "run_tag": plan["run_tag"],
        "remote_run_dir": plan["remote_run_dir"],
        "artifact_names": sorted(plan["exact_artifact_names"]),
        "final_resource": final_resource,
    }
    (local_run / "remote_run.json").write_text(
        json.dumps(evidence, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence


def run_stage(
    *,
    name,
    plan,
    execution_env,
    command_runner=_run,
):
    if execution_env != REQUIRED_EXECUTION_ENV:
        raise ValueError(
            "exact KRB5CCNAME execution environment is required"
        )
    if plan.get("resource_policy") != resource_policy.CONTROLLED_SHARED:
        raise ValueError("controlled-shared root plan is required")
    if name == "preflight":
        return _run_preflight(plan, command_runner)
    if name == "run":
        return _run_authority(plan, command_runner)
    if name == "download":
        return runner.execute_download(
            run_tag=plan["run_tag"],
            repo_root=plan["repo_root"],
            command_runner=command_runner,
        )
    if name == "verify":
        return runner.execute_verify(
            run_tag=plan["run_tag"],
            repo_root=plan["repo_root"],
            command_runner=command_runner,
        )
    raise ValueError("controlled-shared root stage is unsupported")
