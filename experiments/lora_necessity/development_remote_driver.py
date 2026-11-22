import ast
import base64
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import time
import types

import candidate_execution_admission
import development_coordinator as coordinator
import development_final_admission
import development_startup_evidence
from development_coordinator import BoundaryProtocolError, DevelopmentRunError
from observed_generation import _failure, _save, generate_observed
from pilot_protocol import FLAGS
from tokenizer_measurement import _state as tokenizer_state, measure
from validation_contract import digest, encoded, parse_json


LAUNCHER_SHA256 = "e15df5c44094b5cf8bb3fadc40006cab4ec7f4ba01df373f9879f0e63a8cd465"
WORKER_SHA256 = "20d7cd8abb3f09350c01ec47d30613761804dd0849620f808b82606110690896"
CHILD_LIMITS = ("startup_timeout", "candidate_timeout", "output_limit", "result_limit", "reap_timeout")
HOST_ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
MODULE_ORDER = (
    "pilot_protocol", "validation_contract", "candidate_execution_protocol", "candidate_feedback",
    "development_final_protocol", "development_final_scoring", "prompt_rendering",
    "development_schedule", "candidate_execution_controller", "candidate_execution_admission",
    "development_final_controller", "development_final_admission", "tokenizer_measurement",
    "generation_core", "observed_generation", "training_inputs", "training_worker",
    "development_coordinator", "development_startup_evidence", "development_remote_driver",
)
DATA_NAMES = ("supervision_v1.json", "prompts_v1.json", "task_sources_v1.json", "manifest_v1.json")
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false", "USE_TF": "0", "USE_FLAX": "0", "USE_TORCH": "1",
    "WANDB_MODE": "disabled", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


class ExportError(RuntimeError):
    def __init__(self, receipt):
        super().__init__("development evidence export incomplete")
        self.receipt = receipt


class DriverRunError(RuntimeError):
    def __init__(self, envelope):
        super().__init__("development driver aborted; preserved evidence is authoritative")
        self.envelope = envelope


def _need(condition, message):
    if not condition:
        raise BoundaryProtocolError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _snapshot(value):
    return coordinator._native(value)


def _new_state(run_id, deadline_ns):
    return {
        "run_id": run_id, "deadline_ns": deadline_ns, "root": None, "emit": None,
        "frozen": False, "events": [], "callbacks": [], "violations": [], "secondary_errors": [],
        "record_cleanup": {"failed_count": 0, "failures": []},
        "integrity": {"initial": None, "final": None},
        "costs": dict.fromkeys((
            "runtime_ns", "gpu_ns", "checkpoint_ns", "tokenizer_load_ns", "model_load_ns",
            "initial_integrity_ns", "final_integrity_ns", "shared_preparation_ns",
            "coordinator_ns", "work_wall_ns", "driver_wall_ns", "export_ns",
            "export_complete_wall_ns", "terminal_emit_ns",
        )),
    }


def _deadline(state):
    if time.monotonic_ns() >= state["deadline_ns"]:
        raise TimeoutError("development driver deadline")


def _record(root, name, value, state):
    _need(not state["frozen"], "driver persistence frozen for export")
    _save(root, name, _snapshot(value), cleanup=state["record_cleanup"])


def _secondary(state, phase, error):
    state["secondary_errors"].append({"phase": phase, "error": _error_details(error)})


def _violation(state, phase, error):
    state["violations"].append({"phase": phase, "error": _error_details(error)})


def _callback_record(directory, row, state, primary):
    if primary is not None:
        row["failure"] = _error_details(primary)
        if isinstance(primary, BoundaryProtocolError):
            _violation(state, row["phase"], primary)
    try:
        _record(directory, "driver-callback.json", row, state)
    except BaseException as error:
        _secondary(state, "callback_record", error)
        if primary is None:
            raise


def save_measurement(directory, call_id, messages, measured, elapsed_ns, state):
    _record(directory, "driver-measurement.json", {
        "kind": "LORA_DRIVER_MEASUREMENT", "version": 1, "run_id": state["run_id"],
        "call_id": call_id, "messages": messages, "messages_sha256": digest(messages),
        "measurement": measured, "measurement_sha256": digest(measured), "elapsed_ns": elapsed_ns,
    }, state)


def verify_real_generation(result, messages, measured, tokenizer, bindings):
    value = _snapshot(result)
    coordinator._fields(value, ("generation", "observation", "record_cleanup"))
    observed = value["observation"]
    generation = coordinator._generation_values(value["generation"], observed, messages, bindings)
    actual_state, vocabulary = coordinator._checked(tokenizer_state, tokenizer)
    coordinator._same(actual_state, bindings["tokenizer_state"], "real tokenizer binding")
    _need(measured["kind"] == "TOKENIZER_MEASUREMENT_UNATTESTED" and measured["status"] == "OK"
          and measured["target"] is None and measured["state_sha256"] == digest(actual_state),
          "successful measurement identity")
    inputs = measured["input"]
    _need(inputs["messages_sha256"] == digest(messages)
          and inputs["prompt_sha256"] == _sha(inputs["prompt"].encode("utf-8"))
          and inputs["prompt_sha256"] == generation["prompt_sha256"], "measured prompt identity")
    tokens = inputs["input_ids"]
    _need(type(tokens) is list and all(type(token) is int and token in vocabulary for token in tokens)
          and len(tokens) == inputs["input_tokens"] == generation["input_tokens"]
          and inputs["remaining_input_tokens"] == 4096 - len(tokens), "measured input tokens")
    _need(generation["device_type"] == "cuda" and generation["peak_allocated_bytes"] > 0
          and generation["peak_reserved_bytes"] >= generation["peak_allocated_bytes"],
          "observed CUDA memory")
    output = generation["output_ids"]
    _need(all(token in vocabulary for token in output), "output vocabulary membership")
    decoded = tokenizer.decode(
        output[:-1] if generation["stop_reason"] == "eos" else output,
        skip_special_tokens=False, clean_up_tokenization_spaces=False)
    _need(type(decoded) is str and decoded == generation["candidate"]
          and _sha(decoded.encode("utf-8")) == generation["candidate_sha256"], "real output decode")
    coordinator._same(coordinator._checked(tokenizer_state, tokenizer)[0], actual_state,
                      "tokenizer changed during decode")


def verify_startup(stdout_raw, worker):
    _need(type(worker) is bytes and _sha(worker) == WORKER_SHA256, "sealed worker identity")
    _need(type(stdout_raw) is bytes, "startup byte stream")
    lines = stdout_raw.split(b"\n", 4)
    _need(len(lines) == 5, "incomplete startup prefix")
    return coordinator._checked(
        development_startup_evidence.verify_startup, b"\n".join(lines[:4]) + b"\n")


def _collect_execution(directory, row, state, worker, primary):
    stdout = None
    for name in ("admission.json", "claim.json", "observation.json", "receipt.json",
                 "stdout.bin", "stderr.bin", "control.bin"):
        try:
            raw = (directory / name).read_bytes()
            row["files"][name] = {"bytes": len(raw), "sha256": _sha(raw)}
            if name == "stdout.bin":
                stdout = raw
        except FileNotFoundError:
            continue
        except OSError as error:
            row["collection_errors"].append({"file": name, "error": _failure(error)})
            if primary is None:
                raise
    if primary is None:
        row["startup"] = verify_startup(stdout, worker)
    elif stdout is not None and len(stdout.split(b"\n", 4)) == 5:
        try:
            row["startup"] = verify_startup(stdout, worker)
        except BoundaryProtocolError as error:
            row["protocol_error"] = _failure(error)
            _violation(state, row["phase"], error)


def make_callbacks(model, tokenizer, *, bindings, launcher, worker, driver_state):
    declaration = coordinator._bindings(bindings)
    _need(type(launcher) is bytes and _sha(launcher) == LAUNCHER_SHA256, "sealed launcher identity")
    _need(type(worker) is bytes and _sha(worker) == WORKER_SHA256, "sealed worker identity")

    def generate(messages, *, expected_tokenizer_sha256, call_id, artifact_dir):
        row = {
            "phase": "generation", "call_id": call_id, "measurement_entered": False,
            "measurement_elapsed_ns": None, "observer_entered": False,
            "observer_returned": False, "record_cleanup": None, "failure": None,
        }
        driver_state["callbacks"].append(row)
        primary = None
        directory = Path(artifact_dir)
        try:
            _deadline(driver_state)
            snapshot = _snapshot(messages)
            _need(expected_tokenizer_sha256 == digest(declaration["tokenizer_state"]),
                  "callback tokenizer identity")
            started = time.monotonic_ns()
            row["measurement_entered"] = True
            try:
                measured = measure(tokenizer, snapshot, answer=None)
            finally:
                row["measurement_elapsed_ns"] = time.monotonic_ns() - started
            coordinator._same(snapshot, messages, "measurement changed messages")
            save_measurement(directory, call_id, snapshot, measured, row["measurement_elapsed_ns"], driver_state)
            _deadline(driver_state)
            row["observer_entered"] = True
            result = generate_observed(model, tokenizer, snapshot,
                                       expected_tokenizer_sha256=expected_tokenizer_sha256,
                                       call_id=call_id, artifact_dir=directory)
            row["observer_returned"] = True
            row["record_cleanup"] = _snapshot(result["record_cleanup"])
            try:
                coordinator._checked(verify_real_generation, result, messages, measured, tokenizer, declaration)
                for key in ("generation", "observation"):
                    _need((directory / (key + ".json")).read_bytes() == encoded(result[key]),
                          "returned observer record differs from saved bytes")
                coordinator._same(snapshot, messages, "driver callback messages changed")
            except BoundaryProtocolError as error:
                error.record_cleanup = result["record_cleanup"]
                raise
            return result
        except BaseException as error:
            primary = error
            try:
                if getattr(error, "record_cleanup", None) is not None:
                    row["record_cleanup"] = _snapshot(error.record_cleanup)
            except BaseException as metadata_error:
                _secondary(driver_state, "generation_cleanup_metadata", metadata_error)
            raise
        finally:
            _callback_record(directory, row, driver_state, primary)

    def execution(phase, request, artifact_dir):
        row = {"phase": phase, "request_sha256": digest(request), "runner_entered": False,
               "runner_returned": False, "files": {}, "collection_errors": [], "startup": None,
               "failure": None}
        driver_state["callbacks"].append(row)
        primary = None
        directory = Path(artifact_dir)
        try:
            _deadline(driver_state)
            module = candidate_execution_admission if phase == "public" else development_final_admission
            bound = declaration["execution"][phase]
            row["runner_entered"] = True
            result = module.run_candidate_request(
                request, bound["argv"], artifact_dir=directory, cwd=directory,
                env=dict(HOST_ENVIRONMENT), **{name: bound[name] for name in CHILD_LIMITS})
            row["runner_returned"] = True
        except BaseException as error:
            primary = error
        try:
            _collect_execution(directory, row, driver_state, worker, primary)
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                _secondary(driver_state, phase + "_collection", error)
        _callback_record(directory, row, driver_state, primary)
        if primary is not None:
            raise primary
        return result

    def execute_public(request, *, artifact_dir):
        return execution("public", request, artifact_dir)

    def execute_final(request, *, artifact_dir):
        return execution("final", request, artifact_dir)

    return generate, execute_public, execute_final


def _identity(metadata):
    return (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns,
            metadata.st_ctime_ns, stat.S_IMODE(metadata.st_mode))


def _read_regular(root, relative):
    path = root / relative
    _need(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), "export path escape")
    for parent in path.parents:
        if parent == root:
            break
        _need(not parent.is_symlink(), "export parent symlink")
    before = path.lstat()
    _need(stat.S_ISREG(before.st_mode), "export regular file required")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as source:
        _need(_identity(os.fstat(source.fileno())) == _identity(before), "export file replaced")
        raw = source.read()
        _need(_identity(os.fstat(source.fileno())) == _identity(before), "export file changed while reading")
    _need(_identity(path.lstat()) == _identity(before) and len(raw) == before.st_size,
          "export file changed after reading")
    return raw, before


def export_evidence(root, *, run_id, emit):
    root = Path(root)
    _need(coordinator._hex(run_id, 32), "export run identity")
    _need(root.is_dir() and not root.is_symlink(), "owned export directory")
    inventory, errors, identities = [], [], {}
    primary = None
    active_path = "."

    def failed(path, error):
        nonlocal primary
        errors.append({"path": path, "error": _failure(error)})
        if primary is None:
            primary = error

    def walk(directory):
        nonlocal active_path
        active_path = directory.relative_to(root).as_posix()
        try:
            entries = sorted(directory.iterdir())
        except TimeoutError:
            raise
        except OSError as error:
            failed(directory.relative_to(root).as_posix(), error)
            return
        for path in entries:
            relative = path.relative_to(root).as_posix()
            active_path = relative
            try:
                metadata = path.lstat()
                _need(not stat.S_ISLNK(metadata.st_mode), "export symlink")
                if stat.S_ISDIR(metadata.st_mode):
                    walk(path)
                    continue
                raw, metadata = _read_regular(root, relative)
                inventory.append({"path": relative, "bytes": len(raw), "sha256": _sha(raw),
                                  "mode": stat.S_IMODE(metadata.st_mode)})
                identities[relative] = _identity(metadata)
            except TimeoutError:
                raise
            except Exception as error:
                failed(relative, error)

    sent = 0

    def frame(phase, **values):
        emit({"phase": phase, "run_id": run_id, **values})

    try:
        walk(root)
        inventory.sort(key=lambda item: item["path"])
        frame("EXPORT_BEGIN", inventory=inventory, inventory_sha256=digest(inventory),
              errors=_snapshot(errors))
        for item in inventory:
            active_path = item["path"]
            try:
                raw, metadata = _read_regular(root, item["path"])
                _need(_identity(metadata) == identities[item["path"]] and _sha(raw) == item["sha256"],
                      "export inventory changed")
            except TimeoutError:
                raise
            except Exception as error:
                failed(item["path"], error)
                continue
            for index, offset in enumerate(range(0, len(raw), 65536)):
                chunk = raw[offset:offset + 65536]
                frame("FILE_CHUNK", path=item["path"], index=index, offset=offset,
                      bytes=len(chunk), base64=base64.b64encode(chunk).decode("ascii"))
            try:
                current, metadata = _read_regular(root, item["path"])
                _need(_identity(metadata) == identities[item["path"]] and _sha(current) == item["sha256"],
                      "export file changed during emission")
            except TimeoutError:
                raise
            except Exception as error:
                failed(item["path"], error)
                continue
            sent += 1
    except BaseException as error:
        failed(active_path, error)
    inventory.sort(key=lambda item: item["path"])
    receipt = {"status": "incomplete" if errors else "completed", "inventory": inventory,
               "inventory_sha256": digest(inventory), "file_count": len(inventory),
               "exported_files": sent, "errors": errors}
    try:
        frame("EXPORT_END", **_snapshot(receipt))
    except BaseException as error:
        failed("EXPORT_END", error)
        receipt["status"] = "incomplete"
    if primary is not None:
        raise ExportError(receipt) from primary
    return receipt


def _validate_package(package):
    coordinator._fields(package, (
        "kind", "version", "run_id", "runtime_manifest", "limits", "module_order",
        "modules", "data", "launcher", "worker",
    ))
    _need(package["kind"] == "LORA_DEVELOPMENT_PACKAGE" and type(package["version"]) is int
          and package["version"] == 1 and coordinator._hex(package["run_id"], 32), "package identity")
    coordinator._same(package["module_order"], list(MODULE_ORDER), "fixed source order")
    for field, names in (("modules", MODULE_ORDER), ("data", DATA_NAMES)):
        coordinator._fields(package[field], names)
        _need(all(type(raw) is bytes and raw for raw in package[field].values()), "decoded byte snapshots")
    _need(type(package["launcher"]) is bytes and _sha(package["launcher"]) == LAUNCHER_SHA256,
          "sealed launcher pin")
    _need(type(package["worker"]) is bytes and _sha(package["worker"]) == WORKER_SHA256, "sealed worker pin")
    runtime = _snapshot(package["runtime_manifest"])
    coordinator._fields(runtime, (
        "hostname", "uid", "python", "executable", "executable_sha256", "packages", "versions",
        "gpu_uuid", "model_root", "checkpoint_files", "artifact_parent", "expected_base_sha256",
        "expected_tokenizer_sha256",
    ))
    _need(type(runtime["uid"]) is int and runtime["uid"] >= 0 and runtime["python"] == "3.12.13",
          "fixed runtime identity")
    for name in ("hostname", "gpu_uuid"):
        _need(type(runtime[name]) is str and bool(runtime[name]), "runtime text identity")
    for name in ("executable", "packages", "model_root", "artifact_parent"):
        _need(type(runtime[name]) is str and Path(runtime[name]).is_absolute()
              and ".." not in Path(runtime[name]).parts, "absolute runtime path")
    for name in ("executable_sha256", "expected_base_sha256", "expected_tokenizer_sha256"):
        _need(coordinator._hex(runtime[name]), "runtime hash identity")
    _need(type(runtime["versions"]) is dict and {"torch", "transformers"} <= runtime["versions"].keys()
          and all(type(name) is str and type(value) is str
                  for name, value in runtime["versions"].items()), "version inventory")
    _need(type(runtime["checkpoint_files"]) is dict and bool(runtime["checkpoint_files"]), "checkpoint inventory")
    for name, pin in runtime["checkpoint_files"].items():
        _need(type(name) is str and name and not Path(name).is_absolute()
              and ".." not in Path(name).parts, "checkpoint relative path")
        coordinator._fields(pin, ("bytes", "sha256"))
        _need(type(pin["bytes"]) is int and pin["bytes"] >= 0
              and coordinator._hex(pin["sha256"]), "checkpoint pin")
    limits = _snapshot(package["limits"])
    coordinator._fields(limits, (
        "public", "final", "driver_timeout_seconds", "export_reserve_seconds", "transport_timeout_seconds",
    ))
    for phase in ("public", "final"):
        coordinator._same(limits[phase], dict(zip(CHILD_LIMITS, (10, 10, 65536, 16384, 2))),
                          "fixed child limits")
    for name, value in (("driver_timeout_seconds", 3600), ("export_reserve_seconds", 120),
                        ("transport_timeout_seconds", 3900)):
        _need(type(limits[name]) is int and limits[name] == value, "approved driver deadlines")
    return {
        **{name: package[name] for name in ("kind", "version", "run_id", "launcher", "worker")},
        "runtime_manifest": runtime, "limits": limits, "module_order": list(MODULE_ORDER),
        "modules": dict(package["modules"]), "data": dict(package["data"]),
    }


def _bindings_for(package, tokenizer):
    runtime = package["runtime_manifest"]
    token_state = coordinator._checked(tokenizer_state, tokenizer)[0]
    _need(digest(token_state) == runtime["expected_tokenizer_sha256"], "prepared tokenizer identity")
    argv = [
        runtime["executable"], "-I", "-S", "-B", "-c", package["launcher"].decode("utf-8"),
        base64.b64encode(package["worker"]).decode("ascii"), _sha(package["worker"]), "{control_fd}",
    ]
    return coordinator._bindings({
        "component_sha256": {name: _sha(package["modules"][name[:-3]]) for name in coordinator.COMPONENTS},
        "runtime_sha256": digest(runtime), "model_sha256": runtime["expected_base_sha256"],
        "tokenizer_state": token_state,
        "execution": {phase: {"argv": list(argv), **package["limits"][phase]} for phase in ("public", "final")},
    })


def _code_records(code):
    result = {}
    for value in code.co_consts:
        if isinstance(value, types.CodeType):
            result[value.co_qualname] = value
            result.update(_code_records(value))
    return result


def _verify_sources(package, *, deferred=()):
    observed = {}
    for name in MODULE_ORDER:
        if name in deferred:
            continue
        module = sys.modules.get(name)
        _need(module is not None and type(getattr(module, "__file__", None)) is str,
              "loaded source module missing: " + name)
        raw = package["modules"][name]
        filename = module.__file__
        if not filename.startswith("<"):
            _need(Path(filename).read_bytes() == raw, "loaded source file changed: " + name)
        expected = _code_records(compile(raw, filename, "exec", dont_inherit=True))

        for qualified, code in expected.items():
            if "<" in qualified or not code.co_flags & 1:
                continue
            value = module
            for part in qualified.split("."):
                value = vars(value).get(part) if hasattr(value, "__dict__") else None
            if isinstance(value, (staticmethod, classmethod)):
                value = value.__func__
            if name == "development_remote_driver" and qualified == "_alarm":
                wrapped = getattr(value, "__wrapped__", None)
                _need(isinstance(value, types.FunctionType) and isinstance(wrapped, types.FunctionType)
                      and value.__code__ == contextmanager(wrapped).__code__
                      and len(value.__closure__ or ()) == 1
                      and value.__closure__[0].cell_contents is wrapped, "loaded contextmanager changed")
                value = wrapped
            _need(isinstance(value, types.FunctionType) and value.__module__ == name
                  and value.__code__ == code, "loaded function code changed: " + name + "." + qualified)
        for statement in ast.parse(raw, filename).body:
            imports = []
            if isinstance(statement, ast.Import):
                imports = [(alias.name, alias.asname or alias.name, None)
                           for alias in statement.names if alias.name in MODULE_ORDER]
            elif isinstance(statement, ast.ImportFrom) and statement.level == 0 \
                    and statement.module in MODULE_ORDER:
                imports = [(statement.module, alias.asname or alias.name, alias.name)
                           for alias in statement.names]
            for provider_name, alias, attribute in imports:
                provider = sys.modules.get(provider_name)
                _need(provider is not None, "loaded import module missing: " + provider_name)
                expected_value = provider if attribute is None else vars(provider).get(attribute)
                _need(expected_value is not None and vars(module).get(alias) is expected_value,
                      "loaded import binding changed: " + name + "." + alias)
        observed[name] = _sha(raw)
    return observed


def _event(state, phase, **values):
    _deadline(state)
    record = {"phase": phase, "run_id": state["run_id"], **values}
    root = state["root"]
    directory = root / "driver-events"
    directory.mkdir(mode=0o700, exist_ok=True)
    _record(directory, f'{len(state["events"]):03d}-{phase}.json', record, state)
    state["events"].append(record)
    if state["emit"] is not None:
        state["emit"](_snapshot(record))


def _file_record(path, state=None):
    before = path.stat()
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            if state is not None:
                _deadline(state)
            checksum.update(chunk)
        _need(_identity(os.fstat(source.fileno())) == _identity(before), "file changed during hash")
    _need(_identity(path.stat()) == _identity(before), "file changed after hash")
    return {"bytes": before.st_size, "sha256": checksum.hexdigest(), "stat": list(_identity(before))}


def _runtime_facts(manifest):
    return {
        "hostname": socket.gethostname(), "uid": os.getuid(), "python": sys.version.split()[0],
        "executable": sys.executable, "executable_sha256": _file_record(Path(sys.executable))["sha256"],
        "versions": {item.metadata["Name"]: item.version
                     for item in importlib.metadata.distributions(path=[manifest["packages"]])},
        "isolated": bool(sys.flags.isolated), "no_site": bool(sys.flags.no_site),
        "dont_write_bytecode": sys.dont_write_bytecode,
    }


def _gpu_outputs(manifest):
    result = {}
    for name, query in (
        ("gpu", "--query-gpu=index,uuid,name,memory.free,memory.used"),
        ("apps", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory"),
    ):
        try:
            result[name] = subprocess.run(
                ["nvidia-smi", "--id=" + manifest["gpu_uuid"], query, "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=15).stdout
        except BaseException as error:
            error.gpu_outputs = dict(result)
            raise
    return result


def _load_runtime(manifest):
    if manifest["packages"] not in sys.path:
        sys.path.append(manifest["packages"])
    import torch
    import transformers
    for module in (torch, transformers):
        _need(module.__version__ == manifest["versions"][module.__name__]
              and Path(module.__file__).resolve().is_relative_to(Path(manifest["packages"]).resolve()),
              "offline package path/version")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _need(torch.cuda.is_available() and torch.cuda.device_count() == 1
          and torch.cuda.get_device_properties(torch.device("cuda:0")).major == 8, "fixed CUDA device")
    importlib.import_module("training_worker")
    return torch, transformers


def _state_hash(model):
    from training_worker import _state, _tensor_digest
    return _tensor_digest(_state(model))


def _integrity(model, tokenizer, state, *, initial):
    torch = state["_torch"]
    manifest = state["_manifest"]
    parameters = list(model.parameters())
    _need(len(model.model.layers) == 36 and parameters
          and all(parameter.dtype == torch.bfloat16 and parameter.device.type == "cuda"
                  and not parameter.requires_grad and parameter.grad is None for parameter in parameters),
          "frozen BF16 base parameters")
    _need(all(not module.training for module in model.modules())
          and model.config.use_cache is False
          and model.config._attn_implementation == "eager", "base eval/eager/no-cache configuration")
    base = _state_hash(model)
    token_state = coordinator._checked(tokenizer_state, tokenizer)[0]
    _need(base == manifest["expected_base_sha256"]
          and digest(token_state) == manifest["expected_tokenizer_sha256"], "base/tokenizer state drift")
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(torch.device("cuda:0"))
    if initial:
        state["_rng"] = (cpu_rng.clone(), cuda_rng.clone())
    else:
        _need("_rng" in state, "initial RNG evidence unavailable")
        _need(torch.equal(state["_rng"][0], cpu_rng) and torch.equal(state["_rng"][1], cuda_rng),
              "RNG state drift")
    return {"verified": True, "base_sha256": base, "tokenizer_state": token_state, "rng_unchanged": not initial}


def _prepare_base(package, root, state):
    manifest = package["runtime_manifest"]
    state["_manifest"] = manifest
    started = time.monotonic_ns()
    try:
        _deadline(state)
        sources = _verify_sources(package, deferred=("training_inputs", "training_worker"))
        facts = _runtime_facts(manifest)
        _record(root, "runtime-observation.json", facts, state)
        for name in ("hostname", "uid", "python", "executable", "executable_sha256", "versions"):
            coordinator._same(facts[name], manifest[name], "runtime identity: " + name)
        _need(all(facts[name] is True for name in ("isolated", "no_site", "dont_write_bytecode")),
              "isolated interpreter flags")
        os.environ.update({**OFFLINE_ENVIRONMENT, "CUDA_VISIBLE_DEVICES": manifest["gpu_uuid"]})
    finally:
        state["costs"]["runtime_ns"] = time.monotonic_ns() - started
    _event(state, "RUNTIME_VERIFIED", facts=facts, source_sha256=sources)
    started = time.monotonic_ns()
    try:
        try:
            gpu = _gpu_outputs(manifest)
        except BaseException as error:
            try:
                _record(root, "gpu-before.json", getattr(error, "gpu_outputs", {}), state)
            except BaseException as record_error:
                _secondary(state, "gpu_observation", record_error)
            raise
        _record(root, "gpu-before.json", gpu, state)
        rows = [line.split(",") for line in gpu["gpu"].strip().splitlines()]
        _need(len(rows) == 1 and len(rows[0]) == 5, "single GPU observation")
        index, uuid, name, free, used = [value.strip() for value in rows[0]]
        _need(uuid == manifest["gpu_uuid"] and "A100" in name and int(free) >= 65536
              and int(used) <= 512 and not gpu["apps"].strip(), "fixed GPU identity/idle state")
    finally:
        state["costs"]["gpu_ns"] = time.monotonic_ns() - started
    _event(state, "GPU_VERIFIED", gpu=gpu)
    started = time.monotonic_ns()
    checkpoint = Path(manifest["model_root"])
    records = {}
    try:
        for name, pin in manifest["checkpoint_files"].items():
            path = checkpoint / name
            _need(path.resolve().is_relative_to(checkpoint.resolve()), "checkpoint path escape")
            records[name] = _file_record(path, state)
            coordinator._same({key: records[name][key] for key in ("bytes", "sha256")}, pin, "checkpoint pin: " + name)
    finally:
        checkpoint_error = sys.exception()
        try:
            _record(root, "checkpoint-observation.json", records, state)
        except BaseException as error:
            if checkpoint_error is None:
                raise
            _secondary(state, "checkpoint_observation", error)
        finally:
            state["costs"]["checkpoint_ns"] = time.monotonic_ns() - started
    _deadline(state)
    torch, transformers = _load_runtime(manifest)
    state["_torch"] = torch
    sources = _verify_sources(package)
    _record(root, "loaded-source-observation.json", sources, state)
    started = time.monotonic_ns()
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(checkpoint), local_files_only=True, trust_remote_code=False)
        state["_tokenizer"] = tokenizer
    finally:
        state["costs"]["tokenizer_load_ns"] = time.monotonic_ns() - started
    _deadline(state)
    started = time.monotonic_ns()
    try:
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats(device)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            str(checkpoint), local_files_only=True, trust_remote_code=False, use_safetensors=True,
            torch_dtype=torch.bfloat16, attn_implementation="eager")
        state["_model"] = model
        model = model.to(device).eval().requires_grad_(False)
        state["_model"] = model
        model.config.use_cache = False
        torch.cuda.synchronize(device)
    finally:
        state["costs"]["model_load_ns"] = time.monotonic_ns() - started
    state["costs"]["model_load_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    state["costs"]["model_load_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    for name, record in records.items():
        _need(list(_identity((checkpoint / name).stat())) == record["stat"], "checkpoint changed after load")
    started = time.monotonic_ns()
    try:
        state["integrity"]["initial"] = coordinator._checked(_integrity, model, tokenizer, state, initial=True)
    finally:
        state["costs"]["initial_integrity_ns"] = time.monotonic_ns() - started
    _event(state, "MODEL_VERIFIED", integrity=state["integrity"]["initial"])
    return model, tokenizer


def _final_integrity(model, tokenizer, state):
    _deadline(state)
    return coordinator._checked(_integrity, model, tokenizer, state, initial=False)


@contextmanager
def _alarm(deadline_ns):
    remaining = (deadline_ns - time.monotonic_ns()) / 1e9
    if remaining <= 0:
        raise TimeoutError("development deadline exhausted")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def expired(signum, frame):
        raise TimeoutError("development deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        prior = previous_timer[0] - (time.monotonic() - started)
        if previous_timer[0] and prior > 0:
            signal.setitimer(signal.ITIMER_REAL, prior, previous_timer[1])


def _error_details(error):
    seen = set()
    remaining = 32

    def visit(current, depth):
        nonlocal remaining
        result = _failure(current)
        if id(current) in seen or depth >= 8 or remaining == 0:
            result["truncated"] = "cycle" if id(current) in seen else "limit"
            return result
        seen.add(id(current))
        remaining -= 1
        for name in ("record_cleanup", "cleanup_errors", "record_cleanup_complete", "persistence_errors"):
            try:
                value = getattr(current, name, None)
                if value is not None:
                    result[name] = _snapshot(value)
            except BaseException as metadata_error:
                result[name + "_error"] = _failure(metadata_error)
        for attribute, field in (("__cause__", "cause"), ("observation_error", "observation_error")):
            try:
                nested = getattr(current, attribute, None)
                if nested is not None:
                    result[field] = visit(nested, depth + 1)
            except BaseException as metadata_error:
                result[field + "_error"] = _failure(metadata_error)
        return result

    return visit(error, 0)


def _coordinator_envelope(error):
    return _snapshot({name: getattr(error, name) for name in (
        "report", "terminal_path", "terminal_sha256", "record_cleanup",
        "record_cleanup_complete", "persistence_errors",
    )})


def run_driver(package, *, artifact_root, deadline_ns, emit):
    started = time.monotonic_ns()
    run_id = package.get("run_id") if type(package) is dict else None
    state = _new_state(run_id if coordinator._hex(run_id, 32) else None, deadline_ns)
    state["emit"] = emit
    primary = None
    result = None
    owned = False
    terminal_path = None
    terminal_sha256 = None
    try:
        snapshot = _validate_package(package)
        _need(type(deadline_ns) is int and deadline_ns > 0 and callable(emit), "driver deadline/emitter")
        _deadline(state)
        root = Path(snapshot["runtime_manifest"]["artifact_parent"]) / ("development-" + run_id)
        _need(Path(artifact_root) == root, "fixed artifact root")
        root.mkdir(mode=0o700)
        state["root"] = root
        owned = True
        _record(root, "driver-claim.json", {
            "kind": "LORA_DEVELOPMENT_DRIVER_CLAIM", "version": 1, "run_id": run_id,
            "runtime_manifest": snapshot["runtime_manifest"], "limits": snapshot["limits"],
            "deadline_ns": deadline_ns, "source_sha256": {name: _sha(raw) for name, raw in snapshot["modules"].items()},
            "data_sha256": {name: _sha(raw) for name, raw in snapshot["data"].items()},
            "launcher_sha256": _sha(snapshot["launcher"]), "worker_sha256": _sha(snapshot["worker"]),
        }, state)
        with _alarm(deadline_ns):
            _event(state, "STARTED", artifact_root=str(root))
            setup_started = time.monotonic_ns()
            try:
                model, tokenizer = _prepare_base(snapshot, root, state)
                state.update(_model=model, _tokenizer=tokenizer)
            finally:
                state["costs"]["shared_preparation_ns"] = time.monotonic_ns() - setup_started
            bindings = _bindings_for(snapshot, tokenizer)
            callbacks = make_callbacks(model, tokenizer, bindings=bindings, launcher=snapshot["launcher"],
                                       worker=snapshot["worker"], driver_state=state)
            _deadline(state)
            coordinator_started = time.monotonic_ns()
            try:
                result = coordinator.run_development(
                    *(snapshot["data"][name] for name in DATA_NAMES), run_id=run_id,
                    artifact_dir=root / "coordinator", bindings=bindings,
                    generate=callbacks[0], execute_public=callbacks[1], execute_final=callbacks[2])
            except DevelopmentRunError as error:
                result = _coordinator_envelope(error)
                raise
            finally:
                state["costs"]["coordinator_ns"] = time.monotonic_ns() - coordinator_started
                coordinator_error = sys.exception()
                try:
                    _event(state, "COORDINATOR_ENTERED")
                except BaseException as error:
                    if coordinator_error is None:
                        raise
                    _secondary(state, "coordinator_event", error)
    except BaseException as error:
        primary = error
    if "_model" in state and "_tokenizer" in state:
        integrity_started = time.monotonic_ns()
        try:
            with _alarm(deadline_ns):
                state["integrity"]["final"] = _final_integrity(state["_model"], state["_tokenizer"], state)
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                _secondary(state, "final_integrity", error)
        finally:
            state["costs"]["final_integrity_ns"] = time.monotonic_ns() - integrity_started
    state["costs"]["work_wall_ns"] = time.monotonic_ns() - started
    envelope = {
        "run_id": state["run_id"], "status": "completed" if primary is None else "aborted",
        "coordinator": result, "integrity": {**state["integrity"], "violations": state["violations"]},
        "costs": {**state["costs"], "callbacks": state["callbacks"]},
        "failure": None if primary is None else _error_details(primary),
        "secondary_errors": state["secondary_errors"], "record_cleanup": state["record_cleanup"],
        "export": None, "training_steps": 0, "adapter_mounted": False, **dict.fromkeys(FLAGS, False),
    }
    if owned:
        for name in (("driver-result.json", "driver-failure.json") if primary is None else ("driver-failure.json",)):
            try:
                saved = _snapshot(envelope)
                _record(root, name, saved, state)
                terminal_path, terminal_sha256 = name, digest(saved)
                break
            except BaseException as error:
                if primary is None:
                    primary = error
                    envelope.update(status="aborted", failure=_error_details(error))
                else:
                    _secondary(state, "driver_terminal", error)
        state["frozen"] = True
        export_started = time.monotonic_ns()
        try:
            export_deadline = time.monotonic_ns() + snapshot["limits"]["export_reserve_seconds"] * 10 ** 9
            with _alarm(export_deadline):
                def export_emit(frame):
                    emit({**frame, "terminal_path": terminal_path, "terminal_sha256": terminal_sha256})
                envelope["export"] = export_evidence(root, run_id=run_id, emit=export_emit)
        except BaseException as error:
            if isinstance(error, ExportError):
                envelope["export"] = error.receipt
                export_error = error.__cause__ or error
            else:
                export_error = error
                envelope["export"] = {"status": "incomplete", "errors": [_failure(error)]}
            if primary is None:
                primary = export_error
                envelope.update(status="aborted", failure=_error_details(primary))
            else:
                _secondary(state, "export", export_error)
        finally:
            envelope["costs"]["export_ns"] = time.monotonic_ns() - export_started
    envelope["costs"]["export_complete_wall_ns"] = time.monotonic_ns() - started
    envelope.update(terminal_path=terminal_path, terminal_sha256=terminal_sha256)
    if owned:
        phase = "DONE" if primary is None and envelope["export"]["status"] == "completed" else \
            "EXPORT_INCOMPLETE" if envelope["export"]["status"] != "completed" else "FAILED"
        emit_started = time.monotonic_ns()
        try:
            emit({"phase": phase, "run_id": run_id, "envelope": _snapshot(envelope)})
        except BaseException as error:
            if primary is None:
                primary = error
                envelope.update(status="aborted", failure=_error_details(error))
            else:
                _secondary(state, "terminal_emit", error)
        finally:
            envelope["costs"]["terminal_emit_ns"] = time.monotonic_ns() - emit_started
    envelope["costs"]["driver_wall_ns"] = time.monotonic_ns() - started
    envelope = _snapshot(envelope)
    if primary is not None:
        raise DriverRunError(envelope) from primary
    return envelope
