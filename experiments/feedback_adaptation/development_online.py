import copy
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

import development_cpu as cpu
from active_verification_contract import canonical, fields, natural, require
from development_protocol import (
    checksum, manifest_digest, model_request, next_step, score, tool_request,
)


SOURCE_ROOT = Path(__file__).resolve().parent
SOURCE_NAMES = (
    "development_online.py", "development_protocol.py", "development_cpu.py",
    "development_tasks.py", "development_programs.py", "active_verification_contract.py",
)
LOADED_HASHES = {
    name: hashlib.sha256((SOURCE_ROOT / name).read_bytes()).hexdigest() for name in SOURCE_NAMES
}


def write_json(path, value):
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def strict_json(data):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            require(name not in result, "duplicate JSON key")
            result[name] = value
        return result
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


class ModelProcess:
    evidence_kind = "subprocess_model_unverified"

    def __init__(self, argv, output_dir, timeout_seconds=600, expected_ready=None):
        self.argv = list(argv)
        self.output_dir = Path(output_dir)
        self.timeout_seconds = timeout_seconds
        self.expected_ready = dict(expected_ready or {})
        self.process = None
        self.selector = None
        self.raw = {}
        self.sizes = {"stdout": 0, "stderr": 0}
        self.buffer = b""
        self.stdout_closed = False
        self.calls = 0
        self.finished = False

    def start(self):
        require(self.process is None, "model cannot restart")
        self.output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        write_json(self.output_dir / "invocation.json", {"argv": self.argv, "timeout_seconds": self.timeout_seconds})
        self.deadline = time.monotonic() + self.timeout_seconds
        self.raw = {name: (self.output_dir / (name + ".raw")).open("xb") for name in self.sizes}
        self.process = subprocess.Popen(
            self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, bufsize=0,
        )
        self.selector = selectors.DefaultSelector()
        for name in ("stdout", "stderr"):
            stream = getattr(self.process, name)
            os.set_blocking(stream.fileno(), False)
            self.selector.register(stream, selectors.EVENT_READ, name)
        os.set_blocking(self.process.stdin.fileno(), False)
        ready = self._read(min(self.deadline, time.monotonic() + 120))
        require(ready.get("event") == "ready", "model readiness failed")
        for name, expected in self.expected_ready.items():
            require(canonical(ready.get(name)) == canonical(expected), "ready identity mismatch: " + name)
        return ready

    def _read(self, deadline):
        while True:
            if b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                require(len(line) <= 1024 * 1024, "model line cap")
                value = strict_json(line)
                require(type(value) is dict, "model record must be object")
                return value
            require(len(self.buffer) <= 1024 * 1024, "model line cap")
            if self.stdout_closed:
                require(not self.buffer, "partial model JSON line")
                raise EOFError("model stdout closed")
            remaining = min(deadline, self.deadline) - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("model transport deadline")
            self._pump(remaining)

    def _pump(self, timeout):
        for key, _mask in self.selector.select(timeout):
            chunk = os.read(key.fileobj.fileno(), 65536)
            name = key.data
            if not chunk:
                self.selector.unregister(key.fileobj)
                if name == "stdout":
                    self.stdout_closed = True
                continue
            self.sizes[name] += len(chunk)
            self.raw[name].write(chunk)
            self.raw[name].flush()
            require(self.sizes[name] <= 16 * 1024 * 1024, "model stream cap")
            if name == "stdout":
                self.buffer += chunk

    def _drain(self, deadline):
        while self.selector is not None and self.selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("model stream drain deadline")
            self._pump(remaining)

    def __call__(self, request):
        require(self.process is not None and not self.finished, "model not running")
        require(self.calls < 144, "model call budget")
        data = canonical(request) + b"\n"
        require(len(data) <= 131072, "model request line cap")
        deadline = min(self.deadline, time.monotonic() + 30)
        with selectors.DefaultSelector() as writable:
            writable.register(self.process.stdin, selectors.EVENT_WRITE)
            while data:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not writable.select(remaining):
                    raise TimeoutError("model write deadline")
                written = os.write(self.process.stdin.fileno(), data)
                data = data[written:]
        self.calls += 1
        response = self._read(deadline)
        require(response.get("event") == "output", "unexpected model response")
        return response

    def finish(self):
        require(self.process is not None and not self.finished, "model not running")
        self.process.stdin.close()
        response = self._read(min(self.deadline, time.monotonic() + 30))
        require(response.get("event") == "finished" and type(response.get("calls")) is int
                and response["calls"] == self.calls, "model finish mismatch")
        self._drain(min(self.deadline, time.monotonic() + 5))
        require(not self.buffer, "extra model bytes after finish")
        self.process.wait(timeout=max(0.01, min(5, self.deadline - time.monotonic())))
        require(self.process.returncode == 0, "model process failed")
        self.finished = True
        return response

    def close(self):
        try:
            if self.process is not None:
                deadline = time.monotonic() + 2
                if self.process.poll() is None:
                    try:
                        os.killpg(self.process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                try:
                    self._drain(deadline)
                    try:
                        self.process.wait(timeout=max(0.001, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        pass
                finally:
                    if self.process.poll() is None:
                        try:
                            os.killpg(self.process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        self.process.wait(timeout=2)
                    self._drain(time.monotonic() + 2)
        finally:
            if self.process is not None:
                for name in ("stdin", "stdout", "stderr"):
                    getattr(self.process, name).close()
            if self.selector is not None:
                self.selector.close()
            for stream in self.raw.values():
                if not stream.closed:
                    try:
                        stream.flush()
                        os.fsync(stream.fileno())
                    finally:
                        stream.close()


def model_result(request, response, elapsed):
    fields(response, ("event", "id", "request_sha256", "text", "terminated", "input_ids",
                      "output_ids", "generation_ns", "call_ns", "prompt_sha256"))
    require(response["event"] == "output" and response["id"] == request["id"]
            and response["request_sha256"] == checksum(request), "response binding")
    for name in ("generation_ns", "call_ns"):
        natural(response[name])
    require(response["generation_ns"] <= response["call_ns"], "model timing order")
    require(type(response["prompt_sha256"]) is str and len(response["prompt_sha256"]) == 64
            and all(char in "0123456789abcdef" for char in response["prompt_sha256"]), "prompt digest")
    return {
        "kind": "model", "status": "completed", "wall_ns": elapsed,
        "request_id": request["id"], "request_sha256": checksum(request),
        **{name: response[name] for name in ("text", "terminated", "input_ids", "output_ids")},
    }


def run_probe(output_dir, cell, method, run_id):
    started = time.perf_counter_ns()
    event = {
        "kind": "tool", "request_id": tool_request(cell, method, run_id)["request_id"],
        "status": "unavailable", "results": [], "executed_cases": None, "wall_ns": None,
    }
    try:
        result = cpu.run(output_dir, "probe", cell["task_id"], cell["initial"])
        if result["status"] == "completed":
            probe_manifest = strict_json((output_dir / "manifest.json").read_bytes())
            observation = strict_json((output_dir / "stdout.json").read_bytes())
            comparison = cpu.compare(probe_manifest, observation)
            require(canonical(comparison) == canonical(result["comparison"]), "probe result drift")
            event.update(status="completed", executed_cases=comparison["observed_calls"])
            event["results"] = [
                {**row, "id": visible["id"]}
                for row, visible in zip(comparison["tool_view"]["results"], cell["view"]["check_inputs"])
            ]
        else:
            event["status"] = "timeout" if result["status"] == "timeout" else "error"
    except (ValueError, OSError) as error:
        if output_dir.exists():
            write_json(output_dir / "adapter-error.json", {"error": type(error).__name__ + ": " + str(error)})
        event["status"] = "error"
    event["wall_ns"] = time.perf_counter_ns() - started
    return event


def run_campaign(output_dir, manifest, model, audit):
    started = time.perf_counter_ns()
    manifest, audit = copy.deepcopy(manifest), copy.deepcopy(audit)
    frozen_digest = manifest_digest(manifest)
    score(manifest, frozen_digest, [], audit)
    output = Path(output_dir)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    source_dir = output / "source"
    source_dir.mkdir()
    for name, digest in LOADED_HASHES.items():
        data = (SOURCE_ROOT / name).read_bytes()
        require(hashlib.sha256(data).hexdigest() == digest, "source changed after import")
        with (source_dir / name).open("xb") as stream:
            stream.write(data)
    write_json(output / "manifest.json", manifest)
    write_json(output / "audit.json", audit)
    write_json(output / "freeze.json", {"manifest_sha256": frozen_digest, "source_sha256": LOADED_HASHES})
    records, measurements = [], []
    cell_map = {cell["episode_id"]: cell for cell in manifest["cells"]}
    status, error, cleanup_error = "completed", None, None
    model_calls = tool_calls = 0
    ready = finished = None
    with (output / "journal.jsonl").open("xb") as journal:
        def emit(value):
            journal.write(canonical(value) + b"\n")
            journal.flush()
            os.fsync(journal.fileno())

        try:
            emit({"event": "starting", "manifest_sha256": frozen_digest})
            ready = model.start()
            emit({"event": "model_ready", "value": ready})
            for scheduled in manifest["schedule"]:
                require(time.perf_counter_ns() - started < 600 * 10**9, "campaign deadline")
                cell = cell_map[scheduled["episode_id"]]
                method = scheduled["method"]
                record = {**scheduled, "manifest_sha256": frozen_digest, "events": [], "episode_wall_ns": None}
                records.append(record)
                episode_started = time.perf_counter_ns()
                emit({"event": "episode_start", **scheduled})
                try:
                    while True:
                        state = next_step(cell, method, record["events"], manifest["run_id"])
                        if state["complete"]:
                            break
                        require(time.perf_counter_ns() - started < 600 * 10**9, "campaign deadline")
                        if state["next"] == "tool":
                            require(tool_calls < 48, "campaign tool budget")
                            probe_dir = output / f"probe-{tool_calls:03d}"
                            tool_calls += 1
                            emit({"event": "tool_start", **scheduled, "directory": probe_dir.name})
                            event = run_probe(probe_dir, cell, method, manifest["run_id"])
                        else:
                            require(model_calls < 144, "campaign model budget")
                            request = model_request(cell, method, record["events"], manifest["run_id"])
                            emit({"event": "model_start", **scheduled, "request": request})
                            model_calls += 1
                            call_started = time.perf_counter_ns()
                            try:
                                response = model(copy.deepcopy(request))
                                elapsed = time.perf_counter_ns() - call_started
                                emit({"event": "model_raw", **scheduled, "response": response})
                                event = model_result(request, response, elapsed)
                                next_step(cell, method, record["events"] + [event], manifest["run_id"])
                                measurements.append({
                                    **scheduled, "generation_ns": response["generation_ns"],
                                    "call_ns": response["call_ns"], "roundtrip_ns": elapsed,
                                })
                            except Exception as failure:
                                event = {
                                    "kind": "model", "status": "error", "text": "", "terminated": False,
                                    "input_ids": None, "output_ids": None,
                                    "wall_ns": time.perf_counter_ns() - call_started,
                                    "request_id": request["id"], "request_sha256": checksum(request),
                                }
                                record["events"].append(event)
                                emit({"event": "accepted", **scheduled, "value": event})
                                raise failure
                        record["events"].append(event)
                        emit({"event": "accepted", **scheduled, "value": event})
                finally:
                    record["episode_wall_ns"] = time.perf_counter_ns() - episode_started
                    emit({"event": "episode_end", "record": record})
            finished = model.finish()
            emit({"event": "model_finished", "value": finished})
            require(time.perf_counter_ns() - started <= 600 * 10**9, "campaign completion deadline")
        except Exception as failure:
            status, error = "aborted", type(failure).__name__ + ": " + str(failure)
            emit({"event": "aborted", "error": error})
        finally:
            try:
                model.close()
            except Exception as failure:
                cleanup_error = type(failure).__name__ + ": " + str(failure)
                status = "aborted"
                emit({"event": "cleanup_failed", "error": cleanup_error})
        write_json(output / "records.json", records)
        records_digest = checksum(records)
        emit({"event": "sealed", "records_sha256": records_digest})
        report = score(manifest, frozen_digest, records, audit)
        emit({"event": "scored", "report_sha256": checksum(report)})
    result = {
        "kind": "FIXED_ONLINE_DEVELOPMENT_DIAGNOSTIC", "status": status, "error": error,
        "cleanup_error": cleanup_error,
        "model_origin": getattr(model, "evidence_kind", "unverified_callback"),
        "records_sha256": records_digest, "records": records, "score": report,
        "ready": ready, "finished": finished, "model_measurements": measurements,
        "controller_wall_ns_before_final_write": time.perf_counter_ns() - started,
        "benchmark_eligible": False, "isolated": False,
    }
    write_json(output / "result.json", result)
    return result
