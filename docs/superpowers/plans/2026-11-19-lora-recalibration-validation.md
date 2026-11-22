# LoRA Recalibration Schedule and Receipts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the independent 480-position scheduler, bounded backend packets and pure receipt reconstruction, without executing any stored source.

**Architecture:** Rebuild identities from the six frozen raw files at each public entry point. Reuse only the existing pure JSON, outcome and process-event parsers, not their old D01/1416 high-level reports. This is one testable subproject of the approved adapter; tokenizer/rendering remain outside it.

**Tech Stack:** Existing Python standard library, unittest, canonical JSON/SHA256.

## Global Constraints

- Work in `/Users/bytedance/dev/TinyLLMForge` on the existing branch; no branch/worktree creation or commit/push.
- 本阶段实现和测试只使用本地 JSON/AST、静态投影和合成回执，不运行存储的题目函数，不启动真实 tokenizer。
- 不安装依赖、创建分支、提交/推送或清理既有文件。
- No SSH, network, telemetry/bootstrap/flush, subprocess launch of stored source, tokenizer/model loading, generation or training.
- Preserve all existing data and code, including r1 pending/false fields. Write only the two named new Python files; controller owns plan, ledger and result documentation.
- C01–C12; source order `reference_a/reference_b/initial/variant_1/variant_2`; case order P01/P02/H01–H06; ordinals 0–479.
- 15 new batches of 32; no old D01 prefix, retries, deduplication or resumption.
- worker 仅收到 `request_id/source/entry_point/args`；`request_id` 固定为 `request-0000` 至 `request-0479`。
- Source ≤4096 bytes, request ≤8192 bytes, worker ≤16384 bytes, bundle ≤131072 bytes. Worker SHA256 `d2c8095b4d1e854a2f497b084226a92b47aa8b083d59c8ab272b088544fedca4`.
- Backend kind `LORA_REMOTE_VALIDATION_BUNDLE`, mode `matrix`, 32-character lowercase hexadecimal run IDs, one distinct ID per batch.
- 现有后端的单项 2 秒与批次 90 秒边界不在本设计中放宽。
- 192 个参考位置必须全部符合正确期望。
- 36 个错误源码的具名见证必须符合预写错误预测，且区别于正确期望。
- JSON 结果采用严格类型与 canonical 字节比较，不能让 Python 的 `True == 1` 或类似宽松比较消除答案差异。
- 正常返回但与正确答案不同，是语义观察，不是基础设施失败。
- 合成回执测试只验证上述判定逻辑，不提供真实执行来源证明。
- `benchmark_eligible`、`formal_evaluation_ready`、`net_benefit_established`、通用 `isolated` 标志保持 false，不自动推进训练或模型对照。
- Use apply_patch for authored files; no inline code comments, no single-letter variable names.

---

## Scope and Evidence Decisions

Approved parent: `docs/superpowers/specs/2026-11-19-lora-recalibration-validation-adapter-design.md`, sections 3/4 and the relevant acceptance cases in section 7.
Completed dependency: `recalibration_data_contract.load_dataset(raw_files)`.
Do not reimplement it or change its tests.

Caller supplies all 15 run IDs. This pure layer never generates IDs, opens files, starts processes or attests to dispatch. A supplied record means the collector claims that batch was attempted. A missing transport is explicitly represented by `None`; all positions of that attempted batch are unknown. Batches without records are `not_dispatched` in the supplied collection history, not independent proof about an external machine.

Malformed local input (wrong record schema/order, non-JSON transport values, invalid pinned input, repeated run IDs or a record after a failed/incomplete batch) raises ValueError. A correctly shaped record with bad bundle/receipt identity or invalid event evidence instead yields a stopped batch with `evidence_error`, quarantining all 32 positions as `unknown_after_dispatch`; previous valid batches are retained. Raw transport and bundle digests remain in the report. A malformed batch cannot contribute favorable observations. This is conservative rather than attributing unverifiable rows.

Valid incomplete receipts retain their observed prefix. An explicitly observed process failure is `observed` with its failure status; it is not a semantic wrong answer. Missing positions inside an attempted batch remain unknown, even if a stopped done event suggests they were not reached. No additional batch is accepted after any incomplete batch. Merely exhausting a shorter prefix of complete batch records yields `records_exhausted` and is not falsely called a process failure.

`matrix_complete` requires all 15 complete batches and 480 valid completed outcomes.
`data_checks_satisfied` requires matrix_complete plus 192 correct references and 36 correct, disagreeing witness predictions. All provenance/admission flags remain false even when these two pure judgments are true. These flags are not a future real-execution admission API.

Reports are private: they contain outcomes and raw transports. Public publication is not included.

### Task 1: Independent Schedule, Packets and Receipt Reconstruction

**Files:**
- Create: `experiments/lora_necessity/recalibration_validation.py`
- Test: `experiments/lora_necessity/test_recalibration_validation.py`

**Interfaces:**
- Consumes `load_dataset(raw_files: dict[str, bytes]) -> dict`, `encoded/digest/parse_json`, `_process_result/_events/_observations` from existing pure modules.
- Produces `build_schedule(raw_files) -> dict`.
- Produces `build_batches(raw_files, worker_raw, run_ids) -> list[dict]`.
- Produces `build_report(raw_files, worker_raw, run_ids, records, *, expected_runtime) -> dict`.
- Records are an ordered list of exact `{"batch_index": int, "bundle_raw": bytes, "transport": dict | None}` objects.
- Batch wrappers bind `batch_index/start/stop/run_id/plan_sha256/bundle_sha256/worker_sha256/bundle_raw`; bytes are passed unmodified to a future backend, not executed here.
- Plan rows include frozen file pins, task/source/case/entry identity, source/input/expected/request hashes, optional witness prediction/hash, canonical row hash and private source/args/expected.
- Report includes all 480 rows, reference and witness planned/observed/matched counts, per-task counts/checks, exact packet and transport digests, runtime binding, batch status and first stop.

- [x] **Step 1: Write the failing tests.**

Start with the complete test module below. Before implementation, its setUp fails with an explicit assertion, not an import error. Each mutation tests real parser behavior on synthetic JSONL; no task source is executed. Add focused cases if the implementation exposes a missing requirement, recording their failure before fixing code.

```python
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest

from pilot_protocol import FLAGS
from validation_contract import digest, encoded, parse_json
from recalibration_data_contract import FILE_PINS

if importlib.util.find_spec("recalibration_validation") is not None:
    import recalibration_validation as adapter
else:
    adapter = None


ROOT = Path(__file__).resolve().parent


def process(raw=b"", status="completed"):
    return {
        "status": status, "returncode": 0 if status == "completed" else -9,
        "stdout_hex": raw.hex(), "stderr_hex": "", "elapsed_ns": 1,
        "process_reaped": True, "stdin_complete": True,
    }


class RecalibrationValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = {
            name: (ROOT / "data" / (
                name if name == "supervision_v1.json"
                else "recalibration_v2_r1/" + name)).read_bytes()
            for name in FILE_PINS
        }
        cls.worker = (ROOT / "validation_task_worker.py").read_bytes()
        cls.run_ids = ["%032x" % position for position in range(15)]
        cls.runtime = {"host": "synthetic-only", "uid": 123}

    def setUp(self):
        self.assertIsNotNone(adapter, "recalibration validation not implemented")
        self.plan = adapter.build_schedule(self.raw)
        self.batches = adapter.build_batches(self.raw, self.worker, self.run_ids)

    def events(self, batch_index):
        batch = self.batches[batch_index]
        events = [{
            "kind": "ready", "run_id": batch["run_id"],
            "bundle_sha256": batch["bundle_sha256"],
            "worker_sha256": batch["worker_sha256"], "runtime": self.runtime,
        }]
        for local, row in enumerate(self.plan["rows"][batch["start"]:batch["stop"]]):
            outcome = row["witness_prediction"] or row["expected"]
            events.append({
                "kind": "row", "run_id": batch["run_id"], "ordinal": local,
                "request_sha256": row["request_sha256"],
                "result": process(encoded(outcome)),
            })
        events.append({
            "kind": "done", "run_id": batch["run_id"], "received": 32,
            "elapsed_ns": 1, "stopped": False,
        })
        return copy.deepcopy(events)

    def record(self, batch_index, events=None, status="completed"):
        selected = self.events(batch_index) if events is None else events
        return {
            "batch_index": batch_index,
            "bundle_raw": self.batches[batch_index]["bundle_raw"],
            "transport": process(b"".join(encoded(event) + b"\n"
                                          for event in selected), status),
        }

    def report(self, records):
        return adapter.build_report(self.raw, self.worker, self.run_ids, records,
                                    expected_runtime=self.runtime)

    def test_schedule_identity(self):
        self.assertEqual(self.plan, adapter.build_schedule(self.raw))
        self.assertEqual(self.plan["kind"], "LORA_RECALIBRATION_VALIDATION_PLAN")
        self.assertEqual(len(self.plan["rows"]), 480)
        expected = [(task, role, case)
                    for task in ("C%02d" % value for value in range(1, 13))
                    for role in ("reference_a", "reference_b", "initial",
                                 "variant_1", "variant_2")
                    for case in ("P01", "P02", "H01", "H02", "H03", "H04", "H05", "H06")]
        self.assertEqual([(row["task_id"], row["role"], row["case_id"])
                          for row in self.plan["rows"]], expected)
        for ordinal, row in enumerate(self.plan["rows"]):
            self.assertEqual(row["ordinal"], ordinal)
            self.assertEqual(row["row_sha256"], digest({
                name: value for name, value in row.items() if name != "row_sha256"}))
        self.assertEqual(sum(row["witness_prediction"] is not None
                             for row in self.plan["rows"]), 36)

    def test_minimal_packets(self):
        self.assertEqual(len(self.batches), 15)
        requests = []
        for index, batch in enumerate(self.batches):
            self.assertEqual((batch["batch_index"], batch["start"], batch["stop"]),
                             (index, index * 32, (index + 1) * 32))
            self.assertEqual(batch["plan_sha256"], digest(self.plan))
            self.assertEqual(batch["bundle_sha256"],
                             hashlib.sha256(batch["bundle_raw"]).hexdigest())
            self.assertLessEqual(len(batch["bundle_raw"]), 131072)
            bundle = parse_json(batch["bundle_raw"])
            self.assertEqual(bundle["mode"], "matrix")
            self.assertEqual(len(bundle["requests"]), 32)
            for request in bundle["requests"]:
                self.assertEqual(set(request), {"request_id", "source", "entry_point", "args"})
                self.assertLessEqual(len(encoded(request)), 8192)
            requests.extend(bundle["requests"])
        self.assertEqual([request["request_id"] for request in requests],
                         ["request-%04d" % ordinal for ordinal in range(480)])

    def test_reject_changed_pins_worker_and_run_ids(self):
        changed = dict(self.raw)
        changed["oracles.json"] += b" "
        with self.assertRaises(ValueError):
            adapter.build_schedule(changed)
        for worker in (self.worker + b"\n", b"x" * 16385, "not bytes"):
            with self.assertRaises(ValueError):
                adapter.build_batches(self.raw, worker, self.run_ids)
        for identifiers in (self.run_ids[:-1], ["0" * 32] * 15,
                            ["X" * 32] + self.run_ids[1:], tuple(self.run_ids)):
            with self.assertRaises(ValueError):
                adapter.build_batches(self.raw, self.worker, identifiers)

    def test_complete_synthetic_checks_without_admission(self):
        report = self.report([self.record(index) for index in range(15)])
        self.assertTrue(report["matrix_complete"])
        self.assertTrue(report["data_checks_satisfied"])
        self.assertEqual(report["reference_counts"],
                         {"planned": 192, "observed": 192, "matched": 192})
        self.assertEqual(report["witness_counts"],
                         {"planned": 36, "observed": 36, "matched": 36})
        self.assertEqual(len(report["rows"]), 480)
        self.assertIsNone(report["first_stop"])
        for name in (*FLAGS, "execution_provenance_verified", "run_authorized",
                     "full_dynamic_validation_complete"):
            self.assertIs(report[name], False)
        self.assertTrue(all(task["completed"] == 40 for task in report["tasks"]))

    def test_semantic_mismatch_continues_and_bool_is_not_int(self):
        records = [self.record(index) for index in range(15)]
        events = self.events(0)
        events[1]["result"] = process(encoded({"kind": "return", "value": True}))
        records[0] = self.record(0, events)
        report = self.report(records)
        self.assertTrue(report["matrix_complete"])
        self.assertFalse(report["data_checks_satisfied"])
        self.assertFalse(report["rows"][0]["expected_match"])
        self.assertEqual(report["completed"], 480)
        self.assertIsNone(report["first_stop"])

    def test_wrong_witness_prediction_is_data_failure_only(self):
        target = next(row for row in self.plan["rows"]
                      if row["witness_prediction"] is not None)
        records = [self.record(index) for index in range(15)]
        batch_index, local = divmod(target["ordinal"], 32)
        events = self.events(batch_index)
        events[local + 1]["result"] = process(encoded(target["expected"]))
        records[batch_index] = self.record(batch_index, events)
        report = self.report(records)
        self.assertTrue(report["matrix_complete"])
        self.assertFalse(report["data_checks_satisfied"])
        self.assertEqual(report["witness_counts"]["matched"], 35)

    def test_unknown_and_not_dispatched_differ(self):
        record = self.record(0)
        record["transport"] = None
        report = self.report([record])
        self.assertEqual(report["unknown_after_dispatch"], 32)
        self.assertEqual(report["not_dispatched"], 448)
        self.assertFalse(report["matrix_complete"])
        self.assertEqual(self.report([])["not_dispatched"], 480)

    def test_observed_failure_and_unknown_tail(self):
        events = self.events(0)
        events[3]["result"] = process(status="timeout")
        done = {**events[-1], "received": 3, "stopped": True}
        report = self.report([self.record(0, events[:4] + [done])])
        self.assertEqual(report["rows"][2]["status"], "timeout")
        self.assertEqual(report["rows"][2]["dispatch_state"], "observed")
        self.assertEqual(report["unknown_after_dispatch"], 29)
        self.assertEqual(report["not_dispatched"], 448)
        self.assertEqual(report["first_stop"]["ordinal"], 2)

    def test_short_batch_requires_stop(self):
        events = self.events(0)
        done = {**events[-1], "received": 2, "stopped": True}
        report = self.report([self.record(0, events[:3] + [done])])
        self.assertEqual(report["completed"], 2)
        self.assertEqual(report["unknown_after_dispatch"], 30)
        done["stopped"] = False
        report = self.report([self.record(0, events[:3] + [done])])
        self.assertEqual(report["batches"][0]["stop_reason"], "evidence_error")
        self.assertEqual(report["unknown_after_dispatch"], 32)

    def test_malformed_event_identities_quarantine_batch(self):
        mutations = [
            (0, "run_id", "f" * 32), (0, "worker_sha256", "0" * 64),
            (0, "bundle_sha256", "0" * 64), (0, "runtime", {"host": "drift"}),
            (1, "ordinal", True), (1, "ordinal", 32),
            (1, "request_sha256", "0" * 64), (-1, "received", True),
            (-1, "received", 31), (-1, "stopped", 0),
        ]
        for position, name, value in mutations:
            with self.subTest(position=position, name=name, value=value):
                events = self.events(0)
                events[position][name] = value
                report = self.report([self.record(0, events)])
                self.assertEqual(report["batches"][0]["stop_reason"], "evidence_error")
                self.assertEqual(report["unknown_after_dispatch"], 32)
                self.assertFalse(report["data_checks_satisfied"])

    def test_bad_process_or_outcome_quarantines_batch(self):
        for name, value in (("returncode", 1), ("stdin_complete", False),
                            ("process_reaped", False), ("stderr_hex", "61"),
                            ("stdout_hex", encoded({"kind": "return", "value": 1.0}).hex()),
                            ("stdout_hex", b'{"kind":"return","value":NaN}'.hex()),
                            ("stdout_hex", b'{"kind":"return","value":1,"value":2}'.hex())):
            with self.subTest(name=name, value=value):
                events = self.events(0)
                events[1]["result"][name] = value
                report = self.report([self.record(0, events)])
                self.assertEqual(report["batches"][0]["stop_reason"], "evidence_error")

    def test_event_order_and_extra_fields(self):
        original = self.events(0)
        for events in (original[1:], [original[0]] + original,
                       original + [original[-1]],
                       original[:2] + [original[1]] + original[2:],
                       [original[0], original[2], original[1]] + original[3:],
                       [{**original[0], "extra": 1}] + original[1:]):
            with self.subTest(events=len(events)):
                report = self.report([self.record(0, events)])
                self.assertEqual(report["batches"][0]["stop_reason"], "evidence_error")

    def test_missing_done_and_truncated_timeout(self):
        report = self.report([self.record(0, self.events(0)[:-1])])
        self.assertFalse(report["matrix_complete"])
        self.assertEqual(report["batches"][0]["stop_reason"], "missing_done")
        record = self.record(0, self.events(0)[:2], status="timeout")
        record["transport"]["stdout_hex"] += b'{"kind":'.hex()
        report = self.report([record])
        self.assertTrue(report["batches"][0]["truncated_final_line"])
        self.assertEqual(report["completed"], 1)
        self.assertEqual(report["unknown_after_dispatch"], 31)

    def test_batch_order_and_record_after_stop_rejected(self):
        for records in ([self.record(1)], [self.record(0), self.record(0)],
                        [self.record(0, []), self.record(1)],
                        [{**self.record(0), "batch_index": False}],
                        [{**self.record(0), "extra": 1}]):
            with self.subTest(records=len(records)):
                with self.assertRaises(ValueError):
                    self.report(records)

    def test_bundle_tampering_and_complete_prefix(self):
        record = self.record(0)
        bundle = parse_json(record["bundle_raw"])
        bundle["requests"][0]["args"] = []
        record["bundle_raw"] = encoded(bundle)
        report = self.report([record])
        self.assertEqual(report["unknown_after_dispatch"], 32)
        prefix = self.report([self.record(0)])
        self.assertEqual(prefix["completed"], 32)
        self.assertEqual(prefix["not_dispatched"], 448)
        self.assertEqual(prefix["first_stop"]["reason"], "records_exhausted")


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run RED and retain raw output and exit status.**

Run from repo root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_recalibration_validation -v
```

Expected: 15 assertion failures with `recalibration validation not implemented`.
No missing dependency, network or stored-source execution is acceptable.
Save raw output and exit code in the private task evidence root supplied at dispatch.

- [x] **Step 3: Implement the pure module.**

The complete candidate below is an implementation starting point, not permission to retain a defect. Requirements above govern; fixes need failure evidence. Its internal `_packet` helper enables direct artificial size-boundary tests without weakening the frozen public API.

```python
import copy
import hashlib
import re

from pilot_protocol import FLAGS, fields, require
from recalibration_data_contract import FILE_PINS, TASK_IDS, _plain, load_dataset
from validation_contract import digest, encoded
from validation_real_report import _events, _observations, _process_result


WORKER_SHA256 = "d2c8095b4d1e854a2f497b084226a92b47aa8b083d59c8ab272b088544fedca4"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _request(row):
    return {
        "request_id": "request-%04d" % row["ordinal"], "source": row["source"],
        "entry_point": row["entry_point"], "args": copy.deepcopy(row["args"]),
    }


def build_schedule(raw_files):
    dataset = load_dataset(raw_files)
    rows = []
    for prompt, oracle, metadata in zip(
            dataset["prompts"], dataset["oracles"], dataset["structure"]["tasks"]):
        task_sha256 = digest({"prompt": prompt, "oracle": oracle, "metadata": metadata})
        for program in oracle["references"] + oracle["incorrect_programs"]:
            for case in oracle["cases"]:
                witness = program.get("witness")
                prediction = (witness["predicted"] if witness is not None
                              and witness["case_id"] == case["id"] else None)
                row = {
                    "ordinal": len(rows), "task_id": prompt["id"],
                    "role": program["id"], "case_id": case["id"],
                    "entry_point": prompt["entry_point"], "file_sha256": dict(FILE_PINS),
                    "task_sha256": task_sha256, "source": program["source"],
                    "source_sha256": _sha(program["source"].encode("utf-8")),
                    "args": copy.deepcopy(case["args"]), "input_sha256": digest(case["args"]),
                    "expected": copy.deepcopy(case["expected"]),
                    "expected_sha256": digest(case["expected"]),
                    "witness_prediction": copy.deepcopy(prediction),
                    "witness_prediction_sha256": None if prediction is None else digest(prediction),
                }
                row["request_sha256"] = digest(_request(row))
                row["row_sha256"] = digest(row)
                rows.append(row)
    require(len(rows) == 480, "recalibration schedule count")
    return {
        "kind": "LORA_RECALIBRATION_VALIDATION_PLAN", "version": 1,
        "data_version": "recalibration_v2_r1", "file_sha256": dict(FILE_PINS), "rows": rows,
    }


def _packet(rows, worker_raw, run_id):
    require(type(worker_raw) is bytes and len(worker_raw) <= 16384, "worker byte limit")
    require(len(rows) == 32, "batch count")
    requests = [_request(row) for row in rows]
    for request in requests:
        require(len(request["source"].encode("utf-8")) <= 4096, "source byte limit")
        require(len(encoded(request)) <= 8192, "request byte limit")
    bundle_raw = encoded({
        "kind": "LORA_REMOTE_VALIDATION_BUNDLE", "mode": "matrix", "run_id": run_id,
        "worker_source": worker_raw.decode("utf-8"), "requests": requests,
    })
    require(len(bundle_raw) <= 131072, "bundle byte limit")
    return bundle_raw


def _batches(plan, worker_raw, run_ids):
    require(type(worker_raw) is bytes and len(worker_raw) <= 16384
            and _sha(worker_raw) == WORKER_SHA256, "fixed worker identity")
    require(type(run_ids) is list and len(run_ids) == 15
            and all(type(value) is str and re.fullmatch(r"[0-9a-f]{32}", value)
                    for value in run_ids) and len(set(run_ids)) == 15, "batch run IDs")
    plan_sha256 = digest(plan)
    batches = []
    for batch_index, run_id in enumerate(run_ids):
        start = batch_index * 32
        stop = start + 32
        raw = _packet(plan["rows"][start:stop], worker_raw, run_id)
        batches.append({
            "batch_index": batch_index, "start": start, "stop": stop, "run_id": run_id,
            "plan_sha256": plan_sha256, "bundle_sha256": _sha(raw),
            "worker_sha256": WORKER_SHA256, "bundle_raw": raw,
        })
    return batches


def build_batches(raw_files, worker_raw, run_ids):
    return _batches(build_schedule(raw_files), worker_raw, run_ids)


def _rows(planned_rows, observed, dispatch_state):
    rows = []
    for position, planned in enumerate(planned_rows):
        observation = observed[position] if position < len(observed) else None
        result = None if observation is None else observation["result"]
        outcome = None if observation is None else observation["outcome"]
        prediction = planned["witness_prediction"]
        rows.append({
            **{name: planned[name] for name in (
                "ordinal", "task_id", "role", "case_id", "row_sha256",
                "task_sha256", "source_sha256", "input_sha256", "expected_sha256",
                "request_sha256", "witness_prediction_sha256")},
            "dispatch_state": dispatch_state if result is None else "observed",
            "status": dispatch_state if result is None else result["status"],
            "result": copy.deepcopy(result), "outcome": copy.deepcopy(outcome),
            "expected_match": None if outcome is None else
            encoded(outcome) == encoded(planned["expected"]),
            "prediction_match": None if outcome is None or prediction is None else
            encoded(outcome) == encoded(prediction),
        })
    return rows


def _receipt(planned_rows, batch, record, expected_runtime):
    transport = record["transport"]
    observed = []
    runtime = None
    done = None
    truncated = False
    local_stop = None
    reason = None
    detail = None
    stdout_sha256 = None
    try:
        require(record["bundle_raw"] == batch["bundle_raw"], "exact planned bundle")
        if transport is None:
            reason = "missing_transport"
        else:
            stdout = _process_result(transport, outer=True)
            stdout_sha256 = _sha(stdout)
            events, truncated = _events(stdout, timed_out=transport["status"] == "timeout")
            for event in events:
                if type(event) is dict and event.get("kind") == "ready":
                    require(encoded(event.get("runtime")) == encoded(expected_runtime),
                            "runtime differs from expected runtime")
            runtime, done, observed, local_stop = _observations(
                events, planned_rows, batch["run_id"], batch["bundle_sha256"], WORKER_SHA256)
            if done is not None:
                require(done["received"] == 32 or done["stopped"], "short batch must stop")
            if local_stop is not None:
                reason = "worker_" + observed[local_stop]["result"]["status"]
            elif transport["status"] != "completed":
                reason = "transport_" + transport["status"]
            elif runtime is None:
                reason = "missing_ready"
            elif done is None:
                reason = "missing_done"
            elif done["stopped"]:
                reason = "backend_stopped"
            elif truncated or len(observed) != 32:
                reason = "incomplete_receipt"
    except ValueError as error:
        observed = []
        runtime = None
        done = None
        local_stop = None
        reason = "evidence_error"
        detail = str(error)
    return {
        **{name: value for name, value in batch.items() if name != "bundle_raw"},
        "actual_bundle_sha256": _sha(record["bundle_raw"]),
        "transport": copy.deepcopy(transport), "transport_sha256": digest(transport),
        "stdout_sha256": stdout_sha256, "runtime": copy.deepcopy(runtime),
        "runtime_sha256": None if runtime is None else digest(runtime),
        "done": copy.deepcopy(done), "truncated_final_line": truncated,
        "rows": _rows(planned_rows, observed, "unknown_after_dispatch"),
        "received": len(observed), "batch_complete": reason is None,
        "stop_reason": reason, "evidence_error": detail,
        "stop_ordinal": None if local_stop is None else batch["start"] + local_stop,
    }


def _checks(rows):
    references = [row for row in rows if row["role"] in ("reference_a", "reference_b")]
    witnesses = [row for row in rows if row["witness_prediction_sha256"] is not None]
    return {
        "reference_counts": {
            "planned": len(references),
            "observed": sum(row["outcome"] is not None for row in references),
            "matched": sum(row["expected_match"] is True for row in references),
        },
        "witness_counts": {
            "planned": len(witnesses),
            "observed": sum(row["outcome"] is not None for row in witnesses),
            "matched": sum(row["prediction_match"] is True and row["expected_match"] is False
                           for row in witnesses),
        },
    }


def build_report(raw_files, worker_raw, run_ids, records, *, expected_runtime):
    plan = build_schedule(raw_files)
    planned_batches = _batches(plan, worker_raw, run_ids)
    _plain(expected_runtime)
    require(type(expected_runtime) is dict and bool(expected_runtime), "expected runtime required")
    require(type(records) is list and len(records) <= 15, "batch records")
    rows = _rows(plan["rows"], [], "not_dispatched")
    batches = []
    for position, record in enumerate(records):
        fields(record, ("batch_index", "bundle_raw", "transport"))
        require(type(record["batch_index"]) is int and record["batch_index"] == position,
                "batch gap, duplicate or ordering")
        require(not batches or batches[-1]["batch_complete"], "record after incomplete batch")
        require(type(record["bundle_raw"]) is bytes
                and len(record["bundle_raw"]) <= 131072, "record bundle bytes")
        _plain(record["transport"])
        require(record["transport"] is None or type(record["transport"]) is dict,
                "transport object or absent")
        batch = planned_batches[position]
        report = _receipt(plan["rows"][batch["start"]:batch["stop"]],
                          batch, record, expected_runtime)
        rows[batch["start"]:batch["stop"]] = report["rows"]
        batches.append(report)
    checks = _checks(rows)
    completed = sum(row["status"] == "completed" for row in rows)
    complete = len(batches) == 15 and all(batch["batch_complete"] for batch in batches)
    complete = complete and completed == 480
    first_stop = None
    if batches and not batches[-1]["batch_complete"]:
        batch = batches[-1]
        first_stop = {"batch_index": batch["batch_index"], "ordinal": batch["stop_ordinal"],
                      "reason": batch["stop_reason"]}
    elif not complete:
        first_stop = {"batch_index": len(batches), "ordinal": len(batches) * 32,
                      "reason": "records_exhausted"}
    tasks = []
    for task_id in TASK_IDS:
        selected = [row for row in rows if row["task_id"] == task_id]
        task_checks = _checks(selected)
        tasks.append({
            "task_id": task_id, "scheduled": 40,
            "completed": sum(row["status"] == "completed" for row in selected),
            "unknown_after_dispatch": sum(row["dispatch_state"] == "unknown_after_dispatch"
                                          for row in selected),
            "not_dispatched": sum(row["dispatch_state"] == "not_dispatched" for row in selected),
            **task_checks,
            "reference_ok": task_checks["reference_counts"]["matched"] == 16,
            "witnesses_ok": task_checks["witness_counts"]["matched"] == 3,
        })
    evidence = [{"batch_index": batch["batch_index"],
                 "bundle_sha256": batch["actual_bundle_sha256"],
                 "transport_sha256": batch["transport_sha256"]} for batch in batches]
    return {
        "kind": "LORA_RECALIBRATION_VALIDATION_REPORT", "version": 1,
        "data_version": "recalibration_v2_r1", "file_sha256": dict(FILE_PINS),
        "plan_sha256": digest(plan), "worker_sha256": WORKER_SHA256,
        "run_ids": list(run_ids), "expected_runtime": copy.deepcopy(expected_runtime),
        "expected_runtime_sha256": digest(expected_runtime),
        "evidence_sha256": digest(evidence), "batches": batches,
        "scheduled": 480, "completed": completed,
        "observed": sum(row["dispatch_state"] == "observed" for row in rows),
        "unknown_after_dispatch": sum(row["dispatch_state"] == "unknown_after_dispatch"
                                      for row in rows),
        "not_dispatched": sum(row["dispatch_state"] == "not_dispatched" for row in rows),
        "rows": rows, "tasks": tasks, **checks, "first_stop": first_stop,
        "matrix_complete": complete,
        "data_checks_satisfied": complete and checks["reference_counts"]["matched"] == 192
        and checks["witness_counts"]["matched"] == 36,
        "execution_provenance_verified": False, "run_authorized": False,
        "full_dynamic_validation_complete": False,
        **{flag: False for flag in FLAGS},
    }
```

Additional concrete boundary tests to add before accepting Step 3:

```python
    def test_packet_individual_and_total_limits(self):
        rows = copy.deepcopy(self.plan["rows"][:32])
        rows[0]["source"] = "x" * 4097
        with self.assertRaisesRegex(ValueError, "source byte limit"):
            adapter._packet(rows, self.worker, self.run_ids[0])
        rows[0]["source"] = "x" * 4096
        rows[0]["args"] = ["x" * 8192]
        with self.assertRaisesRegex(ValueError, "request byte limit"):
            adapter._packet(rows, self.worker, self.run_ids[0])
        for row in rows:
            row["source"] = "x" * 4096
            row["args"] = ["x" * 100]
        with self.assertRaisesRegex(ValueError, "bundle byte limit"):
            adapter._packet(rows, self.worker, self.run_ids[0])
        with self.assertRaisesRegex(ValueError, "worker byte limit"):
            adapter._packet(self.plan["rows"][:32], b"x" * 16385, self.run_ids[0])

    def test_json_values_and_outer_lifecycle(self):
        record = self.record(0)
        record["transport"]["returncode"] = True
        report = self.report([record])
        self.assertEqual(report["batches"][0]["stop_reason"], "evidence_error")
        for runtime in ({}, {"uid": float("inf")}, {"host": ("tuple",)}):
            with self.assertRaises(ValueError):
                adapter.build_report(self.raw, self.worker, self.run_ids, [],
                                     expected_runtime=runtime)
        for name in ("stdin_complete", "process_reaped"):
            record = self.record(0)
            record["transport"][name] = False
            report = self.report([record])
            self.assertEqual(report["batches"][0]["stop_reason"], "evidence_error")

    def test_observation_after_worker_stop_is_invalid(self):
        events = self.events(0)
        events[1]["result"] = process(status="program_error")
        report = self.report([self.record(0, events)])
        self.assertEqual(report["batches"][0]["stop_reason"], "evidence_error")

    def test_no_execution_imports_or_calls(self):
        import ast
        tree = ast.parse((ROOT / "recalibration_validation.py").read_text())
        forbidden = {"subprocess", "socket", "paramiko", "transformers", "torch",
                     "validation_remote", "validation_task_worker", "tokenizer_session"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse(any(alias.name.split(".")[0] in forbidden
                                     for alias in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module.split(".")[0], forbidden)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"exec", "eval", "compile", "open", "__import__"})
```

- [x] **Step 4: Run GREEN and narrow regression; retain logs and exact hashes.**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_recalibration_validation -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_validation_contract test_prompt_rendering test_recalibration_data_contract test_recalibration_validation -v
```

Expected: all focused tests and all narrow regression tests pass, no skips.
The 30 existing tests are included in the regression, not additional unique passes.
Do not run tests that launch remote/backend/worker code. Tests hash/read stored text and synthesize outputs only.

- [x] **Step 5: Self-review, independent review and preservation verification.**

No commit: explicit user prohibition overrides the workflow's commit step.
Implementation report must include actual RED/GREEN commands, exit codes, raw log paths, all changed paths, source/test SHA256, and concerns.
Controller supplies an actual two-file addition diff to a fresh reviewer. Reviewer checks requirements and quality independently without rerunning same-hash suites. Fixes retain new failure/pass evidence.
Controller verifies all baseline file hashes, HEAD/index/tracked state, the final implementation/test hashes and reports; then updates the dedicated adapter ledger and a new result document. No whole-branch merge review is claimed: this is the second completed subproject, not completion of the full adapter.

## Plan Self-Review

- Section 4 coverage: frozen full identity and canonical row hashes → build_schedule; independent packets and all limits → build_batches; 480 denominator, complete receipt binding, semantic/infrastructure distinction and reference/witness checks → build_report.
- Section 7 coverage in this subproject: ordering/identity, projection privacy, partial states, bool/int, mismatch/stop, malformed evidence, limit guards, purity, preservation and existing narrow regression.
- Sections 5–6 are explicitly deferred to later rendering/tokenizer subplans; no unsupported completed capability.
- Interface names and record schemas are consistent across all candidate calls. Local helper reuse intentionally retains the existing 32-row wire protocol.
- Candidate code is reviewable implementation guidance, not a substitute for failure evidence or an independent review.

## Implementation Evidence

- Evidence root: `/private/tmp/lora-recalibration-validation-20261119-OXqFOy`.
- Implementation report: `task-1-report.md`; actual two-file diff: `task-1.diff`.
- Initial RED: 15 intended assertion failures before production file creation.
- Expanded RED: 25 intended failures; initial GREEN: 25 passed.
- Self-review found non-builtin string record keys accepted by the existing field
  equality helper. A new focused test failed in three subcases before the new
  module added an exact builtin-key check. No existing dependency was changed.
- Initial focused GREEN: 26 passed; narrow regression: 56 passed, including those
  26 and the 30 pre-existing tests. No skips; raw exits both zero.
- Parent verified current source/test hashes against the report, read raw tests,
  checked all 170 preservation files and matched all 15 independently projected
  packet hashes.
- Review repairs corrected two ineffective test inputs: run-ID negatives now use
  a valid worker and require the specific run-ID error; canonical comparison
  tests preserve the expected/predicted list shape while replacing True with 1.
  A new witness test covers the second comparison path.
- Mutation RED evidence shows the repaired cases fail when the run-ID guard is
  bypassed or canonical comparisons become ordinary Python equality. The new
  production file was restored byte-identically after these bounded experiments.
- Post-repair focused GREEN: 27 passed; narrow regression: 57 passed, no skips.
  Final test SHA256: `cd44a8f45ae12ebb1b2abd4b6643b1b1267a8d6ecf0274cdd0fcfe331cce5dc3`.
- A proposed outcome-domain finding was withdrawn after inspecting the backend's
  pre-emission validation/status rewrite. No old parser/backend/domain changed.
- Final independent re-review: Spec Compliance Compliant / Task Quality Approved;
  0 active Critical/Important/Minor. Initial findings and their dispositions are
  preserved in `task-1-review.md`; current full diff is `task-1-fixed.diff`.
- Final report SHA256: `46e25ce8fe48d3acfeca66ef6e85da5e0d843c887bec057c5a4d4e5e7dea89d7`.
- Final review SHA256: `0b407fda8796acf4144566f3d901291c989234f95922a59ec3dfc0d0dfe37e2d`.
- No commits or execution permission; the rendering/tokenizer subproject remains
  separate and unfinished.
