# LoRA Recalibration Data Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为新版验证器提供只读、固定身份、无源码执行副作用的数据契约。

**Architecture:** 只新增数据契约及对应测试。加载器先校验六份原始文件的固定摘要，
再做跨文件、源码签名、用例和见证检查；公开的单题结构检查不是运行准入。
后续排程与长度组件消费该契约，不回写题库。

**Tech Stack:** Python 标准库、AST、unittest；沿用已有 canonical JSON 与严格解析。

## Global Constraints

- 工作目录 `/Users/bytedance/dev/TinyLLMForge`，既有分支 `experiments/feedback-active-verification`；不新建 worktree/分支，不提交/推送。
- 只新增 `experiments/lora_necessity/recalibration_data_contract.py` 和 `experiments/lora_necessity/test_recalibration_data_contract.py`。
- 不修改旧源码、题库、监督文件、封存报告；报告/计划/ledger 可单独新增或记进度。
- 不 import/exec 存储的题目源码；只允许 JSON/AST 检查。
- 不 SSH、模型生成、训练、真实 tokenizer、安装或遥测上传。
- 用户范围优先于测试技能的联网 bootstrap/flush；采用现有本地 unittest，不执行这些联网脚本。
- 不加源码内注释或单字母变量。
- 12 题、60 源、96 用例、36 见证、480 个未来动态位置；实际执行数始终为零。
- 所有准入/收益标志为 false；静态数据加载不等于动态通过。

---

## 子计划边界和后续依赖

上位规格：
`docs/superpowers/specs/2026-11-19-lora-recalibration-validation-adapter-design.md`，
SHA256 `58edbdc88133a12965494669aac602d4ed99d4c727563d632ed89c4544684c7d`。

本子计划只完成规格第 3 节的数据模块，以及第 4.1/7 节在数据层的约束。
不会声称覆盖全部规格。其余模块按依赖顺序另写完整子计划：

1. `recalibration_validation.py`：480 项排程、15 批和回执判定。
2. `recalibration_rendering.py` 与 `recalibration_tokenizer_inventory.py`：
   公开消息边界、实际反馈依赖、120 行清单。
3. `recalibration_tokenizer_report.py` 与 `recalibration_tokenizer_audit.py`：
   新版记录重建与会话生命周期。
4. 全部本地实现独立审阅后，才讨论具体远程启动包。

这里的单题检查器可检查内存中的测试对象，但不认可其数据身份、来源或执行权限。
只有加载器验证固定字节；返回的 dict 是可修改的普通快照，后续组件必须重新绑定
原始文件/内容摘要，不能把它作为不可伪造的准入 token。

## 文件结构

- 新增 `experiments/lora_necessity/recalibration_data_contract.py`：
  固定身份加载、单题 schema、跨文件和 AST 签名检查。
- 新增 `experiments/lora_necessity/test_recalibration_data_contract.py`：
  真实固定文件只读测试、内存反例、无运行行为检查。
- 复用 `pilot_protocol.fields/require/FLAGS` 和
  `validation_contract.encoded/parse_json/load_frozen`，不改这些函数。

### Task 1: 固定字节加载与单题静态契约

**Files:**
- Create: `experiments/lora_necessity/recalibration_data_contract.py`
- Create: `experiments/lora_necessity/test_recalibration_data_contract.py`

**Interfaces:**
- Consumes: `raw_files: dict[str, bytes]`，键为下面六个 `FILE_PINS` 文件名。
  五个 r1 文件从新版目录读取；supervision_v1.json 从原 data 目录读取。
- Produces: `load_dataset(raw_files) -> dict`，返回独立解析的静态快照，
  包括 `prompts/oracles/structure/manifest/supervision/file_sha256/counts`。
- Produces: `validate_task(prompt, oracle, metadata) -> None`；
  三个参数为严格内建 JSON 对象，静态检查失败统一抛 `ValueError`。
  不执行源码、不提供安全或运行准入。
- `FILE_PINS`、`TASK_IDS`、`CASE_IDS`、`SOURCE_IDS` 只定义本版本。

- [x] **Step 1: 写失败测试**

创建测试文件如下；固定源码都只作为字符串/AST 使用。允许在实现前补充能直接
对应上述契约的反例测试，但不能减少这些断言或调用题目执行入口。

```python
import copy
import importlib
import json
from pathlib import Path
import unittest


class RecalibrationDataContractTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(
            Path(__file__).with_name("recalibration_data_contract.py").is_file(),
            "recalibration data contract not implemented",
        )
        self.module = importlib.import_module("recalibration_data_contract")
        self.root = Path(__file__).with_name("data")
        self.raw = {
            name: (self.root / "recalibration_v2_r1" / name).read_bytes()
            for name in ("manifest.json", "prompts.json", "oracles.json",
                         "structure.json", "review.md")
        }
        self.raw["supervision_v1.json"] = (
            self.root / "supervision_v1.json").read_bytes()
        self.dataset = self.module.load_dataset(self.raw)

    def test_frozen_dataset_counts_and_no_admission(self):
        self.assertEqual(self.dataset["kind"], "LORA_RECALIBRATION_DATA_STATIC_ONLY")
        self.assertEqual(self.dataset["data_version"], "recalibration_v2_r1")
        self.assertEqual(self.dataset["counts"], {
            "tasks": 12, "sources": 60, "cases": 96, "witnesses": 36,
            "future_dynamic_positions": 480,
        })
        self.assertEqual(len(self.dataset["supervision"]), 8)
        self.assertEqual([task["id"] for task in self.dataset["prompts"]],
                         list(self.module.TASK_IDS))
        for name in ("run_authorized", "benchmark_eligible",
                     "formal_evaluation_ready", "net_benefit_established", "isolated"):
            self.assertIs(self.dataset[name], False)

    def test_every_pin_rejects_changed_or_reformatted_bytes(self):
        for name in self.raw:
            with self.subTest(name=name):
                changed = dict(self.raw)
                changed[name] += b"\n"
                with self.assertRaises(ValueError):
                    self.module.load_dataset(changed)

    def test_raw_input_shape_and_types(self):
        for invalid in (None, [], {}, {**self.raw, "unexpected": b"{}"}):
            with self.subTest(kind=type(invalid).__name__):
                with self.assertRaises(ValueError):
                    self.module.load_dataset(invalid)
        for replacement in ("{}", bytearray(b"{}"), b"\xff", b"{}" * 600000):
            changed = dict(self.raw)
            changed["prompts.json"] = replacement
            with self.subTest(kind=type(replacement).__name__):
                with self.assertRaises(ValueError):
                    self.module.load_dataset(changed)

    def test_returned_objects_do_not_alias_future_loads(self):
        expected = copy.deepcopy(self.dataset)
        self.dataset["prompts"][0]["public_cases"][0]["args"].append("changed")
        self.dataset["file_sha256"].clear()
        self.assertEqual(self.module.load_dataset(self.raw), expected)

    def test_static_checker_rejects_cross_file_and_witness_changes(self):
        mutations = (
            lambda prompt, oracle, metadata: prompt.update(id="C99"),
            lambda prompt, oracle, metadata: oracle.update(entry_point="different"),
            lambda prompt, oracle, metadata: prompt.update(unexpected=True),
            lambda prompt, oracle, metadata: oracle["cases"].reverse(),
            lambda prompt, oracle, metadata: oracle["references"].reverse(),
            lambda prompt, oracle, metadata: oracle["incorrect_programs"].pop(),
            lambda prompt, oracle, metadata: metadata.update(parameters=["different"]),
            lambda prompt, oracle, metadata: oracle["incorrect_programs"][0]["witness"].update(
                case_id="H01"),
            lambda prompt, oracle, metadata: oracle["incorrect_programs"][0]["witness"].update(
                predicted=copy.deepcopy(oracle["cases"][0]["expected"])),
            lambda prompt, oracle, metadata: prompt["public_cases"][0].update(
                expected={"kind": "return", "value": [1, 0, 1, 0, 1]}),
        )
        for ordinal, mutate in enumerate(mutations):
            prompt = copy.deepcopy(self.dataset["prompts"][0])
            oracle = copy.deepcopy(self.dataset["oracles"][0])
            metadata = copy.deepcopy(self.dataset["structure"]["tasks"][0])
            mutate(prompt, oracle, metadata)
            with self.subTest(ordinal=ordinal), self.assertRaises(ValueError):
                self.module.validate_task(prompt, oracle, metadata)

    def test_static_checker_rejects_source_syntax_without_execution(self):
        for source in ("this is not python", "raise RuntimeError('must not run')",
                       "def admit_reservations(capacity, requests=[]):\n    return []\n",
                       "def admit_reservations(capacity, requests):\n    import os\n",
                       "@print('must not run')\ndef admit_reservations(capacity, requests):\n    return []\n"):
            prompt = copy.deepcopy(self.dataset["prompts"][0])
            oracle = copy.deepcopy(self.dataset["oracles"][0])
            oracle["references"][0]["source"] = source
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.module.validate_task(
                    prompt, oracle, self.dataset["structure"]["tasks"][0])

    def test_static_checker_rejects_non_json_and_nonfinite_values(self):
        for invalid in (float("nan"), float("inf"), (1, 2), {1: "value"}):
            oracle = copy.deepcopy(self.dataset["oracles"][0])
            oracle["cases"][0]["args"].append(invalid)
            with self.subTest(kind=type(invalid).__name__), self.assertRaises(ValueError):
                self.module.validate_task(
                    self.dataset["prompts"][0], oracle,
                    self.dataset["structure"]["tasks"][0])

    def test_strict_parser_retains_duplicate_key_rejection(self):
        from validation_contract import parse_json
        for raw in (b'{"id":"C01","id":"C02"}', b"NaN", b"1e999", b"\xff"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_json(raw)
        self.assertNotEqual(json.dumps(True), json.dumps(1))
```

- [x] **Step 2: 运行并记录 RED**

Run:
`PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/lora_necessity -p 'test_recalibration_data_contract.py' -v`

Expected: 非零退出，断言 `recalibration data contract not implemented` 失败；
不是 import 路径错误，也不先创建空实现使错误消失。
将命令、退出码、完整输出保存到本任务私有报告目录。

- [x] **Step 3: 实现最小契约**

下面是完整实现候选；接口、边界和验收条件为要求，代码仍须经过测试和审阅，
不得把候选代码本身当正确性证明。发现缺陷先增加失败用例，再修复。

```python
import ast
import hashlib
import math

from pilot_protocol import FLAGS, fields, require
from validation_contract import encoded, load_frozen, parse_json


FILE_PINS = {
    "manifest.json": "2d8bc46cb387ca80948aea6a9c9a42264e66560666bc515dbc3cf79fde1e9690",
    "prompts.json": "0b8b037017edda84c28afdf0b7a6dd547d6bfd50582a60287c733f58f6ac1e38",
    "oracles.json": "a07803a677db263e40e71423d0f1c423c093e7a26f99939274fdcd91ddef4be2",
    "structure.json": "b5ede6dd62e380530fb1ce158a0e2aefb2bb6690f0a2dcd3ee2b3d3bea729e6e",
    "review.md": "012f49362d11547b7d4cf11baf3341210a8e9dc102a425b33f94a45b49a8b2fc",
    "supervision_v1.json": "4dafb0ac759697e9b80c5bf097b363082a17a34e6b78fb3d3b91d37a76beb78a",
}
TASK_IDS = tuple("C%02d" % ordinal for ordinal in range(1, 13))
CASE_IDS = ("P01", "P02", "H01", "H02", "H03", "H04", "H05", "H06")
SOURCE_IDS = ("reference_a", "reference_b", "initial", "variant_1", "variant_2")


def _text(value):
    require(type(value) is str and bool(value), "nonempty text required")
    value.encode("utf-8")


def _plain(value, depth=0):
    require(depth <= 64, "JSON nesting limit")
    kind = type(value)
    if kind is dict:
        for name, item in value.items():
            require(type(name) is str, "JSON key type")
            name.encode("utf-8")
            _plain(item, depth + 1)
    elif kind is list:
        for item in value:
            _plain(item, depth + 1)
    elif kind is str:
        value.encode("utf-8")
    else:
        require(value is None or kind in (bool, int, float), "JSON value type")
        require(kind is not float or math.isfinite(value), "finite JSON number")


def _outcome(value):
    require(type(value) is dict, "outcome object")
    if value.get("kind") == "return":
        fields(value, ("kind", "value"))
    else:
        fields(value, ("kind", "type"))
        require(value["kind"] == "raises" and value["type"] == "ValueError",
                "supported outcome")


def _source(source, entry_point, parameters):
    _text(source)
    require(len(source.encode("utf-8")) <= 4096, "source byte limit")
    tree = ast.parse(source)
    require(len(tree.body) == 1 and type(tree.body[0]) is ast.FunctionDef,
            "single function")
    function = tree.body[0]
    require(function.name == entry_point and not function.decorator_list
            and function.returns is None and not getattr(function, "type_params", []),
            "function identity")
    arguments = function.args
    require(not arguments.posonlyargs and not arguments.kwonlyargs
            and arguments.vararg is None and arguments.kwarg is None
            and not arguments.defaults and not arguments.kw_defaults,
            "plain positional signature")
    require([argument.arg for argument in arguments.args] == parameters,
            "parameter identity")
    for node in ast.walk(tree):
        require(not isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef,
                                      ast.AsyncFunctionDef)), "unsupported syntax")
        require(not isinstance(node, ast.FunctionDef) or node is function,
                "nested function")
        require(not isinstance(node, ast.arg) or node.annotation is None,
                "argument annotation")


def _validate_task(prompt, oracle, metadata):
    for value in (prompt, oracle, metadata):
        _plain(value)
    fields(prompt, ("id", "entry_point", "spec", "initial_source", "public_cases"))
    fields(oracle, ("id", "entry_point", "cases", "references", "incorrect_programs"))
    fields(metadata, ("id", "family", "entry_point", "parameters", "mechanics",
                      "initial_repair_obligation"))
    require(type(prompt["id"]) is str and prompt["id"] in TASK_IDS
            and prompt["id"] == oracle["id"] == metadata["id"], "task identity")
    entry_point = prompt["entry_point"]
    _text(entry_point)
    require(entry_point.isidentifier()
            and entry_point == oracle["entry_point"] == metadata["entry_point"],
            "entry point identity")
    _text(prompt["spec"])
    parameters = metadata["parameters"]
    require(type(parameters) is list and bool(parameters)
            and all(type(name) is str and name.isidentifier() for name in parameters)
            and len(parameters) == len(set(parameters)), "parameters")
    cases = oracle["cases"]
    require(type(cases) is list and len(cases) == 8, "case count")
    for case, identifier in zip(cases, CASE_IDS):
        fields(case, ("id", "args", "expected", "derivation", "coverage"))
        require(case["id"] == identifier and type(case["args"]) is list
                and len(case["args"]) == len(parameters), "case identity or arguments")
        _outcome(case["expected"])
        _text(case["derivation"])
        require(type(case["coverage"]) is list and bool(case["coverage"]), "coverage")
        for label in case["coverage"]:
            _text(label)
    require(len({encoded(case["args"]) for case in cases}) == 8, "distinct inputs")
    public = prompt["public_cases"]
    require(type(public) is list and len(public) == 2, "public count")
    for actual, expected in zip(public, cases[:2]):
        fields(actual, ("id", "args", "expected"))
        require(encoded(actual) == encoded({
            name: expected[name] for name in ("id", "args", "expected")}),
            "public binding")
    references = oracle["references"]
    incorrect = oracle["incorrect_programs"]
    require(type(references) is list and len(references) == 2
            and type(incorrect) is list and len(incorrect) == 3, "source count")
    for reference, identifier in zip(references, SOURCE_IDS[:2]):
        fields(reference, ("id", "source", "decomposition", "author"))
        require(reference["id"] == identifier, "reference identity")
        _text(reference["decomposition"])
        _text(reference["author"])
        _source(reference["source"], entry_point, parameters)
    by_case = {case["id"]: case for case in cases}
    for program, identifier in zip(incorrect, SOURCE_IDS[2:]):
        fields(program, ("id", "source", "defect", "witness"))
        require(program["id"] == identifier, "incorrect source identity")
        _text(program["defect"])
        _source(program["source"], entry_point, parameters)
        witness = program["witness"]
        fields(witness, ("case_id", "predicted", "reason"))
        require(type(witness["case_id"]) is str and witness["case_id"] in by_case,
                "witness identity")
        _outcome(witness["predicted"])
        _text(witness["reason"])
        require(encoded(witness["predicted"]) != encoded(
            by_case[witness["case_id"]]["expected"]), "witness must differ")
        if identifier == "initial":
            require(witness["case_id"] in CASE_IDS[:2], "public initial witness")
            require(program["source"] == prompt["initial_source"], "initial binding")


def validate_task(prompt, oracle, metadata):
    try:
        _validate_task(prompt, oracle, metadata)
    except (SyntaxError, UnicodeError, RecursionError) as error:
        raise ValueError("invalid static task encoding or syntax") from error


def load_dataset(raw_files):
    fields(raw_files, FILE_PINS)
    for name, expected in FILE_PINS.items():
        raw = raw_files[name]
        require(type(raw) is bytes and len(raw) <= 1048576, "frozen file bytes")
        require(hashlib.sha256(raw).hexdigest() == expected, "frozen file changed")
    parsed = {name: parse_json(raw) for name, raw in raw_files.items()
              if name.endswith(".json")}
    manifest = parsed["manifest.json"]
    structure = parsed["structure.json"]
    for name in ("structure.json", "prompts.json", "oracles.json", "review.md"):
        require(manifest["files"][name] == {
            "sha256": FILE_PINS[name], "bytes": len(raw_files[name])}, "manifest binding")
    for document in (manifest, structure):
        for name in ("run_authorized", *FLAGS):
            require(document[name] is False, "unadmitted static snapshot")
        require(document["dynamic_status"] == document["tokenizer_length_status"]
                == "pending", "static snapshot state")
    prompts = parsed["prompts.json"]
    oracles = parsed["oracles.json"]
    for tasks in (prompts, oracles, structure["tasks"]):
        require(type(tasks) is list and len(tasks) == 12, "task count")
        require(tuple(task["id"] for task in tasks) == TASK_IDS, "task order")
    for prompt, oracle, metadata in zip(prompts, oracles, structure["tasks"]):
        validate_task(prompt, oracle, metadata)
    supervision = load_frozen("supervision_v1.json", raw_files["supervision_v1.json"])
    require([task["id"] for task in supervision] == [
        "S%02d" % ordinal for ordinal in range(1, 9)], "supervision identity")
    return {
        "kind": "LORA_RECALIBRATION_DATA_STATIC_ONLY", "version": 1,
        "data_version": "recalibration_v2_r1", "file_sha256": dict(FILE_PINS),
        "manifest": manifest, "structure": structure, "prompts": prompts,
        "oracles": oracles, "supervision": supervision,
        "counts": {"tasks": 12, "sources": 60, "cases": 96, "witnesses": 36,
                   "future_dynamic_positions": 480},
        "run_authorized": False, **{name: False for name in FLAGS},
    }
```

- [x] **Step 4: 运行 GREEN 和窄回归**

Run focused:
`PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/lora_necessity -p 'test_recalibration_data_contract.py' -v`

Run regression:
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/lora_necessity python3 -m unittest test_validation_contract test_prompt_rendering test_recalibration_data_contract -v`

Expected: 全部通过、无跳过、无题目函数执行。若某个回归模块不存在，先报告实际
文件情况，不悄悄换成无关测试；本计划写作时将核对该命令目标。
只运行与改动有关的测试，不启动全仓训练/远程测试。

- [x] **Step 5: 自查、审阅与留存**

检查新文件的 whitespace，核对上位规格和原文件 SHA，记录 RED/GREEN 命令、
退出码、测试数、源码/测试 SHA256 和任何疑点到私有报告。
不 commit。主控生成两个新文件的完整 diff，交独立 reviewer 给出
Spec Compliance 和 Task Quality 两个判定；有阻断项则先测试复现再修复。
任务通过后更新专用 ledger，不覆盖既有 `.superpowers/sdd/progress.md`。

## 计划自查

本子计划实现规格的数据层条目；后续五个模块和所有远程能力明确不在此交付范围。
参数名、文件名、快照返回字段和测试调用一致；无新依赖或隐藏运行许可。
计划源码是可审阅候选，不是静态审查/实际运行/安全保证。

## 实施结果

Task 1 已完成，独立 Spec Compliance / Task Quality 均为 PASS，发现项为 0。
实际实现修复了候选代码的三个问题，均有补充失败测试：
C10 固定 metadata 中的 concern 字段、metadata 描述类型、原始映射键的 str 子类。
上面的候选代码保留作实施起点记录，不用它覆盖实际实现。

- 首次 RED：8 个缺少实现断言失败，退出码 1。
- 最终 focused GREEN：21 tests，0 failures/errors/skips，退出码 0。
- 窄回归 GREEN：30 tests，0 failures/errors/skips，退出码 0。
- 原始八个测试及 setUp 的 AST 保持一致；167 个既有 LoRA 文件保持不变。
- 源码 SHA256：`52695a9e2fb7e90b04f4fc7c95dcdc8b0310bbb0c269e6779890b9dff989ed02`。
- 测试 SHA256：`cb74252172abb3b24daf93cc45b750c8ad38c5988849699efc21b1f04758df1d`。
- 证据根：`/private/tmp/lora-recalibration-data-contract-20261119-xRMkGn`。
- 实施报告：`task-1-report.md`；独立审阅：`task-1-review.md`；完整 diff：`task-1.diff`。

只完成数据契约子计划。其余五个组件及实际 480 项/120 行检查仍未完成；
没有源码执行、真实 tokenizer、远程运行、训练、commit/push。
