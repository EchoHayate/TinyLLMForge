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
    concern_fields = ("concern",) if prompt["id"] == "C10" else ()
    fields(metadata, ("id", "family", "entry_point", "parameters", "mechanics",
                      "initial_repair_obligation", *concern_fields))
    for name in ("family", "mechanics", "initial_repair_obligation", *concern_fields):
        _text(metadata[name])
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
    require(type(raw_files) is dict and all(type(name) is str for name in raw_files),
            "frozen file mapping")
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
