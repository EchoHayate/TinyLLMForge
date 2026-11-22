import ast
import os
from pathlib import Path

from candidate_execution_protocol import InfrastructureError, prepare_request
from validation_contract import digest, encoded, parse_json


CALLS = frozenset((
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int",
    "len", "list", "map", "max", "min", "range", "reversed", "round", "set",
    "sorted", "str", "sum", "tuple", "zip", "ValueError", "TypeError",
    "RuntimeError", "ZeroDivisionError", "Exception",
))
METHODS = frozenset((
    "append", "extend", "pop", "insert", "remove", "clear", "copy", "reverse",
    "sort", "count", "index", "get", "keys", "values", "items", "update",
    "setdefault", "add", "discard", "union", "intersection", "difference",
    "issubset", "issuperset", "lower", "upper", "casefold", "capitalize",
    "title", "strip", "lstrip", "rstrip", "split", "rsplit", "splitlines",
    "join", "replace", "find", "rfind", "startswith", "endswith", "isdigit",
    "isalpha", "isalnum", "isspace", "islower", "isupper", "ljust", "rjust",
    "center", "zfill", "removeprefix", "removesuffix",
))
NODE_NAMES = frozenset((
    "Module", "FunctionDef", "arguments", "arg", "Return", "Assign", "AugAssign",
    "Expr", "If", "For", "While", "Break", "Continue", "Pass", "Assert",
    "Raise", "Try", "ExceptHandler", "List", "Tuple", "Dict", "Set",
    "ListComp", "SetComp", "DictComp", "GeneratorExp", "comprehension",
    "Subscript", "Slice", "BinOp", "UnaryOp", "BoolOp", "Compare", "IfExp",
    "Name", "Constant", "Call", "keyword", "Attribute", "Load", "Store",
    "JoinedStr", "FormattedValue", "Add", "Sub", "Mult", "MatMult", "Div",
    "FloorDiv", "Mod", "Pow", "LShift", "RShift", "BitOr", "BitXor", "BitAnd",
    "Invert", "Not", "UAdd", "USub", "And", "Or", "Eq", "NotEq", "Lt",
    "LtE", "Gt", "GtE", "Is", "IsNot", "In", "NotIn",
))
POLICY_VERSION = 2
POLICY_SHA256 = digest({
    "version": POLICY_VERSION, "calls": sorted(CALLS), "methods": sorted(METHODS),
    "nodes": sorted(NODE_NAMES), "unknown_names": "deny",
    "attributes": "method_call_only", "denial": "abort_protocol",
    "bindings": "lexical_function_and_comprehension",
    "comprehension_outermost_iterable": "enclosing_scope",
    "reserved_bindings": "calls_and_entry_all_scopes",
    "direct_calls": "unshadowed_builtin_or_entry", "input_types": "exact_builtin_json",
})
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


class CandidateAdmissionError(InfrastructureError):
    pass


def _require(condition, reason):
    if not condition:
        raise CandidateAdmissionError(reason)


def _plain_json(value, remaining=1048576):
    _require(remaining > 0, "JSON traversal budget")
    remaining -= 1
    kind = type(value)
    if kind is dict:
        for name, item in value.items():
            _require(type(name) is str, "JSON object key type")
            remaining = _plain_json(item, remaining)
    elif kind is list:
        for item in value:
            remaining = _plain_json(item, remaining)
    else:
        _require(value is None or kind is bool or kind is int or kind is float
                 or kind is str, "JSON value type")
    return remaining


def _bindings(node):
    if isinstance(node, _COMPREHENSIONS):
        return set()
    names = set()
    if isinstance(node, ast.arg):
        names.add(node.arg)
    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
        names.add(node.id)
    elif isinstance(node, ast.ExceptHandler) and node.name is not None:
        names.add(node.name)
    for child in ast.iter_child_nodes(node):
        names.update(_bindings(child))
    return names


def _local_bindings(names, reserved):
    _require(all(not name.startswith("__") for name in names), "private binding")
    _require(names.isdisjoint(reserved), "reserved callable binding")
    return names


def _check_scope(node, bound, reserved):
    if isinstance(node, ast.Name):
        _require(not node.id.startswith("__") and node.id in bound, "unqualified name")
    if isinstance(node, _COMPREHENSIONS):
        _check_scope(node.generators[0].iter, bound, reserved)
        names = set()
        for generator in node.generators:
            names.update(_bindings(generator.target))
        inner = bound | _local_bindings(names, reserved)
        for ordinal, generator in enumerate(node.generators):
            _check_scope(generator.target, inner, reserved)
            if ordinal:
                _check_scope(generator.iter, inner, reserved)
            for condition in generator.ifs:
                _check_scope(condition, inner, reserved)
        if isinstance(node, ast.DictComp):
            _check_scope(node.key, inner, reserved)
            _check_scope(node.value, inner, reserved)
        else:
            _check_scope(node.elt, inner, reserved)
        return
    for child in ast.iter_child_nodes(node):
        _check_scope(child, bound, reserved)


def admit_request(request):
    try:
        _plain_json(request)
        request_sha256, unused_payload = prepare_request(request)
    except (ValueError, RecursionError) as error:
        raise CandidateAdmissionError("request not qualified") from error
    tree = ast.parse(request["source"])
    function = tree.body[0]
    nodes = list(ast.walk(tree))
    parents = {child: parent for parent in nodes for child in ast.iter_child_nodes(parent)}
    _require(not function.name.startswith("__") and function.name not in CALLS,
             "reserved function name")
    for node in nodes:
        _require(type(node).__name__ in NODE_NAMES, "unsupported computation syntax")
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                _require(target.id in CALLS or target.id == function.name,
                         "indirect call")
            else:
                _require(isinstance(target, ast.Attribute) and target.attr in METHODS,
                         "unqualified call target")
        elif isinstance(node, ast.Attribute):
            parent = parents.get(node)
            _require(isinstance(parent, ast.Call) and parent.func is node
                     and isinstance(node.ctx, ast.Load) and node.attr in METHODS,
                     "attribute is not an approved method call")
        elif isinstance(node, ast.comprehension):
            _require(not node.is_async, "async comprehension")
        elif isinstance(node, ast.keyword):
            _require(node.arg is not None and not node.arg.startswith("__"),
                     "keyword unpacking or private keyword")
        elif isinstance(node, ast.Dict):
            _require(all(key is not None for key in node.keys), "dictionary unpacking")
    reserved = CALLS | {function.name}
    try:
        bound = _local_bindings(_bindings(function), reserved) | reserved
        _check_scope(function, bound, reserved)
    except RecursionError as error:
        raise CandidateAdmissionError("computation syntax depth") from error
    return {"kind": "COMPUTATION_AST_ADMISSION", "version": POLICY_VERSION,
            "policy_sha256": POLICY_SHA256, "request_sha256": request_sha256}


def run_candidate_request(request, argv, *, artifact_dir, cwd, env, **limits):
    import candidate_execution_controller

    try:
        _plain_json(request)
        prepare_request(request)
        snapshot = parse_json(encoded(request), limit=1052672)
    except (ValueError, RecursionError) as error:
        raise CandidateAdmissionError("request not qualified") from error
    admission = admit_request(snapshot)
    try:
        descriptor = os.open(Path(artifact_dir) / "admission.json",
                             os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            raw = encoded(admission)
            _require(output.write(raw) == len(raw), "incomplete admission record write")
    except OSError as error:
        raise CandidateAdmissionError("admission record could not be persisted") from error
    result = candidate_execution_controller.run_request(
        snapshot, argv, artifact_dir=artifact_dir, cwd=cwd, env=env, **limits)
    _require(result["receipt"]["request_sha256"] == admission["request_sha256"],
             "admitted request changed")
    return {**result, "admission": admission}
