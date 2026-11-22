import ast
import _json
import builtins
import json
import math
import os
from pathlib import Path
import sys
import types


READY = b"TLF_CANDIDATE_READY_V1\n"
_finite = math.isfinite


def _pairs(entries):
    result = {}
    for name, value in entries:
        if name in result:
            raise ValueError("duplicate key")
        result[name] = value
    return result


def _constant(value):
    raise ValueError("nonfinite JSON")


def _number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite JSON")
    return result


def _json_value(value, ancestors=None, depth=0, remaining=None):
    if ancestors is None:
        ancestors = set()
        remaining = [4096]
    remaining[0] -= 1
    if remaining[0] < 0 or depth > 32:
        raise ValueError("result structure limit")
    if type(value) in (type(None), bool, int):
        return
    if type(value) is float and _finite(value):
        return
    if type(value) is str:
        return
    if type(value) not in (list, dict) or id(value) in ancestors:
        raise ValueError("unsupported result type or cycle")
    if type(value) is dict and any(type(name) is not str for name in value):
        raise ValueError("result keys must be strings")
    ancestors.add(id(value))
    children = value if type(value) is list else list(value.keys()) + list(value.values())
    for child in children:
        _json_value(child, ancestors, depth + 1, remaining)
    ancestors.remove(id(value))


def evaluate(raw):
    if type(raw) is not bytes or len(raw) > 1048576:
        raise ValueError("request byte limit")
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                             parse_constant=_constant, parse_float=_number)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("invalid request encoding") from error
    if type(payload) is not dict or set(payload) != {"source", "entry_point", "args"}:
        raise ValueError("request fields")
    source, entry, args = payload["source"], payload["entry_point"], payload["args"]
    if (type(source) is not str or len(source.encode("utf-8")) > 65536
            or type(entry) is not str or not entry.isidentifier() or type(args) is not list):
        raise ValueError("request types")
    try:
        tree = ast.parse(source)
        if len(tree.body) != 1 or type(tree.body[0]) is not ast.FunctionDef:
            raise ValueError("single function required")
        function = tree.body[0]
        if (function.name != entry or function.decorator_list or function.returns
                or function.type_params or function.args.defaults
                or any(value is not None for value in function.args.kw_defaults)):
            raise ValueError("unsupported function declaration")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.AsyncFunctionDef)):
                raise ValueError("unsupported source")
            if isinstance(node, ast.FunctionDef) and node is not function:
                raise ValueError("nested function")
            if isinstance(node, ast.arg) and node.annotation is not None:
                raise ValueError("argument annotation")
        code = compile(tree, "<candidate>", "exec", dont_inherit=True)
    except (SyntaxError, RecursionError) as error:
        raise ValueError("invalid function syntax") from error
    encoder = _json.make_encoder({}, None, _json.encode_basestring_ascii,
                                 None, ": ", ", ", True, False, False)
    private = {"__builtins__": vars(builtins).copy(), "_finite": math.isfinite}
    check_value = types.FunctionType(_json_value.__code__, private,
                                    argdefs=_json_value.__defaults__)
    private["_json_value"] = check_value
    exception_type = ValueError
    buffer_type, to_bytes, count = bytearray, bytes, len
    namespace = {}
    exec(code, namespace)
    try:
        value = namespace[entry](*args)
        outcome = {"kind": "return", "value": value}
    except exception_type:
        outcome = {"kind": "raises", "type": "ValueError"}
    check_value(outcome)
    encoded = buffer_type()
    for chunk in encoder(outcome, 0):
        encoded.extend(chunk[:16385 - count(encoded)].encode("ascii"))
        if count(encoded) >= 16385:
            return to_bytes(encoded)
    return to_bytes(encoded) + b"\n"


def serve(control_fd):
    if (sys.platform != "linux" or sys.version_info[:3] != (3, 12, 13)
            or sys.prefix != "/runtime" or os.getuid() != 65534 or os.getgid() != 65534):
        raise RuntimeError("isolated runtime required")
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines()
                  if ":" in line)
    if status["Seccomp"].strip() != "2" or status["NoNewPrivs"].strip() != "1":
        raise RuntimeError("isolation filter required")
    if any(int(status[name].strip(), 16) for name in
           ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")):
        raise RuntimeError("capabilities must be cleared")
    if type(control_fd) is not int or control_fd < 3:
        raise RuntimeError("control descriptor required")
    write = os.write
    read = os.read
    run = evaluate
    count = len
    write(control_fd, READY)
    raw = bytearray()
    while True:
        chunk = read(0, min(8192, 1048577 - len(raw)))
        if not chunk:
            break
        raw.extend(chunk)
        if len(raw) > 1048576:
            raise ValueError("request byte limit")
    result = run(bytes(raw))
    position = 0
    while position < count(result):
        position += write(control_fd, result[position:])
