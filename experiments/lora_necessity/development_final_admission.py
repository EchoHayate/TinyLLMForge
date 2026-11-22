import ast
import os
from pathlib import Path

from candidate_execution_admission import (
    CALLS, METHODS, NODE_NAMES, POLICY_VERSION, POLICY_SHA256,
    CandidateAdmissionError, _require, _plain_json, _bindings,
    _local_bindings, _check_scope,
)
from development_final_protocol import prepare_request
from validation_contract import encoded, parse_json


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
    return {"kind": "DEVELOPMENT_FINAL_AST_ADMISSION", "version": POLICY_VERSION,
            "policy_sha256": POLICY_SHA256, "request_sha256": request_sha256}


def _save_admission(path, admission):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        output = os.fdopen(descriptor, "wb")
    except BaseException:
        os.close(descriptor)
        raise
    with output:
        raw = encoded(admission)
        _require(output.write(raw) == len(raw), "incomplete admission record write")


def run_candidate_request(request, argv, *, artifact_dir, cwd, env, **limits):
    import development_final_controller

    try:
        _plain_json(request)
        prepare_request(request)
        snapshot = parse_json(encoded(request), limit=1052672)
    except (ValueError, RecursionError) as error:
        raise CandidateAdmissionError("request not qualified") from error
    admission = admit_request(snapshot)
    try:
        _save_admission(Path(artifact_dir) / "admission.json", admission)
    except (OSError, ValueError) as error:
        raise CandidateAdmissionError("admission record could not be persisted") from error
    result = development_final_controller.run_request(
        snapshot, argv, artifact_dir=artifact_dir, cwd=cwd, env=env, **limits)
    _require(result["receipt"]["request_sha256"] == admission["request_sha256"],
             "admitted request changed")
    return {**result, "admission": admission}
