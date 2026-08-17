from __future__ import annotations

import ast
from pathlib import Path


SOURCE_NAMES = (
    "qwen35_tp4_correctness_authority_campaign_authorization.py",
    "qwen35_tp4_correctness_authority_campaign_callbacks.py",
    "qwen35_tp4_correctness_authority_campaign_executor.py",
    "qwen35_tp4_correctness_authority_campaign_plan.py",
    "qwen35_tp4_correctness_authority_campaign_preparation.py",
    "qwen35_tp4_correctness_authority_campaign_receipt.py",
    "qwen35_tp4_cached_continuation_remote_execution_executor.py",
    "qwen35_tp4_cached_continuation_remote_execution_plan.py",
    "qwen35_tp4_cached_continuation_remote_execution_receipt.py",
    "qwen35_tp4_engine_remote_execution_authorization.py",
    "qwen35_tp4_engine_remote_execution_executor.py",
    "qwen35_tp4_engine_remote_execution_plan.py",
    "qwen35_tp4_engine_remote_execution_receipt.py",
    (
        "qwen35_tp4_hybrid_prefix_benchmark_"
        "remote_execution_authorization.py"
    ),
    (
        "qwen35_tp4_hybrid_prefix_benchmark_"
        "remote_execution_executor.py"
    ),
    (
        "qwen35_tp4_hybrid_prefix_benchmark_"
        "remote_execution_plan.py"
    ),
    (
        "qwen35_tp4_hybrid_prefix_benchmark_"
        "remote_execution_receipt.py"
    ),
    "qwen35_tp4_root_logit_remote_execution_authorization.py",
    "qwen35_tp4_root_logit_remote_execution_executor.py",
    "qwen35_tp4_root_logit_remote_execution_plan.py",
    "qwen35_tp4_root_logit_remote_execution_receipt.py",
)
ADAPTER_NAME = "qwen35_tp4_engine_remote_subprocess_adapter.py"
FORBIDDEN_IMPORTS = {"subprocess"}
FORBIDDEN_CALLS = {
    "os.system",
    "os.popen",
    "subprocess.run",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
}
COMPILE_HELPER = "tinyvllm/utils/torch_compile.py"
CONDITIONAL_COMPILE_LAYERS = (
    "tinyvllm/layers/activation.py",
    "tinyvllm/layers/layernorm.py",
    "tinyvllm/layers/rotary_embedding.py",
)
LINEAR_SOURCE = "tinyvllm/layers/linear.py"


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _parse(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("execution source is missing or linked")
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise ValueError("execution source is invalid") from error


def _required_keyword(function, name):
    arguments = function.args
    keyword_names = [argument.arg for argument in arguments.kwonlyargs]
    if name not in keyword_names:
        return False
    index = keyword_names.index(name)
    default = arguments.kw_defaults[index]
    return default is None


def inspect_executor(path):
    tree = _parse(path)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    execute = functions.get("execute_plan")
    verified = (
        functions.get("execute_verified_plan_file")
        or functions.get("execute_verified_campaign_file")
    )
    if execute is None or verified is None:
        raise ValueError("executor entrypoints are missing")
    return {
        "execute_plan_requires_runner": any(
            _required_keyword(execute, name)
            for name in (
                "command_runner",
                "stage_runner",
                "child_executors",
            )
        ),
        "verified_entrypoint_requires_plan_verifier": (
            _required_keyword(verified, "plan_verifier")
        ),
        "verified_entrypoint_requires_execution_env": (
            _required_keyword(verified, "execution_env")
        ),
    }


def inspect_preparation_source(path):
    tree = _parse(path)
    names = {
        _call_name(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    identifiers = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    }
    text = "\n".join(sorted(strings | identifiers | names))
    return {
        "process_execution_surface": any(
            name in FORBIDDEN_CALLS for name in names
        ),
        "runner_surface": "runner" in text.lower(),
        "executor_dependency": (
            "campaign_executor" in text
            or "remote_execution_executor" in text
            or "execute_verified_campaign_file" in text
            or "execute_verified_plan_file" in text
        ),
        "callback_dependency": (
            "campaign_callbacks" in text
            or "build_campaign_callbacks" in text
        ),
    }


def _imports_name(tree, module, name):
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == module
        and any(alias.name == name for alias in node.names)
        for node in tree.body
    )


def _top_level_imports_module(tree, module):
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(
                alias.name == module
                or alias.name.startswith(f"{module}.")
                for alias in node.names
            ):
                return True
        if isinstance(node, ast.ImportFrom):
            if (
                node.module == module
                or (
                    node.module is not None
                    and node.module.startswith(f"{module}.")
                )
            ):
                return True
    return False


def inspect_runtime_compile_policy(repo_root):
    repo_root = Path(repo_root)
    helper_tree = _parse(repo_root / COMPILE_HELPER)
    helper_functions = {
        node.name: node
        for node in helper_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    helper = helper_functions.get("compile_if_enabled")
    if helper is None:
        raise ValueError("conditional compile helper is missing")
    helper_text = ast.unparse(helper)
    if (
        "TORCH_COMPILE_DISABLE" not in helper_text
        or "torch.compile" not in helper_text
    ):
        raise ValueError("conditional compile helper is invalid")

    for relative in CONDITIONAL_COMPILE_LAYERS:
        tree = _parse(repo_root / relative)
        if not _imports_name(
            tree,
            "tinyvllm.utils.torch_compile",
            "compile_if_enabled",
        ):
            raise ValueError("conditional compile import is missing")
        decorators = [
            decorator
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            for decorator in node.decorator_list
        ]
        if not any(
            isinstance(decorator, ast.Call)
            and _call_name(decorator.func) == "compile_if_enabled"
            for decorator in decorators
        ):
            raise ValueError("conditional compile decorator is missing")
        if any(
            isinstance(decorator, ast.Call)
            and _call_name(decorator.func) == "torch.compile"
            for decorator in decorators
        ):
            raise ValueError("direct torch.compile decorator is forbidden")

    linear_tree = _parse(repo_root / LINEAR_SOURCE)
    if _top_level_imports_module(linear_tree, "bitsandbytes"):
        raise ValueError("eager bitsandbytes import is forbidden")
    functions = {
        node.name: node
        for node in linear_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    loader = functions.get("_load_bitsandbytes_functional")
    if loader is None or not any(
        isinstance(node, ast.Call)
        and _call_name(node.func) == "importlib.import_module"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "bitsandbytes.functional"
        for node in ast.walk(loader)
    ):
        raise ValueError("lazy bitsandbytes loader is missing")
    return {
        "compile_helper": COMPILE_HELPER,
        "conditional_layers": list(CONDITIONAL_COMPILE_LAYERS),
        "bitsandbytes_import": "lazy",
    }


def _verify_tree(tree, *, allow_main):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name.split(".", 1)[0] in FORBIDDEN_IMPORTS
                for alias in node.names
            ):
                raise ValueError("subprocess import is forbidden")
        if isinstance(node, ast.ImportFrom):
            if (
                node.module
                and node.module.split(".", 1)[0] in FORBIDDEN_IMPORTS
            ):
                raise ValueError("subprocess import is forbidden")
        if isinstance(node, ast.Call):
            if _call_name(node.func) in FORBIDDEN_CALLS:
                raise ValueError("process execution call is forbidden")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "main" and not allow_main:
                raise ValueError("main execution entrypoint is forbidden")
            for default in [
                *node.args.defaults,
                *[
                    value
                    for value in node.args.kw_defaults
                    if value is not None
                ],
            ]:
                if _call_name(default) in {
                    "default_runner",
                    "_default_runner",
                }:
                    raise ValueError("default command runner is forbidden")
        if isinstance(node, ast.If):
            if (
                "__main__" in ast.unparse(node.test)
                and not allow_main
            ):
                raise ValueError("__main__ execution is forbidden")


def _verify_adapter(tree):
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "run_command" not in functions or "main" in functions:
        raise ValueError("subprocess adapter surface is invalid")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in FORBIDDEN_CALLS - {"subprocess.Popen"}:
                raise ValueError(
                    "subprocess adapter process API is forbidden"
                )
            if name == "subprocess.Popen":
                for keyword in node.keywords:
                    if (
                        keyword.arg == "shell"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True
                    ):
                        raise ValueError(
                            "subprocess adapter shell execution is forbidden"
                        )
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            raise ValueError("subprocess adapter main is forbidden")


def verify_sources(root):
    root = Path(root)
    for name in SOURCE_NAMES:
        _verify_tree(
            _parse(root / name),
            allow_main=(
                name
                in {
                    "qwen35_tp4_cached_continuation_remote_execution_plan.py",
                    "qwen35_tp4_cached_continuation_remote_execution_receipt.py",
                    (
                        "qwen35_tp4_correctness_authority_"
                        "campaign_preparation.py"
                    ),
                    "qwen35_tp4_engine_remote_execution_plan.py",
                    "qwen35_tp4_engine_remote_execution_receipt.py",
                    (
                        "qwen35_tp4_hybrid_prefix_benchmark_"
                        "remote_execution_plan.py"
                    ),
                    "qwen35_tp4_root_logit_remote_execution_plan.py",
                }
            ),
        )
    preparation = inspect_preparation_source(
        root
        / "qwen35_tp4_correctness_authority_campaign_preparation.py"
    )
    if any(preparation.values()):
        raise ValueError(
            "campaign preparation has an execution dependency"
        )
    _verify_adapter(_parse(root / ADAPTER_NAME))
    for name in (
        "qwen35_tp4_correctness_authority_campaign_executor.py",
        "qwen35_tp4_engine_remote_execution_executor.py",
        "qwen35_tp4_cached_continuation_remote_execution_executor.py",
        (
            "qwen35_tp4_hybrid_prefix_benchmark_"
            "remote_execution_executor.py"
        ),
        "qwen35_tp4_root_logit_remote_execution_executor.py",
    ):
        executor = inspect_executor(root / name)
        if not all(executor.values()):
            raise ValueError(
                "executor does not require explicit authorities"
            )
    inspect_runtime_compile_policy(root.parent)
    return {
        "classification": "PASS",
        "modules": sorted(SOURCE_NAMES),
        "adapter": ADAPTER_NAME,
        "adapter_execution_surface": "explicit_runner_only",
        "default_execution_surface": False,
        "explicit_runner_required": True,
        "runtime_compile_policy": (
            "conditional_compile_and_lazy_bnb"
        ),
    }
