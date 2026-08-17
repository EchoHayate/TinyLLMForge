from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name, filename):
    path = TOOLS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load(
    "qwen35_tp4_engine_remote_execution_source_contract",
    "qwen35_tp4_engine_remote_execution_source_contract.py",
)


def test_source_contract_accepts_current_execution_modules():
    result = contract.verify_sources(TOOLS)
    assert result == {
        "classification": "PASS",
        "modules": [
            "qwen35_tp4_cached_continuation_remote_execution_executor.py",
            "qwen35_tp4_cached_continuation_remote_execution_plan.py",
            "qwen35_tp4_cached_continuation_remote_execution_receipt.py",
            (
                "qwen35_tp4_correctness_authority_"
                "campaign_authorization.py"
            ),
            "qwen35_tp4_correctness_authority_campaign_callbacks.py",
            "qwen35_tp4_correctness_authority_campaign_executor.py",
            "qwen35_tp4_correctness_authority_campaign_plan.py",
            (
                "qwen35_tp4_correctness_authority_"
                "campaign_preparation.py"
            ),
            "qwen35_tp4_correctness_authority_campaign_receipt.py",
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
        ],
        "adapter": (
            "qwen35_tp4_engine_remote_subprocess_adapter.py"
        ),
        "adapter_execution_surface": "explicit_runner_only",
        "default_execution_surface": False,
        "explicit_runner_required": True,
        "runtime_compile_policy": "conditional_compile_and_lazy_bnb",
    }


def test_source_contract_rejects_subprocess_main_or_default_runner():
    original = contract.SOURCE_NAMES
    try:
        cases = {
            "subprocess.py": (
                "import subprocess\n"
                "def execute_plan(command_runner=None):\n"
                "    return subprocess.run(['true'])\n"
            ),
            "main.py": (
                "def main(argv=None):\n"
                "    return 0\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            ),
            "default.py": (
                "def default_runner():\n"
                "    return None\n"
                "def execute_plan(command_runner=default_runner):\n"
                "    return command_runner()\n"
            ),
        }
        import tempfile

        for name, source in cases.items():
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / name).write_text(source)
                contract.SOURCE_NAMES = (name,)
                try:
                    contract.verify_sources(root)
                except ValueError:
                    pass
                else:
                    raise AssertionError(
                        f"unsafe source {name} was accepted"
                    )
    finally:
        contract.SOURCE_NAMES = original


def test_executor_requires_runner_before_plan_commands():
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
        result = contract.inspect_executor(TOOLS / name)
        assert result["execute_plan_requires_runner"] is True
        assert result["verified_entrypoint_requires_plan_verifier"] is True
        assert result["verified_entrypoint_requires_execution_env"] is True


def test_source_contract_rejects_unsafe_subprocess_adapter():
    import tempfile

    original = contract.ADAPTER_NAME
    cases = {
        "main.py": (
            "import subprocess\n"
            "def main(): return subprocess.Popen(['ssh'])\n"
        ),
        "shell.py": (
            "import subprocess\n"
            "def run_command():\n"
            " return subprocess.Popen(['ssh'], shell=True)\n"
        ),
        "helper.py": (
            "import subprocess\n"
            "def run_command(): return subprocess.run(['ssh'])\n"
        ),
    }
    try:
        for name, source in cases.items():
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for source_name in contract.SOURCE_NAMES:
                    (root / source_name).write_text(
                        (
                            "def execute_plan(*, command_runner): pass\n"
                            "def execute_verified_plan_file("
                            "*, plan_verifier, execution_env): pass\n"
                        )
                        if "executor" in source_name
                        else ""
                    )
                (root / name).write_text(source)
                contract.ADAPTER_NAME = name
                try:
                    contract.verify_sources(root)
                except ValueError:
                    pass
                else:
                    raise AssertionError(
                        f"unsafe adapter {name} was accepted"
                    )
    finally:
        contract.ADAPTER_NAME = original


def test_preparation_source_has_no_execution_dependencies():
    result = contract.inspect_preparation_source(
        TOOLS
        / "qwen35_tp4_correctness_authority_campaign_preparation.py"
    )
    assert result == {
        "process_execution_surface": False,
        "runner_surface": False,
        "executor_dependency": False,
        "callback_dependency": False,
    }


def test_runtime_compile_policy_accepts_current_repository():
    result = contract.inspect_runtime_compile_policy(ROOT)
    assert result == {
        "compile_helper": "tinyvllm/utils/torch_compile.py",
        "conditional_layers": [
            "tinyvllm/layers/activation.py",
            "tinyvllm/layers/layernorm.py",
            "tinyvllm/layers/rotary_embedding.py",
        ],
        "bitsandbytes_import": "lazy",
    }


def test_runtime_compile_policy_rejects_direct_compile_and_eager_bnb():
    import tempfile

    cases = {
        "direct_compile": (
            "@torch.compile(dynamic=True)\n"
            "def forward(value): return value\n"
        ),
        "eager_bnb": "import bitsandbytes.functional\n",
    }
    for name, unsafe_source in cases.items():
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tinyvllm/utils").mkdir(parents=True)
            (root / "tinyvllm/layers").mkdir(parents=True)
            (root / "tinyvllm/utils/torch_compile.py").write_text(
                "def compile_if_enabled(**kwargs):\n"
                " return lambda function: function\n"
            )
            for relative in (
                "activation.py",
                "layernorm.py",
                "rotary_embedding.py",
            ):
                (root / "tinyvllm/layers" / relative).write_text(
                    (
                        unsafe_source
                        if name == "direct_compile"
                        and relative == "activation.py"
                        else (
                            "from tinyvllm.utils.torch_compile "
                            "import compile_if_enabled\n"
                            "@compile_if_enabled(dynamic=True)\n"
                            "def forward(value): return value\n"
                        )
                    )
                )
            (root / "tinyvllm/layers/linear.py").write_text(
                (
                    unsafe_source
                    if name == "eager_bnb"
                    else (
                        "import importlib\n"
                        "def _load_bitsandbytes_functional():\n"
                        " return importlib.import_module("
                        "'bitsandbytes.functional')\n"
                    )
                )
            )
            try:
                contract.inspect_runtime_compile_policy(root)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"unsafe runtime compile policy {name} was accepted"
                )


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 Engine remote execution source contract tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
