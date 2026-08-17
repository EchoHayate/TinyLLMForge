import importlib.util
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_config_module():
    module_name = "tinyvllm.config"
    path = ROOT / "tinyvllm/config.py"
    fake_transformers = ModuleType("transformers")

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model):
            with open(
                Path(model) / "config.json",
                encoding="utf-8",
            ) as config_file:
                values = json.load(config_file)
            return type("FakeHfConfig", (), values)()

    fake_transformers.AutoConfig = FakeAutoConfig
    sys.modules["transformers"] = fake_transformers
    module = ModuleType(module_name)
    module.__file__ = os.fspath(path)
    sys.modules[module_name] = module
    source = path.read_text(encoding="utf-8")
    exec(
        compile(
            "from __future__ import annotations\n" + source,
            os.fspath(path),
            "exec",
        ),
        module.__dict__,
    )
    return module


codec_module = ModuleType("tinyvllm.engine.qwen35_recurrent_int8_codec")
codec_module.QWEN35_RECURRENT_INT8_CODEC = (
    "qwen35_recurrent_symmetric_int8_per_row_v1"
)
sys.modules[codec_module.__name__] = codec_module
representation_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_representation",
    "tinyvllm/engine/qwen35_hybrid_prefix_representation.py",
)
config_module = _load_config_module()

QWEN35_HYBRID_PREFIX_DEFAULT = (
    representation_module.QWEN35_HYBRID_PREFIX_DEFAULT
)
resolve_qwen35_hybrid_prefix_representation = (
    representation_module.resolve_qwen35_hybrid_prefix_representation
)
Config = config_module.Config


def _expect_exception(exception_types, function):
    try:
        function()
    except exception_types:
        return
    raise AssertionError(f"expected {exception_types}")


def _write_model_config(model_dir):
    (Path(model_dir) / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "num_hidden_layers": 1,
                "max_position_embeddings": 4096,
            }
        ),
        encoding="utf-8",
    )


def test_default_representation_is_exact_and_int8_is_explicit():
    assert QWEN35_HYBRID_PREFIX_DEFAULT == "exact_restore"
    assert resolve_qwen35_hybrid_prefix_representation(
        "exact_restore"
    ).codec is None
    int8 = resolve_qwen35_hybrid_prefix_representation(
        "recurrent_int8_per_row"
    )
    assert int8.version == "qwen35_hybrid_prefix_recurrent_int8_v1"
    assert int8.codec == "qwen35_recurrent_symmetric_int8_per_row_v1"


def test_unknown_or_kv_quantization_names_are_rejected():
    for value in ("int8", "kv8", "", None):
        _expect_exception(
            (TypeError, ValueError),
            lambda value=value: (
                resolve_qwen35_hybrid_prefix_representation(value)
            ),
        )


def test_config_accepts_only_closed_representation_values():
    with TemporaryDirectory() as model_dir:
        _write_model_config(model_dir)
        assert (
            Config(model=model_dir).qwen35_hybrid_prefix_representation
            == "exact_restore"
        )
        assert Config(
            model=model_dir,
            qwen35_hybrid_prefix_representation=(
                "recurrent_int8_per_row"
            ),
        ).qwen35_hybrid_prefix_representation == "recurrent_int8_per_row"
        for value in ("int8", "kv8", "", None):
            _expect_exception(
                ValueError,
                lambda value=value: Config(
                    model=model_dir,
                    qwen35_hybrid_prefix_representation=value,
                ),
            )


if __name__ == "__main__":
    test_default_representation_is_exact_and_int8_is_explicit()
    test_unknown_or_kv_quantization_names_are_rejected()
    test_config_accepts_only_closed_representation_values()
    print("qwen35 hybrid prefix representation tests passed (3 tests)")
