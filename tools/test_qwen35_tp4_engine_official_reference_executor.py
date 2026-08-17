from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name, filename):
    path = TOOLS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


executor_module = _load(
    "qwen35_tp4_engine_executor_for_official_reference_test",
    "qwen35_tp4_engine_correctness_executor.py",
)
reference = _load(
    "qwen35_tp4_engine_reference_tokens_for_official_test",
    "qwen35_tp4_engine_reference_tokens.py",
)
official = _load(
    "qwen35_tp4_engine_official_reference_executor",
    "qwen35_tp4_engine_official_reference_executor.py",
)


def _configuration():
    return executor_module.ExecutorConfiguration(
        model_dir="/models/qwen35",
        model_manifest_path="/authority/model_manifest.json",
        model_manifest_sha256="a" * 64,
        source_tree_sha256="b" * 64,
        workload_manifest_path="/authority/workload_manifest.json",
        workload_manifest_sha256="c" * 64,
        model_fingerprint="qwen35-m8-authority",
        gpu_indices=(4, 5, 6, 7),
        dist_port=31001,
        master_port=31002,
        max_cache_entries=8,
        max_cache_bytes=1 << 30,
        timeout_s=600.0,
    )


class FakeBackend:

    def __init__(self):
        self.calls = []
        self.closed = False

    def generate_greedy(
        self,
        *,
        prompt_token_ids,
        generated_tokens,
        generation_policy,
    ):
        self.calls.append({
            "prompt_token_ids": list(prompt_token_ids),
            "generated_tokens": generated_tokens,
            "generation_policy": dict(generation_policy),
        })
        return list(prompt_token_ids) + list(range(generated_tokens))

    def close(self):
        self.closed = True


class StepwiseFakeBackend(FakeBackend):

    def generate_greedy_with_step_logits(
        self,
        *,
        prompt_token_ids,
        generated_tokens,
        generation_policy,
    ):
        self.calls.append({
            "prompt_token_ids": list(prompt_token_ids),
            "generated_tokens": generated_tokens,
            "generation_policy": dict(generation_policy),
        })
        return {
            "output_token_ids": (
                list(prompt_token_ids) + list(range(generated_tokens))
            ),
            "step_logits": [
                f"logits-{index}"
                for index in range(generated_tokens)
            ],
        }


def test_executor_returns_completion_and_stepwise_logits():
    backend = StepwiseFakeBackend()
    executor = official.OfficialGreedyReferenceExecutor(
        _configuration(),
        backend_factory=lambda *args, **kwargs: backend,
    )
    result = executor.generate_reference_with_step_logits(
        scenario="publish_source",
        prompt_token_ids=[11, 12, 13],
        generated_tokens=2,
        generation_policy=dict(reference.GENERATION_POLICY),
    )
    assert result == {
        "output_token_ids": [0, 1],
        "step_logits": ["logits-0", "logits-1"],
    }
    assert backend.calls == [{
        "prompt_token_ids": [11, 12, 13],
        "generated_tokens": 2,
        "generation_policy": {
            "temperature": 0.0,
            "ignore_eos": True,
        },
    }]


def test_executor_is_lazy_uses_rank0_gpu_and_exact_greedy_policy():
    backends = []

    def backend_factory(configuration, *, gpu_index):
        assert configuration.to_payload() == _configuration().to_payload()
        assert gpu_index == 4
        backend = FakeBackend()
        backends.append(backend)
        return backend

    executor = official.OfficialGreedyReferenceExecutor(
        _configuration(),
        backend_factory=backend_factory,
    )
    assert backends == []
    output = executor.generate_reference(
        scenario="publish_source",
        prompt_token_ids=[11, 12, 13],
        generated_tokens=2,
        generation_policy=dict(reference.GENERATION_POLICY),
    )
    assert output == [0, 1]
    assert len(backends) == 1
    assert backends[0].calls == [{
        "prompt_token_ids": [11, 12, 13],
        "generated_tokens": 2,
        "generation_policy": {
            "temperature": 0.0,
            "ignore_eos": True,
        },
    }]
    executor.close()
    executor.close()
    assert backends[0].closed is True


def test_executor_rejects_policy_prefix_and_output_length_mismatch():
    class BadBackend(FakeBackend):
        def __init__(self, result):
            super().__init__()
            self.result = result

        def generate_greedy(self, **kwargs):
            return list(self.result)

    for policy, message in (
        ({"temperature": 0.1, "ignore_eos": True}, "policy"),
        ({"temperature": 0.0, "ignore_eos": False}, "policy"),
    ):
        executor = official.OfficialGreedyReferenceExecutor(
            _configuration(),
            backend_factory=lambda *args, **kwargs: FakeBackend(),
        )
        try:
            executor.generate_reference(
                scenario="publish_source",
                prompt_token_ids=[1, 2],
                generated_tokens=1,
                generation_policy=policy,
            )
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("invalid generation policy was accepted")

    for result, message in (
        ([1, 9, 7], "prefix"),
        ([1, 2], "length"),
    ):
        executor = official.OfficialGreedyReferenceExecutor(
            _configuration(),
            backend_factory=lambda *args, **kwargs: BadBackend(result),
        )
        try:
            executor.generate_reference(
                scenario="publish_source",
                prompt_token_ids=[1, 2],
                generated_tokens=1,
                generation_policy=dict(reference.GENERATION_POLICY),
            )
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"invalid {message} output was accepted")


def test_executor_rejects_unsupported_scenario_and_invalid_tokens():
    executor = official.OfficialGreedyReferenceExecutor(
        _configuration(),
        backend_factory=lambda *args, **kwargs: FakeBackend(),
    )
    invalid = (
        {
            "scenario": "construct_and_bind",
            "prompt_token_ids": [1],
            "generated_tokens": 1,
        },
        {
            "scenario": "publish_source",
            "prompt_token_ids": [],
            "generated_tokens": 1,
        },
        {
            "scenario": "publish_source",
            "prompt_token_ids": [True],
            "generated_tokens": 1,
        },
        {
            "scenario": "publish_source",
            "prompt_token_ids": [1],
            "generated_tokens": 0,
        },
    )
    for values in invalid:
        try:
            executor.generate_reference(
                **values,
                generation_policy=dict(reference.GENERATION_POLICY),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid reference request was accepted")


def test_factory_canonicalizes_configuration_across_module_identity():
    factory = official.build_official_reference_executor_factory(
        _configuration(),
        backend_factory=lambda *args, **kwargs: FakeBackend(),
    )
    executor = factory()
    assert (
        executor.configuration.to_payload()
        == _configuration().to_payload()
    )
    executor.close()


class FakeTensor:

    def __init__(self, values):
        self.values = [list(row) for row in values]
        self.ndim = 2
        self.shape = (
            len(self.values),
            len(self.values[0]),
        )
        self.dtype = "int64"

    def tolist(self):
        return [list(row) for row in self.values]

    def __getitem__(self, index):
        if isinstance(index, int):
            return FakeTensor([self.values[index]])
        if (
            isinstance(index, tuple)
            and len(index) == 3
            and index[0] == slice(None)
            and index[1] == -1
            and index[2] == slice(None)
        ):
            return FakeTensor([self.values[-1]])
        if (
            isinstance(index, tuple)
            and len(index) == 2
            and index[0] == 0
            and index[1] == -1
        ):
            return FakeVector(self.values[-1])
        raise TypeError("unsupported fake tensor index")

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def argmax(self, dim=-1):
        assert dim == -1
        return FakeScalar(max(
            range(len(self.values[0])),
            key=self.values[0].__getitem__,
        ))

    def clone(self):
        return FakeTensor(self.values)

    def __eq__(self, other):
        return (
            isinstance(other, FakeTensor)
            and self.values == other.values
        )


class FakeVector:

    def __init__(self, values):
        self.values = list(values)
        self.ndim = 1
        self.shape = (len(self.values),)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return FakeVector(self.values)

    def argmax(self, dim=-1):
        assert dim == -1
        return FakeScalar(max(
            range(len(self.values)),
            key=self.values.__getitem__,
        ))

    def __eq__(self, other):
        return (
            isinstance(other, FakeVector)
            and self.values == other.values
        )


class FakeScalar:

    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeModel:

    def __init__(self):
        self.calls = []
        self.device = None
        self.eval_called = False

    def to(self, *, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["input_ids"].tolist()[0]
        generated = kwargs["max_new_tokens"]
        return FakeTensor([prompt + list(range(generated))])


class FakeAutoModel:
    model = None
    calls = []

    @classmethod
    def from_pretrained(cls, model_dir, **kwargs):
        cls.calls.append((str(model_dir), dict(kwargs)))
        cls.model = FakeModel()
        return cls.model


def test_transformers_backend_scopes_custom_op_annotation_compatibility():
    with tempfile.TemporaryDirectory() as temporary:
        configuration = _real_configuration(Path(temporary))
        events = []

        class Compatibility:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, exc_type, exc, traceback):
                events.append("exit")
                return False

        class CompatibilityAutoModel(FakeAutoModel):
            @classmethod
            def from_pretrained(cls, model_dir, **kwargs):
                assert events == ["enter"]
                events.append("load")
                return super().from_pretrained(model_dir, **kwargs)

        backend = official.TransformersGreedyReferenceBackend(
            configuration,
            gpu_index=4,
            torch_module=FakeTorch(),
            auto_model=CompatibilityAutoModel,
            custom_op_compatibility=lambda: Compatibility(),
        )
        backend._model()
        assert events == ["enter", "load", "exit"]


def test_custom_op_annotation_compatibility_resolves_and_restores():
    calls = []
    tensor_type = type("Tensor", (), {})
    previous_torch = globals().get("torch")
    globals()["torch"] = SimpleNamespace(Tensor=tensor_type)

    def operation(
        input: "torch.Tensor",
        weight: "torch.Tensor",
        offs: "torch.Tensor",
    ) -> "torch.Tensor":
        return input

    original_annotations = dict(operation.__annotations__)

    def infer_schema(function, mutates_args=()):
        calls.append((
            dict(function.__annotations__),
            tuple(mutates_args),
        ))
        return "(Tensor input, Tensor weight, Tensor offs) -> Tensor"

    try:
        owner = SimpleNamespace(infer_schema=infer_schema)
        with official.torch_custom_op_annotation_compatibility(
            infer_schema_owner=owner,
        ):
            assert owner.infer_schema(operation, ()) == (
                "(Tensor input, Tensor weight, Tensor offs) -> Tensor"
            )
            assert operation.__annotations__ == original_annotations
        assert owner.infer_schema is infer_schema
        assert calls == [({
            "input": tensor_type,
            "weight": tensor_type,
            "offs": tensor_type,
            "return": tensor_type,
        }, ())]
    finally:
        if previous_torch is None:
            globals().pop("torch", None)
        else:
            globals()["torch"] = previous_torch


class FakeGenerateOutput:

    def __init__(self, sequences, logits):
        self.sequences = sequences
        self.logits = tuple(logits)


class StepwiseFakeModel(FakeModel):

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        step = len(self.calls) - 1
        logits = [
            [0.0, 1.0 + step, 0.5]
        ]
        return SimpleNamespace(
            logits=FakeTensor(logits),
            past_key_values=f"past-{step}",
        )


class StepwiseFakeAutoModel(FakeAutoModel):

    @classmethod
    def from_pretrained(cls, model_dir, **kwargs):
        cls.calls.append((str(model_dir), dict(kwargs)))
        cls.model = StepwiseFakeModel()
        return cls.model


class FakeCuda:

    def __init__(self):
        self.empty_cache_calls = 0

    def is_available(self):
        return True

    def empty_cache(self):
        self.empty_cache_calls += 1


class FakeTorch:
    int64 = "int64"
    bfloat16 = "bfloat16"

    def __init__(self):
        self.cuda = FakeCuda()
        self.inference_mode_entries = 0

    def device(self, value):
        return value

    def tensor(self, values, *, dtype, device):
        assert dtype == self.int64
        assert device == "cuda:0"
        return FakeTensor(values)

    def arange(self, start, end, *, dtype, device):
        assert dtype == self.int64
        assert device == "cuda:0"
        return FakeVector(range(start, end))

    def inference_mode(self):
        outer = self

        class Context:
            def __enter__(self):
                outer.inference_mode_entries += 1

            def __exit__(self, exc_type, exc, traceback):
                return False

        return Context()


def _real_configuration(root):
    model_dir = root / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    manifest = root / "model_manifest.json"
    manifest.write_text(
        json.dumps({
            "files": {
                "config.json": {
                    "size": 2,
                    "sha256": official._sha256_file(
                        model_dir / "config.json"
                    ),
                },
            },
        })
        + "\n"
    )
    payload = _configuration().to_payload()
    payload.pop("world_size")
    payload["model_dir"] = str(model_dir)
    payload["model_manifest_path"] = str(manifest)
    payload["model_manifest_sha256"] = (
        official._sha256_file(manifest)
    )
    payload["gpu_indices"] = tuple(payload["gpu_indices"])
    return executor_module.ExecutorConfiguration(**payload)


def test_transformers_backend_loads_local_model_and_generates_exact_tokens():
    with tempfile.TemporaryDirectory() as temporary:
        configuration = _real_configuration(Path(temporary))
        torch_module = FakeTorch()
        FakeAutoModel.calls = []
        backend = official.TransformersGreedyReferenceBackend(
            configuration,
            gpu_index=4,
            torch_module=torch_module,
            auto_model=FakeAutoModel,
        )
        assert FakeAutoModel.calls == []
        output = backend.generate_greedy(
            prompt_token_ids=[11, 12],
            generated_tokens=3,
            generation_policy=dict(reference.GENERATION_POLICY),
        )
        assert output == [11, 12, 0, 1, 2]
        assert FakeAutoModel.calls == [(
            configuration.model_dir,
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "dtype": "bfloat16",
                "attn_implementation": "eager",
            },
        )]
        model = FakeAutoModel.model
        assert model.device == "cuda:0"
        assert model.eval_called is True
        assert torch_module.inference_mode_entries == 1
        assert model.calls == [{
            "input_ids": model.calls[0]["input_ids"],
            "do_sample": False,
            "max_new_tokens": 3,
            "use_cache": True,
            "pad_token_id": None,
            "eos_token_id": None,
        }]
        backend.close()
        backend.close()
        assert torch_module.cuda.empty_cache_calls == 1


def test_transformers_backend_returns_generated_sequences_and_step_logits():
    with tempfile.TemporaryDirectory() as temporary:
        configuration = _real_configuration(Path(temporary))
        torch_module = FakeTorch()
        StepwiseFakeAutoModel.calls = []
        backend = official.TransformersGreedyReferenceBackend(
            configuration,
            gpu_index=4,
            torch_module=torch_module,
            auto_model=StepwiseFakeAutoModel,
        )
        result = backend.generate_greedy_with_step_logits(
            prompt_token_ids=[21, 22],
            generated_tokens=2,
            generation_policy=dict(reference.GENERATION_POLICY),
        )
        assert result == {
            "output_token_ids": [21, 22, 1, 1],
            "step_logits": [
                FakeTensor([[0.0, 1.0, 0.5]]),
                FakeTensor([[0.0, 2.0, 0.5]]),
            ],
        }
        calls = StepwiseFakeAutoModel.model.calls
        assert len(calls) == 2
        assert calls[0]["input_ids"].tolist() == [[21, 22]]
        assert calls[0]["use_cache"] is True
        assert calls[0]["return_dict"] is True
        assert calls[0]["past_key_values"] is None
        assert calls[0]["cache_position"].values == [0, 1]
        assert calls[1]["input_ids"].tolist() == [[1]]
        assert calls[1]["use_cache"] is True
        assert calls[1]["return_dict"] is True
        assert calls[1]["past_key_values"] == "past-0"
        assert calls[1]["cache_position"].values == [2]


def test_transformers_backend_rejects_manifest_or_cuda_mismatch():
    with tempfile.TemporaryDirectory() as temporary:
        configuration = _real_configuration(Path(temporary))
        Path(configuration.model_manifest_path).write_text("{}\n")
        try:
            official.TransformersGreedyReferenceBackend(
                configuration,
                gpu_index=4,
                torch_module=FakeTorch(),
                auto_model=FakeAutoModel,
            )
        except ValueError as error:
            assert "manifest" in str(error)
        else:
            raise AssertionError("tampered model manifest was accepted")

    with tempfile.TemporaryDirectory() as temporary:
        configuration = _real_configuration(Path(temporary))
        (Path(configuration.model_dir) / "config.json").write_text("[]")
        try:
            official.TransformersGreedyReferenceBackend(
                configuration,
                gpu_index=4,
                torch_module=FakeTorch(),
                auto_model=FakeAutoModel,
            )
        except ValueError as error:
            assert "file" in str(error)
        else:
            raise AssertionError("tampered model file was accepted")

    with tempfile.TemporaryDirectory() as temporary:
        configuration = _real_configuration(Path(temporary))
        torch_module = FakeTorch()
        torch_module.cuda.is_available = lambda: False
        try:
            official.TransformersGreedyReferenceBackend(
                configuration,
                gpu_index=4,
                torch_module=torch_module,
                auto_model=FakeAutoModel,
            )
        except ValueError as error:
            assert "CUDA" in str(error)
        else:
            raise AssertionError("missing CUDA was accepted")


class FakeWorkerConnection:

    def __init__(self):
        self.sent = []
        self.responses = [{
            "status": "ready",
            "gpu_index": 4,
        }]
        self.closed = False

    def send(self, payload):
        self.sent.append(payload)
        if payload["action"] == "generate":
            prompt = payload["prompt_token_ids"]
            generated = payload["generated_tokens"]
            self.responses.append({
                "status": "ok",
                "output_token_ids": (
                    list(prompt) + list(range(generated))
                ),
            })
        elif payload["action"] == "generate_step_logits":
            prompt = payload["prompt_token_ids"]
            generated = payload["generated_tokens"]
            self.responses.append({
                "status": "ok",
                "output_token_ids": (
                    list(prompt) + list(range(generated))
                ),
                "step_logits": [
                    FakeTensor([[index, index + 1]])
                    for index in range(generated)
                ],
            })
        elif payload["action"] == "close":
            self.responses.append({
                "status": "closed",
                "cleanup_complete": True,
            })

    def poll(self, timeout):
        assert timeout == 600.0
        return bool(self.responses)

    def recv(self):
        return self.responses.pop(0)

    def close(self):
        self.closed = True


class FakeWorkerProcess:

    def __init__(self):
        self.pid = 1234
        self.exitcode = None
        self.started = False
        self.joined = False

    def start(self):
        self.started = True

    def is_alive(self):
        return self.exitcode is None

    def join(self, timeout=None):
        assert timeout == 600.0
        self.joined = True
        if self.exitcode is None:
            self.exitcode = 0

    def kill(self):
        self.exitcode = -9


class FakeWorkerContext:

    def __init__(self):
        self.parent = FakeWorkerConnection()
        self.child = FakeWorkerConnection()
        self.process = FakeWorkerProcess()
        self.process_args = None

    def Pipe(self, duplex=True):
        assert duplex is True
        return self.parent, self.child

    def Process(self, *, target, args):
        self.process_args = (target, args)
        return self.process


def test_process_backend_uses_spawn_transport_and_proves_cleanup():
    context = FakeWorkerContext()
    backend = official.ProcessGreedyReferenceBackend(
        _configuration(),
        gpu_index=4,
        context=context,
    )
    assert context.process.started is True
    target, args = context.process_args
    assert target is official._official_reference_worker_main
    assert args[0] is context.child
    assert args[1] == _configuration().to_payload()
    assert args[2] == 4
    assert backend.generate_greedy(
        prompt_token_ids=[1, 2],
        generated_tokens=2,
        generation_policy=dict(reference.GENERATION_POLICY),
    ) == [1, 2, 0, 1]
    backend.close()
    backend.close()
    assert context.parent.sent == [
        {
            "action": "generate",
            "prompt_token_ids": [1, 2],
            "generated_tokens": 2,
            "generation_policy": dict(
                reference.GENERATION_POLICY
            ),
        },
        {"action": "close"},
    ]
    assert context.process.joined is True
    assert context.process.exitcode == 0
    assert context.parent.closed is True
    assert context.child.closed is True


def test_process_backend_transports_stepwise_logits():
    context = FakeWorkerContext()
    backend = official.ProcessGreedyReferenceBackend(
        _configuration(),
        gpu_index=4,
        context=context,
    )
    result = backend.generate_greedy_with_step_logits(
        prompt_token_ids=[4, 5],
        generated_tokens=2,
        generation_policy=dict(reference.GENERATION_POLICY),
    )
    assert result == {
        "output_token_ids": [4, 5, 0, 1],
        "step_logits": [
            FakeTensor([[0, 1]]),
            FakeTensor([[1, 2]]),
        ],
    }
    assert context.parent.sent == [{
        "action": "generate_step_logits",
        "prompt_token_ids": [4, 5],
        "generated_tokens": 2,
        "generation_policy": dict(reference.GENERATION_POLICY),
    }]
    backend.close()


def test_process_backend_rejects_worker_error_or_failed_cleanup():
    context = FakeWorkerContext()
    backend = official.ProcessGreedyReferenceBackend(
        _configuration(),
        gpu_index=4,
        context=context,
    )
    context.parent.responses = [{
        "status": "error",
        "error_type": "RuntimeError",
        "error_detail": "synthetic failure",
    }]
    try:
        backend.generate_greedy(
            prompt_token_ids=[1],
            generated_tokens=1,
            generation_policy=dict(reference.GENERATION_POLICY),
        )
    except RuntimeError as error:
        assert "synthetic failure" in str(error)
    else:
        raise AssertionError("worker generation error was hidden")
    assert backend.closed is True

    cleanup_context = FakeWorkerContext()
    cleanup_backend = official.ProcessGreedyReferenceBackend(
        _configuration(),
        gpu_index=4,
        context=cleanup_context,
    )
    cleanup_context.parent.responses = [{
        "status": "closed",
        "cleanup_complete": False,
    }]
    try:
        cleanup_backend.close()
    except RuntimeError as error:
        assert "cleanup" in str(error)
    else:
        raise AssertionError("failed worker cleanup was accepted")

    timeout_context = FakeWorkerContext()
    timeout_backend = official.ProcessGreedyReferenceBackend(
        _configuration(),
        gpu_index=4,
        context=timeout_context,
    )
    timeout_context.parent.poll = lambda timeout: False
    try:
        timeout_backend.close()
    except TimeoutError as error:
        assert "timed out" in str(error)
    else:
        raise AssertionError("worker cleanup timeout was accepted")
    assert timeout_context.process.exitcode == -9
    assert timeout_context.parent.closed is True
    assert timeout_backend.closed is True


def test_process_backend_timeout_poison_closes_transport():
    context = FakeWorkerContext()
    context.parent.poll = lambda timeout: False
    try:
        official.ProcessGreedyReferenceBackend(
            _configuration(),
            gpu_index=4,
            context=context,
        )
    except TimeoutError as error:
        assert "timed out" in str(error)
    else:
        raise AssertionError("worker startup timeout was accepted")
    assert context.process.exitcode == -9
    assert context.parent.closed is True
    assert context.child.closed is True

    context = FakeWorkerContext()
    backend = official.ProcessGreedyReferenceBackend(
        _configuration(),
        gpu_index=4,
        context=context,
    )
    context.parent.poll = lambda timeout: False
    try:
        backend.generate_greedy(
            prompt_token_ids=[1],
            generated_tokens=1,
            generation_policy=dict(reference.GENERATION_POLICY),
        )
    except TimeoutError as error:
        assert "timed out" in str(error)
    else:
        raise AssertionError("worker generation timeout was accepted")
    assert context.process.exitcode == -9
    assert context.parent.closed is True
    assert backend.closed is True


def test_default_backend_factory_returns_process_isolated_backend():
    original = official.ProcessGreedyReferenceBackend
    calls = []

    class RecordingBackend:
        def __init__(self, configuration, *, gpu_index):
            calls.append((
                configuration.to_payload(),
                gpu_index,
            ))

    official.ProcessGreedyReferenceBackend = RecordingBackend
    try:
        result = official._default_backend_factory(
            _configuration(),
            gpu_index=4,
        )
    finally:
        official.ProcessGreedyReferenceBackend = original
    assert isinstance(result, RecordingBackend)
    assert calls == [(_configuration().to_payload(), 4)]


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 official reference executor tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
