from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
import typing


def _load_module(module_name, filename):
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


reference = _load_module(
    "qwen35_tp4_engine_reference_tokens",
    "qwen35_tp4_engine_reference_tokens.py",
)
executor_module = _load_module(
    "qwen35_tp4_engine_correctness_executor",
    "qwen35_tp4_engine_correctness_executor.py",
)


_ORIGINAL_CUSTOM_OP_INFER_SCHEMA = None


def _resolve_custom_op_schema(function, mutates_args=()):
    original = _ORIGINAL_CUSTOM_OP_INFER_SCHEMA
    if original is None:
        raise RuntimeError(
            "custom-op schema compatibility is not active"
        )
    annotations = getattr(function, "__annotations__", None)
    if not annotations or not any(
        isinstance(value, str) for value in annotations.values()
    ):
        return original(function, mutates_args)
    resolved = typing.get_type_hints(
        function,
        globalns=function.__globals__,
    )
    function.__annotations__ = resolved
    try:
        return original(function, mutates_args)
    finally:
        function.__annotations__ = annotations


@contextmanager
def torch_custom_op_annotation_compatibility(
    *,
    infer_schema_owner=None,
):
    global _ORIGINAL_CUSTOM_OP_INFER_SCHEMA
    if infer_schema_owner is None:
        import torch._custom_op.impl as infer_schema_owner
    if _ORIGINAL_CUSTOM_OP_INFER_SCHEMA is not None:
        raise RuntimeError(
            "custom-op schema compatibility is nested"
        )
    original = infer_schema_owner.infer_schema
    _ORIGINAL_CUSTOM_OP_INFER_SCHEMA = original
    infer_schema_owner.infer_schema = _resolve_custom_op_schema
    try:
        yield
    finally:
        infer_schema_owner.infer_schema = original
        _ORIGINAL_CUSTOM_OP_INFER_SCHEMA = None


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_model_manifest(configuration):
    manifest_path = Path(configuration.model_manifest_path)
    model_dir = Path(configuration.model_dir)
    if (
        not manifest_path.is_file()
        or _sha256_file(manifest_path)
        != configuration.model_manifest_sha256
    ):
        raise ValueError("official reference model manifest mismatch")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "official reference model manifest is invalid"
        ) from error
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(
            "official reference model manifest files are invalid"
        )
    for name, row in files.items():
        path = model_dir / name
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(row, dict)
            or isinstance(row.get("size"), bool)
            or not isinstance(row.get("size"), int)
            or row["size"] < 0
            or not isinstance(row.get("sha256"), str)
            or len(row["sha256"]) != 64
            or not path.is_file()
            or path.stat().st_size != row["size"]
            or _sha256_file(path) != row["sha256"]
        ):
            raise ValueError(
                f"official reference model file mismatch: {name}"
            )


class TransformersGreedyReferenceBackend:

    def __init__(
        self,
        configuration,
        *,
        gpu_index,
        torch_module=None,
        auto_model=None,
        custom_op_compatibility=None,
    ):
        self.configuration = _normalize_configuration(configuration)
        if (
            isinstance(gpu_index, bool)
            or not isinstance(gpu_index, int)
            or gpu_index < 0
            or gpu_index not in self.configuration.gpu_indices
        ):
            raise ValueError(
                "official reference GPU index is invalid"
            )
        _verify_model_manifest(self.configuration)
        if torch_module is None:
            import torch as torch_module
        real_auto_model = auto_model is None
        if real_auto_model:
            from transformers import AutoModelForCausalLM
            auto_model = AutoModelForCausalLM
        if custom_op_compatibility is None:
            custom_op_compatibility = (
                torch_custom_op_annotation_compatibility
                if real_auto_model
                else nullcontext
            )
        if not callable(custom_op_compatibility):
            raise TypeError(
                "custom_op_compatibility must be callable"
            )
        if not bool(torch_module.cuda.is_available()):
            raise ValueError("official reference CUDA is unavailable")
        self.gpu_index = gpu_index
        self.torch = torch_module
        self.auto_model = auto_model
        self.custom_op_compatibility = custom_op_compatibility
        self.model = None
        self.closed = False

    def _model(self):
        if self.model is None:
            with self.custom_op_compatibility():
                self.model = self.auto_model.from_pretrained(
                    self.configuration.model_dir,
                    local_files_only=True,
                    trust_remote_code=False,
                    dtype=self.torch.bfloat16,
                    attn_implementation="eager",
                )
            self.model = self.model.to(
                device=self.torch.device("cuda:0")
            )
            self.model.eval()
        return self.model

    def generate_greedy(
        self,
        *,
        prompt_token_ids,
        generated_tokens,
        generation_policy,
    ):
        if self.closed:
            raise RuntimeError(
                "official reference backend is closed"
            )
        if generation_policy != reference.GENERATION_POLICY:
            raise ValueError("reference generation policy mismatch")
        input_ids = self.torch.tensor(
            [list(prompt_token_ids)],
            dtype=self.torch.int64,
            device=self.torch.device("cuda:0"),
        )
        with self.torch.inference_mode():
            output = self._model().generate(
                input_ids=input_ids,
                do_sample=False,
                max_new_tokens=generated_tokens,
                use_cache=True,
                pad_token_id=None,
                eos_token_id=None,
            )
        if (
            getattr(output, "ndim", None) != 2
            or tuple(getattr(output, "shape", ()))
            != (1, len(prompt_token_ids) + generated_tokens)
        ):
            raise ValueError(
                "official reference generated tensor shape mismatch"
            )
        rows = output.tolist()
        if (
            not isinstance(rows, list)
            or len(rows) != 1
            or not isinstance(rows[0], list)
        ):
            raise ValueError(
                "official reference generated tensor is invalid"
            )
        return rows[0]

    def generate_greedy_with_step_logits(
        self,
        *,
        prompt_token_ids,
        generated_tokens,
        generation_policy,
    ):
        if self.closed:
            raise RuntimeError(
                "official reference backend is closed"
            )
        if generation_policy != reference.GENERATION_POLICY:
            raise ValueError("reference generation policy mismatch")
        input_ids = self.torch.tensor(
            [list(prompt_token_ids)],
            dtype=self.torch.int64,
            device=self.torch.device("cuda:0"),
        )
        output_token_ids = list(prompt_token_ids)
        step_logits = []
        past_key_values = None
        sequence_length = 0
        with self.torch.inference_mode():
            for _ in range(generated_tokens):
                input_length = int(input_ids.shape[-1])
                cache_position = self.torch.arange(
                    sequence_length,
                    sequence_length + input_length,
                    dtype=self.torch.int64,
                    device=self.torch.device("cuda:0"),
                )
                output = self._model()(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    cache_position=cache_position,
                )
                logits = getattr(output, "logits", None)
                past_key_values = getattr(
                    output,
                    "past_key_values",
                    None,
                )
                if past_key_values is None:
                    raise ValueError(
                        "official reference KV cache is missing"
                    )
                score = logits[:, -1, :]
                if (
                    getattr(score, "ndim", None) != 2
                    or tuple(getattr(score, "shape", ()))[:1] != (1,)
                ):
                    raise ValueError(
                        "official reference step logits shape mismatch"
                    )
                score = score.detach().float().cpu().clone()
                step_logits.append(score)
                token_id = int(score.argmax(dim=-1).item())
                output_token_ids.append(token_id)
                sequence_length += input_length
                input_ids = self.torch.tensor(
                    [[token_id]],
                    dtype=self.torch.int64,
                    device=self.torch.device("cuda:0"),
                )
        for score in step_logits:
            if (
                getattr(score, "ndim", None) != 2
                or tuple(getattr(score, "shape", ()))[:1] != (1,)
            ):
                raise ValueError(
                    "official reference step logits shape mismatch"
                )
        return {
            "output_token_ids": output_token_ids,
            "step_logits": step_logits,
        }

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.model = None
        self.torch.cuda.empty_cache()


def _configuration_from_payload(payload):
    payload = dict(payload)
    if payload.pop("world_size", None) != 4:
        raise ValueError("configuration world_size mismatch")
    if isinstance(payload.get("gpu_indices"), list):
        payload["gpu_indices"] = tuple(payload["gpu_indices"])
    return executor_module.ExecutorConfiguration(**payload)


def _official_reference_worker_main(
    connection,
    configuration_payload,
    gpu_index,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    backend = None
    try:
        configuration = _configuration_from_payload(
            configuration_payload
        )
        backend = TransformersGreedyReferenceBackend(
            configuration,
            gpu_index=gpu_index,
        )
        connection.send({
            "status": "ready",
            "gpu_index": gpu_index,
        })
        while True:
            request = connection.recv()
            action = request.get("action")
            if action == "close":
                backend.close()
                backend = None
                connection.send({
                    "status": "closed",
                    "cleanup_complete": True,
                })
                return
            if action not in {"generate", "generate_step_logits"}:
                raise ValueError(
                    f"unsupported reference worker action: {action}"
                )
            try:
                arguments = {
                    "prompt_token_ids": request["prompt_token_ids"],
                    "generated_tokens": request["generated_tokens"],
                    "generation_policy": request["generation_policy"],
                }
                if action == "generate":
                    output = backend.generate_greedy(**arguments)
                else:
                    output = (
                        backend.generate_greedy_with_step_logits(
                            **arguments
                        )
                    )
            except Exception as error:
                connection.send({
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error_detail": str(error),
                })
            else:
                if action == "generate":
                    connection.send({
                        "status": "ok",
                        "output_token_ids": output,
                    })
                else:
                    connection.send({
                        "status": "ok",
                        "output_token_ids": output[
                            "output_token_ids"
                        ],
                        "step_logits": output["step_logits"],
                    })
    except Exception as error:
        try:
            connection.send({
                "status": "error",
                "error_type": type(error).__name__,
                "error_detail": str(error),
            })
        finally:
            raise
    finally:
        if backend is not None:
            backend.close()
        connection.close()


class ProcessGreedyReferenceBackend:

    def __init__(
        self,
        configuration,
        *,
        gpu_index,
        context=None,
    ):
        self.configuration = _normalize_configuration(configuration)
        self.gpu_index = gpu_index
        self.timeout_s = self.configuration.timeout_s
        if context is None:
            context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self.connection = parent
        self.process = context.Process(
            target=_official_reference_worker_main,
            args=(
                child,
                self.configuration.to_payload(),
                gpu_index,
            ),
        )
        self.process.start()
        child.close()
        self.closed = False
        try:
            ready = self._receive()
            if ready != {
                "status": "ready",
                "gpu_index": gpu_index,
            }:
                raise RuntimeError(
                    "official reference worker ready handshake failed"
                )
        except BaseException:
            self._abort()
            raise

    def _receive(self):
        if not self.connection.poll(self.timeout_s):
            raise TimeoutError(
                "official reference worker timed out"
            )
        response = self.connection.recv()
        if not isinstance(response, dict):
            raise RuntimeError(
                "official reference worker response is invalid"
            )
        return response

    def _abort(self):
        if self.process.is_alive():
            self.process.kill()
        self.process.join(timeout=self.timeout_s)
        self.connection.close()
        self.closed = True

    def generate_greedy(
        self,
        *,
        prompt_token_ids,
        generated_tokens,
        generation_policy,
    ):
        if self.closed:
            raise RuntimeError(
                "official reference worker is closed"
            )
        try:
            self.connection.send({
                "action": "generate",
                "prompt_token_ids": list(prompt_token_ids),
                "generated_tokens": generated_tokens,
                "generation_policy": dict(generation_policy),
            })
            response = self._receive()
        except BaseException:
            self._abort()
            raise
        if response.get("status") == "error":
            self._abort()
            raise RuntimeError(
                "official reference worker failed: "
                f"{response.get('error_type')}: "
                f"{response.get('error_detail')}"
            )
        if set(response) != {"status", "output_token_ids"} or (
            response["status"] != "ok"
        ):
            self._abort()
            raise RuntimeError(
                "official reference worker output response is invalid"
            )
        return response["output_token_ids"]

    def generate_greedy_with_step_logits(
        self,
        *,
        prompt_token_ids,
        generated_tokens,
        generation_policy,
    ):
        if self.closed:
            raise RuntimeError(
                "official reference worker is closed"
            )
        try:
            self.connection.send({
                "action": "generate_step_logits",
                "prompt_token_ids": list(prompt_token_ids),
                "generated_tokens": generated_tokens,
                "generation_policy": dict(generation_policy),
            })
            response = self._receive()
        except BaseException:
            self._abort()
            raise
        if response.get("status") == "error":
            self._abort()
            raise RuntimeError(
                "official reference worker failed: "
                f"{response.get('error_type')}: "
                f"{response.get('error_detail')}"
            )
        if (
            set(response)
            != {"status", "output_token_ids", "step_logits"}
            or response["status"] != "ok"
        ):
            self._abort()
            raise RuntimeError(
                "official reference worker stepwise response is invalid"
            )
        return {
            "output_token_ids": response["output_token_ids"],
            "step_logits": response["step_logits"],
        }

    def close(self):
        if self.closed:
            return
        try:
            self.connection.send({"action": "close"})
            response = self._receive()
            self.process.join(timeout=self.timeout_s)
        except BaseException:
            self._abort()
            raise
        self.connection.close()
        self.closed = True
        if (
            response != {
                "status": "closed",
                "cleanup_complete": True,
            }
            or self.process.is_alive()
            or self.process.exitcode != 0
        ):
            raise RuntimeError(
                "official reference worker cleanup failed"
            )


def _default_backend_factory(configuration, *, gpu_index):
    return ProcessGreedyReferenceBackend(
        configuration,
        gpu_index=gpu_index,
    )


def _normalize_configuration(configuration):
    if isinstance(
        configuration,
        executor_module.ExecutorConfiguration,
    ):
        return configuration
    to_payload = getattr(configuration, "to_payload", None)
    if not callable(to_payload):
        raise TypeError(
            "configuration must provide a canonical payload"
        )
    payload = dict(to_payload())
    if payload.pop("world_size", None) != 4:
        raise ValueError("configuration world_size mismatch")
    if isinstance(payload.get("gpu_indices"), list):
        payload["gpu_indices"] = tuple(payload["gpu_indices"])
    return executor_module.ExecutorConfiguration(**payload)


class OfficialGreedyReferenceExecutor:

    def __init__(
        self,
        configuration,
        *,
        backend_factory=_default_backend_factory,
    ):
        if not callable(backend_factory):
            raise TypeError("backend_factory must be callable")
        self.configuration = _normalize_configuration(configuration)
        self.backend_factory = backend_factory
        self.backend = None
        self.closed = False

    def _backend(self):
        if self.backend is None:
            self.backend = self.backend_factory(
                self.configuration,
                gpu_index=self.configuration.gpu_indices[0],
            )
        return self.backend

    def generate_reference(
        self,
        *,
        scenario,
        prompt_token_ids,
        generated_tokens,
        generation_policy,
    ):
        if self.closed:
            raise RuntimeError("official reference executor is closed")
        if scenario not in reference.REFERENCE_SCENARIOS:
            raise ValueError(
                f"unsupported reference scenario: {scenario}"
            )
        if (
            not isinstance(prompt_token_ids, (list, tuple))
            or not prompt_token_ids
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in prompt_token_ids
            )
        ):
            raise ValueError("reference prompt token IDs are invalid")
        if (
            isinstance(generated_tokens, bool)
            or not isinstance(generated_tokens, int)
            or generated_tokens <= 0
        ):
            raise ValueError(
                "reference generated token count is invalid"
            )
        if generation_policy != reference.GENERATION_POLICY:
            raise ValueError("reference generation policy mismatch")
        prompt = list(prompt_token_ids)
        result = self._backend().generate_greedy(
            prompt_token_ids=prompt,
            generated_tokens=generated_tokens,
            generation_policy=dict(generation_policy),
        )
        if not isinstance(result, (list, tuple)):
            raise ValueError("reference backend output is invalid")
        result = list(result)
        expected_length = len(prompt) + generated_tokens
        if len(result) != expected_length:
            raise ValueError("reference backend output length mismatch")
        if result[:len(prompt)] != prompt:
            raise ValueError("reference backend prompt prefix mismatch")
        completion = result[len(prompt):]
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in completion
        ):
            raise ValueError(
                "reference backend completion is invalid"
            )
        return completion

    def generate_reference_with_step_logits(
        self,
        *,
        scenario,
        prompt_token_ids,
        generated_tokens,
        generation_policy,
    ):
        if self.closed:
            raise RuntimeError("official reference executor is closed")
        if scenario not in reference.REFERENCE_SCENARIOS:
            raise ValueError(
                f"unsupported reference scenario: {scenario}"
            )
        if (
            not isinstance(prompt_token_ids, (list, tuple))
            or not prompt_token_ids
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in prompt_token_ids
            )
        ):
            raise ValueError("reference prompt token IDs are invalid")
        if (
            isinstance(generated_tokens, bool)
            or not isinstance(generated_tokens, int)
            or generated_tokens <= 0
        ):
            raise ValueError(
                "reference generated token count is invalid"
            )
        if generation_policy != reference.GENERATION_POLICY:
            raise ValueError("reference generation policy mismatch")
        prompt = list(prompt_token_ids)
        result = self._backend().generate_greedy_with_step_logits(
            prompt_token_ids=prompt,
            generated_tokens=generated_tokens,
            generation_policy=dict(generation_policy),
        )
        if (
            not isinstance(result, dict)
            or set(result) != {"output_token_ids", "step_logits"}
        ):
            raise ValueError(
                "reference backend stepwise output is invalid"
            )
        output = result["output_token_ids"]
        logits = result["step_logits"]
        if not isinstance(output, (list, tuple)):
            raise ValueError("reference backend output is invalid")
        output = list(output)
        expected_length = len(prompt) + generated_tokens
        if len(output) != expected_length:
            raise ValueError("reference backend output length mismatch")
        if output[:len(prompt)] != prompt:
            raise ValueError("reference backend prompt prefix mismatch")
        completion = output[len(prompt):]
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in completion
        ):
            raise ValueError(
                "reference backend completion is invalid"
            )
        if (
            not isinstance(logits, (list, tuple))
            or len(logits) != generated_tokens
        ):
            raise ValueError(
                "reference backend step logits count mismatch"
            )
        return {
            "output_token_ids": completion,
            "step_logits": list(logits),
        }

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.backend is not None:
            self.backend.close()


def build_official_reference_executor_factory(
    configuration,
    *,
    backend_factory=_default_backend_factory,
):
    configuration = _normalize_configuration(configuration)

    def factory():
        return OfficialGreedyReferenceExecutor(
            configuration,
            backend_factory=backend_factory,
        )

    return factory
