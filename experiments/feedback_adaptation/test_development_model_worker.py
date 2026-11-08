import copy
from contextlib import nullcontext
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


WORKER_PATH = Path(__file__).with_name("development_model_worker.py")
GPU_UUID = "GPU-00000000-0000-4000-8000-000000000001"


def request(number=0, content="Choose A."):
    return {
        "id": f"{number:032x}",
        "messages": [{"role": "user", "content": content}],
    }


def generated():
    return {
        "text": "A", "terminated": True, "input_ids": [12, 34],
        "output_ids": [65, 99], "generation_ns": 10, "call_ns": 20,
        "prompt_sha256": "a" * 64,
    }


def lines(*requests):
    return io.StringIO("".join(
        json.dumps(value, ensure_ascii=False) + "\n" for value in requests
    ))


class DevelopmentModelWorkerTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(WORKER_PATH.is_file(), "standalone worker is not implemented")
        spec = importlib.util.spec_from_file_location("development_model_worker", WORKER_PATH)
        self.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.worker)

    def test_request_accepts_only_single_public_user_message(self):
        self.assertIsNone(self.worker.validate_request(request()))
        self.worker.validate_request(request(content="字" * 32000))
        for value in (None, [], {}, {"id": "0" * 32}, {**request(), "oracle": []}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.worker.validate_request(value)

    def test_request_ids_are_exact_hex_strings(self):
        self.worker.validate_request({**request(), "id": "A" * 32})
        for identity in (True, 123, None, "x" * 32, "a" * 31, "a" * 33, "a" * 32 + "\n"):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                self.worker.validate_request({**request(), "id": identity})

    def test_request_rejects_private_message_fields_and_wrong_roles(self):
        for messages in (
            [], [request()["messages"][0]] * 2, {}, None,
            [{"role": "system", "content": "A"}],
            [{"role": "user", "content": "A", "task": "private"}],
            [{"role": "user", "content": ""}],
            [{"role": "user", "content": " \n"}],
            [{"role": "user", "content": 1}],
            [{"role": "user", "content": "a" * 32001}],
            [{"role": "user", "content": "\ud800"}],
        ):
            with self.subTest(messages=str(messages)[:100]), self.assertRaises(ValueError):
                self.worker.validate_request({**request(), "messages": messages})

    def test_serve_binds_canonical_request_and_emits_each_result_once(self):
        events, calls = [], []
        original = request(content="选择 A")
        expected_digest = hashlib.sha256(json.dumps(
            original, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")).hexdigest()

        def generate(value):
            calls.append(copy.deepcopy(value))
            value["id"] = "f" * 32
            return generated()

        self.assertEqual(self.worker.serve(lines(original), generate, events.append), 1)
        self.assertEqual(calls, [original])
        self.assertEqual(events, [{
            "event": "output", "id": original["id"],
            "request_sha256": expected_digest, **generated(),
        }])

    def test_serve_rejects_duplicate_or_case_variant_id_before_generation(self):
        for second_id in ("a" * 32, "A" * 32):
            calls, events = [], []

            def generate(value):
                calls.append(value)
                return generated()

            with self.subTest(second_id=second_id), self.assertRaises(ValueError):
                self.worker.serve(
                    lines({**request(), "id": "a" * 32}, {**request(), "id": second_id}),
                    generate, events.append,
                )
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(events), 1)

    def test_serve_caps_calls_before_145th_generation(self):
        calls, events = [], []

        def generate(value):
            calls.append(value)
            return generated()

        with self.assertRaises(ValueError):
            self.worker.serve(lines(*(request(index) for index in range(145))),
                              generate, events.append)
        self.assertEqual(len(calls), 144)
        self.assertEqual(len(events), 144)

    def test_serve_validates_cap_and_supports_lower_cap(self):
        for cap in (0, -1, 145, True, 1.0, "144"):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                self.worker.serve(io.StringIO(""), lambda value: self.fail(), self.fail, cap)
        events = []
        self.assertEqual(self.worker.serve(lines(request()), lambda value: generated(),
                                          events.append, 1), 1)
        self.assertEqual(self.worker.serve(io.StringIO(""), self.fail, self.fail), 0)

    def test_serve_rejects_json_errors_before_generation(self):
        invalid = (
            "NaN\n", "null\n", "\n", "{bad}\n",
            '{"id":"%s","id":"%s","messages":[]}\n' % ("0" * 32, "1" * 32),
            json.dumps({**request(), "oracle": True}) + "\n",
            json.dumps(request()).replace('"Choose A."', "Infinity") + "\n",
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.worker.serve(io.StringIO(raw), self.fail, self.fail)
        with self.assertRaises(ValueError):
            self.worker.serve(io.BytesIO(b"\xff\n"), self.fail, self.fail)

    def test_serve_line_limit_is_utf8_bytes_not_characters(self):
        raw = json.dumps(request(content="😀" * 32000), ensure_ascii=False)
        padding = 131072 - len(raw.encode("utf-8")) - 1
        accepted = raw + " " * padding + "\n"
        for stream in (io.StringIO(accepted), io.BytesIO(accepted.encode("utf-8"))):
            self.assertEqual(self.worker.serve(stream, lambda value: generated(),
                                              lambda value: None), 1)
        for raw_line in (accepted[:-1] + " \n", " " * 131073):
            for stream in (io.StringIO(raw_line), io.BytesIO(raw_line.encode("utf-8"))):
                with self.subTest(length=len(raw_line)), self.assertRaises(ValueError):
                    self.worker.serve(stream, self.fail, self.fail)

    def test_serve_never_requests_an_unbounded_read(self):
        class BoundedStream(io.StringIO):
            def readline(inner, size=-1):
                self.assertGreater(size, 0)
                self.assertLessEqual(size, 131072)
                return super().readline(size)

        self.assertEqual(self.worker.serve(
            BoundedStream(json.dumps(request())), lambda value: generated(),
            lambda value: None,
        ), 1)

    def test_serve_bounds_text_stream_consumption_before_oversize_rejection(self):
        stream = io.StringIO("😀" * 131072)
        with self.assertRaises(ValueError):
            self.worker.serve(stream, self.fail, self.fail)
        self.assertLessEqual(stream.tell(), 131072 // 4 + 1)

    def test_oversize_binary_lookahead_is_at_most_one_byte(self):
        stream = io.BytesIO(b" " * 262144)
        with self.assertRaises(ValueError):
            self.worker.serve(stream, self.fail, self.fail)
        self.assertEqual(stream.tell(), 131073)

    def test_model_index_cannot_reference_unverified_or_external_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for shard in ("../weights.safetensors", "missing.safetensors"):
                payload = json.dumps({"weight_map": {"weight": shard}}).encode()
                (root / "model.safetensors.index.json").write_bytes(payload)
                with self.subTest(shard=shard), self.assertRaises(ValueError):
                    self.worker.check_model(root, {
                        "model.safetensors.index.json": hashlib.sha256(payload).hexdigest(),
                    })

    def test_worker_import_has_only_standard_library_top_level_dependencies(self):
        import ast

        tree = ast.parse(WORKER_PATH.read_text())
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertIn(alias.name.split(".")[0], sys.stdlib_module_names)
            if isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
                self.assertIn(node.module.split(".")[0], sys.stdlib_module_names)
        self.assertNotIn("run_model_choice", WORKER_PATH.read_text())

    def test_serve_generation_and_emit_failures_never_retry(self):
        calls, events = [], []

        def fail_generation(value):
            calls.append(value)
            raise RuntimeError("generation failed")

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            self.worker.serve(lines(request(), request(1)), fail_generation, events.append)
        self.assertEqual(len(calls), 1)
        self.assertEqual(events, [])

        def generate(value):
            calls.append(value)
            return generated()

        def fail_emit(value):
            events.append(value)
            raise RuntimeError("emit failed")

        calls.clear()
        with self.assertRaisesRegex(RuntimeError, "emit failed"):
            self.worker.serve(lines(request(), request(1)), generate, fail_emit)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(events), 1)

    def test_generated_fields_are_strict_and_cannot_override_binding(self):
        invalid = [
            {**generated(), "event": "ready"}, {**generated(), "id": "f" * 32},
            {**generated(), "input_ids": [True]}, {**generated(), "output_ids": [False]},
            {**generated(), "output_ids": [-1]}, {**generated(), "terminated": 1},
            {**generated(), "call_ns": True}, {**generated(), "generation_ns": -1},
            {**generated(), "generation_ns": 21}, {**generated(), "text": 1},
            {**generated(), "prompt_sha256": "bad"},
            {**generated(), "input_ids": [1] * 4097},
            {**generated(), "output_ids": [1] * 33},
        ]
        for result in invalid:
            with self.subTest(result=str(result)[:80]), self.assertRaises(ValueError):
                self.worker.serve(lines(request()), lambda value: result, self.fail)

    def test_gpu_requires_pinned_idle_visible_target(self):
        devices = f"0, GPU-other, 40000, 100\n7, {GPU_UUID}, 100, 0\n"
        self.worker.validate_gpu(devices, "GPU-other, 123\n", GPU_UUID, GPU_UUID, 7)
        for queried, processes, visible, index in (
            (devices, "", "7", 7), (devices, "", GPU_UUID, 6),
            (devices, f"{GPU_UUID}, 123\n", GPU_UUID, 7),
            (f"7, {GPU_UUID}, 101, 0", "", GPU_UUID, 7),
            (f"7, {GPU_UUID}, 0, 1", "", GPU_UUID, 7),
            (devices + devices, "", GPU_UUID, 7),
            ("malformed", "", GPU_UUID, 7),
            (devices, "malformed", GPU_UUID, 7),
            (f"7, {GPU_UUID}, N/A, 0", "", GPU_UUID, 7),
            (devices, "", GPU_UUID, True),
        ):
            with self.subTest(queried=queried), self.assertRaises(ValueError):
                self.worker.validate_gpu(queried, processes, visible, GPU_UUID, index)

    def test_model_hashes_detect_changes_and_unverified_loading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_bytes(b"{}")
            hashes = {"config.json": hashlib.sha256(b"{}").hexdigest()}
            self.assertEqual(self.worker.check_model(root, hashes), hashes)
            (root / "tokenizer.model").write_bytes(b"unverified")
            with self.assertRaises(ValueError):
                self.worker.check_model(root, hashes)
            (root / "tokenizer.model").unlink()
            (root / "config.json").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                self.worker.check_model(root, hashes)

    def test_model_hashes_reject_paths_symlink_escape_and_bad_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model"
            root.mkdir()
            outside = Path(directory) / "outside.json"
            outside.write_bytes(b"{}")
            (root / "config.json").symlink_to(outside)
            for hashes in (
                {}, [], {"../outside.json": "a" * 64},
                {str(outside): "a" * 64}, {"config.json": "a" * 64},
                {"nested/file": "a" * 64}, {"config.json": True},
            ):
                with self.subTest(hashes=hashes), self.assertRaises(ValueError):
                    self.worker.check_model(root, hashes)

    def test_preparation_uses_effective_runtime_config_and_explicit_overrides(self):
        self.assertTrue(hasattr(self.worker, "prepare_generation"), "runtime preparation missing")
        prepared_calls = []

        class Config:
            def __init__(inner, **kwargs):
                inner.__dict__.update(kwargs)

            def get_generation_mode(inner):
                return SimpleNamespace(value="greedy_search")

        class Model:
            generation_config = SimpleNamespace(eos_token_id=[99])

            def _prepare_generation_config(inner, config, **kwargs):
                prepared_calls.append(kwargs)
                return config, {}

        config, mode = self.worker.prepare_generation(
            Model(), SimpleNamespace(pad_token_id=0), Config,
        )
        self.assertEqual(prepared_calls, [{"use_model_defaults": False, "do_sample": False}])
        self.assertEqual(mode, "greedy_search")
        self.assertEqual(config.max_new_tokens, 32)
        self.assertFalse(config.do_sample)
        self.assertEqual(config.num_beams, 1)

    def test_preparation_rejects_effective_sampling_even_if_requested_greedy(self):
        self.assertTrue(hasattr(self.worker, "prepare_generation"), "runtime preparation missing")

        class Config:
            def __init__(inner, **kwargs):
                inner.__dict__.update(kwargs)

            def get_generation_mode(inner):
                return SimpleNamespace(value="sample")

        class Model:
            generation_config = SimpleNamespace(eos_token_id=[99])

            def _prepare_generation_config(inner, config, **kwargs):
                return config, {}

        with self.assertRaisesRegex(ValueError, "greedy"):
            self.worker.prepare_generation(Model(), SimpleNamespace(pad_token_id=0), Config)

    def runtime_fakes(self, input_ids=None, output_ids=None, suffix=True):
        calls = []
        input_ids = [12, 34] if input_ids is None else input_ids
        output_ids = [65, 99] if output_ids is None else output_ids

        class Tensor:
            def __init__(inner, values):
                inner.values = values
                inner.shape = (len(values),)

            def __getitem__(inner, key):
                value = inner.values[key]
                return Tensor(value) if isinstance(value, list) else value

            def to(inner, device):
                calls.append(("tensor_to", device))
                return inner

            def tolist(inner):
                return inner.values

        class Tokenizer:
            pad_token_id = 0

            @classmethod
            def from_pretrained(cls, path, **kwargs):
                calls.append(("tokenizer_load", path, kwargs))
                return cls()

            def apply_chat_template(inner, messages, **kwargs):
                calls.append(("template", messages, kwargs))
                return "public prompt" + ("<think>\n\n</think>\n\n" if suffix else "")

            def __call__(inner, prompt, **kwargs):
                calls.append(("tokenize", kwargs))
                return {"input_ids": Tensor([input_ids])}

            def decode(inner, values, **kwargs):
                calls.append(("decode", values, kwargs))
                return "A"

        class Config:
            def __init__(inner, **kwargs):
                inner.__dict__.update(kwargs)

            def get_generation_mode(inner):
                return SimpleNamespace(value="greedy_search")

            def to_dict(inner):
                return inner.__dict__.copy()

        class Model:
            generation_config = SimpleNamespace(eos_token_id=[99])

            @classmethod
            def from_pretrained(cls, path, **kwargs):
                calls.append(("model_load", path, kwargs))
                return cls()

            def to(inner, device):
                calls.append(("model_to", device))
                return inner

            def eval(inner):
                calls.append(("eval",))
                return inner

            def _prepare_generation_config(inner, config, **kwargs):
                calls.append(("prepare", kwargs))
                return config, {}

            def generate(inner, **kwargs):
                calls.append(("generate", kwargs))
                return Tensor([input_ids + output_ids])

        def record(name, result=None):
            def method(*args):
                calls.append((name, *args))
                return result
            return method

        torch = SimpleNamespace(
            bfloat16="bf16", set_num_threads=record("threads"),
            manual_seed=record("seed"), inference_mode=nullcontext,
            cuda=SimpleNamespace(
                device_count=record("device_count", 1),
                manual_seed_all=record("cuda_seed"),
                synchronize=record("synchronize"),
                max_memory_allocated=record("allocated", 1024),
                max_memory_reserved=record("reserved", 2048),
            ),
        )
        transformers = SimpleNamespace(
            AutoTokenizer=Tokenizer, AutoModelForCausalLM=Model, GenerationConfig=Config,
        )
        return calls, torch, transformers

    def test_generation_is_one_call_with_raw_eos_and_explicit_overrides(self):
        self.assertTrue(hasattr(self.worker, "generate_request"), "runtime generation missing")
        calls, torch, transformers = self.runtime_fakes()
        model = transformers.AutoModelForCausalLM()
        tokenizer = transformers.AutoTokenizer()
        config, mode = self.worker.prepare_generation(model, tokenizer, transformers.GenerationConfig)
        result = self.worker.generate_request(request(), model, tokenizer, torch, config)
        self.assertEqual(result["output_ids"], [65, 99])
        self.assertEqual(result["input_ids"], [12, 34])
        self.assertTrue(result["terminated"])
        self.assertGreaterEqual(result["call_ns"], result["generation_ns"])
        self.assertEqual(result["prompt_sha256"], hashlib.sha256(
            b"public prompt<think>\n\n</think>\n\n",
        ).hexdigest())
        generations = [call for call in calls if call[0] == "generate"]
        self.assertEqual(len(generations), 1)
        self.assertFalse(generations[0][1]["use_model_defaults"])
        self.assertFalse(generations[0][1]["do_sample"])
        self.assertNotIn("past_key_values", generations[0][1])

    def test_generation_validates_prompt_and_token_budget_before_device_transfer(self):
        self.assertTrue(hasattr(self.worker, "generate_request"), "runtime generation missing")
        for kwargs in ({"suffix": False}, {"input_ids": []}, {"input_ids": [1] * 4097}):
            calls, torch, transformers = self.runtime_fakes(**kwargs)
            model, tokenizer = transformers.AutoModelForCausalLM(), transformers.AutoTokenizer()
            config, mode = self.worker.prepare_generation(model, tokenizer, transformers.GenerationConfig)
            with self.subTest(kwargs=str(kwargs)[:50]), self.assertRaises(ValueError):
                self.worker.generate_request(request(), model, tokenizer, torch, config)
            self.assertFalse(any(call[0] in {"generate", "tensor_to", "synchronize"} for call in calls))

    def test_generation_checks_raw_token_limit_and_marks_non_eos_truncation(self):
        self.assertTrue(hasattr(self.worker, "generate_request"), "runtime generation missing")
        for output_ids in ([1] * 32, [1] * 33):
            calls, torch, transformers = self.runtime_fakes(output_ids=output_ids)
            model, tokenizer = transformers.AutoModelForCausalLM(), transformers.AutoTokenizer()
            config, mode = self.worker.prepare_generation(model, tokenizer, transformers.GenerationConfig)
            if len(output_ids) == 33:
                with self.assertRaises(ValueError):
                    self.worker.generate_request(request(), model, tokenizer, torch, config)
            else:
                result = self.worker.generate_request(request(), model, tokenizer, torch, config)
                self.assertFalse(result["terminated"])
                self.assertEqual(result["output_ids"], output_ids)

    def launch_fixture(self, directory):
        root = Path(directory)
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_bytes(b"{}")
        source = root / WORKER_PATH.name
        source.write_bytes(WORKER_PATH.read_bytes())
        launch = {
            "run_id": "b" * 32, "hostname": platform.node(), "uid": os.getuid(),
            "gpu_uuid": GPU_UUID, "gpu_index": 7, "model_dir": str(model),
            "model_hashes": {"config.json": hashlib.sha256(b"{}").hexdigest()},
            "worker_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "max_calls": 144,
        }
        (root / "launch.json").write_text(json.dumps(launch), encoding="utf-8")
        return root, source, launch

    def test_main_admits_before_import_load_and_emits_ready_output_finished(self):
        self.assertTrue(hasattr(self.worker, "main"), "runtime entrypoint missing")
        calls, torch, transformers = self.runtime_fakes()
        events = []
        with tempfile.TemporaryDirectory() as directory:
            root, source, launch = self.launch_fixture(directory)

            def query(argv, **kwargs):
                self.assertFalse(any(call[0] in {"device_count", "model_load"} for call in calls))
                calls.append(("query", argv))
                return f"7, {GPU_UUID}, 0, 0\n" if "--query-gpu" in argv[1] else ""

            with (
                patch.object(self.worker, "__file__", str(source)),
                patch.dict(sys.modules, {"torch": torch, "transformers": transformers}),
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU_UUID}),
                patch("importlib.metadata.version", side_effect=self.worker.EXPECTED_VERSIONS.__getitem__),
                patch("platform.python_version", return_value="3.12.13"),
                patch("subprocess.check_output", side_effect=query),
            ):
                self.worker.main(root, lines(request()), events.append)
        self.assertEqual([event["event"] for event in events], ["ready", "output", "finished"])
        self.assertEqual(events[0]["effective_mode"], "greedy_search")
        self.assertEqual(events[0]["run_id"], launch["run_id"])
        self.assertEqual(events[2]["calls"], 1)
        self.assertEqual(events[2]["peak_allocated_bytes"], 1024)
        self.assertEqual(events[2]["peak_reserved_bytes"], 2048)
        self.assertEqual(sum(call[0] == "generate" for call in calls), 1)
        model_load = next(call for call in calls if call[0] == "model_load")
        self.assertEqual(model_load[2], {
            "local_files_only": True, "trust_remote_code": False, "torch_dtype": "bf16",
            "low_cpu_mem_usage": False, "attn_implementation": "eager",
        })
        self.assertEqual(calls[0][0], "query")
        self.assertEqual(calls[1][0], "query")

    def test_main_rejects_invalid_launch_before_gpu_queries_or_imports(self):
        self.assertTrue(hasattr(self.worker, "main"), "runtime entrypoint missing")
        for changed in (
            {"hostname": "not-this-host"}, {"uid": True}, {"worker_sha256": "0" * 64},
            {"extra": 1}, {"max_calls": 145}, {"gpu_index": True}, {"run_id": "bad"},
        ):
            with tempfile.TemporaryDirectory() as directory:
                root, source, launch = self.launch_fixture(directory)
                (root / "launch.json").write_text(json.dumps({**launch, **changed}))
                with (
                    patch.object(self.worker, "__file__", str(source)),
                    patch("subprocess.check_output", side_effect=AssertionError("GPU query before admission")),
                    self.subTest(changed=changed),
                    self.assertRaises(ValueError),
                ):
                    self.worker.main(root, io.StringIO(""), self.fail)

    def test_main_rejects_versions_and_busy_gpu_without_cuda_calls(self):
        self.assertTrue(hasattr(self.worker, "main"), "runtime entrypoint missing")
        for wrong_version in (True, False):
            calls, torch, transformers = self.runtime_fakes()
            with tempfile.TemporaryDirectory() as directory:
                root, source, launch = self.launch_fixture(directory)
                with (
                    patch.object(self.worker, "__file__", str(source)),
                    patch.dict(sys.modules, {"torch": torch, "transformers": transformers}),
                    patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU_UUID}),
                    patch("platform.python_version", return_value="3.12.13"),
                    patch("importlib.metadata.version", side_effect=(
                        lambda name: "wrong" if wrong_version else self.worker.EXPECTED_VERSIONS[name]
                    )),
                    patch("subprocess.check_output", side_effect=[
                        f"7, {GPU_UUID}, 1000, 99\n", "",
                    ]) as queries,
                    self.assertRaises(ValueError),
                ):
                    self.worker.main(root, io.StringIO(""), self.fail)
                self.assertEqual(calls, [])
                if wrong_version:
                    queries.assert_not_called()

    def test_empty_runtime_includes_import_and_setup_in_preparation_time(self):
        calls, torch, transformers = self.runtime_fakes()
        clock = [0]
        events = []

        def set_seed(seed):
            clock[0] += 100

        torch.manual_seed = set_seed
        with tempfile.TemporaryDirectory() as directory:
            root, source, launch = self.launch_fixture(directory)
            with (
                patch.object(self.worker, "__file__", str(source)),
                patch.dict(sys.modules, {"torch": torch, "transformers": transformers}),
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU_UUID}),
                patch("platform.python_version", return_value="3.12.13"),
                patch("importlib.metadata.version", side_effect=self.worker.EXPECTED_VERSIONS.__getitem__),
                patch("subprocess.check_output", side_effect=[f"7, {GPU_UUID}, 0, 0\n", ""]),
                patch("time.perf_counter_ns", side_effect=lambda: clock[0]),
            ):
                self.worker.main(root, io.StringIO(""), events.append)
        self.assertEqual([event["event"] for event in events], ["ready", "finished"])
        self.assertEqual(events[0]["preparation_ns"], 100)
        self.assertEqual(events[1]["calls"], 0)
        self.assertFalse(any(call[0] == "generate" for call in calls))

    def test_effective_preparation_rejects_unused_api_overrides_and_bad_eos(self):
        for eos_ids, unused in (([99], {"use_model_defaults": False}), ([True], {}), (None, {})):
            calls, torch, transformers = self.runtime_fakes()
            model = transformers.AutoModelForCausalLM()
            model.generation_config = SimpleNamespace(eos_token_id=eos_ids)
            model._prepare_generation_config = lambda config, **kwargs: (config, unused)
            with self.subTest(eos_ids=eos_ids, unused=unused), self.assertRaises(ValueError):
                self.worker.prepare_generation(
                    model, transformers.AutoTokenizer(), transformers.GenerationConfig,
                )

    def test_isolated_cli_reports_fatal_without_loading_runtime(self):
        result = subprocess.run(
            [sys.executable, "-I", "-B", str(WORKER_PATH)],
            capture_output=True, text=True, timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(result.stdout.strip(), "fatal JSON event missing")
        event = json.loads(result.stdout)
        self.assertEqual(event["event"], "fatal")
        self.assertEqual(event["error_type"], "ValueError")
        self.assertIsInstance(event["message"], str)


if __name__ == "__main__":
    unittest.main()
