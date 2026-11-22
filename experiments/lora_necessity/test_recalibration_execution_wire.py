import ast
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
import types
import unittest


MODULES_BEFORE = frozenset(sys.modules)
if importlib.util.find_spec("recalibration_execution_wire") is not None:
    import recalibration_execution_wire as wire
else:
    wire = None
MODULES_AFTER = frozenset(sys.modules)

FRAME_NAMES = (
    "ready_raw", "backend_stdout_raw", "backend_stderr_raw", "diagnostic_raw", "done_raw",
)
FRAME_LIMITS = (4096, 8388608, 131072, 65536, 4096)
PAYLOADS = (b"r", b"\x00\xff", b"e", b"d", b"f")


class BytesSubclass(bytes):
    pass


class IntSubclass(int):
    pass


class StringSubclass(str):
    pass


def frame(channel, payload, declared=None, magic=b"LRV01"):
    size = len(payload) if declared is None else declared
    return magic + bytes([channel]) + size.to_bytes(8, "big") + payload


def complete_wire(payloads=PAYLOADS):
    return b"".join(frame(channel, payload)
                    for channel, payload in enumerate(payloads, 1))


class WireDecoderTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(wire, "wire decoder not implemented")

    def decode(self, raw):
        return wire.decode_wire(raw, expected_sha256=hashlib.sha256(raw).hexdigest())

    def test_independent_golden_header_and_big_endian_length(self):
        payload = bytes(range(256)) + b"\x00\xff"
        header = b"LRV01\x02\x00\x00\x00\x00\x00\x00\x01\x02"
        self.assertEqual(len(header), 14)
        self.assertEqual(len(payload), 258)
        self.assertEqual(wire.decode_frame(header + payload, expected_channel=2), payload)

    def test_fixed_constants(self):
        self.assertEqual(wire.MAGIC, b"LRV01")
        self.assertIs(type(wire.MAGIC), bytes)
        self.assertEqual(wire.HEADER.format, "!5sBQ")
        self.assertEqual(wire.HEADER.size, 14)
        self.assertIs(type(wire.WIRE_LIMIT), int)
        self.assertEqual(wire.WIRE_LIMIT, 8388608)
        self.assertIs(type(wire.FRAME_LIMITS), tuple)
        self.assertEqual(wire.FRAME_LIMITS, FRAME_LIMITS)
        self.assertIs(type(wire.FRAME_NAMES), tuple)
        self.assertEqual(wire.FRAME_NAMES, FRAME_NAMES)

    def test_preserves_all_opaque_bytes(self):
        payloads = (
            b'{"unverified":true}', b"\x00\xff\n", b"\xfe",
            b"diagnostic\x00", b'{"failed":true}',
        )
        decoded = self.decode(complete_wire(payloads))
        self.assertEqual(list(decoded.values()), list(payloads))
        self.assertEqual(tuple(decoded), FRAME_NAMES)
        for value in decoded.values():
            self.assertIs(type(value), bytes)

    def test_embedded_headers_are_payload_not_delimiters(self):
        payload = b"\x00LRV01\x05\x00\x00\x00\x00\x00\x00\x00\x00\xff\n"
        payloads = (payload,) * 5
        self.assertEqual(list(self.decode(complete_wire(payloads)).values()), list(payloads))
        for channel in range(1, 6):
            with self.subTest(channel=channel):
                self.assertEqual(
                    wire.decode_frame(frame(channel, payload), expected_channel=channel),
                    payload,
                )

    def test_empty_payloads_are_valid_opaque_frames(self):
        payloads = (b"",) * 5
        decoded = self.decode(complete_wire(payloads))
        self.assertEqual(decoded, dict.fromkeys(FRAME_NAMES, b""))
        self.assertEqual(tuple(decoded), FRAME_NAMES)
        for channel in range(1, 6):
            with self.subTest(channel=channel):
                self.assertEqual(wire.decode_frame(frame(channel, b""),
                                                   expected_channel=channel), b"")

    def test_non_json_control_and_duplicate_fields_are_not_interpreted(self):
        for payload in (b"\xff\xfe\x00\r\n", b"not-json", b'{"a":1,"a":2}',
                        b'{"success":true,"authorized":true,"extra":1}'):
            with self.subTest(payload=payload):
                decoded = self.decode(complete_wire((payload,) * 5))
                self.assertEqual(tuple(decoded), FRAME_NAMES)
                self.assertEqual(list(decoded.values()), [payload] * 5)

    def test_fresh_deterministic_dicts_do_not_alias_or_mutate_inputs(self):
        raw = complete_wire()
        original_digest = hashlib.sha256(raw).hexdigest()
        first = self.decode(raw)
        second = self.decode(raw)
        expected = dict(zip(FRAME_NAMES, PAYLOADS))
        self.assertIs(type(first), dict)
        self.assertIsNot(first, second)
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        first["ready_raw"] = b"changed"
        first.clear()
        second["diagnostic_raw"] = b"changed again"
        self.assertEqual(self.decode(raw), expected)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), original_digest)
        self.assertEqual(wire.FRAME_NAMES, FRAME_NAMES)
        self.assertEqual(wire.FRAME_LIMITS, FRAME_LIMITS)

    def test_each_frame_decodes_without_a_complete_wire(self):
        for channel, payload in enumerate(PAYLOADS, 1):
            with self.subTest(channel=channel):
                decoded = wire.decode_frame(frame(channel, payload),
                                            expected_channel=channel)
                self.assertIs(type(decoded), bytes)
                self.assertEqual(decoded, payload)

    def test_complete_wire_is_not_one_exact_frame(self):
        with self.assertRaisesRegex(ValueError, "trailing frame bytes"):
            wire.decode_frame(complete_wire(), expected_channel=1)

    def test_strict_frame_and_wire_bytes_types(self):
        for raw in (frame(1, b""), complete_wire()):
            for invalid in (BytesSubclass(raw), bytearray(raw), memoryview(raw),
                            raw.decode("latin1"), None, 1, [raw]):
                with self.subTest(raw_size=len(raw), invalid_type=type(invalid).__name__):
                    with self.assertRaises(ValueError):
                        wire.decode_frame(invalid, expected_channel=1)
                    with self.assertRaises(ValueError):
                        wire.decode_wire(invalid, expected_sha256=hashlib.sha256(raw).hexdigest())

    def test_strict_expected_channel_types_and_range(self):
        for channel in (True, False, IntSubclass(1), 1.0, "1", 0, -1, 6, None):
            with self.subTest(channel=channel, channel_type=type(channel).__name__):
                with self.assertRaises(ValueError):
                    wire.decode_frame(frame(1, b""), expected_channel=channel)

    def test_strict_digest_type_and_syntax(self):
        raw = complete_wire()
        digest = hashlib.sha256(raw).hexdigest()
        invalid_digests = (
            StringSubclass(digest), digest.upper(), digest[:-1], digest + "0",
            digest + "\n", "\n" + digest, " " + digest[1:], "g" * 64, "",
            None, 1, True, 1.0, digest.encode(), bytearray(digest.encode()),
            memoryview(digest.encode()), [digest],
        )
        for invalid in invalid_digests:
            with self.subTest(digest=invalid, digest_type=type(invalid).__name__):
                with self.assertRaisesRegex(ValueError, "expected wire digest"):
                    wire.decode_wire(raw, expected_sha256=invalid)

    def test_wrong_magic_in_every_channel(self):
        for channel in range(1, 6):
            for magic in (b"LRV00", b"LRV02", b"lrv01", b"LRV0\x00", b"\x00RV01"):
                with self.subTest(channel=channel, magic=magic):
                    invalid = frame(channel, b"", magic=magic)
                    with self.assertRaisesRegex(ValueError, "frame magic"):
                        wire.decode_frame(invalid, expected_channel=channel)
                    frames = [frame(ordinal, b"") for ordinal in range(1, 6)]
                    frames[channel - 1] = invalid
                    with self.assertRaisesRegex(ValueError, "frame magic"):
                        self.decode(b"".join(frames))

    def test_wrong_frame_channel_is_rejected(self):
        for expected in range(1, 6):
            for actual in (0, 1, 2, 3, 4, 5, 6, 255):
                if actual == expected:
                    continue
                with self.subTest(expected=expected, actual=actual):
                    with self.assertRaisesRegex(ValueError, "frame channel"):
                        wire.decode_frame(frame(actual, b""), expected_channel=expected)

    def test_unknown_and_repeated_wire_channels_are_rejected(self):
        for position in range(5):
            for channel in (0, 6, 255, ((position + 1) % 5) + 1):
                with self.subTest(position=position, channel=channel):
                    channels = [1, 2, 3, 4, 5]
                    channels[position] = channel
                    with self.assertRaises(ValueError):
                        self.decode(b"".join(frame(actual, b"") for actual in channels))

    def test_missing_and_duplicated_frames_are_rejected(self):
        frames = [frame(channel, b"") for channel in range(1, 6)]
        for position in range(5):
            with self.subTest(missing=position + 1):
                with self.assertRaises(ValueError):
                    self.decode(b"".join(frames[:position] + frames[position + 1:]))
            with self.subTest(duplicated=position + 1):
                with self.assertRaises(ValueError):
                    self.decode(b"".join(
                        frames[:position] + [frames[position]] + frames[position:]))

    def test_swapped_channels_are_rejected(self):
        for position in range(4):
            with self.subTest(position=position):
                frames = [frame(channel, b"") for channel in range(1, 6)]
                frames[position], frames[position + 1] = frames[position + 1], frames[position]
                with self.assertRaises(ValueError):
                    self.decode(b"".join(frames))

    def test_trailing_bytes_are_rejected(self):
        for trailing in (b"\x00", b"\n", frame(5, b""), complete_wire()):
            with self.subTest(trailing=trailing):
                with self.assertRaisesRegex(ValueError, "trailing wire bytes"):
                    self.decode(complete_wire() + trailing)
                with self.assertRaisesRegex(ValueError, "trailing frame bytes"):
                    wire.decode_frame(frame(1, b"") + trailing, expected_channel=1)

    def test_every_wire_prefix_fails_without_salvage(self):
        raw = complete_wire()
        for stop in range(len(raw)):
            with self.subTest(stop=stop):
                with self.assertRaises(ValueError):
                    self.decode(raw[:stop])

    def test_every_frame_header_and_body_prefix_fails(self):
        for channel in range(1, 6):
            raw = frame(channel, b"\x00\xff\n")
            for stop in range(len(raw)):
                with self.subTest(channel=channel, stop=stop):
                    with self.assertRaises(ValueError):
                        wire.decode_frame(raw[:stop], expected_channel=channel)

    def test_smaller_and_larger_declared_lengths_fail(self):
        for channel in range(1, 6):
            for declared in (0, 2, 4):
                with self.subTest(channel=channel, declared=declared):
                    invalid = frame(channel, b"abc", declared=declared)
                    with self.assertRaises(ValueError):
                        wire.decode_frame(invalid, expected_channel=channel)
                    frames = [frame(ordinal, b"") for ordinal in range(1, 6)]
                    frames[channel - 1] = invalid
                    with self.assertRaises(ValueError):
                        self.decode(b"".join(frames))

    def test_oversized_declarations_fail_before_payload_availability(self):
        for channel, limit in enumerate(FRAME_LIMITS, 1):
            for declared in (limit + 1, 2**64 - 1):
                with self.subTest(channel=channel, declared=declared):
                    invalid = frame(channel, b"", declared=declared)
                    self.assertEqual(len(invalid), 14)
                    with self.assertRaisesRegex(ValueError, "frame payload limit"):
                        wire.decode_frame(invalid, expected_channel=channel)
                    frames = [frame(ordinal, b"") for ordinal in range(1, 6)]
                    frames[channel - 1] = invalid
                    with self.assertRaisesRegex(ValueError, "frame payload limit"):
                        self.decode(b"".join(frames))

    def test_little_endian_multibyte_lengths_are_rejected(self):
        for channel in range(1, 6):
            with self.subTest(channel=channel):
                invalid = b"LRV01" + bytes([channel]) + (258).to_bytes(8, "little")
                invalid += b"\xff" * 258
                with self.assertRaises(ValueError):
                    wire.decode_frame(invalid, expected_channel=channel)
                frames = [frame(ordinal, b"") for ordinal in range(1, 6)]
                frames[channel - 1] = invalid
                with self.assertRaises(ValueError):
                    self.decode(b"".join(frames))

    def test_single_frame_payload_limits_are_inclusive(self):
        for channel, limit in enumerate(FRAME_LIMITS, 1):
            with self.subTest(channel=channel, limit=limit):
                payload = b"\xff" * limit
                raw = frame(channel, payload)
                self.assertEqual(wire.decode_frame(raw, expected_channel=channel), payload)
                del payload, raw

    def test_single_frame_payload_limit_plus_one_is_rejected(self):
        for channel, limit in enumerate(FRAME_LIMITS, 1):
            with self.subTest(channel=channel, limit=limit):
                raw = frame(channel, b"\xff" * (limit + 1))
                with self.assertRaises(ValueError):
                    wire.decode_frame(raw, expected_channel=channel)
                del raw

    def test_whole_wire_limit_includes_all_seventy_header_bytes(self):
        for extra in (0, 1):
            with self.subTest(extra=extra):
                payload = b"\xff" * (8388608 - 70 + extra)
                raw = complete_wire((b"", payload, b"", b"", b""))
                self.assertEqual(len(raw), 8388608 + extra)
                self.assertLessEqual(len(payload), 8388608)
                if extra:
                    with self.assertRaisesRegex(ValueError, "wire bytes"):
                        self.decode(raw)
                else:
                    decoded = self.decode(raw)
                    self.assertEqual(tuple(decoded), FRAME_NAMES)
                    self.assertEqual(list(decoded.values()), [b"", payload, b"", b"", b""])
                    del decoded
                del payload, raw

    def test_maximum_stdout_frame_cannot_fit_in_complete_wire(self):
        payload = b"\x00" * 8388608
        raw = frame(2, payload)
        self.assertEqual(wire.decode_frame(raw, expected_channel=2), payload)
        self.assertEqual(len(raw), 8388622)
        del raw
        raw = complete_wire((b"", payload, b"", b"", b""))
        self.assertEqual(len(raw), 8388678)
        with self.assertRaisesRegex(ValueError, "wire bytes"):
            self.decode(raw)

    def test_stale_digest_rejects_same_length_mutations_in_each_payload(self):
        raw = complete_wire()
        digest = hashlib.sha256(raw).hexdigest()
        for position in range(5):
            with self.subTest(channel=position + 1):
                payloads = list(PAYLOADS)
                payloads[position] = bytes([payloads[position][0] ^ 1]) + payloads[position][1:]
                mutated = complete_wire(payloads)
                self.assertEqual(len(mutated), len(raw))
                self.assertNotEqual(mutated, raw)
                with self.assertRaisesRegex(ValueError, "^wire digest$"):
                    wire.decode_wire(mutated, expected_sha256=digest)
                self.assertEqual(list(self.decode(mutated).values()), payloads)

    def test_digest_is_checked_before_any_malformed_header(self):
        for raw in (b"", b"\x00", b"WRONG" + complete_wire()[5:],
                    frame(255, b""), frame(1, b"", declared=2**64 - 1)):
            with self.subTest(raw=raw):
                digest = hashlib.sha256(raw).hexdigest()
                stale = ("0" if digest[0] != "0" else "1") + digest[1:]
                with self.assertRaisesRegex(ValueError, "^wire digest$"):
                    wire.decode_wire(raw, expected_sha256=stale)
                with self.assertRaises(ValueError):
                    wire.decode_wire(raw, expected_sha256=digest)

    def test_public_api_and_keyword_only_signatures(self):
        functions = {
            name for name, value in vars(wire).items()
            if inspect.isfunction(value) and not name.startswith("_")
        }
        self.assertEqual(functions, {"decode_frame", "decode_wire"})
        for name, positional, keyword in (
                ("decode_frame", "frame_raw", "expected_channel"),
                ("decode_wire", "wire_raw", "expected_sha256")):
            with self.subTest(name=name):
                function = getattr(wire, name)
                parameters = inspect.signature(function).parameters
                self.assertEqual(tuple(parameters), (positional, keyword))
                self.assertIs(parameters[positional].kind,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD)
                self.assertIs(parameters[keyword].kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIs(parameters[keyword].default, inspect.Parameter.empty)
                with self.assertRaises(TypeError):
                    function(b"", 1)
                with self.assertRaises(TypeError):
                    function(b"")

    def test_ast_and_module_observations_show_only_pure_dependencies(self):
        source = Path(wire.__file__).read_text()
        tree = ast.parse(source)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
                imports.add(node.module)
                self.assertEqual(node.module, "pilot_protocol")
                self.assertEqual([alias.name for alias in node.names], ["require"])
        self.assertEqual(imports, {"hashlib", "re", "struct", "pilot_protocol"})
        allowed_names = {"len", "type", "enumerate", "_require", "require"}
        allowed_attributes = {"Struct", "unpack_from", "sha256", "hexdigest", "fullmatch"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    self.assertTrue(node.func.id in allowed_names or
                                    node.func.id.startswith("_read_"), node.func.id)
                else:
                    self.assertIsInstance(node.func, ast.Attribute)
                    self.assertIn(node.func.attr, allowed_attributes)
        for node in tree.body:
            self.assertIsInstance(node, (ast.Import, ast.ImportFrom, ast.Assign,
                                         ast.FunctionDef))
        modules = {
            value.__name__ for value in vars(wire).values()
            if isinstance(value, types.ModuleType)
        }
        self.assertLessEqual(modules, {"hashlib", "re", "struct"})
        before_calls = frozenset(sys.modules)
        self.decode(complete_wire())
        wire.decode_frame(frame(1, b""), expected_channel=1)
        introduced = (MODULES_AFTER - MODULES_BEFORE) | (frozenset(sys.modules) - before_calls)
        forbidden = {
            "os", "pathlib", "subprocess", "socket", "urllib", "requests", "http",
            "time", "random", "secrets", "torch", "transformers", "tokenizers",
            "vllm", "mlx", "validation_remote", "validation_task_worker",
            "recalibration_data_contract", "recalibration_execution_package",
        }
        self.assertFalse({name.split(".")[0] for name in introduced} & forbidden)


if __name__ == "__main__":
    unittest.main()
