import copy
import importlib
from pathlib import Path
import unittest

from validation_contract import encoded


CHECKS = (
    "stat_metadata", "lstat_metadata", "seccomp_filter_active",
    "no_new_privileges", "all_capability_sets_zero", "root_and_runtime_readonly",
    "host_directories_absent", "environment_clean", "workdir_fixed",
    "site_packages_hidden", "no_ipv4_routes", "numerical_work", "json_roundtrip",
    "unlisted_getppid_denied", "socket_denied", "fork_denied", "execve_denied",
)
SYSCALLS = (
    "read", "write", "close", "stat", "lstat", "fstat", "newfstatat", "lseek", "pread64",
    "mmap", "mprotect", "munmap", "mremap", "madvise", "brk",
    "rt_sigaction", "rt_sigprocmask", "rt_sigreturn", "sigaltstack",
    "futex", "clock_gettime", "getrandom", "getpid", "gettid",
    "getuid", "geteuid", "getgid", "getegid", "openat", "readlink",
    "getcwd", "getdents64", "fcntl", "ioctl", "statfs", "fstatfs",
    "prlimit64", "getrusage", "wait4", "exit", "exit_group",
)
CAPABILITIES = ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")


def startup_record():
    return {
        "kind": "FIXED_HARDENING_SMOKE_ONLY", "python": "3.12.13",
        "checks": {name: True for name in CHECKS}, "passed": True,
        "capabilities": {name: "0000000000000000" for name in CAPABILITIES},
        "allowed_syscalls": list(SYSCALLS), "generated_candidate_executed": False,
        "candidate_execution_admitted": False,
    }


def startup_bytes(record=None, policy=b"enabled"):
    if record is None:
        record = startup_record()
    return (b"USERNS_POLICY_" + policy + b"\nHOST_NO_NEW_PRIVILEGES\n"
            b"ALL_CAPABILITIES_ZERO\n" + encoded(record) + b"\n")


class StartupEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(
            Path(__file__).with_name("development_startup_evidence.py").is_file(),
            "startup evidence component missing",
        )
        self.component = importlib.import_module("development_startup_evidence")

    def test_both_fixed_user_namespace_markers(self):
        for policy in (b"enabled", b"unavailable"):
            with self.subTest(policy=policy):
                self.assertEqual(self.component.verify_startup(startup_bytes(policy=policy)),
                                 startup_record())

    def test_saved_record_is_independent_and_keeps_pre_input_flags(self):
        raw = startup_bytes()
        result = self.component.verify_startup(raw)
        self.assertIsInstance(result, dict)
        self.assertIs(result["generated_candidate_executed"], False)
        self.assertIs(result["candidate_execution_admitted"], False)
        result["checks"]["stat_metadata"] = False
        self.assertEqual(self.component.verify_startup(raw), startup_record())

    def test_accepts_worker_pretty_json_spacing(self):
        import json

        raw = startup_bytes().rsplit(b"\n", 2)[0] + b"\n"
        raw += json.dumps(startup_record(), sort_keys=True).encode() + b"\n"
        self.assertEqual(self.component.verify_startup(raw), startup_record())

    def test_rejects_wrong_native_type_and_output_limit(self):
        class ByteSubclass(bytes):
            pass

        for raw in (None, "", bytearray(startup_bytes()), ByteSubclass(startup_bytes()),
                    b"", b"x" * 65537):
            with self.subTest(kind=type(raw).__name__), self.assertRaises(ValueError):
                self.component.verify_startup(raw)

    def test_output_byte_boundary_with_otherwise_valid_record(self):
        original = startup_bytes()
        boundary = original[:-1] + b" " * (65536 - len(original)) + b"\n"
        self.assertEqual(len(boundary), 65536)
        self.assertEqual(self.component.verify_startup(boundary), startup_record())
        with self.assertRaises(ValueError):
            self.component.verify_startup(boundary[:-1] + b" \n")

    def test_rejects_wrong_missing_reordered_and_extra_markers(self):
        raw = startup_bytes()
        variants = (
            raw.replace(b"USERNS_POLICY_enabled", b"USERNS_POLICY_disabled"),
            raw.replace(b"HOST_NO_NEW_PRIVILEGES\n", b""),
            raw.replace(b"ALL_CAPABILITIES_ZERO", b"CAPABILITIES_UNKNOWN"),
            raw.replace(b"HOST_NO_NEW_PRIVILEGES\nALL_CAPABILITIES_ZERO",
                        b"ALL_CAPABILITIES_ZERO\nHOST_NO_NEW_PRIVILEGES"),
            b"IDENTITY_OK\n" + raw, raw + b"unexpected output\n",
            raw + raw, raw[:-1], raw.replace(b"\n", b"\r\n"),
        )
        for ordinal, value in enumerate(variants):
            with self.subTest(variant=ordinal), self.assertRaises(ValueError):
                self.component.verify_startup(value)

    def test_rejects_malformed_duplicate_and_nonfinite_json(self):
        prefix = startup_bytes().rsplit(b"\n", 2)[0] + b"\n"
        for suffix in (b"{", b"[]", b"null",
                       encoded(startup_record())[:-1] + b',"passed":true}',
                       encoded(startup_record()).replace(b'"passed":true', b'"passed":NaN'),
                       encoded(startup_record()).replace(b'"passed":true', b'"passed":1e400')):
            with self.subTest(suffix=suffix[:30]), self.assertRaises(ValueError):
                self.component.verify_startup(prefix + suffix + b"\n")

    def test_rejects_every_missing_or_extra_record_field(self):
        for name in startup_record():
            record = startup_record()
            del record[name]
            with self.subTest(missing=name), self.assertRaises(ValueError):
                self.component.verify_startup(startup_bytes(record))
        record = startup_record()
        record["extra"] = True
        with self.assertRaises(ValueError):
            self.component.verify_startup(startup_bytes(record))

    def test_rejects_kind_version_passed_and_phase_substitutions(self):
        variants = {
            "kind": ("different", None, True), "python": ("3.12.12", 3.12, None),
            "passed": (False, 1, "true"),
            "generated_candidate_executed": (True, 0, None),
            "candidate_execution_admitted": (True, 0, None),
        }
        for name, values in variants.items():
            for value in values:
                record = startup_record()
                record[name] = value
                with self.subTest(field=name, value=value), self.assertRaises(ValueError):
                    self.component.verify_startup(startup_bytes(record))

    def test_rejects_each_false_or_nonboolean_check(self):
        for name in CHECKS:
            for value in (False, 1, None):
                record = startup_record()
                record["checks"][name] = value
                with self.subTest(check=name, value=value), self.assertRaises(ValueError):
                    self.component.verify_startup(startup_bytes(record))

    def test_rejects_missing_extra_or_nondictionary_checks(self):
        for name in CHECKS:
            record = startup_record()
            del record["checks"][name]
            with self.subTest(check=name), self.assertRaises(ValueError):
                self.component.verify_startup(startup_bytes(record))
        for checks in ({**startup_record()["checks"], "extra": True}, [], None):
            record = startup_record()
            record["checks"] = checks
            with self.subTest(checks=checks), self.assertRaises(ValueError):
                self.component.verify_startup(startup_bytes(record))

    def test_rejects_capability_drift_and_inexact_zero(self):
        for name in CAPABILITIES:
            for value in ("0000000000000001", "0", 0, False, None):
                record = startup_record()
                record["capabilities"][name] = value
                with self.subTest(capability=name, value=value), self.assertRaises(ValueError):
                    self.component.verify_startup(startup_bytes(record))
            record = startup_record()
            del record["capabilities"][name]
            with self.subTest(missing=name), self.assertRaises(ValueError):
                self.component.verify_startup(startup_bytes(record))
        for capabilities in ([], None, {**startup_record()["capabilities"], "Extra": "0"}):
            record = startup_record()
            record["capabilities"] = capabilities
            with self.subTest(capabilities=capabilities), self.assertRaises(ValueError):
                self.component.verify_startup(startup_bytes(record))

    def test_rejects_syscall_removal_addition_reorder_and_duplicates(self):
        original = list(SYSCALLS)
        variants = (original[:-1], original + ["socket"], original[::-1],
                    original + [original[0]], [], None, {})
        for allowed in variants:
            record = copy.deepcopy(startup_record())
            record["allowed_syscalls"] = allowed
            with self.subTest(allowed=allowed), self.assertRaises(ValueError):
                self.component.verify_startup(startup_bytes(record))


if __name__ == "__main__":
    unittest.main()
