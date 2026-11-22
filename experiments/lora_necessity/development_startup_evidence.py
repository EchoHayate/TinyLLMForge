from pilot_protocol import require
from validation_contract import parse_json


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


def verify_startup(stdout_raw):
    """Check pinned pre-input diagnostics, not actual isolation or later execution."""
    require(type(stdout_raw) is bytes and 0 < len(stdout_raw) <= 65536,
            "startup stdout bytes")
    require(stdout_raw.endswith(b"\n") and b"\r" not in stdout_raw,
            "startup stdout framing")
    lines = stdout_raw[:-1].split(b"\n")
    require(len(lines) == 4, "startup stdout line count")
    require(lines[0] in (b"USERNS_POLICY_enabled", b"USERNS_POLICY_unavailable")
            and lines[1] == b"HOST_NO_NEW_PRIVILEGES"
            and lines[2] == b"ALL_CAPABILITIES_ZERO", "startup markers")
    record = parse_json(lines[3], limit=65536)
    require(type(record) is dict and set(record) == {
        "kind", "python", "checks", "passed", "capabilities", "allowed_syscalls",
        "generated_candidate_executed", "candidate_execution_admitted",
    }, "startup record fields")
    require(record["kind"] == "FIXED_HARDENING_SMOKE_ONLY"
            and record["python"] == "3.12.13" and record["passed"] is True,
            "startup record identity")
    checks = record["checks"]
    require(type(checks) is dict and set(checks) == set(CHECKS)
            and all(value is True for value in checks.values()), "startup checks")
    require(record["capabilities"] == {
        name: "0000000000000000" for name in CAPABILITIES
    }, "startup capabilities")
    require(record["allowed_syscalls"] == list(SYSCALLS), "startup syscall policy")
    require(record["generated_candidate_executed"] is False
            and record["candidate_execution_admitted"] is False, "startup phase")
    return record
