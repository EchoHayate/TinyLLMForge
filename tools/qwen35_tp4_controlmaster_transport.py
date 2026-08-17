from __future__ import annotations

from pathlib import Path
import stat


def _validate_control_path(control_path, stat_fn):
    path = Path(control_path)
    try:
        mode = stat_fn(path).st_mode
    except FileNotFoundError as error:
        raise ValueError("ControlMaster socket is missing") from error
    if not stat.S_ISSOCK(mode):
        raise ValueError("ControlMaster path is not a socket")
    return path


def _has_control_path(argv):
    for index, value in enumerate(argv[1:], start=1):
        if value == "-S":
            return True
        if value == "-o" and index + 1 < len(argv):
            if argv[index + 1].startswith("ControlPath="):
                return True
        if value.startswith("-oControlPath="):
            return True
    return False


def _without_neutral_control_path(argv):
    result = [argv[0]]
    index = 1
    while index < len(argv):
        value = argv[index]
        if (
            value == "-o"
            and index + 1 < len(argv)
            and argv[index + 1] == "ControlPath=none"
        ):
            index += 2
            continue
        if value == "-oControlPath=none":
            index += 1
            continue
        result.append(value)
        index += 1
    return result


def _transport_argv(argv, control_path):
    if argv[0] not in {"ssh", "scp"}:
        return list(argv)
    forwarded = _without_neutral_control_path(argv)
    if _has_control_path(forwarded):
        raise ValueError("transport already defines a control path")
    if argv[0] == "ssh":
        return ["ssh", "-S", str(control_path), *forwarded[1:]]
    return [
        "scp",
        "-o",
        f"ControlPath={control_path}",
        *forwarded[1:],
    ]


def controlmaster_command_runner(
    *,
    base_runner,
    control_path,
    stat_fn=Path.stat,
    command_timeouts_s=None,
):
    if not callable(base_runner):
        raise ValueError("base command runner is required")
    if not callable(stat_fn):
        raise ValueError("control path stat function is invalid")
    if command_timeouts_s is None:
        command_timeouts_s = {}
    if (
        not isinstance(command_timeouts_s, dict)
        or any(
            not isinstance(name, str)
            or not name
            or isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
            for name, timeout in command_timeouts_s.items()
        )
    ):
        raise ValueError("command timeout policy is invalid")
    control_path = _validate_control_path(control_path, stat_fn)
    command_timeouts_s = dict(command_timeouts_s)

    def run_command(**kwargs):
        argv = kwargs.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(value, str) for value in argv)
        ):
            raise ValueError("transport argv is invalid")
        forwarded = dict(kwargs)
        forwarded["argv"] = _transport_argv(argv, control_path)
        if (
            "command_timeout_s" not in forwarded
            and forwarded.get("name") in command_timeouts_s
        ):
            forwarded["command_timeout_s"] = (
                command_timeouts_s[forwarded["name"]]
            )
        return base_runner(**forwarded)

    return run_command
