import json
import resource
import sys


def unique_pairs(entries):
    result = {}
    for name, value in entries:
        if name in result:
            raise ValueError("duplicate key")
        result[name] = value
    return result


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    raw = sys.stdin.buffer.read(8193)
    if len(raw) > 8192:
        raise ValueError("request size")
    request = json.loads(raw, object_pairs_hook=unique_pairs)
    if type(request) is not dict or set(request) != {
        "request_id", "source", "entry_point", "args",
    }:
        raise ValueError("request fields")
    if (type(request["request_id"]) is not str
            or type(request["source"]) is not str
            or len(request["source"].encode()) > 4096
            or type(request["entry_point"]) is not str
            or not request["entry_point"].isidentifier()
            or type(request["args"]) is not list):
        raise ValueError("request types")
    namespace = {}
    exec(compile(request["source"], "<candidate>", "exec"), namespace)
    try:
        value = namespace[request["entry_point"]](*request["args"])
        outcome = {"kind": "return", "value": value}
    except ValueError:
        outcome = {"kind": "raises", "type": "ValueError"}
    output = json.dumps(outcome, sort_keys=True, ensure_ascii=True, allow_nan=False)
    sys.stdout.write(output + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        raise SystemExit(20)
