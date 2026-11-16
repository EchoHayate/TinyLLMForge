import copy
import hashlib
import json
import math

from pilot_protocol import build_manifest, canonical, require


PINS = {
    "task_sources_v1.json": "4711c72508b78f5deeafa5ed8de13ec4045d38c8ffbe5f463cb16651c29c693c",
    "manifest_v1.json": "cdac4ae1218701c585f8a38a27cd86583b9d4f71dcc60a3826f9dcfd2a2e7320",
    "supervision_v1.json": "4dafb0ac759697e9b80c5bf097b363082a17a34e6b78fb3d3b91d37a76beb78a",
    "prompts_v1.json": "3668d996bb0e05d600859fd12780b9f582a65aa2ad6b1881fd0bcce9832b5a05",
}


def encoded(value):
    return canonical(value).encode("ascii")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def parse_json(raw, limit=1048576):
    require(type(raw) is bytes and len(raw) <= limit, "JSON byte budget")

    def pairs(entries):
        result = {}
        for name, value in entries:
            require(name not in result, "duplicate JSON key")
            result[name] = value
        return result

    def constant(value):
        raise ValueError("nonfinite JSON value")

    def number(value):
        result = float(value)
        require(math.isfinite(result), "nonfinite JSON number")
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=constant, parse_float=number)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("invalid JSON encoding or depth") from error


def load_frozen(name, raw):
    require(type(name) is str and name in PINS, "unknown frozen file")
    require(type(raw) is bytes and len(raw) <= 1048576, "frozen file bytes")
    require(hashlib.sha256(raw).hexdigest() == PINS[name], "frozen bytes changed")
    return parse_json(raw)


def _tasks(source_raw, manifest_raw):
    source = load_frozen("task_sources_v1.json", source_raw)
    manifest = load_frozen("manifest_v1.json", manifest_raw)
    metadata = [
        {**{name: task[name] for name in ("id", "split", "family", "template_id")},
         "content_sha256": digest(task)}
        for task in source["tasks"]
    ]
    require(encoded(build_manifest(metadata)) == encoded(manifest["cpu_manifest"]),
            "task metadata binding")
    return sorted(source["tasks"], key=lambda task: task["id"])


def _request(row):
    return {
        "request_id": f"request-{row['ordinal']:04d}",
        "entry_point": row["entry_point"], "source": row["source"],
        "args": copy.deepcopy(row["args"]),
    }


def build_schedule(source_raw, manifest_raw):
    rows = []
    for task in _tasks(source_raw, manifest_raw):
        versions = [
            ("reference", task["reference_source"], None, None, None),
            ("initial", task["initial_source"], "public",
             task["initial_witness"]["public_case_id"],
             task["initial_witness"]["predicted_initial"]),
        ]
        versions.extend(
            (mutant["id"], mutant["source"], "scoring", mutant["scoring_case_id"],
             mutant["predicted_mutant"])
            for mutant in sorted(task["mutants"], key=lambda mutant: mutant["id"])
        )
        for role, source, witness_group, witness_id, prediction in versions:
            for group, name in (("public", "public_cases"), ("scoring", "scoring_cases")):
                for case in sorted(task[name], key=lambda case: case["id"]):
                    witness = group == witness_group and case["id"] == witness_id
                    row = {
                        "ordinal": len(rows), "task_id": task["id"], "role": role,
                        "group": group, "case_id": case["id"],
                        "task_sha256": digest(task), "source_sha256":
                        hashlib.sha256(source.encode("utf-8")).hexdigest(),
                        "input_sha256": digest(case["args"]),
                        "expected_sha256": digest(case["expected"]),
                        "entry_point": task["entry_point"], "source": source,
                        "args": copy.deepcopy(case["args"]),
                        "expected": copy.deepcopy(case["expected"]),
                        "witness_prediction": copy.deepcopy(prediction) if witness else None,
                    }
                    if witness:
                        require(encoded(prediction) != encoded(case["expected"]),
                                "witness must disagree")
                    row["request_sha256"] = digest(_request(row))
                    rows.append(row)
    require(len(rows) == 1416, "frozen schedule size")
    return {
        "kind": "LORA_VALIDATION_PLAN", "version": 1,
        "source_sha256": PINS["task_sources_v1.json"],
        "manifest_sha256": PINS["manifest_v1.json"], "rows": rows,
    }


def worker_requests(source_raw, manifest_raw):
    return [_request(row) for row in build_schedule(source_raw, manifest_raw)["rows"]]


def reference_jobs(source_raw, manifest_raw):
    return [{"id": task["id"], "reference_source": task["reference_source"]}
            for task in _tasks(source_raw, manifest_raw)]
