import copy
import hashlib
import json
import re
from collections import Counter


ARMS = ("base", "text", "lora", "lora_text")
SPLITS = {"supervision": 8, "development": 4, "transfer": 16, "retention": 16}
FLAGS = ("benchmark_eligible", "formal_evaluation_ready", "net_benefit_established", "isolated")
CHECKS = ("protocol_valid", "parameters_valid", "isolation_valid", "cleanup_complete")
COSTS = ("input_tokens", "output_tokens", "episode_ns")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(value, names):
    require(type(value) is dict and set(value) == set(names), "unexpected fields")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def configuration():
    return {
        "arms": list(ARMS), "splits": dict(SPLITS),
        "training": {
            "rank": 8, "alpha": 16, "dropout": 0, "seed": 42,
            "targets": ["q_proj", "v_proj"], "base_dtype": "bfloat16",
            "adapter_dtype": "float32", "optimizer": "AdamW",
            "learning_rate": 0.00005, "weight_decay": 0,
            "gradient_clip": 1, "batch_size": 1, "accumulation": 1,
            "epochs": 2, "steps": 16, "answer_only_loss": True,
        },
        "limits": {
            "input_tokens": 4096, "output_tokens": 256, "episode_calls": 2,
            "total_calls": 272, "workflow_seconds": 3600, "gpu_count": 1,
        },
        "screen": {"transfer_net_min": 2, "retention_regression_max": 0},
    }


def _validate_configuration(value, expected):
    require(type(value) is type(expected), "configuration type")
    if type(expected) is dict:
        fields(value, expected)
        for name in expected:
            _validate_configuration(value[name], expected[name])
    elif type(expected) is list:
        require(len(value) == len(expected), "configuration length")
        for actual_item, expected_item in zip(value, expected):
            _validate_configuration(actual_item, expected_item)
    else:
        require(value == expected, "configuration changed")


def validate_manifest(manifest):
    fields(manifest, ("kind", "version", "config", "tasks"))
    require(manifest["kind"] == "LORA_NECESSITY_CPU_MANIFEST", "manifest kind")
    require(type(manifest["version"]) is int and manifest["version"] == 1, "version")
    _validate_configuration(manifest["config"], configuration())
    tasks = manifest["tasks"]
    require(type(tasks) is list and len(tasks) == 44, "44 task metadata rows required")
    identifiers, templates, digests = set(), set(), set()
    groups = {split: [] for split in SPLITS}
    for task in tasks:
        fields(task, ("id", "split", "family", "template_id", "content_sha256"))
        for name in task:
            require(type(task[name]) is str and re.fullmatch(r"[A-Za-z0-9_-]+", task[name]),
                    "invalid task metadata")
        require(task["split"] in SPLITS, "unknown split")
        require(re.fullmatch(r"[0-9a-f]{64}", task["content_sha256"]), "invalid digest")
        for value, seen in ((task["id"], identifiers), (task["template_id"], templates),
                            (task["content_sha256"], digests)):
            require(value not in seen, "duplicate task, template or content")
            seen.add(value)
        groups[task["split"]].append(task["family"])
    families = {}
    for split, expected in SPLITS.items():
        require(len(groups[split]) == expected, "split count")
        counts = Counter(groups[split])
        require(len(counts) == 4 and set(counts.values()) == {expected // 4}, "family balance")
        families[split] = set(counts)
    require(families["supervision"] == families["transfer"], "transfer families")
    require(families["retention"].isdisjoint(families["supervision"]), "retention overlap")
    return hashlib.sha256(canonical(manifest).encode()).hexdigest()


def build_manifest(tasks):
    manifest = {
        "kind": "LORA_NECESSITY_CPU_MANIFEST", "version": 1,
        "config": configuration(), "tasks": copy.deepcopy(tasks),
    }
    validate_manifest(manifest)
    return manifest


def _summarize(rows):
    summary = {
        "scheduled": len(rows), "received": sum(row is not None for row in rows),
        "passed": sum(row is not None and row["status"] == "pass" for row in rows),
        "statuses": dict(Counter("missing" if row is None else row["status"] for row in rows)),
    }
    for name in COSTS:
        values = [row[name] for row in rows if row is not None and row[name] is not None]
        summary[name] = sum(values) if len(values) == len(rows) else None
        summary[f"{name}_observed"] = sum(values)
    return summary


def score_synthetic(manifest, records, checks):
    digest = validate_manifest(manifest)
    fields(checks, CHECKS)
    require(all(value is None or type(value) is bool for value in checks.values()), "checks")
    require(type(records) is list, "record list")
    task_families = {task["id"]: task["family"] for task in manifest["tasks"]}
    schedule = {}
    for task in manifest["tasks"]:
        if task["split"] != "supervision":
            arms = ARMS[:2] if task["split"] == "development" else ARMS
            for arm in arms:
                schedule[(task["id"], arm)] = task["split"]
    indexed = {}
    for record in records:
        fields(record, ("kind", "task_id", "arm", "status", "model_calls", *COSTS))
        require(record["kind"] == "synthetic", "synthetic records only")
        require(type(record["task_id"]) is str and type(record["arm"]) is str, "record identity")
        key = record["task_id"], record["arm"]
        require(key in schedule and key not in indexed, "unknown or duplicate episode")
        require(record["status"] in ("pass", "fail", "invalid", "timeout"), "record status")
        require(type(record["model_calls"]) is int and 0 <= record["model_calls"] <= 2,
                "model call count")
        require(record["status"] == "timeout" or record["model_calls"] == 2,
                "completed episode needs two model calls")
        for name in COSTS:
            value = record[name]
            require(value is None or type(value) is int and value >= 0, "invalid cost")
        for name, cap in (("input_tokens", 4096), ("output_tokens", 256)):
            require(record[name] is None or record[name] <= cap * record["model_calls"],
                    "token budget")
        indexed[key] = record
    by_split, by_family = {}, {}
    for split in ("development", "transfer", "retention"):
        by_split[split] = {}
        families = sorted({task["family"] for task in manifest["tasks"] if task["split"] == split})
        by_family[split] = {family: {} for family in families}
        for arm in (ARMS[:2] if split == "development" else ARMS):
            keys = [key for key, stage in schedule.items() if stage == split and key[1] == arm]
            by_split[split][arm] = _summarize([indexed.get(key) for key in keys])
            for family in families:
                by_family[split][family][arm] = _summarize([
                    indexed.get(key) for key in keys if task_families[key[0]] == family
                ])
    pairs = {}
    for split in ("transfer", "retention"):
        counts = Counter()
        for task in manifest["tasks"]:
            if task["split"] != split:
                continue
            before = indexed.get((task["id"], "text"))
            after = indexed.get((task["id"], "lora_text"))
            if before is None or after is None:
                counts["missing"] += 1
            else:
                pattern = (before["status"] == "pass", after["status"] == "pass")
                counts[{(True, True): "both_right", (False, False): "both_wrong",
                        (True, False): "right_to_wrong", (False, True): "wrong_to_right"}[pattern]] += 1
        pairs[split] = {name: counts[name] for name in
                       ("both_right", "both_wrong", "right_to_wrong", "wrong_to_right", "missing")}
        pairs[split]["net"] = counts["wrong_to_right"] - counts["right_to_wrong"]
    incomplete = (len(indexed) != len(schedule)
                  or any(row["status"] == "timeout" for row in indexed.values())
                  or any(row[name] is None for row in indexed.values() for name in COSTS)
                  or any(value is None for value in checks.values()))
    development = by_split["development"]
    calibration_bad = (all(development[arm]["passed"] == 4 for arm in ARMS[:2])
                       or all(development[arm]["passed"] == 0 for arm in ARMS[:2])
                       or any(development[arm]["statuses"].get("invalid", 0) for arm in ARMS[:2]))
    if any(value is False for value in checks.values()):
        decision = "INVALID_PROTOCOL"
    elif incomplete:
        decision = "INCONCLUSIVE_INCOMPLETE"
    elif calibration_bad:
        decision = "INCONCLUSIVE_CALIBRATION"
    elif pairs["retention"]["right_to_wrong"]:
        decision = "OBSERVED_REGRESSION"
    elif pairs["transfer"]["net"] >= 2:
        decision = "PROMISING_FOR_REPLICATION"
    else:
        decision = "NO_GAIN_IN_THIS_PILOT"
    return {
        "status": "SYNTHETIC_ONLY", "manifest_sha256": digest,
        "simulated_decision": decision, "by_split": by_split, "by_family": by_family, "pairs": pairs,
        "model_calls": sum(row["model_calls"] for row in indexed.values()),
        "scheduled_model_calls": 272, "checks": copy.deepcopy(checks),
        **{flag: False for flag in FLAGS},
    }
