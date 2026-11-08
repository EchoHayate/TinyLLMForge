import argparse
import hashlib
import itertools
import json

from active_verification_contract import METHODS, probe_request, require
from active_verification_manifest import ROLES, build_manifest, manifest_digest
from active_verification_score import score_batch


SCENARIOS = ("mixed", "ceiling", "incomplete")


def synthetic_tasks():
    tasks = []
    for index, information in enumerate(("sufficient", "insufficient"), start=1):
        tasks.append({
            "evidence_kind": "synthetic", "task_id": f"{index:032x}",
            "family_id": f"synthetic-family-{index}", "split": "development",
            "domain": f"synthetic-domain-{index}",
            "specification": f"Synthetic fixture {index}; sources are not executable.",
            "sources": {role: f"opaque fixture {index} {role}" for role in ROLES},
            "check_inputs": [
                {"id": f"probe-{case}", "input": f"opaque check input {case}"}
                for case in range(2)
            ],
            "private": {
                "case_ids": ["scoring-0", "scoring-1"],
                "pass_by_source": {
                    "correct": [True, True], "defect": [True, False], "other": [False, True],
                },
                "feedback_by_initial": {
                    initial: {
                        "correct": f"synthetic {initial} supported claim",
                        "incorrect": f"synthetic {initial} unsupported claim",
                    } for initial in ("correct", "defect")
                },
                "verification_information": information,
                "audit_note": "Synthetic author-assigned outcomes; not an oracle audit.",
            },
        })
    return tasks


def synthetic_model(action, failed=False):
    return {
        "kind": "model", "status": "error" if failed else "completed",
        "text": "" if failed else action, "terminated": not failed,
        "input_ids": None if failed else [11],
        "output_ids": None if failed else [12],
        "wall_ns": None if failed else 10,
    }


def synthetic_tool(cell, method, run_id, failed):
    request = probe_request(cell["view"], run_id, cell["episode_id"], method)
    if failed and method == "always_verify":
        status, results = "unavailable", []
    elif failed:
        status = "timeout"
        results = [{
            **cell["view"]["check_inputs"][0], "expected": "synthetic expected",
            "actual": None, "status": "timeout",
        }]
    else:
        status = "completed"
        results = [{
            **check, "expected": "synthetic expected",
            "actual": "synthetic expected" if cell["initial"] == "correct" else "synthetic mismatch",
            "status": "completed",
        } for check in cell["view"]["check_inputs"]]
    return {
        "kind": "tool", "request_id": request["request_id"], "status": status,
        "results": results, "executed_cases": len(results), "wall_ns": 7,
    }


def synthetic_trajectory(cell, method, scenario, run_id):
    letters = dict(zip(cell["permutation"], "ABC"))
    correct, defect = letters["correct"], letters["defect"]
    initial = letters[cell["initial"]]
    condition = cell["condition"]
    adverse = (
        scenario != "ceiling" and condition == "incorrect"
        and cell["verification_information"] == "insufficient"
    )
    if method == "keep":
        return []
    if method == "direct":
        choice = correct if scenario == "ceiling" or condition == "correct" else (
            initial if condition == "none" else defect
        )
        return [synthetic_model(choice)]
    if method == "rethink":
        choice = correct if scenario == "ceiling" or condition != "incorrect" else defect
        return [synthetic_model(initial), synthetic_model(choice, failed=adverse)]
    if method == "always_verify":
        return [synthetic_tool(cell, method, run_id, adverse), synthetic_model(correct)]
    if condition == "none":
        return [synthetic_model(correct if scenario == "ceiling" else initial)]
    return [
        synthetic_model("VERIFY"), synthetic_tool(cell, method, run_id, adverse),
        synthetic_model(defect if adverse else correct),
    ]


def diagnostic(scenario):
    require(scenario in SCENARIOS, "unknown synthetic scenario")
    run_id = hashlib.sha256(("synthetic-demo-" + scenario).encode("utf-8")).hexdigest()[:32]
    manifest = build_manifest(
        synthetic_tasks(), list(map(list, itertools.permutations(ROLES))),
        run_id, "development", {"max_input_tokens": 8, "max_output_tokens": 4},
    )
    digest = manifest_digest(manifest)
    records = []
    for index, cell in enumerate(manifest["cells"]):
        for method in METHODS:
            if scenario == "incomplete" and index == 0 and method == "active_verify":
                continue
            events = synthetic_trajectory(cell, method, scenario, run_id)
            if scenario == "incomplete" and index == 1 and method == "rethink":
                events = events[:1]
            records.append({
                "evidence_kind": "synthetic", "manifest_sha256": digest,
                "run_id": run_id, "episode_id": cell["episode_id"],
                "method": method, "events": events,
            })
    return {
        "scenario": scenario, "fixture_controlled_choices": True,
        "warning": "Synthetic scripted choices and outcomes, not model or tool execution.",
        "manifest": manifest, "records": records,
        "report": score_batch(manifest, digest, records),
    }


def main():
    parser = argparse.ArgumentParser(description="Offline synthetic active-verification diagnostic")
    parser.add_argument("--scenario", choices=SCENARIOS, default="mixed")
    arguments = parser.parse_args()
    print(json.dumps(diagnostic(arguments.scenario), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
