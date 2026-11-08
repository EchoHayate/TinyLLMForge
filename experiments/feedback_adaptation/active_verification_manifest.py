import copy
import hashlib

from active_verification_contract import (
    canonical, fields, identity, require, text, validate_limits, validate_view,
)


ROLES = ("correct", "defect", "other")
CONDITIONS = ("none", "correct", "incorrect")
SPLITS = ("development", "sealed")


def validate_tasks(tasks):
    require(type(tasks) is list and bool(tasks), "nonempty synthetic tasks")
    identifiers, families, signatures = set(), {}, set()
    for task in tasks:
        fields(task, (
            "evidence_kind", "task_id", "family_id", "split", "domain",
            "specification", "sources", "check_inputs", "private",
        ))
        require(task["evidence_kind"] == "synthetic", "synthetic tasks only")
        identity(task["task_id"])
        require(task["task_id"] not in identifiers, "duplicate task identity")
        identifiers.add(task["task_id"])
        text(task["family_id"])
        text(task["domain"])
        require(task["split"] in SPLITS, "unknown split")
        family, split = task["family_id"], task["split"]
        require(families.get(family, split) == split, "family crosses splits")
        families[family] = split
        fields(task["sources"], ROLES)
        validate_view({
            "specification": task["specification"],
            "initial_source": task["sources"]["correct"],
            "candidates": dict(zip("ABC", (task["sources"][role] for role in ROLES))),
            "check_inputs": task["check_inputs"], "feedback": [],
        })
        signature = canonical([task["specification"], sorted(task["sources"].values())])
        require(signature not in signatures, "duplicate task content")
        signatures.add(signature)
        private = task["private"]
        fields(private, (
            "case_ids", "pass_by_source", "feedback_by_initial",
            "verification_information", "audit_note",
        ))
        case_ids = private["case_ids"]
        require(type(case_ids) is list and bool(case_ids), "missing scoring cases")
        for case_id in case_ids:
            text(case_id)
        require(len(set(case_ids)) == len(case_ids), "duplicate scoring case")
        fields(private["pass_by_source"], ROLES)
        for role, outcomes in private["pass_by_source"].items():
            require(
                type(outcomes) is list and len(outcomes) == len(case_ids)
                and all(type(value) is bool for value in outcomes), "scoring outcome shape",
            )
            require(all(outcomes) == (role == "correct"), "source role contradicts outcomes")
        fields(private["feedback_by_initial"], ("correct", "defect"))
        for claims in private["feedback_by_initial"].values():
            fields(claims, ("correct", "incorrect"))
            for claim in claims.values():
                text(claim)
            require(claims["correct"] != claims["incorrect"], "indistinguishable feedback")
        require(private["verification_information"] in ("sufficient", "insufficient"), "information label")
        text(private["audit_note"])
        canonical(task)


def build_manifest(tasks, permutations, run_id, split, limits):
    validate_tasks(tasks)
    validate_limits(limits)
    identity(run_id)
    require(split in SPLITS, "unknown split")
    require(type(permutations) is list and bool(permutations), "missing permutations")
    seen = set()
    for permutation in permutations:
        require(
            type(permutation) is list and len(permutation) == 3
            and all(type(role) is str for role in permutation)
            and set(permutation) == set(ROLES), "invalid permutation",
        )
        key = tuple(permutation)
        require(key not in seen, "duplicate permutation")
        seen.add(key)
    cells, identifiers = [], set()
    for task in tasks:
        if task["split"] != split:
            continue
        for permutation in permutations:
            for initial in ("correct", "defect"):
                for condition in CONDITIONS:
                    feedback = [] if condition == "none" else [
                        task["private"]["feedback_by_initial"][initial][condition]
                    ]
                    view = {
                        "specification": task["specification"],
                        "initial_source": task["sources"][initial],
                        "candidates": dict(zip("ABC", (
                            task["sources"][role] for role in permutation
                        ))),
                        "check_inputs": task["check_inputs"], "feedback": feedback,
                    }
                    validate_view(view)
                    episode_id = hashlib.sha256(canonical([run_id, view])).hexdigest()[:32]
                    require(episode_id not in identifiers, "duplicate episode identity")
                    identifiers.add(episode_id)
                    cells.append({
                        "episode_id": episode_id, "task_id": task["task_id"],
                        "family_id": task["family_id"], "domain": task["domain"],
                        "split": split, "permutation": permutation, "initial": initial,
                        "condition": condition,
                        "verification_information": task["private"]["verification_information"],
                        "view": view,
                        "pass_by_letter": dict(zip("ABC", (
                            task["private"]["pass_by_source"][role] for role in permutation
                        ))),
                    })
    require(bool(cells), "empty selected split")
    count = len(cells)
    return copy.deepcopy({
        "evidence_kind": "synthetic", "run_id": run_id, "split": split,
        "limits": limits, "tasks": tasks, "permutations": permutations, "cells": cells,
        "budgets": {
            "cells": count, "records": 5 * count, "model_calls": 6 * count,
            "tool_calls": 2 * count,
            "executed_cases": 2 * sum(len(cell["view"]["check_inputs"]) for cell in cells),
            "input_tokens": 6 * count * limits["max_input_tokens"],
            "output_tokens": 6 * count * limits["max_output_tokens"],
        },
    })


def validate_manifest(manifest):
    fields(manifest, (
        "evidence_kind", "run_id", "split", "limits", "tasks",
        "permutations", "cells", "budgets",
    ))
    rebuilt = build_manifest(
        manifest["tasks"], manifest["permutations"], manifest["run_id"],
        manifest["split"], manifest["limits"],
    )
    require(canonical(manifest) == canonical(rebuilt), "frozen manifest drift")


def manifest_digest(manifest):
    validate_manifest(manifest)
    return hashlib.sha256(canonical(manifest)).hexdigest()
