import copy

from pilot_protocol import fields, require
from prompt_rendering import INSTRUCTION, PUBLIC_FIELDS, _examples
from recalibration_data_contract import CASE_IDS, TASK_IDS, _outcome, _plain
from validation_contract import digest, encoded, load_frozen


EXAMPLES_SHA256 = "e1634ad790ed551484a91fe55db36fc2f74b9615d2e5f273989552ea4f6cb290"
REVISION = "Return the final complete function, even if you believe the first candidate is correct."


def examples_projection(supervision_raw):
    supervision = load_frozen("supervision_v1.json", supervision_raw)
    examples = _examples(sorted(supervision, key=lambda task: task["id"]))
    require(len(encoded(examples)) == 7514 and digest(examples) == EXAMPLES_SHA256,
            "frozen examples projection")
    return examples


def _public_task(task):
    _plain(task)
    fields(task, PUBLIC_FIELDS)
    require(type(task["id"]) is str and task["id"] in TASK_IDS, "public task identity")
    for name in ("entry_point", "spec", "initial_source"):
        require(type(task[name]) is str and bool(task[name]), "public task text")
    require(task["entry_point"].isidentifier(), "public entry point")
    require(len(task["initial_source"].encode("utf-8")) <= 4096, "initial source byte limit")
    cases = task["public_cases"]
    require(type(cases) is list and len(cases) == 2, "public case coverage")
    for case, identifier in zip(cases, CASE_IDS[:2]):
        fields(case, ("id", "args", "expected"))
        require(case["id"] == identifier and type(case["args"]) is list, "public case identity")
        _outcome(case["expected"])
    return copy.deepcopy(task)


def first_round(task, arm, supervision_raw):
    public = _public_task(task)
    require(type(arm) is str and arm in ("base", "text"), "public arm")
    examples = examples_projection(supervision_raw)
    content = INSTRUCTION
    if arm == "text":
        content += "\n\nSupervised examples:\n" + encoded(examples).decode("ascii")
    content += "\n\nCurrent task:\n" + encoded(public).decode("ascii")
    return [{"role": "user", "content": content}]


def _feedback(branch, feedback):
    require(type(branch) is str and branch in ("feedback", "retry"), "continuation branch")
    if branch == "retry":
        require(feedback is None, "retry withholds feedback")
        return {"kind": "results_withheld"}
    _plain(feedback)
    fields(feedback, ("kind", "cases"))
    require(feedback["kind"] == "visible_tests", "public feedback kind")
    cases = feedback["cases"]
    require(type(cases) is list and len(cases) == 2, "public feedback coverage")
    for case, identifier in zip(cases, CASE_IDS[:2]):
        fields(case, ("id", "status"))
        require(case["id"] == identifier and type(case["status"]) is str
                and case["status"] in ("pass", "mismatch"), "public feedback identity")
    return copy.deepcopy(feedback)


def second_round(task, arm, supervision_raw, branch, feedback=None):
    messages = first_round(task, arm, supervision_raw)
    payload = _feedback(branch, feedback)
    return messages + [
        {"role": "assistant", "content": task["initial_source"]},
        {"role": "user", "content": REVISION + "\n" + encoded(payload).decode("ascii")},
    ]
