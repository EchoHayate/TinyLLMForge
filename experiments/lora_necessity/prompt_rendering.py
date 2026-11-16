import copy

from pilot_protocol import ARMS, fields, require
from validation_contract import digest, encoded, load_frozen


INSTRUCTION = (
    "Repair the function to satisfy its specification and public tests. "
    "Return only one complete Python function with the same name and parameters. "
    "Do not include Markdown, explanations, imports, or test code."
)
REVISION = (
    "Public-test feedback follows. Return the final complete function, "
    "even if the first candidate passed."
)
PUBLIC_FIELDS = ("entry_point", "id", "initial_source", "public_cases", "spec")
FEEDBACK_STATUSES = ("pass", "mismatch", "program_error", "timeout", "output_limit")


def _text(value):
    return encoded(value).decode("ascii")


def _public(task):
    return {name: copy.deepcopy(task[name]) for name in PUBLIC_FIELDS}


def _witness(task):
    witness = task["initial_witness"]
    return {name: copy.deepcopy(witness[name]) for name in
            ("public_case_id", "predicted_initial")}


def _examples(supervision):
    return [
        {**_public(task), "initial_witness": _witness(task),
         "reference_source": task["reference_source"]}
        for task in supervision
    ]


def _message(task, examples=None, witness=None):
    content = INSTRUCTION
    if examples is not None:
        content += "\n\nSupervised examples:\n" + _text(examples)
    content += "\n\nCurrent task:\n" + _text(_public(task))
    if witness is not None:
        content += "\n\nKnown initial failure:\n" + _text(witness)
    return [{"role": "user", "content": content}]


def episode_jobs(supervision_raw, prompts_raw):
    supervision = load_frozen("supervision_v1.json", supervision_raw)
    prompts = load_frozen("prompts_v1.json", prompts_raw)
    examples = _examples(sorted(supervision, key=lambda task: task["id"]))
    jobs = []
    for task in sorted(prompts, key=lambda task: task["id"]):
        arms = ARMS[:2] if task["id"].startswith("D") else ARMS
        for arm in arms:
            included = examples if arm in ("text", "lora_text") else None
            jobs.append({
                "task_id": task["id"], "arm": arm,
                "messages": _message(task, included),
                "examples_sha256": None if included is None else digest(included),
            })
    require(len(jobs) == 136, "episode count")
    return jobs


def training_jobs(supervision_raw):
    supervision = load_frozen("supervision_v1.json", supervision_raw)
    return [
        {"task_id": task["id"], "messages": _message(task, witness=_witness(task)),
         "answer": task["reference_source"]}
        for task in sorted(supervision, key=lambda task: task["id"])
    ]


def second_round(supervision_raw, prompts_raw, task_id, arm, candidate, feedback):
    require(type(task_id) is str and type(arm) is str, "episode identity")
    require(type(candidate) is str, "candidate text")
    try:
        require(len(candidate.encode("utf-8")) <= 65536, "candidate text byte cap")
    except UnicodeError as error:
        raise ValueError("candidate encoding") from error
    jobs = episode_jobs(supervision_raw, prompts_raw)
    matches = [job for job in jobs if job["task_id"] == task_id and job["arm"] == arm]
    require(len(matches) == 1, "unknown episode")
    require(type(feedback) is dict, "feedback object")
    if feedback.get("kind") == "invalid_source":
        fields(feedback, ("kind",))
    else:
        fields(feedback, ("kind", "cases"))
        require(feedback["kind"] == "visible_tests", "feedback kind")
        task = next(task for task in load_frozen("prompts_v1.json", prompts_raw)
                    if task["id"] == task_id)
        cases = feedback["cases"]
        require(type(cases) is list and len(cases) == len(task["public_cases"]),
                "public feedback coverage")
        for actual, expected in zip(cases, task["public_cases"]):
            fields(actual, ("id", "status"))
            require(actual["id"] == expected["id"] and
                    type(actual["status"]) is str and
                    actual["status"] in FEEDBACK_STATUSES, "public feedback case")
    return copy.deepcopy(matches[0]["messages"]) + [
        {"role": "assistant", "content": candidate},
        {"role": "user", "content": REVISION + "\n" + _text(feedback)},
    ]
