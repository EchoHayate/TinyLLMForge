import inspect
import json

from development_programs import PROGRAMS, PROBES, ROLES, domains


DETAILS = (
    (
        "strict-record-highs-v1",
        "Input: a list of 0 to 4 integers, each -1, 0 or 1 (not bool). "
        "Return the number of elements strictly greater than every earlier element. "
        "The first element counts; the empty list returns 0.",
        [0, 0], "sufficient",
        "For each position, universally compare its value with the preceding prefix.",
    ),
    (
        "half-open-membership-v1",
        "Input: [start, end, point], integers (not bool) with 0 <= start <= end <= 3 "
        "and 0 <= point <= 3. Return 1 exactly when point is in [start, end), "
        "otherwise 0. Empty intervals contain no points.",
        [1, 3, 3], "insufficient",
        "Enumerate the integer range from start through end minus one; test membership.",
    ),
    (
        "unsigned-binary-grammar-v1",
        "Input: a string of length 0 to 4 over '0', '1', '+'. Return the value "
        "of a nonempty unsigned binary numeral; leading zeros are allowed. "
        "Return -1 for empty input or any '+', including a leading '+'.",
        "+1", "sufficient",
        "Validate the grammar, then sum independent positional powers of two.",
    ),
    (
        "idempotent-open-close-v1",
        "Input: a string of length 0 to 4 over 'o', 'c', 'r'. Initially closed; "
        "'o' opens and 'c' closes, both idempotently. 'r' reads without changing "
        "state. Return the number of reads that occur while open.",
        "cr", "insufficient",
        "For each read, inspect the most recent preceding open/close event; default closed.",
    ),
)


def anchors():
    return {
        "dev-001": [([], 0), ([-1, -1], 1), ([1, 0, 1], 1),
                    ([-1, 0, 1], 3), ([1, -1, 0], 1)],
        "dev-002": [([0, 0, 0], 0), ([0, 1, 0], 1), ([1, 3, 1], 1),
                    ([1, 3, 3], 0)],
        "dev-003": [("", -1), ("0", 0), ("10", 2), ("+1", -1),
                    ("1+", -1), ("1111", 15)],
        "dev-004": [("", 0), ("r", 0), ("or", 1), ("cr", 0),
                    ("oor", 1), ("orrr", 3)],
    }


def reference(task_id, value):
    allowed = domains()
    if type(task_id) is not str or task_id not in allowed:
        raise ValueError("unknown task")
    if task_id in ("dev-001", "dev-002"):
        if type(value) is not list or any(type(item) is not int for item in value):
            raise ValueError("input must be a list of integers")
    elif type(value) is not str:
        raise ValueError("input must be a string")
    serialized = json.dumps(value, sort_keys=True, allow_nan=False)
    if serialized not in {json.dumps(item) for item in allowed[task_id]}:
        raise ValueError("input outside finite domain")
    if task_id == "dev-001":
        return sum(all(item > earlier for earlier in value[:index])
                   for index, item in enumerate(value))
    if task_id == "dev-002":
        start, end, point = value
        return int(point in range(start, end))
    if task_id == "dev-003":
        if not value or not set(value) <= {"0", "1"}:
            return -1
        return sum((2 ** index) for index, char in enumerate(reversed(value)) if char == "1")
    return sum(
        next((event for event in reversed(value[:index]) if event != "r"), "c") == "o"
        for index, event in enumerate(value) if event == "r"
    )


def catalog():
    tasks = []
    for (task_id, values), details in zip(domains().items(), DETAILS):
        family, specification, witness, information, note = details
        examples = anchors()[task_id]
        for value, expected in examples:
            if reference(task_id, value) != expected:
                raise ValueError("literal/reference disagreement")
        cases = [
            {"id": f"case-{index:03d}", "input": value, "expected": reference(task_id, value)}
            for index, value in enumerate(values)
        ]
        probes = [
            {"id": row["id"], "input": row["input"]}
            for row in cases if row["input"] in PROBES[task_id]
        ]
        sources = {}
        for role, function in zip(ROLES, PROGRAMS[task_id]):
            source = inspect.getsource(function)
            sources[role] = source.replace(f"def {function.__name__}(", "def solve(", 1)
        witness_id = next(row["id"] for row in cases if row["input"] == witness)
        feedback = {}
        for initial in ("correct", "defect"):
            match = f"On input {json.dumps(witness)}, the initial program matches the specification."
            mismatch = f"On input {json.dumps(witness)}, the initial program does not match the specification."
            truth, falsehood = (match, mismatch) if initial == "correct" else (mismatch, match)
            feedback[initial] = {"correct": truth, "incorrect": falsehood}
        tasks.append({
            "evidence_kind": "fixed_development", "task_id": task_id,
            "family_id": family, "split": "development", "specification": specification,
            "sources": sources, "check_inputs": probes,
            "private": {
                "cases": cases,
                "score_ids": [row["id"] for row in cases
                              if row["id"] not in {probe["id"] for probe in probes}],
                "witness_id": witness_id, "anchors": examples,
                "verification_information": information, "feedback_by_initial": feedback,
                "reference_note": note, "independent_human_review": False,
            },
        })
    return tasks
