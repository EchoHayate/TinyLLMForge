import copy
import hashlib

from pilot_protocol import FLAGS, require
from prompt_rendering import episode_jobs, second_round, training_jobs
from validation_contract import PINS, digest, load_frozen, reference_jobs


REFERENCE_SHA256 = "2333003d187a1f12618b4505ca3522b7c2a20e3295c48ae835f3e879ed601cf3"


def build_inventory(source_raw, manifest_raw, supervision_raw, prompts_raw):
    references = reference_jobs(source_raw, manifest_raw)
    require(digest(references) == REFERENCE_SHA256, "reference projection changed")
    training = training_jobs(supervision_raw)
    episodes = episode_jobs(supervision_raw, prompts_raw)
    public = {task["id"]: task for task in load_frozen("prompts_v1.json", prompts_raw)}
    rows = []
    base_messages = {}

    def append(category, task_id, arm, messages, answer=None, sample=None):
        row = {
            "ordinal": len(rows), "category": category, "task_id": task_id,
            "arm": arm, "messages": copy.deepcopy(messages), "answer": answer,
            "sample": copy.deepcopy(sample),
        }
        rows.append({**row, "row_sha256": digest(row)})

    for job in training:
        append("training", job["task_id"], None, job["messages"], job["answer"])
        base_messages[job["task_id"]] = job["messages"]
    for job in episodes:
        append("first", job["task_id"], job["arm"], job["messages"])
        if job["arm"] == "base":
            base_messages[job["task_id"]] = job["messages"]
    for job in references:
        append("references", job["id"], None,
               base_messages[job["id"]], job["reference_source"])
    for job in episodes:
        task = public[job["task_id"]]
        candidate = task["initial_source"]
        feedback = {"kind": "visible_tests", "cases": [
            {"id": case["id"], "status": "mismatch"} for case in task["public_cases"]
        ]}
        messages = second_round(supervision_raw, prompts_raw, job["task_id"],
                                job["arm"], candidate, feedback)
        sample = {
            "kind": "INITIAL_SOURCE_SYNTHETIC_FEEDBACK",
            "candidate_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
            "feedback_sha256": digest(feedback),
        }
        append("sampled_second", job["task_id"], job["arm"], messages, sample=sample)
    require(len(training) == 8 and len(episodes) == 136 and
            len(references) == 44 and len(rows) == 324, "inventory size")
    return {
        "kind": "LORA_TOKENIZER_INVENTORY", "version": 1,
        "data_sha256": dict(PINS), "references_sha256": REFERENCE_SHA256,
        "rows": rows, "all_possible_second_rounds_validated": False,
        **{flag: False for flag in FLAGS},
    }
