import copy
import hashlib

from pilot_protocol import FLAGS, fields, require
from prompt_rendering import episode_jobs, second_round, training_jobs
from validation_contract import PINS, digest, encoded, load_frozen


REFERENCE_JOBS_SHA256 = "2333003d187a1f12618b4505ca3522b7c2a20e3295c48ae835f3e879ed601cf3"


def _ids(values):
    require(type(values) is list and all(type(value) is int and value >= 0
                                       for value in values), "token IDs")
    return list(values)


def simulation_identity(tokenizer):
    template = tokenizer.chat_template
    require(type(template) is str and bool(template), "chat template")
    vocabulary = tokenizer.get_vocab()
    require(type(vocabulary) is dict and bool(vocabulary), "token vocabulary")
    require(all(type(name) is str and type(value) is int and value >= 0
                for name, value in vocabulary.items()), "vocabulary entries")
    for value in (tokenizer.eos_token_id, tokenizer.pad_token_id):
        require(type(value) is int and value in vocabulary.values(), "special token ID")
    return {
        "kind": "SYNTHETIC_TOKENIZER_IDENTITY", "version": 1,
        "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "vocabulary_sha256": digest(vocabulary),
        "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
    }


def _identity(tokenizer, identity):
    require(encoded(identity) == encoded(simulation_identity(tokenizer)),
            "synthetic tokenizer identity changed")


def _messages(messages, completed=False):
    require(type(messages) is list, "message list")
    roles = [message.get("role") if type(message) is dict else None for message in messages]
    allowed = (["user", "assistant"], ["user", "assistant", "user", "assistant"]) \
        if completed else (["user"], ["user", "assistant", "user"])
    require(roles in allowed, "message role sequence")
    for message in messages:
        fields(message, ("role", "content"))
        require(type(message["content"]) is str, "message content")


def _tokenize(tokenizer, messages, generation):
    _messages(messages, completed=not generation)
    copied = copy.deepcopy(messages)
    original = encoded(copied)
    arguments = {"add_generation_prompt": generation, "enable_thinking": False}
    text = tokenizer.apply_chat_template(copied, tokenize=False, **arguments)
    require(type(text) is str, "rendered text")
    direct = _ids(tokenizer.apply_chat_template(copied, tokenize=True, **arguments))
    actual = tokenizer(text, add_special_tokens=False, truncation=False)
    require(type(actual) is dict and "input_ids" in actual, "tokenizer output")
    tokens = _ids(actual["input_ids"])
    require(encoded(copied) == original, "tokenizer mutated messages")
    require(tokens == direct and bool(tokens), "template/tokenization disagreement")
    require(set(tokens).issubset(set(tokenizer.get_vocab().values())),
            "token outside declared vocabulary")
    return text, tokens


def inspect_call(tokenizer, identity, messages):
    _identity(tokenizer, identity)
    text, tokens = _tokenize(tokenizer, messages, True)
    _identity(tokenizer, identity)
    return {
        "kind": "SIMULATED_TOKEN_INPUT", "identity_sha256": digest(identity),
        "messages_sha256": digest(messages),
        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "prompt": text, "input_ids": tokens, "input_tokens": len(tokens),
        "remaining_input_tokens": 4096 - len(tokens), "within_limit": len(tokens) <= 4096,
    }


def check_call(tokenizer, identity, messages):
    row = inspect_call(tokenizer, identity, messages)
    require(row["within_limit"], "input exceeds 4096 tokens")
    return row


def inspect_target(tokenizer, identity, messages, answer):
    require(type(answer) is str and bool(answer), "answer text")
    prefix = inspect_call(tokenizer, identity, messages)
    completed = copy.deepcopy(messages) + [{"role": "assistant", "content": answer}]
    full_text, full_ids = _tokenize(tokenizer, completed, False)
    _identity(tokenizer, identity)
    prefix_ids = prefix["input_ids"]
    require(full_text.startswith(prefix["prompt"]), "answer text prefix mismatch")
    require(full_text[len(prefix["prompt"]):].startswith(answer), "answer text changed")
    require(full_ids[:len(prefix_ids)] == prefix_ids, "answer token prefix mismatch")
    suffix = full_ids[len(prefix_ids):]
    endings = [index for index, value in enumerate(suffix) if value == identity["eos_token_id"]]
    require(len(endings) == 1, "unique answer termination token required")
    endpoint = endings[0] + 1
    target = suffix[:endpoint]
    return {
        "kind": "SIMULATED_TOKEN_TARGET", "identity_sha256": digest(identity),
        "messages_sha256": digest(messages),
        "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "full_prompt_sha256": hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
        "prompt_tokens": len(prefix_ids), "prompt_within_limit": prefix["within_limit"],
        "full_sequence_tokens": len(full_ids), "full_sequence_ids": full_ids,
        "target_start": len(prefix_ids), "target_ids": target, "target_tokens": len(target),
        "target_within_limit": len(target) <= 256, "template_tail_ids": suffix[endpoint:],
    }


def audit_simulated(supervision_raw, prompts_raw, answers, tokenizer, identity):
    require(digest(answers) == REFERENCE_JOBS_SHA256, "frozen reference projection")
    _identity(tokenizer, identity)
    episodes = episode_jobs(supervision_raw, prompts_raw)
    training = training_jobs(supervision_raw)
    public = {task["id"]: task for task in load_frozen("prompts_v1.json", prompts_raw)}
    first_rows = []
    sample_rows = []
    base_messages = {}
    for job in episodes:
        key = {"task_id": job["task_id"], "arm": job["arm"]}
        first_rows.append({**key, **inspect_call(tokenizer, identity, job["messages"])})
        if job["arm"] == "base":
            base_messages[job["task_id"]] = job["messages"]
        task = public[job["task_id"]]
        feedback = {"kind": "visible_tests", "cases": [
            {"id": case["id"], "status": "mismatch"} for case in task["public_cases"]
        ]}
        messages = second_round(supervision_raw, prompts_raw, job["task_id"], job["arm"],
                                task["initial_source"], feedback)
        sample_rows.append({
            **key, "sample_kind": "INITIAL_SOURCE_SYNTHETIC_FEEDBACK",
            "candidate_sha256": hashlib.sha256(task["initial_source"].encode()).hexdigest(),
            "feedback_sha256": digest(feedback), **inspect_call(tokenizer, identity, messages),
        })
    training_rows = []
    for job in training:
        base_messages[job["task_id"]] = job["messages"]
        training_rows.append({
            "task_id": job["task_id"],
            **inspect_target(tokenizer, identity, job["messages"], job["answer"]),
        })
    reference_rows = [
        {"task_id": job["id"], **inspect_target(
            tokenizer, identity, base_messages[job["id"]], job["reference_source"])}
        for job in answers
    ]
    return {
        "kind": "LORA_TOKEN_BUDGET_SIMULATION", "version": 1,
        "identity": copy.deepcopy(identity),
        "supervision_sha256": PINS["supervision_v1.json"],
        "prompts_sha256": PINS["prompts_v1.json"],
        "references_sha256": REFERENCE_JOBS_SHA256,
        "training": training_rows, "first": first_rows,
        "references": reference_rows, "sampled_second": sample_rows,
        "all_possible_second_rounds_validated": False,
        **{flag: False for flag in FLAGS},
    }
