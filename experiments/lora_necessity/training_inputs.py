from pilot_protocol import fields, require
from prompt_rendering import training_jobs
from tokenizer_measurement import measure
from validation_contract import PINS, digest


DATASET_FIELDS = (
    "kind", "version", "supervision_sha256", "tokenizer_state_sha256",
    "eos_token_id", "pad_token_id", "rows", "dataset_sha256",
)
ROW_FIELDS = (
    "task_id", "input_ids", "attention_mask", "labels", "prompt_tokens",
    "target_tokens", "messages_sha256", "answer_sha256",
)
TASK_IDS = tuple(f"S{ordinal:02d}" for ordinal in range(1, 9))


def _hash(value):
    require(type(value) is str and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value),
            "invalid SHA256")


def validate_training(dataset):
    fields(dataset, DATASET_FIELDS)
    require(all(type(name) is str for name in dataset), "JSON string field names")
    require(type(dataset["kind"]) is str and
            dataset["kind"] == "LORA_TRAINING_INPUTS_UNATTESTED", "training input kind")
    require(type(dataset["version"]) is int and dataset["version"] == 1,
            "training input version")
    for name in ("supervision_sha256", "tokenizer_state_sha256", "dataset_sha256"):
        _hash(dataset[name])
    require(dataset["supervision_sha256"] == PINS["supervision_v1.json"],
            "supervision source pin")
    for name in ("eos_token_id", "pad_token_id"):
        require(type(dataset[name]) is int and dataset[name] >= 0, "special token ID")
    rows = dataset["rows"]
    require(type(rows) is list and len(rows) == len(TASK_IDS), "eight training rows")
    for row, task_id in zip(rows, TASK_IDS):
        fields(row, ROW_FIELDS)
        require(all(type(name) is str for name in row), "JSON string field names")
        require(type(row["task_id"]) is str and row["task_id"] == task_id,
                "ordered supervision task IDs")
        for name in ("messages_sha256", "answer_sha256"):
            _hash(row[name])
        for name in ("prompt_tokens", "target_tokens"):
            require(type(row[name]) is int and row[name] > 0, "positive token counts")
        prompt_tokens = row["prompt_tokens"]
        target_tokens = row["target_tokens"]
        total_tokens = prompt_tokens + target_tokens
        require(prompt_tokens <= 4096 and target_tokens <= 256, "training token limits")
        for name in ("input_ids", "attention_mask", "labels"):
            values = row[name]
            require(type(values) is list and len(values) == total_tokens,
                    "training sequence length")
            require(all(type(value) is int for value in values), "integer token lists")
        require(all(value >= 0 for value in row["input_ids"]), "nonnegative token IDs")
        require(all(value == 1 for value in row["attention_mask"]), "unmasked attention")
        target_ids = row["input_ids"][prompt_tokens:]
        require(target_ids[-1] == dataset["eos_token_id"] and
                target_ids.count(dataset["eos_token_id"]) == 1, "unique terminal target EOS")
        require(row["labels"] == [-100] * prompt_tokens + target_ids,
                "answer-only unshifted labels")
    expected = digest({name: value for name, value in dataset.items()
                       if name != "dataset_sha256"})
    require(dataset["dataset_sha256"] == expected, "training dataset digest")
    return expected


def prepare_training(tokenizer, supervision_raw):
    jobs = training_jobs(supervision_raw)
    require(tuple(job["task_id"] for job in jobs) == TASK_IDS,
            "ordered supervision task IDs")
    rows = []
    state_sha256 = None
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    for job in jobs:
        measured = measure(tokenizer, job["messages"], job["answer"])
        require(measured["status"] == "OK", "training measurement token limits")
        if state_sha256 is None:
            state_sha256 = measured["state_sha256"]
        require(measured["state_sha256"] == state_sha256 and
                tokenizer.eos_token_id == eos_token_id and
                tokenizer.pad_token_id == pad_token_id, "tokenizer state changed")
        prompt = measured["input"]
        target = measured["target"]
        input_ids = list(prompt["input_ids"]) + list(target["target_ids"])
        rows.append({
            "task_id": job["task_id"],
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * prompt["input_tokens"] + list(target["target_ids"]),
            "prompt_tokens": prompt["input_tokens"],
            "target_tokens": target["target_tokens"],
            "messages_sha256": prompt["messages_sha256"],
            "answer_sha256": target["answer_sha256"],
        })
    dataset = {
        "kind": "LORA_TRAINING_INPUTS_UNATTESTED",
        "version": 1,
        "supervision_sha256": PINS["supervision_v1.json"],
        "tokenizer_state_sha256": state_sha256,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "rows": rows,
    }
    dataset["dataset_sha256"] = digest(dataset)
    validate_training(dataset)
    return dataset
