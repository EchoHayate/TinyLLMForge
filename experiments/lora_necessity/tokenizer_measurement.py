import copy
import hashlib
from collections.abc import Mapping

from pilot_protocol import fields, require
from validation_contract import digest, encoded


def _text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _VocabularyDigest:
    def __init__(self):
        self._vocabulary = None
        self._sha256 = None

    def __call__(self, vocabulary):
        observed = vocabulary.copy()
        require(all(type(name) is str and type(value) is int and value >= 0
                    for name, value in observed.items()), "vocabulary entries")
        if observed != self._vocabulary:
            self._sha256 = digest(observed)
            self._vocabulary = observed
        return self._sha256


def _state(tokenizer, vocabulary_digest=None):
    template = tokenizer.chat_template
    vocabulary = tokenizer.get_vocab()
    require(type(template) is str and bool(template), "chat template")
    require(type(vocabulary) is dict and bool(vocabulary), "token vocabulary")
    require(all(type(name) is str and type(value) is int and value >= 0
                for name, value in vocabulary.items()), "vocabulary entries")
    for value in (tokenizer.eos_token_id, tokenizer.pad_token_id):
        require(type(value) is int and value in vocabulary.values(), "special token ID")
    return {
        "template_sha256": _text_hash(template),
        "vocabulary_sha256": (vocabulary_digest or digest)(vocabulary),
        "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
    }, set(vocabulary.values())


def _ids(values, vocabulary):
    require(type(values) is list and bool(values), "nonempty token ID list")
    require(all(type(value) is int and value >= 0 and value in vocabulary
                for value in values), "declared integer token IDs")
    return list(values)


def _render(tokenizer, messages, generation, state, vocabulary, answer_prefix=None,
            vocabulary_digest=None):
    require(type(messages) is list, "message list")
    roles = []
    for message in messages:
        fields(message, ("role", "content"))
        require(type(message["role"]) is str and type(message["content"]) is str,
                "message strings")
        roles.append(message["role"])
    allowed = (["user"], ["user", "assistant", "user"]) if generation else (
        ["user", "assistant"], ["user", "assistant", "user", "assistant"])
    require(roles in allowed, "message roles")
    copied = copy.deepcopy(messages)
    original = encoded(copied)
    arguments = {"add_generation_prompt": generation, "enable_thinking": False}
    text = tokenizer.apply_chat_template(copied, tokenize=False, **arguments)
    require(type(text) is str, "rendered text")
    require(encoded(copied) == original, "messages changed")
    require(_state(tokenizer, vocabulary_digest)[0] == state, "tokenizer state changed")
    direct = _ids(tokenizer.apply_chat_template(
        copied, tokenize=True, truncation=False, **arguments), vocabulary)
    require(encoded(copied) == original, "messages changed")
    require(_state(tokenizer, vocabulary_digest)[0] == state, "tokenizer state changed")
    actual = tokenizer(text, add_special_tokens=False, truncation=False)
    require(isinstance(actual, Mapping) and "input_ids" in actual, "tokenizer mapping")
    tokens = _ids(actual["input_ids"], vocabulary)
    require(tokens == direct, "template/tokenization disagreement")
    require(_state(tokenizer, vocabulary_digest)[0] == state, "tokenizer state changed")
    require(encoded(copied) == original, "messages changed")
    if answer_prefix is not None:
        boundary = tokenizer(answer_prefix, add_special_tokens=False, truncation=False)
        require(isinstance(boundary, Mapping) and "input_ids" in boundary, "tokenizer mapping")
        boundary_ids = _ids(boundary["input_ids"], vocabulary)
        require(tokens[:len(boundary_ids)] == boundary_ids,
                "answer boundary token prefix mismatch")
        require(state["eos_token_id"] in tokens[len(boundary_ids):],
                "termination token must follow complete answer")
        require(_state(tokenizer, vocabulary_digest)[0] == state, "tokenizer state changed")
        require(encoded(copied) == original, "messages changed")
    return text, tokens


def measure(tokenizer, messages, answer=None):
    require(answer is None or type(answer) is str and bool(answer), "answer text")
    vocabulary_digest = _VocabularyDigest()
    state, vocabulary = _state(tokenizer, vocabulary_digest)
    prompt, prompt_ids = _render(
        tokenizer, messages, True, state, vocabulary, vocabulary_digest=vocabulary_digest)
    input_record = {
        "messages_sha256": digest(messages), "prompt": prompt,
        "prompt_sha256": _text_hash(prompt), "input_ids": prompt_ids,
        "input_tokens": len(prompt_ids), "remaining_input_tokens": 4096 - len(prompt_ids),
    }
    result = {
        "kind": "TOKENIZER_MEASUREMENT_UNATTESTED", "state_sha256": digest(state),
        "status": "OK", "input": input_record, "target": None,
    }
    if len(prompt_ids) > 4096:
        return {**result, "status": "INPUT_LIMIT"}
    if answer is None:
        return result
    completed = copy.deepcopy(messages) + [{"role": "assistant", "content": answer}]
    full_text, full_ids = _render(
        tokenizer, completed, False, state, vocabulary, answer_prefix=prompt + answer,
        vocabulary_digest=vocabulary_digest)
    require(full_text.startswith(prompt), "answer text prefix mismatch")
    require(full_text[len(prompt):].startswith(answer), "answer text changed")
    require(full_ids[:len(prompt_ids)] == prompt_ids, "answer token prefix mismatch")
    suffix = full_ids[len(prompt_ids):]
    endings = [index for index, value in enumerate(suffix)
               if value == state["eos_token_id"]]
    require(len(endings) == 1, "unique answer termination token required")
    endpoint = endings[0] + 1
    target_ids = suffix[:endpoint]
    result["target"] = {
        "answer_sha256": _text_hash(answer), "full_prompt_sha256": _text_hash(full_text),
        "full_sequence_ids": full_ids, "full_sequence_tokens": len(full_ids),
        "target_start": len(prompt_ids), "target_ids": target_ids,
        "target_tokens": len(target_ids), "template_tail_ids": suffix[endpoint:],
    }
    if len(target_ids) > 256:
        result["status"] = "OUTPUT_LIMIT"
    return result
