import copy
import hashlib
from importlib import metadata
import platform
import re

from pilot_protocol import FLAGS, fields, require
from validation_contract import parse_json


REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
FILE_PINS = {
    "config.json": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
}
REQUIRED_PACKAGES = {"python", "transformers", "tokenizers", "huggingface-hub", "jinja2"}


def validate_profile(raw, approved_sha256):
    require(type(raw) is bytes and len(raw) <= 1048576, "profile bytes")
    require(type(approved_sha256) is str and
            re.fullmatch("[0-9a-f]{64}", approved_sha256) is not None, "approval digest")
    require(hashlib.sha256(raw).hexdigest() == approved_sha256, "profile approval mismatch")
    profile = parse_json(raw)
    fields(profile, ("kind", "version", "run_id", "revision", "file_sha256", "versions"))
    require(profile["kind"] == "TOKENIZER_RUNTIME_PROFILE", "profile kind")
    require(type(profile["version"]) is int and profile["version"] == 1, "profile version")
    require(type(profile["run_id"]) is str and
            re.fullmatch("[0-9a-f]{32}", profile["run_id"]) is not None, "run ID")
    require(profile["revision"] == REVISION, "revision declaration")
    require(profile["file_sha256"] == FILE_PINS, "file pins")
    versions = profile["versions"]
    require(type(versions) is dict and REQUIRED_PACKAGES <= set(versions), "version inventory")
    for name, version in versions.items():
        require(type(name) is str and re.fullmatch("[a-z0-9]+(?:-[a-z0-9]+)*", name)
                is not None, "canonical package name")
        require(type(version) is str and bool(version) and
                not any(character.isspace() for character in version), "exact version")
    require(versions["python"] == "3.12.13" and
            versions["transformers"] == "4.57.1", "pinned runtime versions")
    return profile


def runtime_inventory():
    versions = {"python": platform.python_version()}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        require(type(name) is str and bool(name), "distribution name")
        canonical = re.sub("[-_.]+", "-", name).lower()
        require(canonical not in versions, "duplicate installed distribution")
        versions[canonical] = distribution.version
    return versions


def inspect_runtime(raw, approved_sha256):
    profile = validate_profile(raw, approved_sha256)
    observed = runtime_inventory()
    require(observed == profile["versions"], "installed inventory mismatch")
    return {
        "kind": "TOKENIZER_RUNTIME_METADATA_ONLY", "run_id": profile["run_id"],
        "profile_sha256": approved_sha256, "versions": copy.deepcopy(observed),
        "tokenizer_loaded": False, "all_possible_second_rounds_validated": False,
        **{flag: False for flag in FLAGS},
    }
