"""Action identity and side-effect classification.

Token-level speculation can discard a rejected token for free. An
agent action cannot: a rejected action may already have mutated an
external system. Every downstream profitability argument in this
package is therefore conditioned on an explicit, fail-closed
side-effect classification.

Trajectory-lossless invariant enforced by this module's consumers:

    The actor's emitted action sequence is the sole authority. A
    speculative observation may only be reused when the speculated
    action signature is byte-identical to the actor's action
    signature. Otherwise the speculative work is discarded and the
    actor path is executed.

This is weaker than the token-level distributional-losslessness
guarantee of ``tinyvllm.speculative`` and must never be described as
equivalent to it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Literal


SideEffectClass = Literal[
    "read_only",
    "sandboxable",
    "reversible",
    "irreversible",
    "unknown",
]

SPECULATION_ELIGIBLE_SIDE_EFFECT_CLASSES = (
    "read_only",
    "sandboxable",
    "reversible",
)


@dataclass(frozen=True)
class ActionSignature:
    """Canonical identity of one agent action.

    ``tool_name`` plus canonically serialised ``arguments`` define
    the match test. Two actions are considered identical only when
    their ``digest`` values are equal.
    """

    tool_name: str
    arguments_json: str

    @property
    def digest(self) -> str:
        payload = "\u0000".join((self.tool_name, self.arguments_json))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def matches(self, other: object) -> bool:
        if not isinstance(other, ActionSignature):
            return False
        return self.digest == other.digest


@dataclass(frozen=True)
class ToolContract:
    """Static, operator-declared properties of one tool."""

    tool_name: str
    side_effect_class: SideEffectClass
    rollback_seconds: float
    sandbox_available: bool = False

    @property
    def speculation_eligible(self) -> bool:
        if self.side_effect_class not in (
            SPECULATION_ELIGIBLE_SIDE_EFFECT_CLASSES
        ):
            return False
        if self.side_effect_class == "sandboxable":
            return bool(self.sandbox_available)
        return True

    @property
    def ineligibility_reason(self) -> str | None:
        if self.speculation_eligible:
            return None
        if self.side_effect_class == "unknown":
            return "side_effect_class unknown; fail closed"
        if self.side_effect_class == "irreversible":
            return "irreversible side effect; speculation forbidden"
        return "sandboxable tool without an available sandbox"


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_non_negative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    normalized = float(value)
    if normalized < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return normalized


def canonical_arguments_json(arguments: object) -> str:
    """Serialise tool arguments deterministically.

    Key order, whitespace and unicode escaping are all pinned so that
    the match test cannot be influenced by dictionary iteration order
    or formatting.
    """

    if not isinstance(arguments, dict):
        raise ValueError("arguments must be a dict")
    for key in arguments:
        if not isinstance(key, str):
            raise ValueError("argument keys must be strings")
    return json.dumps(
        arguments,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def build_action_signature(
    *,
    tool_name: object,
    arguments: object,
) -> ActionSignature:
    return ActionSignature(
        _require_str(tool_name, "tool_name"),
        canonical_arguments_json(arguments),
    )


def build_tool_contract(
    *,
    tool_name: object,
    side_effect_class: object,
    rollback_seconds: object,
    sandbox_available: object = False,
) -> ToolContract:
    name = _require_str(tool_name, "tool_name")
    if side_effect_class not in (
        "read_only",
        "sandboxable",
        "reversible",
        "irreversible",
        "unknown",
    ):
        raise ValueError("side_effect_class is not a known class")
    if not isinstance(sandbox_available, bool):
        raise ValueError("sandbox_available must be a bool")
    return ToolContract(
        name,
        side_effect_class,
        _require_non_negative_float(
            rollback_seconds,
            "rollback_seconds",
        ),
        sandbox_available,
    )


def tool_contract_to_dict(contract: ToolContract) -> dict:
    payload = asdict(contract)
    payload["speculation_eligible"] = contract.speculation_eligible
    payload["ineligibility_reason"] = contract.ineligibility_reason
    return payload
