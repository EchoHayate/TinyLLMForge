from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tinyvllm.engine.speculative_selection import (
    SpeculativeSelectionConfig,
)
from tinyvllm.speculative.adapter import (
    DraftAdapter,
    DraftCapabilities,
    validate_draft_capabilities,
)


class DraftLifecycle(Protocol):
    def register_sequence(
        self,
        sequence_id: int,
        verified_token_ids: tuple[int, ...],
    ) -> None:
        ...

    def synchronize_verified_history(
        self,
        sequence_id: int,
        verified_token_ids: tuple[int, ...],
    ) -> int:
        ...

    def release_sequence(self, sequence_id: int) -> None:
        ...


@dataclass(frozen=True)
class ModelRunnerProposalExecutorDescriptor:
    executor_id: str
    capabilities: DraftCapabilities


@dataclass(frozen=True)
class EngineSpeculativeRuntime:
    draft_adapter: DraftAdapter | None = None
    model_runner_executor: (
        ModelRunnerProposalExecutorDescriptor | None
    ) = None
    lifecycle: DraftLifecycle | None = None

    @property
    def capabilities(self) -> DraftCapabilities:
        configured_count = int(
            self.draft_adapter is not None
        ) + int(
            self.model_runner_executor is not None
        )
        if configured_count != 1:
            raise ValueError(
                "runtime must configure exactly one proposal source"
            )
        if self.draft_adapter is not None:
            return getattr(
                self.draft_adapter,
                "capabilities",
                None,
            )
        return getattr(
            self.model_runner_executor,
            "capabilities",
            None,
        )


def build_engine_speculative_selection_config(
    runtime: EngineSpeculativeRuntime,
    *,
    model_runner,
) -> SpeculativeSelectionConfig:
    if not isinstance(runtime, EngineSpeculativeRuntime):
        raise ValueError(
            "runtime must be EngineSpeculativeRuntime"
        )
    capabilities = runtime.capabilities
    if runtime.draft_adapter is not None:
        capabilities = validate_draft_capabilities(
            capabilities,
            expected_execution_domain="host",
        )
        if not callable(
            getattr(
                runtime.draft_adapter,
                "propose_batch",
                None,
            )
        ):
            raise ValueError(
                "draft adapter propose_batch must be callable"
            )
    else:
        descriptor = runtime.model_runner_executor
        if not isinstance(
            descriptor,
            ModelRunnerProposalExecutorDescriptor,
        ):
            raise ValueError(
                "model runner executor descriptor is invalid"
            )
        if (
            not isinstance(descriptor.executor_id, str)
            or not descriptor.executor_id
        ):
            raise ValueError(
                "model runner executor ID must be non-empty"
            )
        capabilities = validate_draft_capabilities(
            capabilities,
            expected_execution_domain="model_runner",
        )
        if getattr(model_runner, "world_size", None) != 1:
            raise ValueError(
                "model runner proposal execution supports TP1 only"
            )
        config = getattr(model_runner, "config", None)
        if getattr(config, "kv_offload_mvp0", False):
            raise ValueError(
                "model runner proposal execution requires "
                "KV offload disabled"
            )
    if not callable(getattr(model_runner, "call", None)):
        raise ValueError(
            "model runner callback bridge is unavailable"
        )
    lifecycle = runtime.lifecycle
    if lifecycle is not None:
        for name in (
            "register_sequence",
            "synchronize_verified_history",
            "release_sequence",
        ):
            if not callable(getattr(lifecycle, name, None)):
                raise ValueError(
                    "draft lifecycle must expose callable "
                    f"{name}"
                )
    return SpeculativeSelectionConfig(
        enabled=True,
        max_proposal_tokens=(
            capabilities.max_proposal_tokens
        ),
    )


def validate_engine_speculative_runtime(
    runtime: EngineSpeculativeRuntime,
    *,
    scheduler,
    model_runner,
) -> EngineSpeculativeRuntime:
    candidate = build_engine_speculative_selection_config(
        runtime,
        model_runner=model_runner,
    )
    selection = getattr(
        scheduler,
        "speculative_selection_config",
        None,
    )
    if getattr(selection, "enabled", None) is not True:
        raise ValueError(
            "Scheduler speculative selection must be enabled"
        )
    if getattr(
        selection,
        "max_proposal_tokens",
        None,
    ) != candidate.max_proposal_tokens:
        raise ValueError(
            "Scheduler proposal limit must match "
            "proposal source"
        )
    return runtime
