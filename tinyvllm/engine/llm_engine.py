from tinyvllm.sampling_params import SamplingParams
from tinyvllm.config import Config
from tinyvllm.engine.model_runner import ModelRunner
from tinyvllm.engine.h2d_slot_reuse_diagnostic import (
    H2D_SLOT_REUSE_SCHEMA,
)
from tinyvllm.engine.scheduler import Scheduler, ScheduledOutputRow
from tinyvllm.engine.sequence import Sequence
from tinyvllm.engine.speculative_execution import (
    build_engine_prepared_speculative_commit_rows,
    build_engine_speculative_partition,
    build_speculative_residency_precommit_rows,
    build_speculative_residency_prepare_rows,
)
from tinyvllm.engine.speculative_model_runner import (
    build_model_runner_side_state_callbacks,
    build_model_runner_proposal_provider,
    commit_model_runner_proposal_finalize_batch,
    prepare_model_runner_proposal_finalize_batch,
    release_model_runner_proposal_sequence,
    rollback_model_runner_proposal_finalize_batch,
    run_model_runner_tail_batch,
)
from tinyvllm.engine.speculative_runtime import (
    EngineSpeculativeRuntime,
    build_engine_speculative_selection_config,
    validate_engine_speculative_runtime,
)
from tinyvllm.engine.speculative_residency import (
    build_kv_block_identity_rows,
)
from tinyvllm.speculative import (
    prepare_native_speculative_batch,
    rollback_prepared_native_speculative_batch,
)
from tinyvllm.speculative.batch_runtime import (
    apply_prepared_speculative_side_state,
    build_prepared_proposal_finalize_rows,
    rollback_prepared_speculative_side_state,
    seal_prepared_speculative_side_state,
)
from tinyvllm.engine.model_runner_command_ack import (
    ModelRunnerCommandAckCollector,
)
from tinyvllm.engine.qwen35_hybrid_prefix_engine_restore import (
    Qwen35HybridPrefixEngineRestoreCoordinator,
)
from tinyvllm.engine.qwen35_hybrid_prefix_publication_ticket import (
    Qwen35HybridPrefixPublicationPayload,
)
from tinyvllm.engine.qwen35_hybrid_prefix_engine_publication import (
    Qwen35HybridPrefixEnginePublicationCoordinator,
)
from tinyvllm.engine.qwen35_hybrid_prefix_source_publication import (
    Qwen35HybridPrefixSourcePublisher,
)
from tinyvllm.engine.qwen35_hybrid_prefix_runtime_identity import (
    Qwen35HybridPrefixRuntimeIdentity,
    validate_qwen35_model_fingerprint,
)
from tinyvllm.engine.qwen35_hybrid_prefix_cache import (
    Qwen35HybridPrefixKey,
)
from tinyvllm.engine.hybrid_state import HybridStateSlotAllocator

from dataclasses import fields
from itertools import count

import gc
from time import perf_counter
import time
import atexit
from tqdm.auto import tqdm
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer


def _try_qwen35_hybrid_prefix_restore(
    engine,
    sequence,
    *,
    key_type=None,
):
    if key_type is None:
        key_type = Qwen35HybridPrefixKey
    coordinator = getattr(
        engine,
        "qwen35_hybrid_prefix_engine_restore_coordinator",
        None,
    )
    identity = getattr(
        engine,
        "qwen35_hybrid_prefix_runtime_identity",
        None,
    )
    if coordinator is None or identity is None:
        return None
    block_manager = engine.scheduler.block_manager
    token_count = block_manager.max_reusable_tokens(sequence)
    if token_count <= 0:
        sequence.hybrid_prefix_restore_attempted = True
        sequence.hybrid_prefix_restore_hit = False
        return False
    token_ids = tuple(sequence.token_ids[:token_count])
    terminal_hash = -1
    for start in range(0, token_count, block_manager.block_size):
        terminal_hash = block_manager.compute_hash(
            list(
                token_ids[
                    start:start + block_manager.block_size
                ]
            ),
            terminal_hash,
        )
    key = key_type(
        token_hash=terminal_hash,
        token_count=token_count,
        terminal_block_hash=terminal_hash,
        block_size=block_manager.block_size,
        model_fingerprint=identity.model_fingerprint,
        layout_fingerprint=identity.layout_fingerprint,
        tensor_parallel_size=engine.model_runner.world_size,
        dtype=identity.dtype,
    )
    engine.flush_pending_hybrid_state_releases(
        timeout_s=coordinator.timeout_s,
    )
    restored = bool(
        engine.acquire_qwen35_hybrid_prefix(
            sequence,
            key,
            token_ids,
        )
    )
    sequence.hybrid_prefix_restore_attempted = True
    sequence.hybrid_prefix_restore_hit = restored
    return restored


def _call_speculative_proposal_lifecycle(
    engine,
    method_name,
    *args,
):
    local_result, worker_acks = (
        engine.call_model_runner_acknowledged(
            method_name,
            *args,
            timeout_s=60.0,
        )
    )
    expected_ranks = tuple(
        range(1, engine.model_runner.world_size)
    )
    observed = {}
    for acknowledgement in worker_acks:
        rank = getattr(acknowledgement, "rank", None)
        result = getattr(acknowledgement, "result", None)
        if (
            rank not in expected_ranks
            or rank in observed
            or result != local_result
        ):
            engine._poison_model_runner_ack_collector(
                "proposal lifecycle acknowledgement is invalid"
            )
            raise RuntimeError(
                "proposal lifecycle acknowledgement is invalid"
            )
        observed[rank] = result
    if tuple(sorted(observed)) != expected_ranks:
        engine._poison_model_runner_ack_collector(
            "proposal lifecycle acknowledgement ranks are incomplete"
        )
        raise RuntimeError(
            "proposal lifecycle acknowledgement ranks are incomplete"
        )
    history = getattr(
        engine,
        "speculative_proposal_lifecycle_ack_rows",
        None,
    )
    if history is None:
        history = []
        engine.speculative_proposal_lifecycle_ack_rows = history
    history.append({
        "method_name": method_name,
        "worker_ranks": list(expected_ranks),
    })
    return local_result


def _commit_prepared_speculative_publication(
    engine,
    runtime,
    prepared_runtime,
    kv_plans,
    prepared_scheduler,
    *,
    clock=None,
):
    if clock is None:
        clock = __import__("time").perf_counter
    started_at = clock()
    lifecycle_dispatch = (
        lambda method_name, *args: (
            _call_speculative_proposal_lifecycle(
                engine,
                method_name,
                *args,
            )
        )
    )
    finalize_rows = build_prepared_proposal_finalize_rows(
        prepared_runtime
    )
    finalize_ticket = None
    descriptor = getattr(
        runtime,
        "model_runner_executor",
        None,
    )
    if finalize_rows:
        if descriptor is None:
            raise RuntimeError(
                "proposal finalization rows require a "
                "ModelRunner executor descriptor"
            )
        finalize_ticket = (
            prepare_model_runner_proposal_finalize_batch(
                engine.model_runner,
                descriptor,
                finalize_rows,
                dispatch=lifecycle_dispatch,
            )
        )
    side_applied = False
    try:
        apply_prepared_speculative_side_state(
            prepared_runtime
        )
        side_applied = (
            prepared_runtime.side_state_state == "applied"
        )
        if kv_plans:
            (
                engine.scheduler.block_manager
                .commit_speculative_kv_commit_batch(kv_plans)
            )
        try:
            engine.scheduler.commit_prepared_postprocess(
                prepared_scheduler
            )
        except BaseException:
            for plan in kv_plans:
                if plan.transaction.state == "committed":
                    plan.transaction.state = "materialized"
            raise
    except BaseException as error:
        for plan in kv_plans:
            if plan.transaction.state == "committed":
                plan.transaction.state = "materialized"
        rollback_errors = []
        if finalize_ticket is not None:
            try:
                rollback_model_runner_proposal_finalize_batch(
                    engine.model_runner,
                    descriptor,
                    finalize_ticket,
                    dispatch=lifecycle_dispatch,
                )
            except BaseException as rollback_error:
                rollback_errors.append((
                    "proposal finalization",
                    rollback_error,
                ))
        if (
            prepared_runtime.side_state_callbacks is not None
            and prepared_runtime.side_state_state
            != "rolled_back"
        ):
            try:
                rollback_prepared_speculative_side_state(
                    prepared_runtime
                )
            except BaseException as rollback_error:
                rollback_errors.append((
                    "speculative side state",
                    rollback_error,
                ))
        if rollback_errors:
            name, rollback_error = rollback_errors[0]
            engine.speculative_runtime_poisoned = True
            engine.speculative_runtime_poison_reason = (
                f"{name} rollback failed: {rollback_error}"
            )
            raise rollback_error from error
        raise
    prepared_runtime.state = "committed"
    if finalize_ticket is not None:
        try:
            commit_model_runner_proposal_finalize_batch(
                engine.model_runner,
                descriptor,
                finalize_ticket,
                dispatch=lifecycle_dispatch,
            )
        except BaseException as error:
            engine.speculative_runtime_poisoned = True
            engine.speculative_runtime_poison_reason = (
                "proposal finalization commit failed: "
                f"{error}"
            )
            raise
    try:
        seal_prepared_speculative_side_state(
            prepared_runtime
        )
    except BaseException as error:
        engine.speculative_runtime_poisoned = True
        engine.speculative_runtime_poison_reason = (
            "speculative side-state seal failed: "
            f"{error}"
        )
        raise
    timing_ms = getattr(
        prepared_runtime,
        "timing_ms",
        None,
    )
    if isinstance(timing_ms, dict):
        timing_ms["commit_metadata_ms"] = float(
            timing_ms.get("commit_metadata_ms", 0.0)
        ) + (clock() - started_at) * 1000.0


class LLMEngine:

    def _kv_offload_identity_rows(
        self,
        seqs: tuple[Sequence, ...],
    ):
        if not self.model_runner.config.kv_offload_mvp0:
            return ()
        return build_kv_block_identity_rows(
            self.scheduler.block_manager,
            seqs,
        )

    @staticmethod
    def _create_worker_ack_channels(ctx, worker_count):
        if (
            isinstance(worker_count, bool)
            or not isinstance(worker_count, int)
            or worker_count < 0
        ):
            raise ValueError(
                "worker_count must be a non-negative integer"
            )
        receivers = []
        senders = []
        for rank in range(1, worker_count + 1):
            receiver, sender = ctx.Pipe(duplex=False)
            receivers.append((rank, receiver))
            senders.append((rank, sender))
        return tuple(receivers), tuple(senders)

    def _close_worker_ack_channels(self):
        for _, receiver in getattr(
            self,
            "model_runner_ack_receivers",
            (),
        ):
            receiver.close()
        for _, sender in getattr(
            self,
            "model_runner_ack_parent_senders",
            (),
        ):
            sender.close()

    def _is_worker_rank_alive(self, rank):
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank <= 0
            or rank > len(self.ps)
        ):
            raise ValueError(f"worker rank out of range: {rank}")
        return bool(self.ps[rank - 1].is_alive())
    
    def __init__(self, model, **kwargs):
        self._clock_ns = kwargs.pop("_clock_ns", time.monotonic_ns)
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}    #过滤掉和config无关的参数
        config  = Config(model, **config_kwargs)       
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")                       # 生成全新解释器，继承基本资源，全局变量，打开的文件，线程不会被继承
        (
            self.model_runner_ack_receivers,
            self.model_runner_ack_parent_senders,
        ) = self._create_worker_ack_channels(
            ctx,
            config.tensor_parallel_size - 1,
        )
        ack_senders = dict(
            self.model_runner_ack_parent_senders
        )
        for i in range(1, config.tensor_parallel_size):     # 生成所有的子进程
            event = ctx.Event()                             #进程间同步的“信号量”，用于进程间通信
            sender = ack_senders[i]
            process = ctx.Process(
                target=ModelRunner,
                args=(config, i, event, sender),
            ) #创建子进程对象 modelrunner是子进程要执行的目标函数
            process.start()
            sender.close()
            self.ps.append(process)
            self.events.append(event)

        self.model_runner = ModelRunner(config, 0, self.events)     # 生成主进程
        self.model_runner_ack_collector = (
            ModelRunnerCommandAckCollector(
                self.model_runner_ack_receivers
            )
            if self.model_runner_ack_receivers
            else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast = True)
        config.eos = self.tokenizer.eos_token_id
        qwen35_owner = getattr(
            self.model_runner,
            "qwen35_hybrid_model_owner",
            None,
        )
        state_allocator = (
            HybridStateSlotAllocator(qwen35_owner.pool.capacity)
            if qwen35_owner is not None
            else None
        )
        self.scheduler = Scheduler(
            config,
            hybrid_state_allocator=state_allocator,
        )
        self.qwen35_hybrid_prefix_engine_restore_coordinator = None
        self.qwen35_hybrid_prefix_engine_publication_coordinator = None
        self.qwen35_hybrid_prefix_restore_configuration = None
        self.qwen35_hybrid_prefix_source_publisher = None
        self.qwen35_hybrid_prefix_source_publisher_hook = None
        self.qwen35_hybrid_prefix_source_publisher_configuration = None
        self.qwen35_hybrid_prefix_runtime_identity = None
        self.qwen35_hybrid_prefix_runtime_identity_configuration = None
        self.qwen35_hybrid_prefix_publication_runtime_configuration = None
        self.qwen35_hybrid_prefix_publication_runtime_publisher = None
        self.qwen35_loaded_checkpoint_candidate_binding_configuration = (
            None
        )
        self.qwen35_loaded_checkpoint_candidate_binding_rows = None
        self.last_batch_kind = None
        self.last_scheduled_seqs = []
        self.last_step_observation = None
        self.speculative_runtime = None
        self.speculative_runtime_poisoned = False
        self.speculative_runtime_poison_reason = None
        self._speculative_residency_ticket_ids = count(1)
        atexit.register(self.exit)
        self._exit_handler_registered = True

    def capacity_snapshot(self):
        return {
            "num_kvcache_blocks": int(
                self.model_runner.config.num_kvcache_blocks
            ),
            "block_size": int(self.scheduler.block_manager.block_size),
        }

    def activate_speculative_runtime(
        self,
        runtime: EngineSpeculativeRuntime,
    ) -> None:
        candidate_selection = (
            build_engine_speculative_selection_config(
                runtime,
                model_runner=self.model_runner,
            )
        )
        current_runtime = self.speculative_runtime
        current_selection = (
            self.scheduler.speculative_selection_config
        )
        selection_installed = bool(
            self.scheduler._speculative_selection_installed
        )
        if current_runtime is runtime:
            if (
                selection_installed
                and current_selection == candidate_selection
            ):
                return
            raise RuntimeError(
                "speculative runtime and Scheduler selection "
                "are not atomically active"
            )
        if current_runtime is not None:
            raise RuntimeError(
                "speculative runtime is already installed"
            )
        if (
            selection_installed
            and current_selection != candidate_selection
        ):
            raise RuntimeError(
                "speculative selection config is already installed"
            )

        previous_runtime = current_runtime
        previous_poisoned = (
            self.speculative_runtime_poisoned
        )
        previous_poison_reason = (
            self.speculative_runtime_poison_reason
        )
        previous_selection = current_selection
        previous_selection_installed = selection_installed
        try:
            self.scheduler.install_speculative_selection(
                candidate_selection
            )
            self.install_speculative_runtime(runtime)
        except BaseException:
            self.scheduler.speculative_selection_config = (
                previous_selection
            )
            self.scheduler._speculative_selection_installed = (
                previous_selection_installed
            )
            self.speculative_runtime = previous_runtime
            self.speculative_runtime_poisoned = (
                previous_poisoned
            )
            self.speculative_runtime_poison_reason = (
                previous_poison_reason
            )
            raise

    def install_speculative_runtime(
        self,
        runtime: EngineSpeculativeRuntime,
    ) -> None:
        current = self.speculative_runtime
        if current is runtime:
            return
        if current is not None:
            raise RuntimeError(
                "speculative runtime is already installed"
            )
        validate_engine_speculative_runtime(
            runtime,
            scheduler=self.scheduler,
            model_runner=self.model_runner,
        )
        self.speculative_runtime = runtime
        self.speculative_runtime_poisoned = False
        self.speculative_runtime_poison_reason = None
        
    def exit(self):
        existing_receipt = getattr(self, "_exit_receipt", None)
        if existing_receipt is not None:
            return dict(existing_receipt)
        if getattr(self, "_exit_handler_registered", False):
            atexit.unregister(self.exit)
            self._exit_handler_registered = False
        drain_releases = getattr(
            self.scheduler,
            "drain_hybrid_state_release_events",
            None,
        )
        released_leases = (
            drain_releases()
            if drain_releases is not None
            else ()
        )
        if released_leases:
            self.model_runner.call(
                "release_hybrid_state",
                released_leases,
            )
        rank_cleanup_receipts = []
        try:
            acknowledged_exit = getattr(
                self,
                "call_model_runner_acknowledged",
                None,
            )
            if acknowledged_exit is None:
                local_receipt = self.model_runner.call("exit")
                worker_receipts = ()
            else:
                local_receipt, worker_acks = acknowledged_exit(
                    "exit",
                    timeout_s=60.0,
                )
                worker_receipts = tuple(
                    ack.result for ack in worker_acks
                )
            if not isinstance(local_receipt, dict):
                raise RuntimeError(
                    "rank 0 cleanup receipt is unavailable"
                )
            rank_cleanup_receipts.append(dict(local_receipt))
            rank_cleanup_receipts.extend(
                dict(receipt)
                for receipt in worker_receipts
                if isinstance(receipt, dict)
            )
            del self.model_runner                           # 显式释放
            for p in self.ps:
                p.join()
            gc.collect()
            torch.cuda.empty_cache()
            rank_exit_codes = [0] + [
                process.exitcode for process in self.ps
            ]
            owned_children_remaining = [
                rank
                for rank, process in enumerate(self.ps, start=1)
                if process.is_alive()
            ]
            expected_ranks = list(range(len(self.ps) + 1))
            receipt_ranks = sorted(
                receipt.get("rank")
                for receipt in rank_cleanup_receipts
                if isinstance(receipt.get("rank"), int)
            )
            process_group_destroyed = (
                receipt_ranks == expected_ranks
                and all(
                    receipt.get("process_group_destroyed") is True
                    for receipt in rank_cleanup_receipts
                )
                and rank_exit_codes == [0] * len(expected_ranks)
                and not owned_children_remaining
            )
            receipt = {
                "process_group_destroyed": process_group_destroyed,
                "rank_exit_codes": rank_exit_codes,
                "owned_children_remaining": owned_children_remaining,
                "rank_cleanup_receipts": sorted(
                    rank_cleanup_receipts,
                    key=lambda row: row["rank"],
                ),
            }
            self._exit_receipt = dict(receipt)
            return receipt
        finally:
            close_ack_channels = getattr(
                self,
                "_close_worker_ack_channels",
                None,
            )
            if close_ack_channels is not None:
                close_ack_channels()

    def call_model_runner_acknowledged(
        self,
        method_name,
        *args,
        timeout_s,
    ):
        envelope = self.model_runner.dispatch_command(
            method_name,
            *args,
            requires_ack=(self.model_runner.world_size > 1),
        )
        collector = self.model_runner_ack_collector
        if self.model_runner.world_size > 1 and collector is None:
            raise RuntimeError(
                "ModelRunner acknowledgement collector is not installed"
            )

        def record_terminal_error(error):
            if envelope.trace_identity is None:
                return
            try:
                self.model_runner.command_timeline.record_terminal_error(
                    envelope.command_id,
                    finished_ns=self._clock_ns(),
                    error_type=type(error).__name__,
                    error_detail=str(error),
                )
            except BaseException:
                pass

        try:
            local_result = (
                self.model_runner.execute_command_envelope(envelope)
            )
        except BaseException as error:
            if collector is not None:
                collector.poison(
                    "rank 0 command failed after dispatch: "
                    f"{type(error).__name__}: {error}"
                )
            record_terminal_error(error)
            raise
        if self.model_runner.world_size == 1:
            return local_result, ()
        ack_wait_started = self._clock_ns()
        if envelope.trace_identity is not None:
            self.model_runner.command_timeline.record_ack_wait_start(
                envelope.command_id,
                started_ns=ack_wait_started,
            )
        try:
            worker_acks = collector.collect(
                envelope.command_id,
                expected_ranks=tuple(
                    range(1, self.model_runner.world_size)
                ),
                timeout_s=timeout_s,
                is_rank_alive=self._is_worker_rank_alive,
            )
        except BaseException as error:
            record_terminal_error(error)
            raise
        ack_wait_finished = self._clock_ns()
        if envelope.trace_identity is not None:
            self.model_runner.command_timeline.record_ack_wait_end(
                envelope.command_id,
                finished_ns=ack_wait_finished,
            )
        return local_result, worker_acks

    def configure_command_timeline(
        self,
        enabled,
        max_rows,
        timeout_s,
    ):
        local_result, worker_acks = (
            self.call_model_runner_acknowledged(
                "configure_command_timeline",
                enabled,
                max_rows,
                timeout_s=timeout_s,
            )
        )
        rows = (local_result,) + tuple(
            acknowledgement.result
            for acknowledgement in worker_acks
        )
        expected_ranks = tuple(
            range(self.model_runner.world_size)
        )
        if (
            tuple(row.get("rank") for row in rows) != expected_ranks
            or any(
                row.get("enabled") is not enabled
                or row.get("max_rows") != max_rows
                for row in rows
            )
        ):
            raise ValueError(
                "command timeline configuration rank inventory mismatch"
            )
        return {
            "enabled": enabled,
            "max_rows": max_rows,
            "rank_inventory": list(expected_ranks),
        }

    def reset_command_timeline(self, timeout_s):
        local_result, worker_acks = (
            self.call_model_runner_acknowledged(
                "reset_command_timeline",
                timeout_s=timeout_s,
            )
        )
        return (local_result,) + tuple(
            acknowledgement.result
            for acknowledgement in worker_acks
        )

    def command_timeline_snapshots(self, timeout_s):
        local_result, worker_acks = (
            self.call_model_runner_acknowledged(
                "command_timeline_snapshot",
                timeout_s=timeout_s,
            )
        )
        return (local_result,) + tuple(
            acknowledgement.result
            for acknowledgement in worker_acks
        )

    def _call_speculative_side_state_phase(
        self,
        method_name,
        *args,
    ):
        local_result, _ = self.call_model_runner_acknowledged(
            method_name,
            *args,
            timeout_s=60.0,
        )
        return local_result

    def _call_speculative_residency_phase(
        self,
        method_name,
        ticket_id,
        payload=None,
        *,
        expected_operation,
        expected_status,
        expected_sequence_ids,
        expected_committed_block_identities,
        expected_rejected_block_identities,
        timeout_s,
    ):
        args = (
            (ticket_id,)
            if payload is None
            else (ticket_id, payload)
        )
        local_result, worker_acks = (
            self.call_model_runner_acknowledged(
                method_name,
                *args,
                timeout_s=timeout_s,
            )
        )
        ranked = [(0, local_result)]
        ranked.extend(
            (ack.rank, ack.result)
            for ack in worker_acks
        )
        required_fields = {
            "ticket_id",
            "participant_id",
            "operation",
            "status",
            "sequence_ids",
            "committed_block_identities",
            "rejected_block_identities",
            "detail",
        }
        rows = {}
        invalid = False
        for outer_rank, row in ranked:
            if (
                not isinstance(row, dict)
                or set(row) != required_fields
                or row.get("ticket_id") != ticket_id
                or row.get("participant_id") != outer_rank
                or row.get("operation")
                != expected_operation
                or row.get("status") != expected_status
                or row.get("sequence_ids")
                != expected_sequence_ids
                or row.get("committed_block_identities")
                != expected_committed_block_identities
                or row.get("rejected_block_identities")
                != expected_rejected_block_identities
                or row.get("detail") != ""
                or outer_rank in rows
            ):
                invalid = True
                break
            rows[outer_rank] = dict(row)
        expected_ranks = tuple(
            range(self.model_runner.world_size)
        )
        if (
            invalid
            or tuple(sorted(rows)) != expected_ranks
        ):
            reason = (
                "speculative residency acknowledgement is invalid"
            )
            collector = getattr(
                self,
                "model_runner_ack_collector",
                None,
            )
            if collector is not None:
                collector.poison(reason)
            raise RuntimeError(reason)
        return tuple(
            rows[rank]
            for rank in expected_ranks
        )

    def kv_offload_summaries(
        self,
        *,
        timeout_s=60.0,
    ):
        local_result, worker_acks = (
            self.call_model_runner_acknowledged(
                "kv_offload_summary",
                timeout_s=timeout_s,
            )
        )
        ranked = [(0, local_result)]
        ranked.extend(
            (ack.rank, ack.result)
            for ack in worker_acks
        )
        rows = {}
        for rank, summary in ranked:
            if (
                not isinstance(summary, dict)
                or rank in rows
            ):
                raise RuntimeError(
                    "KV offload summary acknowledgement is invalid"
                )
            rows[rank] = dict(summary)
        expected_ranks = tuple(
            range(self.model_runner.world_size)
        )
        if tuple(sorted(rows)) != expected_ranks:
            raise RuntimeError(
                "KV offload summary ranks are incomplete"
            )
        return tuple(
            rows[rank]
            for rank in expected_ranks
        )

    def configure_decode_internal_profile(
        self,
        enabled,
        profile_label,
        *,
        timeout_s,
    ):
        local_result, worker_acks = (
            self.call_model_runner_acknowledged(
                "configure_decode_internal_profile",
                enabled,
                profile_label,
                timeout_s=timeout_s,
            )
        )
        ranked = [(0, local_result)]
        ranked.extend(
            (ack.rank, ack.result)
            for ack in worker_acks
        )
        rows = {}
        for outer_rank, row in ranked:
            if (
                not isinstance(row, dict)
                or row.get("rank") != outer_rank
                or row.get("enabled") is not enabled
                or row.get("profile_label") != profile_label
                or outer_rank in rows
            ):
                raise RuntimeError(
                    "decode internal profile configuration "
                    "acknowledgement is invalid"
                )
            rows[outer_rank] = dict(row)
        expected = tuple(range(self.model_runner.world_size))
        if tuple(sorted(rows)) != expected:
            raise RuntimeError(
                "decode internal profile configuration ranks "
                "are incomplete"
            )
        return {
            "enabled": enabled,
            "rank_inventory": list(expected),
        }

    def finalize_decode_internal_profile(self, *, timeout_s):
        local_result, worker_acks = (
            self.call_model_runner_acknowledged(
                "finalize_decode_internal_profile",
                timeout_s=timeout_s,
            )
        )
        ranked = [(0, local_result)]
        ranked.extend(
            (ack.rank, ack.result)
            for ack in worker_acks
        )
        rows = {}
        for outer_rank, row in ranked:
            if (
                not isinstance(row, dict)
                or row.get("rank") != outer_rank
                or row.get("enabled") is not True
                or row.get("finalization_status") != "complete"
                or not isinstance(row.get("steps"), list)
                or not isinstance(row.get("collectives"), list)
                or outer_rank in rows
            ):
                raise RuntimeError(
                    "decode internal profile finalization "
                    "acknowledgement is invalid"
                )
            rows[outer_rank] = dict(row)
        expected = tuple(range(self.model_runner.world_size))
        if tuple(sorted(rows)) != expected:
            raise RuntimeError(
                "decode internal profile finalization ranks "
                "are incomplete"
            )
        return {
            "enabled": True,
            "rank_inventory": list(expected),
            "ranks": [rows[rank] for rank in expected],
        }

    def _collect_qwen35_recurrent_capture_rows(
        self,
        method_name,
        expected_fields,
        *args,
        timeout_s,
    ):
        local_result, worker_acks = (
            self.call_model_runner_acknowledged(
                method_name,
                *args,
                timeout_s=timeout_s,
            )
        )
        ranked = [(0, local_result)]
        ranked.extend(
            (ack.rank, ack.result)
            for ack in worker_acks
        )
        expected_fields = set(expected_fields)
        expected_ranks = tuple(range(self.model_runner.world_size))
        rows = {}
        reference = None
        for outer_rank, row in ranked:
            if (
                type(outer_rank) is not int
                or outer_rank in rows
                or not isinstance(row, dict)
                or set(row) != expected_fields
                or type(row.get("rank")) is not int
                or row.get("rank") != outer_rank
            ):
                raise ValueError(
                    "recurrent capture rank result fields or rank mismatch"
                )
            non_rank = {
                key: value
                for key, value in row.items()
                if key != "rank"
            }
            if reference is None:
                reference = non_rank
            elif non_rank != reference:
                raise ValueError(
                    "recurrent capture rank result mismatch"
                )
            rows[outer_rank] = dict(row)
        if tuple(sorted(rows)) != expected_ranks:
            raise ValueError(
                "recurrent capture rank inventory mismatch"
            )
        return tuple(rows[rank] for rank in expected_ranks)

    def configure_qwen35_recurrent_capture(
        self,
        *,
        capture_root,
        model_manifest_sha256,
        source_tree_sha256,
        workload_manifest_sha256,
        world_size,
        workload_ids,
        timeout_s,
    ):
        if world_size != self.model_runner.world_size:
            raise ValueError(
                "recurrent capture world_size does not match engine"
            )
        configuration = {
            "capture_root": capture_root,
            "model_manifest_sha256": model_manifest_sha256,
            "source_tree_sha256": source_tree_sha256,
            "workload_manifest_sha256": workload_manifest_sha256,
            "world_size": world_size,
            "workload_ids": workload_ids,
        }
        rows = self._collect_qwen35_recurrent_capture_rows(
            "configure_qwen35_recurrent_capture",
            {
                "rank",
                "configured",
                "workload_ids",
                "linear_layer_indices",
            },
            configuration,
            timeout_s=timeout_s,
        )
        if (
            rows[0]["configured"] is not True
            or rows[0]["workload_ids"] != tuple(workload_ids)
        ):
            raise ValueError(
                "recurrent capture configuration acknowledgement mismatch"
            )
        return rows

    def arm_qwen35_recurrent_capture(
        self,
        workload_id,
        *,
        timeout_s,
    ):
        rows = self._collect_qwen35_recurrent_capture_rows(
            "arm_qwen35_recurrent_capture",
            {"rank", "workload_id", "armed"},
            workload_id,
            timeout_s=timeout_s,
        )
        if rows[0]["workload_id"] != workload_id:
            raise ValueError(
                "recurrent capture workload acknowledgement mismatch"
            )
        if rows[0]["armed"] is not True:
            raise ValueError(
                "recurrent capture armed status mismatch"
            )
        return rows

    def finish_qwen35_recurrent_capture_workload(
        self,
        workload_id,
        *,
        timeout_s,
    ):
        rows = self._collect_qwen35_recurrent_capture_rows(
            "finish_qwen35_recurrent_capture_workload",
            {"rank", "workload_id", "complete"},
            workload_id,
            timeout_s=timeout_s,
        )
        if rows[0]["workload_id"] != workload_id:
            raise ValueError(
                "recurrent capture workload acknowledgement mismatch"
            )
        if rows[0]["complete"] is not True:
            raise ValueError(
                "recurrent capture complete status mismatch"
            )
        return rows

    def enable_step_logits_authority_recording(
        self,
        enabled,
        *,
        timeout_s,
    ):
        local_result, worker_acks = (
            self.call_model_runner_acknowledged(
                "enable_step_logits_recording",
                bool(enabled),
                timeout_s=timeout_s,
            )
        )
        rows = [local_result]
        rows.extend(ack.result for ack in worker_acks)
        world_size = self.model_runner.world_size
        if (
            len(rows) != world_size
            or any(
                not isinstance(row, dict)
                or set(row) != {"rank", "enabled"}
                or row["enabled"] is not bool(enabled)
                for row in rows
            )
            or sorted(row["rank"] for row in rows)
            != list(range(world_size))
        ):
            raise ValueError(
                "step logits authority rank inventory mismatch"
            )
        return {
            "enabled": bool(enabled),
            "rank_inventory": list(range(world_size)),
        }

    def configure_h2d_slot_reuse_diagnostic(
        self,
        mode,
        *,
        timeout_s,
    ):
        local, acks = self.call_model_runner_acknowledged(
            "configure_h2d_slot_reuse_diagnostic",
            mode,
            timeout_s=timeout_s,
        )
        ranked = [(0, local)]
        ranked.extend((ack.rank, ack.result) for ack in acks)
        world_size = self.model_runner.world_size
        rows = {}
        for outer_rank, row in ranked:
            if (
                not isinstance(row, dict)
                or set(row) != {"rank", "mode"}
                or row.get("rank") != outer_rank
                or row.get("mode") != mode
                or outer_rank in rows
            ):
                raise ValueError(
                    "H2D slot-reuse configure rank or mode mismatch"
                )
            rows[outer_rank] = row
        expected = list(range(world_size))
        if sorted(rows) != expected:
            raise ValueError(
                "H2D slot-reuse configure rank inventory mismatch"
            )
        return {
            "mode": mode,
            "rank_inventory": expected,
        }

    def set_h2d_slot_reuse_diagnostic_context(
        self,
        engine_step,
        attention_stage="decode",
        *,
        timeout_s,
    ):
        local, acks = self.call_model_runner_acknowledged(
            "set_h2d_slot_reuse_diagnostic_context",
            engine_step,
            attention_stage,
            timeout_s=timeout_s,
        )
        ranked = [(0, local)]
        ranked.extend((ack.rank, ack.result) for ack in acks)
        world_size = self.model_runner.world_size
        rows = {}
        expected_fields = {
            "rank",
            "engine_step",
            "attention_stage",
        }
        for outer_rank, row in ranked:
            if (
                not isinstance(row, dict)
                or set(row) != expected_fields
                or row.get("rank") != outer_rank
                or row.get("engine_step") != engine_step
                or row.get("attention_stage")
                != attention_stage
                or outer_rank in rows
            ):
                raise ValueError(
                    "H2D slot-reuse context rank mismatch"
                )
            rows[outer_rank] = row
        expected = list(range(world_size))
        if sorted(rows) != expected:
            raise ValueError(
                "H2D slot-reuse context rank inventory mismatch"
            )
        return {
            "engine_step": engine_step,
            "attention_stage": attention_stage,
            "rank_inventory": expected,
        }

    def drain_h2d_slot_reuse_diagnostic(
        self,
        *,
        timing_epsilon_ms,
        expected_mode,
        timeout_s,
    ):
        local, acks = self.call_model_runner_acknowledged(
            "drain_h2d_slot_reuse_diagnostic",
            timing_epsilon_ms,
            timeout_s=timeout_s,
        )
        ranked = [(0, local)]
        ranked.extend((ack.rank, ack.result) for ack in acks)
        fields = {
            "rank",
            "schema",
            "mode",
            "stream_inventory",
            "read_rows",
            "overwrite_rows",
        }
        rows = {}
        for outer_rank, row in ranked:
            if (
                not isinstance(row, dict)
                or set(row) != fields
                or row.get("rank") != outer_rank
                or row.get("schema")
                != H2D_SLOT_REUSE_SCHEMA
                or row.get("mode") != expected_mode
                or outer_rank in rows
            ):
                raise ValueError(
                    "H2D slot-reuse drain rank, schema, "
                    "or mode mismatch"
                )
            rows[outer_rank] = row
        expected = tuple(range(self.model_runner.world_size))
        if tuple(sorted(rows)) != expected:
            raise ValueError(
                "H2D slot-reuse drain rank inventory mismatch"
            )
        return tuple(rows[rank] for rank in expected)

    def h2d_slot_reuse_diagnostic_summaries(
        self,
        *,
        timeout_s,
    ):
        local, acks = self.call_model_runner_acknowledged(
            "h2d_slot_reuse_diagnostic_summary",
            timeout_s=timeout_s,
        )
        ranked = [(0, local)]
        ranked.extend((ack.rank, ack.result) for ack in acks)
        fields = {
            "rank",
            "mode",
            "retained_event_count",
            "read_row_count",
            "overwrite_row_count",
        }
        rows = {}
        for outer_rank, row in ranked:
            if (
                not isinstance(row, dict)
                or set(row) != fields
                or row.get("rank") != outer_rank
                or outer_rank in rows
            ):
                raise ValueError(
                    "H2D slot-reuse summary rank mismatch"
                )
            rows[outer_rank] = row
        expected = tuple(range(self.model_runner.world_size))
        if tuple(sorted(rows)) != expected:
            raise ValueError(
                "H2D slot-reuse summary rank inventory mismatch"
            )
        return tuple(rows[rank] for rank in expected)

    def read_step_logits_authority(self):
        if self.model_runner.rank != 0:
            raise RuntimeError(
                "step logits authority is available only on rank zero"
            )
        logits = self.model_runner.last_step_logits()
        if logits is None:
            raise RuntimeError(
                "step logits authority evidence is unavailable"
            )
        return logits.clone()

    def qwen35_hybrid_prefix_cache_snapshots(self, *, timeout_s):
        local_result, worker_acks = (
            self.call_model_runner_acknowledged(
                "qwen35_hybrid_prefix_cache_snapshot",
                timeout_s=timeout_s,
            )
        )
        ranked = [(0, local_result)]
        ranked.extend(
            (ack.rank, ack.result)
            for ack in worker_acks
        )
        rows = {}
        for outer_rank, row in ranked:
            if (
                not isinstance(row, dict)
                or row.get("rank") != outer_rank
                or outer_rank in rows
            ):
                raise ValueError(
                    "hybrid prefix cache snapshot rank mismatch"
                )
            rows[outer_rank] = dict(row)
        expected = tuple(range(self.model_runner.world_size))
        if tuple(sorted(rows)) != expected:
            raise ValueError(
                "hybrid prefix cache snapshot rank inventory mismatch"
            )
        return tuple(rows[rank] for rank in expected)

    def autoregressive_draft_authority_snapshots(
        self,
        *,
        timeout_s,
    ):
        local_result, worker_acks = (
            self.call_model_runner_acknowledged(
                "autoregressive_draft_authority_snapshot",
                timeout_s=timeout_s,
            )
        )
        ranked = [(0, local_result)]
        ranked.extend(
            (ack.rank, ack.result)
            for ack in worker_acks
        )
        fields = {
            "rank",
            "world_size",
            "registered",
            "registration_consensus_sha256",
            "executor_descriptor",
            "checkpoint_identity",
            "tokenizer_contract",
            "registration_error",
            "executor",
        }
        rows = {}
        world_size = self.model_runner.world_size
        for outer_rank, row in ranked:
            if not isinstance(row, dict) or set(row) != fields:
                raise ValueError(
                    "autoregressive draft authority snapshot "
                    "fields mismatch"
                )
            if row["rank"] != outer_rank or outer_rank in rows:
                raise ValueError(
                    "autoregressive draft authority snapshot "
                    "rank mismatch"
                )
            if row["world_size"] != world_size:
                raise ValueError(
                    "autoregressive draft authority snapshot "
                    "world size mismatch"
                )
            rows[outer_rank] = dict(row)
        expected = tuple(range(world_size))
        if tuple(sorted(rows)) != expected:
            raise ValueError(
                "autoregressive draft authority snapshot "
                "rank inventory mismatch"
            )
        return tuple(rows[rank] for rank in expected)

    def qwen35_hybrid_prefix_authority_snapshots(self, *, timeout_s):
        local_result, worker_acks = (
            self.call_model_runner_acknowledged(
                "qwen35_hybrid_prefix_authority_snapshot",
                timeout_s=timeout_s,
            )
        )
        ranked = [(0, local_result)]
        ranked.extend(
            (ack.rank, ack.result)
            for ack in worker_acks
        )
        fields = {
            "rank",
            "current_entries",
            "hits",
            "misses",
            "publication_commits",
            "invalidations",
            "clears",
            "last_publication_block_identities",
        }
        rows = {}
        for outer_rank, row in ranked:
            if (
                not isinstance(row, dict)
                or set(row) != fields
                or row["rank"] != outer_rank
                or outer_rank in rows
            ):
                raise ValueError(
                    "hybrid prefix authority snapshot rank mismatch"
                )
            for name in fields - {
                "rank",
                "last_publication_block_identities",
            }:
                value = row[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise ValueError(
                        "hybrid prefix authority counter is invalid"
                    )
            block_identities = row[
                "last_publication_block_identities"
            ]
            if (
                not isinstance(block_identities, list)
                or any(
                    not isinstance(identity, list)
                    or len(identity) != 3
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                        for value in identity
                    )
                    for identity in block_identities
                )
            ):
                raise ValueError(
                    "hybrid prefix authority block identity is invalid"
                )
            rows[outer_rank] = dict(row)
        expected = tuple(range(self.model_runner.world_size))
        if tuple(sorted(rows)) != expected:
            raise ValueError(
                "hybrid prefix authority snapshot rank inventory mismatch"
            )
        reference = {
            name: rows[0][name]
            for name in fields if name != "rank"
        }
        if any(
            {
                name: rows[rank][name]
                for name in fields if name != "rank"
            } != reference
            for rank in expected[1:]
        ):
            raise ValueError(
                "hybrid prefix authority snapshot rank parity mismatch"
            )
        return tuple(rows[rank] for rank in expected)

    def _collect_qwen35_hybrid_prefix_cache_mutation(
        self,
        method_name,
        result_field,
        *args,
        timeout_s,
    ):
        local_result, worker_acks = (
            self.call_model_runner_acknowledged(
                method_name,
                *args,
                timeout_s=timeout_s,
            )
        )
        ranked = [(0, local_result)]
        ranked.extend(
            (ack.rank, ack.result)
            for ack in worker_acks
        )
        rows = {}
        for outer_rank, row in ranked:
            if (
                not isinstance(row, dict)
                or set(row) != {"rank", result_field}
                or row["rank"] != outer_rank
                or outer_rank in rows
            ):
                raise ValueError(
                    "hybrid prefix cache mutation rank mismatch"
                )
            value = row[result_field]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    "hybrid prefix cache mutation count is invalid"
                )
            rows[outer_rank] = dict(row)
        expected = tuple(range(self.model_runner.world_size))
        if tuple(sorted(rows)) != expected:
            raise ValueError(
                "hybrid prefix cache mutation rank inventory mismatch"
            )
        counts = {rows[rank][result_field] for rank in expected}
        if len(counts) != 1:
            raise ValueError(
                "hybrid prefix cache mutation count parity mismatch"
            )
        return tuple(rows[rank] for rank in expected)

    def clear_qwen35_hybrid_prefix_caches(self, *, timeout_s):
        return self._collect_qwen35_hybrid_prefix_cache_mutation(
            "clear_qwen35_hybrid_prefix_cache",
            "cleared_entries",
            timeout_s=timeout_s,
        )

    def invalidate_qwen35_hybrid_prefix_blocks(
        self,
        block_identities,
        *,
        timeout_s,
    ):
        block_identities = tuple(
            tuple(identity)
            for identity in block_identities
        )
        return self._collect_qwen35_hybrid_prefix_cache_mutation(
            "invalidate_qwen35_hybrid_prefix_blocks",
            "invalidated_entries",
            block_identities,
            timeout_s=timeout_s,
        )

    def memory_snapshots(self, *, timeout_s):
        local_result, worker_acks = (
            self.call_model_runner_acknowledged(
                "memory_snapshot",
                timeout_s=timeout_s,
            )
        )
        ranked = [(0, local_result)]
        ranked.extend(
            (ack.rank, ack.result)
            for ack in worker_acks
        )
        rows = {}
        for outer_rank, row in ranked:
            if not isinstance(row, dict) or outer_rank in rows:
                raise ValueError("memory snapshot rank mismatch")
            inner_rank = row.get("rank", outer_rank)
            if inner_rank != outer_rank:
                raise ValueError("memory snapshot rank mismatch")
            normalized = dict(row)
            normalized["rank"] = outer_rank
            rows[outer_rank] = normalized
        expected = tuple(range(self.model_runner.world_size))
        if tuple(sorted(rows)) != expected:
            raise ValueError("memory snapshot rank inventory mismatch")
        return tuple(rows[rank] for rank in expected)

    def reset_peak_memory_stats(self, *, timeout_s):
        local_result, worker_acks = (
            self.call_model_runner_acknowledged(
                "reset_peak_memory_stats",
                timeout_s=timeout_s,
            )
        )
        ranked = [(0, local_result)]
        ranked.extend(
            (ack.rank, ack.result)
            for ack in worker_acks
        )
        rows = {}
        for outer_rank, row in ranked:
            if not isinstance(row, dict) or outer_rank in rows:
                raise ValueError("peak reset rank mismatch")
            inner_rank = row.get("rank", outer_rank)
            if inner_rank != outer_rank:
                raise ValueError("peak reset rank mismatch")
            normalized = dict(row)
            normalized["rank"] = outer_rank
            rows[outer_rank] = normalized
        expected = tuple(range(self.model_runner.world_size))
        if tuple(sorted(rows)) != expected:
            raise ValueError("peak reset rank inventory mismatch")
        return tuple(rows[rank] for rank in expected)

    def _poison_model_runner_ack_collector(self, reason):
        collector = getattr(
            self,
            "model_runner_ack_collector",
            None,
        )
        if collector is not None:
            collector.poison(reason)

    def _validate_hybrid_prefix_restore_results(
        self,
        payload,
        operation,
        local_result,
        worker_acks,
        *,
        allowed_statuses,
    ):
        if (
            not isinstance(operation, str)
            or operation not in {
                "prepare",
                "validate",
                "commit",
                "rollback",
            }
        ):
            raise ValueError(
                f"unsupported hybrid prefix restore operation: {operation}"
            )
        ranked_rows = [(0, local_result)]
        ranked_rows.extend(
            (
                acknowledgement.rank,
                acknowledgement.result,
            )
            for acknowledgement in worker_acks
        )
        expected_ids = tuple(
            range(self.model_runner.world_size)
        )
        validated = {}
        try:
            for expected_participant_id, row in ranked_rows:
                if not isinstance(row, dict):
                    raise ValueError(
                        "hybrid prefix restore result must be a dict"
                    )
                required = {
                    "ticket_id",
                    "participant_id",
                    "operation",
                    "status",
                    "detail",
                }
                if set(row) != required:
                    raise ValueError(
                        "hybrid prefix restore result fields are invalid"
                    )
                if row["ticket_id"] != payload.ticket_id:
                    raise ValueError(
                        "hybrid prefix restore ticket id mismatch"
                    )
                participant_id = row["participant_id"]
                if (
                    isinstance(participant_id, bool)
                    or not isinstance(participant_id, int)
                    or participant_id not in expected_ids
                ):
                    raise ValueError(
                        "hybrid prefix restore participant id mismatch"
                    )
                if participant_id != expected_participant_id:
                    raise ValueError(
                        "hybrid prefix restore inner participant id "
                        "does not match outer rank"
                    )
                if participant_id in validated:
                    raise ValueError(
                        "duplicate hybrid prefix restore participant result"
                    )
                if row["operation"] != operation:
                    raise ValueError(
                        "hybrid prefix restore operation mismatch"
                    )
                if row["status"] not in allowed_statuses:
                    raise ValueError(
                        "hybrid prefix restore status is invalid"
                    )
                if not isinstance(row["detail"], str):
                    raise ValueError(
                        "hybrid prefix restore detail must be a string"
                    )
                validated[participant_id] = dict(row)
            if tuple(sorted(validated)) != expected_ids:
                raise ValueError(
                    "hybrid prefix restore participant results are incomplete"
                )
        except (ValueError, RuntimeError) as error:
            self._poison_model_runner_ack_collector(
                "invalid hybrid prefix restore nested result: "
                f"{error}"
            )
            raise
        return tuple(
            validated[participant_id]
            for participant_id in expected_ids
        )

    def _validate_hybrid_prefix_publication_payloads(
        self,
        payloads,
    ):
        world_size = self.model_runner.world_size
        if (
            not isinstance(payloads, tuple)
            or len(payloads) != world_size
            or any(
                not isinstance(
                    payload,
                    Qwen35HybridPrefixPublicationPayload,
                )
                for payload in payloads
            )
        ):
            raise ValueError(
                "publication payload matrix must contain one "
                "payload per ModelRunner rank"
            )
        payloads = tuple(sorted(
            payloads,
            key=lambda payload: payload.participant_id,
        ))
        expected_ids = tuple(range(world_size))
        if tuple(
            payload.participant_id for payload in payloads
        ) != expected_ids:
            raise ValueError(
                "publication payload participant ids must match ranks"
            )
        reference = payloads[0]
        for payload in payloads[1:]:
            for name in (
                "ticket_id",
                "request_id",
                "key",
                "token_ids",
                "block_identities",
            ):
                if getattr(payload, name) != getattr(reference, name):
                    raise ValueError(
                        "publication payload identity mismatch across "
                        f"ranks: {name}"
                    )
        if reference.key.tensor_parallel_size != world_size:
            raise ValueError(
                "publication payload tensor parallel size must match "
                "ModelRunner world size"
            )
        return payloads

    def _validate_hybrid_prefix_publication_results(
        self,
        payloads,
        operation,
        local_result,
        worker_acks,
        *,
        allowed_statuses,
    ):
        if operation not in {
            "prepare",
            "precommit",
            "finalize",
            "seal",
            "rollback",
        }:
            raise ValueError(
                "unsupported hybrid prefix publication operation: "
                f"{operation}"
            )
        payloads = self._validate_hybrid_prefix_publication_payloads(
            payloads
        )
        ranked_rows = [(0, local_result)]
        ranked_rows.extend(
            (
                acknowledgement.rank,
                acknowledgement.result,
            )
            for acknowledgement in worker_acks
        )
        expected_ids = tuple(range(self.model_runner.world_size))
        required = {
            "ticket_id",
            "participant_id",
            "operation",
            "status",
            "detail",
        }
        validated = {}
        try:
            for outer_rank, row in ranked_rows:
                if not isinstance(row, dict):
                    raise ValueError(
                        "hybrid prefix publication result must be a dict"
                    )
                if set(row) != required:
                    raise ValueError(
                        "hybrid prefix publication result fields are invalid"
                    )
                participant_id = row["participant_id"]
                if (
                    isinstance(outer_rank, bool)
                    or not isinstance(outer_rank, int)
                    or outer_rank not in expected_ids
                    or isinstance(participant_id, bool)
                    or not isinstance(participant_id, int)
                    or participant_id not in expected_ids
                ):
                    raise ValueError(
                        "hybrid prefix publication participant id mismatch"
                    )
                if participant_id != outer_rank:
                    raise ValueError(
                        "hybrid prefix publication inner participant id "
                        "does not match outer rank"
                    )
                if participant_id in validated:
                    raise ValueError(
                        "duplicate hybrid prefix publication participant "
                        "result"
                    )
                payload = payloads[participant_id]
                if row["ticket_id"] != payload.ticket_id:
                    raise ValueError(
                        "hybrid prefix publication ticket id mismatch"
                    )
                if row["operation"] != operation:
                    raise ValueError(
                        "hybrid prefix publication operation mismatch"
                    )
                if row["status"] not in allowed_statuses:
                    raise ValueError(
                        "hybrid prefix publication status is invalid"
                    )
                if not isinstance(row["detail"], str):
                    raise ValueError(
                        "hybrid prefix publication detail must be a string"
                    )
                validated[participant_id] = dict(row)
            if tuple(sorted(validated)) != expected_ids:
                raise ValueError(
                    "hybrid prefix publication participant results "
                    "are incomplete"
                )
        except (ValueError, RuntimeError) as error:
            self._poison_model_runner_ack_collector(
                "invalid hybrid prefix publication nested result: "
                f"{error}"
            )
            raise
        return tuple(
            validated[participant_id]
            for participant_id in expected_ids
        )

    def _call_model_runner_hybrid_prefix_publication_phase(
        self,
        operation,
        payloads,
        *,
        timeout_s,
    ):
        method_names = {
            "prepare": "prepare_hybrid_prefix_publication",
            "precommit": "precommit_hybrid_prefix_publication",
            "finalize": "finalize_hybrid_prefix_publication",
            "seal": "seal_hybrid_prefix_publication",
            "rollback": "rollback_hybrid_prefix_publication",
        }
        status_matrix = {
            "prepare": {"prepared", "rejected", "error"},
            "precommit": {"precommitted", "error"},
            "finalize": {"finalized", "error"},
            "seal": {"committed", "error"},
            "rollback": {"rolled_back", "error"},
        }
        if operation not in method_names:
            raise ValueError(
                "unsupported hybrid prefix publication operation: "
                f"{operation}"
            )
        payloads = self._validate_hybrid_prefix_publication_payloads(
            payloads
        )
        local_result, worker_acks = (
            self.call_model_runner_acknowledged(
                method_names[operation],
                payloads,
                timeout_s=timeout_s,
            )
        )
        return self._validate_hybrid_prefix_publication_results(
            payloads,
            operation,
            local_result,
            worker_acks,
            allowed_statuses=status_matrix[operation],
        )

    def prepare_model_runner_hybrid_prefix_publication(
        self,
        payloads,
        *,
        timeout_s,
    ):
        return self._call_model_runner_hybrid_prefix_publication_phase(
            "prepare",
            payloads,
            timeout_s=timeout_s,
        )

    def precommit_model_runner_hybrid_prefix_publication(
        self,
        payloads,
        *,
        timeout_s,
    ):
        return self._call_model_runner_hybrid_prefix_publication_phase(
            "precommit",
            payloads,
            timeout_s=timeout_s,
        )

    def finalize_model_runner_hybrid_prefix_publication(
        self,
        payloads,
        *,
        timeout_s,
    ):
        return self._call_model_runner_hybrid_prefix_publication_phase(
            "finalize",
            payloads,
            timeout_s=timeout_s,
        )

    def seal_model_runner_hybrid_prefix_publication(
        self,
        payloads,
        *,
        timeout_s,
    ):
        return self._call_model_runner_hybrid_prefix_publication_phase(
            "seal",
            payloads,
            timeout_s=timeout_s,
        )

    def rollback_model_runner_hybrid_prefix_publication(
        self,
        payloads,
        *,
        timeout_s,
    ):
        return self._call_model_runner_hybrid_prefix_publication_phase(
            "rollback",
            payloads,
            timeout_s=timeout_s,
        )

    def install_qwen35_hybrid_prefix_engine_publication_coordinator(
        self,
        coordinator,
    ):
        if not isinstance(
            coordinator,
            Qwen35HybridPrefixEnginePublicationCoordinator,
        ):
            raise ValueError(
                "coordinator must be a "
                "Qwen35HybridPrefixEnginePublicationCoordinator"
            )
        if coordinator.engine is not self:
            raise ValueError(
                "coordinator must target this LLMEngine"
            )
        current = getattr(
            self,
            "qwen35_hybrid_prefix_engine_publication_coordinator",
            None,
        )
        if current is not None:
            if current is coordinator:
                return
            raise RuntimeError(
                "Engine hybrid prefix publication coordinator "
                "is already installed"
            )
        self.qwen35_hybrid_prefix_engine_publication_coordinator = (
            coordinator
        )

    def publish_qwen35_hybrid_prefix(self, payloads):
        coordinator = getattr(
            self,
            "qwen35_hybrid_prefix_engine_publication_coordinator",
            None,
        )
        if coordinator is None:
            raise RuntimeError(
                "Engine hybrid prefix publication coordinator "
                "is not installed"
            )
        return coordinator.publish(payloads)

    def install_qwen35_hybrid_prefix_source_publisher(
        self,
        *,
        model_fingerprint,
        layout_fingerprint,
        dtype,
    ):
        configuration = (
            model_fingerprint,
            layout_fingerprint,
            dtype,
        )
        current = getattr(
            self,
            "qwen35_hybrid_prefix_source_publisher",
            None,
        )
        if current is not None:
            if getattr(
                self,
                "qwen35_hybrid_prefix_source_publisher_configuration",
                None,
            ) == configuration:
                return current
            raise RuntimeError(
                "Engine hybrid prefix source publisher "
                "is already installed"
            )
        publisher = Qwen35HybridPrefixSourcePublisher(
            self,
            model_fingerprint=model_fingerprint,
            layout_fingerprint=layout_fingerprint,
            dtype=dtype,
        )
        hook = publisher.publish
        self.scheduler.install_prefill_commit_hook(hook)
        self.qwen35_hybrid_prefix_source_publisher = publisher
        self.qwen35_hybrid_prefix_source_publisher_hook = hook
        self.qwen35_hybrid_prefix_source_publisher_configuration = (
            configuration
        )
        return publisher

    def install_configured_qwen35_hybrid_prefix_source_publisher(
        self,
    ):
        identity = getattr(
            self,
            "qwen35_hybrid_prefix_runtime_identity",
            None,
        )
        if not isinstance(
            identity,
            Qwen35HybridPrefixRuntimeIdentity,
        ):
            raise RuntimeError(
                "Qwen3.5 hybrid prefix runtime identity "
                "is not configured"
            )
        return self.install_qwen35_hybrid_prefix_source_publisher(
            model_fingerprint=identity.model_fingerprint,
            layout_fingerprint=identity.layout_fingerprint,
            dtype=identity.dtype,
        )

    def configure_qwen35_hybrid_prefix_publication_runtime(
        self,
        *,
        model_fingerprint,
        max_entries,
        max_bytes,
        representation="exact_restore",
        timeout_s,
    ):
        model_fingerprint = validate_qwen35_model_fingerprint(
            model_fingerprint
        )
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or max_entries <= 0
        ):
            raise ValueError("max_entries must be a positive integer")
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes <= 0
        ):
            raise ValueError("max_bytes must be a positive integer")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be positive")
        timeout_s = float(timeout_s)
        if representation not in {
            "exact_restore",
            "recurrent_int8_per_row",
        }:
            raise ValueError(
                "unsupported Qwen3.5 hybrid prefix "
                f"representation: {representation}"
            )
        configuration = (
            model_fingerprint,
            max_entries,
            max_bytes,
            representation,
            timeout_s,
        )
        current = getattr(
            self,
            "qwen35_hybrid_prefix_publication_runtime_configuration",
            None,
        )
        if current is not None and current != configuration:
            raise RuntimeError(
                "Qwen3.5 hybrid prefix publication runtime "
                "is already configured"
            )

        restore_configuration = getattr(
            self,
            "qwen35_hybrid_prefix_restore_configuration",
            None,
        )
        if (
            representation == "exact_restore"
            and isinstance(restore_configuration, tuple)
            and len(restore_configuration) == 3
        ):
            restore_configuration = (
                restore_configuration[0],
                restore_configuration[1],
                representation,
                restore_configuration[2],
            )
        expected_restore_configuration = (
            max_entries,
            max_bytes,
            representation,
            timeout_s,
        )
        if (
            restore_configuration is not None
            and restore_configuration
            != expected_restore_configuration
        ):
            raise RuntimeError(
                "Qwen3.5 hybrid prefix restore configuration "
                "conflicts with publication runtime"
            )
        restore_coordinator = getattr(
            self,
            "qwen35_hybrid_prefix_engine_restore_coordinator",
            None,
        )
        if (
            restore_configuration is None
            and restore_coordinator is not None
        ):
            raise RuntimeError(
                "Qwen3.5 hybrid prefix restore coordinator "
                "is installed before restore configuration"
            )
        if restore_configuration is not None and (
            not isinstance(
                restore_coordinator,
                Qwen35HybridPrefixEngineRestoreCoordinator,
            )
            or restore_coordinator.engine is not self
            or restore_coordinator.timeout_s != timeout_s
        ):
            raise RuntimeError(
                "Qwen3.5 hybrid prefix restore state is incomplete"
            )

        identity_configuration = getattr(
            self,
            "qwen35_hybrid_prefix_runtime_identity_configuration",
            None,
        )
        expected_identity_configuration = (
            model_fingerprint,
            timeout_s,
        )
        identity = getattr(
            self,
            "qwen35_hybrid_prefix_runtime_identity",
            None,
        )
        if (
            identity_configuration is None
            and identity is not None
        ):
            raise RuntimeError(
                "Qwen3.5 hybrid prefix runtime identity state "
                "is incomplete"
            )
        if identity_configuration is not None:
            if restore_configuration is None:
                raise RuntimeError(
                    "Qwen3.5 hybrid prefix runtime identity "
                    "is installed before restore configuration"
                )
            if identity_configuration != expected_identity_configuration:
                raise RuntimeError(
                    "Qwen3.5 hybrid prefix runtime identity "
                    "conflicts with publication runtime"
                )
            if not isinstance(
                identity,
                Qwen35HybridPrefixRuntimeIdentity,
            ):
                raise RuntimeError(
                    "Qwen3.5 hybrid prefix runtime identity "
                    "state is incomplete"
                )

        coordinator = getattr(
            self,
            "qwen35_hybrid_prefix_engine_publication_coordinator",
            None,
        )
        if coordinator is not None:
            if identity_configuration is None:
                raise RuntimeError(
                    "Qwen3.5 hybrid prefix publication coordinator "
                    "is installed before runtime identity"
                )
            if (
                not isinstance(
                    coordinator,
                    Qwen35HybridPrefixEnginePublicationCoordinator,
                )
                or coordinator.engine is not self
                or coordinator.timeout_s != timeout_s
            ):
                raise RuntimeError(
                    "Qwen3.5 hybrid prefix publication coordinator "
                    "conflicts with publication runtime"
                )

        publisher = getattr(
            self,
            "qwen35_hybrid_prefix_source_publisher",
            None,
        )
        publisher_configuration = getattr(
            self,
            "qwen35_hybrid_prefix_source_publisher_configuration",
            None,
        )
        if (publisher is None) != (publisher_configuration is None):
            raise RuntimeError(
                "Qwen3.5 hybrid prefix source publisher state "
                "is incomplete"
            )
        if publisher is not None:
            if coordinator is None or identity_configuration is None:
                raise RuntimeError(
                    "Qwen3.5 hybrid prefix source publisher "
                    "is installed before publication dependencies"
                )
            expected_publisher_configuration = (
                identity.model_fingerprint,
                identity.layout_fingerprint,
                identity.dtype,
            )
            if (
                identity.model_fingerprint != model_fingerprint
                or publisher_configuration
                != expected_publisher_configuration
            ):
                raise RuntimeError(
                    "Qwen3.5 hybrid prefix source publisher "
                    "conflicts with publication runtime"
                )

        aggregate_publisher = getattr(
            self,
            "qwen35_hybrid_prefix_publication_runtime_publisher",
            None,
        )
        if current is None and aggregate_publisher is not None:
            raise RuntimeError(
                "Qwen3.5 hybrid prefix publication runtime "
                "completion state is incomplete"
            )
        if current is not None:
            if (
                aggregate_publisher is None
                or aggregate_publisher is not publisher
            ):
                raise RuntimeError(
                    "Qwen3.5 hybrid prefix publication runtime "
                    "completion state is incomplete"
                )
            return aggregate_publisher

        restore_kwargs = {
            "max_entries": max_entries,
            "max_bytes": max_bytes,
            "timeout_s": timeout_s,
        }
        if representation != "exact_restore":
            restore_kwargs["representation"] = representation
        self.configure_qwen35_hybrid_prefix_restore(
            **restore_kwargs,
        )
        self.configure_qwen35_hybrid_prefix_runtime_identity(
            model_fingerprint=model_fingerprint,
            timeout_s=timeout_s,
        )
        if coordinator is None:
            coordinator = (
                Qwen35HybridPrefixEnginePublicationCoordinator(
                    self,
                    timeout_s=timeout_s,
                )
            )
            self.install_qwen35_hybrid_prefix_engine_publication_coordinator(
                coordinator
            )
        publisher = (
            self
            .install_configured_qwen35_hybrid_prefix_source_publisher()
        )
        self.qwen35_hybrid_prefix_publication_runtime_publisher = (
            publisher
        )
        self.qwen35_hybrid_prefix_publication_runtime_configuration = (
            configuration
        )
        return publisher

    def configure_qwen35_hybrid_prefix_runtime_identity(
        self,
        *,
        model_fingerprint,
        timeout_s,
    ):
        model_fingerprint = validate_qwen35_model_fingerprint(
            model_fingerprint
        )
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be positive")
        configuration = (
            model_fingerprint,
            float(timeout_s),
        )
        current = getattr(
            self,
            "qwen35_hybrid_prefix_runtime_identity_configuration",
            None,
        )
        if current is not None:
            if current == configuration:
                return self.qwen35_hybrid_prefix_runtime_identity
            raise RuntimeError(
                "Qwen3.5 hybrid prefix runtime identity "
                "is already configured"
            )
        local_result, worker_acks = (
            self.call_model_runner_acknowledged(
                "bind_qwen35_hybrid_prefix_runtime_identity",
                model_fingerprint,
                timeout_s=timeout_s,
            )
        )
        ranked_rows = [(0, local_result)]
        ranked_rows.extend(
            (ack.rank, ack.result)
            for ack in worker_acks
        )
        expected_ranks = tuple(
            range(self.model_runner.world_size)
        )
        required = {
            "participant_id",
            "model_fingerprint",
            "layout_fingerprint",
            "dtype",
        }
        rows = {}
        dtype_by_name = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        try:
            for outer_rank, row in ranked_rows:
                if not isinstance(row, dict) or set(row) != required:
                    raise ValueError(
                        "hybrid prefix runtime identity fields "
                        "are invalid"
                    )
                participant_id = row["participant_id"]
                if (
                    isinstance(outer_rank, bool)
                    or not isinstance(outer_rank, int)
                    or outer_rank not in expected_ranks
                    or participant_id != outer_rank
                ):
                    raise ValueError(
                        "hybrid prefix runtime identity rank "
                        "is invalid"
                    )
                validate_qwen35_model_fingerprint(
                    row["model_fingerprint"]
                )
                if (
                    not isinstance(row["layout_fingerprint"], str)
                    or not row["layout_fingerprint"]
                ):
                    raise ValueError(
                        "hybrid prefix runtime identity layout "
                        "is invalid"
                    )
                if row["dtype"] not in dtype_by_name:
                    raise ValueError(
                        "hybrid prefix runtime identity dtype "
                        "is invalid"
                    )
                if outer_rank in rows:
                    raise ValueError(
                        "duplicate hybrid prefix runtime identity rank"
                    )
                rows[outer_rank] = dict(row)
            if tuple(sorted(rows)) != expected_ranks:
                raise ValueError(
                    "hybrid prefix runtime identity ranks "
                    "are incomplete"
                )
            reference = rows[0]
            for rank in expected_ranks:
                row = rows[rank]
                for name in (
                    "model_fingerprint",
                    "layout_fingerprint",
                    "dtype",
                ):
                    if row[name] != reference[name]:
                        raise ValueError(
                            "hybrid prefix runtime identity "
                            f"mismatch: {name}"
                        )
            if reference["model_fingerprint"] != model_fingerprint:
                raise ValueError(
                    "hybrid prefix runtime identity model "
                    "does not match request"
                )
            identity = Qwen35HybridPrefixRuntimeIdentity(
                model_fingerprint=reference["model_fingerprint"],
                layout_fingerprint=reference["layout_fingerprint"],
                dtype=dtype_by_name[reference["dtype"]],
            )
        except (ValueError, RuntimeError) as error:
            self._poison_model_runner_ack_collector(
                "invalid hybrid prefix runtime identity: "
                f"{error}"
            )
            raise
        self.qwen35_hybrid_prefix_runtime_identity = identity
        self.qwen35_hybrid_prefix_runtime_identity_configuration = (
            configuration
        )
        return identity

    def bind_qwen35_loaded_checkpoint_candidates(
        self,
        *,
        timeout_s,
    ):
        timeout_value = (
            float(timeout_s)
            if (
                not isinstance(timeout_s, bool)
                and isinstance(timeout_s, (int, float))
            )
            else None
        )
        if (
            timeout_value is None
            or timeout_value != timeout_value
            or timeout_value in (float("inf"), float("-inf"))
            or timeout_value <= 0
        ):
            raise ValueError("timeout_s must be positive and finite")
        timeout_s = timeout_value
        current = getattr(
            self,
            "qwen35_loaded_checkpoint_candidate_binding_configuration",
            None,
        )
        rows = getattr(
            self,
            "qwen35_loaded_checkpoint_candidate_binding_rows",
            None,
        )
        if current is not None:
            if (
                not isinstance(current, tuple)
                or len(current) != 4
                or not all(
                    isinstance(value, str) and value
                    for value in current[:3]
                )
            ):
                raise RuntimeError(
                    "Qwen3.5 loaded checkpoint candidate binding "
                    "completion state is incomplete"
                )
            try:
                validate_qwen35_model_fingerprint(current[0])
            except ValueError as error:
                raise RuntimeError(
                    "Qwen3.5 loaded checkpoint candidate binding "
                    "completion state is incomplete"
                ) from error
            if current[2] not in {
                "float16",
                "bfloat16",
                "float32",
            }:
                raise RuntimeError(
                    "Qwen3.5 loaded checkpoint candidate binding "
                    "completion state is incomplete"
                )
            if current[3] != timeout_s:
                raise RuntimeError(
                    "Qwen3.5 loaded checkpoint candidates "
                    "are already configured"
                )
            if (
                not isinstance(rows, tuple)
                or len(rows) != self.model_runner.world_size
            ):
                raise RuntimeError(
                    "Qwen3.5 loaded checkpoint candidate binding "
                    "completion state is incomplete"
                )
            required = {
                "participant_id",
                "operation",
                "status",
                "model_fingerprint",
                "layout_fingerprint",
                "dtype",
                "detail",
            }
            for rank, row in enumerate(rows):
                if (
                    not isinstance(row, dict)
                    or set(row) != required
                    or row["participant_id"] != rank
                    or row["operation"]
                    != "bind_loaded_checkpoint_candidate"
                    or row["status"] != "bound"
                    or row["model_fingerprint"] != current[0]
                    or row["layout_fingerprint"] != current[1]
                    or row["dtype"] != current[2]
                    or row["detail"] != ""
                ):
                    raise RuntimeError(
                        "Qwen3.5 loaded checkpoint candidate binding "
                        "completion state is incomplete"
                    )
            return rows
        if rows is not None:
            raise RuntimeError(
                "Qwen3.5 loaded checkpoint candidate binding "
                "completion state is incomplete"
            )

        local_result, worker_acks = (
            self.call_model_runner_acknowledged(
                "bind_published_qwen35_loaded_checkpoint_candidate",
                timeout_s=timeout_s,
            )
        )
        ranked_rows = [(0, local_result)]
        ranked_rows.extend(
            (ack.rank, ack.result)
            for ack in worker_acks
        )
        expected_ranks = tuple(
            range(self.model_runner.world_size)
        )
        if len(ranked_rows) != len(expected_ranks):
            raise RuntimeError(
                "loaded checkpoint candidate binding result count "
                "is incomplete"
            )
        required = {
            "participant_id",
            "operation",
            "status",
            "model_fingerprint",
            "layout_fingerprint",
            "dtype",
            "detail",
        }
        validated = {}
        for outer_rank, row in ranked_rows:
            if not isinstance(row, dict) or set(row) != required:
                raise RuntimeError(
                    "loaded checkpoint candidate binding result "
                    "fields are invalid"
                )
            participant_id = row["participant_id"]
            if (
                isinstance(outer_rank, bool)
                or not isinstance(outer_rank, int)
                or outer_rank not in expected_ranks
                or participant_id != outer_rank
                or participant_id in validated
            ):
                raise RuntimeError(
                    "loaded checkpoint candidate binding participant "
                    "is invalid"
                )
            if (
                row["operation"]
                != "bind_loaded_checkpoint_candidate"
            ):
                raise RuntimeError(
                    "loaded checkpoint candidate binding operation "
                    "is invalid"
                )
            if row["status"] not in {"bound", "error"}:
                raise RuntimeError(
                    "loaded checkpoint candidate binding status "
                    "is invalid"
                )
            if not isinstance(row["detail"], str):
                raise RuntimeError(
                    "loaded checkpoint candidate binding detail "
                    "is invalid"
                )
            for name in (
                "model_fingerprint",
                "layout_fingerprint",
                "dtype",
            ):
                if not isinstance(row[name], str):
                    raise RuntimeError(
                        "loaded checkpoint candidate binding "
                        f"{name} is invalid"
                    )
            if row["status"] == "error":
                if any(
                    row[name]
                    for name in (
                        "model_fingerprint",
                        "layout_fingerprint",
                        "dtype",
                    )
                ):
                    raise RuntimeError(
                        "loaded checkpoint candidate binding error "
                        "identity must be empty"
                    )
            else:
                if (
                    not row["model_fingerprint"]
                    or not row["layout_fingerprint"]
                    or not row["dtype"]
                    or row["detail"]
                ):
                    raise RuntimeError(
                        "loaded checkpoint candidate bound identity "
                        "is invalid"
                    )
                try:
                    validate_qwen35_model_fingerprint(
                        row["model_fingerprint"]
                    )
                except ValueError as error:
                    raise RuntimeError(
                        "loaded checkpoint candidate binding "
                        "model_fingerprint is invalid"
                    ) from error
                if row["dtype"] not in {
                    "float16",
                    "bfloat16",
                    "float32",
                }:
                    raise RuntimeError(
                        "loaded checkpoint candidate binding "
                        "dtype is invalid"
                    )
            validated[participant_id] = dict(row)
        if tuple(sorted(validated)) != expected_ranks:
            raise RuntimeError(
                "loaded checkpoint candidate binding ranks "
                "are incomplete"
            )
        rows = tuple(
            validated[rank]
            for rank in expected_ranks
        )
        failed = next(
            (row for row in rows if row["status"] == "error"),
            None,
        )
        if failed is not None:
            raise RuntimeError(
                "loaded checkpoint candidate binding failed: "
                f"rank={failed['participant_id']}, "
                f"detail={failed['detail']}"
            )
        reference = rows[0]
        for rank in expected_ranks:
            row = rows[rank]
            for name in (
                "model_fingerprint",
                "layout_fingerprint",
                "dtype",
            ):
                if row[name] != reference[name]:
                    raise RuntimeError(
                        "loaded checkpoint candidate binding "
                        f"mismatch: {name}"
                    )
        configuration = (
            reference["model_fingerprint"],
            reference["layout_fingerprint"],
            reference["dtype"],
            timeout_s,
        )
        self.qwen35_loaded_checkpoint_candidate_binding_rows = rows
        self.qwen35_loaded_checkpoint_candidate_binding_configuration = (
            configuration
        )
        return rows

    def prepare_model_runner_hybrid_prefix_restore(
        self,
        payload,
        *,
        timeout_s,
    ):
        local_result, worker_acks = (
            self.call_model_runner_acknowledged(
                "prepare_hybrid_prefix_restore",
                payload,
                timeout_s=timeout_s,
            )
        )
        return self._validate_hybrid_prefix_restore_results(
            payload,
            "prepare",
            local_result,
            worker_acks,
            allowed_statuses={
                "prepared",
                "miss",
                "error",
            },
        )

    def _call_model_runner_hybrid_prefix_restore_operation(
        self,
        operation,
        payload,
        *,
        timeout_s,
    ):
        if operation not in {
            "validate",
            "commit",
            "rollback",
        }:
            raise ValueError(
                f"unsupported hybrid prefix restore operation: {operation}"
            )
        local_result, worker_acks = (
            self.call_model_runner_acknowledged(
                f"{operation}_hybrid_prefix_restore",
                payload,
                timeout_s=timeout_s,
            )
        )
        rows = self._validate_hybrid_prefix_restore_results(
            payload,
            operation,
            local_result,
            worker_acks,
            allowed_statuses={"ok", "error"},
        )
        for row in rows:
            if row["status"] != "ok":
                error = RuntimeError(
                    "hybrid prefix restore operation failed: "
                    f"participant={row['participant_id']}, "
                    f"operation={operation}, detail={row['detail']}"
                )
                self._poison_model_runner_ack_collector(str(error))
                raise error
        return rows

    def validate_model_runner_hybrid_prefix_restore(
        self,
        payload,
        *,
        timeout_s,
    ):
        return self._call_model_runner_hybrid_prefix_restore_operation(
            "validate",
            payload,
            timeout_s=timeout_s,
        )

    def commit_model_runner_hybrid_prefix_restore(
        self,
        payload,
        *,
        timeout_s,
    ):
        return self._call_model_runner_hybrid_prefix_restore_operation(
            "commit",
            payload,
            timeout_s=timeout_s,
        )

    def rollback_model_runner_hybrid_prefix_restore(
        self,
        payload,
        *,
        timeout_s,
    ):
        return self._call_model_runner_hybrid_prefix_restore_operation(
            "rollback",
            payload,
            timeout_s=timeout_s,
        )

    def install_qwen35_hybrid_prefix_engine_restore_coordinator(
        self,
        coordinator,
    ):
        if not isinstance(
            coordinator,
            Qwen35HybridPrefixEngineRestoreCoordinator,
        ):
            raise ValueError(
                "coordinator must be a "
                "Qwen35HybridPrefixEngineRestoreCoordinator"
            )
        if coordinator.engine is not self:
            raise ValueError(
                "coordinator must target this LLMEngine"
            )
        if coordinator.block_manager is not (
            self.scheduler.block_manager
        ):
            raise ValueError(
                "coordinator must use the Scheduler BlockManager"
            )
        if coordinator.state_allocator is not (
            self.scheduler.hybrid_state_allocator
        ):
            raise ValueError(
                "coordinator must use the Scheduler state allocator"
            )
        current = (
            self.qwen35_hybrid_prefix_engine_restore_coordinator
        )
        if current is not None and current is not coordinator:
            raise RuntimeError(
                "Engine hybrid prefix restore coordinator "
                "is already installed"
            )
        self.qwen35_hybrid_prefix_engine_restore_coordinator = (
            coordinator
        )

    def acquire_qwen35_hybrid_prefix(
        self,
        sequence,
        key,
        token_ids,
    ):
        coordinator = (
            self.qwen35_hybrid_prefix_engine_restore_coordinator
        )
        if coordinator is None:
            raise RuntimeError(
                "Engine hybrid prefix restore coordinator "
                "is not installed"
            )
        return coordinator.acquire(sequence, key, token_ids)

    def configure_qwen35_hybrid_prefix_restore(
        self,
        *,
        max_entries,
        max_bytes,
        representation="exact_restore",
        timeout_s,
    ):
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or max_entries <= 0
        ):
            raise ValueError("max_entries must be a positive integer")
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes <= 0
        ):
            raise ValueError("max_bytes must be a positive integer")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be positive")
        representation_identities = {
            "exact_restore": (
                "qwen35_hybrid_prefix_exact_v1",
                None,
            ),
            "recurrent_int8_per_row": (
                "qwen35_hybrid_prefix_recurrent_int8_v1",
                "qwen35_recurrent_symmetric_int8_per_row_v1",
            ),
        }
        if representation not in representation_identities:
            raise ValueError(
                "unsupported Qwen3.5 hybrid prefix "
                f"representation: {representation}"
            )
        representation_version, codec = (
            representation_identities[representation]
        )
        configuration = (
            max_entries,
            max_bytes,
            representation,
            float(timeout_s),
        )
        current = getattr(
            self,
            "qwen35_hybrid_prefix_restore_configuration",
            None,
        )
        if current is not None:
            if current == configuration:
                return (
                    self
                    .qwen35_hybrid_prefix_engine_restore_coordinator
                )
            raise RuntimeError(
                "Qwen3.5 hybrid prefix restore is already configured"
            )
        allocator = getattr(
            self.scheduler,
            "hybrid_state_allocator",
            None,
        )
        if allocator is None:
            raise RuntimeError(
                "Scheduler hybrid state allocator is not installed"
            )
        local_result, worker_acks = (
            self.call_model_runner_acknowledged(
                "configure_qwen35_hybrid_prefix_restore_owner",
                max_entries,
                max_bytes,
                representation,
                timeout_s=timeout_s,
            )
        )
        ranked_rows = [(0, local_result)]
        ranked_rows.extend(
            (ack.rank, ack.result)
            for ack in worker_acks
        )
        expected_ranks = tuple(
            range(self.model_runner.world_size)
        )
        rows = {}
        required = {
            "participant_id",
            "capacity",
            "layout_fingerprint",
            "bytes_per_slot",
            "max_entries",
            "max_bytes",
            "representation",
            "representation_version",
            "codec",
        }
        try:
            for outer_rank, row in ranked_rows:
                if not isinstance(row, dict):
                    raise ValueError(
                        "hybrid prefix restore owner identity "
                        "fields are invalid"
                    )
                row = dict(row)
                missing = required - set(row)
                if missing:
                    legacy_identity_fields = {
                        "representation",
                        "representation_version",
                        "codec",
                    }
                    if (
                        representation != "exact_restore"
                        or missing != legacy_identity_fields
                        or set(row) != required - legacy_identity_fields
                    ):
                        raise ValueError(
                            "hybrid prefix restore owner identity "
                            "fields are invalid"
                        )
                    row.update({
                        "representation": representation,
                        "representation_version": (
                            representation_version
                        ),
                        "codec": codec,
                    })
                elif set(row) != required:
                    raise ValueError(
                        "hybrid prefix restore owner identity "
                        "fields are invalid"
                    )
                if row["participant_id"] != outer_rank:
                    raise ValueError(
                        "hybrid prefix restore owner inner rank "
                        "does not match outer rank"
                    )
                for name in (
                    "participant_id",
                    "capacity",
                    "bytes_per_slot",
                    "max_entries",
                    "max_bytes",
                ):
                    value = row[name]
                    minimum = 0 if name == "participant_id" else 1
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < minimum
                    ):
                        raise ValueError(
                            "hybrid prefix restore owner identity "
                            f"{name} is invalid"
                        )
                if (
                    not isinstance(row["layout_fingerprint"], str)
                    or not row["layout_fingerprint"]
                ):
                    raise ValueError(
                        "hybrid prefix restore owner layout "
                        "fingerprint is invalid"
                    )
                if (
                    row["representation"]
                    != representation
                    or row["representation_version"]
                    != representation_version
                    or row["codec"] != codec
                ):
                    raise ValueError(
                        "hybrid prefix restore owner identity "
                        "mismatch: representation"
                    )
                if outer_rank in rows:
                    raise ValueError(
                        "duplicate hybrid prefix restore owner rank"
                    )
                rows[outer_rank] = row
            if tuple(sorted(rows)) != expected_ranks:
                raise ValueError(
                    "hybrid prefix restore owner ranks are incomplete"
                )
            reference = rows[0]
            for rank in expected_ranks:
                row = rows[rank]
                for name in (
                    "capacity",
                    "layout_fingerprint",
                    "bytes_per_slot",
                    "max_entries",
                    "max_bytes",
                    "representation",
                    "representation_version",
                    "codec",
                ):
                    if row[name] != reference[name]:
                        raise ValueError(
                            "hybrid prefix restore owner identity "
                            f"mismatch: {name}"
                        )
            if reference["capacity"] != allocator.capacity:
                raise RuntimeError(
                    "hybrid prefix restore owner capacity does not "
                    "match Scheduler allocator capacity"
                )
            if (
                reference["max_entries"] != max_entries
                or reference["max_bytes"] != max_bytes
            ):
                raise RuntimeError(
                    "hybrid prefix restore owner cache limits "
                    "do not match request"
                )
        except (ValueError, RuntimeError) as error:
            self._poison_model_runner_ack_collector(
                "invalid hybrid prefix restore owner identity: "
                f"{error}"
            )
            raise
        coordinator = Qwen35HybridPrefixEngineRestoreCoordinator(
            self,
            self.scheduler.block_manager,
            allocator,
            timeout_s=timeout_s,
        )
        self.install_qwen35_hybrid_prefix_engine_restore_coordinator(
            coordinator
        )
        self.qwen35_hybrid_prefix_restore_configuration = (
            configuration
        )
        return coordinator

    def add_request(
        self, 
        prompt: str | list[int], 
        sampling_params: SamplingParams
    ):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        runtime = getattr(
            self,
            "speculative_runtime",
            None,
        )
        lifecycle = (
            None
            if runtime is None
            else runtime.lifecycle
        )
        registered = False
        if lifecycle is not None:
            lifecycle.register_sequence(
                seq.seq_id,
                tuple(seq.token_ids),
            )
            registered = True
        try:
            _try_qwen35_hybrid_prefix_restore(self, seq)
            self.scheduler.add(seq)           #直接加到waiting
        except BaseException:
            if registered:
                lifecycle.release_sequence(seq.seq_id)
            raise

    def flush_pending_hybrid_state_releases(self, *, timeout_s):
        drain_releases = getattr(
            self.scheduler,
            "drain_hybrid_state_release_events",
            None,
        )
        released_leases = (
            drain_releases()
            if drain_releases is not None
            else ()
        )
        if not released_leases:
            return ()
        try:
            self.call_model_runner_acknowledged(
                "release_hybrid_state",
                released_leases,
                timeout_s=timeout_s,
            )
        except BaseException:
            restore_releases = getattr(
                self.scheduler,
                "restore_hybrid_state_release_events",
                None,
            )
            if restore_releases is not None:
                restore_releases(released_leases)
            raise
        return released_leases

    def clear_reusable_prefix_cache(self):
        return self.scheduler.block_manager.clear_reusable_cache()

    def step(self):     #decode阶段：每次step生成新的token加到seq后面
        queue_before = self.scheduler.observation_snapshot()
        decision_now_ns = self._clock_ns()
        scheduled = self.scheduler.schedule(decision_now_ns)
        if len(scheduled) == 4:
            seqs, is_prefill, do_sample, batch_kind = scheduled
        else:
            seqs, is_prefill, do_sample = scheduled
            batch_kind = None
        partition = build_engine_speculative_partition(
            self.scheduler.last_speculative_selection,
            tuple(seqs),
            expected_schedule_generation=(
                self.scheduler.schedule_generation
            ),
        )
        identity_builder = getattr(
            self,
            "_kv_offload_identity_rows",
            None,
        )

        def kv_block_identity_rows_for(rows):
            if identity_builder is None:
                return ()
            return identity_builder(tuple(rows))

        runtime = getattr(self, "speculative_runtime", None)
        if partition.selected_sequences and runtime is None:
            raise RuntimeError(
                "speculative rows selected before engine runtime "
                "installation"
            )
        if (
            partition.selected_sequences
            and self.speculative_runtime_poisoned
        ):
            raise RuntimeError(
                "speculative runtime is poisoned: "
                f"{self.speculative_runtime_poison_reason}"
            )
        self.last_batch_kind = batch_kind
        self.last_scheduled_seqs = seqs
        completion_lengths_before = {
            seq.seq_id: len(seq.completion_token_ids)
            for seq in seqs
        }
        scheduled_rows = [{
            "seq_id": seq.seq_id,
            "is_decode": bool(
                getattr(seq, "step_is_decode", False)
                if batch_kind == "mixed"
                else not is_prefill
            ),
            "do_sample": bool(
                getattr(seq, "step_do_sample", do_sample)
                if batch_kind == "mixed"
                else do_sample
            ),
            "prefill_chunk_start": seq.prefill_chunk_start,
            "prefill_chunk_end": seq.prefill_chunk_end,
            "prefill_chunk_final": bool(seq.prefill_chunk_final),
        } for seq in seqs]
        if batch_kind == "mixed":
            prefill_tokens = sum(
                row["prefill_chunk_end"]
                - row["prefill_chunk_start"]
                for row in scheduled_rows
                if not row["is_decode"]
            )
            decode_tokens = sum(
                1
                for row in scheduled_rows
                if row["is_decode"]
            )
            num_tokens = prefill_tokens + decode_tokens
        elif is_prefill:
            num_tokens = sum(
                row["prefill_chunk_end"]
                - row["prefill_chunk_start"]
                for row in scheduled_rows
            )
        else:
            num_tokens = -len(seqs)
        speculative_output_token_counts = {}
        speculative_accepted_draft_token_counts = {}
        speculative_proposal_token_counts = {}
        speculative_proposal_token_ids_by_seq = {}
        speculative_accepted_draft_token_ids_by_seq = {}
        speculative_proposal_row_count = 0
        speculative_first_target_callback_count = 0
        speculative_fixed_q_group_count = 0
        speculative_runtime_timing_ms = {}
        if partition.selected_sequences:
            model_runner_config = getattr(
                self.model_runner,
                "config",
                None,
            )
            residency_enabled = bool(
                getattr(
                    model_runner_config,
                    "kv_offload_mvp0",
                    False,
                )
            )
            residency_ticket_id = None
            residency_state = None
            residency_prepare_rows = ()
            residency_precommit_rows = ()
            residency_publication_committed = False

            def rollback_residency():
                nonlocal residency_state
                rejected_identities = tuple(
                    identity
                    for row in residency_prepare_rows
                    for identity in (
                        row.reserved_block_identities
                    )
                )
                self._call_speculative_residency_phase(
                    "rollback_speculative_residency_batch",
                    residency_ticket_id,
                    expected_operation="rollback",
                    expected_status="rolled_back",
                    expected_sequence_ids=tuple(
                        row.sequence_id
                        for row in residency_prepare_rows
                    ),
                    expected_committed_block_identities=(),
                    expected_rejected_block_identities=(
                        rejected_identities
                    ),
                    timeout_s=60.0,
                )
                residency_state = "rolled_back"

            ordinary_token_ids = ()
            if partition.suppressed_sequences:
                suppressed_identity_rows = (
                    kv_block_identity_rows_for(
                        partition.suppressed_sequences
                    )
                )
                drain_releases = getattr(
                    self.scheduler,
                    "drain_hybrid_state_release_events",
                    None,
                )
                released_leases = (
                    drain_releases()
                    if drain_releases is not None
                    else ()
                )
                try:
                    if released_leases or suppressed_identity_rows:
                        ordinary_token_ids = self.model_runner.call(
                            "run",
                            partition.suppressed_sequences,
                            is_prefill,
                            do_sample,
                            batch_kind,
                            released_leases,
                            suppressed_identity_rows,
                        )
                    else:
                        ordinary_token_ids = self.model_runner.call(
                            "run",
                            partition.suppressed_sequences,
                            is_prefill,
                            do_sample,
                            batch_kind,
                        )
                except BaseException:
                    restore_releases = getattr(
                        self.scheduler,
                        "restore_hybrid_state_release_events",
                        None,
                    )
                    if restore_releases is not None:
                        restore_releases(released_leases)
                    raise
            else:
                self.flush_pending_hybrid_state_releases(
                    timeout_s=60.0,
                )
            prepared_runtime = None
            try:
                def run_tail_batch(items):
                    nonlocal residency_ticket_id
                    nonlocal residency_state
                    nonlocal residency_prepare_rows
                    if not residency_enabled:
                        return run_model_runner_tail_batch(
                            self.model_runner,
                            items,
                        )
                    if residency_ticket_id is not None:
                        raise RuntimeError(
                            "speculative residency tail callback "
                            "must run exactly once"
                        )
                    residency_prepare_rows = (
                        build_speculative_residency_prepare_rows(
                            items
                        )
                    )
                    residency_ticket_id = next(
                        self._speculative_residency_ticket_ids
                    )
                    sequence_ids = tuple(
                        row.sequence_id
                        for row in residency_prepare_rows
                    )
                    self._call_speculative_residency_phase(
                        "prepare_speculative_residency_batch",
                        residency_ticket_id,
                        residency_prepare_rows,
                        expected_operation="prepare",
                        expected_status="prepared",
                        expected_sequence_ids=sequence_ids,
                        expected_committed_block_identities=(),
                        expected_rejected_block_identities=(),
                        timeout_s=60.0,
                    )
                    residency_state = "prepared"
                    try:
                        return run_model_runner_tail_batch(
                            self.model_runner,
                            items,
                            residency_ticket_id,
                        )
                    except BaseException as error:
                        try:
                            rollback_residency()
                        except BaseException as rollback_error:
                            residency_state = "rollback_failed"
                            self.speculative_runtime_poisoned = True
                            self.speculative_runtime_poison_reason = (
                                "speculative residency rollback "
                                f"failed: {rollback_error}"
                            )
                            raise rollback_error from error
                        raise

                run_first_targets_and_proposals = (
                    build_model_runner_proposal_provider(
                        self.model_runner,
                        runtime,
                        kv_block_identity_rows_for,
                    )
                )
                side_state_callbacks = (
                    build_model_runner_side_state_callbacks(
                        self.model_runner,
                        dispatch=(
                            lambda method_name, *args: (
                                self
                                ._call_speculative_side_state_phase(
                                    method_name,
                                    *args,
                                )
                            )
                        ),
                    )
                )
                prepared_runtime = (
                    prepare_native_speculative_batch(
                        block_manager=(
                            self.scheduler.block_manager
                        ),
                        seqs=partition.selected_sequences,
                        eos_token=self.scheduler.eos,
                        run_first_targets_and_proposals=(
                            run_first_targets_and_proposals
                        ),
                        run_tail_batch=run_tail_batch,
                        side_state_callbacks=(
                            side_state_callbacks
                        ),
                    )
                )
                commit_rows = (
                    build_engine_prepared_speculative_commit_rows(
                        prepared_runtime,
                        partition.selected_sequences,
                        eos_token=self.scheduler.eos,
                    )
                )
                commit_row_by_sequence_id = {
                    row.sequence_id: row
                    for row in commit_rows
                }
                ordinary_token_iter = iter(
                    ordinary_token_ids or ()
                )
                scheduler_rows = []
                for seq in seqs:
                    commit_row = (
                        commit_row_by_sequence_id.get(
                            seq.seq_id
                        )
                    )
                    if commit_row is not None:
                        scheduler_rows.append(
                            ScheduledOutputRow(
                                sequence_id=seq.seq_id,
                                output_tokens=(
                                    commit_row.output_tokens
                                ),
                                speculative=True,
                                accepted_draft_tokens=(
                                    commit_row
                                    .accepted_draft_tokens
                                ),
                            )
                        )
                        continue
                    row_is_decode = (
                        bool(
                            getattr(
                                seq,
                                "step_is_decode",
                                False,
                            )
                        )
                        if batch_kind == "mixed"
                        else not is_prefill
                    )
                    row_do_sample = (
                        bool(
                            getattr(
                                seq,
                                "step_do_sample",
                                do_sample,
                            )
                        )
                        if batch_kind == "mixed"
                        else do_sample
                    )
                    output_tokens = (
                        (next(ordinary_token_iter),)
                        if row_is_decode or row_do_sample
                        else ()
                    )
                    scheduler_rows.append(
                        ScheduledOutputRow(
                            sequence_id=seq.seq_id,
                            output_tokens=output_tokens,
                            speculative=False,
                        )
                    )
                try:
                    next(ordinary_token_iter)
                except StopIteration:
                    pass
                else:
                    raise ValueError(
                        "ordinary speculative-suppressed execution "
                        "returned extra output tokens"
                    )
                scheduler_rows = tuple(scheduler_rows)
                step_end_ns = self._clock_ns()
                prepared_scheduler = (
                    self.scheduler.prepare_postprocess(
                        tuple(seqs),
                        scheduler_rows,
                        is_prefill,
                        do_sample,
                        batch_kind,
                        decision_now_ns=decision_now_ns,
                        step_end_ns=step_end_ns,
                    )
                )
                kv_plans = tuple(
                    self.scheduler.block_manager
                    .prepare_speculative_kv_commit(
                        row.transaction,
                        row.sequence,
                        row.accepted_tokens,
                    )
                    for row in prepared_runtime.sequences
                    if row.transaction is not None
                )
                if residency_ticket_id is not None:
                    residency_precommit_rows = (
                        build_speculative_residency_precommit_rows(
                            kv_plans
                        )
                    )
                    committed_identities = tuple(
                        identity
                        for row in residency_precommit_rows
                        for identity in (
                            row.committed_block_identities
                        )
                    )
                    rejected_identities = tuple(
                        identity
                        for row in residency_precommit_rows
                        for identity in (
                            row.rejected_block_identities
                        )
                    )
                    self._call_speculative_residency_phase(
                        "precommit_speculative_residency_batch",
                        residency_ticket_id,
                        residency_precommit_rows,
                        expected_operation="precommit",
                        expected_status="precommitted",
                        expected_sequence_ids=tuple(
                            row.sequence_id
                            for row in residency_precommit_rows
                        ),
                        expected_committed_block_identities=(
                            committed_identities
                        ),
                        expected_rejected_block_identities=(
                            rejected_identities
                        ),
                        timeout_s=60.0,
                    )
                    residency_state = "precommitted"
                _commit_prepared_speculative_publication(
                    self,
                    runtime,
                    prepared_runtime,
                    kv_plans,
                    prepared_scheduler,
                )
                residency_publication_committed = True
                if residency_ticket_id is not None:
                    committed_identities = tuple(
                        identity
                        for row in residency_precommit_rows
                        for identity in (
                            row.committed_block_identities
                        )
                    )
                    rejected_identities = tuple(
                        identity
                        for row in residency_precommit_rows
                        for identity in (
                            row.rejected_block_identities
                        )
                    )
                    try:
                        self._call_speculative_residency_phase(
                            "seal_speculative_residency_batch",
                            residency_ticket_id,
                            expected_operation="seal",
                            expected_status="sealed",
                            expected_sequence_ids=tuple(
                                row.sequence_id
                                for row in residency_precommit_rows
                            ),
                            expected_committed_block_identities=(
                                committed_identities
                            ),
                            expected_rejected_block_identities=(
                                rejected_identities
                            ),
                            timeout_s=60.0,
                        )
                    except BaseException as error:
                        prepared_runtime.state = "committed"
                        self.speculative_runtime_poisoned = True
                        self.speculative_runtime_poison_reason = (
                            "speculative residency seal failed: "
                            f"{error}"
                        )
                        raise
                    residency_state = "sealed"
                descriptor = getattr(
                    runtime,
                    "model_runner_executor",
                    None,
                )
                capabilities = getattr(
                    descriptor,
                    "capabilities",
                    None,
                )
                if (
                    descriptor is not None
                    and getattr(
                        capabilities,
                        "requires_proposal_lifecycle",
                        False,
                    )
                ):
                    try:
                        for seq in seqs:
                            if not seq.is_finished:
                                continue
                            release_model_runner_proposal_sequence(
                                self.model_runner,
                                descriptor,
                                seq.seq_id,
                                int(
                                    getattr(
                                        seq,
                                        "sequence_epoch",
                                        0,
                                    )
                                ),
                                dispatch=(
                                    lambda method_name, *args: (
                                        _call_speculative_proposal_lifecycle(
                                            self,
                                            method_name,
                                            *args,
                                        )
                                    )
                                ),
                            )
                    except BaseException as error:
                        self.speculative_runtime_poisoned = True
                        self.speculative_runtime_poison_reason = (
                            "proposal executor sequence release "
                            f"failed: {error}"
                        )
                        raise
                prepared_runtime.state = "committed"
                speculative_output_token_counts = {
                    row.sequence_id: len(row.output_tokens)
                    for row in commit_rows
                }
                speculative_accepted_draft_token_counts = {
                    row.sequence_id: len(
                        row.accepted_draft_tokens
                    )
                    for row in commit_rows
                }
                speculative_proposal_token_counts = {
                    row.sequence_id: len(
                        row.proposal.token_ids
                    )
                    for row in prepared_runtime.sequences
                }
                speculative_proposal_token_ids_by_seq = {
                    row.sequence_id: list(
                        row.proposal.token_ids
                    )
                    for row in prepared_runtime.sequences
                }
                speculative_accepted_draft_token_ids_by_seq = {
                    row.sequence_id: list(
                        row.accepted_draft_tokens
                    )
                    for row in commit_rows
                }
                speculative_proposal_row_count = sum(
                    1
                    for count in (
                        speculative_proposal_token_counts
                        .values()
                    )
                    if count > 0
                )
                speculative_first_target_callback_count = (
                    prepared_runtime
                    .first_target_callback_count
                )
                speculative_fixed_q_group_count = (
                    prepared_runtime.tail_callback_count
                )
                speculative_runtime_timing_ms = dict(
                    prepared_runtime.timing_ms
                )
            except BaseException as error:
                residency_rollback_error = None
                if (
                    residency_ticket_id is not None
                    and residency_state
                    in ("prepared", "precommitted")
                    and not residency_publication_committed
                ):
                    try:
                        rollback_residency()
                    except BaseException as rollback_error:
                        residency_state = "rollback_failed"
                        residency_rollback_error = rollback_error
                        self.speculative_runtime_poisoned = True
                        self.speculative_runtime_poison_reason = (
                            "speculative residency rollback failed: "
                            f"{rollback_error}"
                        )
                if (
                    prepared_runtime is not None
                    and prepared_runtime.state == "prepared"
                ):
                    rollback_prepared_native_speculative_batch(
                        block_manager=(
                            self.scheduler.block_manager
                        ),
                        prepared=prepared_runtime,
                    )
                if residency_rollback_error is not None:
                    raise residency_rollback_error from error
                raise
            lifecycle = runtime.lifecycle
            if lifecycle is not None:
                try:
                    for seq, row in zip(
                        seqs,
                        scheduler_rows,
                    ):
                        if not row.output_tokens:
                            continue
                        lifecycle.synchronize_verified_history(
                            seq.seq_id,
                            tuple(seq.token_ids),
                        )
                        if seq.is_finished:
                            lifecycle.release_sequence(
                                seq.seq_id
                            )
                except BaseException as error:
                    self.speculative_runtime_poisoned = True
                    self.speculative_runtime_poison_reason = (
                        "draft lifecycle synchronization failed: "
                        f"{error}"
                    )
                    raise
            token_ids = ()
        else:
            ordinary_identity_rows = (
                kv_block_identity_rows_for(seqs)
            )
            drain_releases = getattr(
                self.scheduler,
                "drain_hybrid_state_release_events",
                None,
            )
            released_leases = (
                drain_releases()
                if drain_releases is not None
                else ()
            )
            try:
                if released_leases or ordinary_identity_rows:
                    token_ids = self.model_runner.call(
                        "run",
                        seqs,
                        is_prefill,
                        do_sample,
                        batch_kind,
                        released_leases,
                        ordinary_identity_rows,
                    )
                else:
                    token_ids = self.model_runner.call(
                        "run",
                        seqs,
                        is_prefill,
                        do_sample,
                        batch_kind,
                    )
            except BaseException:
                restore_releases = getattr(
                    self.scheduler,
                    "restore_hybrid_state_release_events",
                    None,
                )
                if restore_releases is not None:
                    restore_releases(released_leases)
                raise
            step_end_ns = self._clock_ns()
        if not partition.selected_sequences:
            self.scheduler.postprocess(
                seqs,
                token_ids,
                is_prefill,
                do_sample,
                batch_kind,
                decision_now_ns=decision_now_ns,
                step_end_ns=step_end_ns,
            )
            lifecycle = (
                None
                if runtime is None
                else runtime.lifecycle
            )
            if lifecycle is not None:
                try:
                    for seq in seqs:
                        if (
                            len(seq.completion_token_ids)
                            == completion_lengths_before[
                                seq.seq_id
                            ]
                        ):
                            continue
                        lifecycle.synchronize_verified_history(
                            seq.seq_id,
                            tuple(seq.token_ids),
                        )
                        if seq.is_finished:
                            lifecycle.release_sequence(
                                seq.seq_id
                            )
                except BaseException as error:
                    self.speculative_runtime_poisoned = True
                    self.speculative_runtime_poison_reason = (
                        "draft lifecycle synchronization failed: "
                        f"{error}"
                    )
                    raise
            descriptor = getattr(
                runtime,
                "model_runner_executor",
                None,
            )
            capabilities = getattr(
                descriptor,
                "capabilities",
                None,
            )
            if (
                descriptor is not None
                and getattr(
                    capabilities,
                    "requires_proposal_lifecycle",
                    False,
                )
            ):
                try:
                    for seq in seqs:
                        if not seq.is_finished:
                            continue
                        release_model_runner_proposal_sequence(
                            self.model_runner,
                            descriptor,
                            seq.seq_id,
                            int(
                                getattr(
                                    seq,
                                    "sequence_epoch",
                                    0,
                                )
                            ),
                            dispatch=(
                                lambda method_name, *args: (
                                    _call_speculative_proposal_lifecycle(
                                        self,
                                        method_name,
                                        *args,
                                    )
                                )
                            ),
                        )
                except BaseException as error:
                    self.speculative_runtime_poisoned = True
                    self.speculative_runtime_poison_reason = (
                        "proposal executor sequence release "
                        f"failed: {error}"
                    )
                    raise
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]       #output包含seq_id和已经生成的token列表
        token_deltas = {
            seq.seq_id: list(
                seq.completion_token_ids[
                    completion_lengths_before[seq.seq_id]:
                ]
            )
            for seq in seqs
        }
        timing_observation = self.scheduler.last_slo_observation()
        self.last_step_observation = {
            "policy_branch": self.scheduler.last_policy_branch,
            "batch_kind": batch_kind,
            "is_prefill": bool(is_prefill),
            "do_sample": bool(do_sample),
            "speculative_schedule_generation": (
                partition.schedule_generation
            ),
            "speculative_selected_seq_ids": list(
                partition.selected_sequence_ids
            ),
            "speculative_suppressed_seq_ids": list(
                partition.suppressed_sequence_ids
            ),
            "scheduled": scheduled_rows,
            "queue_before": queue_before,
            "queue_after": self.scheduler.observation_snapshot(),
            "new_completion_tokens_by_seq": token_deltas,
            "finished_seq_ids": [
                seq.seq_id for seq in seqs if seq.is_finished
            ],
            "speculative_output_token_counts": (
                speculative_output_token_counts
            ),
            "speculative_accepted_draft_token_counts": (
                speculative_accepted_draft_token_counts
            ),
            "speculative_proposal_token_counts": (
                speculative_proposal_token_counts
            ),
            "speculative_proposal_token_ids_by_seq": (
                speculative_proposal_token_ids_by_seq
            ),
            "speculative_accepted_draft_token_ids_by_seq": (
                speculative_accepted_draft_token_ids_by_seq
            ),
            "speculative_proposal_row_count": (
                speculative_proposal_row_count
            ),
            "speculative_first_target_callback_count": (
                speculative_first_target_callback_count
            ),
            "speculative_fixed_q_group_count": (
                speculative_fixed_q_group_count
            ),
            "speculative_runtime_timing_ms": (
                speculative_runtime_timing_ms
            ),
            "memory": self.model_runner.memory_snapshot(),
            **timing_observation,
        }
        return outputs, num_tokens      #计算的是每个step的单次增量

    def is_finished(self):
        return self.scheduler.is_finished()

    def hybrid_state_release_event_count(self):
        events = getattr(
            self.scheduler,
            "_hybrid_state_release_events",
            None,
        )
        if events is None:
            raise RuntimeError(
                "Scheduler hybrid state release events are unavailable"
            )
        return len(events)

    def generate(
        self, 
        prompts: list[str] | list[list[int]],               #输入提示：可以是字符串列表（未分词）也可以是token id列表（已分词）
        sampling_params: SamplingParams | list[SamplingParams], 
        use_tqdm: bool = True, 
    ) -> list[int]:
        if use_tqdm: 
            pbar = tqdm(total = len(prompts), desc = "Generating", dynamic_ncols = True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)    #保证每个prompt都有一组sampling_params
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():           #根据waiting和running队列是否为空判断
            t = perf_counter()                  #纳秒级别的高精度时间（自计算机启动经过的时间）
            output, num_tokens = self.step()
            if use_tqdm:
                # prefill
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                # decode
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)  #为了区分decode和prefill 规定decode阶段的num_tokens都是-1 （decode每个step阶段都是生成1个token）
                pbar.set_postfix({
                    "prefill": f"{int(prefill_throughput)} tok/s",     #一次step的吞吐
                    "Decode": f"{int(decode_throughput)} tok/s"
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
    
