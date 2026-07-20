from __future__ import annotations

from collections import deque
from types import MappingProxyType

from tinyvllm.config import Config
from tinyvllm.engine.sequence import Sequence, SequenceStatus
from tinyvllm.engine.block_manager import BlockManager

ADAPTIVE_MIXED_INACTIVE = "inactive"
ADAPTIVE_MIXED_ACTIVE = "active"
ADAPTIVE_MIXED_DRAINING = "draining"
INT64_MAX = (1 << 63) - 1


def build_slo_chunk_ladder(
    max_chunk_tokens: int,
    min_chunk_tokens: int,
) -> tuple[int, ...]:
    if (
        isinstance(max_chunk_tokens, bool)
        or isinstance(min_chunk_tokens, bool)
        or not isinstance(max_chunk_tokens, int)
        or not isinstance(min_chunk_tokens, int)
        or min_chunk_tokens <= 0
        or max_chunk_tokens < min_chunk_tokens
        or max_chunk_tokens % min_chunk_tokens != 0
    ):
        raise ValueError("invalid SLO chunk ladder")
    return tuple(
        range(max_chunk_tokens, min_chunk_tokens - 1, -min_chunk_tokens)
    )


def select_slo_chunk(
    *,
    remaining_slack_ns: int,
    cost_intercept_ns: int,
    cost_per_prefill_token_ns: int,
    token_ladder: tuple[int, ...],
) -> tuple[int | None, int | None]:
    for tokens in token_ladder:
        if (
            cost_per_prefill_token_ns
            > (INT64_MAX - cost_intercept_ns) // tokens
        ):
            raise OverflowError("P5 predicted step cost overflows int64")
        predicted = cost_intercept_ns + tokens * cost_per_prefill_token_ns
        if predicted <= remaining_slack_ns:
            return tokens, predicted
    return None, None


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_model_len = getattr(config, "max_model_len", 0) or 0
        self.max_num_prefill_tokens_per_step = getattr(config, "max_num_prefill_tokens_per_step", 0)
        self.chunked_prefill_decode_first = getattr(config, "chunked_prefill_decode_first", True)
        self.chunked_prefill_max_consecutive_chunks = getattr(config, "chunked_prefill_max_consecutive_chunks", 0)
        self.chunked_prefill_mixed_batch = getattr(config, "chunked_prefill_mixed_batch", False)
        self.chunked_prefill_mixed_min_prompt_tokens = getattr(config, "chunked_prefill_mixed_min_prompt_tokens", 0)
        self.chunked_prefill_adaptive_mixed = getattr(
            config, "chunked_prefill_adaptive_mixed", False
        )
        self.chunked_prefill_adaptive_enter_waiting = getattr(
            config, "chunked_prefill_adaptive_enter_waiting", 8
        )
        self.chunked_prefill_adaptive_exit_waiting = getattr(
            config, "chunked_prefill_adaptive_exit_waiting", 2
        )
        self.chunked_prefill_adaptive_transition_steps = getattr(
            config, "chunked_prefill_adaptive_transition_steps", 2
        )
        self.chunked_prefill_adaptive_max_mixed_steps = getattr(
            config, "chunked_prefill_adaptive_max_mixed_steps", 2
        )
        self.chunked_prefill_slo_mixed = getattr(
            config, "chunked_prefill_slo_mixed", False
        )
        self.chunked_prefill_slo_target_gap_ns = getattr(
            config, "chunked_prefill_slo_target_gap_ns", 0
        )
        self.chunked_prefill_slo_reserve_ns = getattr(
            config, "chunked_prefill_slo_reserve_ns", 0
        )
        self.chunked_prefill_slo_cost_intercept_ns = getattr(
            config, "chunked_prefill_slo_cost_intercept_ns", 0
        )
        self.chunked_prefill_slo_cost_per_prefill_token_ns = getattr(
            config, "chunked_prefill_slo_cost_per_prefill_token_ns", 0
        )
        self.chunked_prefill_slo_min_chunk_tokens = getattr(
            config, "chunked_prefill_slo_min_chunk_tokens", 16
        )
        self.slo_chunk_ladder = (
            build_slo_chunk_ladder(
                self.max_num_prefill_tokens_per_step,
                self.chunked_prefill_slo_min_chunk_tokens,
            )
            if self.chunked_prefill_slo_mixed
            else ()
        )
        self.adaptive_mixed_state = ADAPTIVE_MIXED_INACTIVE
        self.adaptive_high_streak = 0
        self.adaptive_low_streak = 0
        self.adaptive_consecutive_mixed_steps = 0
        self._consecutive_prefill_chunks = 0
        self.last_policy_branch: str | None = None
        self.last_slo_decision = MappingProxyType({})
        self._last_slo_postprocess: dict = {}
        self.decode_progress_ns_by_seq_id: dict[int, int] = {}
        self.slo_clock_invalid = False
        self.slo_clock_invalid_reason: str | None = None
        self._last_slo_decision_now_ns: int | None = None
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()     #未分配 KV 缓存块
        self.prefilling: deque[Sequence] = deque()  #chunked prefill 中，已分配 KV 但 prompt 未完整算完
        self.running: deque[Sequence] = deque()     #已分配 KV 缓存块  参与decode阶段生成

    @property
    def chunked_prefill_enabled(self) -> bool:
        return self.max_num_prefill_tokens_per_step > 0

    def is_finished(self):
        return not self.waiting and not self.prefilling and not self.running

    def observation_snapshot(self) -> dict:
        block_manager = self.block_manager
        return {
            "waiting_seq_ids": [seq.seq_id for seq in self.waiting],
            "prefilling_seq_ids": [seq.seq_id for seq in self.prefilling],
            "running_seq_ids": [seq.seq_id for seq in self.running],
            "free_kv_blocks": len(block_manager.free_block_ids),
            "used_kv_blocks": len(block_manager.used_block_ids),
            "total_kv_blocks": len(block_manager.blocks),
            "kv_block_size_tokens": block_manager.block_size,
            "consecutive_prefill_chunks": self._consecutive_prefill_chunks,
            "adaptive_mixed_state": self.adaptive_mixed_state,
            "adaptive_high_streak": self.adaptive_high_streak,
            "adaptive_low_streak": self.adaptive_low_streak,
            "adaptive_consecutive_mixed_steps": self.adaptive_consecutive_mixed_steps,
        }

    def _reset_adaptive_mixed_controller(self) -> None:
        self.adaptive_mixed_state = ADAPTIVE_MIXED_INACTIVE
        self.adaptive_high_streak = 0
        self.adaptive_low_streak = 0
        self.adaptive_consecutive_mixed_steps = 0

    def _maybe_reset_adaptive_mixed_controller(self) -> None:
        if not self.waiting and not self.prefilling and not self.running:
            self._reset_adaptive_mixed_controller()
            self.decode_progress_ns_by_seq_id.clear()
            self._last_slo_decision_now_ns = None

    def _adaptive_transition_eligible(self) -> bool:
        return bool(
            (
                self.chunked_prefill_adaptive_mixed
                or self.chunked_prefill_slo_mixed
            )
            and self.chunked_prefill_enabled
            and self.running
            and (self.waiting or self.prefilling)
        )

    def _update_adaptive_mixed_state(self, waiting_depth: int) -> None:
        if not self._adaptive_transition_eligible():
            self.adaptive_high_streak = 0
            self.adaptive_low_streak = 0
            return

        transition_steps = self.chunked_prefill_adaptive_transition_steps
        if self.adaptive_mixed_state == ADAPTIVE_MIXED_INACTIVE:
            self.adaptive_low_streak = 0
            if waiting_depth >= self.chunked_prefill_adaptive_enter_waiting:
                self.adaptive_high_streak += 1
            else:
                self.adaptive_high_streak = 0
            if self.adaptive_high_streak >= transition_steps:
                self.adaptive_mixed_state = ADAPTIVE_MIXED_ACTIVE
                self.adaptive_high_streak = 0
            return

        if self.adaptive_mixed_state == ADAPTIVE_MIXED_ACTIVE:
            self.adaptive_high_streak = 0
            if waiting_depth <= self.chunked_prefill_adaptive_exit_waiting:
                self.adaptive_low_streak += 1
            else:
                self.adaptive_low_streak = 0
            if self.adaptive_low_streak >= transition_steps:
                self.adaptive_mixed_state = (
                    ADAPTIVE_MIXED_DRAINING
                    if self.prefilling
                    else ADAPTIVE_MIXED_INACTIVE
                )
                self.adaptive_low_streak = 0
            return

        self.adaptive_low_streak = 0
        if waiting_depth >= self.chunked_prefill_adaptive_enter_waiting:
            self.adaptive_high_streak += 1
        else:
            self.adaptive_high_streak = 0
        if self.adaptive_high_streak >= transition_steps:
            self.adaptive_mixed_state = ADAPTIVE_MIXED_ACTIVE
            self.adaptive_high_streak = 0
        elif not self.prefilling:
            self.adaptive_mixed_state = ADAPTIVE_MIXED_INACTIVE
            self.adaptive_consecutive_mixed_steps = 0

    def _return_schedule(self, scheduled: tuple, branch: str):
        self.last_policy_branch = branch
        return scheduled

    def _publish_slo_decision(self, values: dict) -> None:
        self.last_slo_decision = MappingProxyType(dict(values))
        self._last_slo_postprocess = {}

    def last_slo_observation(self) -> dict:
        return {
            **dict(self.last_slo_decision),
            **dict(self._last_slo_postprocess),
        }

    def _invalidate_slo_clock(self, reason: str) -> None:
        if not self.slo_clock_invalid:
            self.slo_clock_invalid = True
            self.slo_clock_invalid_reason = reason

    def _validate_slo_decision_time(
        self,
        decision_now_ns: int | None,
    ) -> bool:
        if (
            isinstance(decision_now_ns, bool)
            or not isinstance(decision_now_ns, int)
            or decision_now_ns < 0
        ):
            self._invalidate_slo_clock("invalid_decision_timestamp")
            return False
        if (
            self._last_slo_decision_now_ns is not None
            and decision_now_ns < self._last_slo_decision_now_ns
        ):
            self._invalidate_slo_clock("decision_clock_regressed")
            return False
        self._last_slo_decision_now_ns = decision_now_ns
        return not self.slo_clock_invalid

    def _validate_slo_step_end(
        self,
        decision_now_ns: int | None,
        step_end_ns: int | None,
    ) -> bool:
        if (
            isinstance(step_end_ns, bool)
            or not isinstance(step_end_ns, int)
            or step_end_ns < 0
            or isinstance(decision_now_ns, bool)
            or not isinstance(decision_now_ns, int)
            or step_end_ns < decision_now_ns
        ):
            self._invalidate_slo_clock("invalid_step_end_timestamp")
            return False
        return True

    def _oldest_runnable_decode(
        self,
        decision_now_ns: int,
    ) -> tuple[int, int, int] | None:
        oldest = None
        for seq in self.running:
            progress_ns = self.decode_progress_ns_by_seq_id.get(seq.seq_id)
            if progress_ns is None:
                return None
            if progress_ns > decision_now_ns:
                self._invalidate_slo_clock("progress_timestamp_in_future")
                return None
            candidate = (progress_ns, seq.seq_id)
            if oldest is None or candidate < oldest:
                oldest = candidate
        if oldest is None:
            return None
        progress_ns, seq_id = oldest
        return seq_id, progress_ns, decision_now_ns - progress_ns

    def _new_slo_decision(
        self,
        *,
        decision_now_ns: int | None,
        demand_state_before: str,
    ) -> dict:
        return {
            "decision_now_ns": decision_now_ns,
            "target_gap_ns": self.chunked_prefill_slo_target_gap_ns,
            "reserve_ns": self.chunked_prefill_slo_reserve_ns,
            "oldest_decode_seq_id": None,
            "oldest_decode_progress_ns": None,
            "oldest_decode_age_ns": None,
            "remaining_slack_ns": None,
            "cost_intercept_ns":
                self.chunked_prefill_slo_cost_intercept_ns,
            "cost_per_prefill_token_ns":
                self.chunked_prefill_slo_cost_per_prefill_token_ns,
            "candidate_chunk_tokens": list(self.slo_chunk_ladder),
            "predicted_step_ns": None,
            "selected_chunk_tokens": None,
            "actual_prefill_tokens": 0,
            "scheduled_decode_seq_ids": [],
            "demand_state_before": demand_state_before,
            "demand_state_after": demand_state_before,
            "suppression_reason": None,
            "clock_invalid": self.slo_clock_invalid,
            "clock_invalid_reason": self.slo_clock_invalid_reason,
        }

    def _return_slo_decode(
        self,
        decision: dict,
        *,
        branch: str,
        suppression_reason: str,
    ) -> tuple[list[Sequence], bool, bool]:
        decision.update({
            "suppression_reason": suppression_reason,
            "clock_invalid": self.slo_clock_invalid,
            "clock_invalid_reason": self.slo_clock_invalid_reason,
        })
        self._publish_slo_decision(decision)
        return self._return_schedule(
            (*self._schedule_decode(), True),
            branch,
        )

    def _schedule_slo_mixed(
        self,
        waiting_depth: int,
        decision_now_ns: int | None,
    ) -> tuple[list[Sequence], bool, bool] | tuple[
        list[Sequence], bool, bool, str
    ]:
        demand_state_before = self.adaptive_mixed_state
        decision = self._new_slo_decision(
            decision_now_ns=decision_now_ns,
            demand_state_before=demand_state_before,
        )
        if not self.running:
            self._update_adaptive_mixed_state(waiting_depth)
            decision["demand_state_after"] = self.adaptive_mixed_state
            prefill = self._schedule_chunked_prefill()
            self._publish_slo_decision(decision)
            if prefill is not None:
                return self._return_schedule(
                    prefill,
                    "slo_mixed_no_running_prefill",
                )
            return self._return_schedule(
                (*self._schedule_decode(), True),
                "slo_mixed_no_running_prefill",
            )

        if not self._validate_slo_decision_time(decision_now_ns):
            return self._return_slo_decode(
                decision,
                branch="slo_mixed_clock_invalid_decode",
                suppression_reason="clock_invalid",
            )
        oldest = self._oldest_runnable_decode(decision_now_ns)
        if self.slo_clock_invalid:
            return self._return_slo_decode(
                decision,
                branch="slo_mixed_clock_invalid_decode",
                suppression_reason="clock_invalid",
            )
        if oldest is None:
            return self._return_slo_decode(
                decision,
                branch="slo_mixed_missing_progress_decode",
                suppression_reason="missing_decode_progress",
            )
        (
            oldest_decode_seq_id,
            oldest_decode_progress_ns,
            oldest_decode_age_ns,
        ) = oldest
        remaining_slack_ns = (
            self.chunked_prefill_slo_target_gap_ns
            - self.chunked_prefill_slo_reserve_ns
            - oldest_decode_age_ns
        )
        decision.update({
            "oldest_decode_seq_id": oldest_decode_seq_id,
            "oldest_decode_progress_ns": oldest_decode_progress_ns,
            "oldest_decode_age_ns": oldest_decode_age_ns,
            "remaining_slack_ns": remaining_slack_ns,
        })

        self._update_adaptive_mixed_state(waiting_depth)
        decision["demand_state_after"] = self.adaptive_mixed_state
        if self.adaptive_mixed_state == ADAPTIVE_MIXED_INACTIVE:
            return self._return_slo_decode(
                decision,
                branch="slo_mixed_inactive_decode",
                suppression_reason="inactive",
            )
        if remaining_slack_ns <= 0:
            return self._return_slo_decode(
                decision,
                branch="slo_mixed_no_slack_decode",
                suppression_reason="no_slack",
            )

        selected_chunk_tokens, predicted_step_ns = select_slo_chunk(
            remaining_slack_ns=remaining_slack_ns,
            cost_intercept_ns=self.chunked_prefill_slo_cost_intercept_ns,
            cost_per_prefill_token_ns=(
                self.chunked_prefill_slo_cost_per_prefill_token_ns
            ),
            token_ladder=self.slo_chunk_ladder,
        )
        if selected_chunk_tokens is None:
            return self._return_slo_decode(
                decision,
                branch="slo_mixed_cost_suppressed_decode",
                suppression_reason="cost_suppressed",
            )
        decision.update({
            "selected_chunk_tokens": selected_chunk_tokens,
            "predicted_step_ns": predicted_step_ns,
        })

        mixed = self._schedule_mixed_prefill_decode(
            allow_waiting_admission=(
                self.adaptive_mixed_state == ADAPTIVE_MIXED_ACTIVE
            ),
            require_decode=True,
            required_decode_seq_id=oldest_decode_seq_id,
            max_prefill_tokens=selected_chunk_tokens,
        )
        if mixed is None:
            return self._return_slo_decode(
                decision,
                branch="slo_mixed_transaction_fallback_decode",
                suppression_reason="transaction_fallback",
            )

        actual_prefill_tokens = sum(
            seq.prefill_chunk_end - seq.prefill_chunk_start
            for seq in mixed[0]
            if not getattr(seq, "step_is_decode", False)
        )
        scheduled_decode_seq_ids = [
            seq.seq_id
            for seq in mixed[0]
            if getattr(seq, "step_is_decode", False)
        ]
        assert actual_prefill_tokens <= selected_chunk_tokens
        assert oldest_decode_seq_id in scheduled_decode_seq_ids
        decision.update({
            "actual_prefill_tokens": actual_prefill_tokens,
            "scheduled_decode_seq_ids": scheduled_decode_seq_ids,
        })
        self._publish_slo_decision(decision)
        branch = (
            "slo_mixed_draining_prefill_decode"
            if self.adaptive_mixed_state == ADAPTIVE_MIXED_DRAINING
            else "slo_mixed_prefill_decode"
        )
        return self._return_schedule(mixed, branch)

    def add(self, seq: Sequence):
        self._validate_admission(seq)
        self.waiting.append(seq)

    def _validate_admission(self, seq: Sequence):
        max_tokens = max(0, getattr(seq, "max_tokens", 0))
        total_tokens = len(seq) + max_tokens
        if self.max_model_len > 0 and total_tokens > self.max_model_len:
            raise ValueError(
                "request length exceeds max_model_len: "
                f"prompt_tokens={len(seq)}, max_tokens={max_tokens}, "
                f"total_tokens={total_tokens}, max_model_len={self.max_model_len}"
            )

        # KV cache stores prompt tokens plus generated tokens that are needed as
        # context for later decode steps. The last requested output token is not
        # decoded again, so it does not need its own KV slot.
        kv_tokens = len(seq) + max(0, max_tokens - 1)
        required_blocks = (kv_tokens + self.block_manager.block_size - 1) // self.block_manager.block_size
        available_blocks = len(self.block_manager.blocks)
        if required_blocks > available_blocks:
            raise ValueError(
                "request length exceeds KV cache capacity: "
                f"prompt_tokens={len(seq)}, max_tokens={max_tokens}, "
                f"kv_tokens={kv_tokens}, required_blocks={required_blocks}, "
                f"available_blocks={available_blocks}, block_size={self.block_manager.block_size}"
            )

    def schedule(
        self,
        decision_now_ns: int | None = None,
    ) -> tuple[list[Sequence], bool, bool]:
        waiting_depth = len(self.waiting)
        self._maybe_reset_adaptive_mixed_controller()
        if self.chunked_prefill_enabled and self.chunked_prefill_slo_mixed:
            return self._schedule_slo_mixed(
                waiting_depth,
                decision_now_ns,
            )
        if self.chunked_prefill_enabled and self.chunked_prefill_adaptive_mixed:
            return self._schedule_adaptive_mixed(waiting_depth)

        if self.chunked_prefill_enabled:
            if self.chunked_prefill_decode_first and self.running:
                self._consecutive_prefill_chunks = 0
                return self._return_schedule(
                    (*self._schedule_decode(), True),
                    "decode_first",
                )
            if (self.running
                    and self.chunked_prefill_max_consecutive_chunks > 0
                    and self._consecutive_prefill_chunks >= self.chunked_prefill_max_consecutive_chunks):
                self._consecutive_prefill_chunks = 0
                return self._return_schedule(
                    (*self._schedule_decode(), True),
                    "bounded_prefill_yield",
                )
            if self.chunked_prefill_mixed_batch and self.running:
                if not self._mixed_prefill_admission_allowed():
                    self._consecutive_prefill_chunks = 0
                    return self._return_schedule(
                        (*self._schedule_decode(), True),
                        "decode_fallback",
                    )
                mixed = self._schedule_mixed_prefill_decode()
                if mixed is not None:
                    if len(mixed) == 4:
                        self._consecutive_prefill_chunks = 0
                        branch = "mixed_prefill_decode"
                    else:
                        self._consecutive_prefill_chunks += 1
                        branch = "chunked_prefill"
                    return self._return_schedule(mixed, branch)
            prefill = self._schedule_chunked_prefill()
            if prefill is not None:
                self._consecutive_prefill_chunks += 1
                return self._return_schedule(prefill, "chunked_prefill")
            self._consecutive_prefill_chunks = 0
            return self._return_schedule(
                (*self._schedule_decode(), True),
                "decode_fallback",
            )

        # prefill, 从 waiting 队列中取出 seq   prefill阶段：处理输入 prompt 的所有 token（批量计算，生成初始 KV 缓存）。
        scheduled_seqs = [] #scheduled_seqs和waiting队列的区别：scheduled_seqs 是从 waiting 队列中筛选出来的、满足调度条件的序列集合
        num_seqs = 0        #number of sequence in the current batch
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]                   # 这里不使用 popleft的原因是 waiting 队列不一定调度成功（如下if判断） 如果调度不成功 这个token就不在waiting队列里了
            reusable_tokens, required_free_blocks = (
                self.block_manager.estimate_admission(seq)
            )
            prefill_tokens = len(seq) - reusable_tokens
            if (
                num_batched_tokens + prefill_tokens
                > self.max_num_batched_tokens
                or len(self.block_manager.free_block_ids)
                < required_free_blocks
            ):
                break
            num_seqs += 1
            max_cached_tokens = (
                self.block_manager.max_reusable_tokens(seq)
            )
            self.block_manager.allocate(
                seq,
                publish_hashes=False,
                max_cached_tokens=max_cached_tokens,
            )
            seq.prefill_chunk_start = seq.num_cached_tokens
            seq.prefill_chunk_end = len(seq)
            seq.prefill_chunk_final = True
            assert (
                seq.prefill_chunk_end > seq.prefill_chunk_start
            )
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return self._return_schedule(
                (scheduled_seqs, True, True),
                "legacy_prefill",
            )

        # decode，从 running 队列中取出 seq   Decode 阶段：逐 token 生成（利用已有 KV 缓存，每次生成一个新 token）。
        return self._return_schedule(
            (*self._schedule_decode(), True),
            "legacy_decode",
        )

    def _schedule_adaptive_mixed(
            self,
            waiting_depth: int) -> tuple[list[Sequence], bool, bool] | tuple[list[Sequence], bool, bool, str]:
        self._update_adaptive_mixed_state(waiting_depth)

        if not self.running:
            self.adaptive_consecutive_mixed_steps = 0
            prefill = self._schedule_chunked_prefill()
            if prefill is not None:
                return self._return_schedule(
                    prefill,
                    "adaptive_mixed_chunked_prefill",
                )
            return self._return_schedule(
                (*self._schedule_decode(), True),
                "adaptive_mixed_decode_fallback",
            )

        if (
            self.adaptive_mixed_state == ADAPTIVE_MIXED_DRAINING
            and not self.prefilling
        ):
            self.adaptive_mixed_state = ADAPTIVE_MIXED_INACTIVE
            self.adaptive_consecutive_mixed_steps = 0

        if self.adaptive_mixed_state == ADAPTIVE_MIXED_INACTIVE:
            self.adaptive_consecutive_mixed_steps = 0
            return self._return_schedule(
                (*self._schedule_decode(), True),
                "adaptive_mixed_decode_first",
            )

        if (
            self.adaptive_consecutive_mixed_steps
            >= self.chunked_prefill_adaptive_max_mixed_steps
        ):
            self.adaptive_consecutive_mixed_steps = 0
            return self._return_schedule(
                (*self._schedule_decode(), True),
                "adaptive_mixed_decode_yield",
            )

        mixed = self._schedule_mixed_prefill_decode(
            allow_waiting_admission=(
                self.adaptive_mixed_state == ADAPTIVE_MIXED_ACTIVE
            ),
            require_decode=True,
        )
        if mixed is None:
            self.adaptive_consecutive_mixed_steps = 0
            return self._return_schedule(
                (*self._schedule_decode(), True),
                "adaptive_mixed_decode_fallback",
            )
        self.adaptive_consecutive_mixed_steps += 1
        return self._return_schedule(
            mixed,
            "adaptive_mixed_prefill_decode",
        )

    def _schedule_decode(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_seqs = 0
        while self.running and num_seqs < self.max_num_seqs:        
            seq = self.running.popleft();          # 这里是preempt抢占资源保证 running队列一定调度成功
            #[thinking] 这里可能有一个能够优化的点 就是在抢占资源的时候默认是t出running的第一个 但是第一个腾出来的空间未必够新的seq使用 所以可以考虑合理规划选一个大小相近的seq去剔除
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))       #当前step结束 但未到达终止条件 所以需要在返回running队列
        return scheduled_seqs, False

    def _mixed_prefill_admission_allowed(self) -> bool:
        if self.chunked_prefill_mixed_min_prompt_tokens <= 0:
            return True
        if self.prefilling:
            # 已经进入 chunked prefill 的请求继续允许 mixed drain；阈值只控制新请求接入。
            return True
        if not self.waiting:
            return True
        candidate = self.waiting[0]
        remaining_prompt_tokens = max(0, len(candidate) - getattr(candidate, "num_computed_tokens", 0))
        return remaining_prompt_tokens >= self.chunked_prefill_mixed_min_prompt_tokens

    def _schedule_chunked_prefill(
            self,
            max_prefill_seqs: int | None = None,
            max_prefill_tokens: int | None = None,
            *,
            allow_waiting_admission: bool = True,
            reserved_free_blocks: int = 0) -> tuple[list[Sequence], bool, bool] | None:
        max_prefill_seqs = self.max_num_seqs if max_prefill_seqs is None else max_prefill_seqs
        max_prefill_tokens = self.max_num_batched_tokens if max_prefill_tokens is None else max_prefill_tokens
        if self.prefilling:
            seq = self.prefilling.popleft()
            return self._schedule_one_prefill_chunk(seq, max_chunk_tokens=max_prefill_tokens)

        if not allow_waiting_admission or not self.waiting:
            return None

        candidate = self.waiting[0]
        _, required_free_blocks = self.block_manager.estimate_admission(
            candidate
        )
        if (
            len(self.block_manager.free_block_ids)
            - required_free_blocks
            < reserved_free_blocks
        ):
            return None
        seq = self.waiting.popleft()
        max_cached_tokens = self.block_manager.max_reusable_tokens(seq)
        self.block_manager.allocate(
            seq,
            publish_hashes=False,
            max_cached_tokens=max_cached_tokens,
        )
        seq.status = SequenceStatus.PREFILLING
        first = self._schedule_one_prefill_chunk(seq, max_chunk_tokens=max_prefill_tokens)
        if first is None:
            return None
        scheduled, is_prefill, do_sample = first
        if not do_sample:
            return first

        num_batched_tokens = scheduled[0].prefill_chunk_end - scheduled[0].prefill_chunk_start
        while self.waiting and len(scheduled) < max_prefill_seqs:
            candidate = self.waiting[0]
            reusable_tokens, required_free_blocks = (
                self.block_manager.estimate_admission(candidate)
            )
            prefill_tokens = len(candidate) - reusable_tokens
            # Conservative short-prompt batching: only admit prompts whose
            # uncached suffix finishes in one chunk. Prefix state is read before
            # allocation, and new hashes remain unpublished until postprocess.
            if prefill_tokens > self.max_num_prefill_tokens_per_step:
                break
            if num_batched_tokens + prefill_tokens > max_prefill_tokens:
                break
            if (
                len(self.block_manager.free_block_ids)
                - required_free_blocks
                < reserved_free_blocks
            ):
                break
            seq = self.waiting.popleft()
            max_cached_tokens = (
                self.block_manager.max_reusable_tokens(seq)
            )
            self.block_manager.allocate(
                seq,
                publish_hashes=False,
                max_cached_tokens=max_cached_tokens,
            )
            seq.status = SequenceStatus.PREFILLING
            one = self._schedule_one_prefill_chunk(seq, max_chunk_tokens=max_prefill_tokens - num_batched_tokens)
            if one is None or not one[2]:
                self.prefilling.appendleft(seq)
                break
            scheduled.append(seq)
            num_batched_tokens += seq.prefill_chunk_end - seq.prefill_chunk_start
        return scheduled, is_prefill, do_sample

    def _schedule_one_prefill_chunk(
            self,
            seq: Sequence,
            max_chunk_tokens: int | None = None) -> tuple[list[Sequence], bool, bool] | None:
        if seq.num_computed_tokens >= len(seq):
            # 全 prompt 命中 prefix cache 时仍需重算最后一个 prompt token 拿 logits，采样首个输出 token。
            seq.prefill_chunk_start = max(0, len(seq) - 1)
            seq.prefill_chunk_end = len(seq)
            seq.prefill_chunk_final = True
            return [seq], True, True

        start = seq.num_computed_tokens
        max_chunk_tokens = self.max_num_prefill_tokens_per_step if max_chunk_tokens is None else max_chunk_tokens
        chunk_len = min(self.max_num_prefill_tokens_per_step, max_chunk_tokens, len(seq) - start)
        end = start + chunk_len
        seq.prefill_chunk_start = start
        seq.prefill_chunk_end = end
        seq.prefill_chunk_final = (end == len(seq))
        if seq.prefill_chunk_final:
            assert (
                seq.prefill_chunk_end > seq.prefill_chunk_start
            )
        return [seq], True, seq.prefill_chunk_final

    def _mixed_decode_reservation(
        self,
        required_decode_seq_id: int | None = None,
    ) -> tuple[int, int] | None:
        if self.max_num_seqs < 2 or self.max_num_batched_tokens < 2:
            return None
        candidates = (
            [
                seq for seq in self.running
                if seq.seq_id == required_decode_seq_id
            ]
            if required_decode_seq_id is not None
            else list(self.running)
        )
        for seq in candidates:
            required_free_blocks = int(
                len(seq) % self.block_manager.block_size == 1
            )
            if len(self.block_manager.free_block_ids) >= required_free_blocks:
                return seq.seq_id, required_free_blocks
        return None

    def _schedule_mixed_prefill_decode(
            self,
            *,
            allow_waiting_admission: bool = True,
            require_decode: bool = False,
            required_decode_seq_id: int | None = None,
            max_prefill_tokens: int | None = None) -> tuple[list[Sequence], bool, bool, str] | tuple[list[Sequence], bool, bool] | None:
        reserved_seq_id = None
        reserved_free_blocks = 0
        if require_decode or required_decode_seq_id is not None:
            reservation = self._mixed_decode_reservation(
                required_decode_seq_id
            )
            if reservation is None:
                return None
            reserved_seq_id, reserved_free_blocks = reservation

        prefill_slots = max(1, self.max_num_seqs - 1)
        decode_query_tokens = 1 if self.running else 0
        prefill_budget = (
            max(1, self.max_num_batched_tokens - decode_query_tokens)
            if max_prefill_tokens is None
            else min(
                max_prefill_tokens,
                max(1, self.max_num_batched_tokens - decode_query_tokens),
            )
        )
        prefill = self._schedule_chunked_prefill(
            max_prefill_seqs=prefill_slots,
            max_prefill_tokens=prefill_budget,
            allow_waiting_admission=allow_waiting_admission,
            reserved_free_blocks=reserved_free_blocks,
        )
        if prefill is None:
            return None
        prefill_seqs, is_prefill, prefill_do_sample = prefill
        assert is_prefill
        prefill_tokens = sum(seq.prefill_chunk_end - seq.prefill_chunk_start for seq in prefill_seqs)
        decode_seqs = []
        if reserved_seq_id is not None:
            reserved_index = next(
                index for index, seq in enumerate(self.running)
                if seq.seq_id == reserved_seq_id
            )
            reserved_seq = self.running[reserved_index]
            del self.running[reserved_index]
            self.block_manager.may_append(reserved_seq)
            decode_seqs.append(reserved_seq)

        while (self.running
               and len(prefill_seqs) + len(decode_seqs) < self.max_num_seqs
               and prefill_tokens + len(decode_seqs) < self.max_num_batched_tokens):
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    seq = None
                    break
            if seq is None:
                continue
            self.block_manager.may_append(seq)
            decode_seqs.append(seq)

        if not decode_seqs:
            assert not require_decode
            return prefill

        assert prefill_seqs
        for seq in prefill_seqs:
            seq.step_is_decode = False
            seq.step_do_sample = prefill_do_sample
        for seq in decode_seqs:
            seq.step_is_decode = True
            seq.step_do_sample = True
        return prefill_seqs + decode_seqs, True, True, "mixed"

    def preempt(self, seq: Sequence):       #将正在running队列中的seq给“踢”出去 
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int] | None,
                    is_prefill: bool = False, do_sample: bool = True,
                    batch_kind: str | None = None, *,
                    decision_now_ns: int | None = None,
                    step_end_ns: int | None = None):
        self._last_slo_postprocess = {}
        timestamps_present = (
            decision_now_ns is not None or step_end_ns is not None
        )
        timestamp_valid = (
            self._validate_slo_step_end(decision_now_ns, step_end_ns)
            if timestamps_present
            else False
        )
        progress_updates = {}
        finished_progress_entries_removed = []
        if batch_kind == "mixed":
            progress_updates, finished_progress_entries_removed = (
                self._postprocess_mixed(
                    seqs,
                    token_ids,
                    step_end_ns=step_end_ns if timestamp_valid else None,
                )
            )
            self._publish_slo_postprocess(
                decision_now_ns,
                step_end_ns,
                timestamp_valid,
                progress_updates,
                finished_progress_entries_removed,
            )
            self._maybe_reset_adaptive_mixed_controller()
            return
        if is_prefill and self.chunked_prefill_enabled:
            progress_updates, finished_progress_entries_removed = (
                self._postprocess_chunked_prefill(
                    seqs,
                    token_ids,
                    do_sample,
                    step_end_ns=step_end_ns if timestamp_valid else None,
                )
            )
            self._publish_slo_postprocess(
                decision_now_ns,
                step_end_ns,
                timestamp_valid,
                progress_updates,
                finished_progress_entries_removed,
            )
            self._maybe_reset_adaptive_mixed_controller()
            return
        if is_prefill:
            for seq in seqs:
                old_end = seq.num_computed_tokens
                new_end = seq.prefill_chunk_end
                assert new_end > seq.prefill_chunk_start
                self.block_manager.commit_prefill(
                    seq,
                    old_end,
                    new_end,
                )
                seq.num_computed_tokens = new_end
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            self._record_decode_progress(
                seq,
                step_end_ns if timestamp_valid else None,
                progress_updates,
            )
            # 如果不能忽略句子终止符号，并且遇到了终止符号
            # 或者生成的长度已经达到了最大值
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                self._remove_finished_progress(
                    seq,
                    finished_progress_entries_removed,
                )
        self._publish_slo_postprocess(
            decision_now_ns,
            step_end_ns,
            timestamp_valid,
            progress_updates,
            finished_progress_entries_removed,
        )
        self._maybe_reset_adaptive_mixed_controller()

    def _publish_slo_postprocess(
        self,
        decision_now_ns: int | None,
        step_end_ns: int | None,
        timestamp_valid: bool,
        progress_updates: dict[int, int],
        finished_progress_entries_removed: list[int],
    ) -> None:
        self._last_slo_postprocess = {
            "step_end_ns": step_end_ns,
            "actual_step_duration_ns": (
                step_end_ns - decision_now_ns
                if timestamp_valid
                else None
            ),
            "decode_progress_updates": {
                str(seq_id): timestamp
                for seq_id, timestamp in sorted(progress_updates.items())
            },
            "finished_progress_entries_removed": sorted(
                finished_progress_entries_removed
            ),
        }

    def _record_decode_progress(
        self,
        seq: Sequence,
        step_end_ns: int | None,
        progress_updates: dict[int, int],
    ) -> None:
        if not self.chunked_prefill_slo_mixed or step_end_ns is None:
            return
        self.decode_progress_ns_by_seq_id[seq.seq_id] = step_end_ns
        progress_updates[seq.seq_id] = step_end_ns

    def _remove_finished_progress(
        self,
        seq: Sequence,
        finished_progress_entries_removed: list[int],
    ) -> None:
        if self.decode_progress_ns_by_seq_id.pop(seq.seq_id, None) is not None:
            finished_progress_entries_removed.append(seq.seq_id)

    def _postprocess_chunked_prefill(
        self,
        seqs: list[Sequence],
        token_ids: list[int] | None,
        do_sample: bool,
        *,
        step_end_ns: int | None = None,
    ) -> tuple[dict[int, int], list[int]]:
        token_iter = iter(token_ids or [])
        progress_updates = {}
        finished_progress_entries_removed = []
        for seq in seqs:
            old_end = seq.num_computed_tokens
            new_end = max(seq.num_computed_tokens, seq.prefill_chunk_end)
            self.block_manager.commit_prefill(seq, old_end, new_end)
            seq.num_computed_tokens = new_end

            if not do_sample:
                seq.status = SequenceStatus.PREFILLING
                self.prefilling.append(seq)
                continue

            token_id = next(token_iter)
            seq.append_token(token_id)
            self._record_decode_progress(
                seq,
                step_end_ns,
                progress_updates,
            )
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self._remove_finished_progress(
                    seq,
                    finished_progress_entries_removed,
                )
            else:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
        return progress_updates, finished_progress_entries_removed

    def _postprocess_mixed(
        self,
        seqs: list[Sequence],
        token_ids: list[int] | None,
        *,
        step_end_ns: int | None = None,
    ) -> tuple[dict[int, int], list[int]]:
        token_iter = iter(token_ids or [])
        progress_updates = {}
        finished_progress_entries_removed = []
        for seq in seqs:
            if getattr(seq, "step_is_decode", False):
                token_id = next(token_iter)
                seq.append_token(token_id)
                self._record_decode_progress(
                    seq,
                    step_end_ns,
                    progress_updates,
                )
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self._remove_finished_progress(
                        seq,
                        finished_progress_entries_removed,
                    )
                else:
                    seq.status = SequenceStatus.RUNNING
                    self.running.append(seq)
                seq.step_is_decode = False
                seq.step_do_sample = True
                continue

            old_end = seq.num_computed_tokens
            new_end = max(seq.num_computed_tokens, seq.prefill_chunk_end)
            self.block_manager.commit_prefill(seq, old_end, new_end)
            seq.num_computed_tokens = new_end
            if not getattr(seq, "step_do_sample", True):
                seq.status = SequenceStatus.PREFILLING
                self.prefilling.append(seq)
            else:
                token_id = next(token_iter)
                seq.append_token(token_id)
                self._record_decode_progress(
                    seq,
                    step_end_ns,
                    progress_updates,
                )
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self._remove_finished_progress(
                        seq,
                        finished_progress_entries_removed,
                    )
                else:
                    seq.status = SequenceStatus.RUNNING
                    self.running.append(seq)
            seq.step_is_decode = False
            seq.step_do_sample = True
        return progress_updates, finished_progress_entries_removed
