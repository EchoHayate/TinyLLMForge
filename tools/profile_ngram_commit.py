"""S3/S4 profiler for accepted-token online n-gram speculation.

By default this preserves the S3 narrow smoke: it runs two identical greedy
requests in one TinyLLM instance. One request remains a normal baseline. The
other performs at most one n-gram speculative verify+commit event.

For S4, pass ``--max-commit-events 0`` to keep attempting online speculative
verify+commit events until the candidate finishes. The final candidate output
must still match the baseline request exactly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass, field

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_NGRAM_PATH = os.path.join(_REPO_ROOT, "tinyvllm", "speculative", "ngram.py")
_NGRAM_SPEC = importlib.util.spec_from_file_location("ngram_commit_profile", _NGRAM_PATH)
ngram = importlib.util.module_from_spec(_NGRAM_SPEC)
sys.modules["ngram_commit_profile"] = ngram
_NGRAM_SPEC.loader.exec_module(ngram)

count_accepted_prefix = ngram.count_accepted_prefix
propose_ngram_draft = ngram.propose_ngram_draft


DEFAULT_PROMPTS = [
    "Repeat the following phrase five times: alpha beta gamma alpha beta gamma.",
]


@dataclass
class DraftProposal:
    tokens: list[int]
    source: str
    metadata: dict = field(default_factory=dict)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--kv-offload-migration-smoke", action="store_true", default=False,
                   help="Run a synthetic KVOffloadMVP0 eviction+H2D reload correctness smoke without loading a model.")
    p.add_argument("--kv-offload-thrash-smoke", action="store_true", default=False,
                   help="Run a synthetic KV offload thrash smoke with repeated batched eviction/reload windows.")
    p.add_argument("--blockwise-attn-smoke", action="store_true", default=False,
                   help="Run an exact streaming/blockwise attention online-softmax smoke without loading a model.")
    p.add_argument("--blockwise-prefill-attn-smoke", action="store_true", default=False,
                   help="Run an exact streaming/blockwise chunked-prefill attention smoke without loading a model.")
    p.add_argument("--thrash-rounds", type=int, default=16)
    p.add_argument("--thrash-window-blocks", type=int, default=2)
    p.add_argument("--thrash-logical-blocks", type=int, default=8)
    p.add_argument("--thrash-gpu-blocks", type=int, default=3)
    p.add_argument("--blockwise-attn-batch", type=int, default=2)
    p.add_argument("--blockwise-attn-heads", type=int, default=4)
    p.add_argument("--blockwise-attn-kv-heads", type=int, default=2)
    p.add_argument("--blockwise-attn-head-dim", type=int, default=32)
    p.add_argument("--blockwise-attn-tokens", type=int, default=1024)
    p.add_argument("--blockwise-attn-window-tokens", type=int, default=128)
    p.add_argument("--blockwise-prefill-prefix-tokens", type=int, default=1024)
    p.add_argument("--blockwise-prefill-chunk-tokens", type=int, default=128)
    p.add_argument("--prompt", action="append", default=None,
                   help="Prompt to benchmark. Can be passed multiple times. Defaults to the S3/S4 single prompt.")
    p.add_argument("--max-output-len", type=int, default=64)
    p.add_argument("--warmup-output-len", type=int, default=0,
                   help="Run one untimed warmup request before measurement. This removes cold CUDA/kernel setup from timing.")
    p.add_argument("--simulate-kv-upload-mb", type=float, default=0.0,
                   help="Simulate CPU->GPU KV page upload cost per decode/verify target forward by copying this many MiB.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="S3 commit smoke currently requires greedy decoding, so this must be 0.0.")
    p.add_argument("--draft-source", type=str, default="ngram", choices=["ngram", "dflash-toy", "dflash-toy-ngram-or-repeat"],
                   help="Draft source for candidate-only/paired speculative verify+commit experiments.")
    p.add_argument("--ngram-size", type=int, default=3)
    p.add_argument("--dflash-toy-context-tokens", type=int, default=1,
                   help="Minimum trailing context tokens required before dflash-toy proposes a block.")
    p.add_argument("--max-draft-tokens", type=int, default=4)
    p.add_argument("--max-commit-events", type=int, default=1,
                   help="Maximum accepted commit events per candidate. Use 0 for unlimited S4 online benchmark.")
    p.add_argument("--allow-zero-accept", action="store_true", default=False,
                   help="Allow speculative plumbing smokes to pass even if no draft tokens are accepted.")
    p.add_argument("--debug-target-hidden", action="store_true", default=False,
                   help="Profiler-only debug: capture target hidden shape/dtype/device during verify+commit.")
    p.add_argument("--debug-hidden-to-draft-stub", action="store_true", default=False,
                   help="Profiler-only debug: attach a target-hidden-to-top-k draft adapter stub preview to verify events.")
    p.add_argument("--hidden-to-draft-adapter", type=str, default="topk-stub", choices=["topk-stub", "linear-stub"],
                   help="Profiler-only hidden-to-draft adapter stub to report when --debug-hidden-to-draft-stub is set.")
    p.add_argument("--debug-hidden-to-draft-top-k", type=int, default=3,
                   help="Top-k token count for --debug-hidden-to-draft-stub previews.")
    p.add_argument("--mode", type=str, default="paired",
                   choices=["paired", "baseline-only", "candidate-only"],
                   help="paired compares baseline+candidate in one engine; baseline-only/candidate-only are for separated timing.")
    p.add_argument("--out-json", type=str, default=None)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--enforce-eager", action="store_true", default=True)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--max-num-prefill-tokens-per-step", type=int, default=0,
                   help="Enable chunked prefill when >0; useful for KV offload blockwise long-context decode smoke.")
    p.add_argument("--quantization", type=str, default=None, choices=[None, "int8", "int4", "int2"])
    p.add_argument("--quant-group-size", type=int, default=128)
    p.add_argument("--act-quant-bits", type=int, default=0, choices=[0, 8])
    p.add_argument("--smoothquant-scale-path", type=str, default=None)
    p.add_argument("--act-quant-skip-first", type=int, default=0)
    p.add_argument("--act-quant-skip-last", type=int, default=0)
    p.add_argument("--kv-quant-bits", type=int, default=0, choices=[0, 4, 8])
    p.add_argument("--kv-quant-group-size", type=int, default=128)
    p.add_argument("--kv-offload-mvp0", action="store_true", default=False,
                   help="Enable real KV offload MVP-0: fp16/bf16 KV, full attention, eager logical->physical GPU slot remap.")
    p.add_argument("--kv-offload-gpu-blocks", type=int, default=0,
                   help="GPU staging KV blocks for --kv-offload-mvp0. 0 uses the auto-computed KV block count.")
    p.add_argument("--kv-offload-logical-blocks", type=int, default=0,
                   help="Logical KV blocks backed by CPU pinned memory for --kv-offload-mvp0. 0 uses the auto-computed KV block count.")
    p.add_argument("--kv-offload-no-async-copy", dest="kv_offload_async_copy", action="store_false", default=True,
                   help="Disable KV offload copy stream/event for debugging.")
    p.add_argument("--kv-offload-no-batch-copy", dest="kv_offload_batch_copy", action="store_false", default=True,
                   help="Disable KV offload contiguous span batching for debugging.")
    p.add_argument("--kv-offload-writeback-on-evict", action="store_true", default=False,
                   help="Delay dirty KV D2H writeback until eviction instead of after every forward.")
    p.add_argument("--kv-offload-evict-policy", type=str, default="lru_cost", choices=["lru", "lru_cost"])
    p.add_argument("--kv-offload-blockwise-decode", action="store_true", default=False,
                   help="Use exact blockwise decode attention for KV offload so visible blocks may exceed staging slots.")
    p.add_argument("--kv-offload-blockwise-prefill", action="store_true", default=False,
                   help="Use exact blockwise chunked prefill attention for KV offload so prefix blocks may exceed staging slots.")
    p.add_argument("--kv-offload-blockwise-blocks", type=int, default=1,
                   help="Logical KV blocks per blockwise KV offload attention window.")
    p.add_argument("--quest-top-k-blocks", type=int, default=-1)
    p.add_argument("--quest-min-seq-len", type=int, default=512)
    return p.parse_args()


def cuda_sync_if_available():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def propose_draft(history: list[int], args) -> DraftProposal:
    if args.draft_source == "ngram":
        draft = propose_ngram_draft(history, args.ngram_size, args.max_draft_tokens)
        return DraftProposal(
            tokens=list(draft.tokens),
            source="ngram",
            metadata={
                "match_start": draft.match_start,
                "ngram_size": draft.ngram_size,
            },
        )
    if args.draft_source == "dflash-toy-ngram-or-repeat":
        draft = propose_ngram_draft(history, args.ngram_size, args.max_draft_tokens)
        if draft.tokens:
            return DraftProposal(
                tokens=list(draft.tokens),
                source="dflash-toy-ngram-or-repeat",
                metadata={
                    "toy_strategy": "ngram_or_repeat",
                    "selected_strategy": "ngram",
                    "match_start": draft.match_start,
                    "ngram_size": draft.ngram_size,
                },
            )
    if args.draft_source in ("dflash-toy", "dflash-toy-ngram-or-repeat"):
        context_tokens = max(1, int(args.dflash_toy_context_tokens))
        if args.max_draft_tokens <= 0 or len(history) < context_tokens:
            return DraftProposal(
                tokens=[],
                source=args.draft_source,
                metadata={"reason": "insufficient_history", "context_tokens": context_tokens},
            )
        window = list(history[-max(1, min(len(history), args.max_draft_tokens)):])
        tokens = []
        while len(tokens) < args.max_draft_tokens:
            tokens.extend(window)
        return DraftProposal(
            tokens=tokens[:args.max_draft_tokens],
            source=args.draft_source,
            metadata={
                "toy_strategy": "repeat_recent_tokens" if args.draft_source == "dflash-toy" else "ngram_or_repeat",
                "selected_strategy": "repeat_recent_tokens",
                "context_tokens": context_tokens,
                "window_tokens": len(window),
            },
        )
    raise ValueError(f"unsupported draft_source={args.draft_source}")


def summarize_hidden_to_draft_stub(hidden_states, logits, top_k: int = 3, adapter: str = "topk-stub") -> dict:
    """Return a JSON-friendly hidden-to-draft adapter interface preview.

    This does not sample or alter runtime behavior. It only records what a
    future hidden-to-draft adapter could inspect and return: target hidden
    metadata, explicit input/output schema, adapter timing, and a top-k logits
    preview for each verify row.
    """
    total_t0 = time.perf_counter()
    top_k = max(1, int(top_k))
    if adapter not in ("topk-stub", "linear-stub"):
        raise ValueError(f"unsupported hidden-to-draft adapter={adapter}")
    t0 = time.perf_counter()
    if hasattr(logits, "detach"):
        rows = logits.detach().float().cpu().tolist()
    else:
        rows = [[float(value) for value in row] for row in logits]
    logits_to_cpu_ms = (time.perf_counter() - t0) * 1000.0
    hidden_rows = None
    hidden_to_cpu_ms = 0.0
    if adapter == "linear-stub":
        t0 = time.perf_counter()
        if hasattr(hidden_states, "detach"):
            hidden_rows = hidden_states.detach().float().cpu().tolist()
        elif hasattr(hidden_states, "values"):
            hidden_rows = [[float(value) for value in row] for row in hidden_states.values]
        else:
            hidden_rows = [[float(value) for value in row] for row in hidden_states]
        hidden_to_cpu_ms = (time.perf_counter() - t0) * 1000.0
    preview = []
    t0 = time.perf_counter()
    projection_metadata = None
    if adapter == "linear-stub":
        hidden_dim = len(hidden_rows[0]) if hidden_rows else 0
        candidate_count = min(8, len(rows[0]) if rows else hidden_dim)
        candidate_token_ids = list(range(max(0, candidate_count)))
        projection_metadata = {
            "seed": 17,
            "candidate_token_ids": candidate_token_ids,
            "hidden_dim": hidden_dim,
            "candidate_count": candidate_count,
        }
        for row_index, hidden_row in enumerate(hidden_rows or []):
            scores = []
            for candidate_index, token_id in enumerate(candidate_token_ids):
                score = 0.0
                for dim_index, value in enumerate(hidden_row):
                    weight = (((dim_index + 1) * (candidate_index + 3) + 17) % 11 - 5) / 4.0
                    score += float(value) * weight
                scores.append((token_id, score))
            ranked = sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]
            preview.append({
                "row": int(row_index),
                "token_ids": [int(token_id) for token_id, _ in ranked],
                "scores": [float(score) for _, score in ranked],
            })
    else:
        for row_index, row in enumerate(rows):
            ranked = sorted(enumerate(row), key=lambda item: item[1], reverse=True)[:top_k]
            preview.append({
                "row": int(row_index),
                "token_ids": [int(token_id) for token_id, _ in ranked],
                "scores": [float(score) for _, score in ranked],
            })
    topk_ms = (time.perf_counter() - t0) * 1000.0
    linear_projection_ms = 0.0
    adapter_name = "target_hidden_topk_stub"
    if adapter == "linear-stub":
        linear_projection_ms = topk_ms
        adapter_name = "target_hidden_linear_stub"
    first_tokens = [item["token_ids"][0] for item in preview if item["token_ids"]]
    first_scores = [item["scores"][0] for item in preview if item["scores"]]
    row_count = len(preview)
    vocab_preview = len(rows[0]) if rows else 0
    hidden_schema = {
        "shape": [int(dim) for dim in hidden_states.shape],
        "dtype": str(hidden_states.dtype),
        "device": str(hidden_states.device),
    }
    return {
        "interface_version": 1,
        "adapter": adapter_name,
        "runtime_mutation": False,
        "input_schema": {
            "hidden_states": hidden_schema,
            "logits": {
                "shape": [row_count, vocab_preview],
                "dtype": "float32_preview",
                "device": "cpu_preview",
            },
            "adapter": adapter,
            "top_k": top_k,
        },
        "output_schema": {
            "draft_token_ids": "list[int]",
            "draft_scores": "list[float]",
            "num_rows": "int",
            "source": "profiler_only_hidden_to_draft_adapter",
            "projection": "deterministic_hidden_linear_stub" if adapter == "linear-stub" else "logits_topk",
        },
        "output": {
            "draft_token_ids": first_tokens,
            "draft_scores": first_scores,
            "num_rows": row_count,
            "source": adapter_name,
        },
        "timing_ms": {
            "adapter_total_ms": (time.perf_counter() - total_t0) * 1000.0,
            "logits_to_cpu_ms": logits_to_cpu_ms,
            **({"hidden_to_cpu_ms": hidden_to_cpu_ms, "linear_projection_ms": linear_projection_ms} if adapter == "linear-stub" else {}),
            "topk_ms": topk_ms,
        },
        **({"projection_metadata": projection_metadata} if projection_metadata is not None else {}),
        "shape": hidden_schema["shape"],
        "dtype": hidden_schema["dtype"],
        "device": hidden_schema["device"],
        "top_k": top_k,
        "rows": row_count,
        "preview": preview,
    }


def _simulate_kv_upload(llm, mb: float) -> float:
    if mb <= 0:
        return 0.0
    import torch

    nbytes = int(mb * 1024 * 1024)
    if nbytes <= 0:
        return 0.0
    if not hasattr(llm, "_sim_kv_upload") or llm._sim_kv_upload.get("nbytes") != nbytes:
        host = torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=True)
        device = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        host.fill_(1)
        llm._sim_kv_upload = {"nbytes": nbytes, "host": host, "device": device}
    state = llm._sim_kv_upload
    t0 = time.perf_counter()
    state["device"].copy_(state["host"], non_blocking=True)
    cuda_sync_if_available()
    return (time.perf_counter() - t0) * 1000.0


def _find_running_seq(llm, seq_id: int):
    for seq in llm.scheduler.running:
        if seq.seq_id == seq_id:
            return seq
    return None


def _finish_if_needed(llm, seq, committed_tokens: list[int]) -> bool:
    if not committed_tokens:
        return False
    saw_eos = (not seq.ignore_eos) and any(int(token_id) == int(llm.scheduler.eos) for token_id in committed_tokens)
    if saw_eos or seq.num_completion_tokens >= seq.max_tokens:
        seq.status = type(seq.status).FINISHED
        llm.scheduler.block_manager.deallocate(seq)
        try:
            llm.scheduler.running.remove(seq)
        except ValueError:
            pass
        return True
    return False


def verify_and_commit_block(
    llm,
    seq,
    draft_tokens: list[int],
    *,
    draft_source: str = "unknown",
    simulate_kv_upload_mb: float = 0.0,
    debug_target_hidden: bool = False,
    debug_hidden_to_draft_stub: bool = False,
    hidden_to_draft_adapter: str = "topk-stub",
    debug_hidden_to_draft_top_k: int = 3,
) -> dict:
    """Verify a speculative draft block with the target model and commit accepted tokens.

    This is intentionally draft-source agnostic so n-gram, toy DFlash-style
    block drafts, or a future real DFlash draft model can share the same target
    verification and KV metadata path.
    """
    import torch
    from tinyvllm.utils.context import reset_context, set_context

    total_t0 = time.perf_counter()
    history_len = len(seq)
    block_manager = llm.scheduler.block_manager
    t0 = time.perf_counter()
    reserved_blocks = block_manager.reserve_append_blocks(seq, len(draft_tokens))
    timing_ms = {
        "reserve_blocks_ms": (time.perf_counter() - t0) * 1000.0,
    }

    try:
        t0 = time.perf_counter()
        input_tokens = [seq.last_token] + list(draft_tokens)
        query_len = len(input_tokens)
        proxy_block_table = list(seq.block_table) + list(reserved_blocks)
        slot_positions = list(range(history_len - 1, history_len + len(draft_tokens)))
        dirty_blocks = [proxy_block_table[pos // seq.block_size] for pos in slot_positions]
        if getattr(llm.model_runner, "kv_offload", None) is not None:
            first_write_offset_by_block = {}
            for pos in slot_positions:
                block_id = proxy_block_table[pos // seq.block_size]
                offset = pos % seq.block_size
                first_write_offset_by_block[block_id] = min(
                    offset, first_write_offset_by_block.get(block_id, offset))
            valid_read_blocks = [
                block_id for block_id in seq.block_table
                if first_write_offset_by_block.get(block_id, 1) > 0
            ]
            future_blocks = set(int(block_id) for block_id in proxy_block_table)
            future_blocks.update(int(block_id) for block_id in dirty_blocks)
            llm.model_runner.kv_offload.stats["prefetch_plans"] += 1
            llm.model_runner.kv_offload.stats["prefetch_read_blocks"] += len(set(valid_read_blocks))
            llm.model_runner.kv_offload.stats["prefetch_write_blocks"] += len(set(dirty_blocks))
            llm.model_runner.kv_offload.ensure_resident(
                valid_read_blocks,
                require_valid=True,
                future_logical_blocks=future_blocks,
            )
            llm.model_runner.kv_offload.ensure_resident(
                dirty_blocks,
                require_valid=False,
                future_logical_blocks=future_blocks,
            )
            physical_proxy_block_table = [
                llm.model_runner.kv_offload.logical_to_slot[int(block_id)]
                for block_id in proxy_block_table
            ]
        else:
            physical_proxy_block_table = proxy_block_table
        slot_mapping_data = [
            physical_proxy_block_table[pos // seq.block_size] * seq.block_size + (pos % seq.block_size)
            for pos in slot_positions
        ]
        input_ids = llm.model_runner._list_to_cuda(input_tokens, "commit_input_ids", torch.int64)
        # Match TinyLLM's decode convention: the last token of a length-H
        # sequence is evaluated with position H, not H-1.
        positions = llm.model_runner._list_to_cuda(
            list(range(history_len, history_len + query_len)), "commit_positions", torch.int64)
        cu_seqlens_q = llm.model_runner._list_to_cuda([0, query_len], "commit_cu_seqlens_q", torch.int32)
        cu_seqlens_k = llm.model_runner._list_to_cuda(
            [0, history_len + len(draft_tokens)], "commit_cu_seqlens_k", torch.int32)
        slot_mapping = llm.model_runner._list_to_cuda(slot_mapping_data, "commit_slot_mapping", torch.int32)
        block_tables = llm.model_runner.prepare_block_tables_from_rows([physical_proxy_block_table], "commit_block_tables")
        logits_indices = llm.model_runner._list_to_cuda(
            list(range(len(draft_tokens))), "commit_logits_indices", torch.int64)
        set_context(True, cu_seqlens_q, cu_seqlens_k, query_len, history_len + len(draft_tokens),
                    slot_mapping, None, block_tables, logits_indices)
        cuda_sync_if_available()
        timing_ms["verify_prepare_ms"] = (time.perf_counter() - t0) * 1000.0

        timing_ms["simulated_kv_upload_ms"] = _simulate_kv_upload(llm, simulate_kv_upload_mb)

        t0 = time.perf_counter()
        hidden_debug = None
        hidden_to_draft_stub = None
        if getattr(llm.model_runner, "kv_offload", None) is not None:
            llm.model_runner._kv_offload_before_forward()
        if debug_target_hidden or debug_hidden_to_draft_stub:
            logits, hidden_states = llm.model_runner.run_model(
                input_ids, positions, is_prefill=True, return_hidden=True)
            hidden_debug = {
                "shape": [int(dim) for dim in hidden_states.shape],
                "dtype": str(hidden_states.dtype),
                "device": str(hidden_states.device),
            }
            if debug_hidden_to_draft_stub:
                hidden_to_draft_stub = summarize_hidden_to_draft_stub(
                    hidden_states, logits, debug_hidden_to_draft_top_k, hidden_to_draft_adapter)
        else:
            logits = llm.model_runner.run_model(input_ids, positions, is_prefill=True)
        cuda_sync_if_available()
        if getattr(llm.model_runner, "kv_offload", None) is not None:
            llm.model_runner.kv_offload.mark_dirty(dirty_blocks)
            if not llm.model_runner.kv_offload.writeback_on_evict:
                llm.model_runner.kv_offload.writeback_dirty(dirty_blocks)
        timing_ms["target_forward_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        target_tokens = [int(token_id) for token_id in logits.argmax(dim=-1).tolist()]
        accepted = count_accepted_prefix(draft_tokens, target_tokens)
        accepted_tokens = list(draft_tokens[:accepted])
        if not seq.ignore_eos and llm.scheduler.eos in accepted_tokens:
            eos_index = accepted_tokens.index(llm.scheduler.eos)
            accepted_tokens = accepted_tokens[:eos_index + 1]
        remaining_budget = max(0, seq.max_tokens - seq.num_completion_tokens)
        accepted_tokens = accepted_tokens[:remaining_budget]
        timing_ms["accept_sample_ms"] = (time.perf_counter() - t0) * 1000.0

        event_reserved_blocks = list(reserved_blocks)
        t0 = time.perf_counter()
        block_manager.commit_accepted_tokens(seq, accepted_tokens, reserved_blocks)
        reserved_blocks = []
        timing_ms["commit_metadata_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        finished = _finish_if_needed(llm, seq, accepted_tokens)
        timing_ms["finish_check_ms"] = (time.perf_counter() - t0) * 1000.0
        timing_ms["verify_commit_total_ms"] = (time.perf_counter() - total_t0) * 1000.0
        return {
            "draft_source": draft_source,
            "history_len": history_len,
            "draft_tokens": list(draft_tokens),
            "target_tokens": target_tokens,
            "accepted_tokens": accepted_tokens,
            "accepted_count": len(accepted_tokens),
            "reserved_blocks": event_reserved_blocks,
            "block_table_after": list(seq.block_table),
            "num_tokens_after": seq.num_tokens,
            "last_token_after": int(seq.last_token),
            "finished": finished,
            "timing_ms": timing_ms,
            "target_hidden_debug": hidden_debug,
            "hidden_to_draft_stub": hidden_to_draft_stub,
        }
    except Exception:
        block_manager.release_reserved_blocks(reserved_blocks)
        raise
    finally:
        reset_context()


def _target_verify_and_commit(llm, seq, draft_tokens: list[int], simulate_kv_upload_mb: float = 0.0) -> dict:
    return verify_and_commit_block(
        llm,
        seq,
        draft_tokens,
        draft_source="ngram",
        simulate_kv_upload_mb=simulate_kv_upload_mb,
    )


def _create_llm(args):
    from tinyvllm import LLM, SamplingParams

    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_prefill_tokens_per_step=args.max_num_prefill_tokens_per_step,
        quantization=args.quantization,
        quant_group_size=args.quant_group_size,
        act_quant_bits=args.act_quant_bits,
        smoothquant_scale_path=args.smoothquant_scale_path,
        act_quant_skip_first=args.act_quant_skip_first,
        act_quant_skip_last=args.act_quant_skip_last,
        kv_quant_bits=args.kv_quant_bits,
        kv_quant_group_size=args.kv_quant_group_size,
        kv_offload_mvp0=args.kv_offload_mvp0,
        kv_offload_gpu_blocks=args.kv_offload_gpu_blocks,
        kv_offload_logical_blocks=args.kv_offload_logical_blocks,
        kv_offload_async_copy=args.kv_offload_async_copy,
        kv_offload_batch_copy=args.kv_offload_batch_copy,
        kv_offload_writeback_on_evict=args.kv_offload_writeback_on_evict,
        kv_offload_evict_policy=args.kv_offload_evict_policy,
        kv_offload_blockwise_decode=args.kv_offload_blockwise_decode,
        kv_offload_blockwise_prefill=args.kv_offload_blockwise_prefill,
        kv_offload_blockwise_blocks=args.kv_offload_blockwise_blocks,
        quest_top_k_blocks=args.quest_top_k_blocks,
        quest_min_seq_len=args.quest_min_seq_len,
    )
    prompts = args.prompt or DEFAULT_PROMPTS
    sp = SamplingParams(temperature=args.temperature, ignore_eos=False, max_tokens=args.max_output_len)
    return llm, prompts, sp


def _run_warmup(llm, args, prompts: list[str]) -> dict | None:
    if args.warmup_output_len <= 0:
        return None

    from tinyvllm import SamplingParams

    prompt = prompts[0] if prompts else DEFAULT_PROMPTS[0]
    sp = SamplingParams(temperature=args.temperature, ignore_eos=False, max_tokens=args.warmup_output_len)
    llm.add_request(prompt, sp)
    t0 = time.perf_counter()
    cuda_sync_if_available()
    steps = 0
    output_tokens = 0
    while not llm.is_finished():
        out, _ = llm.step()
        cuda_sync_if_available()
        steps += 1
        for _, token_ids in out:
            output_tokens = max(output_tokens, len(token_ids))
    return {
        "warmup_output_len": args.warmup_output_len,
        "warmup_steps": steps,
        "warmup_output_tokens": output_tokens,
        "warmup_elapsed_s": time.perf_counter() - t0,
    }


def _base_summary(args, prompts: list[str], elapsed_s: float, step_records: list[dict]) -> dict:
    return {
        "mode": args.mode,
        "num_prompts": len(prompts),
        "max_output_len": args.max_output_len,
        "decode_steps": sum(1 for record in step_records if record["num_tokens"] < 0),
        "elapsed_s": elapsed_s,
    }


def _accumulate_timing_ms(total: dict, event: dict):
    for key, value in event.get("timing_ms", {}).items():
        total[key] = total.get(key, 0.0) + float(value)


def _sum_timing_ms(events: list[dict]) -> dict:
    total = {}
    for event in events:
        _accumulate_timing_ms(total, event)
    return total


def _summarize_simulated_upload(
    args,
    step_records: list[dict],
    verify_timing_ms: dict | None = None,
    verify_upload_events: int = 0,
) -> dict:
    """Aggregate the synthetic KV upload pressure metrics into summary fields.

    The profiler injects upload copies in two places: normal decode steps and
    target-verify forwards. Keeping both totals in the top-level summary makes
    separated baseline-only/candidate-only JSONs directly comparable without
    post-processing the full step/event traces.
    """
    verify_timing_ms = verify_timing_ms or {}
    simulate_enabled = args.simulate_kv_upload_mb > 0
    normal_upload_ms = sum(float(record.get("simulated_kv_upload_ms", 0.0)) for record in step_records)
    verify_upload_ms = float(verify_timing_ms.get("simulated_kv_upload_ms", 0.0))
    normal_upload_events = (
        sum(1 for record in step_records if record.get("num_tokens", 0) < 0)
        if simulate_enabled else 0
    )
    verify_upload_events = verify_upload_events if simulate_enabled else 0
    total_upload_events = normal_upload_events + verify_upload_events
    return {
        "simulate_kv_upload_mb": args.simulate_kv_upload_mb,
        "normal_decode_simulated_kv_upload_ms": normal_upload_ms,
        "verify_simulated_kv_upload_ms": verify_upload_ms,
        "total_simulated_kv_upload_ms": normal_upload_ms + verify_upload_ms,
        "normal_decode_simulated_kv_upload_events": normal_upload_events,
        "verify_simulated_kv_upload_events": verify_upload_events,
        "total_simulated_kv_upload_events": total_upload_events,
        "normal_decode_simulated_kv_upload_mib": normal_upload_events * args.simulate_kv_upload_mb,
        "verify_simulated_kv_upload_mib": verify_upload_events * args.simulate_kv_upload_mb,
        "total_simulated_kv_upload_mib": total_upload_events * args.simulate_kv_upload_mb,
    }


def run_kv_offload_migration_smoke() -> dict:
    """Synthetic eviction/reload test for KVOffloadMVP0.

    This intentionally bypasses the full LLM so it can force page migration with
    a tiny cache: write logical blocks 0/1, evict them with 2/3, then reload 0/1
    from CPU pinned backing store and verify byte-for-byte content.
    """
    import torch
    from tinyvllm.engine.model_runner import KVOffloadMVP0

    if not torch.cuda.is_available():
        raise RuntimeError("--kv-offload-migration-smoke requires CUDA")

    torch.cuda.set_device(0)
    kv_cache = torch.empty(2, 1, 2, 4, 1, 2, dtype=torch.float16, device="cuda")
    manager = KVOffloadMVP0(kv_cache, logical_blocks=4, block_size=4)

    expected = {}
    for logical_block in (0, 1):
        manager.ensure_resident([logical_block], require_valid=False)
        slot = manager.logical_to_slot[logical_block]
        value = float(logical_block + 1)
        kv_cache[:, :, slot].fill_(value)
        expected[logical_block] = torch.full_like(kv_cache[:, :, slot], value).cpu()
    manager.mark_dirty([0, 1])
    manager.writeback_dirty([0, 1])

    for logical_block in (2, 3):
        manager.ensure_resident([logical_block], require_valid=False)
        slot = manager.logical_to_slot[logical_block]
        value = float(logical_block + 1)
        kv_cache[:, :, slot].fill_(value)
        expected[logical_block] = torch.full_like(kv_cache[:, :, slot], value).cpu()
    manager.mark_dirty([2, 3])
    manager.writeback_dirty([2, 3])

    manager.ensure_resident([0, 1], require_valid=True, wait=True)
    manager.synchronize_copies()
    mismatches = []
    reloaded_slots = {}
    for logical_block in (0, 1):
        slot = manager.logical_to_slot[logical_block]
        reloaded_slots[logical_block] = slot
        got = kv_cache[:, :, slot].cpu()
        if not torch.equal(got, expected[logical_block]):
            mismatches.append(logical_block)

    summary = manager.summary()
    gate_fail_reasons = []
    if mismatches:
        gate_fail_reasons.append(f"content_mismatch={mismatches}")
    if summary["evictions"] < 4:
        gate_fail_reasons.append(f"evictions<{4}: {summary['evictions']}")
    if summary["h2d_copies"] < 2:
        gate_fail_reasons.append(f"h2d_copies<{2}: {summary['h2d_copies']}")
    if summary["d2h_copies"] < 4:
        gate_fail_reasons.append(f"d2h_copies<{4}: {summary['d2h_copies']}")
    return {
        "mode": "kv-offload-migration-smoke",
        "summary": {
            "gate_pass": not gate_fail_reasons,
            "gate_fail_reasons": gate_fail_reasons,
            **summary,
        },
        "reloaded_slots": reloaded_slots,
        "slot_to_logical": manager.slot_to_logical,
        "logical_to_slot": manager.logical_to_slot,
    }


def run_kv_offload_thrash_smoke(args) -> dict:
    """Synthetic MVP-1 thrash test with repeated batched eviction/reload windows."""
    import torch
    from tinyvllm.engine.model_runner import KVOffloadMVP0

    if not torch.cuda.is_available():
        raise RuntimeError("--kv-offload-thrash-smoke requires CUDA")
    if args.thrash_window_blocks > args.thrash_gpu_blocks:
        raise ValueError("--thrash-window-blocks must be <= --thrash-gpu-blocks")
    if args.thrash_logical_blocks < args.thrash_gpu_blocks:
        raise ValueError("--thrash-logical-blocks must be >= --thrash-gpu-blocks")

    torch.cuda.set_device(0)
    kv_cache = torch.empty(2, 1, args.thrash_gpu_blocks, 4, 1, 2, dtype=torch.float16, device="cuda")
    manager = KVOffloadMVP0(
        kv_cache,
        logical_blocks=args.thrash_logical_blocks,
        block_size=4,
        async_copy=args.kv_offload_async_copy,
        batch_copy=args.kv_offload_batch_copy,
        writeback_on_evict=args.kv_offload_writeback_on_evict,
        evict_policy=args.kv_offload_evict_policy,
    )

    expected = {}
    for start in range(0, args.thrash_logical_blocks, args.thrash_gpu_blocks):
        blocks = list(range(start, min(start + args.thrash_gpu_blocks, args.thrash_logical_blocks)))
        manager.ensure_resident(blocks, require_valid=False, future_logical_blocks=set(blocks), wait=True)
        for logical_block in blocks:
            slot = manager.logical_to_slot[logical_block]
            value = float(logical_block + 1)
            kv_cache[:, :, slot].fill_(value)
            expected[logical_block] = torch.full_like(kv_cache[:, :, slot], value).cpu()
        manager.mark_dirty(blocks)
        manager.writeback_dirty(blocks)
        manager.synchronize_copies()

    windows = []
    for round_idx in range(args.thrash_rounds):
        start = (round_idx * args.thrash_window_blocks) % args.thrash_logical_blocks
        window = [(start + offset) % args.thrash_logical_blocks for offset in range(args.thrash_window_blocks)]
        windows.append(window)

    mismatches = []
    for window in windows:
        manager.stats["prefetch_plans"] += 1
        manager.stats["prefetch_read_blocks"] += len(set(window))
        manager.ensure_resident(window, require_valid=True, future_logical_blocks=set(window), wait=True)
        manager.synchronize_copies()
        for logical_block in window:
            slot = manager.logical_to_slot[logical_block]
            got = kv_cache[:, :, slot].cpu()
            if not torch.equal(got, expected[logical_block]):
                mismatches.append(logical_block)

    summary = manager.summary()
    gate_fail_reasons = []
    if mismatches:
        gate_fail_reasons.append(f"content_mismatch={sorted(set(mismatches))}")
    if summary["evictions"] <= 0:
        gate_fail_reasons.append("evictions<=0")
    if summary["h2d_copies"] <= 0:
        gate_fail_reasons.append("h2d_copies<=0")
    if summary["d2h_copies"] <= 0:
        gate_fail_reasons.append("d2h_copies<=0")
    if args.kv_offload_batch_copy and summary["h2d_batches"] <= 0:
        gate_fail_reasons.append("h2d_batches<=0")

    return {
        "mode": "kv-offload-thrash-smoke",
        "args": {
            "thrash_rounds": args.thrash_rounds,
            "thrash_window_blocks": args.thrash_window_blocks,
            "thrash_logical_blocks": args.thrash_logical_blocks,
            "thrash_gpu_blocks": args.thrash_gpu_blocks,
            "kv_offload_async_copy": args.kv_offload_async_copy,
            "kv_offload_batch_copy": args.kv_offload_batch_copy,
            "kv_offload_writeback_on_evict": args.kv_offload_writeback_on_evict,
            "kv_offload_evict_policy": args.kv_offload_evict_policy,
        },
        "summary": {
            "gate_pass": not gate_fail_reasons,
            "gate_fail_reasons": gate_fail_reasons,
            **summary,
        },
        "windows": windows,
        "slot_to_logical": manager.slot_to_logical,
        "logical_to_slot": manager.logical_to_slot,
    }


def _repeat_kv_heads_for_gqa(kv, num_heads: int):
    if kv.size(2) == num_heads:
        return kv
    if num_heads % kv.size(2) != 0:
        raise ValueError(f"num_heads={num_heads} must be divisible by num_kv_heads={kv.size(2)}")
    return kv.repeat_interleave(num_heads // kv.size(2), dim=2)


def _full_decode_attention(q, k, v, context_lens, scale: float):
    import torch

    k = _repeat_kv_heads_for_gqa(k, q.size(1)).to(torch.float32)
    v = _repeat_kv_heads_for_gqa(v, q.size(1)).to(torch.float32)
    q = q.to(torch.float32)
    scores = torch.einsum("bhd,bthd->bht", q, k) * scale
    positions = torch.arange(k.size(1), device=k.device).view(1, 1, -1)
    mask = positions < context_lens.view(-1, 1, 1)
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bht,bthd->bhd", probs, v)


def _blockwise_decode_attention(q, k, v, context_lens, scale: float, window_tokens: int):
    import torch

    k = _repeat_kv_heads_for_gqa(k, q.size(1)).to(torch.float32)
    v = _repeat_kv_heads_for_gqa(v, q.size(1)).to(torch.float32)
    q = q.to(torch.float32)
    batch, num_heads, head_dim = q.shape
    max_tokens = k.size(1)
    running_m = torch.full((batch, num_heads), float("-inf"), device=q.device, dtype=torch.float32)
    running_l = torch.zeros((batch, num_heads), device=q.device, dtype=torch.float32)
    running_o = torch.zeros((batch, num_heads, head_dim), device=q.device, dtype=torch.float32)
    chunks = 0
    streamed_tokens = 0

    for start in range(0, max_tokens, window_tokens):
        end = min(start + window_tokens, max_tokens)
        k_chunk = k[:, start:end]
        v_chunk = v[:, start:end]
        scores = torch.einsum("bhd,bthd->bht", q, k_chunk) * scale
        positions = torch.arange(start, end, device=q.device).view(1, 1, -1)
        mask = positions < context_lens.view(-1, 1, 1)
        valid_chunk = mask.any(dim=-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        chunk_m = scores.max(dim=-1).values
        chunk_m_safe = torch.where(valid_chunk, chunk_m, torch.zeros_like(chunk_m))
        exp_scores = torch.exp(scores - chunk_m_safe.unsqueeze(-1)).masked_fill(~mask, 0.0)
        chunk_l = exp_scores.sum(dim=-1)
        chunk_o = torch.einsum("bht,bthd->bhd", exp_scores, v_chunk)

        merged_m = torch.maximum(running_m, chunk_m)
        merged_m = torch.where(valid_chunk, merged_m, running_m)
        old_weight = torch.exp(running_m - merged_m).masked_fill(torch.isneginf(running_m), 0.0)
        new_weight = torch.exp(chunk_m - merged_m).masked_fill(~valid_chunk, 0.0)
        merged_l = old_weight * running_l + new_weight * chunk_l
        merged_o = old_weight.unsqueeze(-1) * running_o + new_weight.unsqueeze(-1) * chunk_o

        running_m = merged_m
        running_l = merged_l
        running_o = merged_o
        chunks += 1
        streamed_tokens += end - start

    return running_o / running_l.clamp_min(1e-20).unsqueeze(-1), chunks, streamed_tokens


def _full_prefill_attention(q, k, v, chunk_starts, chunk_lens, scale: float):
    import torch

    k = _repeat_kv_heads_for_gqa(k, q.size(2)).to(torch.float32)
    v = _repeat_kv_heads_for_gqa(v, q.size(2)).to(torch.float32)
    q = q.to(torch.float32)
    batch, max_q, num_heads, _ = q.shape
    max_k = k.size(1)
    out = torch.zeros_like(q)
    key_positions = torch.arange(max_k, device=q.device).view(1, 1, max_k)
    for b in range(batch):
        q_len = int(chunk_lens[b])
        chunk_start = int(chunk_starts[b])
        if q_len <= 0:
            continue
        scores = torch.einsum("qhd,thd->qht", q[b, :q_len], k[b]) * scale
        query_positions = torch.arange(chunk_start, chunk_start + q_len, device=q.device).view(q_len, 1, 1)
        mask = (key_positions[:, :, :max_k] <= query_positions) & (key_positions[:, :, :max_k] < chunk_start + q_len)
        scores = scores.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[b, :q_len] = torch.einsum("qht,thd->qhd", probs, v[b])
    return out


def _blockwise_prefill_attention(q, k, v, chunk_starts, chunk_lens, scale: float, window_tokens: int):
    import torch

    k = _repeat_kv_heads_for_gqa(k, q.size(2)).to(torch.float32)
    v = _repeat_kv_heads_for_gqa(v, q.size(2)).to(torch.float32)
    q = q.to(torch.float32)
    batch, max_q, num_heads, head_dim = q.shape
    out = torch.zeros_like(q)
    chunks = 0
    streamed_tokens = 0

    def merge(running_m, running_l, running_o, scores, values, mask):
        valid = mask.any(dim=-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        chunk_m = scores.max(dim=-1).values
        chunk_m_safe = torch.where(valid, chunk_m, torch.zeros_like(chunk_m))
        exp_scores = torch.exp(scores - chunk_m_safe.unsqueeze(-1)).masked_fill(~mask, 0.0)
        chunk_l = exp_scores.sum(dim=-1)
        chunk_o = torch.einsum("qht,thd->qhd", exp_scores, values)
        merged_m = torch.maximum(running_m, chunk_m)
        merged_m = torch.where(valid, merged_m, running_m)
        old_weight = torch.exp(running_m - merged_m).masked_fill(torch.isneginf(running_m), 0.0)
        new_weight = torch.exp(chunk_m - merged_m).masked_fill(~valid, 0.0)
        running_l = old_weight * running_l + new_weight * chunk_l
        running_o = old_weight.unsqueeze(-1) * running_o + new_weight.unsqueeze(-1) * chunk_o
        running_m = merged_m
        return running_m, running_l, running_o

    for b in range(batch):
        q_len = int(chunk_lens[b])
        chunk_start = int(chunk_starts[b])
        chunk_end = chunk_start + q_len
        if q_len <= 0:
            continue
        running_m = torch.full((q_len, num_heads), float("-inf"), device=q.device, dtype=torch.float32)
        running_l = torch.zeros((q_len, num_heads), device=q.device, dtype=torch.float32)
        running_o = torch.zeros((q_len, num_heads, head_dim), device=q.device, dtype=torch.float32)
        for start in range(0, chunk_start, window_tokens):
            end = min(start + window_tokens, chunk_start)
            k_window = k[b, start:end]
            v_window = v[b, start:end]
            scores = torch.einsum("qhd,thd->qht", q[b, :q_len], k_window) * scale
            mask = torch.ones((q_len, 1, end - start), device=q.device, dtype=torch.bool)
            running_m, running_l, running_o = merge(running_m, running_l, running_o, scores, v_window, mask)
            chunks += 1
            streamed_tokens += end - start
        k_local = k[b, chunk_start:chunk_end]
        v_local = v[b, chunk_start:chunk_end]
        scores = torch.einsum("qhd,thd->qht", q[b, :q_len], k_local) * scale
        q_pos = torch.arange(q_len, device=q.device).view(q_len, 1, 1)
        k_pos = torch.arange(q_len, device=q.device).view(1, 1, q_len)
        mask = k_pos <= q_pos
        running_m, running_l, running_o = merge(running_m, running_l, running_o, scores, v_local, mask)
        chunks += 1
        streamed_tokens += q_len
        out[b, :q_len] = running_o / running_l.clamp_min(1e-20).unsqueeze(-1)
    return out, chunks, streamed_tokens


def run_blockwise_attn_smoke(args) -> dict:
    """Validate exact streaming/blockwise decode attention via online softmax."""
    import math
    import torch

    if args.blockwise_attn_window_tokens <= 0:
        raise ValueError("--blockwise-attn-window-tokens must be > 0")
    if args.blockwise_attn_tokens < args.blockwise_attn_window_tokens:
        raise ValueError("--blockwise-attn-tokens must be >= --blockwise-attn-window-tokens")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(0)
    torch.manual_seed(20260630)

    batch = args.blockwise_attn_batch
    num_heads = args.blockwise_attn_heads
    num_kv_heads = args.blockwise_attn_kv_heads
    head_dim = args.blockwise_attn_head_dim
    tokens = args.blockwise_attn_tokens
    window = args.blockwise_attn_window_tokens
    q = torch.randn(batch, num_heads, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch, tokens, num_kv_heads, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch, tokens, num_kv_heads, head_dim, device=device, dtype=torch.float32)
    context_lens = torch.full((batch,), tokens, device=device, dtype=torch.int64)
    if batch > 1:
        context_lens[-1] = max(1, tokens - window // 2)
    scale = 1.0 / math.sqrt(head_dim)

    full = _full_decode_attention(q, k, v, context_lens, scale)
    blockwise, chunks, streamed_tokens = _blockwise_decode_attention(q, k, v, context_lens, scale, window)
    diff = (full - blockwise).abs()
    max_abs = float(diff.max().item())
    max_ref = float(full.abs().max().item())
    rel = max_abs / max(max_ref, 1e-12)
    tolerance = 2e-5
    gate_fail_reasons = []
    if max_abs > tolerance and rel > tolerance:
        gate_fail_reasons.append(f"attention_mismatch max_abs={max_abs:.3e} rel={rel:.3e}")
    if chunks <= 1:
        gate_fail_reasons.append("chunks<=1")
    return {
        "mode": "blockwise-attn-smoke",
        "args": {
            "batch": batch,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "tokens": tokens,
            "window_tokens": window,
            "device": device,
        },
        "summary": {
            "gate_pass": not gate_fail_reasons,
            "gate_fail_reasons": gate_fail_reasons,
            "chunks": chunks,
            "streamed_tokens": streamed_tokens,
            "max_abs_error": max_abs,
            "relative_error": rel,
            "context_lens": [int(x) for x in context_lens.cpu().tolist()],
        },
    }


def run_blockwise_prefill_attn_smoke(args) -> dict:
    """Validate exact blockwise/chunked prefill attention via online softmax."""
    import math
    import torch

    if args.blockwise_attn_window_tokens <= 0:
        raise ValueError("--blockwise-attn-window-tokens must be > 0")
    if args.blockwise_prefill_prefix_tokens <= 0:
        raise ValueError("--blockwise-prefill-prefix-tokens must be > 0")
    if args.blockwise_prefill_chunk_tokens <= 0:
        raise ValueError("--blockwise-prefill-chunk-tokens must be > 0")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(0)
    torch.manual_seed(20260630)

    batch = args.blockwise_attn_batch
    num_heads = args.blockwise_attn_heads
    num_kv_heads = args.blockwise_attn_kv_heads
    head_dim = args.blockwise_attn_head_dim
    prefix = args.blockwise_prefill_prefix_tokens
    chunk = args.blockwise_prefill_chunk_tokens
    window = args.blockwise_attn_window_tokens
    max_tokens = prefix + chunk
    q = torch.randn(batch, chunk, num_heads, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch, max_tokens, num_kv_heads, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch, max_tokens, num_kv_heads, head_dim, device=device, dtype=torch.float32)
    chunk_starts = torch.full((batch,), prefix, device=device, dtype=torch.int64)
    chunk_lens = torch.full((batch,), chunk, device=device, dtype=torch.int64)
    if batch > 1:
        chunk_starts[-1] = max(1, prefix - window // 2)
    scale = 1.0 / math.sqrt(head_dim)

    full = _full_prefill_attention(q, k, v, chunk_starts, chunk_lens, scale)
    blockwise, chunks, streamed_tokens = _blockwise_prefill_attention(
        q, k, v, chunk_starts, chunk_lens, scale, window)
    diff = (full - blockwise).abs()
    max_abs = float(diff.max().item())
    max_ref = float(full.abs().max().item())
    rel = max_abs / max(max_ref, 1e-12)
    tolerance = 2e-5
    gate_fail_reasons = []
    if max_abs > tolerance and rel > tolerance:
        gate_fail_reasons.append(f"prefill_attention_mismatch max_abs={max_abs:.3e} rel={rel:.3e}")
    if chunks <= batch:
        gate_fail_reasons.append("prefix_chunks<=0")
    return {
        "mode": "blockwise-prefill-attn-smoke",
        "args": {
            "batch": batch,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "prefix_tokens": prefix,
            "chunk_tokens": chunk,
            "window_tokens": window,
            "device": device,
        },
        "summary": {
            "gate_pass": not gate_fail_reasons,
            "gate_fail_reasons": gate_fail_reasons,
            "chunks": chunks,
            "streamed_tokens": streamed_tokens,
            "max_abs_error": max_abs,
            "relative_error": rel,
            "chunk_starts": [int(x) for x in chunk_starts.cpu().tolist()],
            "chunk_lens": [int(x) for x in chunk_lens.cpu().tolist()],
        },
    }


def run_paired_profile(args) -> dict:
    llm, prompts, sp = _create_llm(args)
    warmup = _run_warmup(llm, args, prompts)
    simulated_upload_warmup_ms = _simulate_kv_upload(llm, args.simulate_kv_upload_mb)
    pairs = []
    stats_by_candidate = {}
    for prompt_index, prompt in enumerate(prompts):
        llm.add_request(prompt, sp)
        baseline_id = llm.scheduler.waiting[-1].seq_id
        llm.add_request(prompt, sp)
        candidate_id = llm.scheduler.waiting[-1].seq_id
        pair = {
            "prompt_index": prompt_index,
            "prompt": prompt,
            "baseline_seq_id": baseline_id,
            "candidate_seq_id": candidate_id,
        }
        pairs.append(pair)
        stats_by_candidate[candidate_id] = {
            "prompt_index": prompt_index,
            "baseline_seq_id": baseline_id,
            "candidate_seq_id": candidate_id,
            "commit_events": 0,
            "commit_attempts": 0,
            "zero_accept_events": 0,
            "no_draft_steps": 0,
            "drafted_tokens": 0,
            "accepted_count": 0,
            "verify_timing_ms": {},
            "events": [],
            "verify_events": [],
        }

    outputs = {}
    commit_events = []
    step_records = []
    t0 = time.perf_counter()
    cuda_sync_if_available()
    step_idx = 0
    while not llm.is_finished():
        committed_seq_ids = []
        for candidate_id, stats in stats_by_candidate.items():
            can_commit_more = args.max_commit_events == 0 or stats["commit_events"] < args.max_commit_events
            candidate = _find_running_seq(llm, candidate_id)
            if not can_commit_more or candidate is None:
                continue
            draft = propose_draft(candidate.token_ids, args)
            if not draft.tokens:
                stats["no_draft_steps"] += 1
                continue

            stats["commit_attempts"] += 1
            stats["drafted_tokens"] += len(draft.tokens)
            event = verify_and_commit_block(
                llm,
                candidate,
                draft.tokens,
                draft_source=draft.source,
                simulate_kv_upload_mb=args.simulate_kv_upload_mb,
                debug_target_hidden=args.debug_target_hidden,
                debug_hidden_to_draft_stub=args.debug_hidden_to_draft_stub,
                hidden_to_draft_adapter=args.hidden_to_draft_adapter,
                debug_hidden_to_draft_top_k=args.debug_hidden_to_draft_top_k,
            )
            event["draft_metadata"] = draft.metadata
            event_record = {
                "step": step_idx,
                "prompt_index": stats["prompt_index"],
                "baseline_seq_id": stats["baseline_seq_id"],
                "candidate_seq_id": candidate_id,
                **event,
            }
            stats["verify_events"].append(event_record)
            stats["accepted_count"] += event["accepted_count"]
            _accumulate_timing_ms(stats["verify_timing_ms"], event)
            if event["accepted_count"] <= 0:
                stats["zero_accept_events"] += 1
                continue

            stats["commit_events"] += 1
            stats["events"].append(event_record)
            commit_events.append(event_record)
            committed_seq_ids.append(candidate_id)
            if event["finished"]:
                outputs[candidate_id] = candidate.completion_token_ids

        if llm.is_finished():
            break

        t_step = time.perf_counter()
        out, num_tokens = llm.step()
        simulated_kv_upload_ms = 0.0
        if num_tokens < 0:
            simulated_kv_upload_ms = _simulate_kv_upload(llm, args.simulate_kv_upload_mb)
        cuda_sync_if_available()
        dt_ms = (time.perf_counter() - t_step) * 1000.0
        for seq_id, token_ids in out:
            outputs[seq_id] = token_ids

        step_records.append({
            "step": step_idx,
            "num_tokens": num_tokens,
            "dt_ms": dt_ms,
            "outputs": len(out),
            "committed": bool(committed_seq_ids),
            "committed_seq_ids": committed_seq_ids,
            "simulated_kv_upload_ms": simulated_kv_upload_ms,
        })
        step_idx += 1

    elapsed_s = time.perf_counter() - t0
    per_prompt = []
    for pair in pairs:
        baseline = outputs.get(pair["baseline_seq_id"], [])
        candidate = outputs.get(pair["candidate_seq_id"], [])
        stats = stats_by_candidate[pair["candidate_seq_id"]]
        per_prompt.append({
            **pair,
            "outputs_match": baseline == candidate,
            "baseline_output_tokens": len(baseline),
            "candidate_output_tokens": len(candidate),
            "committed": stats["commit_events"] > 0,
            "commit_events": stats["commit_events"],
            "commit_attempts": stats["commit_attempts"],
            "zero_accept_events": stats["zero_accept_events"],
            "no_draft_steps": stats["no_draft_steps"],
            "drafted_tokens": stats["drafted_tokens"],
            "accepted_count": stats["accepted_count"],
            "verify_timing_ms": stats["verify_timing_ms"],
            "verify_events": stats["verify_events"],
            "acceptance_rate": stats["accepted_count"] / stats["drafted_tokens"] if stats["drafted_tokens"] else 0.0,
            "candidate_autoregressive_steps_avoided": stats["accepted_count"],
            "candidate_step_reduction": stats["accepted_count"] / len(candidate) if candidate else 0.0,
        })

    total_candidate_tokens = sum(item["candidate_output_tokens"] for item in per_prompt)
    accepted_tokens = sum(item["accepted_count"] for item in per_prompt)
    drafted_tokens = sum(item["drafted_tokens"] for item in per_prompt)
    commit_attempts = sum(item["commit_attempts"] for item in per_prompt)
    zero_accept_events = sum(item["zero_accept_events"] for item in per_prompt)
    no_draft_steps = sum(item["no_draft_steps"] for item in per_prompt)
    summary = {
        "num_prompts": len(prompts),
        "baseline_seq_id": pairs[0]["baseline_seq_id"] if len(pairs) == 1 else None,
        "candidate_seq_id": pairs[0]["candidate_seq_id"] if len(pairs) == 1 else None,
        "outputs_match": all(item["outputs_match"] for item in per_prompt),
        "baseline_output_tokens": sum(item["baseline_output_tokens"] for item in per_prompt),
        "candidate_output_tokens": total_candidate_tokens,
        "committed": len(commit_events) > 0,
        "commit_events": len(commit_events),
        "accepted_count": accepted_tokens,
        "drafted_tokens": drafted_tokens,
        "acceptance_rate": accepted_tokens / drafted_tokens if drafted_tokens else 0.0,
        "candidate_autoregressive_steps_avoided": accepted_tokens,
        "candidate_step_reduction": accepted_tokens / total_candidate_tokens if total_candidate_tokens else 0.0,
        "commit_attempts": commit_attempts,
        "zero_accept_events": zero_accept_events,
        "no_draft_steps": no_draft_steps,
        "verify_timing_ms": _sum_timing_ms([{"timing_ms": item["verify_timing_ms"]} for item in per_prompt]),
        "elapsed_s": elapsed_s,
    }
    summary.update(_summarize_simulated_upload(
        args,
        step_records,
        summary["verify_timing_ms"],
        verify_upload_events=commit_attempts,
    ))
    gate_fail_reasons = []
    if not summary["outputs_match"]:
        gate_fail_reasons.append("outputs_match=false")
    if not summary["committed"]:
        gate_fail_reasons.append("committed=false")
    if summary["accepted_count"] <= 0:
        gate_fail_reasons.append("accepted_count<=0")
    if args.allow_zero_accept:
        gate_fail_reasons = [
            reason for reason in gate_fail_reasons
            if reason not in ("committed=false", "accepted_count<=0")
        ]
    summary["gate_pass"] = not gate_fail_reasons
    summary["gate_fail_reasons"] = gate_fail_reasons
    first_baseline = outputs.get(pairs[0]["baseline_seq_id"], []) if pairs else []
    first_candidate = outputs.get(pairs[0]["candidate_seq_id"], []) if pairs else []
    return {
        "args": vars(args),
        "summary": summary,
        "per_prompt": per_prompt,
        "commit_event": commit_events[0] if commit_events else None,
        "commit_events": commit_events,
        "verify_events": [
            event for item in per_prompt for event in item.get("verify_events", [])
        ],
        "baseline_text": llm.tokenizer.decode(first_baseline),
        "candidate_text": llm.tokenizer.decode(first_candidate),
        "step_records": step_records,
        "warmup": warmup,
        "simulated_upload_warmup_ms": simulated_upload_warmup_ms,
        "kv_offload": llm.model_runner.kv_offload_summary(),
    }


def run_baseline_only_profile(args) -> dict:
    llm, prompts, sp = _create_llm(args)
    warmup = _run_warmup(llm, args, prompts)
    simulated_upload_warmup_ms = _simulate_kv_upload(llm, args.simulate_kv_upload_mb)
    seq_to_prompt = {}
    for prompt_index, prompt in enumerate(prompts):
        llm.add_request(prompt, sp)
        seq_id = llm.scheduler.waiting[-1].seq_id
        seq_to_prompt[seq_id] = {"prompt_index": prompt_index, "prompt": prompt, "seq_id": seq_id}

    outputs = {}
    step_records = []
    t0 = time.perf_counter()
    cuda_sync_if_available()
    step_idx = 0
    while not llm.is_finished():
        t_step = time.perf_counter()
        out, num_tokens = llm.step()
        simulated_kv_upload_ms = 0.0
        if num_tokens < 0:
            simulated_kv_upload_ms = _simulate_kv_upload(llm, args.simulate_kv_upload_mb)
        cuda_sync_if_available()
        dt_ms = (time.perf_counter() - t_step) * 1000.0
        for seq_id, token_ids in out:
            outputs[seq_id] = token_ids
        step_records.append({
            "step": step_idx,
            "num_tokens": num_tokens,
            "dt_ms": dt_ms,
            "outputs": len(out),
            "simulated_kv_upload_ms": simulated_kv_upload_ms,
        })
        step_idx += 1

    elapsed_s = time.perf_counter() - t0
    per_prompt = []
    for seq_id, item in seq_to_prompt.items():
        token_ids = outputs.get(seq_id, [])
        per_prompt.append({
            **item,
            "output_tokens": len(token_ids),
            "token_ids": token_ids,
            "text": llm.tokenizer.decode(token_ids),
        })
    output_tokens = sum(item["output_tokens"] for item in per_prompt)
    summary = _base_summary(args, prompts, elapsed_s, step_records)
    summary.update({
        "output_tokens": output_tokens,
        "output_tokens_per_s": output_tokens / elapsed_s if elapsed_s > 0 else 0.0,
        "gate_pass": all(item["output_tokens"] == args.max_output_len for item in per_prompt),
        "gate_fail_reasons": [],
    })
    summary.update(_summarize_simulated_upload(args, step_records))
    if not summary["gate_pass"]:
        summary["gate_fail_reasons"].append("incomplete_output")
    return {
        "args": vars(args),
        "summary": summary,
        "per_prompt": per_prompt,
        "step_records": step_records,
        "warmup": warmup,
        "simulated_upload_warmup_ms": simulated_upload_warmup_ms,
        "kv_offload": llm.model_runner.kv_offload_summary(),
    }


def run_candidate_only_profile(args) -> dict:
    llm, prompts, sp = _create_llm(args)
    warmup = _run_warmup(llm, args, prompts)
    simulated_upload_warmup_ms = _simulate_kv_upload(llm, args.simulate_kv_upload_mb)
    stats_by_candidate = {}
    for prompt_index, prompt in enumerate(prompts):
        llm.add_request(prompt, sp)
        candidate_id = llm.scheduler.waiting[-1].seq_id
        stats_by_candidate[candidate_id] = {
            "prompt_index": prompt_index,
            "prompt": prompt,
            "seq_id": candidate_id,
            "commit_events": 0,
            "commit_attempts": 0,
            "zero_accept_events": 0,
            "no_draft_steps": 0,
            "drafted_tokens": 0,
            "accepted_count": 0,
            "verify_timing_ms": {},
            "events": [],
            "verify_events": [],
        }

    outputs = {}
    commit_events = []
    step_records = []
    t0 = time.perf_counter()
    cuda_sync_if_available()
    step_idx = 0
    while not llm.is_finished():
        committed_seq_ids = []
        for candidate_id, stats in stats_by_candidate.items():
            can_commit_more = args.max_commit_events == 0 or stats["commit_events"] < args.max_commit_events
            candidate = _find_running_seq(llm, candidate_id)
            if not can_commit_more or candidate is None:
                continue
            draft = propose_draft(candidate.token_ids, args)
            if not draft.tokens:
                stats["no_draft_steps"] += 1
                continue
            stats["commit_attempts"] += 1
            stats["drafted_tokens"] += len(draft.tokens)
            event = verify_and_commit_block(
                llm,
                candidate,
                draft.tokens,
                draft_source=draft.source,
                simulate_kv_upload_mb=args.simulate_kv_upload_mb,
                debug_target_hidden=args.debug_target_hidden,
                debug_hidden_to_draft_stub=args.debug_hidden_to_draft_stub,
                hidden_to_draft_adapter=args.hidden_to_draft_adapter,
                debug_hidden_to_draft_top_k=args.debug_hidden_to_draft_top_k,
            )
            event["draft_metadata"] = draft.metadata
            event_record = {"step": step_idx, "prompt_index": stats["prompt_index"], "candidate_seq_id": candidate_id, **event}
            stats["verify_events"].append(event_record)
            stats["accepted_count"] += event["accepted_count"]
            _accumulate_timing_ms(stats["verify_timing_ms"], event)
            if event["accepted_count"] <= 0:
                stats["zero_accept_events"] += 1
                continue
            stats["commit_events"] += 1
            stats["events"].append(event_record)
            commit_events.append(event_record)
            committed_seq_ids.append(candidate_id)
            if event["finished"]:
                outputs[candidate_id] = candidate.completion_token_ids

        if llm.is_finished():
            break

        t_step = time.perf_counter()
        out, num_tokens = llm.step()
        simulated_kv_upload_ms = 0.0
        if num_tokens < 0:
            simulated_kv_upload_ms = _simulate_kv_upload(llm, args.simulate_kv_upload_mb)
        cuda_sync_if_available()
        dt_ms = (time.perf_counter() - t_step) * 1000.0
        for seq_id, token_ids in out:
            outputs[seq_id] = token_ids
        step_records.append({
            "step": step_idx,
            "num_tokens": num_tokens,
            "dt_ms": dt_ms,
            "outputs": len(out),
            "committed": bool(committed_seq_ids),
            "committed_seq_ids": committed_seq_ids,
            "simulated_kv_upload_ms": simulated_kv_upload_ms,
        })
        step_idx += 1

    elapsed_s = time.perf_counter() - t0
    per_prompt = []
    for candidate_id, stats in stats_by_candidate.items():
        token_ids = outputs.get(candidate_id, [])
        per_prompt.append({
            "prompt_index": stats["prompt_index"],
            "prompt": stats["prompt"],
            "seq_id": candidate_id,
            "output_tokens": len(token_ids),
            "token_ids": token_ids,
            "text": llm.tokenizer.decode(token_ids),
            "committed": stats["commit_events"] > 0,
            "commit_events": stats["commit_events"],
            "commit_attempts": stats["commit_attempts"],
            "zero_accept_events": stats["zero_accept_events"],
            "no_draft_steps": stats["no_draft_steps"],
            "drafted_tokens": stats["drafted_tokens"],
            "accepted_count": stats["accepted_count"],
            "verify_timing_ms": stats["verify_timing_ms"],
            "verify_events": stats["verify_events"],
            "acceptance_rate": stats["accepted_count"] / stats["drafted_tokens"] if stats["drafted_tokens"] else 0.0,
            "candidate_step_reduction": stats["accepted_count"] / len(token_ids) if token_ids else 0.0,
        })
    output_tokens = sum(item["output_tokens"] for item in per_prompt)
    accepted_tokens = sum(item["accepted_count"] for item in per_prompt)
    drafted_tokens = sum(item["drafted_tokens"] for item in per_prompt)
    commit_attempts = sum(item["commit_attempts"] for item in per_prompt)
    zero_accept_events = sum(item["zero_accept_events"] for item in per_prompt)
    no_draft_steps = sum(item["no_draft_steps"] for item in per_prompt)
    summary = _base_summary(args, prompts, elapsed_s, step_records)
    summary.update({
        "output_tokens": output_tokens,
        "output_tokens_per_s": output_tokens / elapsed_s if elapsed_s > 0 else 0.0,
        "committed": len(commit_events) > 0,
        "commit_events": len(commit_events),
        "commit_attempts": commit_attempts,
        "zero_accept_events": zero_accept_events,
        "no_draft_steps": no_draft_steps,
        "drafted_tokens": drafted_tokens,
        "accepted_count": accepted_tokens,
        "acceptance_rate": accepted_tokens / drafted_tokens if drafted_tokens else 0.0,
        "candidate_autoregressive_steps_avoided": accepted_tokens,
        "candidate_step_reduction": accepted_tokens / output_tokens if output_tokens else 0.0,
        "verify_timing_ms": _sum_timing_ms([{"timing_ms": item["verify_timing_ms"]} for item in per_prompt]),
    })
    summary.update(_summarize_simulated_upload(
        args,
        step_records,
        summary["verify_timing_ms"],
        verify_upload_events=commit_attempts,
    ))
    gate_fail_reasons = []
    if not all(item["output_tokens"] == args.max_output_len for item in per_prompt):
        gate_fail_reasons.append("incomplete_output")
    if not summary["committed"]:
        gate_fail_reasons.append("committed=false")
    if summary["accepted_count"] <= 0:
        gate_fail_reasons.append("accepted_count<=0")
    if args.allow_zero_accept:
        gate_fail_reasons = [
            reason for reason in gate_fail_reasons
            if reason not in ("committed=false", "accepted_count<=0")
        ]
    summary["gate_pass"] = not gate_fail_reasons
    summary["gate_fail_reasons"] = gate_fail_reasons
    return {
        "args": vars(args),
        "summary": summary,
        "per_prompt": per_prompt,
        "commit_event": commit_events[0] if commit_events else None,
        "commit_events": commit_events,
        "verify_events": [
            event for item in per_prompt for event in item.get("verify_events", [])
        ],
        "step_records": step_records,
        "warmup": warmup,
        "simulated_upload_warmup_ms": simulated_upload_warmup_ms,
        "kv_offload": llm.model_runner.kv_offload_summary(),
    }


def run_profile(args) -> dict:
    if args.kv_offload_migration_smoke:
        return run_kv_offload_migration_smoke()
    if args.kv_offload_thrash_smoke:
        return run_kv_offload_thrash_smoke(args)
    if args.blockwise_attn_smoke:
        return run_blockwise_attn_smoke(args)
    if args.blockwise_prefill_attn_smoke:
        return run_blockwise_prefill_attn_smoke(args)

    if args.model is None:
        raise ValueError("--model is required unless a KV offload synthetic smoke flag is set")

    if args.temperature != 0.0:
        raise ValueError("S3/S4 profiler currently supports greedy decoding only (--temperature 0.0)")

    if args.max_commit_events < 0:
        raise ValueError("--max-commit-events must be >= 0; use 0 for unlimited")
    if args.warmup_output_len < 0:
        raise ValueError("--warmup-output-len must be >= 0")
    if args.simulate_kv_upload_mb < 0:
        raise ValueError("--simulate-kv-upload-mb must be >= 0")

    if args.mode == "baseline-only":
        return run_baseline_only_profile(args)
    if args.mode == "candidate-only":
        return run_candidate_only_profile(args)
    return run_paired_profile(args)


def main():
    args = parse_args()
    result = run_profile(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out_json:
        out_dir = os.path.dirname(os.path.abspath(args.out_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    if not result["summary"]["gate_pass"]:
        reasons = ", ".join(result["summary"]["gate_fail_reasons"])
        print(f"S3/S4 commit smoke gate failed: {reasons}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
