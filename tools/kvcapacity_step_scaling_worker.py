#!/usr/bin/env python3
"""GATE A worker: measure how a decode step scales with context and batch.

Why this exists
---------------
Stage 0 of the latent KV capacity line models the decode step as::

    step_ms(L, B) = c0 + c1 * L * B

with `c0 = 13.05 ms` and `c1 = 0.151 us/token`. Those two constants were fit
from a **batch-1** artifact. The batch term is therefore an *extrapolation*, and
every Stage 0 conclusion, including the entire GO/NO-GO structure, rests on it.

The previous research line in this repository died because a ratio was trusted
while its denominator was measured by a harness that was technically correct and
semantically wrong. This worker exists so that the same mistake is not repeated
one level up. It measures `step_ms(L, B)` directly on the serving path that this
repository ships, and it is allowed to invalidate Stage 0.

What it measures
----------------
For each pre-registered `(context_length, batch)` cell, `batch` independent
requests are admitted, each carrying `context_length` tokens, and the engine is
driven one scheduler step at a time. Only decode steps whose *observed* running
batch equals the target are timed. The observed batch is read from the engine's
own step accounting (`num_tokens = -len(seqs)` for decode) rather than assumed.

Three guards that the measurement can fail on
---------------------------------------------
1. **Prefix sharing.** Block-hash prefix caching is enabled on this engine. If
   the `batch` prompts shared any prefix, their KV blocks would be shared, the
   decode step would read far fewer than `batch * context_length` tokens, and the
   step would look sublinear in batch. That would masquerade as good news. Every
   prompt is therefore drawn independently at full length, and the total prefill
   token count reported by the engine is recorded next to `batch * context_length`
   so a discrepancy is visible instead of silently favourable.
2. **Observed batch drift.** Requests may finish or be preempted at different
   times, so a step nominally at batch `B` can run fewer sequences. Steps whose
   observed batch differs from the target are discarded, not averaged in.
3. **Capacity.** A cell whose KV footprint exceeds the device budget is recorded
   as skipped with a reason. It is never estimated.

This worker measures time and capacity only. It says nothing about output
quality, and the pre-registered plan forbids drawing quality conclusions from it.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time

# Pre-registered measurement grid. Each cell is feasible under a 47.6 GiB KV
# budget at the Qwen3-8B GQA footprint of 147456 bytes per token, which is what
# the Stage 0 artifact assumes. The cells were chosen so that the product
# `context_length * batch` collides across different shapes; those collisions are
# the sharpest available test of the model's central claim.
# Amended after the first full run. Qwen3-8B declares max_position_embeddings =
# 40960, so the engine clamps max_model_len to 40960 and rejected every cell at
# 65536 and 131072: those contexts were never reachable without RoPE scaling,
# which would change the model instead of measuring it. Stage 0 swept contexts up
# to 131072 for a model that cannot reach them, and this grid inherited the error.
#
# Resident token count is extended through batch instead, which preserves the full
# L * B range Stage 0 depends on, up to 262144. That substitution is legitimate
# here precisely because the first run established it: at four separate
# equal-product groups the members agreed within 5.3%, so at fixed L * B the shape
# does not matter. See PREREGISTERED_CELLS_V1 in the verdict tool for the original.
CONTEXT_BATCH_GRID = (
    (8192, (1, 2, 4, 8, 16, 32)),
    (16384, (1, 2, 4, 8, 16)),
    (32768, (1, 2, 4, 8)),
    # 40448 rather than 40960: a request is rejected when prompt + generated tokens
    # exceeds max_model_len, and this worker generates warmup + measured + 2 tokens.
    # Asking for the full 40960 leaves no room to decode and fails outright.
    (40448, (1, 2, 4)),
)

KV_BYTES_PER_TOKEN = 147456
WARMUP_STEPS = 32
MEASURED_STEPS = 24
DEFAULT_SEED = 20260913


def parse_grid_spec(text):
    """Parse a grid override such as ``1024:1,2,4;2048:1,2``.

    The override exists so a smoke run can prove the plumbing works on a small
    model without occupying a GPU for the full grid. It is not a tuning knob. The
    artifact records whether the pre-registered grid was used, so a run on a
    narrowed grid cannot later be presented as the pre-registered measurement.
    """
    groups = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"grid group {chunk!r} is missing its ':' separator")
        head, tail = chunk.split(":", 1)
        context_length = int(head)
        if context_length <= 0:
            raise ValueError("context length must be positive")
        batches = tuple(int(value) for value in tail.split(",") if value.strip())
        if not batches:
            raise ValueError(f"grid group {chunk!r} lists no batch")
        if any(batch <= 0 for batch in batches):
            raise ValueError("batch must be positive")
        if len(set(batches)) != len(batches):
            raise ValueError(f"grid group {chunk!r} repeats a batch")
        groups.append((context_length, batches))
    if not groups:
        raise ValueError("grid specification is empty")
    if len({context for context, _batches in groups}) != len(groups):
        raise ValueError("grid specification repeats a context length")
    return tuple(groups)


def format_grid_spec(grid):
    return ";".join(
        f"{context}:" + ",".join(str(batch) for batch in batches)
        for context, batches in grid
    )


def enumerate_cells(grid=CONTEXT_BATCH_GRID):
    """Flatten the grid into ordered (context_length, batch) pairs."""
    cells = []
    for context_length, batches in grid:
        for batch in batches:
            cells.append((int(context_length), int(batch)))
    return tuple(cells)


def product_collision_groups(cells):
    """Group cells by `context_length * batch`.

    Only groups with more than one member are returned, because a single member
    cannot test anything.
    """
    groups = {}
    for context_length, batch in cells:
        groups.setdefault(context_length * batch, []).append((context_length, batch))
    return {
        product: tuple(members)
        for product, members in sorted(groups.items())
        if len(members) > 1
    }


def kv_bytes_for_cell(context_length, batch, bytes_per_token=KV_BYTES_PER_TOKEN):
    return int(context_length) * int(batch) * int(bytes_per_token)


def cell_fits(context_length, batch, available_kv_bytes, *, headroom=0.95):
    """Whether a cell's resident KV fits, keeping a little headroom."""
    if available_kv_bytes is None:
        return True
    required = kv_bytes_for_cell(context_length, batch)
    return required <= float(available_kv_bytes) * headroom


def build_distinct_prompts(context_length, batch, vocab_size, rng):
    """Draw `batch` independent prompts, each `context_length` tokens long.

    Independence at full length is deliberate. Sharing even the first block
    across two prompts would let block-hash prefix caching merge their KV and
    corrupt the batch scaling this worker exists to measure.
    """
    if vocab_size < 16:
        raise ValueError("vocab_size is implausibly small")
    prompts = []
    seen_first_block = set()
    for _ in range(batch):
        while True:
            prompt = [rng.randrange(4, vocab_size - 1) for _ in range(context_length)]
            marker = tuple(prompt[:256])
            if marker not in seen_first_block:
                seen_first_block.add(marker)
                break
        prompts.append(prompt)
    return prompts


def prompt_digest(prompt):
    payload = ",".join(str(token) for token in prompt).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def measurement_drift(samples):
    """Compare the second half of the measured window with the first.

    This is the guard against a contaminated window. `torch.compile` recompiles
    per input shape on this engine, so a cell that changes batch can spend its
    early decode steps paying for compilation. Averaged in, that inflates the
    constant term and would be indistinguishable from a genuinely expensive
    engine. A window that is still warming up shows a falling trend, so the ratio
    below is reported and checked rather than assumed away.
    """
    if len(samples) < 4:
        return None
    midpoint = len(samples) // 2
    first = statistics.median(samples[:midpoint])
    second = statistics.median(samples[midpoint:])
    if first <= 0:
        return None
    return {
        "first_half_median_ms": first,
        "second_half_median_ms": second,
        "ratio": second / first,
    }


def summarise(samples):
    """Robust summary of a list of step durations in milliseconds."""
    if not samples:
        return None
    ordered = sorted(samples)
    return {
        "count": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p10_ms": ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        "p90_ms": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "stdev_ms": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# GPU section. Everything below imports torch and tinyvllm lazily so that the
# pure helpers above stay importable, and testable, on a laptop.
# ---------------------------------------------------------------------------


def multi_sequence_graph_kwargs(batches):
    """Config overrides that let decode batches above 1 replay a captured graph.

    By default `model_runner.py:12557` fails every decode step with more than one
    sequence closed to eager, because the legacy captured graphs are only
    correctness-validated for a single sequence. Every earlier GATE A run
    therefore measured an uncaptured path for B >= 2 and paid a per-step Python
    cost that a production engine does not, which is why the fitted constant came
    out near 39.8 ms against a weight-bandwidth floor closer to 13 ms.

    The dynamic multi-sequence path is opt-in and dispatches only for batches on
    an explicit allowlist, so the allowlist has to cover the grid or the run
    silently falls back to the same eager path it is meant to replace.
    """
    wanted = tuple(sorted({int(batch) for batch in batches if int(batch) > 1}))
    if not wanted:
        return {}
    return {
        "multi_sequence_cuda_graphs": True,
        "multi_sequence_cuda_graph_batch_allowlist": wanted,
        "multi_sequence_cuda_graph_max_entries": max(8, len(wanted) * 2),
        # The capture budgets below are a serving-safety policy, not a hardware
        # fact, and at their defaults they silently decide the measurement. The
        # smoke run captured batch 4 and batch 8 but rejected batch 2 with
        # `single_capture_budget`, because the first capture in a process also
        # pays torch.compile for the shape and overran the 2 s default. That left
        # batch 2 on the eager path at 31.4 ms next to batch 4 at 4.8 ms, which
        # would have read as a batch-scaling cliff rather than as a budget.
        # One-time capture cost is not what GATE A is measuring, so it is given
        # room and recorded.
        "multi_sequence_cuda_graph_max_single_capture_ns": 120_000_000_000,
        "multi_sequence_cuda_graph_max_total_capture_ns": 900_000_000_000,
        "multi_sequence_cuda_graph_max_static_bytes": 1024 * 1024 * 1024,
        "multi_sequence_cuda_graph_max_reserved_bytes": 4 * 1024 * 1024 * 1024,
    }


def resolve_model_runner(engine):
    """Find the ModelRunner, which is the only object that knows what really ran."""
    for path in ("model_runner", "llm_engine.model_runner", "engine.model_runner"):
        target = engine
        try:
            for attribute in path.split("."):
                target = getattr(target, attribute)
        except AttributeError:
            continue
        if target is not None:
            return target
    return None


def dispatch_observation(engine):
    """Read the engine's own account of how the last step was dispatched.

    Returns None when the engine does not publish the event, which is itself the
    finding: without it a run cannot claim to have measured the graph path.
    """
    runner = resolve_model_runner(engine)
    if runner is None:
        return None
    reader = getattr(runner, "cuda_graph_dispatch_observation", None)
    if not callable(reader):
        return None
    try:
        return reader()
    except Exception:  # noqa: BLE001 - evidence, never control flow
        return None


def dispatch_label(event):
    """Collapse a dispatch event into one countable label.

    `dispatch` alone is not enough. A step that ran eager because the batch was
    not allowlisted and a step that ran eager because the graph was still being
    observed are the same word and different facts, and only the second one
    disappears once the cache warms.
    """
    if event is None:
        return "unobserved"
    dispatch = str(event.get("dispatch") or "unknown")
    if dispatch == "graph":
        return "graph"
    reason = event.get("fallback_reason")
    cache_state = event.get("cache_state")
    suffix = reason or cache_state or "unspecified"
    return f"eager:{suffix}"


class DispatchTracker:
    """Label each decode step by how the engine dispatched it.

    The engine keeps only the most recent dispatch event, and the legacy batch-1
    graph path does not publish one at all. Reading the field blindly therefore
    reports a stale label as if it described the current step, which is the same
    class of error this whole rerun exists to correct: the batch-1 cell of the
    first msgraph smoke run came back tagged `eager:unsupported_mode`, inherited
    from a prefill step, while it was in fact replaying the batch-1 graph. The
    event carries a monotonically increasing `step_id`, so an unchanged id means
    nothing was published for this step and the label must say so.
    """

    def __init__(self):
        self.last_step_id = None

    def observe(self, engine):
        event = dispatch_observation(engine)
        step_id = None if event is None else event.get("step_id")
        if event is None:
            return "unobserved"
        if step_id is not None and step_id == self.last_step_id:
            return "unpublished"
        self.last_step_id = step_id
        return dispatch_label(event)


def summarise_dispatch(labels):
    """Turn per-step dispatch labels into the audit that validates the run.

    The reason this worker is being rerun at all is that the previous GATE A
    measured an eager fallback while believing it measured the serving path. The
    replacement therefore has to be able to fail the same way out loud: if
    `graph_share` is not close to 1 for a cell above batch 1, the cell measured
    the old path again and its constant means the same thing it meant before.
    """
    if not labels:
        return None
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    graph = counts.get("graph", 0)
    return {
        "counts": dict(sorted(counts.items())),
        "steps": len(labels),
        "graph_steps": graph,
        "graph_share": graph / len(labels),
        "all_graph": graph == len(labels),
    }


def _load_engine(*, model_path, max_model_len, enforce_eager, gpu_memory_utilization,
                 max_num_seqs, multi_sequence_graph_batches=None):
    from tinyvllm import LLM

    extra = {}
    if multi_sequence_graph_batches:
        extra = multi_sequence_graph_kwargs(multi_sequence_graph_batches)

    engine = LLM(
        model=model_path,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
        max_num_batched_tokens=max(16384, max_model_len),
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
        **extra,
    )
    return engine


def resolve_config_holder(engine):
    """Find the object that actually carries `config`.

    The wall sweep recorded every identity field as null because `LLM` does not
    expose `config` directly on itself in every build, and the lookup gave up at
    the first AttributeError. Provenance that silently degrades to null is worse
    than no provenance, because the artifact still looks complete.
    """
    for path in ("", "llm_engine", "engine", "model_runner"):
        target = engine
        if path:
            try:
                for attribute in path.split("."):
                    target = getattr(target, attribute)
            except AttributeError:
                continue
        if getattr(target, "config", None) is not None:
            return target
    return engine


def _engine_identity(engine):
    """Best-effort record of what the engine actually allocated.

    Reached defensively: this is evidence, not control flow, and a private
    attribute moving must not fail the run.
    """
    identity = {}
    engine = resolve_config_holder(engine)
    for path, key in (
        ("config.num_kvcache_blocks", "num_kvcache_blocks"),
        ("config.kvcache_block_size", "kvcache_block_size"),
        ("config.max_model_len", "max_model_len"),
        ("config.max_num_seqs", "max_num_seqs"),
        ("config.max_num_batched_tokens", "max_num_batched_tokens"),
        ("config.gpu_memory_utilization", "gpu_memory_utilization"),
        ("config.enforce_eager", "enforce_eager"),
        ("config.multi_sequence_cuda_graphs", "multi_sequence_cuda_graphs"),
        ("config.kv_quant_bits", "kv_quant_bits"),
        ("config.cpu_offload", "cpu_offload"),
        ("config.hf_config.vocab_size", "vocab_size"),
        ("config.hf_config.num_hidden_layers", "num_hidden_layers"),
        ("config.hf_config.num_key_value_heads", "num_key_value_heads"),
        ("config.hf_config.head_dim", "head_dim"),
    ):
        target = engine
        try:
            for attribute in path.split("."):
                target = getattr(target, attribute)
        except AttributeError:
            identity[key] = None
            continue
        identity[key] = target
    # Which decode batch sizes have a captured graph decides whether step time is
    # even a smooth function of batch, so it is recorded as evidence.
    identity["captured_graph_batches"] = None
    for path in ("model_runner.graph_bs", "llm_engine.model_runner.graph_bs"):
        target = engine
        try:
            for attribute in path.split("."):
                target = getattr(target, attribute)
        except AttributeError:
            continue
        try:
            identity["captured_graph_batches"] = sorted(int(v) for v in target)
        except (TypeError, ValueError):
            identity["captured_graph_batches"] = None
        break

    blocks = identity.get("num_kvcache_blocks")
    block_size = identity.get("kvcache_block_size")
    if isinstance(blocks, int) and isinstance(block_size, int) and blocks > 0:
        identity["kv_capacity_tokens"] = blocks * block_size
        identity["kv_capacity_bytes"] = blocks * block_size * KV_BYTES_PER_TOKEN
    else:
        identity["kv_capacity_tokens"] = None
        identity["kv_capacity_bytes"] = None
    return identity


def _measure_cell(engine, *, context_length, batch, vocab_size, rng,
                  warmup_steps, measured_steps, engine_max_model_len):
    """Drive one grid cell and time only the steps that really ran at `batch`."""
    from tinyvllm.sampling_params import SamplingParams

    prompts = build_distinct_prompts(context_length, batch, vocab_size, rng)
    max_tokens = warmup_steps + measured_steps + 2
    if context_length + max_tokens > engine_max_model_len:
        raise ValueError(
            f"context {context_length} plus {max_tokens} generated tokens exceeds "
            f"max_model_len {engine_max_model_len}; the engine would reject the "
            "request outright"
        )
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    for prompt in prompts:
        engine.add_request(prompt, params)

    prefill_tokens_total = 0
    prefill_steps = 0
    decode_steps_by_batch = {}
    samples = []
    trace = []
    observed_batches = []
    measured_dispatch = []
    all_decode_dispatch = []
    dispatch_tracker = DispatchTracker()
    step_index = 0
    guard = (warmup_steps + measured_steps + 8) * max(1, batch) + 64

    while not engine.is_finished():
        step_index += 1
        if step_index > guard:
            break
        start = time.perf_counter()
        _output, num_tokens = engine.step()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if num_tokens > 0:
            prefill_tokens_total += num_tokens
            prefill_steps += 1
            continue
        observed = -num_tokens
        observed_batches.append(observed)
        step_dispatch = dispatch_tracker.observe(engine)
        all_decode_dispatch.append(step_dispatch)
        decode_steps_by_batch[observed] = decode_steps_by_batch.get(observed, 0) + 1
        if observed != batch:
            continue
        if len(trace) < 256:
            trace.append(elapsed_ms)
        if decode_steps_by_batch[observed] <= warmup_steps:
            continue
        if len(samples) < measured_steps:
            samples.append(elapsed_ms)
            measured_dispatch.append(step_dispatch)

    return {
        "context_length": context_length,
        "batch": batch,
        "kv_tokens": context_length * batch,
        "kv_bytes": kv_bytes_for_cell(context_length, batch),
        "prompt_digests": [prompt_digest(prompt) for prompt in prompts],
        "prefill_steps": prefill_steps,
        "prefill_tokens_total": prefill_tokens_total,
        "prefill_tokens_expected": context_length * batch,
        "prefill_tokens_match": prefill_tokens_total == context_length * batch,
        "decode_steps_by_observed_batch": {
            str(key): value for key, value in sorted(decode_steps_by_batch.items())
        },
        "observed_batch_is_stable": bool(
            observed_batches and set(observed_batches) == {batch}
        ),
        "target_batch_step_count": decode_steps_by_batch.get(batch, 0),
        "step": summarise(samples),
        "drift": measurement_drift(samples),
        "dispatch_measured": summarise_dispatch(measured_dispatch),
        "dispatch_all_decode": summarise_dispatch(all_decode_dispatch),
        "target_batch_step_trace_ms": trace,
        "warmup_steps": warmup_steps,
        "measured": bool(samples),
        "skipped_reason": None if samples else "no decode step ran at the target batch",
    }


def run(*, model_path, gpu_memory_utilization, enforce_eager, seed,
        warmup_steps, measured_steps, grid=CONTEXT_BATCH_GRID,
        multi_sequence_cuda_graphs=False):
    """Measure every feasible cell, one engine per context length.

    Context lengths are attempted in ascending order and a group that fails is
    recorded rather than allowed to abort the run. The largest contexts are the
    plausible casualties: admitting a 131072-token prompt requires
    `max_num_batched_tokens` to be at least as large, so a single prefill step can
    demand a great deal of activation memory. Losing that group should cost the
    run those cells and nothing else, because a partial grid still constrains the
    model while a crashed run constrains nothing.
    """
    rows = []
    engines = []
    for context_length, batches in sorted(grid, key=lambda group: group[0]):
        engine = None
        try:
            engine = _load_engine(
                model_path=model_path,
                max_model_len=context_length + 1024,
                enforce_eager=enforce_eager,
                gpu_memory_utilization=gpu_memory_utilization,
                max_num_seqs=max(8, max(batches) + 4),
                multi_sequence_graph_batches=(
                    batches if multi_sequence_cuda_graphs else None
                ),
            )
        except Exception as error:  # noqa: BLE001 - recorded, not swallowed
            reason = f"engine construction failed: {type(error).__name__}: {error}"
            engines.append(
                {"context_length": context_length, "identity": None, "error": reason}
            )
            for batch in batches:
                rows.append(
                    {
                        "context_length": context_length,
                        "batch": batch,
                        "kv_tokens": context_length * batch,
                        "kv_bytes": kv_bytes_for_cell(context_length, batch),
                        "measured": False,
                        "step": None,
                        "skipped_reason": reason,
                    }
                )
            continue

        identity = _engine_identity(engine)
        engines.append({"context_length": context_length, "identity": identity})
        vocab_size = identity.get("vocab_size") or 151936
        available = identity.get("kv_capacity_bytes")
        rng = random.Random(seed + context_length)
        try:
            for batch in batches:
                if not cell_fits(context_length, batch, available):
                    rows.append(
                        {
                            "context_length": context_length,
                            "batch": batch,
                            "kv_tokens": context_length * batch,
                            "kv_bytes": kv_bytes_for_cell(context_length, batch),
                            "measured": False,
                            "step": None,
                            "skipped_reason": (
                                "resident KV exceeds the device budget: "
                                f"{kv_bytes_for_cell(context_length, batch)} bytes "
                                f"required, {available} available"
                            ),
                        }
                    )
                    continue
                try:
                    rows.append(
                        _measure_cell(
                            engine,
                            context_length=context_length,
                            batch=batch,
                            vocab_size=vocab_size,
                            rng=rng,
                            warmup_steps=warmup_steps,
                            measured_steps=measured_steps,
                            engine_max_model_len=(
                                identity.get("max_model_len") or context_length + 1024
                            ),
                        )
                    )
                except Exception as error:  # noqa: BLE001 - recorded, not swallowed
                    rows.append(
                        {
                            "context_length": context_length,
                            "batch": batch,
                            "kv_tokens": context_length * batch,
                            "kv_bytes": kv_bytes_for_cell(context_length, batch),
                            "measured": False,
                            "step": None,
                            "skipped_reason": (
                                f"measurement failed: {type(error).__name__}: {error}"
                            ),
                        }
                    )
        finally:
            del engine
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
    return rows, engines


def build_payload(rows, engines, *, model_path, enforce_eager, seed,
                  gpu_memory_utilization, warmup_steps, measured_steps,
                  grid=CONTEXT_BATCH_GRID, multi_sequence_cuda_graphs=False):
    cells = enumerate_cells(grid)
    preregistered = tuple(grid) == CONTEXT_BATCH_GRID
    payload = {
        "schema": "kvcapacity-gate-a-step-scaling/1",
        "purpose": (
            "Validate the batch term of the Stage 0 decode-step model "
            "step_ms(L, B) = c0 + c1 * L * B on the tinyvllm serving path."
        ),
        "configuration": {
            "model_path": model_path,
            "enforce_eager": enforce_eager,
            "multi_sequence_cuda_graphs": multi_sequence_cuda_graphs,
            "gpu_memory_utilization": gpu_memory_utilization,
            "seed": seed,
            "warmup_steps": warmup_steps,
            "measured_steps": measured_steps,
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        },
        "grid": [list(cell) for cell in cells],
        "grid_spec": format_grid_spec(grid),
        "grid_is_preregistered": preregistered,
        "preregistered_grid_spec": format_grid_spec(CONTEXT_BATCH_GRID),
        "product_collision_groups": {
            str(product): [list(member) for member in members]
            for product, members in product_collision_groups(cells).items()
        },
        "engines": engines,
        "rows": rows,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--multi-sequence-cuda-graphs",
        action="store_true",
        help=(
            "let decode batches above 1 replay a captured graph instead of "
            "falling back to eager; measures the constant a production engine "
            "would pay rather than this engine's uncaptured path"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--measured-steps", type=int, default=MEASURED_STEPS)
    parser.add_argument(
        "--grid-spec",
        help=(
            "override the pre-registered grid, for smoke runs only, "
            "formatted as 1024:1,2,4;2048:1,2"
        ),
    )
    args = parser.parse_args(argv)
    if args.grid_spec:
        try:
            args.grid = parse_grid_spec(args.grid_spec)
        except ValueError as error:
            parser.error(str(error))
    else:
        args.grid = CONTEXT_BATCH_GRID
    return args


def main(argv=None):
    args = parse_args(argv)
    rows, engines = run(
        model_path=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
        grid=args.grid,
        multi_sequence_cuda_graphs=args.multi_sequence_cuda_graphs,
    )
    payload = build_payload(
        rows,
        engines,
        model_path=args.model_path,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
        grid=args.grid,
        multi_sequence_cuda_graphs=args.multi_sequence_cuda_graphs,
    )
    directory = os.path.dirname(os.path.abspath(args.out))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    measured = sum(1 for row in rows if row.get("measured"))
    print(f"measured {measured}/{len(rows)} cells")
    if not payload["grid_is_preregistered"]:
        print(
            "WARNING: this run used a narrowed grid, so it is a plumbing check "
            "and not the pre-registered GATE A measurement"
        )
    print(f"payload_sha256 {payload['payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
