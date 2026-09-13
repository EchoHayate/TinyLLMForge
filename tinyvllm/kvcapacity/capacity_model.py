"""Analytic serving-capacity model for latent KV compression.

The model has one job: decide which resource binds serving capacity for an
agent workload, and therefore whether shrinking the KV cache can buy anything.

Structure
---------
An agent turn is::

    restore (optional, DMA)  ->  warm prefill  ->  N decode steps  ->  tool wait

Only the middle two need the GPU. The tool wait needs nothing. But without an
offload path the KV cache stays resident across the tool wait anyway, because
there is nowhere else to put it. That asymmetry is the entire reason lossless
host offload is a serious competitor to lossy compression, and the model keeps
the two residency regimes separate:

    duty_compute = gpu_phase / period       always
    duty_memory  = 1                        resident KV, held across the wait
    duty_memory  = (restore + gpu_phase) / period    with offload

Steady state
------------
With `N` agents each spending `duty_compute` of its cycle on the GPU, the mean
number of agents decoding together is `N * duty_compute`, which *is* the decode
batch. So batch is not a free parameter: to sustain a mean batch of `B` you need

    N = utilization_target * B / duty_compute

agents in flight, and you need all of them to fit in the KV budget. A row where
the required `N` exceeds what memory can hold is not a slower configuration, it
is an unreachable one, and the model marks it infeasible rather than reporting a
capacity for it.

Batching and why it drives this whole line
------------------------------------------
The decode step is modelled as::

    step_ms(L, B) = c0 + c1 * L * B

`c0` is the batch-invariant part: weight traffic and the projection GEMMs are
read once per step regardless of batch. `c1 * L * B` is the KV attention part,
which scales with total resident KV read by the step. The consequence drives
everything downstream: at `B = 1` the KV term is a small minority of the step,
so compression is nearly worthless; as `B` grows the `c0` term amortises and the
KV term approaches 100% of the step, so compression approaches its full value.

Compression is parametrised by two independent numbers rather than by any
particular decomposition:

    kv_ratio               r >= 1, how many times smaller the KV bytes get
    attention_time_ratio   phi > 0, how the KV attention *time* term scales

`phi = 1 / r` is the memory-bound ideal, where time tracks bytes. `phi = 1`
means bytes were saved but no time was. `phi > 1` means the reconstruction FLOPs
cost more than the byte saving won, which is the real risk for MLA-style latent
attention where every query head attends to a shared latent instead of a shared
group of KV heads. `phi` is exactly what the first GPU stage must measure; the
model treats it as unknown and sweeps it.

Metrics trap this model deliberately closes
-------------------------------------------
Capacity alone can be inflated by making every turn slower: a longer period
lowers each agent's duty cycle and therefore packs more agents. Offload does
exactly this, by adding a restore transfer. So `user_latency_seconds` includes
the restore, and every capacity comparison must be paired with a latency ratio.
Capacity without a latency bound is not a result.
"""

from dataclasses import dataclass

_BYTES_PER_GIB = 1024 ** 3


def _require_positive(value, *, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    numeric = float(value)
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    if allow_zero:
        if numeric < 0.0:
            raise ValueError(f"{name} must be non-negative")
    elif numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    return numeric


def _require_positive_int(value, *, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class ModelGeometry:
    """Baseline KV geometry of the served checkpoint."""

    name: str
    layers: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int

    def __post_init__(self):
        _require_positive_int(self.layers, name="layers")
        _require_positive_int(self.kv_heads, name="kv_heads")
        _require_positive_int(self.head_dim, name="head_dim")
        _require_positive_int(self.dtype_bytes, name="dtype_bytes")
        if not self.name:
            raise ValueError("name must be non-empty")

    @property
    def kv_bytes_per_token(self):
        """Bytes of KV cache one token occupies across the whole model.

        The leading factor two is key and value.
        """
        return 2 * self.layers * self.kv_heads * self.head_dim * self.dtype_bytes


@dataclass(frozen=True)
class DecodeStepFit:
    """Measured affine fit of one decode step.

    step_ms(total_kv_tokens) = constant_ms + per_token_us * total_kv_tokens / 1000

    `total_kv_tokens` is summed over the batch, so a batch of `B` sequences each
    holding `L` tokens contributes `L * B`.
    """

    constant_ms: float
    per_token_us: float

    def __post_init__(self):
        _require_positive(self.constant_ms, name="constant_ms")
        _require_positive(self.per_token_us, name="per_token_us", allow_zero=True)

    def step_ms(self, context_length, decode_batch, attention_time_ratio=1.0):
        length = _require_positive_int(context_length, name="context_length")
        batch = _require_positive_int(decode_batch, name="decode_batch")
        phi = _require_positive(
            attention_time_ratio, name="attention_time_ratio", allow_zero=True
        )
        kv_term_ms = self.per_token_us * length * batch / 1000.0
        return self.constant_ms + phi * kv_term_ms

    def kv_attention_share(self, context_length, decode_batch):
        """Fraction of one uncompressed decode step attributable to KV attention."""
        total = self.step_ms(context_length, decode_batch)
        return (total - self.constant_ms) / total


@dataclass(frozen=True)
class TurnProfile:
    """One agent turn, as measured on real traces.

    `decode_steps` is the number of tokens the actor emits for the turn.
    `warm_prefill_seconds` is what the engine actually recomputes when block-hash
    prefix caching hits, not a cold prefill. Using a cold prefill here is the
    exact error that invalidated the previous line.
    """

    warm_prefill_seconds: float
    decode_steps: int
    tool_seconds: float

    def __post_init__(self):
        _require_positive(
            self.warm_prefill_seconds, name="warm_prefill_seconds", allow_zero=True
        )
        _require_positive_int(self.decode_steps, name="decode_steps")
        _require_positive(self.tool_seconds, name="tool_seconds", allow_zero=True)


@dataclass(frozen=True)
class DeviceBudget:
    """What one device has to spend."""

    kv_bytes: int
    utilization_target: float
    host_bytes: int = 0
    host_bandwidth_bytes_per_second: float = 0.0

    def __post_init__(self):
        _require_positive_int(self.kv_bytes, name="kv_bytes")
        rho = _require_positive(self.utilization_target, name="utilization_target")
        if rho > 1.0:
            raise ValueError("utilization_target must not exceed 1")
        _require_positive(self.host_bytes, name="host_bytes", allow_zero=True)
        _require_positive(
            self.host_bandwidth_bytes_per_second,
            name="host_bandwidth_bytes_per_second",
            allow_zero=True,
        )

    @property
    def kv_gib(self):
        return self.kv_bytes / _BYTES_PER_GIB


@dataclass(frozen=True)
class CompressionSpec:
    """A KV representation, described only by what the model can price."""

    name: str
    kv_ratio: float = 1.0
    attention_time_ratio: float = 1.0
    lossless: bool = True

    def __post_init__(self):
        if not self.name:
            raise ValueError("name must be non-empty")
        ratio = _require_positive(self.kv_ratio, name="kv_ratio")
        if ratio < 1.0:
            raise ValueError("kv_ratio must be at least 1; it is a shrink factor")
        _require_positive(
            self.attention_time_ratio,
            name="attention_time_ratio",
            allow_zero=True,
        )
        if not isinstance(self.lossless, bool):
            raise ValueError("lossless must be a bool")


@dataclass(frozen=True)
class CapacityPoint:
    """Fully derived capacity of one (workload, representation) pair."""

    context_length: int
    decode_batch: int
    compression: str
    offload: bool
    kv_bytes_per_token: float
    bytes_per_agent: float
    step_ms: float
    kv_attention_share: float
    restore_seconds: float
    gpu_phase_seconds: float
    user_latency_seconds: float
    turn_period_seconds: float
    duty_compute: float
    duty_memory: float
    sustained_agents: float
    memory_ceiling: float
    host_ceiling: float
    concurrent_agents: float
    binding_constraint: str
    feasible: bool

    def as_dict(self):
        return {
            "context_length": self.context_length,
            "decode_batch": self.decode_batch,
            "compression": self.compression,
            "offload": self.offload,
            "kv_bytes_per_token": round(self.kv_bytes_per_token, 6),
            "bytes_per_agent": round(self.bytes_per_agent, 3),
            "step_ms": round(self.step_ms, 6),
            "kv_attention_share": round(self.kv_attention_share, 6),
            "restore_seconds": round(self.restore_seconds, 6),
            "gpu_phase_seconds": round(self.gpu_phase_seconds, 6),
            "user_latency_seconds": round(self.user_latency_seconds, 6),
            "turn_period_seconds": round(self.turn_period_seconds, 6),
            "duty_compute": round(self.duty_compute, 6),
            "duty_memory": round(self.duty_memory, 6),
            "sustained_agents": round(self.sustained_agents, 6),
            "memory_ceiling": round(self.memory_ceiling, 6),
            "host_ceiling": (
                None if self.host_ceiling == float("inf")
                else round(self.host_ceiling, 6)
            ),
            "concurrent_agents": round(self.concurrent_agents, 6),
            "binding_constraint": self.binding_constraint,
            "feasible": self.feasible,
        }


def evaluate_point(
    *,
    geometry,
    fit,
    turn,
    budget,
    compression,
    context_length,
    decode_batch,
    offload=False,
):
    """Derive serving capacity for one workload point.

    `offload` models the *lossless* competitor: idle KV is parked in host memory
    and restored over the host link at the start of the turn. The restore is DMA,
    so it is charged to user latency and to memory residency, but not to the GPU
    compute budget.
    """
    if not isinstance(offload, bool):
        raise ValueError("offload must be a bool")
    length = _require_positive_int(context_length, name="context_length")
    batch = _require_positive_int(decode_batch, name="decode_batch")

    bytes_per_token = geometry.kv_bytes_per_token / compression.kv_ratio
    bytes_per_agent = bytes_per_token * length

    step_ms = fit.step_ms(length, batch, compression.attention_time_ratio)
    baseline_share = fit.kv_attention_share(length, batch)

    decode_seconds = turn.decode_steps * step_ms / 1000.0
    gpu_phase_seconds = turn.warm_prefill_seconds + decode_seconds

    if offload:
        if budget.host_bandwidth_bytes_per_second <= 0.0:
            raise ValueError("offload requires a positive host bandwidth")
        restore_seconds = bytes_per_agent / budget.host_bandwidth_bytes_per_second
        host_ceiling = budget.host_bytes / bytes_per_agent
    else:
        restore_seconds = 0.0
        host_ceiling = float("inf")

    user_latency_seconds = restore_seconds + gpu_phase_seconds
    turn_period_seconds = user_latency_seconds + turn.tool_seconds

    duty_compute = gpu_phase_seconds / turn_period_seconds
    # Without an offload path the KV cannot be released during the tool wait,
    # so it is resident for the entire period.
    duty_memory = (
        (restore_seconds + gpu_phase_seconds) / turn_period_seconds if offload else 1.0
    )

    # Steady state: sustaining a mean decode batch of `batch` requires this many
    # agents in flight. utilization_target is headroom for burstiness.
    sustained_agents = budget.utilization_target * batch / duty_compute

    resident_slots = budget.kv_bytes / bytes_per_agent
    memory_ceiling = resident_slots / duty_memory
    host_slots = host_ceiling / duty_memory if offload else float("inf")

    ceilings = (
        ("demand", sustained_agents),
        ("memory", memory_ceiling),
        ("host", host_slots),
    )
    binding_constraint, concurrent_agents = min(ceilings, key=lambda item: item[1])

    # A mean decode batch of `batch` needs at least `batch` agents to exist.
    # If the binding resource cannot hold them, this batch is unreachable.
    feasible = concurrent_agents >= float(batch)

    return CapacityPoint(
        context_length=length,
        decode_batch=batch,
        compression=compression.name,
        offload=offload,
        kv_bytes_per_token=bytes_per_token,
        bytes_per_agent=bytes_per_agent,
        step_ms=step_ms,
        kv_attention_share=baseline_share,
        restore_seconds=restore_seconds,
        gpu_phase_seconds=gpu_phase_seconds,
        user_latency_seconds=user_latency_seconds,
        turn_period_seconds=turn_period_seconds,
        duty_compute=duty_compute,
        duty_memory=duty_memory,
        sustained_agents=sustained_agents,
        memory_ceiling=memory_ceiling,
        host_ceiling=host_ceiling,
        concurrent_agents=concurrent_agents,
        binding_constraint=binding_constraint,
        feasible=feasible,
    )


def capacity_gain(candidate, reference):
    """Capacity of `candidate` relative to `reference`.

    Both arguments are `CapacityPoint`. The comparison is only meaningful at a
    fixed workload point, so a mismatch is an error rather than a silent ratio.
    """
    if candidate.context_length != reference.context_length:
        raise ValueError("capacity_gain requires a common context_length")
    if candidate.decode_batch != reference.decode_batch:
        raise ValueError("capacity_gain requires a common decode_batch")
    return candidate.concurrent_agents / reference.concurrent_agents
