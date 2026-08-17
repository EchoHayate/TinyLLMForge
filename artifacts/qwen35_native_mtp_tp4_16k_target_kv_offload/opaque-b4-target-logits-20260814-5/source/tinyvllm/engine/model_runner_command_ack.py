from dataclasses import dataclass
import math
import time


_ACK_STATUSES = {
    "ok",
    "error",
}
_MAX_ERROR_BYTES = 4096


def _non_negative_integer(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_finite(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _bounded_text(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    encoded = value.encode("utf-8")
    if len(encoded) <= _MAX_ERROR_BYTES:
        return value
    return encoded[:_MAX_ERROR_BYTES].decode(
        "utf-8",
        errors="ignore",
    )


@dataclass(frozen=True)
class ModelRunnerCommandEnvelope:
    command_id: int
    method_name: str
    args: tuple
    requires_ack: bool

    def __post_init__(self):
        _non_negative_integer(self.command_id, "command_id")
        if (
            not isinstance(self.method_name, str)
            or not self.method_name
            or self.method_name.startswith("_")
        ):
            raise ValueError(
                "method_name must be a non-empty public method name"
            )
        if not isinstance(self.args, tuple):
            raise ValueError("args must be a tuple")
        if not isinstance(self.requires_ack, bool):
            raise ValueError("requires_ack must be a bool")


@dataclass(frozen=True)
class ModelRunnerCommandAck:
    command_id: int
    rank: int
    status: str
    result: object = None
    error_type: str = ""
    error_detail: str = ""

    def __post_init__(self):
        _non_negative_integer(self.command_id, "command_id")
        _positive_integer(self.rank, "rank")
        if self.status not in _ACK_STATUSES:
            raise ValueError(
                f"unsupported command acknowledgement status: {self.status}"
            )
        error_type = _bounded_text(self.error_type, "error_type")
        error_detail = _bounded_text(
            self.error_detail,
            "error_detail",
        )
        if self.status == "ok":
            if error_type or error_detail:
                raise ValueError(
                    "successful acknowledgement cannot contain error detail"
                )
        else:
            if self.result is not None:
                raise ValueError(
                    "error acknowledgement result must be None"
                )
            if not error_type:
                raise ValueError(
                    "error acknowledgement requires error_type"
                )
        object.__setattr__(self, "error_type", error_type)
        object.__setattr__(self, "error_detail", error_detail)


def execute_acknowledged_command(
    envelope: ModelRunnerCommandEnvelope,
    *,
    rank: int,
    target: object,
    send_ack,
):
    if not isinstance(envelope, ModelRunnerCommandEnvelope):
        raise ValueError(
            "envelope must be a ModelRunnerCommandEnvelope"
        )
    _positive_integer(rank, "rank")
    method = getattr(target, envelope.method_name, None)
    if method is None or not callable(method):
        if not envelope.requires_ack:
            raise AttributeError(
                f"unknown command method: {envelope.method_name}"
            )
        acknowledgement = ModelRunnerCommandAck(
            command_id=envelope.command_id,
            rank=rank,
            status="error",
            error_type="AttributeError",
            error_detail=(
                f"unknown command method: {envelope.method_name}"
            ),
        )
        send_ack(acknowledgement)
        return None

    if not envelope.requires_ack:
        return method(*envelope.args)

    try:
        result = method(*envelope.args)
    except Exception as error:
        acknowledgement = ModelRunnerCommandAck(
            command_id=envelope.command_id,
            rank=rank,
            status="error",
            error_type=type(error).__name__,
            error_detail=str(error),
        )
        send_ack(acknowledgement)
        return None

    acknowledgement = ModelRunnerCommandAck(
        command_id=envelope.command_id,
        rank=rank,
        status="ok",
        result=result,
    )
    send_ack(acknowledgement)
    return result


class ModelRunnerCommandAckCollector:

    def __init__(
        self,
        receivers,
        *,
        clock=time.monotonic,
        sleeper=time.sleep,
        poll_interval_s=0.001,
    ):
        if not isinstance(receivers, tuple) or not receivers:
            raise ValueError(
                "receivers must be a non-empty tuple"
            )
        receiver_map = {}
        for item in receivers:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(
                    "each receiver must be a rank/connection pair"
                )
            rank, receiver = item
            _positive_integer(rank, "receiver rank")
            if rank in receiver_map:
                raise ValueError(
                    f"duplicate receiver rank: {rank}"
                )
            if (
                not hasattr(receiver, "poll")
                or not callable(receiver.poll)
                or not hasattr(receiver, "recv")
                or not callable(receiver.recv)
            ):
                raise ValueError(
                    "receiver must provide poll() and recv()"
                )
            receiver_map[rank] = receiver
        ordered_ranks = tuple(sorted(receiver_map))
        if (
            ordered_ranks[0] != 1
            or ordered_ranks != tuple(
            range(ordered_ranks[0], ordered_ranks[-1] + 1)
            )
        ):
            raise ValueError(
                "receiver ranks must be contiguous from rank 1"
            )
        if not callable(clock):
            raise ValueError("clock must be callable")
        if not callable(sleeper):
            raise ValueError("sleeper must be callable")
        self.poll_interval_s = _positive_finite(
            poll_interval_s,
            "poll_interval_s",
        )
        self._receivers = receiver_map
        self._clock = clock
        self._sleeper = sleeper
        self._poisoned_error = None

    @property
    def poisoned(self):
        return self._poisoned_error is not None

    def _poison(self, error_type, detail):
        if self._poisoned_error is None:
            self._poisoned_error = (
                str(error_type),
                str(detail),
            )

    def poison(self, reason):
        if not isinstance(reason, str) or not reason:
            raise ValueError("poison reason must be a non-empty string")
        self._poison("external", reason)

    def _ensure_healthy(self):
        if self._poisoned_error is not None:
            error_type, detail = self._poisoned_error
            raise RuntimeError(
                "model runner acknowledgement collector is poisoned: "
                f"{error_type}: {detail}"
            )

    def _fail(self, exception_type, message):
        self._poison(exception_type.__name__, message)
        raise exception_type(message)

    def _validate_collect(
        self,
        command_id,
        expected_ranks,
        timeout_s,
        is_rank_alive,
    ):
        self._ensure_healthy()
        _non_negative_integer(command_id, "command_id")
        if (
            not isinstance(expected_ranks, tuple)
            or not expected_ranks
        ):
            raise ValueError(
                "expected_ranks must be a non-empty tuple"
            )
        for rank in expected_ranks:
            _positive_integer(rank, "expected rank")
        if len(set(expected_ranks)) != len(expected_ranks):
            raise ValueError("expected ranks must be unique")
        if tuple(sorted(expected_ranks)) != expected_ranks:
            raise ValueError("expected ranks must be sorted")
        unknown = set(expected_ranks).difference(self._receivers)
        if unknown:
            raise ValueError(
                "expected rank has no receiver: "
                + ", ".join(str(rank) for rank in sorted(unknown))
            )
        timeout_s = _positive_finite(timeout_s, "timeout_s")
        if not callable(is_rank_alive):
            raise ValueError("is_rank_alive must be callable")
        return timeout_s

    def collect(
        self,
        command_id,
        *,
        expected_ranks,
        timeout_s,
        is_rank_alive,
    ):
        timeout_s = self._validate_collect(
            command_id,
            expected_ranks,
            timeout_s,
            is_rank_alive,
        )
        deadline = self._clock() + timeout_s
        acknowledgements = {}

        while len(acknowledgements) < len(expected_ranks):
            made_progress = False
            for rank in expected_ranks:
                if rank in acknowledgements:
                    continue
                receiver = self._receivers[rank]
                try:
                    ready = receiver.poll(0.0)
                except Exception as error:
                    self._fail(
                        RuntimeError,
                        f"rank {rank} acknowledgement poll failed: {error}",
                    )
                if ready:
                    try:
                        acknowledgement = receiver.recv()
                    except Exception as error:
                        self._fail(
                            RuntimeError,
                            "rank "
                            f"{rank} acknowledgement receive failed: {error}",
                        )
                    if not isinstance(
                        acknowledgement,
                        ModelRunnerCommandAck,
                    ):
                        self._fail(
                            RuntimeError,
                            "rank "
                            f"{rank} returned malformed acknowledgement",
                        )
                    if acknowledgement.rank != rank:
                        self._fail(
                            RuntimeError,
                            "acknowledgement rank mismatch: "
                            f"pipe={rank}, message={acknowledgement.rank}",
                        )
                    if acknowledgement.command_id != command_id:
                        self._fail(
                            RuntimeError,
                            "acknowledgement command mismatch: "
                            f"expected={command_id}, "
                            f"received={acknowledgement.command_id}, "
                            f"rank={rank}",
                        )
                    if rank in acknowledgements:
                        self._fail(
                            RuntimeError,
                            f"duplicate acknowledgement from rank {rank}",
                        )
                    if acknowledgement.status == "error":
                        self._fail(
                            RuntimeError,
                            "worker command failed: "
                            f"rank={rank}, "
                            f"type={acknowledgement.error_type}, "
                            f"detail={acknowledgement.error_detail}",
                        )
                    acknowledgements[rank] = acknowledgement
                    made_progress = True
                    continue
                try:
                    alive = is_rank_alive(rank)
                except Exception as error:
                    self._fail(
                        RuntimeError,
                        f"rank {rank} liveness check failed: {error}",
                    )
                if not isinstance(alive, bool):
                    self._fail(
                        RuntimeError,
                        f"rank {rank} liveness result must be bool",
                    )
                if not alive:
                    self._fail(
                        RuntimeError,
                        "worker rank is not alive while waiting for "
                        f"acknowledgement: {rank}",
                    )

            if len(acknowledgements) == len(expected_ranks):
                break
            now = self._clock()
            if now >= deadline:
                missing = tuple(
                    rank
                    for rank in expected_ranks
                    if rank not in acknowledgements
                )
                self._fail(
                    TimeoutError,
                    "timed out waiting for model runner "
                    "acknowledgements; missing ranks: "
                    + ", ".join(str(rank) for rank in missing),
                )
            if not made_progress:
                self._sleeper(min(
                    self.poll_interval_s,
                    max(0.0, deadline - now),
                ))

        return tuple(
            acknowledgements[rank]
            for rank in expected_ranks
        )
