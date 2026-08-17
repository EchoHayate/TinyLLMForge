from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Qwen35HybridPrefixEnginePublicationTransaction:
    payloads: tuple
    prepare_results: tuple[dict, ...] = ()
    precommit_results: tuple[dict, ...] = ()
    finalize_results: tuple[dict, ...] = ()
    seal_results: tuple[dict, ...] = ()
    rollback_results: tuple[dict, ...] = ()
    state: str = "created"


class Qwen35HybridPrefixEnginePublicationCoordinator:

    def __init__(self, engine, *, timeout_s):
        if engine is None:
            raise ValueError("engine must be provided")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be positive")
        world_size = getattr(
            getattr(engine, "model_runner", None),
            "world_size",
            None,
        )
        if (
            isinstance(world_size, bool)
            or not isinstance(world_size, int)
            or world_size <= 0
        ):
            raise ValueError(
                "engine ModelRunner world_size must be positive"
            )
        self.engine = engine
        self.timeout_s = float(timeout_s)
        self.world_size = world_size
        self._poisoned_error = None
        self.last_transaction: Optional[
            Qwen35HybridPrefixEnginePublicationTransaction
        ] = None

    def _ensure_healthy(self):
        if self._poisoned_error is not None:
            raise RuntimeError(
                "Engine hybrid prefix publication coordinator is "
                f"poisoned: {self._poisoned_error}"
            )

    def _poison(self, error):
        reason = str(error)
        if self._poisoned_error is None:
            self._poisoned_error = reason
        poison_transport = getattr(
            self.engine,
            "_poison_model_runner_ack_collector",
            None,
        )
        if poison_transport is not None:
            poison_transport(reason)

    def _validate_rows(
        self,
        transaction,
        rows,
        operation,
        statuses,
    ):
        rows = tuple(rows)
        if len(rows) != self.world_size:
            raise RuntimeError(
                f"publication {operation} result count is incomplete"
            )
        ticket_id = transaction.payloads[0].ticket_id
        for participant_id, row in enumerate(rows):
            if (
                not isinstance(row, dict)
                or set(row)
                != {
                    "ticket_id",
                    "participant_id",
                    "operation",
                    "status",
                    "detail",
                }
                or row["ticket_id"] != ticket_id
                or row["participant_id"] != participant_id
                or row["operation"] != operation
                or row["status"] not in statuses
                or not isinstance(row["detail"], str)
            ):
                raise RuntimeError(
                    f"publication {operation} result is invalid"
                )
        return rows

    def _rollback(self, transaction):
        try:
            rows = (
                self.engine
                .rollback_model_runner_hybrid_prefix_publication(
                    transaction.payloads,
                    timeout_s=self.timeout_s,
                )
            )
            rows = self._validate_rows(
                transaction,
                rows,
                "rollback",
                {"rolled_back", "error"},
            )
        except BaseException as error:
            self._poison(error)
            raise RuntimeError(
                f"publication rollback failed: {error}"
            ) from error
        failed = next(
            (
                row
                for row in rows
                if row["status"] != "rolled_back"
            ),
            None,
        )
        if failed is not None:
            error = RuntimeError(
                "publication rollback failed: "
                f"participant={failed['participant_id']}, "
                f"detail={failed['detail']}"
            )
            self._poison(error)
            raise error
        transaction.rollback_results = rows
        transaction.state = "rolled_back"

    def _phase(
        self,
        transaction,
        operation,
        success_status,
    ):
        method = getattr(
            self.engine,
            f"{operation}_model_runner_hybrid_prefix_publication",
        )
        try:
            rows = method(
                transaction.payloads,
                timeout_s=self.timeout_s,
            )
            rows = self._validate_rows(
                transaction,
                rows,
                operation,
                {success_status, "error"},
            )
        except BaseException as error:
            self._poison(error)
            raise
        setattr(transaction, f"{operation}_results", rows)
        return rows

    def publish(self, payloads) -> bool:
        self._ensure_healthy()
        payloads = (
            self.engine
            ._validate_hybrid_prefix_publication_payloads(payloads)
        )
        transaction = Qwen35HybridPrefixEnginePublicationTransaction(
            payloads=payloads
        )
        self.last_transaction = transaction

        try:
            prepare_rows = (
                self.engine
                .prepare_model_runner_hybrid_prefix_publication(
                    payloads,
                    timeout_s=self.timeout_s,
                )
            )
            prepare_rows = self._validate_rows(
                transaction,
                prepare_rows,
                "prepare",
                {"prepared", "rejected", "error"},
            )
        except BaseException as error:
            self._poison(error)
            raise
        transaction.prepare_results = prepare_rows
        transaction.state = "prepared"
        if any(
            row["status"] == "rejected"
            for row in prepare_rows
        ):
            self._rollback(transaction)
            return False
        prepare_error = next(
            (
                row
                for row in prepare_rows
                if row["status"] != "prepared"
            ),
            None,
        )
        if prepare_error is not None:
            self._rollback(transaction)
            raise RuntimeError(
                "publication prepare failed: "
                f"participant={prepare_error['participant_id']}, "
                f"detail={prepare_error['detail']}"
            )

        for operation, success_status in (
            ("precommit", "precommitted"),
            ("finalize", "finalized"),
        ):
            rows = self._phase(
                transaction,
                operation,
                success_status,
            )
            failed = next(
                (
                    row
                    for row in rows
                    if row["status"] != success_status
                ),
                None,
            )
            if failed is not None:
                self._rollback(transaction)
                raise RuntimeError(
                    f"publication {operation} failed: "
                    f"participant={failed['participant_id']}, "
                    f"detail={failed['detail']}"
                )
            transaction.state = (
                "precommitted"
                if operation == "precommit"
                else "finalized"
            )

        seal_rows = self._phase(
            transaction,
            "seal",
            "committed",
        )
        failed = next(
            (
                row
                for row in seal_rows
                if row["status"] != "committed"
            ),
            None,
        )
        if failed is not None:
            error = RuntimeError(
                "publication seal failed: "
                f"participant={failed['participant_id']}, "
                f"detail={failed['detail']}"
            )
            self._poison(error)
            raise error
        transaction.state = "committed"
        return True
