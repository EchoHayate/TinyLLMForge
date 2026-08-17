from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import sys


def _default_engine_factory(
    configuration,
    *,
    max_num_batched_tokens=4096,
    max_num_seqs=2,
):
    module = importlib.import_module("tinyvllm.engine.llm_engine")
    environment = {
        "CUDA_VISIBLE_DEVICES": ",".join(
            str(index) for index in configuration.gpu_indices
        ),
        "TINYVLLM_DIST_PORT": str(configuration.dist_port),
        "MASTER_PORT": str(configuration.master_port),
    }
    previous = {
        name: os.environ.get(name)
        for name in environment
    }
    try:
        os.environ.update(environment)
        return module.LLMEngine(
            configuration.model_dir,
            tensor_parallel_size=len(configuration.gpu_indices),
            max_model_len=4096,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _default_reference_token_provider(**kwargs):
    raise RuntimeError(
        "independent reference token provider is not configured"
    )


def _sampling_params(max_tokens):
    module_name = "tinyvllm.sampling_params"
    module = sys.modules.get(module_name)
    if module is None:
        path = (
            Path(__file__).resolve().parents[1]
            / "tinyvllm"
            / "sampling_params.py"
        )
        spec = importlib.util.spec_from_file_location(
            module_name,
            path,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )


class EngineBackendSession:

    def __init__(
        self,
        configuration,
        *,
        scenario,
        expected,
        engine_factory=_default_engine_factory,
        reference_token_provider=_default_reference_token_provider,
    ):
        if not callable(engine_factory):
            raise TypeError("engine_factory must be callable")
        if not callable(reference_token_provider):
            raise TypeError(
                "reference_token_provider must be callable"
            )
        self.configuration = configuration
        self.scenario = scenario
        self.expected = dict(expected)
        self.engine_factory = engine_factory
        self.reference_token_provider = reference_token_provider
        self.engine = None
        self.cleanup_receipt = None
        self.closed = False
        self.payload = None
        self.observation_baseline = None
        self.pending_prompt_token_ids = None
        self.pending_generated_tokens = None

    def _validate_action_context(self, scenario, expected):
        if scenario != self.scenario or expected != self.expected:
            raise ValueError("Engine backend scenario context mismatch")

    def _require_engine(self):
        if self.engine is None:
            raise RuntimeError("Engine backend is not constructed")
        return self.engine

    def _construct_engine(self):
        if self.engine is not None:
            raise RuntimeError("Engine backend is already constructed")
        self.engine = self.engine_factory(self.configuration)
        return {
            "engine_class": (
                "tinyvllm.engine.llm_engine.LLMEngine"
            ),
            "model_runner_class": (
                "tinyvllm.engine.model_runner.ModelRunner"
            ),
        }

    def _load_payload(self):
        if self.payload is None:
            module = importlib.import_module(
                "qwen35_tp4_engine_correctness_executor"
            )
            self.payload = module.build_scenario_payloads()[self.scenario]
        return self.payload

    def _configure_exact_restore(self):
        engine = self._require_engine()
        engine.configure_qwen35_hybrid_prefix_publication_runtime(
            model_fingerprint=self.configuration.model_fingerprint,
            max_entries=self.configuration.max_cache_entries,
            max_bytes=self.configuration.max_cache_bytes,
            timeout_s=self.configuration.timeout_s,
        )
        if self.scenario == "construct_and_bind":
            return {
                "scheduler_steps": 0,
                "model_runner_calls": 0,
                "output_token_ids": [],
                "reference_output_token_ids": [],
            }
        return {}

    def _authority_snapshot(self):
        engine = self._require_engine()
        rows = engine.qwen35_hybrid_prefix_authority_snapshots(
            timeout_s=self.configuration.timeout_s,
        )
        world_size = len(self.configuration.gpu_indices)
        if (
            not isinstance(rows, tuple)
            or len(rows) != world_size
            or [row.get("rank") for row in rows] != list(range(world_size))
        ):
            raise ValueError(
                "Engine authority snapshot rank inventory mismatch"
            )
        reference = {
            name: rows[0][name]
            for name in rows[0]
            if name != "rank"
        }
        if any(
            {
                name: row[name]
                for name in row
                if name != "rank"
            } != reference
            for row in rows[1:]
        ):
            raise ValueError(
                "Engine authority snapshot rank parity mismatch"
            )
        return dict(rows[0])

    def _verify_rank_bindings(self):
        engine = self._require_engine()
        world_size = len(self.configuration.gpu_indices)
        if engine.model_runner.world_size != world_size:
            raise ValueError("Engine ModelRunner world size mismatch")
        if len(engine.ps) != world_size - 1:
            raise ValueError("Engine worker process inventory mismatch")
        return {
            "rank_inventory": list(range(world_size)),
            "ack_ranks": list(range(1, world_size)),
        }

    def _begin_observation(self):
        if self.scenario == "construct_and_bind":
            self.observation_baseline = {
                "current_entries": 0,
                "hits": 0,
                "misses": 0,
                "publication_commits": 0,
                "invalidations": 0,
                "clears": 0,
                "last_publication_block_identities": [],
                "release_events": 0,
            }
            return {
                "publication_commits": 0,
                "restore_hits": 0,
                "restore_misses": 0,
                "release_events": 0,
                "cache_entries_after": 0,
                "cache_identity_match": True,
            }
        snapshot = self._authority_snapshot()
        snapshot["release_events"] = (
            self._require_engine().hybrid_state_release_event_count()
        )
        self.observation_baseline = snapshot
        return {}

    def _submit(self, prompt_token_ids, generated_tokens):
        if self.pending_prompt_token_ids is not None:
            raise RuntimeError("Engine request is already pending")
        prompt_token_ids = list(prompt_token_ids)
        self._require_engine().add_request(
            prompt_token_ids,
            _sampling_params(generated_tokens),
        )
        self.pending_prompt_token_ids = prompt_token_ids
        self.pending_generated_tokens = generated_tokens
        return {}

    def _submit_source_request(self):
        payload = self._load_payload()
        return self._submit(
            payload["source_prompt_token_ids"],
            payload["generated_tokens"],
        )

    def _submit_cached_continuation(self):
        payload = self._load_payload()
        return self._submit(
            payload["request_prompt_token_ids"],
            payload["generated_tokens"],
        )

    def _submit_token_mismatch(self):
        return self._submit_cached_continuation()

    def _run_pending_request(self, *, require_reference):
        if self.pending_prompt_token_ids is None:
            raise RuntimeError("Engine request is not submitted")
        engine = self._require_engine()
        scheduler_steps = 0
        output_token_ids = []
        while not engine.is_finished():
            outputs, _ = engine.step()
            scheduler_steps += 1
            for _, token_ids in outputs:
                token_ids = list(token_ids)
                if len(token_ids) >= self.pending_generated_tokens:
                    output_token_ids = token_ids
                else:
                    output_token_ids.extend(token_ids)
        prompt_token_ids = self.pending_prompt_token_ids
        generated_tokens = self.pending_generated_tokens
        self.pending_prompt_token_ids = None
        self.pending_generated_tokens = None
        if len(output_token_ids) != generated_tokens:
            raise ValueError(
                "Engine output token count mismatch"
            )
        if not require_reference:
            return output_token_ids, scheduler_steps, None
        reference = self.reference_token_provider(
            scenario=self.scenario,
            prompt_token_ids=list(prompt_token_ids),
            generated_tokens=generated_tokens,
        )
        if (
            not isinstance(reference, (list, tuple))
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in reference
            )
        ):
            raise ValueError(
                "independent reference token evidence is invalid"
            )
        return output_token_ids, scheduler_steps, list(reference)

    def _seed_source_fixture(self):
        payload = self._load_payload()
        self._submit(payload["source_prompt_token_ids"], 1)
        self._run_pending_request(require_reference=False)
        snapshot = self._authority_snapshot()
        if (
            snapshot["current_entries"] != 1
            or snapshot["publication_commits"] < 1
            or not snapshot["last_publication_block_identities"]
        ):
            raise ValueError(
                "Engine source fixture publication is incomplete"
            )
        return {}

    def _run_to_completion(self):
        outputs, scheduler_steps, reference = (
            self._run_pending_request(require_reference=True)
        )
        return {
            "scheduler_steps": scheduler_steps,
            "model_runner_calls": scheduler_steps,
            "output_token_ids": outputs,
            "reference_output_token_ids": reference,
        }

    def _invalidate_block_generation(self):
        snapshot = self._authority_snapshot()
        block_identities = snapshot[
            "last_publication_block_identities"
        ]
        if not block_identities:
            raise ValueError(
                "Engine stale invalidation blocks are unavailable"
            )
        self._require_engine().invalidate_qwen35_hybrid_prefix_blocks(
            block_identities,
            timeout_s=self.configuration.timeout_s,
        )
        return {}

    def _clear_reusable_cache(self):
        self._require_engine().clear_qwen35_hybrid_prefix_caches(
            timeout_s=self.configuration.timeout_s,
        )
        return {}

    def _snapshot_cache(self):
        if self.observation_baseline is None:
            raise RuntimeError("Engine observation is not started")
        snapshot = self._authority_snapshot()
        release_events = (
            self._require_engine().hybrid_state_release_event_count()
        )
        baseline = self.observation_baseline
        evidence = {
            "restore_hits": snapshot["hits"] - baseline["hits"],
            "restore_misses": snapshot["misses"] - baseline["misses"],
            "cache_entries_after": snapshot["current_entries"],
            "cache_identity_match": True,
        }
        if self.scenario != "publish_source":
            evidence["publication_commits"] = (
                snapshot["publication_commits"]
                - baseline["publication_commits"]
            )
        if self.scenario != "restore_w1":
            evidence["release_events"] = (
                release_events - baseline["release_events"]
            )
        return evidence

    def _verify_publication_commit(self):
        if self.observation_baseline is None:
            raise RuntimeError("Engine observation is not started")
        snapshot = self._authority_snapshot()
        return {
            "publication_commits": (
                snapshot["publication_commits"]
                - self.observation_baseline["publication_commits"]
            ),
        }

    def _drain_release_events(self):
        if self.observation_baseline is None:
            raise RuntimeError("Engine observation is not started")
        count = self._require_engine().hybrid_state_release_event_count()
        return {
            "release_events": (
                count - self.observation_baseline["release_events"]
            ),
        }

    def _close_engine(self):
        engine = self._require_engine()
        receipt = engine.exit()
        required = {
            "process_group_destroyed",
            "rank_exit_codes",
            "owned_children_remaining",
            "rank_cleanup_receipts",
        }
        if not isinstance(receipt, dict) or set(receipt) != required:
            raise ValueError("Engine cleanup receipt is invalid")
        rank_exit_codes = receipt["rank_exit_codes"]
        rank_receipts = receipt["rank_cleanup_receipts"]
        world_size = len(self.configuration.gpu_indices)
        if (
            receipt["process_group_destroyed"] is not True
            or rank_exit_codes != [0] * world_size
            or receipt["owned_children_remaining"] != []
            or not isinstance(rank_receipts, list)
            or [
                row.get("rank")
                for row in rank_receipts
                if isinstance(row, dict)
            ] != list(range(world_size))
            or any(
                row.get("process_group_destroyed") is not True
                for row in rank_receipts
            )
        ):
            raise ValueError("Engine cleanup receipt did not prove cleanup")
        self.cleanup_receipt = dict(receipt)
        return {"rank_exit_codes": list(rank_exit_codes)}

    def _verify_cleanup(self):
        if self.cleanup_receipt is None:
            raise RuntimeError("Engine cleanup receipt is unavailable")
        return {
            "process_group_destroyed": True,
            "owned_children_remaining": [],
        }

    def execute_action(self, *, action, scenario, expected):
        if self.closed:
            raise RuntimeError("Engine backend session is closed")
        self._validate_action_context(scenario, expected)
        actions = {
            "construct_engine": self._construct_engine,
            "configure_exact_restore": self._configure_exact_restore,
            "verify_rank_bindings": self._verify_rank_bindings,
            "begin_observation": self._begin_observation,
            "seed_source_fixture": self._seed_source_fixture,
            "submit_source_request": self._submit_source_request,
            "submit_cached_continuation": (
                self._submit_cached_continuation
            ),
            "submit_token_mismatch": self._submit_token_mismatch,
            "run_to_completion": self._run_to_completion,
            "verify_publication_commit": (
                self._verify_publication_commit
            ),
            "drain_release_events": self._drain_release_events,
            "snapshot_cache": self._snapshot_cache,
            "invalidate_block_generation": (
                self._invalidate_block_generation
            ),
            "clear_reusable_cache": self._clear_reusable_cache,
            "close_engine": self._close_engine,
            "verify_cleanup": self._verify_cleanup,
        }
        handler = actions.get(action)
        if handler is None:
            raise RuntimeError(
                f"Engine backend action is not implemented: {action}"
            )
        return handler()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.engine is not None and self.cleanup_receipt is None:
            self.cleanup_receipt = self.engine.exit()
