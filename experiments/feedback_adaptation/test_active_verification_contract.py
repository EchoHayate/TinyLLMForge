import copy
import unittest

from active_verification_contract import (
    model_action, probe_request, public_view, tool_view,
    validate_limits, validate_view,
)


RUN = "1" * 32
EPISODE = "2" * 32
LIMITS = {"max_input_tokens": 8, "max_output_tokens": 4}


def sample_view():
    return {
        "specification": "Synthetic specification; never execute sources.",
        "initial_source": "source-B",
        "candidates": {"A": "source-A", "B": "source-B", "C": "source-C"},
        "check_inputs": [{"id": "check-0", "input": "synthetic input"}],
        "feedback": ["A synthetic claim."],
    }


def model_event(text="A", status="completed", terminated=True):
    return {
        "kind": "model", "status": status, "text": text,
        "terminated": terminated, "input_ids": [11], "output_ids": [12],
        "wall_ns": 10,
    }


def tool_event(view=None, method="active_verify", status="completed"):
    view = sample_view() if view is None else view
    request = probe_request(view, RUN, EPISODE, method)
    results = [
        {
            **check, "expected": "expected", "actual": "observed",
            "status": "completed",
        }
        for check in view["check_inputs"]
    ] if status == "completed" else []
    return {
        "kind": "tool", "request_id": request["request_id"], "status": status,
        "results": results, "executed_cases": len(results), "wall_ns": 7,
    }


class ContractTests(unittest.TestCase):
    def test_projection_is_detached_and_private_free(self):
        envelope = {
            "evidence_kind": "synthetic", "public": sample_view(),
            "private": {"answer": "PRIVATE_SENTINEL"},
        }
        before = copy.deepcopy(envelope)
        projected = public_view(envelope)
        self.assertNotIn("PRIVATE_SENTINEL", repr(projected))
        projected["candidates"]["A"] = "changed"
        self.assertEqual(envelope, before)
        envelope["evidence_kind"] = "real"
        with self.assertRaises(ValueError):
            public_view(envelope)

    def test_invalid_public_shapes(self):
        mutations = (
            lambda view: view.update(private={}),
            lambda view: view.update(initial_source="absent"),
            lambda view: view["candidates"].update(C="source-A"),
            lambda view: view.update(check_inputs=[]),
            lambda view: view["check_inputs"].append(view["check_inputs"][0]),
            lambda view: view["check_inputs"][0].update(expected="leak"),
            lambda view: view.update(feedback=["first", "second"]),
            lambda view: view.update(specification="\ud800"),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                view = sample_view()
                mutate(view)
                validate_view(view)

    def test_strict_action_parsing(self):
        for text, expected in (
            (" A ", "A"), ("VERIFY", "VERIFY"), ("A or B", "invalid"),
            ("```A```", "invalid"), ("", "empty"), ("a", "invalid"),
        ):
            with self.subTest(text=text):
                self.assertEqual(model_action(model_event(text), True, LIMITS), expected)
        self.assertEqual(model_action(model_event("VERIFY"), False, LIMITS), "invalid")
        self.assertEqual(
            model_action(model_event("A", terminated=False), True, LIMITS), "truncated",
        )
        for status in ("error", "timeout"):
            self.assertEqual(model_action(model_event(status=status), True, LIMITS), status)

    def test_limits_unknown_cost_and_malformed_records(self):
        record = model_event()
        record["output_ids"] = [12] * 5
        self.assertEqual(model_action(record, False, LIMITS), "budget_exceeded")
        record = model_event(status="error")
        record.update(input_ids=None, output_ids=None, wall_ns=None)
        self.assertEqual(model_action(record, False, LIMITS), "error")
        for field, value in (
            ("wall_ns", True), ("wall_ns", -1), ("wall_ns", 1.5),
            ("input_ids", [True]), ("input_ids", []), ("output_ids", []),
            ("output_ids", None),
            ("terminated", 1), ("status", "unknown"), ("extra", 0),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                record = model_event()
                record[field] = value
                model_action(record, False, LIMITS)
        with self.assertRaises(ValueError):
            validate_limits({"max_input_tokens": True, "max_output_tokens": 4})

    def test_tool_projection_and_binding(self):
        view = sample_view()
        request = probe_request(view, RUN, EPISODE, "active_verify")
        record = tool_event()
        projected = tool_view(record, request)
        self.assertNotIn("request_id", projected)
        projected["results"][0]["actual"] = "changed"
        self.assertEqual(record["results"][0]["actual"], "observed")
        variants = (
            probe_request(view, "3" * 32, EPISODE, "active_verify"),
            probe_request(view, RUN, "4" * 32, "active_verify"),
            probe_request(view, RUN, EPISODE, "always_verify"),
        )
        for other in variants:
            with self.assertRaises(ValueError):
                tool_view(record, other)
        changed = copy.deepcopy(view)
        changed["initial_source"] = changed["candidates"]["A"]
        with self.assertRaises(ValueError):
            tool_view(record, probe_request(changed, RUN, EPISODE, "active_verify"))
        changed = copy.deepcopy(view)
        changed["check_inputs"][0]["input"] = "another input"
        with self.assertRaises(ValueError):
            tool_view(record, probe_request(changed, RUN, EPISODE, "active_verify"))

    def test_partial_tool_failure_and_strict_rows(self):
        view = sample_view()
        view["check_inputs"].append({"id": "check-1", "input": "second"})
        request = probe_request(view, RUN, EPISODE, "active_verify")
        record = tool_event(view)
        record.update(status="timeout", results=record["results"][:1], executed_cases=1)
        record["results"][0].update(status="timeout", actual=None)
        self.assertEqual(tool_view(record, request)["executed_cases"], 1)
        for mutate in (
            lambda event: event.update(status="completed"),
            lambda event: event.update(executed_cases=True),
            lambda event: event.update(executed_cases=2),
            lambda event: event["results"][0].update(actual="invented"),
            lambda event: event["results"][0].update(id="wrong"),
            lambda event: event.update(private="leak"),
        ):
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                changed = copy.deepcopy(record)
                mutate(changed)
                tool_view(changed, request)


if __name__ == "__main__":
    unittest.main()
