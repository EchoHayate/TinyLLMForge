import copy
import unittest

from active_verification_state import replay
from test_active_verification_contract import (
    EPISODE, LIMITS, RUN, model_event, sample_view, tool_event,
)


def state(method, events=None, view=None):
    return replay(
        sample_view() if view is None else view, method, RUN, EPISODE,
        [] if events is None else events, LIMITS,
    )


class StateTests(unittest.TestCase):
    def test_keep_and_direct(self):
        kept = state("keep")
        self.assertTrue(kept["complete"])
        self.assertEqual((kept["selected"], kept["deployed"]), ("B", "B"))
        self.assertEqual(kept["cost"]["model_calls"], 0)
        self.assertIsNone(kept["next_action"])
        direct = state("direct", [model_event("C")])
        self.assertEqual((direct["status"], direct["selected"]), ("valid", "C"))

    def test_active_direct_or_one_tool_then_final(self):
        pending = state("active_verify")
        self.assertEqual(pending["next_action"]["allowed"], ["A", "B", "C", "VERIFY"])
        direct = state("active_verify", [model_event("A")])
        self.assertTrue(direct["complete"])
        self.assertEqual(direct["cost"]["tool_calls"], 0)
        transcript = [model_event("VERIFY")]
        self.assertEqual(state("active_verify", transcript)["next_action"]["kind"], "tool")
        transcript.append(tool_event())
        following = state("active_verify", transcript)["next_action"]
        self.assertEqual((following["kind"], following["round"]), ("model", 2))
        self.assertEqual(following["allowed"], ["A", "B", "C"])
        transcript.append(model_event("A"))
        final = state("active_verify", transcript)
        self.assertEqual(final["selected"], "A")
        self.assertEqual(final["cost"]["model_calls"], 2)
        self.assertEqual(final["cost"]["tool_calls"], 1)
        self.assertEqual(final["cost"]["executed_cases"], 1)
        self.assertEqual(final["cost"]["component_wall_ns"], 27)

    def test_rethink_has_two_models_no_tool(self):
        pending = state("rethink", [model_event("B")])
        self.assertFalse(pending["complete"])
        self.assertEqual(pending["next_action"]["stage"], "final")
        final = state("rethink", [model_event("B"), model_event("A")])
        self.assertEqual(final["selected"], "A")
        self.assertEqual(final["cost"]["model_calls"], 2)
        self.assertEqual(final["cost"]["tool_calls"], 0)

    def test_always_tool_does_not_spend_a_decision_call(self):
        self.assertEqual(state("always_verify")["next_action"]["kind"], "tool")
        final = state("always_verify", [
            tool_event(method="always_verify"), model_event("A"),
        ])
        self.assertEqual(final["selected"], "A")
        self.assertEqual(final["cost"]["model_calls"], 1)
        self.assertEqual(final["cost"]["tool_calls"], 1)

    def test_tool_failure_still_allows_final_choice_without_retry(self):
        for status in ("error", "timeout", "unavailable"):
            with self.subTest(status=status):
                events = [model_event("VERIFY"), tool_event(status=status), model_event("B")]
                final = state("active_verify", events)
                self.assertEqual(final["selected"], "B")
                self.assertEqual(final["cost"]["tool_calls"], 1)
                self.assertEqual(final["events"][1]["status"], status)

    def test_invalid_or_failed_model_stops_and_falls_back(self):
        events = (
            model_event("A or B"),
            model_event(""),
            model_event("A", terminated=False),
            model_event(status="error"),
            model_event(status="timeout"),
        )
        for event in events:
            with self.subTest(event=event):
                final = state("active_verify", [event])
                self.assertTrue(final["complete"])
                self.assertIsNone(final["selected"])
                self.assertEqual(final["deployed"], "B")
                self.assertIsNone(final["next_action"])
                with self.assertRaises(ValueError):
                    state("active_verify", [event, model_event("A")])
        with self.assertRaises(ValueError):
            state("rethink", [model_event("VERIFY"), model_event("A")])

    def test_second_verify_is_invalid_and_cannot_launch_more_tools(self):
        events = [model_event("VERIFY"), tool_event(), model_event("VERIFY")]
        final = state("active_verify", events)
        self.assertEqual(final["status"], "invalid")
        self.assertEqual(final["cost"]["tool_calls"], 1)
        for extra in (model_event("A"), tool_event()):
            with self.assertRaises(ValueError):
                state("active_verify", events + [extra])

    def test_wrong_order_duplicate_and_cross_identity(self):
        invalid = (
            ("keep", [model_event()]),
            ("direct", [tool_event()]),
            ("active_verify", [tool_event()]),
            ("always_verify", [model_event()]),
            ("active_verify", [model_event("VERIFY"), tool_event(), tool_event()]),
            ("active_verify", [model_event("VERIFY"), tool_event(method="always_verify")]),
        )
        for method, events in invalid:
            with self.subTest(method=method), self.assertRaises(ValueError):
                state(method, events)
        with self.assertRaises(ValueError):
            replay(sample_view(), "active_verify", "3" * 32, EPISODE,
                   [model_event("VERIFY"), tool_event()], LIMITS)

    def test_unknown_cost_is_not_zero_and_prefix_is_not_complete(self):
        event = model_event(status="error")
        event.update(input_ids=None, output_ids=None, wall_ns=None)
        final = state("direct", [event])
        self.assertIsNone(final["cost"]["input_tokens"])
        self.assertIsNone(final["cost"]["component_wall_ns"])
        self.assertEqual(final["cost"]["observed_component_wall_ns"], 0)
        self.assertIsNone(final["cost"]["episode_wall_ns"])
        self.assertFalse(state("active_verify", [model_event("VERIFY")])["complete"])

    def test_token_budget_and_isolation(self):
        event = model_event()
        event["input_ids"] = [11] * 9
        self.assertEqual(state("active_verify", [event])["status"], "budget_exceeded")
        view = sample_view()
        events = [model_event("VERIFY"), tool_event()]
        before = copy.deepcopy((view, events))
        first = state("active_verify", events, view)
        self.assertEqual(first, state("active_verify", events, view))
        first["events"][0]["text"] = "changed"
        first["next_action"]["view"]["feedback"].clear()
        self.assertEqual((view, events), before)
        fresh = state("active_verify")
        self.assertEqual(fresh["next_action"]["history"], [])
        self.assertEqual(fresh["evidence_kind"], "SYNTHETIC_ONLY")
        self.assertFalse(fresh["real_execution_eligible"])
        self.assertFalse(fresh["benchmark_eligible"])


if __name__ == "__main__":
    unittest.main()
