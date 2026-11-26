from __future__ import annotations

import json
from pathlib import Path
import unittest


class CrossDomainContractsTests(unittest.TestCase):
    def test_med_and_legal_rule_engines_disjoint_and_zero_oracle_leak(self):
        from experiments.cross_domain_prefix_lora import domain_sop_contracts as contracts

        built = contracts.build_and_verify_cross_domain_dataset()
        pub = built["public_bundle"]
        orc = built["blind_oracles"]
        manifest = built["manifest"]

        self.assertEqual(len(pub["med_triage"]["seeds"]), 8)
        self.assertEqual(len(pub["med_triage"]["unlabeled"]), 20)
        self.assertEqual(len(pub["med_triage"]["blind_test_inputs"]), 16)

        self.assertEqual(len(pub["legal_compliance"]["seeds"]), 8)
        self.assertEqual(len(pub["legal_compliance"]["unlabeled"]), 20)
        self.assertEqual(len(pub["legal_compliance"]["blind_test_inputs"]), 16)

        self.assertEqual(len(pub["general_retention"]["blind_test_inputs"]), 16)
        self.assertEqual(len(pub["background_anchors"]), 16)
        self.assertTrue(manifest["zero_blind_oracle_in_public_bundle"])

        pub_raw = json.dumps(pub, sort_keys=True)
        for dom_key in ("med_triage", "legal_compliance"):
            blind_ids = {item["id"] for item in pub[dom_key]["blind_test_inputs"]}
            self.assertEqual(set(orc[dom_key].keys()), blind_ids)
            for b_item in pub[dom_key]["blind_test_inputs"]:
                self.assertEqual(set(b_item.keys()), {"id", "input", "vignette"})
            for u_item in pub[dom_key]["unlabeled"]:
                self.assertEqual(set(u_item.keys()), {"id", "input", "vignette"})

        self.assertGreaterEqual(len(pub["med_triage"]["sop_manual_text"]), 5500)
        self.assertGreaterEqual(len(pub["legal_compliance"]["sop_manual_text"]), 5000)

    def test_leave_one_out_rendering_and_deterministic_scorers(self):
        from experiments.cross_domain_prefix_lora import domain_sop_contracts as contracts

        built = contracts.build_and_verify_cross_domain_dataset()
        pub = built["public_bundle"]
        orc = built["blind_oracles"]

        s01 = pub["med_triage"]["seeds"][0]
        full_msgs = contracts.render_domain_long_messages("med_triage", s01, pub, exclude_seed_id=None)
        loo_msgs = contracts.render_domain_long_messages("med_triage", s01, pub, exclude_seed_id=s01["id"])
        self.assertIn(f"--- Worked Case {s01['id']} ---", full_msgs[0]["content"])
        self.assertNotIn(f"--- Worked Case {s01['id']} ---", loo_msgs[0]["content"])
        self.assertNotIn(contracts.canonical_json(s01["expected"]), loo_msgs[0]["content"])

        m_t01 = pub["med_triage"]["blind_test_inputs"][0]
        m_ans = orc["med_triage"][m_t01["id"]]
        perfect_json = json.dumps(m_ans, sort_keys=False)
        sc_ok = contracts.score_structured_sop_prediction("med_triage", perfect_json, m_ans)
        self.assertTrue(sc_ok["strict_pass"])
        self.assertEqual(sc_ok["matched_fields"], 5)

        wrong_dose = dict(m_ans)
        wrong_dose["dose_mg"] = int(m_ans["dose_mg"]) + 111
        sc_partial = contracts.score_structured_sop_prediction(
            "med_triage", json.dumps(wrong_dose), m_ans
        )
        self.assertFalse(sc_partial["strict_pass"])
        self.assertEqual(sc_partial["matched_fields"], 4)

        g01 = pub["general_retention"]["blind_test_inputs"][0]
        g_exp = orc["general_retention"][g01["id"]]
        g_ok = contracts.score_general_retention_prediction(
            g01, f"Reasoning step...\nFINAL_ANSWER: {g_exp}", g_exp
        )
        self.assertTrue(g_ok["passed"])
        self.assertFalse(g_ok["contaminated"])

        self.assertTrue(
            contracts.detect_cross_domain_contamination(
                "general_retention", '{"renal_tier": "R4_SEVERE", "dose_mg": 500}'
            )
        )
        self.assertTrue(
            contracts.detect_cross_domain_contamination(
                "med_triage", "def sensor_epoch_deltas(readings):\n    return []"
            )
        )
        self.assertFalse(
            contracts.detect_cross_domain_contamination("general_retention", "FINAL_ANSWER: 42")
        )


if __name__ == "__main__":
    unittest.main()
