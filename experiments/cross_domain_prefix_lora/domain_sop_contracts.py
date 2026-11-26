from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re


MED_KEYS = ("renal_tier", "action", "regimen", "dose_mg", "rule_id")
LEGAL_KEYS = ("risk_grade", "routing", "min_cap_x", "cure_days", "primary_rule")

MED_RENAL_TIERS = ("R1_NORMAL", "R2_MILD", "R3_MOD", "R4_SEVERE")
MED_ACTIONS = ("STD_APPROVE", "ADJUST_APPROVE", "SWITCH_ALT", "HOLD_STAT")
MED_REGIMENS = (
    "CEFEPIME_IV",
    "MEROPENEM_IV",
    "VANCO_IV",
    "APIXABAN_PO",
    "LINEZOLID_PO",
    "PIPTAZO_IV",
    "DAPTO_IV",
    "NONE",
)
MED_RULES = ("MED-R01", "MED-R02", "MED-R03", "MED-R04", "MED-R05")

LEGAL_GRADES = ("LOW", "ELEVATED", "HIGH", "CRITICAL")
LEGAL_ROUTINGS = ("AUTO_APPROVE", "SENIOR_COUNSEL", "VP_LEGAL_CISO", "GC_BOARD_REJECT")
LEGAL_CAPS = (1, 2, 3, 5)
LEGAL_CURES = (0, 15, 30, 45)
LEGAL_RULES = ("LEG-R01", "LEG-R02", "LEG-R03", "LEG-R04", "LEG-R05", "LEG-R06")

MED_DOSE_TABLE = {
    "CEFEPIME_IV": {"R1_NORMAL": 2000, "R2_MILD": 1000, "R3_MOD": 500, "R4_SEVERE": 250},
    "MEROPENEM_IV": {"R1_NORMAL": 1000, "R2_MILD": 1000, "R3_MOD": 500, "R4_SEVERE": 250},
    "VANCO_IV": {"R1_NORMAL": 1500, "R2_MILD": 1000, "R3_MOD": 750, "R4_SEVERE": 500},
    "APIXABAN_PO": {"R1_NORMAL": 10, "R2_MILD": 10, "R3_MOD": 5, "R4_SEVERE": 2},
    "LINEZOLID_PO": {"R1_NORMAL": 600, "R2_MILD": 600, "R3_MOD": 600, "R4_SEVERE": 600},
    "PIPTAZO_IV": {"R1_NORMAL": 4500, "R2_MILD": 3375, "R3_MOD": 2250, "R4_SEVERE": 1125},
    "DAPTO_IV": {"R1_NORMAL": 700, "R2_MILD": 500, "R3_MOD": 350, "R4_SEVERE": 250},
}

LEGAL_CAP_MATRIX = {
    "TIER_STRATEGIC": {"D0_BIO_PII": 5, "D1_CUST_PII": 5, "D2_CONFID": 3, "D3_PUBLIC": 2},
    "TIER_CORE": {"D0_BIO_PII": 5, "D1_CUST_PII": 3, "D2_CONFID": 2, "D3_PUBLIC": 1},
    "TIER_STD": {"D0_BIO_PII": 5, "D1_CUST_PII": 3, "D2_CONFID": 2, "D3_PUBLIC": 1},
}


def canonical_json(obj: dict) -> str:
    return json.dumps(obj, separators=(", ", ": "), ensure_ascii=True)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def med_sop_evaluate(case_input: dict) -> dict:
    raw_egfr = int(case_input["egfr"])
    age = int(case_input["age"])
    albumin = float(case_input["albumin_g_dl"])
    aki_stage = int(case_input["aki_stage"])
    qtc_ms = int(case_input["qtc_ms"])
    potassium = float(case_input["potassium_mmol"])
    platelets = int(case_input["platelets_k"])
    active_bleed = bool(case_input["active_bleed"])
    on_maoi = bool(case_input["on_maoi"])
    requested = str(case_input["requested_regimen"])

    if aki_stage >= 2:
        renal_tier = "R4_SEVERE"
    else:
        eff_egfr = raw_egfr - 12 if (age >= 75 and albumin < 3.2) else raw_egfr
        if eff_egfr >= 60:
            renal_tier = "R1_NORMAL"
        elif eff_egfr >= 30:
            renal_tier = "R2_MILD"
        elif eff_egfr >= 15:
            renal_tier = "R3_MOD"
        else:
            renal_tier = "R4_SEVERE"

    if qtc_ms >= 510:
        return {
            "renal_tier": renal_tier,
            "action": "HOLD_STAT",
            "regimen": "NONE",
            "dose_mg": 0,
            "rule_id": "MED-R01",
        }

    if active_bleed or platelets < 45:
        if requested == "APIXABAN_PO":
            return {
                "renal_tier": renal_tier,
                "action": "HOLD_STAT",
                "regimen": "NONE",
                "dose_mg": 0,
                "rule_id": "MED-R04",
            }

    if on_maoi and requested == "LINEZOLID_PO":
        alt_reg = "DAPTO_IV"
        return {
            "renal_tier": renal_tier,
            "action": "SWITCH_ALT",
            "regimen": alt_reg,
            "dose_mg": MED_DOSE_TABLE[alt_reg][renal_tier],
            "rule_id": "MED-R02",
        }

    if potassium >= 5.7 and requested == "PIPTAZO_IV":
        alt_reg = "MEROPENEM_IV"
        return {
            "renal_tier": renal_tier,
            "action": "SWITCH_ALT",
            "regimen": alt_reg,
            "dose_mg": MED_DOSE_TABLE[alt_reg][renal_tier],
            "rule_id": "MED-R03",
        }

    std_dose = MED_DOSE_TABLE[requested]["R1_NORMAL"]
    tier_dose = MED_DOSE_TABLE[requested][renal_tier]
    action = "STD_APPROVE" if tier_dose == std_dose and renal_tier == "R1_NORMAL" else "ADJUST_APPROVE"
    if tier_dose == std_dose and renal_tier != "R1_NORMAL":
        action = "STD_APPROVE"
    return {
        "renal_tier": renal_tier,
        "action": action,
        "regimen": requested,
        "dose_mg": tier_dose,
        "rule_id": "MED-R05",
    }


def legal_sop_evaluate(case_input: dict) -> dict:
    vendor_tier = str(case_input["vendor_tier"])
    data_class = str(case_input["data_class"])
    cross_border = str(case_input["cross_border"])
    has_sec_assessment = bool(case_input["has_sec_assessment"])
    governing_law = str(case_input["governing_law"])
    ip_indemnity_uncapped = bool(case_input["ip_indemnity_uncapped"])
    auto_renew_months = int(case_input["auto_renew_months"])
    terminate_for_conv = bool(case_input["terminate_for_conv"])
    proposed_cap_x = int(case_input["proposed_cap_x"])

    min_cap_x = LEGAL_CAP_MATRIX[vendor_tier][data_class]

    if cross_border != "NONE" and data_class == "D0_BIO_PII" and not has_sec_assessment:
        return {
            "risk_grade": "CRITICAL",
            "routing": "GC_BOARD_REJECT",
            "min_cap_x": min_cap_x,
            "cure_days": 0,
            "primary_rule": "LEG-R01",
        }

    if governing_law == "FOREIGN_LITIGATION":
        return {
            "risk_grade": "CRITICAL",
            "routing": "GC_BOARD_REJECT",
            "min_cap_x": min_cap_x,
            "cure_days": 15,
            "primary_rule": "LEG-R02",
        }

    if not ip_indemnity_uncapped and data_class in ("D0_BIO_PII", "D1_CUST_PII", "D2_CONFID"):
        return {
            "risk_grade": "HIGH",
            "routing": "VP_LEGAL_CISO",
            "min_cap_x": min_cap_x,
            "cure_days": 15,
            "primary_rule": "LEG-R03",
        }

    if auto_renew_months > 12 and not terminate_for_conv:
        return {
            "risk_grade": "HIGH",
            "routing": "VP_LEGAL_CISO",
            "min_cap_x": min_cap_x,
            "cure_days": 30,
            "primary_rule": "LEG-R04",
        }

    if proposed_cap_x < min_cap_x:
        return {
            "risk_grade": "ELEVATED",
            "routing": "SENIOR_COUNSEL",
            "min_cap_x": min_cap_x,
            "cure_days": 30,
            "primary_rule": "LEG-R05",
        }

    return {
        "risk_grade": "LOW",
        "routing": "AUTO_APPROVE",
        "min_cap_x": min_cap_x,
        "cure_days": 45,
        "primary_rule": "LEG-R06",
    }


MED_SOP_MANUAL_BODY = """================================================================================
INSTITUTIONAL CLINICAL PHARMACY & RENAL TRIAGE PROTOCOL (MED-SOP-2026-V4)
Effective Scope: Inpatient Antimicrobial & Anticoagulant Order Verification
================================================================================

SECTION 1. EFFECTIVE RENAL TIER DETERMINATION (`renal_tier`)
Every inpatient medication order must first be assigned to one of four institutional
renal tiers (`R1_NORMAL`, `R2_MILD`, `R3_MOD`, `R4_SEVERE`) using the exact step order:

  Step 1A (Acute Kidney Injury Override):
    - If `aki_stage >= 2` (KDIGO Stage 2 or Stage 3 acute kidney injury), IMMEDIATELY
      assign `renal_tier = "R4_SEVERE"`, regardless of the raw `egfr` value.

  Step 1B (Geriatric Hypoalbuminemia Adjustment):
    - Otherwise (`aki_stage < 2`), compute `effective_egfr`:
      * If `age >= 75` AND `albumin_g_dl < 3.2`, subtract 12 from raw `egfr`:
        `effective_egfr = egfr - 12`.
      * In all other cases (`age < 75` OR `albumin_g_dl >= 3.2`), use raw `egfr`:
        `effective_egfr = egfr`.

  Step 1C (Tier Threshold Mapping on `effective_egfr`):
    - `effective_egfr >= 60`           -> `"R1_NORMAL"`
    - `30 <= effective_egfr <= 59`     -> `"R2_MILD"`
    - `15 <= effective_egfr <= 29`     -> `"R3_MOD"`
    - `effective_egfr < 15`            -> `"R4_SEVERE"`

--------------------------------------------------------------------------------
SECTION 2. INSTITUTIONAL SINGLE-DOSE MILLIGRAM TABLE (`dose_mg`)
When a regimen is dispensed (either the original `requested_regimen` or a mandated
substitute), select the exact institutional single-administration dose in milligrams
strictly from the matrix below according to the patient's `renal_tier`:

  Regimen Code    | R1_NORMAL | R2_MILD | R3_MOD | R4_SEVERE | Clinical Notes
  ----------------|-----------|---------|--------|-----------|--------------------------
  CEFEPIME_IV     |  2000 mg  | 1000 mg | 500 mg |   250 mg  | Neurotoxicity guard in R3/R4
  MEROPENEM_IV    |  1000 mg  | 1000 mg | 500 mg |   250 mg  | Carbapenem renal step-down
  VANCO_IV        |  1500 mg  | 1000 mg | 750 mg |   500 mg  | Institutional loading cap
  APIXABAN_PO     |    10 mg  |   10 mg |   5 mg |     2 mg  | Inpatient VTE protocol dose
  LINEZOLID_PO    |   600 mg  |  600 mg | 600 mg |   600 mg  | Hepatic clearance; flat 600 mg
  PIPTAZO_IV      |  4500 mg  | 3375 mg | 2250 mg|  1125 mg  | Zosyn composite mg strength
  DAPTO_IV        |   700 mg  |  500 mg | 350 mg |   250 mg  | Mandated MAOI substitute

--------------------------------------------------------------------------------
SECTION 3. HARD SAFETY OVERRIDE HIERARCHY (`rule_id`, `action`, `regimen`)
Evaluate rules strictly from TOP PRIORITY (`MED-R01`) to BOTTOM (`MED-R05`).
The FIRST triggered rule governs the final order decision:

  Priority 1 — Rule `MED-R01` (Severe Torsades / QTc Prolongation Block):
    - Trigger: `qtc_ms >= 510` (applies to ALL orders under acute telemetry lock).
    - Decision:
      * `action = "HOLD_STAT"`
      * `regimen = "NONE"`
      * `dose_mg = 0`
      * `rule_id = "MED-R01"`

  Priority 2 — Rule `MED-R04` (Active Hemorrhage or Severe Thrombocytopenia Anticoagulant Block):
    - Trigger: `requested_regimen == "APIXABAN_PO"` AND (`active_bleed == true` OR `platelets_k < 45`).
    - Decision:
      * `action = "HOLD_STAT"`
      * `regimen = "NONE"`
      * `dose_mg = 0`
      * `rule_id = "MED-R04"`
    - Note: Non-anticoagulant antimicrobial regimens are NOT blocked by `MED-R04`.

  Priority 3 — Rule `MED-R02` (Serotonin Syndrome MAOI Interaction Switch):
    - Trigger: `on_maoi == true` AND `requested_regimen == "LINEZOLID_PO"`.
    - Decision:
      * `action = "SWITCH_ALT"`
      * `regimen = "DAPTO_IV"`
      * `dose_mg` = lookup `DAPTO_IV` under the patient's `renal_tier` in Section 2
        (`700` for `R1_NORMAL`, `500` for `R2_MILD`, `350` for `R3_MOD`, `250` for `R4_SEVERE`).
      * `rule_id = "MED-R02"`

  Priority 4 — Rule `MED-R03` (Severe Hyperkalemia Piperacillin-Tazobactam Switch):
    - Trigger: `potassium_mmol >= 5.7` AND `requested_regimen == "PIPTAZO_IV"`.
    - Decision:
      * `action = "SWITCH_ALT"`
      * `regimen = "MEROPENEM_IV"`
      * `dose_mg` = lookup `MEROPENEM_IV` under the patient's `renal_tier` in Section 2
        (`1000` for `R1_NORMAL`/`R2_MILD`, `500` for `R3_MOD`, `250` for `R4_SEVERE`).
      * `rule_id = "MED-R03"`

  Priority 5 — Rule `MED-R05` (Standard / Renal-Adjusted Formulary Verification):
    - Trigger: None of `MED-R01`..`MED-R04` fired.
    - Decision:
      * `regimen = requested_regimen`
      * `dose_mg` = lookup `requested_regimen` under `renal_tier` in Section 2.
      * `action`:
        - `"STD_APPROVE"` if `dose_mg` equals that drug's `R1_NORMAL` dose in Section 2
          (for example, any `R1_NORMAL` order, `MEROPENEM_IV` or `APIXABAN_PO` in `R2_MILD`,
          or `LINEZOLID_PO` `600 mg` across all renal tiers).
        - `"ADJUST_APPROVE"` whenever `dose_mg` is reduced below that drug's `R1_NORMAL` dose.
      * `rule_id = "MED-R05"`
================================================================================
"""


LEGAL_SOP_MANUAL_BODY = """================================================================================
ENTERPRISE PROCUREMENT & DATA CROSS-BORDER POLICY (LEGAL-SOP-2026-V3)
Effective Scope: Commercial Vendor Agreements, SaaS Subscriptions & Data Addenda
================================================================================

SECTION 1. MINIMUM LIABILITY CAP MULTIPLIER MATRIX (`min_cap_x`)
For every vendor agreement, first determine the mandatory minimum limitation-of-liability
cap (`min_cap_x`, expressed as an integer multiple of annual contract fees) from the
intersection of `vendor_tier` and `data_class`:

  Vendor Tier      | D0_BIO_PII | D1_CUST_PII | D2_CONFID | D3_PUBLIC
  -----------------|------------|-------------|-----------|----------
  TIER_STRATEGIC   |     5x     |      5x     |     3x    |     2x
  TIER_CORE        |     5x     |      3x     |     2x    |     1x
  TIER_STD         |     5x     |      3x     |     2x    |     1x

  Important: `min_cap_x` is ALWAYS reported in the output JSON according to this matrix,
  even when a higher-priority rejection or escalation rule is triggered!

--------------------------------------------------------------------------------
SECTION 2. MANDATORY ESCALATION & REDLINE PRIORITY HIERARCHY
Evaluate contract redlines strictly in order from Priority 1 (`LEG-R01`) down to
Priority 6 (`LEG-R06`). The FIRST triggered rule determines `risk_grade`, `routing`,
`cure_days`, and `primary_rule`:

  Priority 1 — Rule `LEG-R01` (Unassessed Biometric / Core PII Cross-Border Transfer):
    - Trigger: `cross_border != "NONE"` AND `data_class == "D0_BIO_PII"` AND `has_sec_assessment == false`.
    - Decision:
      * `risk_grade = "CRITICAL"`
      * `routing = "GC_BOARD_REJECT"`
      * `cure_days = 0`
      * `primary_rule = "LEG-R01"`

  Priority 2 — Rule `LEG-R02` (Prohibited Exclusive Foreign Court Litigation):
    - Trigger: `governing_law == "FOREIGN_LITIGATION"`
      (Note: `"DOMESTIC_COURT"` and `"NEUTRAL_ARBITRATION"` are permitted and do NOT trigger `LEG-R02`.)
    - Decision:
      * `risk_grade = "CRITICAL"`
      * `routing = "GC_BOARD_REJECT"`
      * `cure_days = 15`
      * `primary_rule = "LEG-R02"`

  Priority 3 — Rule `LEG-R03` (Missing Uncapped Third-Party IP Indemnity Carve-Out):
    - Trigger: `ip_indemnity_uncapped == false` AND `data_class` is in `{"D0_BIO_PII", "D1_CUST_PII", "D2_CONFID"}`.
      (Note: Pure `D3_PUBLIC` agreements are exempt from mandatory uncapped IP indemnity.)
    - Decision:
      * `risk_grade = "HIGH"`
      * `routing = "VP_LEGAL_CISO"`
      * `cure_days = 15`
      * `primary_rule = "LEG-R03"`

  Priority 4 — Rule `LEG-R04` (Multi-Year Evergreen Lock-In Without Convenience Termination):
    - Trigger: `auto_renew_months > 12` AND `terminate_for_conv == false`.
      (Note: `auto_renew_months <= 12` OR `terminate_for_conv == true` passes this check.)
    - Decision:
      * `risk_grade = "HIGH"`
      * `routing = "VP_LEGAL_CISO"`
      * `cure_days = 30`
      * `primary_rule = "LEG-R04"`

  Priority 5 — Rule `LEG-R05` (Proposed Liability Cap Below Mandatory Matrix Floor):
    - Trigger: `proposed_cap_x < min_cap_x` (where `min_cap_x` comes from Section 1).
    - Decision:
      * `risk_grade = "ELEVATED"`
      * `routing = "SENIOR_COUNSEL"`
      * `cure_days = 30`
      * `primary_rule = "LEG-R05"`

  Priority 6 — Rule `LEG-R06` (Fully Compliant Agreement Clear for Automated Execution):
    - Trigger: None of `LEG-R01`..`LEG-R05` fired (`proposed_cap_x >= min_cap_x` and all redlines satisfied).
    - Decision:
      * `risk_grade = "LOW"`
      * `routing = "AUTO_APPROVE"`
      * `cure_days = 45`
      * `primary_rule = "LEG-R06"`
================================================================================
"""


def _format_med_vignette(case_obj: dict) -> str:
    inp = case_obj["input"]
    return (
        f"Patient Order ID: {case_obj['id']}\n"
        f"- Age: {inp['age']} years | Serum Albumin: {inp['albumin_g_dl']:.1f} g/dL\n"
        f"- Baseline eGFR: {inp['egfr']} mL/min/1.73m2 | KDIGO AKI Stage: {inp['aki_stage']}\n"
        f"- ECG QTc: {inp['qtc_ms']} ms | Serum Potassium: {inp['potassium_mmol']:.1f} mmol/L | Platelets: {inp['platelets_k']} K/uL\n"
        f"- Active Bleeding: {str(inp['active_bleed']).lower()} | Concurrent MAOI Therapy: {str(inp['on_maoi']).lower()}\n"
        f"- Requested Inpatient Regimen: {inp['requested_regimen']}"
    )


def _format_legal_vignette(case_obj: dict) -> str:
    inp = case_obj["input"]
    return (
        f"Contract Review ID: {case_obj['id']}\n"
        f"- Vendor Classification: {inp['vendor_tier']} | Data Sensitivity Tier: {inp['data_class']}\n"
        f"- Cross-Border Transfer Destination: {inp['cross_border']} | Regulatory Security Assessment Passed: {str(inp['has_sec_assessment']).lower()}\n"
        f"- Dispute Governing Venue: {inp['governing_law']} | Uncapped 3rd-Party IP Indemnity: {str(inp['ip_indemnity_uncapped']).lower()}\n"
        f"- Auto-Renewal Term: {inp['auto_renew_months']} months | Termination for Convenience Right: {str(inp['terminate_for_conv']).lower()}\n"
        f"- Vendor Proposed Liability Cap: {inp['proposed_cap_x']}x annual fees"
    )


MED_TASK_SUFFIX_SPEC = (
    "Output ONLY a single-line JSON object with exact keys "
    '("renal_tier", "action", "regimen", "dose_mg", "rule_id") and no extra prose.\n'
    "Allowed enum values:\n"
    "- renal_tier: R1_NORMAL | R2_MILD | R3_MOD | R4_SEVERE\n"
    "- action: STD_APPROVE | ADJUST_APPROVE | SWITCH_ALT | HOLD_STAT\n"
    "- regimen: CEFEPIME_IV | MEROPENEM_IV | VANCO_IV | APIXABAN_PO | LINEZOLID_PO | PIPTAZO_IV | DAPTO_IV | NONE\n"
    "- dose_mg: integer milligram single dose (0 when held)\n"
    "- rule_id: MED-R01 | MED-R02 | MED-R03 | MED-R04 | MED-R05"
)

LEGAL_TASK_SUFFIX_SPEC = (
    "Output ONLY a single-line JSON object with exact keys "
    '("risk_grade", "routing", "min_cap_x", "cure_days", "primary_rule") and no extra prose.\n'
    "Allowed enum values:\n"
    "- risk_grade: LOW | ELEVATED | HIGH | CRITICAL\n"
    "- routing: AUTO_APPROVE | SENIOR_COUNSEL | VP_LEGAL_CISO | GC_BOARD_REJECT\n"
    "- min_cap_x: integer multiplier from {1, 2, 3, 5}\n"
    "- cure_days: integer days from {0, 15, 30, 45}\n"
    "- primary_rule: LEG-R01 | LEG-R02 | LEG-R03 | LEG-R04 | LEG-R05 | LEG-R06"
)


def _build_med_cases() -> tuple[list[dict], list[dict], list[dict], dict[str, dict]]:
    seeds_raw = [
        ("M_S01", {"age": 62, "albumin_g_dl": 3.8, "egfr": 74, "aki_stage": 0, "qtc_ms": 524, "potassium_mmol": 4.3, "platelets_k": 210, "active_bleed": False, "on_maoi": False, "requested_regimen": "VANCO_IV"}),
        ("M_S02", {"age": 79, "albumin_g_dl": 2.9, "egfr": 38, "aki_stage": 0, "qtc_ms": 442, "potassium_mmol": 4.5, "platelets_k": 185, "active_bleed": False, "on_maoi": True, "requested_regimen": "LINEZOLID_PO"}),
        ("M_S03", {"age": 54, "albumin_g_dl": 3.6, "egfr": 48, "aki_stage": 0, "qtc_ms": 455, "potassium_mmol": 5.9, "platelets_k": 240, "active_bleed": False, "on_maoi": False, "requested_regimen": "PIPTAZO_IV"}),
        ("M_S04", {"age": 68, "albumin_g_dl": 3.5, "egfr": 52, "aki_stage": 0, "qtc_ms": 438, "potassium_mmol": 4.1, "platelets_k": 34, "active_bleed": False, "on_maoi": False, "requested_regimen": "APIXABAN_PO"}),
        ("M_S05", {"age": 49, "albumin_g_dl": 4.0, "egfr": 68, "aki_stage": 2, "qtc_ms": 460, "potassium_mmol": 4.8, "platelets_k": 190, "active_bleed": False, "on_maoi": False, "requested_regimen": "CEFEPIME_IV"}),
        ("M_S06", {"age": 81, "albumin_g_dl": 2.8, "egfr": 66, "aki_stage": 0, "qtc_ms": 448, "potassium_mmol": 4.2, "platelets_k": 205, "active_bleed": False, "on_maoi": False, "requested_regimen": "VANCO_IV"}),
        ("M_S07", {"age": 58, "albumin_g_dl": 3.7, "egfr": 22, "aki_stage": 0, "qtc_ms": 450, "potassium_mmol": 4.4, "platelets_k": 175, "active_bleed": False, "on_maoi": False, "requested_regimen": "PIPTAZO_IV"}),
        ("M_S08", {"age": 45, "albumin_g_dl": 4.1, "egfr": 82, "aki_stage": 0, "qtc_ms": 430, "potassium_mmol": 4.0, "platelets_k": 260, "active_bleed": False, "on_maoi": False, "requested_regimen": "CEFEPIME_IV"}),
    ]

    unlabeled_raw = [
        ("M_U01", {"age": 77, "albumin_g_dl": 3.0, "egfr": 65, "aki_stage": 0, "qtc_ms": 440, "potassium_mmol": 4.2, "platelets_k": 195, "active_bleed": False, "on_maoi": False, "requested_regimen": "CEFEPIME_IV"}),
        ("M_U02", {"age": 63, "albumin_g_dl": 3.9, "egfr": 55, "aki_stage": 2, "qtc_ms": 452, "potassium_mmol": 4.6, "platelets_k": 160, "active_bleed": False, "on_maoi": False, "requested_regimen": "VANCO_IV"}),
        ("M_U03", {"age": 51, "albumin_g_dl": 3.5, "egfr": 76, "aki_stage": 0, "qtc_ms": 518, "potassium_mmol": 4.1, "platelets_k": 220, "active_bleed": False, "on_maoi": False, "requested_regimen": "MEROPENEM_IV"}),
        ("M_U04", {"age": 72, "albumin_g_dl": 3.4, "egfr": 26, "aki_stage": 0, "qtc_ms": 446, "potassium_mmol": 4.3, "platelets_k": 38, "active_bleed": False, "on_maoi": False, "requested_regimen": "CEFEPIME_IV"}),
        ("M_U05", {"age": 69, "albumin_g_dl": 3.3, "egfr": 44, "aki_stage": 0, "qtc_ms": 439, "potassium_mmol": 4.2, "platelets_k": 180, "active_bleed": True, "on_maoi": False, "requested_regimen": "APIXABAN_PO"}),
        ("M_U06", {"age": 83, "albumin_g_dl": 2.7, "egfr": 24, "aki_stage": 0, "qtc_ms": 461, "potassium_mmol": 4.5, "platelets_k": 210, "active_bleed": False, "on_maoi": True, "requested_regimen": "LINEZOLID_PO"}),
        ("M_U07", {"age": 59, "albumin_g_dl": 3.8, "egfr": 21, "aki_stage": 0, "qtc_ms": 449, "potassium_mmol": 5.8, "platelets_k": 200, "active_bleed": False, "on_maoi": False, "requested_regimen": "PIPTAZO_IV"}),
        ("M_U08", {"age": 47, "albumin_g_dl": 4.2, "egfr": 42, "aki_stage": 0, "qtc_ms": 433, "potassium_mmol": 4.1, "platelets_k": 230, "active_bleed": False, "on_maoi": False, "requested_regimen": "MEROPENEM_IV"}),
        ("M_U09", {"age": 76, "albumin_g_dl": 3.5, "egfr": 64, "aki_stage": 0, "qtc_ms": 441, "potassium_mmol": 4.0, "platelets_k": 215, "active_bleed": False, "on_maoi": False, "requested_regimen": "VANCO_IV"}),
        ("M_U10", {"age": 64, "albumin_g_dl": 2.9, "egfr": 64, "aki_stage": 0, "qtc_ms": 437, "potassium_mmol": 4.3, "platelets_k": 190, "active_bleed": False, "on_maoi": False, "requested_regimen": "PIPTAZO_IV"}),
        ("M_U11", {"age": 55, "albumin_g_dl": 3.6, "egfr": 13, "aki_stage": 0, "qtc_ms": 458, "potassium_mmol": 4.7, "platelets_k": 170, "active_bleed": False, "on_maoi": False, "requested_regimen": "APIXABAN_PO"}),
        ("M_U12", {"age": 60, "albumin_g_dl": 3.7, "egfr": 19, "aki_stage": 0, "qtc_ms": 445, "potassium_mmol": 4.4, "platelets_k": 205, "active_bleed": False, "on_maoi": False, "requested_regimen": "LINEZOLID_PO"}),
        ("M_U13", {"age": 80, "albumin_g_dl": 2.9, "egfr": 40, "aki_stage": 0, "qtc_ms": 451, "potassium_mmol": 4.2, "platelets_k": 188, "active_bleed": False, "on_maoi": False, "requested_regimen": "APIXABAN_PO"}),
        ("M_U14", {"age": 52, "albumin_g_dl": 3.9, "egfr": 71, "aki_stage": 0, "qtc_ms": 436, "potassium_mmol": 6.0, "platelets_k": 245, "active_bleed": False, "on_maoi": False, "requested_regimen": "VANCO_IV"}),
        ("M_U15", {"age": 67, "albumin_g_dl": 3.4, "egfr": 33, "aki_stage": 0, "qtc_ms": 444, "potassium_mmol": 4.3, "platelets_k": 198, "active_bleed": False, "on_maoi": False, "requested_regimen": "DAPTO_IV"}),
        ("M_U16", {"age": 78, "albumin_g_dl": 3.1, "egfr": 70, "aki_stage": 3, "qtc_ms": 463, "potassium_mmol": 4.9, "platelets_k": 155, "active_bleed": False, "on_maoi": False, "requested_regimen": "PIPTAZO_IV"}),
        ("M_U17", {"age": 43, "albumin_g_dl": 4.0, "egfr": 88, "aki_stage": 0, "qtc_ms": 428, "potassium_mmol": 3.9, "platelets_k": 275, "active_bleed": False, "on_maoi": True, "requested_regimen": "LINEZOLID_PO"}),
        ("M_U18", {"age": 71, "albumin_g_dl": 3.6, "egfr": 18, "aki_stage": 0, "qtc_ms": 515, "potassium_mmol": 5.9, "platelets_k": 32, "active_bleed": True, "on_maoi": True, "requested_regimen": "APIXABAN_PO"}),
        ("M_U19", {"age": 57, "albumin_g_dl": 3.8, "egfr": 36, "aki_stage": 0, "qtc_ms": 447, "potassium_mmol": 4.2, "platelets_k": 212, "active_bleed": False, "on_maoi": False, "requested_regimen": "PIPTAZO_IV"}),
        ("M_U20", {"age": 66, "albumin_g_dl": 3.5, "egfr": 27, "aki_stage": 0, "qtc_ms": 453, "potassium_mmol": 4.5, "platelets_k": 184, "active_bleed": False, "on_maoi": False, "requested_regimen": "MEROPENEM_IV"}),
    ]

    blind_raw = [
        ("M_T01", {"age": 76, "albumin_g_dl": 2.9, "egfr": 68, "aki_stage": 0, "qtc_ms": 442, "potassium_mmol": 4.1, "platelets_k": 204, "active_bleed": False, "on_maoi": False, "requested_regimen": "CEFEPIME_IV"}),
        ("M_T02", {"age": 78, "albumin_g_dl": 3.4, "egfr": 63, "aki_stage": 0, "qtc_ms": 438, "potassium_mmol": 4.2, "platelets_k": 218, "active_bleed": False, "on_maoi": False, "requested_regimen": "CEFEPIME_IV"}),
        ("M_T03", {"age": 82, "albumin_g_dl": 2.8, "egfr": 39, "aki_stage": 0, "qtc_ms": 451, "potassium_mmol": 4.4, "platelets_k": 192, "active_bleed": False, "on_maoi": False, "requested_regimen": "MEROPENEM_IV"}),
        ("M_T04", {"age": 84, "albumin_g_dl": 3.0, "egfr": 25, "aki_stage": 0, "qtc_ms": 449, "potassium_mmol": 4.6, "platelets_k": 176, "active_bleed": False, "on_maoi": False, "requested_regimen": "VANCO_IV"}),
        ("M_T05", {"age": 53, "albumin_g_dl": 3.9, "egfr": 79, "aki_stage": 2, "qtc_ms": 456, "potassium_mmol": 4.5, "platelets_k": 188, "active_bleed": False, "on_maoi": False, "requested_regimen": "PIPTAZO_IV"}),
        ("M_T06", {"age": 61, "albumin_g_dl": 3.7, "egfr": 46, "aki_stage": 1, "qtc_ms": 440, "potassium_mmol": 4.3, "platelets_k": 225, "active_bleed": False, "on_maoi": False, "requested_regimen": "PIPTAZO_IV"}),
        ("M_T07", {"age": 59, "albumin_g_dl": 3.6, "egfr": 84, "aki_stage": 0, "qtc_ms": 516, "potassium_mmol": 5.9, "platelets_k": 210, "active_bleed": False, "on_maoi": True, "requested_regimen": "LINEZOLID_PO"}),
        ("M_T08", {"age": 65, "albumin_g_dl": 3.5, "egfr": 35, "aki_stage": 0, "qtc_ms": 445, "potassium_mmol": 4.1, "platelets_k": 39, "active_bleed": False, "on_maoi": False, "requested_regimen": "APIXABAN_PO"}),
        ("M_T09", {"age": 67, "albumin_g_dl": 3.5, "egfr": 35, "aki_stage": 0, "qtc_ms": 445, "potassium_mmol": 4.1, "platelets_k": 36, "active_bleed": True, "on_maoi": False, "requested_regimen": "VANCO_IV"}),
        ("M_T10", {"age": 79, "albumin_g_dl": 2.9, "egfr": 36, "aki_stage": 0, "qtc_ms": 447, "potassium_mmol": 4.3, "platelets_k": 195, "active_bleed": False, "on_maoi": True, "requested_regimen": "LINEZOLID_PO"}),
        ("M_T11", {"age": 48, "albumin_g_dl": 4.1, "egfr": 72, "aki_stage": 0, "qtc_ms": 432, "potassium_mmol": 4.0, "platelets_k": 250, "active_bleed": False, "on_maoi": True, "requested_regimen": "VANCO_IV"}),
        ("M_T12", {"age": 56, "albumin_g_dl": 3.8, "egfr": 24, "aki_stage": 0, "qtc_ms": 454, "potassium_mmol": 5.9, "platelets_k": 190, "active_bleed": False, "on_maoi": False, "requested_regimen": "PIPTAZO_IV"}),
        ("M_T13", {"age": 50, "albumin_g_dl": 3.9, "egfr": 65, "aki_stage": 2, "qtc_ms": 459, "potassium_mmol": 5.8, "platelets_k": 172, "active_bleed": False, "on_maoi": False, "requested_regimen": "PIPTAZO_IV"}),
        ("M_T14", {"age": 64, "albumin_g_dl": 3.7, "egfr": 44, "aki_stage": 0, "qtc_ms": 439, "potassium_mmol": 4.2, "platelets_k": 215, "active_bleed": False, "on_maoi": False, "requested_regimen": "APIXABAN_PO"}),
        ("M_T15", {"age": 70, "albumin_g_dl": 3.6, "egfr": 21, "aki_stage": 0, "qtc_ms": 448, "potassium_mmol": 4.4, "platelets_k": 182, "active_bleed": False, "on_maoi": False, "requested_regimen": "LINEZOLID_PO"}),
        ("M_T16", {"age": 73, "albumin_g_dl": 3.5, "egfr": 12, "aki_stage": 0, "qtc_ms": 452, "potassium_mmol": 4.7, "platelets_k": 168, "active_bleed": False, "on_maoi": False, "requested_regimen": "APIXABAN_PO"}),
    ]

    seeds = []
    for cid, inp in seeds_raw:
        ans = med_sop_evaluate(inp)
        obj = {"id": cid, "input": inp, "expected": ans}
        obj["vignette"] = _format_med_vignette(obj)
        seeds.append(obj)

    unlabeled = []
    for cid, inp in unlabeled_raw:
        obj = {"id": cid, "input": inp}
        obj["vignette"] = _format_med_vignette(obj)
        unlabeled.append(obj)

    blind_inputs = []
    blind_oracles = {}
    for cid, inp in blind_raw:
        ans = med_sop_evaluate(inp)
        obj = {"id": cid, "input": inp}
        obj["vignette"] = _format_med_vignette(obj)
        blind_inputs.append(obj)
        blind_oracles[cid] = ans

    seen_inputs = set()
    for grp in (seeds_raw, unlabeled_raw, blind_raw):
        for cid, inp in grp:
            sig = json.dumps(inp, sort_keys=True)
            if sig in seen_inputs:
                raise ValueError(f"Duplicate medical input signature on {cid}")
            seen_inputs.add(sig)

    return seeds, unlabeled, blind_inputs, blind_oracles


def _build_legal_cases() -> tuple[list[dict], list[dict], list[dict], dict[str, dict]]:
    seeds_raw = [
        ("L_S01", {"vendor_tier": "TIER_STRATEGIC", "data_class": "D0_BIO_PII", "cross_border": "EU_FRANKFURT", "has_sec_assessment": False, "governing_law": "DOMESTIC_COURT", "ip_indemnity_uncapped": True, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 5}),
        ("L_S02", {"vendor_tier": "TIER_CORE", "data_class": "D1_CUST_PII", "cross_border": "NONE", "has_sec_assessment": True, "governing_law": "FOREIGN_LITIGATION", "ip_indemnity_uncapped": True, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 3}),
        ("L_S03", {"vendor_tier": "TIER_STRATEGIC", "data_class": "D2_CONFID", "cross_border": "NONE", "has_sec_assessment": True, "governing_law": "NEUTRAL_ARBITRATION", "ip_indemnity_uncapped": False, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 3}),
        ("L_S04", {"vendor_tier": "TIER_STD", "data_class": "D2_CONFID", "cross_border": "NONE", "has_sec_assessment": True, "governing_law": "DOMESTIC_COURT", "ip_indemnity_uncapped": True, "auto_renew_months": 24, "terminate_for_conv": False, "proposed_cap_x": 2}),
        ("L_S05", {"vendor_tier": "TIER_STRATEGIC", "data_class": "D1_CUST_PII", "cross_border": "SG_SINGAPORE", "has_sec_assessment": True, "governing_law": "NEUTRAL_ARBITRATION", "ip_indemnity_uncapped": True, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 3}),
        ("L_S06", {"vendor_tier": "TIER_CORE", "data_class": "D3_PUBLIC", "cross_border": "NONE", "has_sec_assessment": False, "governing_law": "DOMESTIC_COURT", "ip_indemnity_uncapped": False, "auto_renew_months": 12, "terminate_for_conv": False, "proposed_cap_x": 1}),
        ("L_S07", {"vendor_tier": "TIER_STD", "data_class": "D1_CUST_PII", "cross_border": "NONE", "has_sec_assessment": True, "governing_law": "DOMESTIC_COURT", "ip_indemnity_uncapped": True, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 2}),
        ("L_S08", {"vendor_tier": "TIER_STRATEGIC", "data_class": "D0_BIO_PII", "cross_border": "SG_SINGAPORE", "has_sec_assessment": True, "governing_law": "NEUTRAL_ARBITRATION", "ip_indemnity_uncapped": True, "auto_renew_months": 24, "terminate_for_conv": True, "proposed_cap_x": 5}),
    ]

    unlabeled_raw = [
        ("L_U01", {"vendor_tier": "TIER_CORE", "data_class": "D0_BIO_PII", "cross_border": "US_VIRGINIA", "has_sec_assessment": False, "governing_law": "NEUTRAL_ARBITRATION", "ip_indemnity_uncapped": True, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 5}),
        ("L_U02", {"vendor_tier": "TIER_STD", "data_class": "D0_BIO_PII", "cross_border": "NONE", "has_sec_assessment": False, "governing_law": "DOMESTIC_COURT", "ip_indemnity_uncapped": True, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 5}),
        ("L_U03", {"vendor_tier": "TIER_STRATEGIC", "data_class": "D3_PUBLIC", "cross_border": "NONE", "has_sec_assessment": False, "governing_law": "FOREIGN_LITIGATION", "ip_indemnity_uncapped": False, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 2}),
        ("L_U04", {"vendor_tier": "TIER_CORE", "data_class": "D1_CUST_PII", "cross_border": "JP_TOKYO", "has_sec_assessment": False, "governing_law": "NEUTRAL_ARBITRATION", "ip_indemnity_uncapped": False, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 3}),
        ("L_U05", {"vendor_tier": "TIER_STD", "data_class": "D3_PUBLIC", "cross_border": "EU_DUBLIN", "has_sec_assessment": False, "governing_law": "NEUTRAL_ARBITRATION", "ip_indemnity_uncapped": False, "auto_renew_months": 36, "terminate_for_conv": False, "proposed_cap_x": 1}),
        ("L_U06", {"vendor_tier": "TIER_STRATEGIC", "data_class": "D2_CONFID", "cross_border": "NONE", "has_sec_assessment": True, "governing_law": "DOMESTIC_COURT", "ip_indemnity_uncapped": True, "auto_renew_months": 24, "terminate_for_conv": True, "proposed_cap_x": 2}),
        ("L_U07", {"vendor_tier": "TIER_CORE", "data_class": "D2_CONFID", "cross_border": "NONE", "has_sec_assessment": True, "governing_law": "DOMESTIC_COURT", "ip_indemnity_uncapped": True, "auto_renew_months": 12, "terminate_for_conv": False, "proposed_cap_x": 2}),
        ("L_U08", {"vendor_tier": "TIER_STD", "data_class": "D2_CONFID", "cross_border": "NONE", "has_sec_assessment": True, "governing_law": "DOMESTIC_COURT", "ip_indemnity_uncapped": True, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 1}),
        ("L_U09", {"vendor_tier": "TIER_STRATEGIC", "data_class": "D3_PUBLIC", "cross_border": "NONE", "has_sec_assessment": False, "governing_law": "DOMESTIC_COURT", "ip_indemnity_uncapped": False, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 1}),
        ("L_U10", {"vendor_tier": "TIER_STRATEGIC", "data_class": "D3_PUBLIC", "cross_border": "NONE", "has_sec_assessment": False, "governing_law": "DOMESTIC_COURT", "ip_indemnity_uncapped": False, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 2}),
        ("L_U11", {"vendor_tier": "TIER_CORE", "data_class": "D0_BIO_PII", "cross_border": "SG_SINGAPORE", "has_sec_assessment": True, "governing_law": "NEUTRAL_ARBITRATION", "ip_indemnity_uncapped": True, "auto_renew_months": 18, "terminate_for_conv": False, "proposed_cap_x": 5}),
        ("L_U12", {"vendor_tier": "TIER_STD", "data_class": "D1_CUST_PII", "cross_border": "EU_FRANKFURT", "has_sec_assessment": False, "governing_law": "FOREIGN_LITIGATION", "ip_indemnity_uncapped": False, "auto_renew_months": 24, "terminate_for_conv": False, "proposed_cap_x": 1}),
        ("L_U13", {"vendor_tier": "TIER_CORE", "data_class": "D1_CUST_PII", "cross_border": "NONE", "has_sec_assessment": True, "governing_law": "DOMESTIC_COURT", "ip_indemnity_uncapped": True, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 4}),
        ("L_U14", {"vendor_tier": "TIER_STRATEGIC", "data_class": "D1_CUST_PII", "cross_border": "NONE", "has_sec_assessment": True, "governing_law": "DOMESTIC_COURT", "ip_indemnity_uncapped": False, "auto_renew_months": 24, "terminate_for_conv": False, "proposed_cap_x": 2}),
        ("L_U15", {"vendor_tier": "TIER_STD", "data_class": "D0_BIO_PII", "cross_border": "NONE", "has_sec_assessment": False, "governing_law": "DOMESTIC_COURT", "ip_indemnity_uncapped": True, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 3}),
        ("L_U16", {"vendor_tier": "TIER_CORE", "data_class": "D2_CONFID", "cross_border": "SG_SINGAPORE", "has_sec_assessment": False, "governing_law": "NEUTRAL_ARBITRATION", "ip_indemnity_uncapped": False, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 2}),
        ("L_U17", {"vendor_tier": "TIER_STRATEGIC", "data_class": "D0_BIO_PII", "cross_border": "JP_TOKYO", "has_sec_assessment": False, "governing_law": "FOREIGN_LITIGATION", "ip_indemnity_uncapped": False, "auto_renew_months": 36, "terminate_for_conv": False, "proposed_cap_x": 1}),
        ("L_U18", {"vendor_tier": "TIER_STD", "data_class": "D3_PUBLIC", "cross_border": "NONE", "has_sec_assessment": True, "governing_law": "NEUTRAL_ARBITRATION", "ip_indemnity_uncapped": False, "auto_renew_months": 24, "terminate_for_conv": True, "proposed_cap_x": 1}),
        ("L_U19", {"vendor_tier": "TIER_CORE", "data_class": "D1_CUST_PII", "cross_border": "SG_SINGAPORE", "has_sec_assessment": False, "governing_law": "DOMESTIC_COURT", "ip_indemnity_uncapped": True, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 3}),
        ("L_U20", {"vendor_tier": "TIER_STRATEGIC", "data_class": "D2_CONFID", "cross_border": "EU_FRANKFURT", "has_sec_assessment": True, "governing_law": "NEUTRAL_ARBITRATION", "ip_indemnity_uncapped": True, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 3}),
    ]

    blind_raw = [
        ("L_T01", {"vendor_tier": "TIER_STD", "data_class": "D0_BIO_PII", "cross_border": "SG_SINGAPORE", "has_sec_assessment": False, "governing_law": "NEUTRAL_ARBITRATION", "ip_indemnity_uncapped": True, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 5}),
        ("L_T02", {"vendor_tier": "TIER_CORE", "data_class": "D0_BIO_PII", "cross_border": "NONE", "has_sec_assessment": False, "governing_law": "DOMESTIC_COURT", "ip_indemnity_uncapped": True, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 5}),
        ("L_T03", {"vendor_tier": "TIER_STRATEGIC", "data_class": "D1_CUST_PII", "cross_border": "EU_FRANKFURT", "has_sec_assessment": False, "governing_law": "DOMESTIC_COURT", "ip_indemnity_uncapped": True, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 5}),
        ("L_T04", {"vendor_tier": "TIER_STRATEGIC", "data_class": "D2_CONFID", "cross_border": "NONE", "has_sec_assessment": True, "governing_law": "FOREIGN_LITIGATION", "ip_indemnity_uncapped": False, "auto_renew_months": 24, "terminate_for_conv": False, "proposed_cap_x": 1}),
        ("L_T05", {"vendor_tier": "TIER_CORE", "data_class": "D1_CUST_PII", "cross_border": "NONE", "has_sec_assessment": True, "governing_law": "NEUTRAL_ARBITRATION", "ip_indemnity_uncapped": False, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 5}),
        ("L_T06", {"vendor_tier": "TIER_STRATEGIC", "data_class": "D3_PUBLIC", "cross_border": "EU_DUBLIN", "has_sec_assessment": False, "governing_law": "NEUTRAL_ARBITRATION", "ip_indemnity_uncapped": False, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 2}),
        ("L_T07", {"vendor_tier": "TIER_CORE", "data_class": "D2_CONFID", "cross_border": "NONE", "has_sec_assessment": True, "governing_law": "DOMESTIC_COURT", "ip_indemnity_uncapped": True, "auto_renew_months": 36, "terminate_for_conv": False, "proposed_cap_x": 3}),
        ("L_T08", {"vendor_tier": "TIER_CORE", "data_class": "D2_CONFID", "cross_border": "NONE", "has_sec_assessment": True, "governing_law": "DOMESTIC_COURT", "ip_indemnity_uncapped": True, "auto_renew_months": 36, "terminate_for_conv": True, "proposed_cap_x": 2}),
        ("L_T09", {"vendor_tier": "TIER_STRATEGIC", "data_class": "D1_CUST_PII", "cross_border": "NONE", "has_sec_assessment": True, "governing_law": "DOMESTIC_COURT", "ip_indemnity_uncapped": True, "auto_renew_months": 12, "terminate_for_conv": False, "proposed_cap_x": 4}),
        ("L_T10", {"vendor_tier": "TIER_CORE", "data_class": "D1_CUST_PII", "cross_border": "NONE", "has_sec_assessment": True, "governing_law": "DOMESTIC_COURT", "ip_indemnity_uncapped": True, "auto_renew_months": 12, "terminate_for_conv": False, "proposed_cap_x": 3}),
        ("L_T11", {"vendor_tier": "TIER_STRATEGIC", "data_class": "D2_CONFID", "cross_border": "NONE", "has_sec_assessment": True, "governing_law": "NEUTRAL_ARBITRATION", "ip_indemnity_uncapped": True, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 2}),
        ("L_T12", {"vendor_tier": "TIER_STD", "data_class": "D0_BIO_PII", "cross_border": "JP_TOKYO", "has_sec_assessment": True, "governing_law": "NEUTRAL_ARBITRATION", "ip_indemnity_uncapped": True, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 4}),
        ("L_T13", {"vendor_tier": "TIER_STRATEGIC", "data_class": "D3_PUBLIC", "cross_border": "NONE", "has_sec_assessment": False, "governing_law": "DOMESTIC_COURT", "ip_indemnity_uncapped": False, "auto_renew_months": 12, "terminate_for_conv": False, "proposed_cap_x": 1}),
        ("L_T14", {"vendor_tier": "TIER_STD", "data_class": "D3_PUBLIC", "cross_border": "SG_SINGAPORE", "has_sec_assessment": False, "governing_law": "DOMESTIC_COURT", "ip_indemnity_uncapped": False, "auto_renew_months": 12, "terminate_for_conv": False, "proposed_cap_x": 1}),
        ("L_T15", {"vendor_tier": "TIER_STRATEGIC", "data_class": "D0_BIO_PII", "cross_border": "EU_FRANKFURT", "has_sec_assessment": False, "governing_law": "FOREIGN_LITIGATION", "ip_indemnity_uncapped": False, "auto_renew_months": 24, "terminate_for_conv": False, "proposed_cap_x": 1}),
        ("L_T16", {"vendor_tier": "TIER_CORE", "data_class": "D0_BIO_PII", "cross_border": "EU_FRANKFURT", "has_sec_assessment": True, "governing_law": "NEUTRAL_ARBITRATION", "ip_indemnity_uncapped": True, "auto_renew_months": 12, "terminate_for_conv": True, "proposed_cap_x": 5}),
    ]

    seeds = []
    for cid, inp in seeds_raw:
        ans = legal_sop_evaluate(inp)
        obj = {"id": cid, "input": inp, "expected": ans}
        obj["vignette"] = _format_legal_vignette(obj)
        seeds.append(obj)

    unlabeled = []
    for cid, inp in unlabeled_raw:
        obj = {"id": cid, "input": inp}
        obj["vignette"] = _format_legal_vignette(obj)
        unlabeled.append(obj)

    blind_inputs = []
    blind_oracles = {}
    for cid, inp in blind_raw:
        ans = legal_sop_evaluate(inp)
        obj = {"id": cid, "input": inp}
        obj["vignette"] = _format_legal_vignette(obj)
        blind_inputs.append(obj)
        blind_oracles[cid] = ans

    seen_inputs = set()
    for grp in (seeds_raw, unlabeled_raw, blind_raw):
        for cid, inp in grp:
            sig = json.dumps(inp, sort_keys=True)
            if sig in seen_inputs:
                raise ValueError(f"Duplicate legal input signature on {cid}")
            seen_inputs.add(sig)

    return seeds, unlabeled, blind_inputs, blind_oracles


def _build_general_retention_and_background() -> tuple[list[dict], dict[str, str], list[dict]]:
    general_raw = [
        ("G_R01", "math", "A warehouse has 14 pallets with 18 cartons each. After shipping out 67 cartons and receiving 25 new cartons, how many cartons remain in the warehouse? End your response with `FINAL_ANSWER: <integer>`.", "210"),
        ("G_R02", "math", "A train travels 135 km in 1.5 hours and then 180 km in 2.5 hours. What is its average speed in km/h over the entire 4-hour trip? Express as a decimal with two digits if needed, ending with `FINAL_ANSWER: <number>`.", "78.75"),
        ("G_R03", "math", "Three servers process 45, 60, and 75 requests per minute respectively. Working together at constant rates, how many minutes does it take them to process 2,700 requests? End your response with `FINAL_ANSWER: <integer>`.", "15"),
        ("G_R04", "math", "A jacket priced at 320 yuan is discounted by 25%, and then an additional 15 yuan coupon is applied. What is the final price in yuan? End your response with `FINAL_ANSWER: <integer>`.", "225"),
        ("G_R05", "logic", "Four tasks (Alpha, Beta, Gamma, Delta) run sequentially without overlap. Beta runs immediately before Gamma. Alpha runs after Gamma. Delta runs first. Which task runs third? End your response with `FINAL_ANSWER: <TaskName>`.", "Gamma"),
        ("G_R06", "logic", "Every encrypted backup is compressed. Archive-7 is not compressed. Can Archive-7 be an encrypted backup? Answer strictly YES or NO at the end as `FINAL_ANSWER: <YES|NO>`.", "NO"),
        ("G_R07", "logic", "Among Mia, Noah, and Liam, each specializes in exactly one database: Redis, Postgres, or ClickHouse. Mia does not use Redis or ClickHouse. Liam does not use Redis. Which database does Noah use? End with `FINAL_ANSWER: <Database>`.", "Redis"),
        ("G_R08", "logic", "Aconference room booking rule states: meetings over 8 people require Room-B, unless the projector is needed, which always requires Room-C. A 10-person design review needs the projector. Which room is required? End with `FINAL_ANSWER: <Room>`.", "Room-C"),
        ("G_R09", "extract", "Extract the exact invoice code from the following memo: 'Per Q3 reconciliation, disregard draft INV-2025-8810 and settle superseded voucher under final invoice INV-2026-4092 by Friday.' End with `FINAL_ANSWER: <Code>`.", "INV-2026-4092"),
        ("G_R10", "extract", "Given the quarterly latency table: [Jan: 42ms, Feb: 39ms, Mar: 51ms, Apr: 36ms, May: 44ms], which 3-letter month had the lowest latency? End with `FINAL_ANSWER: <Month>`.", "Apr"),
        ("G_R11", "extract", "Count how many items in the list `[17, 24, 31, 42, 55, 68, 73, 80]` are strictly greater than 30 AND divisible by 2. End with `FINAL_ANSWER: <integer>`.", "3"),
        ("G_R12", "extract", "From the release log: 'v1.4.2 (rolled back), v1.4.3 (canary only), v1.4.5 (production active), v1.5.0-rc1 (staging)', which version string is currently active in production? End with `FINAL_ANSWER: <version>`.", "v1.4.5"),
        ("G_R13", "cn_qa", "某图书馆周一借出图书 86 本，周二比周一多借出 19 本，周三借出数量是周二的一半（向下取整已无需，周二为偶数吗？86+19=105，改为：周一借出 85 本，周二比周一多借出 19 本，周三借出数量恰好是周二的一半）。请问这三天一共借出多少本图书？最后一行请严格按 `FINAL_ANSWER: <整数>` 输出。", "241"),
        ("G_R14", "cn_qa", "甲、乙、丙三人排成一列，已知甲不在第一位，乙在丙的前面，且丙不在最后一位。请问排在第一位的是谁？（仅填“甲”、“乙”或“丙”），最后一行严格输出 `FINAL_ANSWER: <姓名>`。", "乙"),
        ("G_R15", "cn_qa", "请从以下通知中提取正式切换的北京时间（仅输出 HH:MM 格式）：‘原定于 01:30 的网关割接因演练顺延，02:15 完成预检，并于 03:40 正式切换流量，04:10 结束观察。’最后一行严格输出 `FINAL_ANSWER: <HH:MM>`。", "03:40"),
        ("G_R16", "cn_qa", "一箱零件共 96 个，质检抽查发现次品率为 12.5%。请问这箱零件中合格品有多少个？最后一行严格输出 `FINAL_ANSWER: <整数>`。", "84"),
    ]
    general_raw[12] = (
        "G_R13",
        "cn_qa",
        "某图书馆周一借出图书 85 本，周二比周一多借出 19 本，周三借出的图书数量恰好是周二的一半。请问这三天一共借出多少本图书？最后一行请严格输出 `FINAL_ANSWER: <整数>`。",
        "241",
    )
    general_raw[7] = (
        "G_R08",
        "logic",
        "A conference room booking rule states: meetings over 8 people require Room-B, unless the projector is needed, which always requires Room-C. A 10-person design review needs the projector. Which room is required? End with `FINAL_ANSWER: <Room>`.",
        "Room-C",
    )

    blind_inputs = [{"id": gid, "category": cat, "prompt": prm} for gid, cat, prm, _ans in general_raw]
    blind_oracles = {gid: ans for gid, _cat, _prm, ans in general_raw}

    bg_nl_prompts = [
        ("BG_NL01", "Explain in three concise bullet points how TCP three-way handshake establishes a reliable connection."),
        ("BG_NL02", "A bakery packs 9 muffins per box. How many full boxes can be made from 140 muffins, and how many muffins are left over?"),
        ("BG_NL03", "请用简明中文总结数据库索引（B+树索引）加快查询但会略微降低写入速度的原因（不超过120字）。"),
        ("BG_NL04", "Convert the following meeting note into 3 action items with owners: 'Alice will finalize API schema Tuesday; Bob updates unit tests Wednesday; Carol deploys staging Thursday.'"),
        ("BG_NL05", "If 5 identical machines produce 300 widgets in 4 hours, how many widgets will 8 machines produce in 3 hours at the same rate?"),
        ("BG_NL06", "请简述操作系统中进程（Process）与线程（Thread）在地址空间和上下文切换开销上的核心区别。"),
        ("BG_NL07", "Sort the following historical milestones chronologically from earliest to latest: World Wide Web proposal, transistor invention, ARPANET first message, Linux kernel 0.01."),
        ("BG_NL08", "A rectangular garden measures 18 meters by 12 meters. A 1-meter-wide gravel path runs around the outside perimeter. What is the area of the gravel path alone?"),
    ]
    bg_code_prompts = [
        ("BG_CD01", "Fix the Python function `rolling_sum_pairs(nums)` so it returns a list of sums of adjacent pairs `[nums[i] + nums[i+1]]` without raising IndexError on empty or single-element lists. Output only the Python `def` function.\n```python\ndef rolling_sum_pairs(nums):\n    return [nums[i] + nums[i + 1] for i in range(len(nums))]\n```"),
        ("BG_CD02", "Write only the Python function `invert_unique_map(mapping)` that swaps keys and values of a dict, raising `ValueError('duplicate')` if two keys share the same value."),
        ("BG_CD03", "Write only the Python function `clamp_scores(values, low, high)` that raises `ValueError('bounds')` if `low > high` and otherwise returns a list of values clamped into `[low, high]`."),
        ("BG_CD04", "Write only the Python function `first_non_repeating_char(text)` that returns the index of the first character appearing exactly once in `text`, or `-1` if none exists."),
        ("BG_CD05", "Write only the Python function `merge_intervals_sorted(intervals)` that merges overlapping `[start, end]` closed intervals and returns them sorted by `start`."),
        ("BG_CD06", "Write only the Python function `count_vowel_runs(word)` that counts how many contiguous substrings of vowels (`aeiouAEIOU`) appear in `word`."),
        ("BG_CD07", "Write only the Python function `balanced_brackets(expr)` that returns `True` if `()`, `[]`, and `{}` are properly nested and matched in `expr`, else `False`."),
        ("BG_CD08", "Write only the Python function `chunk_list(items, size)` that raises `ValueError('size')` if `size < 1` and otherwise splits `items` into consecutive sublists of length up to `size`."),
    ]
    bg_anchors = [{"id": bid, "modality": "nl", "prompt": bprm} for bid, bprm in bg_nl_prompts] + [
        {"id": bid, "modality": "code", "prompt": bprm} for bid, bprm in bg_code_prompts
    ]
    return blind_inputs, blind_oracles, bg_anchors


def _render_worked_examples(domain: str, seeds: list[dict], exclude_seed_id: str | None = None) -> str:
    blocks = ["SECTION 4. VERIFIED INSTITUTIONAL WORKED CASE EXEMPLARS"]
    for item in seeds:
        if exclude_seed_id is not None and item["id"] == exclude_seed_id:
            continue
        blocks.append(
            f"\n--- Worked Case {item['id']} ---\n"
            f"{item['vignette']}\n"
            f"Verified Output:\n{canonical_json(item['expected'])}"
        )
    return "\n".join(blocks)


def render_domain_long_messages(
    domain: str,
    case_obj: dict,
    public_bundle: dict,
    exclude_seed_id: str | None = None,
) -> list[dict]:
    if domain == "med_triage":
        dom_data = public_bundle["med_triage"]
        header = (
            "You are an institutional clinical pharmacy verification engine. "
            "Apply protocol MED-SOP-2026-V4 strictly without using external assumptions.\n\n"
            + dom_data["sop_manual_text"]
            + "\n"
            + _render_worked_examples("med_triage", dom_data["seeds"], exclude_seed_id=exclude_seed_id)
        )
        suffix = f"\n\n=== CURRENT CASE FOR REVIEW ===\n{case_obj['vignette']}\n\n{MED_TASK_SUFFIX_SPEC}"
        return [{"role": "user", "content": header + suffix}]
    if domain == "legal_compliance":
        dom_data = public_bundle["legal_compliance"]
        header = (
            "You are an enterprise legal compliance verification engine. "
            "Apply policy LEGAL-SOP-2026-V3 strictly without using external assumptions.\n\n"
            + dom_data["sop_manual_text"]
            + "\n"
            + _render_worked_examples("legal_compliance", dom_data["seeds"], exclude_seed_id=exclude_seed_id)
        )
        suffix = f"\n\n=== CURRENT CASE FOR REVIEW ===\n{case_obj['vignette']}\n\n{LEGAL_TASK_SUFFIX_SPEC}"
        return [{"role": "user", "content": header + suffix}]
    raise ValueError(f"Unsupported SOP domain: {domain}")


def render_domain_short_messages(domain: str, case_obj: dict) -> list[dict]:
    if domain == "med_triage":
        short_hdr = (
            "Apply institutional clinical pharmacy protocol MED-SOP-2026-V4 to verify this inpatient order."
        )
        suffix = f"\n\n=== CURRENT CASE FOR REVIEW ===\n{case_obj['vignette']}\n\n{MED_TASK_SUFFIX_SPEC}"
        return [{"role": "user", "content": short_hdr + suffix}]
    if domain == "legal_compliance":
        short_hdr = (
            "Apply enterprise legal procurement policy LEGAL-SOP-2026-V3 to review this vendor contract."
        )
        suffix = f"\n\n=== CURRENT CASE FOR REVIEW ===\n{case_obj['vignette']}\n\n{LEGAL_TASK_SUFFIX_SPEC}"
        return [{"role": "user", "content": short_hdr + suffix}]
    if domain == "general_retention":
        return [{"role": "user", "content": case_obj["prompt"]}]
    raise ValueError(f"Unsupported domain for short messages: {domain}")


def extract_first_json_object(raw_text: str) -> dict | None:
    if not isinstance(raw_text, str):
        return None
    stripped = raw_text.strip()
    search_pos = 0
    while True:
        start = stripped.find("{", search_pos)
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        matched = False
        for idx in range(start, len(stripped)):
            ch = stripped[idx]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        matched = True
                        candidate = stripped[start : idx + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                return parsed
                        except Exception:
                            pass
                        search_pos = start + 1
                        break
        if not matched:
            return None


def detect_cross_domain_contamination(target_domain: str, raw_text: str) -> bool:
    if not isinstance(raw_text, str):
        return True
    has_med_markers = any(tok in raw_text for tok in ('"renal_tier"', '"MED-R0', "R4_SEVERE", "CEFEPIME_IV"))
    has_leg_markers = any(tok in raw_text for tok in ('"primary_rule"', '"LEG-R0', "GC_BOARD_REJECT", "VP_LEGAL_CISO"))
    has_py_def = bool(re.search(r"(^|\n)\s*def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(", raw_text))

    if target_domain == "code_bugfix":
        return has_med_markers or has_leg_markers
    if target_domain == "med_triage":
        return has_leg_markers or has_py_def
    if target_domain == "legal_compliance":
        return has_med_markers or has_py_def
    if target_domain == "general_retention":
        return has_med_markers or has_leg_markers or has_py_def
    raise ValueError(f"Unknown target_domain: {target_domain}")


def score_structured_sop_prediction(domain: str, raw_text: str, expected_dict: dict) -> dict:
    keys = MED_KEYS if domain == "med_triage" else LEGAL_KEYS
    contaminated = detect_cross_domain_contamination(domain, raw_text)
    parsed = extract_first_json_object(raw_text)
    if parsed is None:
        return {
            "valid_json": False,
            "strict_pass": False,
            "matched_fields": 0,
            "total_fields": len(keys),
            "field_matches": {k: False for k in keys},
            "contaminated": contaminated,
        }
    field_matches = {}
    matched = 0
    for key_name in keys:
        exp_val = expected_dict[key_name]
        got_val = parsed.get(key_name)
        is_ok = (got_val == exp_val) and (type(got_val) is type(exp_val))
        field_matches[key_name] = bool(is_ok)
        if is_ok:
            matched += 1
    return {
        "valid_json": True,
        "strict_pass": matched == len(keys) and not contaminated,
        "matched_fields": matched,
        "total_fields": len(keys),
        "field_matches": field_matches,
        "contaminated": contaminated,
        "parsed": parsed,
    }


def score_general_retention_prediction(task_obj: dict, raw_text: str, expected_answer: str) -> dict:
    contaminated = detect_cross_domain_contamination("general_retention", raw_text)
    if not isinstance(raw_text, str):
        return {"passed": False, "extracted": None, "contaminated": True}
    matches = re.findall(r"FINAL_ANSWER:\s*([^\n\r`]+)", raw_text)
    if matches:
        extracted = matches[-1].strip().rstrip(".")
    else:
        lines = [ln.strip() for ln in raw_text.strip().splitlines() if ln.strip()]
        extracted = lines[-1] if lines else ""

    norm_got = extracted.strip().strip("*").strip()
    norm_exp = expected_answer.strip()
    passed = (norm_got.lower() == norm_exp.lower()) and not contaminated
    return {
        "passed": passed,
        "extracted": norm_got,
        "expected": norm_exp,
        "contaminated": contaminated,
    }


def build_and_verify_cross_domain_dataset() -> dict:
    med_seeds, med_unlabeled, med_blind_in, med_oracles = _build_med_cases()
    leg_seeds, leg_unlabeled, leg_blind_in, leg_oracles = _build_legal_cases()
    gen_blind_in, gen_oracles, bg_anchors = _build_general_retention_and_background()

    med_manual_padded = (
        MED_SOP_MANUAL_BODY
        + "\nSECTION 3B. CLINICAL AUDIT CHECKLIST & PHARMACOKINETIC RATIONALE\n"
        + "- Verify raw eGFR lab timestamp within 24 hours of inpatient order entry.\n"
        + "- Confirm serum albumin in g/dL when patient age is 75 years or older: hypoalbuminemia (<3.2 g/dL) increases unbound drug fraction and reflects sarcopenic creatinine overestimation of glomerular filtration, mandating the -12 mL/min/1.73m2 effective eGFR deduction prior to tier lookup.\n"
        + "- KDIGO AKI Stage 2 (serum creatinine 2.0-2.9x baseline) or Stage 3 (>=3.0x baseline or renal replacement) reflects acute non-steady-state filtration collapse; raw eGFR equations lag behind true clearance, so `R4_SEVERE` dosing is mandatory.\n"
        + "- QTc >= 510 ms represents an institutional telemetry redline due to additive hERG potassium channel blockade under acute illness; hold order immediately (`HOLD_STAT`, `NONE`, `0 mg`, `MED-R01`) before evaluating any drug-specific rule.\n"
        + "- For `APIXABAN_PO`, active hemorrhage or platelet count `< 45 K/uL` poses unacceptably high major bleeding risk (`MED-R04`); antimicrobial orders (`VANCO_IV`, `CEFEPIME_IV`, `MEROPENEM_IV`, `LINEZOLID_PO`, `PIPTAZO_IV`, `DAPTO_IV`) proceed through standard verification despite thrombocytopenia.\n"
        + "- Concurrent MAOI therapy with oxazolidinone `LINEZOLID_PO` risks life-threatening serotonin toxicity (`MED-R02`); switch to lipopeptide `DAPTO_IV` dosed strictly per the patient's computed `renal_tier`.\n"
        + "- Serum potassium `>= 5.7 mmol/L` contraindicates standard `PIPTAZO_IV` formulation under renal/electrolyte stewardship (`MED-R03`); switch to `MEROPENEM_IV` dosed strictly per the patient's computed `renal_tier`.\n"
        + "- Under `MED-R05`, compare the dispensed `dose_mg` against that specific regimen's `R1_NORMAL` reference column: use `STD_APPROVE` whenever the milligram amount equals the `R1_NORMAL` dose (including `LINEZOLID_PO` 600 mg in any tier, or `MEROPENEM_IV` 1000 mg / `APIXABAN_PO` 10 mg in `R2_MILD`), and `ADJUST_APPROVE` whenever the milligram amount is stepped down.\n"
    )

    leg_manual_padded = (
        LEGAL_SOP_MANUAL_BODY
        + "\nSECTION 3B. CORPORATE COUNSEL VERIFICATION NOTES & DEFINITIONS\n"
        + "- Always compute `min_cap_x` from Section 1 (`vendor_tier` x `data_class`) first and include that exact integer (`1`, `2`, `3`, or `5`) in every JSON output, even when `GC_BOARD_REJECT` or `VP_LEGAL_CISO` is triggered.\n"
        + "- Biometric and high-sensitivity personal identifiers (`D0_BIO_PII`) transferred to any non-domestic region (`cross_border != \"NONE\"`) without a completed regulatory security assessment (`has_sec_assessment == false`) constitute a statutory blocker (`LEG-R01`, `cure_days = 0`). Domestic storage (`cross_border == \"NONE\"`) or non-D0 tiers (`D1_CUST_PII`, `D2_CONFID`, `D3_PUBLIC`) do NOT trigger `LEG-R01`.\n"
        + "- `FOREIGN_LITIGATION` exposes the enterprise to foreign discovery and injunctive orders (`LEG-R02`, `cure_days = 15`); `NEUTRAL_ARBITRATION` (e.g., HKIAC / SIAC) and `DOMESTIC_COURT` are pre-cleared venues.\n"
        + "- Third-party intellectual property infringement indemnity must be uncapped (`ip_indemnity_uncapped == true`) whenever confidential, customer, or biometric data (`D0_BIO_PII`, `D1_CUST_PII`, `D2_CONFID`) is in scope (`LEG-R03`, `cure_days = 15`); pure `D3_PUBLIC` commodity tools are exempt.\n"
        + "- Multi-year evergreen auto-renewal (`auto_renew_months > 12`) is only permissible when paired with an explicit termination-for-convenience clause (`terminate_for_conv == true`); otherwise escalate under `LEG-R04` (`cure_days = 30`).\n"
        + "- When all four structural redlines (`LEG-R01`..`LEG-R04`) pass, compare `proposed_cap_x` against `min_cap_x`: a shortfall (`proposed_cap_x < min_cap_x`) routes to `SENIOR_COUNSEL` (`LEG-R05`, `ELEVATED`, `cure_days = 30`), whereas meeting or exceeding `min_cap_x` clears `AUTO_APPROVE` (`LEG-R06`, `LOW`, `cure_days = 45`).\n"
    )

    public_bundle = {
        "version": "cross_domain_v1",
        "med_triage": {
            "sop_manual_text": med_manual_padded,
            "seeds": med_seeds,
            "unlabeled": med_unlabeled,
            "blind_test_inputs": med_blind_in,
        },
        "legal_compliance": {
            "sop_manual_text": leg_manual_padded,
            "seeds": leg_seeds,
            "unlabeled": leg_unlabeled,
            "blind_test_inputs": leg_blind_in,
        },
        "general_retention": {
            "blind_test_inputs": gen_blind_in,
        },
        "background_anchors": bg_anchors,
    }

    blind_oracles = {
        "version": "cross_domain_v1",
        "med_triage": med_oracles,
        "legal_compliance": leg_oracles,
        "general_retention": gen_oracles,
    }

    pub_bytes = (json.dumps(public_bundle, indent=2, sort_keys=True) + "\n").encode("utf-8")
    orc_bytes = (json.dumps(blind_oracles, indent=2, sort_keys=True) + "\n").encode("utf-8")
    pub_str = pub_bytes.decode("utf-8")

    zero_leak = True
    for dom_k in ("med_triage", "legal_compliance"):
        for b_obj in public_bundle[dom_k]["blind_test_inputs"]:
            if set(b_obj.keys()) != {"id", "input", "vignette"}:
                zero_leak = False
        for u_obj in public_bundle[dom_k]["unlabeled"]:
            if set(u_obj.keys()) != {"id", "input", "vignette"}:
                zero_leak = False
    for g_obj in public_bundle["general_retention"]["blind_test_inputs"]:
        if set(g_obj.keys()) != {"id", "category", "prompt"}:
            zero_leak = False
        if blind_oracles["general_retention"][g_obj["id"]] in g_obj["prompt"] and g_obj["category"] == "math":
            zero_leak = False

    manifest = {
        "version": "cross_domain_v1",
        "public_bundle_sha256": sha256_bytes(pub_bytes),
        "blind_oracles_sha256": sha256_bytes(orc_bytes),
        "zero_blind_oracle_in_public_bundle": zero_leak,
        "counts": {
            "med_seeds": len(med_seeds),
            "med_unlabeled": len(med_unlabeled),
            "med_blind_test": len(med_blind_in),
            "legal_seeds": len(leg_seeds),
            "legal_unlabeled": len(leg_unlabeled),
            "legal_blind_test": len(leg_blind_in),
            "general_blind_test": len(gen_blind_in),
            "background_anchors": len(bg_anchors),
        },
    }
    return {
        "public_bundle": public_bundle,
        "blind_oracles": blind_oracles,
        "manifest": manifest,
        "public_bundle_bytes": pub_bytes,
        "blind_oracles_bytes": orc_bytes,
    }


def write_cross_domain_data_files(target_dir: Path | None = None) -> dict:
    out_dir = target_dir or (Path(__file__).resolve().parent / "data/cross_domain_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    built = build_and_verify_cross_domain_dataset()
    if not built["manifest"]["zero_blind_oracle_in_public_bundle"]:
        raise RuntimeError("Oracle leak detected in public_bundle")
    (out_dir / "public_bundle.json").write_bytes(built["public_bundle_bytes"])
    (out_dir / "blind_oracles.json").write_bytes(built["blind_oracles_bytes"])
    (out_dir / "manifest.json").write_text(
        json.dumps(built["manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return built["manifest"]


if __name__ == "__main__":
    print(json.dumps(write_cross_domain_data_files(), indent=2))
