"""Label-blind episode decision. The planted mechanism is not an input."""

from __future__ import annotations

import math
from typing import Mapping

RESPONSE_TOLERANCE = 1e-12
PRIMARY_METRIC = {"classification": "log_loss", "regression": "rmse"}
# Controls are paired by the kind of edit, not by a truth label.
CONTROL_OF = {"I1": "P1", "I2": "P1", "I3": "P1", "I4": "P5", "I5": "P5"}
DIAGNOSIS_OF = {
    "I1": "L1_overlap",
    "I2": "L2_group",
    "I3": "L3_temporal",
    "I4": "L4_preprocessing",
    "I5": "L5_proxy",
}


def _blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return value == ""


def rule_outcome(rule_status: object, rule_delta: object, control_status: object, control_delta: object) -> str:
    """Compare one rule with its control on one metric."""

    if rule_status == "not_applicable":
        return "not_applicable"
    if rule_status == "failed" or control_status in {"failed", "not_applicable"} or _blank(rule_delta) or _blank(control_delta):
        return "abstain"
    rule_value = float(rule_delta)
    control_value = float(control_delta)
    if not math.isfinite(rule_value) or not math.isfinite(control_value):
        return "abstain"
    if abs(rule_value) <= RESPONSE_TOLERANCE:
        return "no_change"
    if abs(rule_value) > abs(control_value) + RESPONSE_TOLERANCE:
        return "fire"
    return "no_fire"


def decide_episode(rules: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Decide from applicable rules and their controls.

    ``rules`` maps a rule id to ``status`` and ``oriented_delta`` for the
    task's primary metric. No leakage-family field is read.
    """

    outcomes = {
        rule: rule_outcome(
            rules.get(rule, {}).get("status", "not_applicable"),
            rules.get(rule, {}).get("oriented_delta"),
            rules.get(control, {}).get("status", "not_applicable"),
            rules.get(control, {}).get("oriented_delta"),
        )
        for rule, control in CONTROL_OF.items()
    }
    fires = [rule for rule, outcome in outcomes.items() if outcome == "fire"]
    fires.sort(key=lambda rule: abs(float(rules[rule]["oriented_delta"])), reverse=True)
    if fires:
        decision = "positive"
        if len(fires) > 1 and abs(abs(float(rules[fires[0]]["oriented_delta"])) - abs(float(rules[fires[1]]["oriented_delta"]))) <= RESPONSE_TOLERANCE:
            diagnosis = "abstain_tie"
        else:
            diagnosis = DIAGNOSIS_OF[fires[0]]
    elif any(outcome == "abstain" for outcome in outcomes.values()):
        decision = "abstain"
        diagnosis = "abstain"
    elif any(outcome in {"no_fire", "no_change"} for outcome in outcomes.values()):
        decision = "negative"
        diagnosis = "abstain"
    else:
        decision = "abstain"
        diagnosis = "abstain"
    return {
        "decision": decision,
        "diagnosis": diagnosis,
        "firing_rules": " ".join(fires),
        "rule_outcomes": outcomes,
    }
