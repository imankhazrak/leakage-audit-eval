"""Counterfactual corrective interventions.

A CorrectiveIntervention maps a suspected pipeline to a leakage-safer counterpart
without changing the dataset identity, model family, or random seed unless that
change is the intervention itself (for example, replacing a random split with a
group-aware split).

Prompt 8A adds the label-blind implementation. ``SuspectedPipeline`` is the
auditor's view of a pipeline and cannot carry injection identity; ``rules``
holds the I1-I5 corrective rules and the P1/P5 placebo controls; ``execute``
refits before and after each rule and records the response. The Phase-3
oracle-assisted substitution in ``leakagelens.phase3.execute`` is unchanged and
is documented as oracle-assisted.
"""

from leakagelens.constants import InterventionType
from leakagelens.interventions.pipeline import (
    FORBIDDEN_STATE_FIELDS,
    InformationFlowError,
    SuspectedPipeline,
    suspected_from_state,
)
from leakagelens.interventions.rules import (
    ALL_RULES,
    CORRECTIVE_RULES,
    PLACEBO_RULES,
    RULE_NAMES,
    RuleConfig,
    StructuralChange,
    apply_rule,
)

__all__ = [
    "ALL_RULES",
    "CORRECTIVE_RULES",
    "FORBIDDEN_STATE_FIELDS",
    "InformationFlowError",
    "InterventionType",
    "PLACEBO_RULES",
    "RULE_NAMES",
    "RuleConfig",
    "StructuralChange",
    "SuspectedPipeline",
    "apply_rule",
    "suspected_from_state",
]
