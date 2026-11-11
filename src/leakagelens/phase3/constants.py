"""Frozen Phase-3 mechanism names, severities, and intervention ids.

Severity numbers are mechanism-specific. They are not comparable across families.
"""

from __future__ import annotations

from leakagelens.phase3 import (
    CANONICAL_DETERMINISTIC_SEED,
    MECHANISM_VERSION,
    PROXY_COLUMN,
)

FAMILY_L1 = "L1"
FAMILY_L2 = "L2"
FAMILY_L3 = "L3"
FAMILY_L4 = "L4"
FAMILY_L5 = "L5"
FAMILY_NONE = "none"
FAMILIES = (FAMILY_L1, FAMILY_L2, FAMILY_L3, FAMILY_L4, FAMILY_L5)

SEVERITY_LOW = "low"
SEVERITY_MODERATE = "moderate"
SEVERITY_HIGH = "high"
SEVERITY_NONE = "none"
SEVERITY_ORDER = (SEVERITY_LOW, SEVERITY_MODERATE, SEVERITY_HIGH)

# Requested structural strengths. Keys are labels, values are the requested
# parameter. L4 stores the held-out visibility fraction that the fit scope implies.
SEVERITY_REQUESTED: dict[str, dict[str, float]] = {
    FAMILY_L1: {SEVERITY_LOW: 0.05, SEVERITY_MODERATE: 0.15, SEVERITY_HIGH: 0.30},
    FAMILY_L2: {SEVERITY_LOW: 0.10, SEVERITY_MODERATE: 0.30, SEVERITY_HIGH: 0.50},
    FAMILY_L3: {SEVERITY_LOW: 0.10, SEVERITY_MODERATE: 0.30, SEVERITY_HIGH: 0.60},
    FAMILY_L4: {SEVERITY_LOW: 0.25, SEVERITY_MODERATE: 1.0, SEVERITY_HIGH: 1.0},
    FAMILY_L5: {SEVERITY_LOW: 0.25, SEVERITY_MODERATE: 0.50, SEVERITY_HIGH: 0.75},
}

SEVERITY_PARAMETER = {
    FAMILY_L1: "fraction_of_test_rows_copied_to_train",
    FAMILY_L2: "fraction_of_eligible_heldout_groups_contaminated",
    FAMILY_L3: "fraction_of_eligible_future_validation_rows_exposed_to_training",
    FAMILY_L4: "fraction_of_heldout_rows_visible_to_preprocessor",
    FAMILY_L5: "chance_adjusted_proxy_strength",
    FAMILY_NONE: "none",
}

SEVERITY_UNIT = {
    FAMILY_L1: "fraction_of_test_rows",
    FAMILY_L2: "fraction_of_eligible_heldout_groups",
    FAMILY_L3: "fraction_of_validation_rows",
    FAMILY_L4: "fraction_of_heldout_rows",
    FAMILY_L5: "chance_adjusted_strength",
    FAMILY_NONE: "none",
}

# L4 high includes the test partition as well as validation. The requested
# held-out fraction is therefore 1.0 for both moderate and high; the scope
# label distinguishes them. Achieved fraction is computed from the actual rows.
L4_SCOPE = {
    SEVERITY_LOW: "train_plus_25pct_validation",
    SEVERITY_MODERATE: "train_plus_100pct_validation",
    SEVERITY_HIGH: "train_plus_validation_plus_test",
}

INTERVENTION_I1 = "I1"
INTERVENTION_I2 = "I2"
INTERVENTION_I3 = "I3"
INTERVENTION_I4 = "I4"
INTERVENTION_I5 = "I5"
INTERVENTIONS = (
    INTERVENTION_I1,
    INTERVENTION_I2,
    INTERVENTION_I3,
    INTERVENTION_I4,
    INTERVENTION_I5,
)
INTERVENTION_NAMES = {
    INTERVENTION_I1: "overlap_decontamination",
    INTERVENTION_I2: "group_decontamination",
    INTERVENTION_I3: "chronological_restoration",
    INTERVENTION_I4: "train_only_preprocessing_refit",
    INTERVENTION_I5: "target_proxy_removal",
}
MATCHED_INTERVENTION = {
    FAMILY_L1: INTERVENTION_I1,
    FAMILY_L2: INTERVENTION_I2,
    FAMILY_L3: INTERVENTION_I3,
    FAMILY_L4: INTERVENTION_I4,
    FAMILY_L5: INTERVENTION_I5,
}

TRACK_CLEAN = "clean_control"
TRACK_PRIMARY = "primary_single_family"
TRACK_MIXED = "mixed_or_confounded"
TRACK_AUXILIARY = "auxiliary_conditional"
TRACK_INVALID = "invalid"

STATUS_CHANGED = "changed"
STATUS_NO_VIOLATION = "no_violation_detected"
STATUS_NOT_APPLICABLE = "not_applicable"
STATUS_FAILED = "failed"

LABEL_CLEAN = "clean"
LABEL_LEAKED = "leaked"

RESPONSE_ZERO = "zero_or_near_zero"
RESPONSE_OBSERVED = "observed"

CORE_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "auroc",
    "auprc",
    "log_loss",
    "brier",
    "ece",
    "mean_confidence",
)
PARTITIONS = ("train", "validation", "test")
GAP_PAIRS = (("train", "test"), ("validation", "test"))
GAP_METRICS = ("accuracy", "balanced_accuracy", "auroc", "log_loss")

__all__ = [
    "CANONICAL_DETERMINISTIC_SEED",
    "MECHANISM_VERSION",
    "PROXY_COLUMN",
]
