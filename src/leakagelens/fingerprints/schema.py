"""Schema for planned behavioral fingerprints.

This module names fields and the metric-orientation convention. It does not
compute fingerprints from models or datasets.

Raw and oriented deltas
-----------------------
For a metric M:

    raw_delta = M_suspected - M_corrected
    oriented_delta = s_M * raw_delta

where s_M is +1 when higher values are better and -1 when lower values are
better. A positive oriented delta means the suspected pipeline looks better
on that metric than the corrected pipeline, which is the direction leakage
often (but not always) produces.

Normalization statistics are not computed here. When they are, they must be
fit on detector-training datasets only. See the meta-leakage invariant in
docs/methodology_blueprint.md.
"""

from __future__ import annotations

from typing import Final

from leakagelens.constants import MetricOrientation

FINGERPRINT_SCHEMA_VERSION: Final[str] = "0.1"

FINGERPRINT_BLOCKS: Final[tuple[str, ...]] = (
    "predictive",
    "calibration",
    "structural",
    "feature",
    "stability",
    "counterfactual",
)

# One record per metric. Deltas are not stored as bare numbers without orientation.
METRIC_RECORD_FIELDS: Final[tuple[str, ...]] = (
    "metric_name",
    "block",
    "suspected_value",
    "corrected_value",
    "raw_delta",
    "orientation",
    "orientation_sign",
    "oriented_delta",
    "normalization",
    "replicate_mean",
    "replicate_std",
    "n_replicates",
)

NORMALIZATION_METADATA_FIELDS: Final[tuple[str, ...]] = (
    "method",
    "fit_scope",
    "fit_dataset_ids",
    "schema_version",
)

REQUIRED_FIT_SCOPE: Final[str] = "detector_train_datasets_only"

METRIC_SPECS: Final[dict[str, dict[str, str]]] = {
    "auroc": {"block": "predictive", "orientation": MetricOrientation.HIGHER_IS_BETTER.value},
    "auprc": {"block": "predictive", "orientation": MetricOrientation.HIGHER_IS_BETTER.value},
    "accuracy": {"block": "predictive", "orientation": MetricOrientation.HIGHER_IS_BETTER.value},
    "balanced_accuracy": {
        "block": "predictive",
        "orientation": MetricOrientation.HIGHER_IS_BETTER.value,
    },
    "macro_f1": {"block": "predictive", "orientation": MetricOrientation.HIGHER_IS_BETTER.value},
    "log_loss": {"block": "predictive", "orientation": MetricOrientation.LOWER_IS_BETTER.value},
    "brier": {"block": "calibration", "orientation": MetricOrientation.LOWER_IS_BETTER.value},
    "ece": {"block": "calibration", "orientation": MetricOrientation.LOWER_IS_BETTER.value},
    "mean_confidence": {
        "block": "calibration",
        "orientation": MetricOrientation.UNSIGNED.value,
    },
    "exact_overlap_rate": {
        "block": "structural",
        "orientation": MetricOrientation.LOWER_IS_BETTER.value,
    },
    "near_duplicate_rate": {
        "block": "structural",
        "orientation": MetricOrientation.LOWER_IS_BETTER.value,
    },
    "group_overlap_rate": {
        "block": "structural",
        "orientation": MetricOrientation.LOWER_IS_BETTER.value,
    },
    "temporal_violation_rate": {
        "block": "structural",
        "orientation": MetricOrientation.LOWER_IS_BETTER.value,
    },
    "feature_proximity": {
        "block": "structural",
        "orientation": MetricOrientation.LOWER_IS_BETTER.value,
    },
    "importance_shift": {
        "block": "feature",
        "orientation": MetricOrientation.UNSIGNED.value,
    },
    "permutation_importance_shift": {
        "block": "feature",
        "orientation": MetricOrientation.UNSIGNED.value,
    },
    "proxy_sensitivity": {
        "block": "feature",
        "orientation": MetricOrientation.UNSIGNED.value,
    },
    "suspicious_feature_sensitivity": {
        "block": "feature",
        "orientation": MetricOrientation.UNSIGNED.value,
    },
    "seed_metric_std": {
        "block": "stability",
        "orientation": MetricOrientation.LOWER_IS_BETTER.value,
    },
    "split_metric_std": {
        "block": "stability",
        "orientation": MetricOrientation.LOWER_IS_BETTER.value,
    },
    "post_correction_instability": {
        "block": "stability",
        "orientation": MetricOrientation.LOWER_IS_BETTER.value,
    },
}

# Feature-shift and mean-confidence metrics are unsigned: a larger value is not
# a performance improvement. Oriented deltas are defined only for signed metrics.

ORIENTATION_SIGN: Final[dict[str, int]] = {
    MetricOrientation.HIGHER_IS_BETTER.value: 1,
    MetricOrientation.LOWER_IS_BETTER.value: -1,
}


def orientation_sign(orientation: str) -> int:
    """Return +1 or -1 for a metric orientation label."""

    try:
        return ORIENTATION_SIGN[orientation]
    except KeyError as exc:
        known = sorted(ORIENTATION_SIGN)
        raise ValueError(f"Unknown metric orientation '{orientation}'. Known: {known}") from exc


def raw_delta(suspected: float, corrected: float) -> float:
    """Return M_suspected - M_corrected. Sign is metric-specific."""

    return float(suspected) - float(corrected)


def oriented_delta(suspected: float, corrected: float, orientation: str) -> float | None:
    """Return s_M * (M_suspected - M_corrected), or None if unsigned."""

    if orientation == MetricOrientation.UNSIGNED.value:
        return None
    return orientation_sign(orientation) * raw_delta(suspected, corrected)


def metric_names_by_block() -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {block: [] for block in FINGERPRINT_BLOCKS}
    for name, spec in METRIC_SPECS.items():
        grouped[spec["block"]].append(name)
    return {block: tuple(names) for block, names in grouped.items()}
