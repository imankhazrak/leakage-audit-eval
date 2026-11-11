"""Roles for Phase-3 fields. Behavioral columns are the only wide-matrix fields."""

from __future__ import annotations

from leakagelens.phase3.constants import (
    CORE_METRICS,
    GAP_METRICS,
    GAP_PAIRS,
    INTERVENTIONS,
    PARTITIONS,
)

BEHAVIORAL = "behavioral_feature"
METADATA = "metadata"
LABEL = "label"
AUDIT = "audit_only"

FORBIDDEN_WIDE_FIELDS = (
    "episode_label",
    "leakage_family",
    "severity",
    "severity_label",
    "severity_requested",
    "severity_achieved",
    "dataset",
    "dataset_id",
    "dataset_name",
    "source_family",
    "source_family_id",
    "model_seed",
    "split_seed",
    "rows_leaked",
    "groups_leaked",
    "future_fraction",
    "proxy_accuracy",
    "proxy_mi",
    "preprocessing_contamination_fraction",
)


def behavioral_column_names(interventions: tuple[str, ...] | None = None) -> list[str]:
    """Behavioral column names for the given intervention ids (default: the Phase-3 I1-I5)."""

    columns: list[str] = []
    for intervention in interventions if interventions is not None else INTERVENTIONS:
        for partition in PARTITIONS:
            for metric in CORE_METRICS:
                for suffix in ("before", "after", "raw_delta", "oriented_delta"):
                    columns.append(f"{intervention}__{partition}__{metric}__{suffix}")
        for left, right in GAP_PAIRS:
            for metric in GAP_METRICS:
                for suffix in ("gap_before", "gap_after", "gap_raw_delta", "gap_oriented_delta"):
                    columns.append(f"{intervention}__{left}_minus_{right}__{metric}__{suffix}")
        for partition in ("validation", "test"):
            for metric in (
                "predicted_class_disagreement_rate",
                "mean_absolute_probability_change",
                "median_absolute_probability_change",
                "maximum_absolute_probability_change",
                "mean_confidence_change",
                "mean_jensen_shannon",
                "median_jensen_shannon",
            ):
                columns.append(f"{intervention}__{partition}__{metric}")
        for metric in ("importance_l1", "importance_norm_change", "topk_overlap", "rank_correlation"):
            columns.append(f"{intervention}__{metric}")
    return columns


def feature_schema() -> dict[str, str]:
    roles = {name: BEHAVIORAL for name in behavioral_column_names()}
    roles["episode_id"] = METADATA
    for name in (
        "dataset_id",
        "dataset_name",
        "source_family_id",
        "model_family",
        "model_backend",
        "split_hash",
        "effective_replicate_id",
        "base_reference_id",
        "mechanism_version",
        "episode_track",
        "purity_status",
    ):
        roles[name] = METADATA
    for name in ("episode_label", "leakage_family", "severity_label", "severity_requested", "severity_achieved"):
        roles[name] = LABEL
    for name in (
        "row_overlap_train_test",
        "future_rows_in_train",
        "contaminated_groups",
        "proxy_target_mutual_information",
        "empirical_proxy_accuracy",
        "preprocessor_fit_rows",
    ):
        roles[name] = AUDIT
    return roles


def validate_wide_columns(columns: list[str]) -> list[str]:
    """Return firewall violations. episode_id is the only non-behavioral column."""

    allowed = set(behavioral_column_names()) | {"episode_id"}
    problems = []
    for column in columns:
        if column not in allowed:
            problems.append(column)
        lowered = column.lower()
        for forbidden in FORBIDDEN_WIDE_FIELDS:
            if forbidden == column or lowered == forbidden:
                problems.append(column)
    return problems
