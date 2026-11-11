"""Label-blind corrective rules and placebo controls.

Every rule receives a :class:`SuspectedPipeline` and returns a corrected
pipeline plus a :class:`StructuralChange` record. Rules read partition
memberships, predictor values, declared group or time metadata, and the
preprocessing fit scope. They do not read column names, injection flags,
donor identities, or the original clean split.

Rule semantics (frozen for the Prompt-8 pilot):

* ``I1`` overlap decontamination: training rows whose exact predictor row also
  appears in a held-out partition are removed from training.
* ``I2`` group decontamination: training rows whose group label appears in a
  held-out partition are removed. Not applicable without declared groups.
* ``I3`` chronological restoration: training rows whose time key is at or after
  the earliest held-out time key are removed. Not applicable without time.
* ``I4`` train-only preprocessing refit: preprocessing is refit on the
  estimator's training rows regardless of the suspected scope.
* ``I5`` target-proxy removal: predictors whose training-only association with
  the target exceeds a preregistered threshold are dropped.
* ``P1`` placebo: a random training subset matched in size to ``I1``'s removal
  (or a fixed fraction when ``I1`` removed nothing) is removed.
* ``P5`` placebo: one random predictor is dropped.

Interventions that change nothing are still executed downstream as a refit.
Their response comes from that refit, never from a forced equality.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from leakagelens.interventions.pipeline import SuspectedPipeline
from leakagelens.phase2.splits import duplicate_cluster_ids, time_order_key

RULE_I1 = "I1"
RULE_I2 = "I2"
RULE_I3 = "I3"
RULE_I4 = "I4"
RULE_I5 = "I5"
RULE_P1 = "P1"
RULE_P5 = "P5"
CORRECTIVE_RULES: tuple[str, ...] = (RULE_I1, RULE_I2, RULE_I3, RULE_I4, RULE_I5)
PLACEBO_RULES: tuple[str, ...] = (RULE_P1, RULE_P5)
ALL_RULES: tuple[str, ...] = CORRECTIVE_RULES + PLACEBO_RULES
RULE_NAMES = {
    RULE_I1: "label_blind_overlap_decontamination",
    RULE_I2: "label_blind_group_decontamination",
    RULE_I3: "label_blind_chronological_restoration",
    RULE_I4: "label_blind_train_only_preprocessing_refit",
    RULE_I5: "label_blind_target_association_screen",
    RULE_P1: "placebo_random_training_row_removal",
    RULE_P5: "placebo_random_predictor_drop",
}

APPLIED = "applied"
NO_CHANGE = "applied_no_structural_change"
NOT_APPLICABLE = "not_applicable"


class RuleError(ValueError):
    pass


@dataclass(frozen=True)
class RuleConfig:
    """Preregistered rule parameters. Values come from ``configs/phase8/pilot.yaml``."""

    proxy_screen_threshold: float
    proxy_screen_bins: int
    placebo_row_fraction: float
    placebo_seed_offset: int
    proxy_screen_max_target_levels: int | None = None
    proxy_availability_policy: str = "legacy"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RuleConfig":
        try:
            levels = payload.get("proxy_screen_max_target_levels")
            return cls(
                proxy_screen_threshold=float(payload["proxy_screen_threshold"]),
                proxy_screen_bins=int(payload["proxy_screen_bins"]),
                placebo_row_fraction=float(payload["placebo_row_fraction"]),
                placebo_seed_offset=int(payload["placebo_seed_offset"]),
                proxy_screen_max_target_levels=None if levels is None else int(levels),
                proxy_availability_policy=str(payload.get("proxy_availability_policy", "legacy")),
            )
        except KeyError as exc:
            raise RuleError(f"rule configuration is missing {exc.args[0]!r}") from exc

    def __post_init__(self) -> None:
        if not 0.0 < self.proxy_screen_threshold <= 1.0:
            raise RuleError("proxy_screen_threshold must lie in (0, 1]")
        if self.proxy_screen_bins < 2:
            raise RuleError("proxy_screen_bins must be at least 2")
        if not 0.0 < self.placebo_row_fraction < 1.0:
            raise RuleError("placebo_row_fraction must lie in (0, 1)")
        if self.proxy_screen_max_target_levels is not None and self.proxy_screen_max_target_levels < 2:
            raise RuleError("proxy_screen_max_target_levels must be at least 2")
        if self.proxy_availability_policy not in {"legacy", "require_boolean_v1"}:
            raise RuleError("proxy_availability_policy is not a known policy")


def prospective_rule_config(payload: Mapping[str, Any]) -> RuleConfig:
    """Build a rule config for a new run.

    Historical configs may omit the policy and then use ``legacy``. A
    prospective caller has to name ``require_boolean_v1`` explicitly.
    """

    if payload.get("proxy_availability_policy") != "require_boolean_v1":
        raise RuleError("a prospective run must set proxy_availability_policy to require_boolean_v1")
    return RuleConfig.from_mapping(payload)


@dataclass
class StructuralChange:
    rule_id: str
    rule_name: str
    status: str
    rows_removed_from_training: int
    feature_positions_removed: list[int]
    preprocessor_fit_rows_before: int
    preprocessor_fit_rows_after: int
    training_rows_before: int
    training_rows_after: int
    detail: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _change(
    rule_id: str,
    before: SuspectedPipeline,
    after: SuspectedPipeline,
    *,
    status: str,
    removed_rows: int,
    removed_features: list[int],
    detail: dict[str, Any],
) -> StructuralChange:
    return StructuralChange(
        rule_id=rule_id,
        rule_name=RULE_NAMES[rule_id],
        status=status,
        rows_removed_from_training=int(removed_rows),
        feature_positions_removed=[int(item) for item in removed_features],
        preprocessor_fit_rows_before=int(len(before.preprocessor_fit_idx)),
        preprocessor_fit_rows_after=int(len(after.preprocessor_fit_idx)),
        training_rows_before=int(len(before.train_idx)),
        training_rows_after=int(len(after.train_idx)),
        detail=detail,
    )


def _remove_training_rows(pipeline: SuspectedPipeline, remove: np.ndarray) -> SuspectedPipeline:
    """Drop rows from the estimator's training set and from the preprocessing scope.

    The preprocessing scope keeps any other rows it already had, so a row
    removal rule does not silently also perform ``I4``.
    """

    remove_set = set(int(item) for item in remove)
    if not remove_set:
        return pipeline
    kept = np.asarray([index for index in pipeline.train_idx if int(index) not in remove_set], dtype=int)
    if len(kept) == 0:
        raise RuleError("An intervention cannot remove every training row")
    scope = np.asarray([index for index in pipeline.preprocessor_fit_idx if int(index) not in remove_set], dtype=int)
    return pipeline.with_training_rows(kept).with_preprocessor_scope(scope)


def rule_i1(pipeline: SuspectedPipeline, config: RuleConfig) -> tuple[SuspectedPipeline, StructuralChange]:
    hashes = duplicate_cluster_ids(pipeline.features).to_numpy()
    heldout_hashes = set(hashes[pipeline.heldout_idx].tolist())
    remove = np.asarray([index for index in pipeline.train_idx if hashes[int(index)] in heldout_hashes], dtype=int)
    after = _remove_training_rows(pipeline, remove)
    status = APPLIED if len(remove) else NO_CHANGE
    return after, _change(
        RULE_I1,
        pipeline,
        after,
        status=status,
        removed_rows=len(remove),
        removed_features=[],
        detail={"exact_overlap_rows_found": int(len(remove)), "heldout_unique_rows": int(len(heldout_hashes))},
    )


def rule_i2(pipeline: SuspectedPipeline, config: RuleConfig) -> tuple[SuspectedPipeline, StructuralChange]:
    if pipeline.groups is None:
        return pipeline, _change(
            RULE_I2,
            pipeline,
            pipeline,
            status=NOT_APPLICABLE,
            removed_rows=0,
            removed_features=[],
            detail={"reason": "no declared group metadata"},
        )
    labels = pipeline.groups.astype(str).to_numpy()
    heldout_groups = set(labels[pipeline.heldout_idx].tolist())
    remove = np.asarray([index for index in pipeline.train_idx if labels[int(index)] in heldout_groups], dtype=int)
    after = _remove_training_rows(pipeline, remove)
    return after, _change(
        RULE_I2,
        pipeline,
        after,
        status=APPLIED if len(remove) else NO_CHANGE,
        removed_rows=len(remove),
        removed_features=[],
        detail={
            "crossing_groups": int(len(set(labels[remove].tolist()))) if len(remove) else 0,
            "heldout_groups": int(len(heldout_groups)),
        },
    )


def rule_i3(pipeline: SuspectedPipeline, config: RuleConfig) -> tuple[SuspectedPipeline, StructuralChange]:
    if pipeline.time is None:
        return pipeline, _change(
            RULE_I3,
            pipeline,
            pipeline,
            status=NOT_APPLICABLE,
            removed_rows=0,
            removed_features=[],
            detail={"reason": "no declared temporal metadata"},
        )
    keys = time_order_key(pipeline.time.reset_index(drop=True)).to_numpy(dtype=float)
    heldout = pipeline.heldout_idx
    if len(heldout) == 0:
        raise RuleError("I3 needs at least one held-out row")
    cutoff = float(np.nanmin(keys[heldout]))
    remove = np.asarray([index for index in pipeline.train_idx if keys[int(index)] >= cutoff], dtype=int)
    if len(remove) == len(pipeline.train_idx):
        # On a non-temporal split, chronology can place every training row at
        # or after the first held-out key. There is no estimable corrected
        # pipeline in that case, so preserve the episode and record the rule as
        # structurally inapplicable rather than aborting all other rules.
        return pipeline, _change(
            RULE_I3,
            pipeline,
            pipeline,
            status=NOT_APPLICABLE,
            removed_rows=0,
            removed_features=[],
            detail={
                "reason": "chronological restoration would remove every training row",
                "earliest_heldout_time_key": cutoff,
                "training_rows_at_or_after_cutoff": int(len(remove)),
            },
        )
    after = _remove_training_rows(pipeline, remove)
    return after, _change(
        RULE_I3,
        pipeline,
        after,
        status=APPLIED if len(remove) else NO_CHANGE,
        removed_rows=len(remove),
        removed_features=[],
        detail={"earliest_heldout_time_key": cutoff, "training_rows_at_or_after_cutoff": int(len(remove))},
    )


def rule_i4(pipeline: SuspectedPipeline, config: RuleConfig) -> tuple[SuspectedPipeline, StructuralChange]:
    before_scope = set(int(item) for item in pipeline.preprocessor_fit_idx)
    train_scope = set(int(item) for item in pipeline.train_idx)
    after = pipeline.with_preprocessor_scope(pipeline.train_idx)
    changed = before_scope != train_scope
    return after, _change(
        RULE_I4,
        pipeline,
        after,
        status=APPLIED if changed else NO_CHANGE,
        removed_rows=0,
        removed_features=[],
        detail={"non_training_rows_in_previous_scope": int(len(before_scope - train_scope))},
    )


def _binned(values: pd.Series, n_bins: int) -> np.ndarray:
    """Discretize one predictor on the rows supplied. Missing values form their own level."""

    if pd.api.types.is_numeric_dtype(values.dtype) and not pd.api.types.is_bool_dtype(values.dtype):
        numeric = pd.to_numeric(values, errors="coerce")
        finite = numeric.dropna()
        if finite.nunique() <= n_bins:
            codes = numeric.astype("object").where(numeric.notna(), "missing").astype(str)
            return codes.to_numpy()
        try:
            bins = pd.qcut(finite, q=n_bins, duplicates="drop", labels=False)
        except ValueError:
            bins = pd.Series(np.zeros(len(finite), dtype=int), index=finite.index)
        codes = pd.Series("missing", index=values.index, dtype=object)
        codes.loc[bins.index] = bins.astype(int).astype(str)
        return codes.to_numpy()
    return values.astype("object").where(values.notna(), "missing").astype(str).to_numpy()


def training_target_association(pipeline: SuspectedPipeline, n_bins: int) -> np.ndarray:
    """Normalized mutual information between each predictor and the target on training rows only."""

    rows = pipeline.train_idx
    y = pipeline.target.iloc[rows].astype(str).to_numpy()
    scores = np.zeros(pipeline.n_features, dtype=float)
    for position in range(pipeline.n_features):
        column = pipeline.features.iloc[rows, position].reset_index(drop=True)
        codes = _binned(column, n_bins)
        if len(set(codes.tolist())) < 2 or len(set(y.tolist())) < 2:
            scores[position] = 0.0
            continue
        scores[position] = float(normalized_mutual_info_score(y, codes))
    return scores


def availability_metadata_baseline(pipeline: SuspectedPipeline) -> dict[str, Any]:
    """List predictors the task marks unavailable. Association is not consulted.

    This is a provenance disclosure. It is not a behavioral leakage decision.
    """

    if "feature_available_at_prediction" not in pipeline.provenance:
        return {"status": "unsupported", "reason": "availability metadata absent", "unavailable_positions": []}
    availability = pipeline.provenance["feature_available_at_prediction"]
    if not isinstance(availability, (list, tuple)) or len(availability) != pipeline.n_features:
        return {"status": "unsupported", "reason": "availability metadata is malformed", "unavailable_positions": []}
    if any(type(item) is not bool for item in availability):
        return {"status": "unsupported", "reason": "availability metadata is not boolean", "unavailable_positions": []}
    unavailable = [position for position, available in enumerate(availability) if available is False]
    return {"status": "disclosed", "reason": "task metadata", "unavailable_positions": unavailable}


def _availability_screen(pipeline: SuspectedPipeline, policy: str) -> tuple[str | None, set[int], dict[str, Any]]:
    """Return an unsupported reason, the eligible positions, and a detail fragment."""

    detail: dict[str, Any] = {"availability_policy": policy}
    if policy == "legacy":
        availability = pipeline.provenance.get("feature_available_at_prediction")
        detail["availability_metadata"] = availability is not None
        if availability is None:
            return None, set(range(pipeline.n_features)), detail
        flags = [bool(item) for item in availability]
        if len(flags) != pipeline.n_features:
            raise RuleError("feature_available_at_prediction length does not match the predictors")
        return None, {position for position, available in enumerate(flags) if not available}, detail
    if "feature_available_at_prediction" not in pipeline.provenance:
        detail["reason"] = "availability metadata absent; association is not evidence that a predictor is unavailable"
        return detail["reason"], set(), detail
    availability = pipeline.provenance["feature_available_at_prediction"]
    malformed = (
        not isinstance(availability, (list, tuple))
        or len(availability) != pipeline.n_features
        or any(type(item) is not bool for item in availability)
    )
    if malformed:
        detail["reason"] = "availability metadata is missing or not a boolean per predictor"
        return detail["reason"], set(), detail
    eligible = {position for position, available in enumerate(availability) if available is False}
    detail["availability_metadata"] = True
    detail["unavailable_positions"] = sorted(eligible)
    return None, eligible, detail


def rule_i5(pipeline: SuspectedPipeline, config: RuleConfig) -> tuple[SuspectedPipeline, StructuralChange]:
    if config.proxy_screen_max_target_levels is not None:
        levels = int(pipeline.target.iloc[pipeline.train_idx].nunique(dropna=False))
        if levels > int(config.proxy_screen_max_target_levels):
            return pipeline, _change(
                RULE_I5,
                pipeline,
                pipeline,
                status=NOT_APPLICABLE,
                removed_rows=0,
                removed_features=[],
                detail={
                    "reason": "training target has more distinct values than the screen allows",
                    "target_levels": levels,
                    "max_target_levels": int(config.proxy_screen_max_target_levels),
                    "availability_policy": config.proxy_availability_policy,
                    "continuous_target_coverage": "unsupported",
                },
            )
    reason, eligible, availability_detail = _availability_screen(pipeline, config.proxy_availability_policy)
    if reason is not None:
        return pipeline, _change(
            RULE_I5,
            pipeline,
            pipeline,
            status=NOT_APPLICABLE,
            removed_rows=0,
            removed_features=[],
            detail=availability_detail,
        )
    if config.proxy_availability_policy == "require_boolean_v1" and not eligible:
        availability_detail["reason"] = "no predictor is marked unavailable"
        availability_detail["not_a_clean_finding"] = True
        return pipeline, _change(
            RULE_I5,
            pipeline,
            pipeline,
            status=NO_CHANGE,
            removed_rows=0,
            removed_features=[],
            detail=availability_detail,
        )
    scores = training_target_association(pipeline, config.proxy_screen_bins)
    flagged = [
        int(position)
        for position in np.flatnonzero(scores >= config.proxy_screen_threshold)
        if int(position) in eligible
    ]
    if len(flagged) >= pipeline.n_features:
        # Refusing to remove every predictor is a structural outcome, not a clean finding.
        return pipeline, _change(
            RULE_I5,
            pipeline,
            pipeline,
            status=NOT_APPLICABLE,
            removed_rows=0,
            removed_features=[],
            detail={
                "reason": "every predictor exceeded the screen",
                "association_scores": scores.tolist(),
                "not_a_clean_finding": True,
                **availability_detail,
            },
        )
    after = pipeline.without_feature_positions(flagged) if flagged else pipeline
    return after, _change(
        RULE_I5,
        pipeline,
        after,
        status=APPLIED if flagged else NO_CHANGE,
        removed_rows=0,
        removed_features=flagged,
        detail={
            "threshold": config.proxy_screen_threshold,
            "bins": config.proxy_screen_bins,
            "association_scores": scores.tolist(),
            "max_association": float(scores.max()) if scores.size else 0.0,
            **availability_detail,
        },
    )


def _placebo_rng(pipeline: SuspectedPipeline, config: RuleConfig, salt: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(int(pipeline.model_seed) + int(config.placebo_seed_offset) + int(salt)))


def rule_p1(pipeline: SuspectedPipeline, config: RuleConfig) -> tuple[SuspectedPipeline, StructuralChange]:
    _, i1_change = rule_i1(pipeline, config)
    matched = int(i1_change.rows_removed_from_training)
    n_remove = matched if matched > 0 else int(np.floor(config.placebo_row_fraction * len(pipeline.train_idx)))
    n_remove = max(1, min(n_remove, len(pipeline.train_idx) - 1))
    rng = _placebo_rng(pipeline, config, salt=1)
    remove = rng.choice(pipeline.train_idx, size=n_remove, replace=False)
    after = _remove_training_rows(pipeline, remove)
    return after, _change(
        RULE_P1,
        pipeline,
        after,
        status=APPLIED,
        removed_rows=n_remove,
        removed_features=[],
        detail={"matched_to_i1": matched > 0, "matched_row_count": matched, "fraction_used": None if matched > 0 else config.placebo_row_fraction},
    )


def rule_p5(pipeline: SuspectedPipeline, config: RuleConfig) -> tuple[SuspectedPipeline, StructuralChange]:
    if pipeline.n_features < 2:
        return pipeline, _change(
            RULE_P5,
            pipeline,
            pipeline,
            status=NOT_APPLICABLE,
            removed_rows=0,
            removed_features=[],
            detail={"reason": "a single predictor cannot be dropped"},
        )
    rng = _placebo_rng(pipeline, config, salt=5)
    position = int(rng.integers(0, pipeline.n_features))
    after = pipeline.without_feature_positions([position])
    return after, _change(
        RULE_P5,
        pipeline,
        after,
        status=APPLIED,
        removed_rows=0,
        removed_features=[position],
        detail={},
    )


RULES: dict[str, Callable[[SuspectedPipeline, RuleConfig], tuple[SuspectedPipeline, StructuralChange]]] = {
    RULE_I1: rule_i1,
    RULE_I2: rule_i2,
    RULE_I3: rule_i3,
    RULE_I4: rule_i4,
    RULE_I5: rule_i5,
    RULE_P1: rule_p1,
    RULE_P5: rule_p5,
}


def apply_rule(pipeline: SuspectedPipeline, rule_id: str, config: RuleConfig) -> tuple[SuspectedPipeline, StructuralChange]:
    if rule_id not in RULES:
        raise RuleError(f"Unknown intervention rule {rule_id}")
    return RULES[rule_id](pipeline, config)


__all__ = [
    "ALL_RULES",
    "APPLIED",
    "CORRECTIVE_RULES",
    "NOT_APPLICABLE",
    "NO_CHANGE",
    "PLACEBO_RULES",
    "RULE_NAMES",
    "RuleConfig",
    "RuleError",
    "StructuralChange",
    "apply_rule",
    "training_target_association",
]
