"""Execute label-blind interventions by refitting the suspected pipeline.

The suspected pipeline is fit once ("before"). Each rule produces a corrected
pipeline that is fit again ("after"). Responses are computed from the two
executed fits with the Phase-3 response functions, so the resulting columns
line up with the Phase-3 behavioral schema. No saved clean reference is
consulted and no rule outcome is forced to equal anything.

Model fitting is refused outside a SLURM job unless ``allow_local`` is set,
which exists for tiny synthetic tests only.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leakagelens.interventions.pipeline import SuspectedPipeline
from leakagelens.interventions.rules import (
    ALL_RULES,
    NOT_APPLICABLE,
    RuleConfig,
    apply_rule,
)
from leakagelens.logging_utils import get_logger
from leakagelens.phase2.jsonio import dump_json
from leakagelens.phase2.metrics import classification_metrics
from leakagelens.phase2.models import build_estimator
from leakagelens.phase2.preprocessing import build_preprocessor, transformed_feature_map
from leakagelens.phase2.run_task import _original_importance, _prediction_frame
from leakagelens.phase3.constants import CORE_METRICS, GAP_METRICS, GAP_PAIRS, PARTITIONS
from leakagelens.phase3.response import importance_response, metric_response, prediction_response, response_status
from leakagelens.utils.seeding import seed_everything

LOGGER = get_logger(__name__)

STATUS_EXECUTED = "executed"
STATUS_NOT_APPLICABLE = NOT_APPLICABLE
STATUS_FAILED = "failed"


class FitGuardError(RuntimeError):
    pass


def require_slurm(*, allow_local: bool = False) -> None:
    if allow_local:
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise FitGuardError("Label-blind intervention fitting must run inside a SLURM job")


@dataclass
class FittedSuspect:
    classes: list[str]
    importance: dict[str, float]
    predictions: dict[str, pd.DataFrame]
    metrics: dict[str, dict[str, Any]]
    fit_seconds: float
    structure: dict[str, int] = field(default_factory=dict)


def _importance(preprocessor: Any, estimator: Any, family: str) -> dict[str, float]:
    mapping = transformed_feature_map(preprocessor)
    if family == "logistic_regression":
        values = np.mean(np.abs(estimator.coef_), axis=0)
    else:
        values = np.asarray(estimator.feature_importances_, dtype=float)
    return _original_importance(mapping, values)


def _align(frame: pd.DataFrame, source_classes: list[str], target_classes: list[str]) -> pd.DataFrame:
    if source_classes == target_classes:
        return frame
    aligned = frame.copy()
    source = frame[[column for column in frame.columns if str(column).startswith("proba_")]].to_numpy(dtype=float)
    output = np.zeros((len(frame), len(target_classes)), dtype=float)
    index = {label: position for position, label in enumerate(source_classes)}
    for position, label in enumerate(target_classes):
        if label in index:
            output[:, position] = source[:, index[label]]
    for position in range(len(target_classes)):
        aligned[f"proba_{position}"] = output[:, position]
    extra = [column for column in aligned.columns if str(column).startswith("proba_") and int(str(column).split("_")[1]) >= len(target_classes)]
    aligned = aligned.drop(columns=extra)
    aligned["y_pred"] = [target_classes[int(item)] for item in np.argmax(output, axis=1)]
    return aligned


def fit_suspected(
    pipeline: SuspectedPipeline,
    *,
    episode_id: str,
    class_order: list[str] | None = None,
    allow_local: bool = False,
) -> FittedSuspect:
    """Fit the pipeline exactly as described and score its own partitions."""

    require_slurm(allow_local=allow_local)
    started = time.perf_counter()
    seed_everything(pipeline.model_seed)
    preprocessor, _name = build_preprocessor(pipeline.model_family, pipeline.features)
    estimator = build_estimator(pipeline.model_family, pipeline.model_backend, dict(pipeline.model_params), pipeline.model_seed)
    preprocessor.fit(pipeline.features.iloc[pipeline.preprocessor_fit_idx])
    estimator.fit(
        preprocessor.transform(pipeline.features.iloc[pipeline.train_idx]),
        pipeline.target.iloc[pipeline.train_idx],
    )
    classes = [str(item) for item in estimator.classes_]
    order = class_order or classes
    predictions: dict[str, pd.DataFrame] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for partition, indices in (
        ("train", pipeline.train_idx),
        ("validation", pipeline.validation_idx),
        ("test", pipeline.test_idx),
    ):
        if len(indices) == 0:
            raise FitGuardError(f"{partition} partition is empty")
        probabilities = estimator.predict_proba(preprocessor.transform(pipeline.features.iloc[indices]))
        frame = _prediction_frame(indices, pipeline.target, probabilities, classes, partition=partition, experiment_id=episode_id)
        frame = _align(frame, classes, order)
        predictions[partition] = frame
        metrics[partition] = classification_metrics(
            frame["y_true"].to_numpy(),
            frame[[column for column in frame.columns if str(column).startswith("proba_")]].to_numpy(dtype=float),
            order,
        )
    return FittedSuspect(
        classes=order,
        importance=_importance(preprocessor, estimator, pipeline.model_family),
        predictions=predictions,
        metrics=metrics,
        fit_seconds=float(time.perf_counter() - started),
        structure=pipeline.structural_summary(),
    )


def _metric_value(metrics: dict[str, Any], name: str) -> float | None:
    record = metrics.get(name, {})
    if record.get("status") != "defined":
        return None
    return float(record["value"])


def _metrics_on(frame: pd.DataFrame, class_order: list[str]) -> dict[str, Any]:
    proba = [column for column in frame.columns if str(column).startswith("proba_")]
    return classification_metrics(frame["y_true"].to_numpy(), frame[proba].to_numpy(dtype=float), class_order)


def _restrict(frame: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    wanted = set(int(item) for item in rows)
    return frame.loc[frame["row_index"].astype(int).isin(wanted)].copy()


def response_columns(rule_ids: tuple[str, ...] | list[str]) -> list[str]:
    from leakagelens.phase3.schema import behavioral_column_names

    return behavioral_column_names(interventions=tuple(rule_ids))


def _gap(wide: dict[str, Any], rule_id: str, left: str, right: str, metric: str, side: str) -> float | None:
    left_value = wide.get(f"{rule_id}__{left}__{metric}__{side}")
    right_value = wide.get(f"{rule_id}__{right}__{metric}__{side}")
    if left_value is None or right_value is None:
        return None
    return float(left_value) - float(right_value)


def fill_rule_response(
    wide: dict[str, Any],
    long_rows: list[dict[str, Any]],
    oriented_values: list[float | None],
    *,
    rule_id: str,
    before: FittedSuspect,
    after: FittedSuspect,
    before_pipeline: SuspectedPipeline,
    after_pipeline: SuspectedPipeline,
) -> dict[str, Any]:
    """Compute before/after responses from two executed fits.

    The training anchor is the intersection of both training sets so both
    models are in-sample on it. Held-out anchors are the suspected pipeline's
    own validation and test partitions, which no rule alters.
    """

    train_anchor = np.asarray(sorted(set(before_pipeline.train_idx.tolist()) & set(after_pipeline.train_idx.tolist())), dtype=int)
    anchors = {"train": train_anchor, "validation": before_pipeline.validation_idx, "test": before_pipeline.test_idx}
    for partition in PARTITIONS:
        before_frame = _restrict(before.predictions[partition], anchors[partition])
        after_frame = _restrict(after.predictions[partition], anchors[partition])
        before_metrics = _metrics_on(before_frame, before.classes) if len(before_frame) else {}
        after_metrics = _metrics_on(after_frame, before.classes) if len(after_frame) else {}
        for metric in CORE_METRICS:
            record = metric_response(_metric_value(before_metrics, metric), _metric_value(after_metrics, metric), metric)
            for suffix, value in record.items():
                wide[f"{rule_id}__{partition}__{metric}__{suffix}"] = value
            long_rows.append({"intervention_id": rule_id, "partition": partition, "metric_or_behavior": metric, **record})
            oriented_values.append(record["oriented_delta"])
    for left, right in GAP_PAIRS:
        for metric in GAP_METRICS:
            before_gap = _gap(wide, rule_id, left, right, metric, "before")
            after_gap = _gap(wide, rule_id, left, right, metric, "after")
            raw = None if before_gap is None or after_gap is None else float(after_gap) - float(before_gap)
            prefix = f"{rule_id}__{left}_minus_{right}__{metric}"
            wide[f"{prefix}__gap_before"] = before_gap
            wide[f"{prefix}__gap_after"] = after_gap
            wide[f"{prefix}__gap_raw_delta"] = raw
            wide[f"{prefix}__gap_oriented_delta"] = None
    for partition in ("validation", "test"):
        summary = prediction_response(before.predictions[partition], after.predictions[partition])
        for name, value in summary.items():
            wide[f"{rule_id}__{partition}__{name}"] = value
            oriented_values.append(value if "disagreement" in name or "jensen" in name or "probability" in name else None)
    summary = importance_response(before.importance, after.importance)
    for name, value in summary.items():
        wide[f"{rule_id}__{name}"] = value
    return {"train_anchor_rows": int(len(train_anchor))}


@dataclass
class LabelBlindResult:
    episode_id: str
    wide: dict[str, Any]
    long: list[dict[str, Any]]
    structural: dict[str, dict[str, Any]]
    rule_status: dict[str, dict[str, Any]]
    behavioral_response_status: str
    fits_executed: int
    fit_seconds: dict[str, float]


def _write_fit(destination: Path, fitted: FittedSuspect) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    fitted.predictions["validation"].to_parquet(destination / "predictions_validation.parquet", index=False)
    fitted.predictions["test"].to_parquet(destination / "predictions_test.parquet", index=False)
    dump_json(destination / "metrics.json", fitted.metrics)
    dump_json(
        destination / "fit_summary.json",
        {"classes": fitted.classes, "importance": fitted.importance, "fit_seconds": fitted.fit_seconds, "structure": fitted.structure},
    )


def execute_label_blind(
    pipeline: SuspectedPipeline,
    *,
    episode_id: str,
    config: RuleConfig,
    rules: tuple[str, ...] | list[str] = ALL_RULES,
    destination: Path | None = None,
    allow_local: bool = False,
) -> LabelBlindResult:
    """Fit the suspected pipeline, apply every rule, refit, and record responses."""

    require_slurm(allow_local=allow_local)
    rules = tuple(rules)
    before = fit_suspected(pipeline, episode_id=episode_id, allow_local=allow_local)
    fits = 1
    seconds = {"suspected": before.fit_seconds}
    if destination is not None:
        _write_fit(destination / "suspected", before)
    wide: dict[str, Any] = {name: None for name in response_columns(rules)}
    long_rows: list[dict[str, Any]] = []
    oriented: list[float | None] = []
    structural: dict[str, dict[str, Any]] = {}
    status: dict[str, dict[str, Any]] = {}
    for rule_id in rules:
        corrected, change = apply_rule(pipeline, rule_id, config)
        structural[rule_id] = change.to_record()
        if change.status == NOT_APPLICABLE:
            status[rule_id] = {"status": STATUS_NOT_APPLICABLE, "reason": change.detail.get("reason")}
            if destination is not None:
                folder = destination / rule_id
                folder.mkdir(parents=True, exist_ok=True)
                dump_json(folder / "structural_change.json", change.to_record())
            continue
        after = fit_suspected(corrected, episode_id=episode_id, class_order=before.classes, allow_local=allow_local)
        fits += 1
        seconds[rule_id] = after.fit_seconds
        anchor_info = fill_rule_response(
            wide,
            long_rows,
            oriented,
            rule_id=rule_id,
            before=before,
            after=after,
            before_pipeline=pipeline,
            after_pipeline=corrected,
        )
        status[rule_id] = {"status": STATUS_EXECUTED, "structural_status": change.status, **anchor_info}
        if destination is not None:
            folder = destination / rule_id
            _write_fit(folder, after)
            dump_json(folder / "structural_change.json", change.to_record())
        LOGGER.info("episode=%s rule=%s status=%s", episode_id, rule_id, change.status)
    result = LabelBlindResult(
        episode_id=episode_id,
        wide=wide,
        long=long_rows,
        structural=structural,
        rule_status=status,
        behavioral_response_status=response_status(oriented),
        fits_executed=fits,
        fit_seconds=seconds,
    )
    if destination is not None:
        dump_json(destination / "fingerprint.json", {"wide": wide, "long": long_rows, "behavioral_response_status": result.behavioral_response_status})
        dump_json(destination / "structural_changes.json", structural)
        dump_json(destination / "rule_status.json", status)
        dump_json(destination / "fit_seconds.json", seconds)
    return result


__all__ = [
    "FitGuardError",
    "FittedSuspect",
    "LabelBlindResult",
    "STATUS_EXECUTED",
    "STATUS_FAILED",
    "STATUS_NOT_APPLICABLE",
    "execute_label_blind",
    "fill_rule_response",
    "fit_suspected",
    "require_slurm",
    "response_columns",
]
