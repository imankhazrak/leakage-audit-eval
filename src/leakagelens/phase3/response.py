"""Behavioral response quantities. Structural leakage counts are not included.

Phase 3 defines a positive oriented delta as an improvement after the corrective
intervention. That is the negative of the Phase-0 suspected-minus-corrected
oriented delta. The Phase-0 function is unchanged. This module applies the
Phase-3 definition explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from leakagelens.fingerprints.schema import METRIC_SPECS
from leakagelens.constants import MetricOrientation

PROBABILITY_EPS = 1e-12
NEAR_ZERO = 1e-8


def phase3_raw_delta(before: float, after: float) -> float:
    """Return after - before."""

    return float(after) - float(before)


def phase3_oriented_delta(before: float, after: float, orientation: str) -> float | None:
    """Return the intervention improvement, or None when the metric is unsigned."""

    if orientation == MetricOrientation.UNSIGNED.value:
        return None
    if orientation == MetricOrientation.HIGHER_IS_BETTER.value:
        return float(after) - float(before)
    if orientation == MetricOrientation.LOWER_IS_BETTER.value:
        return float(before) - float(after)
    raise ValueError(f"Unknown orientation {orientation}")


def metric_response(before: float | None, after: float | None, metric: str) -> dict[str, float | None]:
    if before is None or after is None or not np.isfinite(before) or not np.isfinite(after):
        return {"before": before, "after": after, "raw_delta": None, "oriented_delta": None}
    orientation = METRIC_SPECS[metric]["orientation"]
    return {
        "before": float(before),
        "after": float(after),
        "raw_delta": phase3_raw_delta(float(before), float(after)),
        "oriented_delta": phase3_oriented_delta(float(before), float(after), orientation),
    }


def jensen_shannon(left: np.ndarray, right: np.ndarray, eps: float = PROBABILITY_EPS) -> np.ndarray:
    """Row-wise Jensen-Shannon divergence in nats. Rows are renormalized."""

    p = np.asarray(left, dtype=float)
    q = np.asarray(right, dtype=float)
    if p.shape != q.shape or p.ndim != 2:
        raise ValueError(f"Probability matrices must share a shape, got {p.shape} and {q.shape}")
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    mean = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / mean), axis=1) + 0.5 * np.sum(q * np.log(q / mean), axis=1)


def prediction_response(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, float]:
    """Compare two prediction frames on the same row_index values."""

    left = before.set_index("row_index").sort_index()
    right = after.set_index("row_index").sort_index()
    if not left.index.equals(right.index):
        raise ValueError("Prediction frames do not share an evaluation anchor")
    proba_columns = [column for column in left.columns if str(column).startswith("proba_")]
    if proba_columns != [column for column in right.columns if str(column).startswith("proba_")]:
        raise ValueError("Probability columns differ")
    p = left[proba_columns].to_numpy(dtype=float)
    q = right[proba_columns].to_numpy(dtype=float)
    absolute = np.abs(p - q)
    js = jensen_shannon(p, q)
    disagreement = float((left["y_pred"].astype(str).to_numpy() != right["y_pred"].astype(str).to_numpy()).mean())
    confidence_before = p.max(axis=1)
    confidence_after = q.max(axis=1)
    return {
        "predicted_class_disagreement_rate": disagreement,
        "mean_absolute_probability_change": float(absolute.mean()),
        "median_absolute_probability_change": float(np.median(absolute)),
        "maximum_absolute_probability_change": float(absolute.max()) if absolute.size else 0.0,
        "mean_confidence_change": float((confidence_after - confidence_before).mean()),
        "mean_jensen_shannon": float(js.mean()),
        "median_jensen_shannon": float(np.median(js)),
    }


def importance_response(before: dict[str, float], after: dict[str, float], *, top_k: int = 10) -> dict[str, float | None]:
    """Compare original-variable importance maps. Missing names contribute 0."""

    names = sorted(set(before).union(after))
    if not names:
        return {
            "importance_l1": None,
            "importance_norm_change": None,
            "topk_overlap": None,
            "rank_correlation": None,
        }
    left = np.asarray([float(before.get(name, 0.0)) for name in names], dtype=float)
    right = np.asarray([float(after.get(name, 0.0)) for name in names], dtype=float)
    left_scale = np.abs(left).sum()
    right_scale = np.abs(right).sum()
    left_norm = left / left_scale if left_scale else left
    right_norm = right / right_scale if right_scale else right
    top_before = set(np.argsort(-np.abs(left))[:top_k])
    top_after = set(np.argsort(-np.abs(right))[:top_k])
    overlap = len(top_before & top_after) / float(top_k)
    if np.allclose(left, left[0]) or np.allclose(right, right[0]):
        correlation = None
    else:
        correlation = float(spearmanr(left, right).statistic)
    return {
        "importance_l1": float(np.abs(left_norm - right_norm).sum()),
        "importance_norm_change": float(np.linalg.norm(right) - np.linalg.norm(left)),
        "topk_overlap": float(overlap),
        "rank_correlation": correlation,
    }


def response_status(values: list[float | None]) -> str:
    observed = [value for value in values if value is not None and np.isfinite(value)]
    if observed and max(abs(value) for value in observed) > NEAR_ZERO:
        return "observed"
    return "zero_or_near_zero"
