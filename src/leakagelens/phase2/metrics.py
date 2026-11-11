"""Classification, probability, and calibration metrics for clean references.

Brier score, both binary and multiclass, is

    (1 / N) * sum_i sum_k (p_ik - y_ik)^2

with one-hot labels. It is not divided by the number of classes, and it is not
sklearn's binary ``brier_score_loss`` (that function scores only the positive class).

Expected calibration error uses 15 equal-width bins of the predicted confidence,
where confidence is the maximum class probability and correctness is whether the
predicted class matches the label. Empty bins are omitted.

    ECE = sum_b (n_b / N) * |accuracy_b - confidence_b|

Multiclass AUROC is the macro average of one-vs-rest ROC AUC over classes that
are present in the evaluated partition. Multiclass AUPRC is the macro average
of one-vs-rest average precision over those same classes. A metric that is
statistically undefined is stored as null with a reason. It is not replaced by zero.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from leakagelens.fingerprints.schema import METRIC_SPECS, oriented_delta

N_ECE_BINS = 15
PROBABILITY_ATOL = 1e-5


class MetricError(ValueError):
    pass


def validate_probabilities(probabilities: np.ndarray, *, n_rows: int) -> None:
    array = np.asarray(probabilities, dtype=float)
    if array.ndim != 2 or array.shape[0] != n_rows or array.shape[1] < 2:
        raise MetricError(f"Expected probability matrix {(n_rows, 'K>=2')}, got {array.shape}")
    if not np.isfinite(array).all():
        raise MetricError("Probabilities contain non-finite values")
    if (array < -PROBABILITY_ATOL).any() or (array > 1.0 + PROBABILITY_ATOL).any():
        raise MetricError("Probabilities fall outside [0, 1]")
    totals = array.sum(axis=1)
    if not np.allclose(totals, 1.0, atol=1e-4):
        raise MetricError("Probability rows do not sum to 1")


def brier_score(y_true: np.ndarray, probabilities: np.ndarray, classes: list[Any]) -> float:
    """Multiclass Brier score without dividing by the class count."""

    y = np.asarray(y_true)
    proba = np.asarray(probabilities, dtype=float)
    class_ids = [str(item) for item in classes]
    index = {label: position for position, label in enumerate(class_ids)}
    one_hot = np.zeros_like(proba)
    for row, label in enumerate(y.astype(str)):
        one_hot[row, index[label]] = 1.0
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = N_ECE_BINS,
) -> dict[str, Any]:
    """Equal-width confidence ECE. Confidence is the maximum class probability."""

    confidence = np.asarray(probabilities, dtype=float).max(axis=1)
    correct = (np.asarray(y_pred).astype(str) == np.asarray(y_true).astype(str)).astype(float)
    # Bin 0 is [0, 1/n), and confidence 1 lands in the last bin.
    bin_ids = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    total = len(confidence)
    weighted = 0.0
    occupied = 0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        count = int(mask.sum())
        if count == 0:
            continue
        occupied += 1
        weighted += (count / total) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return {
        "value": float(weighted),
        "n_bins": n_bins,
        "binning": "equal_width",
        "confidence": "max_predicted_class_probability",
        "correctness": "predicted_class_equals_true_class",
        "occupied_bins": occupied,
    }


def _defined(value: float) -> dict[str, Any]:
    return {"value": float(value), "status": "defined", "reason": None}


def _undefined(reason: str) -> dict[str, Any]:
    return {"value": None, "status": "undefined", "reason": reason}


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: list[Any],
) -> dict[str, Any]:
    """Score one partition. ``classes`` is the estimator class order."""

    y = np.asarray(y_true)
    proba = np.asarray(probabilities, dtype=float)
    validate_probabilities(proba, n_rows=len(y))
    class_list = list(classes)
    if proba.shape[1] != len(class_list):
        raise MetricError("Probability columns do not match class order")
    pred_index = np.argmax(proba, axis=1)
    y_pred = np.asarray([class_list[index] for index in pred_index])
    y_text = y.astype(str)
    pred_text = y_pred.astype(str)
    class_text = [str(item) for item in class_list]
    present = [label for label in class_text if (y_text == label).any()]

    metrics: dict[str, Any] = {
        "accuracy": _defined(accuracy_score(y_text, pred_text)),
        "balanced_accuracy": _defined(balanced_accuracy_score(y_text, pred_text)),
        "macro_f1": _defined(f1_score(y_text, pred_text, average="macro", zero_division=0, labels=class_text)),
        "log_loss": _defined(log_loss(y_text, proba, labels=class_text)),
        "brier": _defined(brier_score(y_text, proba, class_text)),
        "mean_confidence": _defined(float(proba.max(axis=1).mean())),
    }
    ece = expected_calibration_error(y_text, pred_text, proba)
    metrics["ece"] = _defined(ece["value"])
    metrics["ece_definition"] = {key: value for key, value in ece.items() if key != "value"}

    if len(present) < 2:
        metrics["auroc"] = _undefined("fewer_than_two_classes_in_partition")
        metrics["auprc"] = _undefined("fewer_than_two_classes_in_partition")
    elif len(class_text) == 2:
        positive = class_text[1]
        scores = proba[:, 1]
        binary = (y_text == positive).astype(int)
        metrics["auroc"] = _defined(roc_auc_score(binary, scores))
        metrics["auprc"] = _defined(average_precision_score(binary, scores))
    else:
        present_index = [class_text.index(label) for label in present]
        y_bin = label_binarize(y_text, classes=class_text)
        if y_bin.ndim == 1:
            y_bin = np.column_stack([1 - y_bin, y_bin])
        aurocs = []
        auprcs = []
        for index in present_index:
            aurocs.append(roc_auc_score(y_bin[:, index], proba[:, index]))
            auprcs.append(average_precision_score(y_bin[:, index], proba[:, index]))
        metrics["auroc"] = _defined(float(np.mean(aurocs)))
        metrics["auprc"] = _defined(float(np.mean(auprcs)))
        metrics["auroc_definition"] = "macro_one_vs_rest_over_classes_present_in_partition"
        metrics["auprc_definition"] = "macro_one_vs_rest_average_precision_over_classes_present_in_partition"

    metrics["predicted_class_count"] = int(len(set(pred_text.tolist())))
    metrics["class_order"] = class_text
    return metrics


def generalization_gaps(partition_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Raw train-test and validation-test gaps plus metric orientation."""

    pairs = (("train", "test"), ("validation", "test"))
    names = (
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
    gaps: dict[str, Any] = {}
    for left, right in pairs:
        block: dict[str, Any] = {}
        for name in names:
            left_metric = partition_metrics[left].get(name, {})
            right_metric = partition_metrics[right].get(name, {})
            if left_metric.get("status") != "defined" or right_metric.get("status") != "defined":
                block[name] = {"raw_gap": None, "orientation": None, "oriented_gap": None, "status": "undefined"}
                continue
            raw = float(left_metric["value"]) - float(right_metric["value"])
            spec = METRIC_SPECS.get(name)
            orientation = None if spec is None else spec["orientation"]
            oriented = None if orientation is None else oriented_delta(
                float(left_metric["value"]), float(right_metric["value"]), orientation
            )
            block[name] = {
                "raw_gap": raw,
                "orientation": orientation,
                "oriented_gap": oriented,
                "status": "defined",
            }
        gaps[f"{left}_minus_{right}"] = block
    return gaps
