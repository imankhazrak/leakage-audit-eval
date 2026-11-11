"""Fail-closed checks that a clean split and predictor matrix do not leak."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from leakagelens.phase2.protocols import PROTOCOL_COMBINED, PROTOCOL_GROUP, PROTOCOL_STRATIFIED, PROTOCOL_TEMPORAL
from leakagelens.phase2.splits import CleanSplit, duplicate_cluster_ids, time_order_key

PREDICTOR_ROLE = "predictor"
BLOCKED_ROLES = {"row_identifier", "entity_identifier", "group_identifier", "temporal_metadata", "identifier"}


class CleanSafetyError(ValueError):
    pass


def _overlap(left: np.ndarray, right: np.ndarray) -> int:
    return len(set(left.tolist()) & set(right.tolist()))


def _cluster_crossings(cluster_ids: pd.Series, left: np.ndarray, right: np.ndarray) -> int:
    if len(left) == 0 or len(right) == 0:
        return 0
    return len(set(cluster_ids.iloc[left].astype(str)) & set(cluster_ids.iloc[right].astype(str)))


def validate_predictor_matrix(features: pd.DataFrame, roles: dict[str, str], *, target_name: str) -> dict[str, Any]:
    """Reject any non-predictor column that is physically present in X."""

    present = [str(column) for column in features.columns]
    if target_name in present:
        raise CleanSafetyError(f"Target column {target_name} is present in X")
    blocked = [column for column in present if roles.get(column) != PREDICTOR_ROLE]
    unknown = [column for column in present if column not in roles]
    if unknown:
        raise CleanSafetyError(f"Columns without a frozen role entered X: {unknown}")
    if blocked:
        raise CleanSafetyError(f"Non-predictor columns entered X: {blocked}")
    report = {
        "target_present_in_X": False,
        "identifier_metadata_present_in_X": False,
        "group_metadata_present_in_X": False,
        "temporal_metadata_present_in_X": False,
        "n_predictors": len(present),
        "predictor_columns": present,
    }
    return report


def validate_clean_split(
    split: CleanSplit,
    *,
    n_rows: int,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series | None = None,
    time: pd.Series | None = None,
) -> dict[str, Any]:
    """Check disjointness and the constraints required by the protocol."""

    if split.status != "success":
        raise CleanSafetyError(f"Split status is {split.status}; refusing to train")
    train, validation, test = split.train_idx, split.validation_idx, split.test_idx
    purged = np.asarray(split.purged_idx, dtype=int)
    row_overlap = {
        "train_validation": _overlap(train, validation),
        "train_test": _overlap(train, test),
        "validation_test": _overlap(validation, test),
        "purged_train": _overlap(purged, train),
        "purged_validation": _overlap(purged, validation),
        "purged_test": _overlap(purged, test),
    }
    union = set(train.tolist()) | set(validation.tolist()) | set(test.tolist()) | set(purged.tolist())
    problems: list[str] = []
    if any(row_overlap.values()):
        problems.append(f"row overlap {row_overlap}")
    if union != set(range(n_rows)) or len(union) != n_rows:
        problems.append(f"row coverage {len(union)} of {n_rows}")
    if min(len(train), len(validation), len(test)) <= 0:
        problems.append("empty partition")

    clusters = duplicate_cluster_ids(features)
    xy = features.copy()
    xy["__y__"] = target.astype(str).to_numpy()
    xy_clusters = duplicate_cluster_ids(xy)
    duplicate_cross = {
        "exact_X_train_test": _cluster_crossings(clusters, train, test),
        "exact_X_train_validation": _cluster_crossings(clusters, train, validation),
        "exact_X_validation_test": _cluster_crossings(clusters, validation, test),
        "exact_Xy_train_test": _cluster_crossings(xy_clusters, train, test),
    }
    if split.protocol in {PROTOCOL_STRATIFIED, PROTOCOL_GROUP} and any(
        duplicate_cross[key] for key in ("exact_X_train_test", "exact_X_train_validation", "exact_X_validation_test")
    ):
        problems.append(f"duplicate clusters cross partitions {duplicate_cross}")

    group_overlap = {"train_validation": 0, "train_test": 0, "validation_test": 0}
    if split.protocol in {PROTOCOL_GROUP, PROTOCOL_COMBINED}:
        if groups is None:
            problems.append("group metadata missing")
        else:
            labels = groups.astype(str).reset_index(drop=True)
            sets = {
                "train": set(labels.iloc[train]),
                "validation": set(labels.iloc[validation]),
                "test": set(labels.iloc[test]),
            }
            group_overlap = {
                "train_validation": len(sets["train"] & sets["validation"]),
                "train_test": len(sets["train"] & sets["test"]),
                "validation_test": len(sets["validation"] & sets["test"]),
            }
            if any(group_overlap.values()):
                problems.append(f"group overlap {group_overlap}")

    temporal_violations = 0
    if split.protocol in {PROTOCOL_TEMPORAL, PROTOCOL_COMBINED}:
        if time is None:
            problems.append("temporal metadata missing")
        else:
            keys = time_order_key(time.reset_index(drop=True))
            train_max = float(keys.iloc[train].max())
            validation_min = float(keys.iloc[validation].min())
            validation_max = float(keys.iloc[validation].max())
            test_min = float(keys.iloc[test].min())
            if train_max > validation_min or validation_max > test_min:
                temporal_violations = 1
                problems.append(
                    f"temporal order violated train_max={train_max} validation=({validation_min},{validation_max}) test_min={test_min}"
                )

    report = {
        "protocol": split.protocol,
        "status": "pass" if not problems else "fail",
        "row_overlap": row_overlap,
        "rows_accounted": len(union) == n_rows and not any(row_overlap.values()),
        "group_overlap": group_overlap,
        "temporal_violations": temporal_violations,
        "duplicate_overlap": duplicate_cross,
        "problems": problems,
    }
    if problems:
        raise CleanSafetyError("; ".join(problems))
    return report
