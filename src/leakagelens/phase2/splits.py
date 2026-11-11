"""Deterministic leakage-safe train/validation/test splits.

Duplicate policy
----------------
Exact predictor-row duplicates form one split unit for ordinary and group-safe
splits, including conflicting-label groups. Those rows are not deleted.

Group-safe splits use connected components. Two rows are linked when they share
an entity id or an exact-predictor duplicate cluster, and components are split
instead of rows.

Temporal splits keep whole ordered time units together. Chronology outranks
duplicate separation. Cross-boundary duplicates are reported and are not moved
backward in time.

A dataset with both supported group and temporal constraints is split only when
some chronological boundary also keeps every entity inside one partition.
Otherwise the result is ``infeasible_under_combined_constraints`` and must not
be trained. Purging crossing entities is a separate question, answered by
``purged_split`` and not by this all-row rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from leakagelens.phase2.protocols import (
    PROTOCOL_COMBINED,
    PROTOCOL_GROUP,
    PROTOCOL_STRATIFIED,
    PROTOCOL_TEMPORAL,
)

HASH_KEY = "leakagelens-p2v1"
TARGET_FRACTIONS = (0.70, 0.15, 0.15)
STATUS_SUCCESS = "success"
STATUS_INFEASIBLE = "infeasible_under_combined_constraints"


class SplitError(ValueError):
    pass


@dataclass
class CleanSplit:
    protocol: str
    split_seed: int
    seed_affects_membership: bool
    status: str
    train_idx: np.ndarray
    validation_idx: np.ndarray
    test_idx: np.ndarray
    details: dict[str, Any] = field(default_factory=dict)
    purged_idx: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=int))

    @property
    def feasible(self) -> bool:
        return self.status == STATUS_SUCCESS


def duplicate_cluster_ids(features: pd.DataFrame) -> pd.Series:
    """Stable ids for exact predictor-row duplicates.

    ``hash_key`` makes the digest independent of ``PYTHONHASHSEED``.
    """

    hashed = pd.util.hash_pandas_object(features, index=False, hash_key=HASH_KEY)
    return pd.Series(hashed.to_numpy(), index=features.index, name="duplicate_cluster_id")


def time_order_key(values: pd.Series) -> pd.Series:
    """Map a temporal column onto a sortable numeric key without shuffling."""

    if pd.api.types.is_numeric_dtype(values):
        return pd.Series(values.astype(float).to_numpy(), index=values.index)
    parsed = pd.to_datetime(values.astype(str), errors="coerce")
    if parsed.notna().all():
        return pd.Series(parsed.astype("int64").to_numpy(), index=values.index)
    ordered = sorted(values.astype(str).unique())
    mapping = {label: index for index, label in enumerate(ordered)}
    return values.astype(str).map(mapping).astype(float)


def _majority_label(labels: np.ndarray) -> str:
    text = np.asarray(labels).astype(str)
    unique, counts = np.unique(text, return_counts=True)
    order = sorted(zip(unique.tolist(), counts.tolist()), key=lambda item: (-item[1], item[0]))
    return str(order[0][0])


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size)
        self.rank = np.zeros(size, dtype=int)

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


def _component_ids(labels: list[pd.Series]) -> np.ndarray:
    """Connected components of rows that share any of the supplied labels."""

    size = len(labels[0])
    finder = _UnionFind(size)
    for series in labels:
        groups: dict[str, int] = {}
        for index, raw in enumerate(series.astype(str).tolist()):
            previous = groups.get(raw)
            if previous is None:
                groups[raw] = index
            else:
                finder.union(previous, index)
    return np.asarray([finder.find(index) for index in range(size)], dtype=int)


def _assign_units(unit_ids: np.ndarray, strata: np.ndarray, seed: int) -> tuple[list[int], list[int], list[int], list[str]]:
    """Split atomic units into train/validation/test, targeting 70/15/15."""

    notes: list[str] = []
    ids = [int(item) for item in unit_ids]
    labels = [str(item) for item in strata]
    if len(ids) < 3:
        raise SplitError(f"Need at least 3 split units, found {len(ids)}")

    forced_train: list[int] = []
    rest_ids: list[int] = []
    rest_labels: list[str] = []
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    for unit, label in zip(ids, labels):
        if counts[label] < 2:
            forced_train.append(unit)
        else:
            rest_ids.append(unit)
            rest_labels.append(label)
    if forced_train:
        notes.append("classes_with_a_single_unit_assigned_to_train")

    def _two_way(items: list[int], item_labels: list[str] | None, test_size: float, random_state: int) -> tuple[list[int], list[int]]:
        stratify = item_labels
        if stratify is not None:
            local_counts: dict[str, int] = {}
            for label in stratify:
                local_counts[label] = local_counts.get(label, 0) + 1
            if min(local_counts.values()) < 2:
                stratify = None
                notes.append("stratify_disabled_for_rare_remaining_units")
        try:
            left, right = train_test_split(
                items,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )
        except ValueError:
            notes.append("unstratified_fallback")
            left, right = train_test_split(items, test_size=test_size, random_state=random_state, stratify=None)
        return [int(item) for item in left], [int(item) for item in right]

    if len(rest_ids) < 2:
        raise SplitError("Not enough split units remain after placing rare classes in train")
    train_units, holdout = _two_way(rest_ids, rest_labels if rest_labels else None, 0.30, seed)
    holdout_labels = None
    if rest_labels:
        label_of = dict(zip(rest_ids, rest_labels))
        holdout_labels = [label_of[item] for item in holdout]
    if len(holdout) < 2:
        raise SplitError("Holdout does not contain both validation and test units")
    validation_units, test_units = _two_way(holdout, holdout_labels, 0.50, seed + 1)
    train_units = forced_train + train_units
    if not train_units or not validation_units or not test_units:
        raise SplitError("A partition received no split units")
    return train_units, validation_units, test_units, notes


def _rows_for_units(membership: np.ndarray, units: list[int]) -> np.ndarray:
    wanted = set(units)
    return np.flatnonzero(np.isin(membership, list(wanted))).astype(int)


def _chronological_assignment(order_key: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]] | None:
    """Prefix/middle/suffix cut of whole time units closest to 70/15/15."""

    keys = order_key.to_numpy()
    units = np.array(sorted(pd.unique(keys)))
    if len(units) < 3:
        return None
    counts = np.asarray([(keys == unit).sum() for unit in units], dtype=int)
    prefix = np.cumsum(counts)
    total = int(prefix[-1])
    best: tuple[float, float, int, int] | None = None
    for end_train in range(1, len(units) - 1):
        for end_validation in range(end_train + 1, len(units)):
            n_train = int(prefix[end_train - 1])
            n_validation = int(prefix[end_validation - 1] - n_train)
            n_test = total - int(prefix[end_validation - 1])
            if min(n_train, n_validation, n_test) <= 0:
                continue
            fractions = (n_train / total, n_validation / total, n_test / total)
            score = sum(abs(fractions[index] - TARGET_FRACTIONS[index]) for index in range(3))
            candidate = (score, abs(fractions[0] - TARGET_FRACTIONS[0]), end_train, end_validation)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return None
    _, _, end_train, end_validation = best
    train_units = set(units[:end_train].tolist())
    validation_units = set(units[end_train:end_validation].tolist())
    test_units = set(units[end_validation:].tolist())
    train_idx = np.flatnonzero(np.isin(keys, list(train_units))).astype(int)
    validation_idx = np.flatnonzero(np.isin(keys, list(validation_units))).astype(int)
    test_idx = np.flatnonzero(np.isin(keys, list(test_units))).astype(int)
    details = {
        "n_time_units": int(len(units)),
        "train_time_key_min": float(min(train_units)),
        "train_time_key_max": float(max(train_units)),
        "validation_time_key_min": float(min(validation_units)),
        "validation_time_key_max": float(max(validation_units)),
        "test_time_key_min": float(min(test_units)),
        "test_time_key_max": float(max(test_units)),
        "fractions": {
            "train": len(train_idx) / total,
            "validation": len(validation_idx) / total,
            "test": len(test_idx) / total,
        },
        "target_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
    }
    return train_idx, validation_idx, test_idx, details


def _group_ranges(groups: pd.Series, order_key: pd.Series) -> list[tuple[str, float, float]]:
    frame = pd.DataFrame({"group": groups.astype(str), "time": order_key.to_numpy()})
    aggregated = frame.groupby("group", sort=False)["time"].agg(["min", "max"])
    return [(str(idx), float(row["min"]), float(row["max"])) for idx, row in aggregated.iterrows()]


def _partition_of(value: float, train_max: float, validation_max: float) -> int:
    if value <= train_max:
        return 0
    if value <= validation_max:
        return 1
    return 2


def make_clean_split(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    protocol: str,
    split_seed: int,
    groups: pd.Series | None = None,
    time: pd.Series | None = None,
) -> CleanSplit:
    """Build one clean split. Infeasible combined splits are returned, not trained."""

    if len(features) != len(target):
        raise SplitError("Features and target row counts differ")
    if split_seed < 0:
        raise SplitError("split_seed must be non-negative")
    clusters = duplicate_cluster_ids(features)
    y = target.astype(str).reset_index(drop=True)
    clusters = clusters.reset_index(drop=True)

    if protocol in {PROTOCOL_STRATIFIED, PROTOCOL_GROUP}:
        if protocol == PROTOCOL_GROUP:
            if groups is None:
                raise SplitError("Group-safe split requires group metadata")
            membership = _component_ids([groups.reset_index(drop=True), clusters])
            link = "entity_or_exact_predictor_duplicate"
        else:
            membership = clusters.to_numpy()
            link = "exact_predictor_duplicate"
        unit_ids = np.unique(membership)
        strata = np.asarray(
            [_majority_label(y.to_numpy()[membership == unit]) for unit in unit_ids]
        )
        train_units, validation_units, test_units, notes = _assign_units(unit_ids, strata, split_seed)
        train_idx = _rows_for_units(membership, train_units)
        validation_idx = _rows_for_units(membership, validation_units)
        test_idx = _rows_for_units(membership, test_units)
        return CleanSplit(
            protocol=protocol,
            split_seed=split_seed,
            seed_affects_membership=True,
            status=STATUS_SUCCESS,
            train_idx=train_idx,
            validation_idx=validation_idx,
            test_idx=test_idx,
            details={
                "link": link,
                "n_units": int(len(unit_ids)),
                "assignment_notes": notes,
                "target_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            },
        )

    if protocol == PROTOCOL_TEMPORAL:
        if time is None:
            raise SplitError("Temporal split requires temporal metadata")
        assigned = _chronological_assignment(time_order_key(time.reset_index(drop=True)))
        if assigned is None:
            raise SplitError("Temporal column does not have three ordered units")
        train_idx, validation_idx, test_idx, details = assigned
        return CleanSplit(
            protocol=protocol,
            split_seed=split_seed,
            seed_affects_membership=False,
            status=STATUS_SUCCESS,
            train_idx=train_idx,
            validation_idx=validation_idx,
            test_idx=test_idx,
            details=details,
        )

    if protocol == PROTOCOL_COMBINED:
        if groups is None or time is None:
            raise SplitError("Combined split requires group and temporal metadata")
        order_key = time_order_key(time.reset_index(drop=True))
        ranges = _group_ranges(groups.reset_index(drop=True), order_key)
        spanning = sum(1 for _, low, high in ranges if high > low)
        assigned = _chronological_assignment(order_key)
        min_crossing = None
        if assigned is not None:
            keys = order_key.to_numpy()
            units = np.array(sorted(pd.unique(keys)))
            crossing_values: list[int] = []
            for end_train in range(1, len(units) - 1):
                for end_validation in range(end_train + 1, len(units)):
                    train_max = float(units[end_train - 1])
                    validation_max = float(units[end_validation - 1])
                    crossed = 0
                    for _, low, high in ranges:
                        if _partition_of(low, train_max, validation_max) != _partition_of(
                            high, train_max, validation_max
                        ):
                            crossed += 1
                    crossing_values.append(crossed)
            min_crossing = min(crossing_values) if crossing_values else None
        feasible = min_crossing == 0
        if not feasible or assigned is None:
            return CleanSplit(
                protocol=protocol,
                split_seed=split_seed,
                seed_affects_membership=False,
                status=STATUS_INFEASIBLE,
                train_idx=np.asarray([], dtype=int),
                validation_idx=np.asarray([], dtype=int),
                test_idx=np.asarray([], dtype=int),
                details={
                    "reason": (
                        "No chronological train/validation/test boundary keeps every "
                        "entity inside a single partition."
                    ),
                    "n_groups": len(ranges),
                    "n_groups_spanning_multiple_times": spanning,
                    "minimum_groups_crossing_a_chronological_cut": min_crossing,
                    "alternatives": [
                        "group-disjoint split without chronology",
                        "chronological split without group disjointness",
                    ],
                    "trained": False,
                },
            )
        # A zero-crossing chronological cut exists. Recompute that cut explicitly.
        keys = order_key.to_numpy()
        units = np.array(sorted(pd.unique(keys)))
        total = len(keys)
        best: tuple[float, int, int] | None = None
        for end_train in range(1, len(units) - 1):
            for end_validation in range(end_train + 1, len(units)):
                train_max = float(units[end_train - 1])
                validation_max = float(units[end_validation - 1])
                if any(
                    _partition_of(low, train_max, validation_max) != _partition_of(high, train_max, validation_max)
                    for _, low, high in ranges
                ):
                    continue
                n_train = int((keys <= train_max).sum())
                n_validation = int(((keys > train_max) & (keys <= validation_max)).sum())
                n_test = total - n_train - n_validation
                if min(n_train, n_validation, n_test) <= 0:
                    continue
                fractions = (n_train / total, n_validation / total, n_test / total)
                score = sum(abs(fractions[index] - TARGET_FRACTIONS[index]) for index in range(3))
                candidate = (score, end_train, end_validation)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            raise SplitError("Combined search reported a feasible cut that could not be reconstructed")
        _, end_train, end_validation = best
        train_max = float(units[end_train - 1])
        validation_max = float(units[end_validation - 1])
        train_idx = np.flatnonzero(keys <= train_max).astype(int)
        validation_idx = np.flatnonzero((keys > train_max) & (keys <= validation_max)).astype(int)
        test_idx = np.flatnonzero(keys > validation_max).astype(int)
        return CleanSplit(
            protocol=protocol,
            split_seed=split_seed,
            seed_affects_membership=False,
            status=STATUS_SUCCESS,
            train_idx=train_idx,
            validation_idx=validation_idx,
            test_idx=test_idx,
            details={
                "n_groups": len(ranges),
                "n_groups_spanning_multiple_times": spanning,
                "minimum_groups_crossing_a_chronological_cut": 0,
                "train_time_key_max": train_max,
                "validation_time_key_max": validation_max,
            },
        )

    raise SplitError(f"Unknown clean protocol: {protocol}")
