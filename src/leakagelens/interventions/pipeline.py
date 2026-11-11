"""The auditor's view of a suspected pipeline.

A ``SuspectedPipeline`` holds only what an independent auditor could observe
from a fitted pipeline and its data: the predictor matrix, the labels, the
partition memberships the pipeline used, the preprocessing fit scope, any
declared group or time metadata, and the model specification.

It deliberately cannot hold the injection identity. Fields such as the leakage
family, donor rows, the injected proxy identity, severity, or the original
clean partition are refused at construction time. Label-blind interventions in
:mod:`leakagelens.interventions.rules` see this object and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any, Mapping

import numpy as np
import pandas as pd

# Names that identify an injection rather than describe a pipeline. Any attempt
# to pass one of these into ``SuspectedPipeline`` raises ``InformationFlowError``.
FORBIDDEN_STATE_FIELDS: tuple[str, ...] = (
    "family",
    "leakage_family",
    "episode_label",
    "severity_label",
    "severity_requested",
    "severity_achieved",
    "severity_parameter_name",
    "severity_unit",
    "donor_idx",
    "proxy",
    "proxy_column",
    "anchor_train",
    "anchor_validation",
    "anchor_test",
    "original_train_idx",
    "clean_train_idx",
    "clean_reference",
    "details",
    "status",
    "indicators",
)


class InformationFlowError(RuntimeError):
    """Raised when injection-derived information reaches a label-blind component."""


def _as_index(values: Any) -> np.ndarray:
    return np.asarray(sorted(int(item) for item in values), dtype=int)


@dataclass(frozen=True)
class SuspectedPipeline:
    """Everything a label-blind intervention may read.

    ``train_idx`` are the rows the estimator was fit on. ``preprocessor_fit_idx``
    are the rows the preprocessing transformers were fit on. ``validation_idx``
    and ``test_idx`` are the held-out partitions the suspected pipeline itself
    reports. Group and time series are positional over the full table and are
    present only when the dataset declares them.
    """

    features: pd.DataFrame
    target: pd.Series
    train_idx: np.ndarray
    validation_idx: np.ndarray
    test_idx: np.ndarray
    preprocessor_fit_idx: np.ndarray
    model_family: str
    model_backend: str
    model_params: Mapping[str, Any]
    model_seed: int
    groups: pd.Series | None = None
    time: pd.Series | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in self.provenance:
            if str(name) in FORBIDDEN_STATE_FIELDS:
                raise InformationFlowError(f"provenance carries injection field {name!r}")
        n_rows = len(self.features)
        if len(self.target) != n_rows:
            raise ValueError("features and target lengths differ")
        for label, series in (("groups", self.groups), ("time", self.time)):
            if series is not None and len(series) != n_rows:
                raise ValueError(f"{label} metadata length does not match the table")
        object.__setattr__(self, "train_idx", _as_index(self.train_idx))
        object.__setattr__(self, "validation_idx", _as_index(self.validation_idx))
        object.__setattr__(self, "test_idx", _as_index(self.test_idx))
        object.__setattr__(self, "preprocessor_fit_idx", _as_index(self.preprocessor_fit_idx))
        for label, index in (
            ("train", self.train_idx),
            ("validation", self.validation_idx),
            ("test", self.test_idx),
            ("preprocessor_fit", self.preprocessor_fit_idx),
        ):
            if len(index) and (index.min() < 0 or index.max() >= n_rows):
                raise ValueError(f"{label} indices fall outside the table")
        object.__setattr__(self, "model_params", dict(self.model_params))

    @property
    def heldout_idx(self) -> np.ndarray:
        return _as_index(np.concatenate([self.validation_idx, self.test_idx]))

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])

    def with_training_rows(self, train_idx: np.ndarray) -> "SuspectedPipeline":
        """Return a copy whose estimator and preprocessor fit on ``train_idx`` only."""

        return replace(self, train_idx=_as_index(train_idx), preprocessor_fit_idx=_as_index(train_idx))

    def with_preprocessor_scope(self, fit_idx: np.ndarray) -> "SuspectedPipeline":
        return replace(self, preprocessor_fit_idx=_as_index(fit_idx))

    def without_feature_positions(self, positions: list[int]) -> "SuspectedPipeline":
        """Drop predictors by position. Names are not consulted."""

        keep = [index for index in range(self.n_features) if index not in set(int(item) for item in positions)]
        if not keep:
            raise ValueError("An intervention cannot remove every predictor")
        return replace(self, features=self.features.iloc[:, keep].reset_index(drop=True))

    def structural_summary(self) -> dict[str, int]:
        return {
            "n_train": int(len(self.train_idx)),
            "n_validation": int(len(self.validation_idx)),
            "n_test": int(len(self.test_idx)),
            "n_preprocessor_fit": int(len(self.preprocessor_fit_idx)),
            "n_features": self.n_features,
        }


def suspected_from_state(
    state: Any,
    *,
    features: pd.DataFrame,
    target: pd.Series,
    model_family: str,
    model_backend: str,
    model_params: Mapping[str, Any],
    model_seed: int,
    groups: pd.Series | None = None,
    time: pd.Series | None = None,
) -> SuspectedPipeline:
    """Build the auditor's view from a Phase-3 ``LeakageState``.

    Only the pipeline-describing fields are copied. If the state carries a
    synthetic proxy it is appended as an ordinary predictor column with a
    neutral positional name, because the auditor sees a column, not an
    injection. The family, donors, anchors, severity, and details never leave
    this function.
    """

    table = features.reset_index(drop=True).copy()
    proxy = getattr(state, "proxy", None)
    if proxy is not None:
        neutral = f"feature_{table.shape[1]}"
        while neutral in table.columns:
            neutral = f"{neutral}_"
        table[neutral] = pd.Series(proxy).astype(str).to_numpy()
    return SuspectedPipeline(
        features=table,
        target=target.reset_index(drop=True),
        train_idx=np.asarray(state.model_train_idx, dtype=int),
        validation_idx=np.asarray(state.validation_idx, dtype=int),
        test_idx=np.asarray(state.test_idx, dtype=int),
        preprocessor_fit_idx=np.asarray(state.preprocessor_fit_idx, dtype=int),
        model_family=str(model_family),
        model_backend=str(model_backend),
        model_params=dict(model_params),
        model_seed=int(model_seed),
        groups=None if groups is None else groups.reset_index(drop=True),
        time=None if time is None else time.reset_index(drop=True),
    )


def public_field_names() -> tuple[str, ...]:
    return tuple(item.name for item in fields(SuspectedPipeline))


__all__ = [
    "FORBIDDEN_STATE_FIELDS",
    "InformationFlowError",
    "SuspectedPipeline",
    "public_field_names",
    "suspected_from_state",
]
