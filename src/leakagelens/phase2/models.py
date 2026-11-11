"""Fixed Phase-2 estimators.

Gradient boosting uses LightGBM with a fixed tree count. Early stopping is not
used, so validation does not change the number of trees and seeds stay
comparable. ``n_jobs`` comes from the SLURM allocation rather than the
machine-wide processor count.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from leakagelens.phase2.resources import allocated_cpus

PHASE2_GBDT_BACKEND = "lightgbm"
NO_EARLY_STOPPING_REASON = (
    "Fixed n_estimators keeps every dataset and seed on the same training budget. "
    "Validation is scored after fit and is not used to stop early."
)


def scientific_model_params(family: str, configured: dict[str, Any]) -> dict[str, Any]:
    """Parameters that belong in the experiment identity.

    Resource settings such as ``n_jobs`` are omitted because they follow the
    SLURM allocation rather than the scientific comparison.
    """

    ignored = {"n_jobs", "verbosity", "verbose"}
    return {key: value for key, value in configured.items() if key not in ignored and family}


def build_estimator(family: str, backend: str, params: dict[str, Any], seed: int) -> Any:
    n_jobs = allocated_cpus()
    if family == "logistic_regression":
        if backend != "sklearn":
            raise ValueError(f"logistic_regression backend must be sklearn, got {backend}")
        if str(params.get("penalty", "l2")) != "l2":
            raise ValueError("Phase-2 logistic regression uses L2 regularization")
        # sklearn 1.8 deprecates penalty='l2'. l1_ratio=0 is that same L2 penalty.
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            l1_ratio=0.0,
            solver=str(params.get("solver", "lbfgs")),
            max_iter=int(params.get("max_iter", 2000)),
            random_state=seed,
        )
    if family == "random_forest":
        if backend != "sklearn":
            raise ValueError(f"random_forest backend must be sklearn, got {backend}")
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 500)),
            max_features=params.get("max_features", "sqrt"),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=seed,
            n_jobs=n_jobs,
        )
    if family == "gradient_boosting":
        if backend != PHASE2_GBDT_BACKEND:
            raise ValueError(f"Phase-2 gradient boosting backend is {PHASE2_GBDT_BACKEND}, got {backend}")
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=int(params.get("n_estimators", 500)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            num_leaves=int(params.get("num_leaves", 31)),
            min_child_samples=int(params.get("min_child_samples", 20)),
            subsample=float(params.get("subsample", 1.0)),
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
            deterministic=True,
            force_row_wise=True,
        )
    raise ValueError(f"Unknown model family: {family}")
