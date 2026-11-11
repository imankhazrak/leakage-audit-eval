"""Label-blind corrective edits and the fixed decision rule (no model is fitted)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leakagelens.interventions import InformationFlowError, RuleConfig, SuspectedPipeline, apply_rule
from leakagelens.interventions.episode_decision import decide_episode

CONFIG = RuleConfig(proxy_screen_threshold=0.9, proxy_screen_bins=10, placebo_row_fraction=0.05, placebo_seed_offset=1000)


def _pipeline(copies: int = 0, **provenance) -> SuspectedPipeline:
    rng = np.random.default_rng(0)
    features = pd.DataFrame(rng.normal(size=(120, 4)), columns=[f"f{i}" for i in range(4)])
    target = pd.Series(rng.integers(0, 2, size=120))
    train, test = np.arange(0, 90), np.arange(90, 120)
    if copies:  # the first `copies` training rows become exact copies of evaluation rows
        features.iloc[:copies] = features.iloc[90:90 + copies].to_numpy()
    return SuspectedPipeline(
        features=features, target=target, train_idx=train, validation_idx=np.array([], dtype=int), test_idx=test,
        preprocessor_fit_idx=train, model_family="logistic_regression", model_backend="sklearn", model_params={},
        model_seed=0, provenance=dict(provenance),
    )


@pytest.mark.parametrize("field", ["family", "donor_idx", "proxy", "severity_label", "anchor_train"])
def test_pipeline_refuses_injection_fields(field: str) -> None:
    with pytest.raises(InformationFlowError):
        _pipeline(**{field: "x"})


@pytest.mark.parametrize("copies", [0, 4, 16])
def test_i1_removes_exactly_the_copied_training_rows(copies: int) -> None:
    pipeline = _pipeline(copies)
    edited, change = apply_rule(pipeline, "I1", CONFIG)
    assert change.rows_removed_from_training == copies
    assert len(edited.train_idx) == len(pipeline.train_idx) - copies
    assert set(edited.train_idx) == set(range(copies, 90))


def test_i4_narrows_the_preprocessing_scope_to_training_rows() -> None:
    pipeline = _pipeline().with_preprocessor_scope(np.arange(0, 120))
    edited, change = apply_rule(pipeline, "I4", CONFIG)
    assert change.preprocessor_fit_rows_before == 120
    assert change.preprocessor_fit_rows_after == 90
    assert set(edited.preprocessor_fit_idx) == set(range(90))


def test_decision_fires_only_when_the_correction_beats_its_placebo() -> None:
    fired = decide_episode({
        "I1": {"status": "applied", "oriented_delta": -0.03},
        "P1": {"status": "applied", "oriented_delta": -0.001},
        "I4": {"status": "applied_no_structural_change", "oriented_delta": 0.0},
        "P5": {"status": "applied", "oriented_delta": -0.002},
    })
    assert fired["decision"] == "positive" and fired["diagnosis"] == "L1_overlap"
    quiet = decide_episode({
        "I1": {"status": "applied", "oriented_delta": 0.001},
        "P1": {"status": "applied", "oriented_delta": -0.01},
        "I4": {"status": "applied", "oriented_delta": 0.0},
        "P5": {"status": "applied", "oriented_delta": 0.0},
    })
    assert quiet["decision"] == "negative"


def test_decision_abstains_without_a_control() -> None:
    out = decide_episode({"I1": {"status": "applied", "oriented_delta": -0.2}, "P1": {"status": "not_applicable", "oriented_delta": ""}})
    assert out["decision"] == "abstain"
