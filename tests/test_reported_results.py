"""The saved outputs reproduce the numbers printed in the paper. Nothing is fitted.

These tests recompute statistics from saved out-of-fold predictions and compare
them with the shipped tables. They check the archive's internal consistency,
not the scientific claims.
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

R8 = "review_response_2026-11-03/round8"
FAMILIES = ["openml_1067", "openml_1462", "openml_1464", "openml_1487", "openml_1489", "openml_1494", "openml_44",
            "uci_159", "uci_329", "uci_563"]


def auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def sign_flip_p(d) -> float:
    d = np.asarray(d, float)
    flips = np.array(list(itertools.product((-1, 1), repeat=len(d))))
    return float(np.mean(np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-15))


@pytest.fixture(scope="module")
def truth(research) -> pd.DataFrame:
    files = sorted((research / "stage7_2026-11-02" / "s3").glob("*/episodes_truth.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


@pytest.fixture(scope="module")
def per_family(research, truth) -> pd.DataFrame:
    oof = pd.read_csv(research / R8 / "runs" / "dr" / "oof_predictions_r8.csv")
    oof = oof[(oof.study == "s3") & (oof.version == "replay")]
    label = truth.set_index("case_id").label
    rows = {}
    for view in ("S", "B", "SB"):
        scores = oof[oof.view == view].set_index("case_id")
        rows[view] = {fam: auroc(g.score[label.loc[g.index] == 1], g.score[label.loc[g.index] == 0])
                      for fam, g in scores.groupby("heldout_family")}
    return pd.DataFrame(rows).loc[FAMILIES]


def test_study3_composition(truth) -> None:
    assert len(truth) == 1260
    assert int((truth.label == 1).sum()) == 780 and int((truth.label == 0).sum()) == 480
    assert truth.arm.nunique() == 21 and sorted(truth.source_family.unique()) == sorted(FAMILIES)
    assert (truth.groupby("arm").size() == 60).all()


def test_suite_config_matches_saved_episodes(repo, truth) -> None:
    suite = yaml.safe_load((repo / "configs" / "suite.yaml").read_text())
    assert sorted(f["id"] for f in suite["families"]) == sorted(truth.source_family.unique())
    assert sorted(suite["suite"]["seeds"]) == sorted(int(s) for s in truth.split_seed.unique())
    for role, key in ((1, "positive"), (0, "negative")):
        assert sorted(suite["arms"][key]) == sorted(truth[truth.label == role].arm.unique())
    matched = pd.read_csv(repo / "research" / "stage8_2026-11-02" / "v4" / "openml_44" / "episodes_truth.csv")
    assert set(matched.arm) - set(truth.arm) == set(suite["arms"]["matched_post_hoc"])


def test_structure_beats_behavior_in_every_family(per_family) -> None:
    diff = per_family.S - per_family.B
    assert round(per_family.S.mean(), 3) == 0.857
    assert round(per_family.B.mean(), 3) == 0.610
    assert round(per_family.SB.mean(), 3) == 0.837
    assert round(diff.mean(), 3) == 0.247
    assert int((diff > 0).sum()) == 10
    assert round(sign_flip_p(diff), 3) == 0.002


def test_per_family_values_match_the_shipped_csv(research, per_family) -> None:
    saved = pd.read_csv(research / R8 / "PER_FAMILY_FULL_PRECISION.csv")
    for view in ("S", "B", "SB"):
        column = saved[saved.quantity == f"study3_{view}_per_family_AUROC"].set_index("family").value.loc[FAMILIES]
        np.testing.assert_allclose(per_family[view].to_numpy(), column.to_numpy(), atol=1e-12)


def test_per_family_values_match_the_paper_table(repo, per_family) -> None:
    tex = (repo / "results" / "tables" / "t_perfamily_main.tex").read_text()
    for view in ("S", "B", "SB"):
        line = next(l for l in tex.splitlines() if l.startswith(f"{view} &"))
        printed = [float(x) for x in line.rstrip("\\").split("&")[1:11]]
        assert printed == [round(v, 3) for v in per_family[view]]


def test_matched_arm_tabulation_reproduces(repo, tmp_path) -> None:
    out = tmp_path / "matched.json"
    subprocess.run([sys.executable, str(repo / "scripts" / "matched_l5_tabulation.py"), str(repo / "research"), str(out)], check=True)
    new, saved = json.loads(out.read_text()), json.loads((repo / "results" / "matched_l5_tabulation.json").read_text())
    assert new == saved
    assert saved["replay"]["flags_by_arm"]["PL5m"]["flags"] == 2 and saved["replay"]["flags_by_arm"]["NL5X"]["flags"] == 7


def test_round9_tables_rebuild(repo, tmp_path) -> None:
    script = repo / "research" / "review_response_2026-11-03" / "round9" / "code" / "build_tables_r9.py"
    (tmp_path / "tables").mkdir()   # the builder expects its output folder to exist
    subprocess.run([sys.executable, str(script), str(tmp_path)], check=True, env={**os.environ, "MPLBACKEND": "Agg"})
    built = sorted((tmp_path / "tables").glob("t_*.tex"))
    assert {"t_robust.tex", "t_r9_ho.tex", "t_r9_ops.tex"} <= {b.name for b in built}
    for table in built:
        shipped = repo / "results" / "tables" / table.name
        assert table.read_text() == shipped.read_text(), table.name
    assert (tmp_path / "figures" / "fig_pr_dd.pdf").is_file()
