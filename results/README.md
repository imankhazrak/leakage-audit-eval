# Results

This folder holds the paper's tables and figures as submitted. The numbers
behind them, down to each episode, are under `research/` next to the code that
produced them.

```
results/
├── tables/                       # every table of the paper, as LaTeX (t_*.tex)
├── figures/                      # every figure, as PDF and PNG
└── matched_l5_tabulation.json    # matched proxy vs. legitimate predictor (Sec. 6.3, App. H.3)
```

## Regenerating without fitting

```bash
bash scripts/reproduce_tables.sh            # writes results/rebuilt/
```

| Step | Reads | Writes |
|---|---|---|
| Matched-arm tabulation | saved predictions and features in `research/` | `results/rebuilt/matched_l5_tabulation.json` |
| Robustness tables, precision-recall figure | `research/review_response_2026-11-03/round9/runs/R9_RESULTS.json` | `results/rebuilt/round9/` |
| Study 2-3 and hidden-scope tables (`REBUILD_R8=1`) | `R8_RESULTS.json`, saved runs, **and the source tables** | `research/review_response_2026-11-03/round8/tables/` (in place) |

`pytest` runs the first two steps and compares the output with the shipped files.

Most figures were drawn from the values printed in the tables. Two have a
script here: `scripts/make_fig_perfamily.py` (per-family AUROC) and the
round-9 table builder (`fig_pr_dd`). The others are shipped as rendered files.

## Regenerating by refitting

See [`docs/REPRODUCING.md`](../docs/REPRODUCING.md). The fitted experiments were
run as Slurm array jobs; the job files record each command line.

## Where each number comes from

| Paper item | File |
|---|---|
| Per-family AUROC, full precision | `research/review_response_2026-11-03/round8/PER_FAMILY_FULL_PRECISION.csv` |
| Out-of-fold detector scores, both feature versions | `research/review_response_2026-11-03/round8/runs/dr/oof_predictions_r8.csv` |
| Study 3 ground truth and features | `research/stage7_2026-11-02/s3/<family>/episodes_{truth,features}.csv` |
| Frozen detectors and fold models | `research/stage7_2026-11-02/detector/` |
| Study 2 cells | `research/stage6_2026-11-01/main/` |
| Targeted reruns, hidden scope (ten seeds), C2ST, static analysis | `research/review_response_2026-11-03/round8/R8_RESULTS.json` |
| Robustness experiments | `research/review_response_2026-11-03/round9/runs/R9_RESULTS.json` |

The complete map from sections, tables and figures to files is in
[`docs/PAPER_MAP.md`](../docs/PAPER_MAP.md).

## Reading the results

- **The unit is the source family.** Each per-family AUROC pools that family's
  held-out episodes; the reported mean is over ten families, not over episodes.
  Seeds are repeated measurements within a family.
- **Paired comparisons** report the mean per-family difference, a 95% *t*
  interval, wins/losses of ten, and an exact two-sided sign-flip test over all
  2^10 assignments (smallest attainable *p* = 0.002).
- **Two feature versions.** `oof_predictions_r8.csv` has a `version` column:
  `replay` (saved features, identical to the frozen outputs) and `corrected`
  (exact matches by tuple equality, after a distance-precision defect was
  found). Appendix H.4 of the paper says which table uses which. No conclusion
  changes between them.
- **Post hoc results** are marked with a dagger in the paper. Each has its own
  later protocol; they are not part of the frozen primary analysis.
- **Operating points at 1% and 5% FPR are oracle**: thresholds were set on the
  evaluated episodes. They describe discrimination and are not validated
  operating points.
- **All ten tables were inspected during development.** These are development
  results for these planted mechanisms and audit implementations, not a general
  ranking of evidence types and not evidence about natural leakage.
