# Map from the paper to this repository

Section, table and figure numbers refer to the submitted PDF.

| Paper | Location |
|---|---|
| Panel of ten families, deduplication (Table 8) | `research/stage5_2026-11-01/` (`code/select_frame_v5.py`, `panel/`, `amendment/`) |
| Study 2 (Sec. 6.2, Fig. 2a, Tables 9-10) | `research/stage6_2026-11-01/` (`code/s2_run.py`, `code/analyze_s2.py`, `protocol/`, `main/`) |
| Study 3 episodes, features, behavioral responses | `research/stage7_2026-11-02/code/s3_run.py`, `research/stage7_2026-11-02/s3/` |
| Study 3 learned detectors, frozen version (Sec. 6.3, Table 2, Fig. 3 top group, Table 11, App. D.3) | `research/stage7_2026-11-02/code/s7_detector_*.py`, `research/stage7_2026-11-02/detector/` |
| Interface parity with the reference library | `research/stage7_2026-11-02/parity/`, `research/stage7_2026-11-02/code/s7_parity_*.py` |
| Feature recomputation and verification | `research/stage8_2026-11-02/` |
| The 23-arm episode set with the two matched L5 arms | `research/stage8_2026-11-02/v4/<family>/` (`episodes_truth.csv`, `features_v3S.csv`, `features_v4.csv`) |
| Matched proxy and legitimate-predictor arms (Sec. 6.3, App. H.3, Table 19) | `scripts/matched_l5_tabulation.py`, `results/matched_l5_tabulation.json` |
| Positive control (1-NN, gradient boosting) | `research/review_response_2026-11-03/positive_control/` |
| Hidden scope, three seeds, and its comparators | `research/review_response_2026-11-03/l4_undeclared/`, `research/review_response_2026-11-03/l4u_comparator/` |
| Post hoc reanalyses and targeted reruns | `research/review_response_2026-11-03/round2/` ... `round7/`, `research/review_response_2026-11-03/code/` |
| Hidden scope at ten seeds, corrected-feature refits, targeted rerun and dilution, C2ST, static analysis (Sec. 6.3-6.5, Fig. 2b, Fig. 3 middle and bottom groups, Figs. 4-5, App. H.4-H.8) | `research/review_response_2026-11-03/round8/` (`PROTOCOL_R8.md`, `code/`, `runs/`, `R8_RESULTS.json`, `tables/`, `PER_FAMILY_FULL_PRECISION.csv`) |
| Robustness: stronger learners, record-linkage and clone statistics, hidden-scope extensions, unseen recipes, operating points (Sec. 6.5-6.6, Table 3, Fig. 12, App. H.9) | `research/review_response_2026-11-03/round9/` (`PROTOCOL_R9.md`, `code/`, `runs/`, `runs/R9_RESULTS.json`) |
| Deepchecks driver | `research/multifamily_stage0_continuation_c4_2026-10-09/src/` |
| Audit library used by the runners | `src/leakagelens/interventions/` |
| Dataset documentation (App. C) | `docs/DATASHEET.md` |

## Tables and figures as submitted

| Paper item | File in `results/` |
|---|---|
| Fig. 1, evidence type by mechanism | `figures/fig_map.{pdf,png}` |
| Fig. 2, flag rates on deduplicated tables | `figures/fig_s2.{pdf,png}` |
| Fig. 3, paired family-level comparisons | `figures/fig_paired.{pdf,png}` |
| Fig. 4, AUROC by mechanism | `figures/fig_mech.{pdf,png}` |
| Fig. 5, hidden scope at ten seeds | `figures/fig_l4u.{pdf,png}` |
| Appendix figures | `figures/fig_app_*.{pdf,png}`, `figures/fig_pr_dd.{pdf,png}` |
| Not in the PDF: dot plot of the Table 2 values | `figures/fig_perfamily.{pdf,png}` (`scripts/make_fig_perfamily.py`) |
| Table 2, per-family AUROC by view | `tables/t_perfamily_main.tex` |
| Table 3, robustness | `tables/t_robust.tex` |
| Dataset table | `tables/t_data.tex` |
| Audit overview | `tables/t_cheatsheet.tex` |
| Appendix tables | the remaining `tables/t_*.tex` |

## About the experiment tree

`research/` holds, for every experiment, its code, its frozen protocol and its
saved outputs in one place. Directory names carry dates and stage labels
because the runners resolve paths through them and the freeze hashes refer to
them; they were left unchanged.

- Each post hoc experiment has its own `PROTOCOL_*.md`, with a freeze hash
  recorded before any of its jobs ran. The matched-arm tabulation is the
  exception: it fits nothing, has no protocol, and is descriptive.
- Freeze files (`FREEZE*.sha256`, `SHA256SUMS`, `MAIN_FREEZE.sha256`) were
  computed before identifying strings were replaced by placeholders such as
  `<REPO>` and `<ACCOUNT>`, so a file that was scrubbed will not match its
  recorded freeze hash. `research/MANIFEST.sha256` lists the hashes of the
  files as shipped; check them with `python scripts/verify_manifest.py`.
- Two files over 5 MB were left out; they are listed in
  `docs/OMITTED_OVER_SIZE_CAP.txt`.
