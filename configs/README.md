# Configuration

| File | Purpose |
|---|---|
| `suite.yaml` | Readable index of the evaluation suite: families, seeds, split, arms and study sizes. |
| `containers/leakage_analysis.def` | Apptainer definition that builds the static-analysis comparator (Yang et al., ASE 2022) at a pinned commit. |
| `containers/leakage_detector_runtime.def` | Apptainer definition of the detector runtime. |

## Where the frozen settings live

Every experiment was run under a protocol frozen before execution. Those
protocols are the authoritative settings and are kept next to the code and
outputs they govern, because the freeze hashes refer to those paths:

| Experiment | Frozen protocol |
|---|---|
| Panel selection and deduplication | `research/stage5_2026-11-01/amendment/`, `research/stage5_2026-11-01/panel/` |
| Study 2 | `research/stage6_2026-11-01/protocol/` |
| Study 3 and learned detectors | `research/stage7_2026-11-02/protocol/` |
| Feature recomputation, 23-arm set | `research/stage8_2026-11-02/protocol/` |
| Targeted reruns, hidden scope at ten seeds, C2ST, static analysis | `research/review_response_2026-11-03/round8/PROTOCOL_R8.md` |
| Robustness: stronger learners, linkage and clone statistics, unseen recipes, operating points | `research/review_response_2026-11-03/round9/PROTOCOL_R9.md` |

Thresholds, feature whitelists and detector hyperparameter grids are part of
those protocols. Do not tune them; anything added later is labelled post hoc in
the paper.

The image path used by the static-analysis runner is set in
`research/review_response_2026-11-03/round8/code/yang_run.py`.
