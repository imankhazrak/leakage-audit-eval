# Reproducing the results

There are three levels. Each needs more than the one before.

| Level | What it shows | Needs | Time |
|---|---|---|---|
| 1. Check | the saved outputs are intact and give the printed numbers | this repository | about a minute |
| 2. Rebuild | tables are regenerated from saved results | level 1, plus the source tables for some | minutes |
| 3. Refit | episodes, features and detectors are recomputed | source tables, a Slurm cluster | CPU hours |

Level 1, and the two level-2 steps that need no source table, were run from this
layout. The round-8 table builder was run in the original working tree, not
from this layout. **Level 3 has not been rerun from this repository**, by the
authors or independently; the job files record what was run.

## Environment

```bash
conda env create --file environment.yml
conda activate leakagelens
pip install -e ".[dev]"
```

The reported results used Python 3.12 with numpy 2.3.3, pandas 2.5.3, scipy
1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0, pyarrow 25.0.1 and openml 0.15.1.
The library itself supports Python 3.10 and later.

## Level 1: check

```bash
pytest -q                               # unit tests and consistency checks
pytest -q -m slow                       # hash every file of research/
python scripts/verify_manifest.py       # the same hash check, as a script
```

`tests/test_reported_results.py` recomputes the per-family AUROCs of the
structural, behavioral and combined views from the saved out-of-fold scores and
compares them with the shipped table and CSV.

## Level 2: rebuild tables without fitting

```bash
bash scripts/reproduce_tables.sh                 # matched-arm tabulation and round-9 tables
REBUILD_R8=1 bash scripts/reproduce_tables.sh    # also the Study 2-3 tables; needs the source tables
```

The round-8 builder recomputes the table inventory from the source tables, so
prepare them first (see `data/README.md`). It writes into
`research/review_response_2026-11-03/round8/tables/` in place.

## Level 3: refit

1. **Prepare the data** as in `data/README.md` and check the hashes.
2. **Edit the job files.** Every `*.sbatch` under `research/` records the exact
   command line and the array of families. Replace the placeholders
   `<ACCOUNT>`, `<REPO>`, `<HOME>` and `<PROJECT_STORAGE>`. Commands are run
   from the repository root.
3. **Submit in order.** Later stages read the outputs of earlier ones:

   | Order | Stage | Job files |
   |---|---|---|
   | 1 | Panel and deduplication | `research/stage5_2026-11-01/panel/run.sbatch` |
   | 2 | Study 2 | `research/stage6_2026-11-01/main/run_*.sbatch` |
   | 3 | Study 3 episodes, then detectors | `research/stage7_2026-11-02/s3/run_s3.sbatch`, `research/stage7_2026-11-02/detector/run_detector.sbatch` |
   | 4 | Feature recomputation, 23-arm set | `research/stage8_2026-11-02/v4/run_features.sbatch`, `research/stage8_2026-11-02/models/run_train.sbatch` |
   | 5 | Positive control, hidden scope | `research/review_response_2026-11-03/{positive_control,l4_undeclared,l4u_comparator}/run_*.sbatch` |
   | 6 | Targeted reruns, ten seeds, C2ST, static analysis | `research/review_response_2026-11-03/round8/slurm/r8_*.sbatch` |
   | 7 | Robustness | `research/review_response_2026-11-03/round9/slurm/r9_*.sbatch` |

4. **Notes.**
   - Runners write into fresh directories and refuse to overwrite. To rerun a
     step, move its existing output directory aside first.
   - Runners refuse to start outside a Slurm job (they check `SLURM_JOB_ID`).
     To run one task on a workstation, set `SLURM_JOB_ID` and
     `SLURM_ARRAY_TASK_ID` by hand.
   - Each job uses one or two CPU cores and 8-16 GB of memory. No GPU is needed.

### Static-analysis comparator

The comparator runs the tool of Yang et al. (ASE 2022) inside an Apptainer
image. Build it from `configs/containers/leakage_analysis.def`, which pins the
upstream commit, and set the image path in
`research/review_response_2026-11-03/round8/code/yang_run.py`. The image, its
intermediate Datalog facts and its HTML reports are not stored here; the
generated programs and the tool's receipts are.

## What is not in the repository

- The source tables and per-episode working copies of their rows.
- The static-analysis container image and its intermediate files.
- Job logs and scheduler accounting.
- Two files over 5 MB (`docs/OMITTED_OVER_SIZE_CAP.txt`).
