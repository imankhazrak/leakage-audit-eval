# Data

**No source table is stored in this repository.** The suite is built from ten
public binary-classification tables that you download from their repositories.
Each is kept as numeric predictors and complete rows, in a raw and a
deduplicated (dd) variant.

| Family | Name | Predictors | Rows | Rows (dedup.) | Class-1 share | License |
|---|---|---:|---:|---:|---:|---|
| OpenML 1067 | kc1 | 21 | 2,109 | 1,192 | 0.15 | Public (OpenML) |
| OpenML 1462 | banknote | 4 | 1,372 | 1,348 | 0.44 | Public (OpenML) |
| OpenML 1464 | blood-transfusion | 4 | 748 | 502 | 0.76 | Public (OpenML) |
| OpenML 1487 | ozone-level | 72 | 2,534 | 2,526 | 0.06 | Public (OpenML) |
| OpenML 1489 | phoneme | 5 | 5,404 | 5,395 | 0.29 | Public (OpenML) |
| OpenML 1494 | qsar-biodeg | 41 | 1,055 | 1,052 | 0.34 | Public (OpenML) |
| OpenML 44 | spambase | 57 | 4,601 | 4,207 | 0.61 | Public (OpenML) |
| UCI 159 | MAGIC gamma | 10 | 19,020 | 18,905 | 0.35 | CC BY 4.0 |
| UCI 329 | diabetic retinopathy | 19 | 1,151 | 1,146 | 0.53 | CC BY 4.0 |
| UCI 563 | Iranian churn | 13 | 3,150 | 2,836 | 0.16 | CC BY 4.0 |

All OpenML tables are dataset version 1. The license column repeats what each
repository states; check the terms yourself before redistributing a table.

## Preparation

Run every command from the repository root.

1. **Six OpenML tables** (44, 1067, 1462, 1464, 1487, 1489). The runners read
   `data/raw/openml/<id>/version_1/{features,target}.parquet`:

   ```bash
   pip install -e ".[data]"
   python scripts/download_data.py
   ```

2. **OpenML 1494 and the three UCI tables.** The panel-selection script
   downloads them, keeps numeric predictors and complete rows, and writes
   `research/stage5_2026-11-01/panel/data_v5/<family>/data.csv`:

   ```bash
   python research/stage5_2026-11-01/code/select_frame_v5.py
   ```

3. **Check the downloads** against the hashes of the files as used in the paper:

   ```bash
   sha256sum -c data/DATA_HASHES.sha256
   ```

   A parquet file written by a different `pyarrow` version can differ in bytes
   while holding the same table. If a parquet hash fails, install the pinned
   versions from `environment.yml` and download again before concluding that
   the data changed.

## Layout after preparation

```
data/
├── README.md
├── DATA_HASHES.sha256
└── raw/                         # git-ignored
    └── openml/<id>/version_1/{features,target}.parquet
research/stage5_2026-11-01/panel/data_v5/<family>/data.csv   # git-ignored
```

## Rules

- `data/raw/` is immutable after download. Do not edit a source table.
- Do not commit source tables or per-episode working copies; `.gitignore` excludes them.
- Every training, evaluation and reserve file is regenerated from the source
  tables and the seeds (104729, 130363, 155921), so splits are recoverable.
- Ground truth comes from source identifiers that are stored apart from every
  method's input (`episodes_truth.csv` next to `episodes_features.csv`).

The full dataset documentation is in [`docs/DATASHEET.md`](../docs/DATASHEET.md).
