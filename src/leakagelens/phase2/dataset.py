"""Load one frozen dataset into predictors, labels, and separated metadata.

Raw files are read and never rewritten. Derived group and time columns are read
from ``data/interim`` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from leakagelens.data.openml_adapter import OpenMLAdapter
from leakagelens.paths import get_project_root
from leakagelens.phase2.freeze_gate import load_frozen_manifest
from leakagelens.phase2.protocols import select_clean_protocol
from leakagelens.phase2.validation import validate_predictor_matrix

BLOCKED_IN_RAW = {"biological_mouse_id", "drift_batch"}


class DatasetLoadError(RuntimeError):
    pass


@dataclass
class DatasetBundle:
    dataset_id: str
    name: str
    provider: str
    provider_dataset_id: int
    provider_version: int
    task_type: str
    target_name: str
    features: pd.DataFrame
    target: pd.Series
    groups: pd.Series | None
    conditional_groups: pd.Series | None
    time: pd.Series | None
    conditional_time: pd.Series | None
    roles: dict[str, str]
    protocol: str
    group_status: str
    temporal_status: str
    benchmark_version: str


def _read_interim_column(root: Path, dataset_id: str, filename: str) -> pd.Series | None:
    path = root / "data" / "interim" / "phase1" / dataset_id / filename
    if not path.is_file():
        return None
    frame = pd.read_parquet(path)
    if frame.shape[1] != 1:
        raise DatasetLoadError(f"{path} should contain one metadata column")
    return frame.iloc[:, 0]


def load_dataset_bundle(dataset_id: str, *, root: str | Path | None = None) -> DatasetBundle:
    project = Path(root or get_project_root())
    manifest = load_frozen_manifest(project)
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or dataset_id not in datasets:
        raise DatasetLoadError(f"{dataset_id} is not in the frozen benchmark")
    entry: dict[str, Any] = datasets[dataset_id]
    provider = str(entry["provider"])
    if provider != "openml":
        raise DatasetLoadError(f"Unsupported provider {provider}")
    dataset_number = int(entry["provider_dataset_id"])
    version = int(entry["provider_version"])
    raw_root = project / "data" / "raw"
    cache = raw_root / "openml" / str(dataset_number) / f"version_{version}"
    if not (cache / "features.parquet").is_file():
        raise DatasetLoadError(f"Missing raw cache for {dataset_id} at {cache}")
    loaded = OpenMLAdapter(raw_root).load_from_cache(dataset_number, version)
    roles = {str(key): str(value) for key, value in (entry.get("column_roles") or {}).items()}
    raw_columns = [str(column) for column in loaded.features.columns]
    derived_in_raw = sorted(BLOCKED_IN_RAW.intersection(raw_columns))
    if derived_in_raw:
        raise DatasetLoadError(f"Raw cache contains derived columns {derived_in_raw}")
    predictors = [column for column in raw_columns if roles.get(column) == "predictor"]
    if len(predictors) != int(entry["n_predictor_features"]):
        raise DatasetLoadError(
            f"{dataset_id} predictor count {len(predictors)} != frozen {entry['n_predictor_features']}"
        )
    features = loaded.features.loc[:, predictors].reset_index(drop=True)
    target = loaded.target.reset_index(drop=True)
    target.name = str(entry["target"])
    if len(features) != len(target) or len(features) != int(entry["n_samples"]):
        raise DatasetLoadError(f"{dataset_id} row count does not match the frozen manifest")
    validate_predictor_matrix(features, roles, target_name=str(entry["target"]))

    group_status = entry["leakage_applicability"]["group"]["status"]
    temporal_status = entry["leakage_applicability"]["temporal"]["status"]
    group_series = _read_interim_column(project, dataset_id, "group.parquet")
    time_series = _read_interim_column(project, dataset_id, "temporal.parquet")
    for series, label in ((group_series, "group"), (time_series, "time")):
        if series is not None and len(series) != len(features):
            raise DatasetLoadError(f"{dataset_id} {label} metadata length does not match predictors")
    groups = group_series if group_status == "supported" else None
    conditional_groups = group_series if group_status == "conditional" else None
    time = time_series if temporal_status == "supported" else None
    conditional_time = time_series if temporal_status == "conditional" else None
    if group_status == "supported" and groups is None:
        raise DatasetLoadError(f"{dataset_id} is L2 supported but group metadata is missing")
    if temporal_status == "supported" and time is None:
        raise DatasetLoadError(f"{dataset_id} is L3 supported but temporal metadata is missing")
    return DatasetBundle(
        dataset_id=dataset_id,
        name=str(entry["name"]),
        provider=provider,
        provider_dataset_id=dataset_number,
        provider_version=version,
        task_type=str(entry["task_type"]),
        target_name=str(entry["target"]),
        features=features,
        target=target,
        groups=None if groups is None else groups.reset_index(drop=True),
        conditional_groups=None if conditional_groups is None else conditional_groups.reset_index(drop=True),
        time=None if time is None else time.reset_index(drop=True),
        conditional_time=None if conditional_time is None else conditional_time.reset_index(drop=True),
        roles=roles,
        protocol=select_clean_protocol(group_status, temporal_status),
        group_status=group_status,
        temporal_status=temporal_status,
        benchmark_version=str(manifest["benchmark_manifest_version"]),
    )
