"""OpenML dataset retrieval with raw-cache provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from leakagelens.data.checksum import sha256_file, sha256_bytes
from leakagelens.logging_utils import get_logger
from leakagelens.paths import get_project_root

LOGGER = get_logger("data.openml")


@dataclass(frozen=True)
class LoadedDataset:
    features: pd.DataFrame
    target: pd.Series
    metadata: dict[str, Any]


class OpenMLAdapter:
    """Load OpenML datasets into X, y with target excluded from X."""

    def __init__(self, raw_root: str | Path) -> None:
        self.raw_root = Path(raw_root)

    def cache_dir(self, dataset_id: int, version: int | None = None) -> Path:
        base = self.raw_root / "openml" / str(dataset_id)
        if version is not None:
            return base / f"version_{version}"
        return base

    def load(
        self,
        dataset_id: int,
        *,
        version: int | None = None,
        target_column: str | None = None,
        task_id: int | None = None,
        include_row_id: bool = False,
        include_ignore_attribute: bool = False,
    ) -> LoadedDataset:
        import openml

        if task_id is not None:
            task = openml.tasks.get_task(task_id, download_data=True)
            dataset = task.get_dataset()
            target_name = task.target_name or dataset.default_target_attribute
            x, y, _, _ = dataset.get_data(
                target=target_name,
                dataset_format="dataframe",
                include_row_id=include_row_id,
                include_ignore_attribute=include_ignore_attribute,
            )
        else:
            dataset = openml.datasets.get_dataset(dataset_id, version=version)
            target_name = target_column or dataset.default_target_attribute
            x, y, _, _ = dataset.get_data(
                target=target_name,
                dataset_format="dataframe",
                include_row_id=include_row_id,
                include_ignore_attribute=include_ignore_attribute,
            )

        if target_name in x.columns:
            x = x.drop(columns=[target_name])
        y = pd.Series(y, name=target_name)

        license_value = getattr(dataset, "licence", None) or getattr(dataset, "license", None)
        if license_value in (None, ""):
            license_value = "not_stated"

        meta = {
            "provider": "openml",
            "openml_dataset_id": dataset_id,
            "openml_dataset_version": int(getattr(dataset, "version", version) or 0) or version,
            "openml_task_id": task_id,
            "target": target_name,
            "dataset_name": dataset.name,
            "license": str(license_value),
            "include_row_id": include_row_id,
            "include_ignore_attribute": include_ignore_attribute,
            "original_source": getattr(dataset, "citation", None) or getattr(dataset, "url", None),
        }
        return LoadedDataset(features=x, target=y, metadata=meta)

    def download_and_cache(
        self,
        dataset_id: int,
        *,
        version: int | None = None,
        task_id: int | None = None,
        include_row_id: bool = False,
        include_ignore_attribute: bool = False,
    ) -> dict[str, Any]:
        """Download, serialize a canonical copy, and record checksums.

        ``checksum_metadata_payload_sha256`` hashes the canonical metadata object
        *excluding* checksum fields themselves (never a self-referential file hash).
        Feature/target file checksums are SHA-256 of the final parquet bytes.
        """

        loaded = self.load(
            dataset_id,
            version=version,
            task_id=task_id,
            include_row_id=include_row_id,
            include_ignore_attribute=include_ignore_attribute,
        )
        version_resolved = loaded.metadata.get("openml_dataset_version") or version
        if version_resolved is None:
            raise ValueError(f"Could not resolve OpenML version for dataset {dataset_id}")
        cache = self.cache_dir(dataset_id, int(version_resolved))
        cache.mkdir(parents=True, exist_ok=True)

        features_path = cache / "features.parquet"
        target_path = cache / "target.parquet"
        meta_path = cache / "metadata.json"

        loaded.features.to_parquet(features_path, index=False)
        loaded.target.to_frame().to_parquet(target_path, index=False)

        record: dict[str, Any] = {
            **loaded.metadata,
            "retrieved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "local_features_path": str(features_path.relative_to(get_project_root())),
            "local_target_path": str(target_path.relative_to(get_project_root())),
            "raw_features_sha256": sha256_file(features_path),
            "raw_target_sha256": sha256_file(target_path),
            "n_samples": int(len(loaded.features)),
            "n_features": int(loaded.features.shape[1]),
            "n_classes": int(loaded.target.nunique(dropna=False)),
        }
        payload_for_hash = {k: v for k, v in record.items() if not str(k).endswith("_sha256")}
        record["metadata_payload_sha256"] = sha256_bytes(
            json.dumps(payload_for_hash, sort_keys=True, default=str).encode("utf-8")
        )
        meta_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        LOGGER.info("Cached OpenML dataset %s version %s at %s", dataset_id, version_resolved, cache)
        return record

    def resolve_cache_dir(self, dataset_id: int, version: int | None = None) -> Path:
        """Resolve cache path for an exact provider version.

        ``version`` is required. Never silently falls back to another version.
        """

        if version is None:
            raise ValueError(
                f"Exact provider_version is required to load OpenML dataset {dataset_id}"
            )
        direct = self.cache_dir(dataset_id, int(version))
        if (direct / "metadata.json").is_file():
            return direct
        raise FileNotFoundError(
            f"No cached metadata for OpenML dataset {dataset_id} version {version} under {direct}"
        )

    def load_from_cache(self, dataset_id: int, version: int | None = None) -> LoadedDataset:
        cache = self.resolve_cache_dir(dataset_id, version)
        meta = normalize_cache_metadata(
            json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
        )
        cached_version = meta.get("openml_dataset_version")
        if version is not None and cached_version is not None and int(cached_version) != int(version):
            raise FileNotFoundError(
                f"Cache version mismatch for {dataset_id}: requested {version}, found {cached_version}"
            )
        features = pd.read_parquet(cache / "features.parquet")
        # Guard: derived metadata must never live in raw feature files
        forbidden = {"biological_mouse_id", "drift_batch"}
        present = forbidden.intersection(map(str, features.columns))
        if present:
            raise ValueError(
                f"Raw cache for OpenML {dataset_id} contains derived columns {sorted(present)}; "
                "re-acquire clean provider data"
            )
        target_frame = pd.read_parquet(cache / "target.parquet")
        target_col = meta.get("target") or target_frame.columns[0]
        target = target_frame[target_col] if target_col in target_frame.columns else target_frame.iloc[:, 0]
        return LoadedDataset(features=features, target=target, metadata=meta)


def normalize_cache_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Migrate legacy checksum field names; drop deprecated self-hash fields."""

    out = dict(meta)
    if "checksum_features_sha256" in out and "raw_features_sha256" not in out:
        out["raw_features_sha256"] = out.pop("checksum_features_sha256")
    else:
        out.pop("checksum_features_sha256", None)
    if "checksum_target_sha256" in out and "raw_target_sha256" not in out:
        out["raw_target_sha256"] = out.pop("checksum_target_sha256")
    else:
        out.pop("checksum_target_sha256", None)
    if "checksum_metadata_payload_sha256" in out and "metadata_payload_sha256" not in out:
        out["metadata_payload_sha256"] = out.pop("checksum_metadata_payload_sha256")
    else:
        out.pop("checksum_metadata_payload_sha256", None)
    # Deprecated: never a valid final-file hash once the hash was written into the file
    out.pop("checksum_metadata_sha256", None)
    return out
