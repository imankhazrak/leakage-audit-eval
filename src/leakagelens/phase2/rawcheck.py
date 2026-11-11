"""Checksum selected raw caches without opening them for writing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yaml

from leakagelens.paths import get_project_root


def selected_raw_checksums(root: str | Path | None = None) -> pd.DataFrame:
    project = Path(root or get_project_root())
    manifest = yaml.safe_load((project / "configs/datasets/benchmark_manifest.yaml").read_text(encoding="utf-8"))
    rows = []
    for dataset_id, entry in manifest["datasets"].items():
        cache = (
            project
            / "data"
            / "raw"
            / "openml"
            / str(entry["provider_dataset_id"])
            / f"version_{entry['provider_version']}"
        )
        for name in ("features.parquet", "target.parquet"):
            path = cache / name
            digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "path": str(path.relative_to(project)),
                    "sha256": digest,
                    "bytes": path.stat().st_size if path.is_file() else 0,
                }
            )
    return pd.DataFrame(rows)
