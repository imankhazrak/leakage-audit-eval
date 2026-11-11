"""Load Phase-2 YAML without creating a second config schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from leakagelens.paths import get_project_root

PHASE2_CONFIG = Path("configs/phase2/phase2.yaml")


def load_yaml(relative: str | Path, root: Path | None = None) -> dict[str, Any]:
    path = Path(relative)
    if not path.is_absolute():
        path = Path(root or get_project_root()) / path
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a mapping")
    return payload


def load_phase2_config(root: Path | None = None) -> dict[str, Any]:
    return load_yaml(PHASE2_CONFIG, root)


def load_seeds(root: Path | None = None) -> list[int]:
    config = load_phase2_config(root)
    payload = load_yaml(config["seeds_file"], root)
    seeds = payload.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(isinstance(item, bool) or not isinstance(item, int) for item in seeds):
        raise ValueError("Phase-2 seeds must be a non-empty list of integers")
    return [int(item) for item in seeds]


def load_model_specs(root: Path | None = None) -> dict[str, dict[str, Any]]:
    config = load_phase2_config(root)
    specs: dict[str, dict[str, Any]] = {}
    for family, relative in config["models"].items():
        spec = load_yaml(relative, root)
        if spec.get("family") != family:
            raise ValueError(f"{relative} family does not match {family}")
        specs[family] = spec
    return specs
