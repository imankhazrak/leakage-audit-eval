"""Experiment identity for one clean reference run.

The hash reuses ``build_experiment_id``. It includes dataset, benchmark version,
split protocol and seed, preprocessing, and model family/backend/parameters/seed.
It does not include timestamps, SLURM ids, or machine paths.
"""

from __future__ import annotations

from typing import Any

from leakagelens.phase2.freeze_gate import EXPECTED_BASELINE, EXPECTED_VERSION
from leakagelens.phase2.models import scientific_model_params
from leakagelens.phase2.preprocessing import preprocessing_config
from leakagelens.phase2.protocols import PROTOCOL_GROUP, PROTOCOL_STRATIFIED, PROTOCOL_TEMPORAL
from leakagelens.utils.experiment import build_experiment_id, config_hash

_STRATEGY = {
    PROTOCOL_STRATIFIED: "stratified",
    PROTOCOL_GROUP: "group",
    PROTOCOL_TEMPORAL: "temporal",
    "combined_group_temporal": "temporal",
}


def build_run_config(
    *,
    dataset_id: str,
    dataset_name: str,
    provider: str,
    provider_dataset_id: int,
    provider_version: int,
    task_type: str,
    protocol: str,
    split_seed: int,
    model_seed: int,
    family: str,
    backend: str,
    model_params: dict[str, Any],
    group_column: str | None,
    time_column: str | None,
    smoke: bool,
) -> dict[str, Any]:
    params = scientific_model_params(family, model_params)
    return {
        "dataset": {
            "name": dataset_name,
            "source": provider,
            "task": task_type,
            "role": "clean_reference",
            "dataset_id": dataset_id,
            "version": str(provider_version),
            "params": {
                "provider_dataset_id": provider_dataset_id,
                "benchmark_manifest_version": EXPECTED_VERSION,
                "git_commit_at_freeze": EXPECTED_BASELINE,
                "preprocessing": preprocessing_config(family),
            },
        },
        "split": {
            "strategy": _STRATEGY[protocol],
            "test_size": 0.15,
            "validation_size": 0.15,
            "seed": int(split_seed),
            "group_column": group_column,
            "time_column": time_column,
            "params": {
                "protocol": protocol,
                "duplicate_policy": "exact_predictor_clusters_atomic_for_random_splits",
                "temporal_membership": "chronological_time_units",
            },
        },
        "model": {"family": family, "backend": backend, "params": params},
        "leakage": {
            "family": "none",
            "mechanism": "none",
            "severity": {
                "level": "none",
                "parameter_name": "none",
                "parameter_value": 0.0,
                "parameter_unit": "none",
            },
            "params": {},
        },
        "intervention": {"type": "none", "params": {}},
        "experiment": {
            "seed": int(model_seed),
            "output_dir": "artifacts/phase2/runs",
            "params": {"phase": "phase2", "smoke": bool(smoke)},
        },
    }


def run_identity(config: dict[str, Any]) -> tuple[str, str]:
    return build_experiment_id(config), config_hash(config)
