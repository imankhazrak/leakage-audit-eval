"""Build deterministic splits and the Phase-2 run manifest. Does not fit models."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from leakagelens.logging_utils import get_logger
from leakagelens.paths import get_project_root
from leakagelens.phase2.catalog import load_model_specs, load_phase2_config, load_seeds
from leakagelens.phase2.dataset import load_dataset_bundle
from leakagelens.phase2.freeze_gate import EXPECTED_VERSION, verify_frozen_benchmark
from leakagelens.phase2.identity import build_run_config, run_identity
from leakagelens.phase2.jsonio import dump_json
from leakagelens.phase2.protocols import PROTOCOL_COMBINED, PROTOCOL_TEMPORAL
from leakagelens.phase2.rawcheck import selected_raw_checksums
from leakagelens.phase2.splits import STATUS_INFEASIBLE, CleanSplit, duplicate_cluster_ids, make_clean_split
from leakagelens.phase2.validation import validate_clean_split
from leakagelens.utils.experiment import git_commit_hash

LOGGER = get_logger("phase2.prepare")


def _index_hash(indices: list[int]) -> str:
    payload = ",".join(str(index) for index in indices)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _counts(target: pd.Series, indices: list[int]) -> dict[str, int]:
    if not indices:
        return {}
    values = target.iloc[indices].astype(str)
    return {str(key): int(value) for key, value in values.value_counts().sort_index().items()}


def split_record(
    split: CleanSplit,
    bundle: Any,
    *,
    benchmark_version: str,
) -> dict[str, Any]:
    train = [int(index) for index in split.train_idx.tolist()]
    validation = [int(index) for index in split.validation_idx.tolist()]
    test = [int(index) for index in split.test_idx.tolist()]
    record: dict[str, Any] = {
        "dataset": bundle.dataset_id,
        "dataset_name": bundle.name,
        "provider": bundle.provider,
        "provider_version": bundle.provider_version,
        "split_protocol": split.protocol,
        "split_seed": split.split_seed,
        "seed_affects_membership": split.seed_affects_membership,
        "status": split.status,
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "class_counts": {
            "train": _counts(bundle.target, train),
            "validation": _counts(bundle.target, validation),
            "test": _counts(bundle.target, test),
        },
        "train_index_hash": _index_hash(sorted(train)),
        "validation_index_hash": _index_hash(sorted(validation)),
        "test_index_hash": _index_hash(sorted(test)),
        "train_indices": train,
        "validation_indices": validation,
        "test_indices": test,
        "benchmark_version": benchmark_version,
        "git_commit": git_commit_hash(),
        "details": split.details,
    }
    if bundle.groups is not None and train:
        labels = bundle.groups.astype(str)
        record["group_counts"] = {
            "train": int(labels.iloc[train].nunique()),
            "validation": int(labels.iloc[validation].nunique()),
            "test": int(labels.iloc[test].nunique()),
        }
    if bundle.time is not None and train:
        values = bundle.time.astype(str)
        record["time_ranges"] = {
            "train": [values.iloc[train].min(), values.iloc[train].max()],
            "validation": [values.iloc[validation].min(), values.iloc[validation].max()],
            "test": [values.iloc[test].min(), values.iloc[test].max()],
        }
    if split.status == "success":
        clusters = duplicate_cluster_ids(bundle.features)
        xy = bundle.features.copy()
        xy["__y__"] = bundle.target.astype(str).to_numpy()
        xy_clusters = duplicate_cluster_ids(xy)

        def cross(ids: pd.Series, left: list[int], right: list[int]) -> int:
            return len(set(ids.iloc[left].astype(str)) & set(ids.iloc[right].astype(str)))

        record["duplicate_overlap"] = {
            "exact_X_train_test": cross(clusters, train, test),
            "exact_Xy_train_test": cross(xy_clusters, train, test),
        }
    return record


def _metadata_column(series: pd.Series | None) -> str | None:
    if series is None or series.name is None:
        return None
    return str(series.name)


def prepare_phase2(root: str | Path | None = None) -> dict[str, Any]:
    project = Path(root or get_project_root())
    freeze = verify_frozen_benchmark(project)
    checksums = selected_raw_checksums(project)
    checksum_path = project / "artifacts" / "phase2" / "raw_checksums_before.csv"
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    checksums.to_csv(checksum_path, index=False)
    if (checksums["sha256"] == "MISSING").any():
        missing = checksums.loc[checksums["sha256"] == "MISSING", "path"].tolist()
        raise RuntimeError(f"Missing raw caches: {missing}")
    config = load_phase2_config(project)
    seeds = load_seeds(project)
    specs = load_model_specs(project)
    manifest = yaml.safe_load((project / "configs/datasets/benchmark_manifest.yaml").read_text(encoding="utf-8"))
    dataset_ids = list(manifest["datasets"])
    split_root = project / "artifacts" / "phase2" / "splits"
    rows: list[dict[str, Any]] = []
    infeasible: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    task_id = 0
    for dataset_id in dataset_ids:
        bundle = load_dataset_bundle(dataset_id, root=project)
        LOGGER.info("Preparing %s protocol=%s", dataset_id, bundle.protocol)
        per_seed: dict[int, CleanSplit] = {}
        if bundle.protocol in {PROTOCOL_TEMPORAL, PROTOCOL_COMBINED}:
            base = make_clean_split(
                bundle.features,
                bundle.target,
                protocol=bundle.protocol,
                split_seed=seeds[0],
                groups=bundle.groups,
                time=bundle.time,
            )
            for seed in seeds:
                per_seed[seed] = replace(base, split_seed=seed)
        else:
            for seed in seeds:
                per_seed[seed] = make_clean_split(
                    bundle.features,
                    bundle.target,
                    protocol=bundle.protocol,
                    split_seed=seed,
                    groups=bundle.groups,
                    time=bundle.time,
                )
        for seed in seeds:
            split = per_seed[seed]
            record = split_record(split, bundle, benchmark_version=EXPECTED_VERSION)
            destination = split_root / dataset_id / f"{bundle.protocol}__seed{seed}.json"
            dump_json(destination, record)
            if split.status == "success":
                safety = validate_clean_split(
                    split,
                    n_rows=len(bundle.features),
                    features=bundle.features,
                    target=bundle.target,
                    groups=bundle.groups,
                    time=bundle.time,
                )
                validation_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "split_seed": seed,
                        "protocol": bundle.protocol,
                        "status": "pass",
                        "n_train": record["n_train"],
                        "n_validation": record["n_validation"],
                        "n_test": record["n_test"],
                        "group_overlap_train_test": safety["group_overlap"]["train_test"],
                        "temporal_violations": safety["temporal_violations"],
                        "exact_X_train_test": safety["duplicate_overlap"]["exact_X_train_test"],
                    }
                )
            else:
                infeasible.append({"dataset_id": dataset_id, "split_seed": seed, **split.details, "protocol": bundle.protocol})
                validation_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "split_seed": seed,
                        "protocol": bundle.protocol,
                        "status": STATUS_INFEASIBLE,
                        "n_train": 0,
                        "n_validation": 0,
                        "n_test": 0,
                        "group_overlap_train_test": None,
                        "temporal_violations": None,
                        "exact_X_train_test": None,
                    }
                )
        for family, spec in specs.items():
            for seed in seeds:
                split = per_seed[seed]
                enabled = split.status == "success"
                model_params = dict(spec.get("params") or {})
                run_config = build_run_config(
                    dataset_id=bundle.dataset_id,
                    dataset_name=bundle.name,
                    provider=bundle.provider,
                    provider_dataset_id=bundle.provider_dataset_id,
                    provider_version=bundle.provider_version,
                    task_type=bundle.task_type,
                    protocol=bundle.protocol,
                    split_seed=seed,
                    model_seed=seed,
                    family=family,
                    backend=str(spec["backend"]),
                    model_params=model_params,
                    group_column=_metadata_column(bundle.groups),
                    time_column=_metadata_column(bundle.time),
                    smoke=False,
                )
                experiment_id, _ = run_identity(run_config)
                output_dir = f"artifacts/phase2/runs/{experiment_id}"
                rows.append(
                    {
                        "task_id": task_id,
                        "experiment_id": experiment_id,
                        "dataset_id": bundle.dataset_id,
                        "dataset_name": bundle.name,
                        "provider": bundle.provider,
                        "provider_version": bundle.provider_version,
                        "model_family": family,
                        "model_backend": spec["backend"],
                        "split_protocol": bundle.protocol,
                        "split_seed": seed,
                        "model_seed": seed,
                        "config_path": "configs/phase2/phase2.yaml",
                        "output_dir": output_dir,
                        "enabled": enabled,
                        "benchmark_version": EXPECTED_VERSION,
                        "smoke": False,
                    }
                )
                task_id += 1
    frame = pd.DataFrame(rows)
    manifest_path = project / "configs" / "phase2" / "run_manifest.csv"
    frame.to_csv(manifest_path, index=False)
    enabled_ids = frame.loc[frame["enabled"] == True, "task_id"].astype(int).tolist()  # noqa: E712
    enabled_path = project / "artifacts" / "phase2" / "enabled_task_ids.txt"
    enabled_path.parent.mkdir(parents=True, exist_ok=True)
    enabled_path.write_text("\n".join(str(item) for item in enabled_ids) + "\n", encoding="utf-8")
    for row in rows:
        if not row["enabled"]:
            continue
        env_path = project / "artifacts" / "phase2" / "task_env" / f"{row['task_id']}.env"
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text(
            "\n".join(
                [
                    f"MODEL_SEED={row['model_seed']}",
                    f"SPLIT_SEED={row['split_seed']}",
                    f"DATASET_ID={row['dataset_id']}",
                    f"MODEL_FAMILY={row['model_family']}",
                    f"EXPERIMENT_ID={row['experiment_id']}",
                    f"BENCHMARK_VERSION={EXPECTED_VERSION}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    infeasible_path = project / "configs" / "phase2" / "infeasible_runs.csv"
    pd.DataFrame(infeasible).to_csv(infeasible_path, index=False)
    validation_path = project / "artifacts" / "phase2" / "split_validation.csv"
    pd.DataFrame(validation_rows).to_csv(validation_path, index=False)
    smoke_rows = _smoke_rows(project, config, specs)
    smoke_path = project / "configs" / "phase2" / "smoke_manifest.csv"
    pd.DataFrame(smoke_rows).to_csv(smoke_path, index=False)
    smoke_ids = project / "artifacts" / "phase2" / "smoke_task_ids.txt"
    smoke_ids.write_text("\n".join(str(row["task_id"]) for row in smoke_rows) + "\n", encoding="utf-8")
    for row in smoke_rows:
        env_path = project / "artifacts" / "phase2" / "smoke_task_env" / f"{row['task_id']}.env"
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text(
            "\n".join(
                [
                    f"MODEL_SEED={row['model_seed']}",
                    f"SPLIT_SEED={row['split_seed']}",
                    f"DATASET_ID={row['dataset_id']}",
                    f"MODEL_FAMILY={row['model_family']}",
                    f"EXPERIMENT_ID={row['experiment_id']}",
                    f"BENCHMARK_VERSION={EXPECTED_VERSION}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    summary = {
        "freeze": freeze,
        "datasets": len(dataset_ids),
        "manifest_rows": len(rows),
        "enabled": len(enabled_ids),
        "disabled": len(rows) - len(enabled_ids),
        "smoke_tasks": len(smoke_rows),
        "manifest_path": str(manifest_path.relative_to(project)),
    }
    dump_json(project / "artifacts" / "phase2" / "prepare_summary.json", summary)
    (project / "artifacts" / "phase2" / "prepare_ok").write_text("ok\n", encoding="utf-8")
    LOGGER.info("Prepared %s enabled runs of %s", len(enabled_ids), len(rows))
    return summary


def _smoke_rows(project: Path, config: dict[str, Any], specs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    smoke = config["smoke"]
    seed = int(smoke["seed"])
    rows = []
    for task_id, task in enumerate(smoke["tasks"]):
        dataset_id = task["dataset_id"]
        family = task["model_family"]
        bundle = load_dataset_bundle(dataset_id, root=project)
        spec = specs[family]
        params = dict(spec.get("params") or {})
        if family in {"random_forest", "gradient_boosting"}:
            params["n_estimators"] = int(smoke["n_estimators"])
        run_config = build_run_config(
            dataset_id=bundle.dataset_id,
            dataset_name=bundle.name,
            provider=bundle.provider,
            provider_dataset_id=bundle.provider_dataset_id,
            provider_version=bundle.provider_version,
            task_type=bundle.task_type,
            protocol=bundle.protocol,
            split_seed=seed,
            model_seed=seed,
            family=family,
            backend=str(spec["backend"]),
            model_params=params,
            group_column=_metadata_column(bundle.groups),
            time_column=_metadata_column(bundle.time),
            smoke=True,
        )
        experiment_id, _ = run_identity(run_config)
        rows.append(
            {
                "task_id": task_id,
                "experiment_id": experiment_id,
                "dataset_id": dataset_id,
                "dataset_name": bundle.name,
                "provider": bundle.provider,
                "provider_version": bundle.provider_version,
                "model_family": family,
                "model_backend": spec["backend"],
                "split_protocol": bundle.protocol,
                "split_seed": seed,
                "model_seed": seed,
                "config_path": "configs/phase2/phase2.yaml",
                "output_dir": f"artifacts/phase2/smoke_runs/{experiment_id}",
                "enabled": True,
                "benchmark_version": EXPECTED_VERSION,
                "smoke": True,
                "reason": task["reason"],
                "n_estimators": params.get("n_estimators"),
            }
        )
    return rows


def main() -> None:
    from leakagelens.logging_utils import configure_logging

    configure_logging("INFO")
    summary = prepare_phase2()
    LOGGER.info("prepare complete enabled=%s", summary["enabled"])


if __name__ == "__main__":
    main()
