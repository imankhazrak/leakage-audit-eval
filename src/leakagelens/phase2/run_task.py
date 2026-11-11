"""Execute exactly one Phase-2 manifest row. Model fitting happens here."""

from __future__ import annotations

import argparse
import os
import socket
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from leakagelens.logging_utils import configure_logging, get_logger
from leakagelens.paths import get_project_root
from leakagelens.phase2.catalog import load_model_specs, load_phase2_config
from leakagelens.phase2.dataset import DatasetBundle, load_dataset_bundle
from leakagelens.phase2.freeze_gate import EXPECTED_VERSION, verify_frozen_benchmark
from leakagelens.phase2.identity import build_run_config, run_identity
from leakagelens.phase2.jsonio import dump_json
from leakagelens.phase2.metrics import classification_metrics, generalization_gaps
from leakagelens.phase2.models import PHASE2_GBDT_BACKEND, build_estimator
from leakagelens.phase2.preprocessing import build_preprocessor, preprocessing_config, transformed_feature_map
from leakagelens.phase2.prepare import _metadata_column
from leakagelens.phase2.splits import make_clean_split
from leakagelens.phase2.validation import CleanSafetyError, validate_clean_split, validate_predictor_matrix
from leakagelens.utils.experiment import collect_provenance
from leakagelens.utils.seeding import seed_everything

LOGGER = get_logger("phase2.run")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _load_row(manifest: Path, task_id: int) -> dict[str, Any]:
    frame = pd.read_csv(manifest)
    matched = frame.loc[frame["task_id"] == task_id]
    if matched.empty:
        raise KeyError(f"task_id {task_id} is not in {manifest}")
    return matched.iloc[0].to_dict()


def _model_params(family: str, smoke: bool, root: Path) -> tuple[str, dict[str, Any]]:
    spec = load_model_specs(root)[family]
    params = dict(spec.get("params") or {})
    if smoke and family in {"random_forest", "gradient_boosting"}:
        params["n_estimators"] = int(load_phase2_config(root)["smoke"]["n_estimators"])
    return str(spec["backend"]), params


def _original_importance(mapping: list[dict[str, str]], values: np.ndarray) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row, value in zip(mapping, values):
        original = row["original"]
        totals[original] = totals.get(original, 0.0) + float(value)
    return totals


def _feature_summary(pipeline: Pipeline, family: str) -> dict[str, Any]:
    preprocessor = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    mapping = transformed_feature_map(preprocessor)
    summary: dict[str, Any] = {
        "transformed_feature_count": len(mapping),
        "transformed_to_original": mapping,
    }
    if family == "logistic_regression":
        coef = np.mean(np.abs(estimator.coef_), axis=0)
        summary["coefficient_mean_absolute_by_original"] = _original_importance(mapping, coef)
    elif family == "random_forest":
        summary["impurity_importance_by_original"] = _original_importance(mapping, estimator.feature_importances_)
    elif family == "gradient_boosting":
        summary["gain_importance_by_original"] = _original_importance(mapping, estimator.feature_importances_)
        booster = estimator.booster_
        summary["split_importance_by_original"] = _original_importance(
            mapping, np.asarray(booster.feature_importance(importance_type="split"), dtype=float)
        )
    return summary


def _fit_scope_report(pipeline: Pipeline, features: pd.DataFrame, train_idx: np.ndarray) -> dict[str, Any]:
    """Confirm imputer statistics match the training partition, not the full matrix."""

    preprocessor = pipeline.named_steps["preprocess"]
    train = features.iloc[train_idx]
    checked: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if transformer == "drop" or not hasattr(transformer, "named_steps"):
            continue
        imputer = transformer.named_steps.get("imputer")
        if imputer is None:
            continue
        column_list = [str(column) for column in list(columns)]
        if imputer.strategy == "median":
            expected = train[column_list].median(numeric_only=True).to_numpy(dtype=float)
            actual = np.asarray(imputer.statistics_, dtype=float)
            if expected.shape != actual.shape or not np.allclose(actual, expected, equal_nan=True):
                raise CleanSafetyError(f"{name} imputer statistics do not match the training median")
        elif imputer.strategy == "most_frequent":
            actual = list(imputer.statistics_)
            if len(actual) != len(column_list):
                raise CleanSafetyError(f"{name} imputer width does not match training columns")
            for got, column in zip(actual, column_list):
                observed = train[column].dropna()
                if observed.empty:
                    if not pd.isna(got):
                        raise CleanSafetyError(f"{name} imputer filled an empty training column")
                    continue
                if str(got) not in set(observed.astype(str)):
                    raise CleanSafetyError(
                        f"{name} imputer statistic for {column} is absent from the training partition"
                    )
        else:
            raise CleanSafetyError(f"Unexpected imputer strategy {imputer.strategy}")
        checked.append(str(name))
    if not checked:
        raise CleanSafetyError("No fitted imputer was found")
    return {
        "preprocessing_fit_scope": "train_only",
        "imputers_checked": checked,
        "n_train": int(len(train_idx)),
        "statistics_match_training_partition": True,
    }


def _prediction_frame(
    indices: np.ndarray,
    y_true: pd.Series,
    probabilities: np.ndarray,
    classes: list[Any],
    *,
    partition: str,
    experiment_id: str,
) -> pd.DataFrame:
    pred = [classes[int(index)] for index in np.argmax(probabilities, axis=1)]
    frame = pd.DataFrame(
        {
            "row_index": indices.astype(int),
            "y_true": y_true.iloc[indices].astype(str).to_numpy(),
            "y_pred": [str(label) for label in pred],
            "partition": partition,
            "experiment_id": experiment_id,
        }
    )
    for position, label in enumerate(classes):
        frame[f"proba_{position}"] = probabilities[:, position]
    frame.attrs["class_order"] = [str(label) for label in classes]
    return frame


def _write_failure(output_dir: Path, payload: dict[str, Any]) -> None:
    dump_json(output_dir / "failure.json", payload)


def execute_task(manifest: Path, task_id: int, *, root: Path | None = None) -> None:
    project = Path(root or get_project_root())
    freeze = verify_frozen_benchmark(project)
    row = _load_row(manifest, task_id)
    if not _as_bool(row["enabled"]):
        raise CleanSafetyError(f"task {task_id} is disabled and must not be trained")
    smoke = _as_bool(row["smoke"])
    output_dir = project / str(row["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    family = str(row["model_family"])
    backend, params = _model_params(family, smoke, project)
    bundle: DatasetBundle = load_dataset_bundle(str(row["dataset_id"]), root=project)
    split_seed = int(row["split_seed"])
    model_seed = int(row["model_seed"])
    run_config = build_run_config(
        dataset_id=bundle.dataset_id,
        dataset_name=bundle.name,
        provider=bundle.provider,
        provider_dataset_id=bundle.provider_dataset_id,
        provider_version=bundle.provider_version,
        task_type=bundle.task_type,
        protocol=str(row["split_protocol"]),
        split_seed=split_seed,
        model_seed=model_seed,
        family=family,
        backend=backend,
        model_params=params,
        group_column=_metadata_column(bundle.groups),
        time_column=_metadata_column(bundle.time),
        smoke=smoke,
    )
    experiment_id, config_digest = run_identity(run_config)
    if experiment_id != str(row["experiment_id"]):
        raise RuntimeError(f"Experiment id mismatch: manifest {row['experiment_id']} rebuilt {experiment_id}")
    predictor_report = validate_predictor_matrix(bundle.features, bundle.roles, target_name=bundle.target_name)
    split = make_clean_split(
        bundle.features,
        bundle.target,
        protocol=str(row["split_protocol"]),
        split_seed=split_seed,
        groups=bundle.groups,
        time=bundle.time,
    )
    safety = validate_clean_split(
        split,
        n_rows=len(bundle.features),
        features=bundle.features,
        target=bundle.target,
        groups=bundle.groups,
        time=bundle.time,
    )
    safety.update(predictor_report)
    dump_json(output_dir / "clean_pipeline_validation.json", safety)
    seed_everything(model_seed)
    preprocessor, preprocess_name = build_preprocessor(family, bundle.features)
    estimator = build_estimator(family, backend, params, model_seed)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    train_idx = split.train_idx
    started = time.perf_counter()
    pipeline.fit(bundle.features.iloc[train_idx], bundle.target.iloc[train_idx])
    fit_seconds = time.perf_counter() - started
    scope = _fit_scope_report(pipeline, bundle.features, train_idx)
    safety.update(scope)
    dump_json(output_dir / "clean_pipeline_validation.json", safety)
    classes = [item for item in pipeline.named_steps["model"].classes_]
    partition_metrics: dict[str, Any] = {}
    predict_seconds = 0.0
    frames: dict[str, pd.DataFrame] = {}
    for partition, indices in (
        ("train", split.train_idx),
        ("validation", split.validation_idx),
        ("test", split.test_idx),
    ):
        clock = time.perf_counter()
        probabilities = pipeline.predict_proba(bundle.features.iloc[indices])
        predict_seconds += time.perf_counter() - clock
        # predict_proba is aligned to the scored rows, not the full frame.
        local_target = bundle.target.iloc[indices].reset_index(drop=True)
        metrics = classification_metrics(local_target.to_numpy(), probabilities, classes)
        partition_metrics[partition] = metrics
        if partition != "train":
            frames[partition] = _prediction_frame(
                indices,
                bundle.target,
                probabilities,
                classes,
                partition=partition,
                experiment_id=experiment_id,
            )
    import lightgbm

    provenance = collect_provenance(
        run_config,
        extra={
            "benchmark_manifest_version": EXPECTED_VERSION,
            "benchmark_baseline_commit": freeze["git_commit_at_freeze"],
            "provider": bundle.provider,
            "provider_version": bundle.provider_version,
            "split_protocol": row["split_protocol"],
            "model_seed": model_seed,
            "preprocessing_config": preprocessing_config(family),
            "preprocessing_name": preprocess_name,
            "config_hash_sha256": config_digest,
            "sklearn_version": sklearn.__version__,
            "lightgbm_version": lightgbm.__version__ if backend == PHASE2_GBDT_BACKEND else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "hostname": socket.gethostname(),
            "smoke": smoke,
            "status": "success",
        },
    )
    (output_dir / "config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    dump_json(output_dir / "metrics.json", {"partitions": partition_metrics, "gaps": generalization_gaps(partition_metrics), "smoke": smoke, "status": "success"})
    dump_json(output_dir / "provenance.json", provenance)
    dump_json(
        output_dir / "split_summary.json",
        {
            "protocol": split.protocol,
            "split_seed": split_seed,
            "model_seed": model_seed,
            "seed_affects_membership": split.seed_affects_membership,
            "n_train": int(len(split.train_idx)),
            "n_validation": int(len(split.validation_idx)),
            "n_test": int(len(split.test_idx)),
            "details": split.details,
        },
    )
    dump_json(output_dir / "preprocessing_summary.json", {"config": preprocessing_config(family), "fit_scope": scope})
    dump_json(
        output_dir / "training_summary.json",
        {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "n_train": int(len(split.train_idx)),
            "n_validation": int(len(split.validation_idx)),
            "n_test": int(len(split.test_idx)),
            "input_predictor_count": int(bundle.features.shape[1]),
            "post_transform_feature_count": int(pipeline.named_steps["preprocess"].get_feature_names_out().shape[0]),
            "model_family": family,
            "model_backend": backend,
            "n_classes": len(classes),
            "class_order": [str(label) for label in classes],
            "smoke": smoke,
            "early_stopping": False,
        },
    )
    dump_json(output_dir / "feature_summary.json", _feature_summary(pipeline, family))
    frames["validation"].to_parquet(output_dir / "predictions_validation.parquet", index=False)
    frames["test"].to_parquet(output_dir / "predictions_test.parquet", index=False)
    dump_json(output_dir / "class_order.json", {"class_order": [str(label) for label in classes]})
    LOGGER.info("Finished %s", experiment_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Phase-2 clean-reference task")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-id", required=True, type=int)
    args = parser.parse_args()
    configure_logging("INFO")
    project = get_project_root()
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = project / manifest
    output_hint = project / "artifacts" / "phase2" / "failures"
    try:
        row = _load_row(manifest, args.task_id)
        output_hint = project / str(row["output_dir"])
        execute_task(manifest, args.task_id, root=project)
    except Exception as exc:
        status = "clean_validation_failed" if isinstance(exc, CleanSafetyError) else "fit_failed"
        message = str(exc)
        if "split" in message.lower() and not isinstance(exc, CleanSafetyError):
            status = "split_failed"
        payload = {
            "status": status,
            "exception_type": type(exc).__name__,
            "message": message,
            "traceback": traceback.format_exc(),
            "task_id": args.task_id,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        }
        try:
            _write_failure(output_hint, payload)
        except Exception:
            LOGGER.exception("Could not write failure record")
        LOGGER.error("Task %s failed: %s", args.task_id, message)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
