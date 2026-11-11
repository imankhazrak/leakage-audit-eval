"""Experiment identifiers and lightweight provenance records."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from leakagelens.paths import get_project_root

if TYPE_CHECKING:
    from leakagelens.config import LeakageLensConfig

_ID_FIELDS = (
    "dataset.name",
    "dataset.source",
    "dataset.task",
    "dataset.role",
    "dataset.dataset_id",
    "dataset.version",
    "split.strategy",
    "split.test_size",
    "split.validation_size",
    "split.seed",
    "split.group_column",
    "split.time_column",
    "model.family",
    "model.backend",
    "leakage.family",
    "leakage.mechanism",
    "leakage.severity.level",
    "leakage.severity.parameter_name",
    "leakage.severity.parameter_value",
    "leakage.severity.parameter_unit",
    "intervention.type",
    "experiment.seed",
)


def _nested_get(mapping: dict[str, Any], dotted: str) -> Any:
    """Read a dotted field. Missing optional keys become ``None``."""

    current: Any = mapping
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def canonical_config_payload(config: LeakageLensConfig | dict[str, Any]) -> dict[str, Any]:
    """Return the subset of config fields that define an experiment identity."""

    data = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    payload = {field: _nested_get(data, field) for field in _ID_FIELDS}
    payload["split.params"] = data.get("split", {}).get("params", {})
    payload["model.params"] = data.get("model", {}).get("params", {})
    payload["leakage.params"] = data.get("leakage", {}).get("params", {})
    payload["intervention.params"] = data.get("intervention", {}).get("params", {})
    payload["dataset.params"] = data.get("dataset", {}).get("params", {})
    payload["experiment.params"] = data.get("experiment", {}).get("params", {})
    return payload


def canonical_json(payload: dict[str, Any]) -> str:
    """Serialize with sorted keys. Key order and timestamps are not part of identity."""

    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def config_hash(config: LeakageLensConfig | dict[str, Any]) -> str:
    """SHA-256 of the canonical identity payload. Not a timestamp."""

    return hashlib.sha256(canonical_json(canonical_config_payload(config)).encode("utf-8")).hexdigest()


def _stable_hash(payload: dict[str, Any]) -> str:
    return config_hash_from_payload(payload)[:10]


def config_hash_from_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def build_experiment_id(config: LeakageLensConfig | dict[str, Any]) -> str:
    """Build a filesystem-safe experiment ID from configuration identity fields.

    The hash is a prefix of ``config_hash`` and does not include timestamps or
    absolute paths. Dictionary key order does not change the ID.

    Severity is identified by parameter name and value, not by a bare number
    that could be compared across leakage families.
    """

    data = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    payload = canonical_config_payload(config)
    digest = _stable_hash(payload)
    severity = data["leakage"]["severity"]
    if not isinstance(severity, dict):
        raise TypeError("leakage.severity must be a mapping, not a bare number")
    severity_token = (
        f"{severity['parameter_name']}-{_format_number(severity['parameter_value'])}"
    )
    parts = [
        data["dataset"]["name"],
        data["model"]["family"],
        data["model"]["backend"],
        data["leakage"]["family"],
        data["leakage"]["mechanism"],
        severity_token,
        data["intervention"]["type"],
        f"seed{data['experiment']['seed']}",
        digest,
    ]
    return "_".join(_sanitize_token(part) for part in parts)


def _format_number(value: Any) -> str:
    number = float(value)
    text = format(number, ".6g")
    return text.replace("-", "m").replace(".", "p")


def _sanitize_token(value: str) -> str:
    allowed = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("-")
    token = "".join(allowed).strip("-_")
    return token or "x"


def git_commit_hash(cwd: str | Path | None = None) -> str | None:
    """Return HEAD commit hash, or ``None`` if Git is unavailable."""

    root = Path(cwd) if cwd is not None else get_project_root()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def git_worktree_dirty(cwd: str | Path | None = None) -> bool | None:
    """Return whether Git reports a dirty worktree, or None if Git is unavailable."""

    root = Path(cwd) if cwd is not None else get_project_root()
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def collect_provenance(
    config: LeakageLensConfig | dict[str, Any] | None = None,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect environment metadata for a future experiment record.

    Unavailable optional fields are ``None``. They are not filled with placeholders.
    """

    record: dict[str, Any] = {
        "experiment_id": None,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit_hash(),
        "git_dirty": git_worktree_dirty(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "package_version": _package_version(),
        "pythonhashseed_at_startup": os.environ.get("PYTHONHASHSEED"),
        "dataset_name": None,
        "dataset_id": None,
        "dataset_version": None,
        "dataset_role": None,
        "model_family": None,
        "model_backend": None,
        "model_parameters": None,
        "split_strategy": None,
        "split_seed": None,
        "experiment_seed": None,
        "leakage_family": None,
        "leakage_mechanism": None,
        "severity_level": None,
        "severity_parameter_name": None,
        "severity_parameter_value": None,
        "severity_parameter_unit": None,
        "intervention": None,
        "config_hash": None,
        "config": None,
    }
    if config is not None:
        data = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        dataset = data.get("dataset", {})
        model = data.get("model", {})
        leakage = data.get("leakage", {})
        severity = leakage.get("severity") if isinstance(leakage.get("severity"), dict) else {}
        split = data.get("split", {})
        experiment = data.get("experiment", {})
        record.update(
            {
                "experiment_id": build_experiment_id(config),
                "dataset_name": dataset.get("name"),
                "dataset_id": dataset.get("dataset_id"),
                "dataset_version": dataset.get("version"),
                "dataset_role": dataset.get("role"),
                "model_family": model.get("family"),
                "model_backend": model.get("backend"),
                "model_parameters": model.get("params"),
                "split_strategy": split.get("strategy"),
                "split_seed": split.get("seed"),
                "experiment_seed": experiment.get("seed"),
                "leakage_family": leakage.get("family"),
                "leakage_mechanism": leakage.get("mechanism"),
                "severity_level": severity.get("level"),
                "severity_parameter_name": severity.get("parameter_name"),
                "severity_parameter_value": severity.get("parameter_value"),
                "severity_parameter_unit": severity.get("parameter_unit"),
                "intervention": data.get("intervention", {}).get("type"),
                "config_hash": config_hash(config),
                "config": data,
            }
        )
    if extra:
        record.update(extra)
    return record


def _package_version() -> str:
    try:
        from leakagelens import __version__
    except Exception:
        return "unknown"
    return __version__
