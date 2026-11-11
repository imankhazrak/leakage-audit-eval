"""Lightweight YAML configuration loading and validation.

Configs are nested mappings with fixed section names. Extra adapter-specific
keys belong under each section's ``params`` mapping, not as hidden globals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from leakagelens.constants import (
    FAMILY_ALLOWED_BACKENDS,
    DatasetRole,
    InterventionType,
    LeakageFamily,
    ModelFamily,
    SeverityLevel,
    SplitStrategy,
    TaskType,
)

REQUIRED_SECTIONS = (
    "dataset",
    "split",
    "model",
    "leakage",
    "intervention",
    "experiment",
)


class ConfigError(ValueError):
    """Invalid experiment configuration."""


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"Section '{name}' must be a mapping, got {type(value).__name__}")
    return value


def _require_str(section: dict[str, Any], key: str, section_name: str) -> str:
    if key not in section:
        raise ConfigError(f"Missing required field '{section_name}.{key}'")
    value = section[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Field '{section_name}.{key}' must be a non-empty string")
    return value.strip()


def _optional_str(section: dict[str, Any], key: str, section_name: str) -> str | None:
    if key not in section or section[key] is None:
        return None
    value = section[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Field '{section_name}.{key}' must be a non-empty string or null")
    return value.strip()


def _require_number(section: dict[str, Any], key: str, section_name: str) -> float:
    if key not in section:
        raise ConfigError(f"Missing required field '{section_name}.{key}'")
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"Field '{section_name}.{key}' must be a number")
    return float(value)


def _require_int(section: dict[str, Any], key: str, section_name: str) -> int:
    if key not in section:
        raise ConfigError(f"Missing required field '{section_name}.{key}'")
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"Field '{section_name}.{key}' must be an integer")
    return value


def _optional_mapping(section: dict[str, Any], key: str, section_name: str) -> dict[str, Any]:
    if key not in section or section[key] is None:
        return {}
    value = section[key]
    if not isinstance(value, dict):
        raise ConfigError(f"Field '{section_name}.{key}' must be a mapping")
    return dict(value)


def _reject_unknown_keys(
    section: dict[str, Any], allowed: set[str], section_name: str
) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ConfigError(f"Unknown keys in '{section_name}': {unknown}")


def _enum_value(enum_cls: type, raw: str, field_name: str) -> str:
    allowed = {item.value for item in enum_cls}
    if raw not in allowed:
        raise ConfigError(
            f"Unknown {field_name} '{raw}'. Allowed values: {sorted(allowed)}"
        )
    return raw


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    source: str
    task: str
    role: str | None = None
    dataset_id: str | None = None
    version: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DatasetConfig:
        _reject_unknown_keys(
            raw,
            {"name", "source", "task", "role", "dataset_id", "version", "params"},
            "dataset",
        )
        task = _enum_value(TaskType, _require_str(raw, "task", "dataset"), "dataset.task")
        role = _optional_str(raw, "role", "dataset")
        if role is not None:
            role = _enum_value(DatasetRole, role, "dataset.role")
        return cls(
            name=_require_str(raw, "name", "dataset"),
            source=_require_str(raw, "source", "dataset"),
            task=task,
            role=role,
            dataset_id=_optional_str(raw, "dataset_id", "dataset"),
            version=_optional_str(raw, "version", "dataset"),
            params=_optional_mapping(raw, "params", "dataset"),
        )


@dataclass(frozen=True)
class SplitConfig:
    strategy: str
    test_size: float
    validation_size: float
    seed: int
    group_column: str | None = None
    time_column: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SplitConfig:
        _reject_unknown_keys(
            raw,
            {
                "strategy",
                "test_size",
                "validation_size",
                "seed",
                "group_column",
                "time_column",
                "params",
            },
            "split",
        )
        strategy = _enum_value(
            SplitStrategy, _require_str(raw, "strategy", "split"), "split.strategy"
        )
        test_size = _require_number(raw, "test_size", "split")
        validation_size = _require_number(raw, "validation_size", "split")
        if not 0.0 < test_size < 1.0:
            raise ConfigError("split.test_size must be in the open interval (0, 1)")
        if not 0.0 <= validation_size < 1.0:
            raise ConfigError("split.validation_size must be in [0, 1)")
        if test_size + validation_size >= 1.0:
            raise ConfigError("split.test_size + split.validation_size must be < 1")
        seed = _require_int(raw, "seed", "split")
        if seed < 0:
            raise ConfigError("split.seed must be non-negative")
        if strategy == SplitStrategy.GROUP.value and not _optional_str(raw, "group_column", "split"):
            raise ConfigError("split.group_column is required when strategy is 'group'")
        if strategy == SplitStrategy.TEMPORAL.value and not _optional_str(raw, "time_column", "split"):
            raise ConfigError("split.time_column is required when strategy is 'temporal'")
        return cls(
            strategy=strategy,
            test_size=test_size,
            validation_size=validation_size,
            seed=seed,
            group_column=_optional_str(raw, "group_column", "split"),
            time_column=_optional_str(raw, "time_column", "split"),
            params=_optional_mapping(raw, "params", "split"),
        )


@dataclass(frozen=True)
class ModelConfig:
    family: str
    backend: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelConfig:
        _reject_unknown_keys(raw, {"family", "backend", "params"}, "model")
        family = _enum_value(
            ModelFamily, _require_str(raw, "family", "model"), "model.family"
        )
        backend = _require_str(raw, "backend", "model")
        allowed = FAMILY_ALLOWED_BACKENDS[family]
        if backend not in allowed:
            raise ConfigError(
                f"model.backend '{backend}' is not allowed for family '{family}'. "
                f"Allowed: {sorted(allowed)}"
            )
        return cls(
            family=family,
            backend=backend,
            params=_optional_mapping(raw, "params", "model"),
        )


@dataclass(frozen=True)
class SeverityConfig:
    """Mechanism-specific severity. Levels are not cross-family equivalents."""

    level: str
    parameter_name: str
    parameter_value: float
    parameter_unit: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SeverityConfig:
        _reject_unknown_keys(
            raw,
            {"level", "parameter_name", "parameter_value", "parameter_unit"},
            "leakage.severity",
        )
        level = _enum_value(
            SeverityLevel, _require_str(raw, "level", "leakage.severity"), "leakage.severity.level"
        )
        parameter_name = _require_str(raw, "parameter_name", "leakage.severity")
        parameter_value = _require_number(raw, "parameter_value", "leakage.severity")
        parameter_unit = _require_str(raw, "parameter_unit", "leakage.severity")
        if parameter_unit == "fraction" and not 0.0 <= parameter_value <= 1.0:
            raise ConfigError(
                "leakage.severity.parameter_value must be in [0, 1] when parameter_unit is 'fraction'"
            )
        if parameter_unit == "binary" and parameter_value not in (0.0, 1.0):
            raise ConfigError(
                "leakage.severity.parameter_value must be 0 or 1 when parameter_unit is 'binary'"
            )
        return cls(
            level=level,
            parameter_name=parameter_name,
            parameter_value=parameter_value,
            parameter_unit=parameter_unit,
        )


@dataclass(frozen=True)
class LeakageConfig:
    family: str
    mechanism: str
    severity: SeverityConfig
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LeakageConfig:
        _reject_unknown_keys(raw, {"family", "mechanism", "severity", "params"}, "leakage")
        if not isinstance(raw.get("severity"), dict):
            raise ConfigError(
                "leakage.severity must be a mapping with level, parameter_name, "
                "parameter_value, and parameter_unit. A bare number is not a severity."
            )
        family = _enum_value(
            LeakageFamily, _require_str(raw, "family", "leakage"), "leakage.family"
        )
        mechanism = _require_str(raw, "mechanism", "leakage")
        severity = SeverityConfig.from_dict(raw["severity"])
        _validate_clean_or_leaked(family, mechanism, severity)
        return cls(
            family=family,
            mechanism=mechanism,
            severity=severity,
            params=_optional_mapping(raw, "params", "leakage"),
        )


def _validate_clean_or_leaked(family: str, mechanism: str, severity: SeverityConfig) -> None:
    if family == LeakageFamily.NONE.value:
        if mechanism != "none":
            raise ConfigError("leakage.mechanism must be 'none' when leakage.family is 'none'")
        if severity.level != SeverityLevel.NONE.value:
            raise ConfigError("leakage.severity.level must be 'none' for a clean episode")
        if severity.parameter_name != "none":
            raise ConfigError("leakage.severity.parameter_name must be 'none' for a clean episode")
        if severity.parameter_value != 0.0:
            raise ConfigError(
                "leakage.severity.parameter_value must be 0 when leakage.family is 'none'"
            )
        if severity.parameter_unit != "none":
            raise ConfigError("leakage.severity.parameter_unit must be 'none' for a clean episode")
        return
    if mechanism == "none":
        raise ConfigError("leakage.mechanism must name the injection when leakage.family is not 'none'")
    if severity.level == SeverityLevel.NONE.value:
        raise ConfigError("leakage.severity.level must not be 'none' for a leaked episode")
    if severity.parameter_name == "none":
        raise ConfigError("leakage.severity.parameter_name is required for a leaked episode")
    if severity.parameter_unit == "none":
        raise ConfigError("leakage.severity.parameter_unit is required for a leaked episode")


@dataclass(frozen=True)
class InterventionConfig:
    type: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> InterventionConfig:
        _reject_unknown_keys(raw, {"type", "params"}, "intervention")
        intervention_type = _enum_value(
            InterventionType,
            _require_str(raw, "type", "intervention"),
            "intervention.type",
        )
        return cls(
            type=intervention_type,
            params=_optional_mapping(raw, "params", "intervention"),
        )


@dataclass(frozen=True)
class ExperimentSettings:
    seed: int
    output_dir: str
    name: str | None = None
    repeats: int = 1
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentSettings:
        _reject_unknown_keys(
            raw, {"seed", "output_dir", "name", "repeats", "params"}, "experiment"
        )
        seed = _require_int(raw, "seed", "experiment")
        if seed < 0:
            raise ConfigError("experiment.seed must be non-negative")
        repeats = raw.get("repeats", 1)
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            raise ConfigError("experiment.repeats must be a positive integer")
        output_dir = _require_str(raw, "output_dir", "experiment")
        if Path(output_dir).is_absolute():
            raise ConfigError(
                "experiment.output_dir must be a project-relative path; "
                "do not commit machine-specific absolute paths"
            )
        return cls(
            seed=seed,
            output_dir=output_dir,
            name=_optional_str(raw, "name", "experiment"),
            repeats=repeats,
            params=_optional_mapping(raw, "params", "experiment"),
        )


@dataclass(frozen=True)
class LeakageLensConfig:
    dataset: DatasetConfig
    split: SplitConfig
    model: ModelConfig
    leakage: LeakageConfig
    intervention: InterventionConfig
    experiment: ExperimentSettings
    source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "dataset": asdict(self.dataset),
            "split": asdict(self.split),
            "model": asdict(self.model),
            "leakage": asdict(self.leakage),
            "intervention": asdict(self.intervention),
            "experiment": asdict(self.experiment),
        }
        return payload

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], *, source_path: str | None = None
    ) -> LeakageLensConfig:
        if not isinstance(raw, dict):
            raise ConfigError("Configuration root must be a mapping")
        missing = [name for name in REQUIRED_SECTIONS if name not in raw]
        if missing:
            raise ConfigError(f"Missing required sections: {missing}")
        _reject_unknown_keys(raw, set(REQUIRED_SECTIONS), "root")
        return cls(
            dataset=DatasetConfig.from_dict(_require_mapping(raw["dataset"], "dataset")),
            split=SplitConfig.from_dict(_require_mapping(raw["split"], "split")),
            model=ModelConfig.from_dict(_require_mapping(raw["model"], "model")),
            leakage=LeakageConfig.from_dict(_require_mapping(raw["leakage"], "leakage")),
            intervention=InterventionConfig.from_dict(
                _require_mapping(raw["intervention"], "intervention")
            ),
            experiment=ExperimentSettings.from_dict(
                _require_mapping(raw["experiment"], "experiment")
            ),
            source_path=source_path,
        )


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        raise ConfigError(f"Configuration file is empty: {config_path}")
    if not isinstance(loaded, dict):
        raise ConfigError(f"Configuration root must be a mapping: {config_path}")
    return loaded


def load_config(path: str | Path) -> LeakageLensConfig:
    """Load and validate a LeakageLens YAML configuration file."""

    config_path = Path(path)
    raw = load_yaml(config_path)
    return LeakageLensConfig.from_dict(raw, source_path=str(config_path))
