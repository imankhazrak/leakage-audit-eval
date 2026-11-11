"""Shared enumerations for configuration validation.

These names are the Prompt-0 contract for configs and experiment IDs.
Algorithms that use them are implemented in later phases.
"""

from __future__ import annotations

from enum import Enum


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"


class SplitStrategy(str, Enum):
    STRATIFIED = "stratified"
    GROUP = "group"
    TEMPORAL = "temporal"
    RANDOM = "random"


class ModelFamily(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"


class ModelBackend(str, Enum):
    """Implementation behind a model family.

    ``UNSPECIFIED`` is allowed only for gradient boosting until a backend is
    chosen and recorded. It is not a claim that backends are interchangeable.
    """

    SKLEARN = "sklearn"
    SKLEARN_HIST_GRADIENT_BOOSTING = "sklearn_hist_gradient_boosting"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    UNSPECIFIED = "unspecified"


FAMILY_ALLOWED_BACKENDS: dict[str, frozenset[str]] = {
    ModelFamily.LOGISTIC_REGRESSION.value: frozenset({ModelBackend.SKLEARN.value}),
    ModelFamily.RANDOM_FOREST.value: frozenset({ModelBackend.SKLEARN.value}),
    ModelFamily.GRADIENT_BOOSTING.value: frozenset(
        {
            ModelBackend.UNSPECIFIED.value,
            ModelBackend.SKLEARN_HIST_GRADIENT_BOOSTING.value,
            ModelBackend.LIGHTGBM.value,
            ModelBackend.XGBOOST.value,
        }
    ),
}


class LeakageFamily(str, Enum):
    """Initial LeakageBench taxonomy.

    ``NONE`` denotes a clean episode. It is a Detector A negative class, not a
    Detector B family label.
    """

    NONE = "none"
    OVERLAP = "overlap"
    GROUP = "group"
    TEMPORAL = "temporal"
    PREPROCESSING = "preprocessing"
    TARGET_PROXY = "target_proxy"


class SeverityLevel(str, Enum):
    """Within-family experimental category.

    ``low`` overlap is not claimed to equal ``low`` proxy leakage.
    """

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class ApplicabilityStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONDITIONAL = "conditional"


class DatasetRole(str, Enum):
    """Planned dataset roles. Assignment of real datasets is a Phase 1 decision."""

    DEVELOPMENT = "development"
    DETECTOR_TRAIN = "detector_train"
    DETECTOR_VALIDATION = "detector_validation"
    DETECTOR_TEST = "detector_test"
    SPECIALIZED_GROUP = "specialized_group"
    SPECIALIZED_TEMPORAL = "specialized_temporal"
    EXTERNAL_ROBUSTNESS = "external_robustness"
    SYNTHETIC_SMOKE = "synthetic_smoke"


class MetricOrientation(str, Enum):
    """How to read a metric when forming an oriented delta.

    ``UNSIGNED`` means the metric has no agreed better-direction. Store the raw
    delta and leave the oriented delta unset.
    """

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    UNSIGNED = "unsigned"


class InterventionType(str, Enum):
    """Corrective interventions. ``NONE`` means no correction is applied."""

    NONE = "none"
    REMOVE_OVERLAP = "remove_overlap"
    GROUP_AWARE_SPLIT = "group_aware_split"
    TEMPORAL_SPLIT = "temporal_split"
    PIPELINE_PREPROCESS = "pipeline_preprocess"
    DROP_PROXY_FEATURES = "drop_proxy_features"


class EvaluationProtocol(str, Enum):
    DATASET_DISJOINT = "dataset_disjoint"
    DATASET_KFOLD = "dataset_kfold"
    LEAVE_ONE_FAMILY_OUT = "leave_one_family_out"
    SEVERITY_TRANSFER = "severity_transfer"
    MODEL_FAMILY_TRANSFER = "model_family_transfer"


KNOWN_LEAKAGE_FAMILIES = tuple(
    family for family in LeakageFamily if family is not LeakageFamily.NONE
)
KNOWN_MODEL_FAMILIES = tuple(ModelFamily)
KNOWN_EVALUATION_PROTOCOLS = tuple(EvaluationProtocol)

# Detector A is binary. Detector B is a conditional family classifier on leaked
# episodes only. ``none`` is intentionally absent from Detector B.
DETECTOR_A_LABELS = ("clean", "leaked")
DETECTOR_B_LABELS = tuple(family.value for family in KNOWN_LEAKAGE_FAMILIES)

# Primary LeakageBench episodes contain one leakage family. Mixed leakage is a
# later robustness study, not an initial family label.
PRIMARY_EPISODE_SCOPE = "single_leakage_family"
MIXED_LEAKAGE_IN_PRIMARY_SCOPE = False

# Qualitative severity labels organize experiments inside one family.
# They are not a universal leaked-information scale.
SEVERITY_LEVELS_ARE_CROSS_FAMILY_COMPARABLE = False
