"""Train-only preprocessing. Transformers are fit only when the pipeline is fit.

Logistic regression scales numeric columns. Tree models do not. Categorical
columns are imputed with the training mode and one-hot encoded with unknown
levels ignored. There is no resampling in the clean reference.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PREPROCESS_LR = "logistic_median_scale_onehot"
PREPROCESS_TREE = "tree_median_onehot"


def column_kinds(features: pd.DataFrame) -> dict[str, list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for name in features.columns:
        dtype = features[name].dtype
        if pd.api.types.is_bool_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(dtype):
            categorical.append(str(name))
        elif pd.api.types.is_numeric_dtype(dtype):
            numeric.append(str(name))
        else:
            categorical.append(str(name))
    return {"numeric": numeric, "categorical": categorical}


def build_preprocessor(model_family: str, features: pd.DataFrame) -> tuple[ColumnTransformer, str]:
    kinds = column_kinds(features)
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    config_name = PREPROCESS_TREE
    if model_family == "logistic_regression":
        numeric_steps.append(("scaler", StandardScaler()))
        config_name = PREPROCESS_LR
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric_pipe = Pipeline(steps=numeric_steps)
    transformers: list[tuple[str, Any, list[str]]] = []
    if kinds["numeric"]:
        transformers.append(("numeric", numeric_pipe, kinds["numeric"]))
    if kinds["categorical"]:
        transformers.append(("categorical", categorical_pipe, kinds["categorical"]))
    if not transformers:
        raise ValueError("No predictor columns available for preprocessing")
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, config_name


def preprocessing_config(model_family: str) -> dict[str, Any]:
    if model_family == "logistic_regression":
        return {
            "name": PREPROCESS_LR,
            "numeric": ["median_imputer", "standard_scaler"],
            "categorical": ["most_frequent_imputer", "one_hot_handle_unknown_ignore"],
            "resampling": None,
            "fit_scope": "train_only",
        }
    return {
        "name": PREPROCESS_TREE,
        "numeric": ["median_imputer"],
        "categorical": ["most_frequent_imputer", "one_hot_handle_unknown_ignore"],
        "resampling": None,
        "fit_scope": "train_only",
    }


def transformed_feature_map(preprocessor: ColumnTransformer) -> list[dict[str, str]]:
    """Map one-hot and numeric outputs back to original predictor names."""

    names = [str(name) for name in preprocessor.get_feature_names_out()]
    mapping: list[dict[str, str]] = []
    for name in names:
        if "__" not in name:
            mapping.append({"transformed": name, "original": name})
            continue
        block, remainder = name.split("__", 1)
        original = remainder
        if block == "categorical":
            # OneHotEncoder names are '<column>_<level>'. The longest matching
            # input column is the original predictor.
            inputs = []
            for _, transformer, columns in preprocessor.transformers_:
                if transformer == "drop" or columns is None:
                    continue
                inputs.extend(str(column) for column in list(columns))
            matches = [column for column in inputs if remainder == column or remainder.startswith(f"{column}_")]
            if matches:
                original = max(matches, key=len)
        mapping.append({"transformed": name, "original": original, "block": block})
    return mapping
