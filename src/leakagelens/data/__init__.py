"""Dataset adapters, manifests, and DatasetSpec loading.

Responsibilities (later phases):
    - load public tabular datasets through registered adapters
    - freeze dataset identity, version, and task metadata
    - expose grouping and temporal columns when they exist

This package must not fit models, inject leakage, or compute fingerprints.
"""

from leakagelens.constants import TaskType

__all__ = ["TaskType"]
