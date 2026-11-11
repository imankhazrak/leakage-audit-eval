"""Phase-2 leakage-safe clean reference pipelines.

This package builds clean splits, train-only preprocessing, and fixed model
baselines. It does not inject leakage and it does not train detectors.
"""

from leakagelens.phase2.protocols import (
    PROTOCOL_COMBINED,
    PROTOCOL_GROUP,
    PROTOCOL_STRATIFIED,
    PROTOCOL_TEMPORAL,
    select_clean_protocol,
)

__all__ = [
    "PROTOCOL_COMBINED",
    "PROTOCOL_GROUP",
    "PROTOCOL_STRATIFIED",
    "PROTOCOL_TEMPORAL",
    "select_clean_protocol",
]
