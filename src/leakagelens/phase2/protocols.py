"""Choose the primary clean-split protocol from frozen applicability."""

from __future__ import annotations

PROTOCOL_STRATIFIED = "stratified_duplicate_safe"
PROTOCOL_GROUP = "group_safe"
PROTOCOL_TEMPORAL = "temporal"
PROTOCOL_COMBINED = "combined_group_temporal"

SUPPORTED = "supported"
CONDITIONAL = "conditional"


def select_clean_protocol(group_status: str, temporal_status: str) -> str:
    """Return the primary Phase-2 split protocol.

    Conditional applicability is not promoted to supported. A conditional group
    or time axis is stored for later experiments and does not change this choice.
    """

    group_supported = group_status == SUPPORTED
    temporal_supported = temporal_status == SUPPORTED
    if group_supported and temporal_supported:
        return PROTOCOL_COMBINED
    if group_supported:
        return PROTOCOL_GROUP
    if temporal_supported:
        return PROTOCOL_TEMPORAL
    return PROTOCOL_STRATIFIED
