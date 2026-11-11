"""Dataset-normalized behavioral fingerprints.

Fingerprint construction consumes paired suspected/corrected diagnostics.
It must not peek at detector-level holdout datasets.
"""

from leakagelens.fingerprints.schema import (
    FINGERPRINT_BLOCKS,
    METRIC_SPECS,
    oriented_delta,
    raw_delta,
)

__all__ = ["FINGERPRINT_BLOCKS", "METRIC_SPECS", "oriented_delta", "raw_delta"]
