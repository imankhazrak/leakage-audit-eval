"""Phase 3 controlled leakage episodes and behavioral fingerprints.

This package plans and executes single-family leakage episodes against the
frozen Phase-2 clean references. It does not train LeakageLens detectors.
"""

MECHANISM_VERSION = "phase3-v1"
PROXY_COLUMN = "__ll_target_proxy__"
CANONICAL_DETERMINISTIC_SEED = 7
