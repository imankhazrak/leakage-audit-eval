"""SLURM allocation helpers.

Model parallelism uses ``SLURM_CPUS_PER_TASK`` rather than the machine-wide
processor count.
"""

from __future__ import annotations

import os


def allocated_cpus(default: int = 1) -> int:
    """Return the CPU count this process is allowed to use.

    Outside SLURM the default is 1 so a login-node import cannot fan out across
    the machine. Compute jobs export ``SLURM_CPUS_PER_TASK`` before Python starts.
    """

    raw = os.environ.get("SLURM_CPUS_PER_TASK")
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"SLURM_CPUS_PER_TASK must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"SLURM_CPUS_PER_TASK must be >= 1, got {value}")
    return value
