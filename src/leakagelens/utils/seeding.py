"""Random-seed control for reproducible experiments.

Two different controls exist:

* **Runtime-controllable** seeds can be set inside a running interpreter:
  ``random.seed``, NumPy's legacy seed, and estimator ``random_state`` values.
* **Startup-controlled** settings cannot. ``PYTHONHASHSEED`` is read when the
  interpreter starts. Assigning ``os.environ["PYTHONHASHSEED"]`` later does
  not make ``hash()`` deterministic in the current process.

Set ``PYTHONHASHSEED`` in the shell, in ``scripts/run_with_hashseed.sh``, or in
the Slurm template before Python starts.
"""

from __future__ import annotations

import os
import random

import numpy as np

PYTHONHASHSEED_ENV = "PYTHONHASHSEED"

RUNTIME_CONTROLLABLE_SEEDS = (
    "random",
    "numpy",
    "sklearn_random_state",
)
STARTUP_CONTROLLED_SETTINGS = (PYTHONHASHSEED_ENV,)

HASHSEED_STARTUP_NOTE = (
    "PYTHONHASHSEED is startup-controlled. It must be set before the Python "
    "interpreter starts. seed_everything does not assign it, because a runtime "
    "assignment does not make hash() deterministic in the current process."
)


def _validate_seed(seed: int) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    return seed


def seed_everything(seed: int) -> dict[str, int | str | None]:
    """Seed runtime generators. Does not set ``PYTHONHASHSEED``.

    scikit-learn, LightGBM, and XGBoost still need an explicit random-state
    argument on each estimator. This function does not touch GPU library state.
    """

    validated = _validate_seed(seed)
    random.seed(validated)
    np.random.seed(validated)
    return {
        "random": validated,
        "numpy": validated,
        "sklearn_random_state": validated,
        "pythonhashseed_at_startup": os.environ.get(PYTHONHASHSEED_ENV),
        "pythonhashseed_control": "startup",
        "pythonhashseed_note": HASHSEED_STARTUP_NOTE,
    }


def sklearn_random_state(seed: int) -> int:
    """Return the integer ``random_state`` to pass into scikit-learn APIs."""

    return _validate_seed(seed)


def hashseed_startup_status() -> dict[str, str | bool | None]:
    """Report whether this process was started with a fixed ``PYTHONHASHSEED``."""

    value = os.environ.get(PYTHONHASHSEED_ENV)
    deterministic = value not in (None, "", "random")
    return {
        "variable": PYTHONHASHSEED_ENV,
        "control": "startup",
        "value": value,
        "deterministic": deterministic,
        "note": HASHSEED_STARTUP_NOTE,
    }
