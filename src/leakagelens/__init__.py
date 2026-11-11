"""LeakageLens: refit-based (behavioral) audits of tabular pipelines.

This is the subset of the library used by the paper's experiments: the
label-blind corrective edits and their execution by refitting
(:mod:`leakagelens.interventions`) and the modules they depend on.
"""

from __future__ import annotations

from leakagelens.config import LeakageLensConfig, load_config
from leakagelens.paths import get_project_root

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "LeakageLensConfig",
    "load_config",
    "get_project_root",
]
