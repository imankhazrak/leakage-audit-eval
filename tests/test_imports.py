"""The library subset used by the experiments imports cleanly."""
from __future__ import annotations

import importlib

import pytest

MODULES = [
    "leakagelens",
    "leakagelens.interventions",
    "leakagelens.interventions.pipeline",
    "leakagelens.interventions.rules",
    "leakagelens.interventions.execute",
    "leakagelens.interventions.episode_decision",
    "leakagelens.phase2.preprocessing",
    "leakagelens.data.openml_adapter",
    "leakagelens.utils.seeding",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_project_root_is_this_repository(repo) -> None:
    from leakagelens.paths import get_project_root

    assert get_project_root() == repo
