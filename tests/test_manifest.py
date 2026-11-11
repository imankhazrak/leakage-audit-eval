"""The experiment tree matches its recorded hashes."""
from __future__ import annotations

import importlib.util

import pytest


def _verify(repo):
    spec = importlib.util.spec_from_file_location("verify_manifest", repo / "scripts" / "verify_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_covers_the_tree(repo) -> None:
    listed = {name for _, name in _verify(repo).entries()}
    present = {p.relative_to(repo).as_posix() for p in (repo / "research").rglob("*") if p.is_file()}
    present.discard("research/MANIFEST.sha256")
    assert listed == present


def test_manifest_sample(repo) -> None:
    n, bad = _verify(repo).check(sample=500)
    assert n == 500 and bad == []


@pytest.mark.slow
def test_manifest_full(repo) -> None:
    n, bad = _verify(repo).check()
    assert bad == []
