"""Read-only check that Phase 2 is using the frozen Phase-1 benchmark."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml

from leakagelens.paths import get_project_root

EXPECTED_VERSION = "1.0.0"
EXPECTED_BASELINE = "a72cda7de9fd620b7a4557493be8bc258a3eebb3"
EXPECTED_FREEZE_COMMIT = "08228714f583e9c0c2e62e4bad3cd41e61b53fab"
EXPECTED_TAG = "leakagelens-phase1-v1.0.0"
MANIFEST_RELATIVE = Path("configs/datasets/benchmark_manifest.yaml")


class FreezeGateError(RuntimeError):
    pass


def manifest_path(root: str | Path | None = None) -> Path:
    return Path(root or get_project_root()) / MANIFEST_RELATIVE


def load_frozen_manifest(root: str | Path | None = None) -> dict[str, Any]:
    path = manifest_path(root)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FreezeGateError("Benchmark manifest root must be a mapping")
    return payload


def _git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def verify_frozen_benchmark(root: str | Path | None = None) -> dict[str, Any]:
    """Fail closed unless the approved freeze metadata is present.

    This function only reads the manifest and Git. It does not rewrite Phase 1.
    """

    project = Path(root or get_project_root())
    manifest = load_frozen_manifest(project)
    version = manifest.get("benchmark_manifest_version")
    status = manifest.get("freeze_status")
    baseline = manifest.get("git_commit_at_freeze")
    problems: list[str] = []
    if version != EXPECTED_VERSION:
        problems.append(f"benchmark_manifest_version={version!r}")
    if status != "frozen":
        problems.append(f"freeze_status={status!r}")
    if baseline != EXPECTED_BASELINE:
        problems.append(f"git_commit_at_freeze={baseline!r}")
    tag_commit = _git(project, "rev-parse", f"{EXPECTED_TAG}^{{commit}}")
    if tag_commit is not None and tag_commit != EXPECTED_FREEZE_COMMIT:
        problems.append(f"tag {EXPECTED_TAG} resolves to {tag_commit}")
    if problems:
        raise FreezeGateError("Phase-1 freeze gate failed: " + "; ".join(problems))
    return {
        "benchmark_manifest_version": version,
        "freeze_status": status,
        "git_commit_at_freeze": baseline,
        "freeze_commit": EXPECTED_FREEZE_COMMIT,
        "freeze_tag": EXPECTED_TAG,
        "freeze_date": manifest.get("freeze_date"),
        "tag_commit": tag_commit,
    }
