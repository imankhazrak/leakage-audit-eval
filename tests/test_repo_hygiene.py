"""No machine paths, credentials or source tables in the repository."""
from __future__ import annotations

import re

TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".cff", ".sh", ".sbatch", ".txt", ".def", ".ipynb", ".tex"}
MACHINE_PATH = re.compile(r"(/users/[A-Za-z0-9_]+/|/home/[A-Za-z0-9_]+/|/fs/(ess|scratch)/)")
SECRET = re.compile(r"(BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16})")
# A frozen test in the experiment tree asserts that a path does NOT contain "/users/".
ALLOWED = {"research/leakagelens_two_week_validation/local_freeze_study_2026-10-27/corrected/test_acceptance.py"}


def _text_files(repo):
    for path in repo.rglob("*"):
        if path.is_file() and path.suffix in TEXT_SUFFIXES and ".git" not in path.parts and "/runs/" not in path.as_posix():
            yield path


def test_no_machine_paths_or_secrets(repo) -> None:
    offenders = []
    for path in _text_files(repo):
        rel = path.relative_to(repo).as_posix()
        if rel in ALLOWED or rel == "tests/test_repo_hygiene.py":
            continue
        text = path.read_text(errors="ignore")
        if MACHINE_PATH.search(text) or SECRET.search(text):
            offenders.append(rel)
    assert offenders == []


def test_no_source_tables_committed(repo) -> None:
    assert not list(repo.rglob("*.parquet"))
    assert not (repo / "data" / "raw").exists() or not any((repo / "data" / "raw").iterdir())


def test_template_layout(repo) -> None:
    for name in ["README.md", "LICENSE", "CITATION.cff", ".gitignore", "pyproject.toml", "data/README.md",
                 "results/README.md", ".github/workflows/tests.yml"]:
        assert (repo / name).is_file(), name
    for name in ["configs", "src/leakagelens", "scripts", "notebooks", "tests", "docs"]:
        assert (repo / name).is_dir(), name
