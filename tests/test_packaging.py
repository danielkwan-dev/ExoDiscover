"""Guards against source files that exist locally but never reach a commit.

An unanchored ``data/`` line in .gitignore once matched the source package
``ml/exodiscover/data/`` as well as the intended top-level data directory. The
files were present on every developer machine, so lint, types, and the whole
test suite passed locally — and CI failed on a fresh clone with
``ModuleNotFoundError: No module named 'exodiscover.data'``. Nothing except a
clean checkout could see the difference, so the check belongs here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("ml", "api", "tests")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture(scope="module")
def tracked_files() -> set[str]:
    try:
        output = _git("ls-files")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("not a git checkout, or git unavailable")
    return set(output.splitlines())


def _source_files() -> list[str]:
    paths: list[str] = []
    for root in SOURCE_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            if "__pycache__" in path.parts or ".egg-info" in str(path):
                continue
            paths.append(path.relative_to(REPO_ROOT).as_posix())
    return paths


def test_no_source_file_is_hidden_by_gitignore() -> None:
    """No .py file under the source roots may match an ignore rule.

    This is the precise failure mode, and it is worth stating precisely: a file
    that is merely untracked is one `git add` from being committed and shows up
    in `git status`, whereas a file matched by .gitignore is invisible there and
    cannot be added without `-f`. That is how ml/exodiscover/data/ stayed out of
    six consecutive commits while every local check passed.
    """
    sources = _source_files()
    assert sources, "no source files found; SOURCE_ROOTS is probably wrong"

    # check-ignore exits 1 when nothing matches, which is the healthy case.
    proc = subprocess.run(
        ["git", "check-ignore", "--stdin", "--verbose"],
        cwd=REPO_ROOT,
        input="\n".join(sources),
        capture_output=True,
        text=True,
    )
    ignored = [line for line in proc.stdout.splitlines() if line.strip()]
    assert not ignored, (
        "source files are matched by .gitignore, so a fresh clone would not "
        "contain them. Each line names the offending rule:\n  "
        + "\n  ".join(ignored)
    )


def test_data_package_is_importable_from_the_index(tracked_files: set[str]) -> None:
    """The leakage contract and the splitters specifically must ship.

    These two modules carry the project's headline claims — no target-encoding
    column reaches the model, and no host star spans a split. A checkout
    missing them is a checkout that cannot substantiate either one.
    """
    required = {
        "ml/exodiscover/data/__init__.py",
        "ml/exodiscover/data/schema.py",
        "ml/exodiscover/data/splits.py",
        "ml/exodiscover/data/ingest.py",
    }
    missing = sorted(required - tracked_files)
    assert not missing, f"untracked: {missing}"


def test_raw_catalogs_stay_out_of_the_index(tracked_files: set[str]) -> None:
    """The counterpart guard: anchoring the pattern must not start committing
    the multi-megabyte NASA CSVs that the original rule existed to exclude."""
    committed_csvs = [
        path
        for path in tracked_files
        if path.startswith("data/") and path.endswith(".csv")
    ]
    assert not committed_csvs, f"raw catalogs must stay untracked: {committed_csvs}"
