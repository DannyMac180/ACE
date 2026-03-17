import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_ls_files(path: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "ls-files", "--error-unmatch", path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_generated_artifacts_are_ignored_and_untracked():
    gitignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "ace.db" in gitignore_text
    assert "*.egg-info/" in gitignore_text

    assert _git_ls_files("ace.db").returncode != 0
    assert _git_ls_files("ace.egg-info/PKG-INFO").returncode != 0
