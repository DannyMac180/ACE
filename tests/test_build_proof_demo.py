import filecmp
import subprocess
import sys
from pathlib import Path


def _run_build(output_dir: Path, doc_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/build_proof_demo.py",
            "--output-dir",
            str(output_dir),
            "--doc-path",
            str(doc_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_build_proof_demo_is_reproducible_for_repeated_runs(tmp_path):
    output_a = tmp_path / "proof-a"
    output_b = tmp_path / "proof-b"
    doc_a = tmp_path / "docs" / "proof-a.md"
    doc_b = tmp_path / "docs" / "proof-b.md"

    first_result = _run_build(output_a, doc_a)
    second_result = _run_build(output_b, doc_b)

    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 0, second_result.stderr

    compared = filecmp.dircmp(output_a, output_b)
    assert compared.left_only == []
    assert compared.right_only == []
    assert compared.diff_files == []


def test_build_proof_demo_supports_output_dir_outside_repo(tmp_path):
    output_dir = tmp_path / "outside-repo-assets"
    doc_path = tmp_path / "outside-repo-docs" / "proof-demo.md"

    result = _run_build(output_dir, doc_path)

    assert result.returncode == 0, result.stderr
    assert (output_dir / "delta.json").exists()
    assert doc_path.exists()
    doc_text = doc_path.read_text(encoding="utf-8")
    assert "outside-repo-assets/terminal-session.txt" in doc_text
