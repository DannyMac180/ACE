import json
import subprocess
import sys


def test_eval_run_can_write_baseline_file(tmp_path):
    baseline_path = tmp_path / "baseline.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ace.cli",
            "eval",
            "run",
            "--suite",
            "retrieval",
            "--json",
            "--write-baseline",
            str(baseline_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    output = json.loads(result.stdout)
    assert output["suite"] == "retrieval"

    baseline = json.loads(baseline_path.read_text())
    assert baseline["retrieval_cases"] == 5.0
    assert baseline["retrieval.cases_run"] == 5.0


def test_eval_run_can_write_markdown_report(tmp_path):
    report_path = tmp_path / "report.md"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ace.cli",
            "eval",
            "run",
            "--suite",
            "retrieval",
            "--format",
            "markdown",
            "--out",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert report_path.exists()

    report = report_path.read_text()
    assert "# ACE Evaluation Results" in report
    assert "**Suite:** `retrieval`" in report
    assert "### Retrieval" in report
    assert "#### Results" in report
