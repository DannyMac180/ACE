import os
import subprocess
import sys


def test_refine_run_module_executes_documented_entrypoint(tmp_path):
    db_path = tmp_path / "ace.db"
    env = os.environ.copy()
    env["ACE_DB_URL"] = f"sqlite:///{db_path}"
    env["ACE_EMBEDDINGS"] = "mock"

    result = subprocess.run(
        [sys.executable, "-m", "ace.refine.run", "--threshold", "0.90"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "merged: 0" in result.stdout
    assert "archived: 0" in result.stdout
