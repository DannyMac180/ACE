import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_example_step(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_getting_started_example_runs_end_to_end(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'ace.db'}"
    env = {
        **os.environ,
        "ACE_DB_URL": db_url,
        "ACE_EMBEDDINGS": "mock",
        "ACE_LOG_LEVEL": "ERROR",
    }

    seed = run_example_step("scripts/seed.py", env=env)
    assert seed.returncode == 0, seed.stderr
    assert "Seeded 11 initial bullets" in seed.stdout

    retrieve = run_example_step(
        "-m",
        "ace.cli",
        "retrieve",
        "hybrid retrieval",
        "--top-k",
        "3",
        env=env,
    )
    assert retrieve.returncode == 0, retrieve.stderr
    assert "Found 3 bullets:" in retrieve.stdout
    assert "[strat-00001]" in retrieve.stdout

    reflection_path = tmp_path / "reflection.json"
    demo_reflect = run_example_step(
        "scripts/demo_reflect.py",
        "docs/assets/ace-proof-demo/demo-task.json",
        env=env,
    )
    assert demo_reflect.returncode == 0, demo_reflect.stderr
    reflection_path.write_text(demo_reflect.stdout, encoding="utf-8")

    curate = run_example_step(
        "-m",
        "ace.cli",
        "curate",
        "--reflection",
        str(reflection_path),
        "--json",
        env=env,
    )
    assert curate.returncode == 0, curate.stderr
    delta = json.loads(curate.stdout)
    assert delta["ops"]

    delta_path = tmp_path / "delta.json"
    delta_path.write_text(curate.stdout, encoding="utf-8")

    commit = run_example_step(
        "-m",
        "ace.cli",
        "commit",
        "--delta",
        str(delta_path),
        "--json",
        env=env,
    )
    assert commit.returncode == 0, commit.stderr
    assert json.loads(commit.stdout) == {"version": 1}

    stats = run_example_step("-m", "ace.cli", "stats", "--json", env=env)
    assert stats.returncode == 0, stats.stderr
    stats_payload = json.loads(stats.stdout)
    assert stats_payload["version"] == 1
    assert stats_payload["total_bullets"] == 12
    assert stats_payload["helpful"] == 2
    assert stats_payload["harmful"] == 0


def test_seed_script_accepts_legacy_relative_db_url_env(tmp_path):
    env = {
        **os.environ,
        "ACE_DB_URL": "ace.db",
        "ACE_EMBEDDINGS": "mock",
        "ACE_LOG_LEVEL": "ERROR",
    }

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "seed.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Seeded 11 initial bullets" in result.stdout
    assert (tmp_path / "ace.db").exists()
