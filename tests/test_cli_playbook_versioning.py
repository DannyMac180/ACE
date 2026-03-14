import json
import os
import subprocess
import sys


def _run_cli(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ace.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_playbook_history_and_dump_specific_version(tmp_path):
    db_path = tmp_path / "ace.db"
    playbook_path = tmp_path / "playbook.json"
    delta_path = tmp_path / "delta.json"

    playbook_path.write_text(
        json.dumps(
            {
                "version": 1,
                "bullets": [
                    {
                        "id": "strat-001",
                        "section": "strategies_and_hard_rules",
                        "content": "original content",
                        "tags": ["topic:test"],
                        "helpful": 0,
                        "harmful": 0,
                        "last_used": None,
                        "added_at": "2026-01-01T00:00:00+00:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    delta_path.write_text(
        json.dumps(
            {
                "ops": [
                    {
                        "op": "PATCH",
                        "target_id": "strat-001",
                        "patch": "updated content",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["ACE_DB_URL"] = f"sqlite:///{db_path}"
    env["ACE_EMBEDDINGS"] = "mock"

    import_result = _run_cli(["playbook", "import", "--file", str(playbook_path), "--json"], env)
    assert import_result.returncode == 0, import_result.stderr

    commit_result = _run_cli(["commit", "--delta", str(delta_path), "--json"], env)
    assert commit_result.returncode == 0, commit_result.stderr

    history_result = _run_cli(["playbook", "history", "--json"], env)
    assert history_result.returncode == 0, history_result.stderr
    history = json.loads(history_result.stdout)
    assert [entry["version"] for entry in history][:3] == [2, 1, 0]

    dump_result = _run_cli(["playbook", "dump", "--version", "1"], env)
    assert dump_result.returncode == 0, dump_result.stderr
    dumped_playbook = json.loads(dump_result.stdout)
    assert dumped_playbook["version"] == 1
    assert dumped_playbook["bullets"][0]["content"] == "original content"


def test_playbook_rollback_restores_previous_snapshot(tmp_path):
    db_path = tmp_path / "ace.db"
    playbook_path = tmp_path / "playbook.json"
    patch_delta_path = tmp_path / "patch.json"
    add_delta_path = tmp_path / "add.json"

    playbook_path.write_text(
        json.dumps(
            {
                "version": 1,
                "bullets": [
                    {
                        "id": "strat-001",
                        "section": "strategies_and_hard_rules",
                        "content": "before rollback",
                        "tags": ["topic:test"],
                        "helpful": 0,
                        "harmful": 0,
                        "last_used": None,
                        "added_at": "2026-01-01T00:00:00+00:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    patch_delta_path.write_text(
        json.dumps(
            {
                "ops": [
                    {
                        "op": "PATCH",
                        "target_id": "strat-001",
                        "patch": "after patch",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    add_delta_path.write_text(
        json.dumps(
            {
                "ops": [
                    {
                        "op": "ADD",
                        "new_bullet": {
                            "id": "strat-002",
                            "section": "troubleshooting_and_pitfalls",
                            "content": "added later",
                            "tags": ["topic:test"],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["ACE_DB_URL"] = f"sqlite:///{db_path}"
    env["ACE_EMBEDDINGS"] = "mock"

    import_result = _run_cli(
        ["playbook", "import", "--file", str(playbook_path), "--json"],
        env,
    )
    assert import_result.returncode == 0
    assert _run_cli(["commit", "--delta", str(patch_delta_path), "--json"], env).returncode == 0
    assert _run_cli(["commit", "--delta", str(add_delta_path), "--json"], env).returncode == 0

    rollback_result = _run_cli(["playbook", "rollback", "--version", "1", "--json"], env)
    assert rollback_result.returncode == 0, rollback_result.stderr
    rollback_data = json.loads(rollback_result.stdout)
    assert rollback_data == {"version": 1, "bullets_restored": 1}

    current_dump = _run_cli(["playbook", "dump"], env)
    assert current_dump.returncode == 0, current_dump.stderr
    current_playbook = json.loads(current_dump.stdout)

    assert current_playbook["version"] == 1
    assert [bullet["id"] for bullet in current_playbook["bullets"]] == ["strat-001"]
    assert current_playbook["bullets"][0]["content"] == "before rollback"
