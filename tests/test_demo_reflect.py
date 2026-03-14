import json
import subprocess
import sys


def test_demo_reflect_outputs_valid_reflection_json(tmp_path):
    doc_path = tmp_path / "demo-task.json"
    doc_path.write_text(
        json.dumps(
            {
                "query": "Improve ACE retrieval wording",
                "retrieved_bullet_ids": ["strat-00001", "seed-retrieval-hygiene"],
                "code_diff": "",
                "test_output": "Initial retrieval was broad.",
                "logs": "",
                "env_meta": {"final_status": "success"},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "scripts/demo_reflect.py", str(doc_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    reflection = json.loads(result.stdout)
    assert reflection["bullet_tags"][0]["id"] == "strat-00001"
    assert reflection["candidate_bullets"][0]["tags"] == [
        "repo:ace",
        "topic:retrieval",
        "tool:cli",
    ]
