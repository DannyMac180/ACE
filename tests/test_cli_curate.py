import json
import subprocess
import sys


def test_curate_cli_accepts_reflection_json_file(tmp_path):
    reflection_path = tmp_path / "reflection.json"
    reflection_path.write_text(
        json.dumps(
            {
                "error_identification": None,
                "root_cause_analysis": None,
                "correct_approach": "Tighten retrieval queries.",
                "key_insight": "Specific task terms improve retrieval quality.",
                "bullet_tags": [
                    {"id": "strat-00001", "tag": "helpful"},
                    {"id": "seed-retrieval-hygiene", "tag": "harmful"},
                ],
                "candidate_bullets": [
                    {
                        "section": "strategies_and_hard_rules",
                        "content": "Use concrete task and tool names in retrieval queries.",
                        "tags": ["topic:retrieval", "tool:cli"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ace.cli",
            "curate",
            "--reflection",
            str(reflection_path),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    output = json.loads(result.stdout)
    assert [op["op"] for op in output["ops"]] == ["INCR_HELPFUL", "INCR_HARMFUL", "ADD"]
