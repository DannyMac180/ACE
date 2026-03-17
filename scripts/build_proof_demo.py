#!/usr/bin/env python3
"""Build the ACE proof-demo assets from a reproducible offline CLI flow."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from html import escape
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "docs" / "assets" / "ace-proof-demo"
DOC_PATH = REPO_ROOT / "docs" / "ace-proof-demo.md"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

DEMO_TASK = {
    "query": "Improve the ACE retrieval wording for a CLI proof post demo",
    "retrieved_bullet_ids": ["strat-00001", "seed-retrieval-hygiene"],
    "code_diff": "",
    "test_output": "Initial retrieve surfaced one tactical bullet and one broad hygiene bullet.",
    "logs": "No runtime failure. Feedback focused on tightening the retrieval query.",
    "env_meta": {
        "final_status": "success",
        "source": "launch-proof-demo",
        "surface": "cli",
    },
}


def resolve_python_bin(repo_root: Path = REPO_ROOT) -> Path:
    """Prefer the repo virtualenv, but fall back to the active interpreter."""
    repo_python = repo_root / ".venv" / "bin" / "python"
    if repo_python.exists():
        return repo_python
    return Path(sys.executable).resolve()


def run_command(command: list[str], cwd: Path, env: dict[str, str]) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout
    if result.stderr:
        output = f"{output}\n{result.stderr}" if output else result.stderr
    return output.strip()


def sanitize_output(raw_output: str) -> str:
    clean_lines: list[str] = []
    for line in ANSI_RE.sub("", raw_output).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('{"level":'):
            continue
        if stripped.startswith("Warning: You are sending unauthenticated requests"):
            continue
        if stripped.startswith("Loading weights:"):
            continue
        if stripped.startswith("Batches:"):
            continue
        if stripped.startswith("BertModel LOAD REPORT"):
            continue
        if stripped.startswith("Key                     | Status"):
            continue
        if stripped.startswith("------------------------+"):
            continue
        if stripped.startswith("embeddings.position_ids"):
            continue
        if stripped.startswith("Notes:"):
            continue
        if stripped.startswith("- UNEXPECTED"):
            continue
        if "HTTP Request:" in stripped:
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines).strip()


def write_svg(output_path: Path) -> None:
    panels = [
        ("retrieve", '$ ace retrieve "hybrid retrieval" --top-k 3', "Found 3 bullets"),
        (
            "reflect",
            "$ python scripts/demo_reflect.py demo-task.json",
            "Generated strict Reflection JSON",
        ),
        ("curate", "$ ace curate --reflection reflection.json --json", "3 deterministic delta ops"),
        ("commit", "$ ace commit --delta delta.json --json", 'Version bumped to {"version": 1}'),
    ]

    width = 1200
    height = 760
    panel_width = 540
    panel_height = 150
    x_positions = [60, 600]
    y_positions = [70, 270]

    panel_elements: list[str] = []
    for index, (title, command, body) in enumerate(panels):
        x = x_positions[index % 2]
        y = y_positions[index // 2]
        panel_elements.append(
            f"""
  <g>
    <rect
      x="{x}"
      y="{y}"
      width="{panel_width}"
      height="{panel_height}"
      rx="18"
      fill="#111827"
      stroke="#1f2937"
      stroke-width="2"
    />
    <circle cx="{x + 26}" cy="{y + 24}" r="6" fill="#ef4444" />
    <circle cx="{x + 46}" cy="{y + 24}" r="6" fill="#f59e0b" />
    <circle cx="{x + 66}" cy="{y + 24}" r="6" fill="#10b981" />
    <text
      x="{x + 90}"
      y="{y + 29}"
      font-family="monospace"
      font-size="18"
      fill="#f9fafb"
    >{escape(title)}</text>
    <text
      x="{x + 24}"
      y="{y + 70}"
      font-family="monospace"
      font-size="18"
      fill="#93c5fd"
    >{escape(command)}</text>
    <text
      x="{x + 24}"
      y="{y + 110}"
      font-family="monospace"
      font-size="18"
      fill="#e5e7eb"
    >{escape(body)}</text>
  </g>
"""
        )

    svg = f"""<svg
  xmlns="http://www.w3.org/2000/svg"
  width="{width}"
  height="{height}"
  viewBox="0 0 {width} {height}"
  role="img"
  aria-labelledby="title desc"
>
  <title id="title">ACE proof demo still</title>
  <desc id="desc">
    Four terminal panels showing retrieve, reflect, curate, and commit in the ACE CLI flow.
  </desc>
  <defs>
    <linearGradient id="bg" x1="0%" x2="100%" y1="0%" y2="100%">
      <stop offset="0%" stop-color="#fff7ed" />
      <stop offset="100%" stop-color="#e0f2fe" />
    </linearGradient>
  </defs>
  <rect width="{width}" height="{height}" fill="url(#bg)" />
  <text x="60" y="48" font-family="Georgia, serif" font-size="32" fill="#111827">
    ACE proof post demo
  </text>
  <text x="60" y="708" font-family="monospace" font-size="18" fill="#374151">
    60-second terminal loop: retrieve -> reflect -> curate -> commit
  </text>
  {''.join(panel_elements)}
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def write_caption_script(output_path: Path) -> None:
    caption = textwrap.dedent(
        """\
        # ACE proof demo captions

        00:00-00:08
        Seed a throwaway playbook so the demo starts from a clean baseline.

        00:08-00:20
        Retrieve the most relevant bullets for a hybrid retrieval query and show
        the exact tactics ACE surfaces.

        00:20-00:33
        Reflect on the run with a strict JSON payload so the feedback stays
        reusable and machine-readable.

        00:33-00:44
        Curate the reflection into deterministic delta operations: two helpful
        votes and one new bullet.

        00:44-00:54
        Commit the delta to the playbook and bump the version without rewriting the whole prompt.

        00:54-01:00
        Show the updated stats so viewers can see the playbook grow by one
        bullet and record feedback immediately.
        """
    )
    output_path.write_text(caption, encoding="utf-8")


def write_explainer(output_path: Path) -> None:
    explainer = textwrap.dedent(
        """\
        # What this demo shows

        This asset demonstrates the core ACE loop on a clean local playbook:

        1. `retrieve` surfaces the bullets most likely to help with the task.
        2. `reflect` turns execution feedback into strict JSON instead of free-form prose.
        3. `curate` converts that reflection into deterministic delta operations.
        4. `commit` applies those deltas and bumps the playbook version.

        The clip is intentionally offline and reproducible. It uses the repo's
        real CLI, deterministic mock embeddings, a seeded local database, and a
        deterministic reflection helper so the proof asset can be rebuilt
        before launch without external API dependencies or model downloads.
        """
    )
    output_path.write_text(explainer, encoding="utf-8")


def _relative_doc_link(doc_path: Path, target_path: Path) -> str:
    """Return a markdown-friendly relative link from the doc to a target file."""
    return Path(os.path.relpath(target_path, start=doc_path.parent)).as_posix()


def write_doc(output_path: Path, asset_dir: Path) -> None:
    terminal_session_link = _relative_doc_link(
        output_path, asset_dir / "terminal-session.txt"
    )
    still_link = _relative_doc_link(output_path, asset_dir / "still.svg")
    caption_script_link = _relative_doc_link(output_path, asset_dir / "caption-script.md")
    what_you_see_link = _relative_doc_link(output_path, asset_dir / "what-you-see.md")
    demo_task_link = _relative_doc_link(output_path, asset_dir / "demo-task.json")
    reflection_link = _relative_doc_link(output_path, asset_dir / "reflection.json")
    delta_link = _relative_doc_link(output_path, asset_dir / "delta.json")
    doc = textwrap.dedent(
        f"""\
        # ACE proof demo asset

        This package is the reusable proof asset for launch posts scheduled the
        week of Monday, March 16, 2026.

        It captures a real ACE loop on a clean local playbook and keeps the
        output stable enough to reuse across X posts, README callouts, and docs.

        ## Files

        - [terminal-session.txt]({terminal_session_link})
        - [still.svg]({still_link})
        - [caption-script.md]({caption_script_link})
        - [what-you-see.md]({what_you_see_link})
        - [demo-task.json]({demo_task_link})
        - [reflection.json]({reflection_link})
        - [delta.json]({delta_link})

        ## Recording notes

        - The terminal transcript is generated from a throwaway workspace.
        - The demo forces `ACE_EMBEDDINGS=mock` so the offline path stays deterministic.
        - Noisy model-load logs are trimmed from the published transcript for readability.
        - The reflection step is deterministic so the asset can be rebuilt offline.
        """
    )
    output_path.write_text(doc, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory where generated assets will be written.",
    )
    parser.add_argument(
        "--doc-path",
        default=str(DOC_PATH),
        help="Markdown explainer path to write alongside the generated assets.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    doc_path = Path(args.doc_path).resolve()
    python_bin = resolve_python_bin()
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "ACE_DB_URL": "ace.db",
            "ACE_EMBEDDINGS": "mock",
            "ACE_LOG_LEVEL": "ERROR",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_VERBOSITY": "error",
        }
    )

    with tempfile.TemporaryDirectory(prefix="ace-proof-demo-") as workspace_str:
        workspace = Path(workspace_str)
        demo_task_path = workspace / "demo-task.json"
        demo_task_path.write_text(json.dumps(DEMO_TASK, indent=2), encoding="utf-8")

        commands = [
            (
                f"$ ACE_EMBEDDINGS=mock {python_bin} {REPO_ROOT / 'scripts' / 'seed.py'}",
                [str(python_bin), str(REPO_ROOT / "scripts" / "seed.py")],
                None,
            ),
            (
                f'$ ACE_DB_URL=ace.db ACE_EMBEDDINGS=mock {python_bin} -m ace.cli '
                'retrieve "hybrid retrieval" --top-k 3',
                [
                    str(python_bin),
                    "-m",
                    "ace.cli",
                    "retrieve",
                    "hybrid retrieval",
                    "--top-k",
                    "3",
                ],
                None,
            ),
            (
                f"$ {python_bin} {REPO_ROOT / 'scripts' / 'demo_reflect.py'} demo-task.json",
                [
                    str(python_bin),
                    str(REPO_ROOT / "scripts" / "demo_reflect.py"),
                    str(demo_task_path),
                ],
                output_dir / "reflection.json",
            ),
            (
                f"$ ACE_DB_URL=ace.db ACE_EMBEDDINGS=mock {python_bin} -m ace.cli "
                "curate --reflection reflection.json --json",
                [
                    str(python_bin),
                    "-m",
                    "ace.cli",
                    "curate",
                    "--reflection",
                    str(output_dir / "reflection.json"),
                    "--json",
                ],
                output_dir / "delta.json",
            ),
            (
                f"$ ACE_DB_URL=ace.db ACE_EMBEDDINGS=mock {python_bin} -m ace.cli "
                "commit --delta delta.json --json",
                [
                    str(python_bin),
                    "-m",
                    "ace.cli",
                    "commit",
                    "--delta",
                    str(output_dir / "delta.json"),
                    "--json",
                ],
                None,
            ),
            (
                f"$ ACE_DB_URL=ace.db ACE_EMBEDDINGS=mock {python_bin} -m ace.cli "
                "stats --json",
                [str(python_bin), "-m", "ace.cli", "stats", "--json"],
                None,
            ),
        ]

        transcript_parts: list[str] = []
        for display_command, command, write_path in commands:
            raw_output = run_command(command, cwd=workspace, env=env)
            if write_path is not None:
                write_path.write_text(raw_output + "\n", encoding="utf-8")
            transcript_parts.append(display_command)
            transcript_parts.append(sanitize_output(raw_output))

        (output_dir / "demo-task.json").write_text(
            json.dumps(DEMO_TASK, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "terminal-session.txt").write_text(
            "\n\n".join(part for part in transcript_parts if part).strip() + "\n",
            encoding="utf-8",
        )

    write_svg(output_dir / "still.svg")
    write_caption_script(output_dir / "caption-script.md")
    write_explainer(output_dir / "what-you-see.md")
    write_doc(doc_path, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
