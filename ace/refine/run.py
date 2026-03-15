"""Module entrypoint for the documented ``python -m ace.refine.run`` command."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ace.cli import cmd_refine
from ace.core.config import load_config
from ace.core.logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone parser for the refine module entrypoint."""
    parser = argparse.ArgumentParser(
        prog="python -m ace.refine.run",
        description="Deduplicate and consolidate ACE playbook bullets.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Similarity threshold for dedup",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without applying changes",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the refine command through the documented module entrypoint."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    config = load_config()
    configure_logging(config.logging.level, config.logging.format)
    cmd_refine(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
