"""Main evaluation harness for ACE system benchmarks"""

import json
from pathlib import Path
from typing import Literal

SuiteType = Literal["retrieval", "reflection", "e2e", "all"]


class EvalRunner:
    """Orchestrates evaluation benchmarks for the ACE system"""

    def __init__(self) -> None:
        """Initialize the EvalRunner"""
        self.retrieval_fixtures = self._load_fixture("retrieval_cases.json")

    @staticmethod
    def _load_fixture(filename: str) -> list[dict]:
        """Load fixture data from JSON file

        Args:
            filename: Name of the fixture file in the fixtures/ directory

        Returns:
            List of test case dictionaries

        Raises:
            FileNotFoundError: If fixture file doesn't exist
            ValueError: If fixture file is malformed or missing 'cases' key
        """
        fixture_path = Path(__file__).parent / "fixtures" / filename

        if not fixture_path.exists():
            raise FileNotFoundError(
                f"Fixture file not found: {fixture_path}. "
                f"Expected fixture at ace/eval/fixtures/{filename}"
            )

        try:
            with fixture_path.open(encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Malformed JSON in fixture file {filename}: {e}"
            ) from e

        if not isinstance(data, dict):
            raise ValueError(
                f"Fixture file {filename} must contain a JSON object, got {type(data).__name__}"
            )

        if "cases" not in data:
            raise ValueError(
                f"Fixture file {filename} must contain a 'cases' key with list of test cases"
            )

        cases = data["cases"]
        if not isinstance(cases, list):
            raise ValueError(
                f"'cases' in {filename} must be a list, got {type(cases).__name__}"
            )

        return cases

    def run(self, suite: str = "all") -> None:
        """Run evaluation suite

        Args:
            suite: Which benchmark suite to run. Options: 'retrieval', 'reflection', 'e2e', 'all'
        """
        print(f"EvalRunner initialized. Attempting to run suite: {suite}")
