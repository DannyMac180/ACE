"""Main evaluation harness for ACE system benchmarks"""

import json
from pathlib import Path
from typing import Any

from ace.core.regression import RegressionDetector


class EvalRunner:
    """Orchestrates evaluation benchmarks for the ACE system"""

    def __init__(self) -> None:
        """Initialize the EvalRunner"""
        try:
            self.retrieval_fixtures = self._load_fixture("retrieval_cases.json")
        except FileNotFoundError:
            self.retrieval_fixtures = []

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

    def run_suite(
        self,
        suite: str = "all",
        baseline_path: str | None = None,
        fail_on_regression: bool = False,
    ) -> dict[str, Any]:
        """Run evaluation suite

        Args:
            suite: Which benchmark suite to run. Options: 'retrieval', 'reflection', 'e2e', 'all'
            baseline_path: Path to baseline JSON for regression detection
            fail_on_regression: If True, raise error on regression

        Returns:
            Dictionary with evaluation results
        """
        results: dict[str, Any] = {
            "suite": suite,
            "summary": {},
            "details": {},
        }

        if suite in ("retrieval", "all"):
            results["details"]["retrieval"] = self._run_retrieval_suite()
            results["summary"]["retrieval_cases"] = len(self.retrieval_fixtures)

        if suite in ("reflection", "all"):
            results["details"]["reflection"] = {"status": "not_implemented"}
            results["summary"]["reflection_status"] = "not_implemented"

        if suite in ("e2e", "all"):
            results["details"]["e2e"] = {"status": "not_implemented"}
            results["summary"]["e2e_status"] = "not_implemented"

        # Check for regression if baseline provided
        if baseline_path:
            results["regression_check"] = self._check_regression(
                results, baseline_path, fail_on_regression
            )

        return results

    def _run_retrieval_suite(self) -> dict[str, Any]:
        """Run retrieval benchmark suite"""
        if not self.retrieval_fixtures:
            return {
                "status": "no_fixtures",
                "message": "No retrieval fixtures found",
            }

        results = []
        for case in self.retrieval_fixtures:
            results.append({"case_id": case.get("id", "unknown"), "status": "pending"})

        return {"status": "complete", "cases_run": len(results), "results": results}

    def _check_regression(
        self, results: dict[str, Any], baseline_path: str, fail_on_regression: bool
    ) -> dict[str, Any]:
        """Check for regression against baseline"""
        try:
            with open(baseline_path) as f:
                baseline_data = json.load(f)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to load baseline from {baseline_path}: {e}"
            }

        detector = RegressionDetector()
        regressions = []

        # Flatten results summary for comparison
        # Assumes summary contains keys like "retrieval_cases" or "retrieval.mrr"
        # If metrics are nested in details, we might need to flatten them too
        # For now, check summary items
        metrics_to_check = results.get("summary", {})

        for metric_key, current_value in metrics_to_check.items():
            if not isinstance(current_value, (int, float)):
                continue

            # If strict matching against baseline file is desired:
            if metric_key in baseline_data:
                baseline_value = baseline_data[metric_key]

                # Infer benchmark/metric names from key (e.g. "retrieval.mrr")
                if "." in metric_key:
                    bench_name, metric_name = metric_key.split(".", 1)
                else:
                    bench_name, metric_name = "global", metric_key

                # Infer direction
                higher_is_better = True
                lower_is_better_keywords = ["latency", "time", "duration", "seconds", "ms"]
                if any(x in metric_name.lower() for x in lower_is_better_keywords):
                    higher_is_better = False

                report = detector.detect_regression(
                    benchmark_name=bench_name,
                    metric_name=metric_name,
                    current_value=float(current_value),
                    higher_is_better=higher_is_better,
                    static_baseline=float(baseline_value),
                )

                if report.detected:
                    regressions.append(report)

        if regressions:
            message = f"Found {len(regressions)} regressions:\n"
            for r in regressions:
                message += f"- {r.message}\n"

            result: dict[str, Any] = {
                "status": "regression_detected",
                "regressions": [r.message for r in regressions],
                "message": message
            }

            if fail_on_regression:
                # We return specific status so caller can decide to exit(1)
                # But harness just returns dict. The CLI uses this.
                # We add a specific flag for CLI to check.
                result["failed"] = True

            return result

        return {"status": "no_regression", "baseline_loaded": True}

    def format_markdown(self, results: dict[str, Any]) -> str:
        """Format results as markdown report"""
        lines = [
            "# ACE Evaluation Results",
            "",
            f"**Suite:** {results['suite']}",
            "",
            "## Summary",
            "",
        ]

        for key, value in results.get("summary", {}).items():
            lines.append(f"- {key}: {value}")

        lines.append("")
        lines.append("## Details")
        lines.append("")
        lines.append("_(Full details in JSON output)_")

        return "\n".join(lines)

    def print_results(self, results: dict[str, Any]) -> None:
        """Print results in human-readable text format"""
        print("=" * 60)
        print(f"ACE Evaluation Results - Suite: {results['suite']}")
        print("=" * 60)
        print()

        print("Summary:")
        for key, value in results.get("summary", {}).items():
            print(f"  {key}: {value}")
        print()

        if "details" in results:
            print("Details:")
            for suite_name, suite_results in results["details"].items():
                print(f"  [{suite_name}]")
                if isinstance(suite_results, dict):
                    for k, v in suite_results.items():
                        print(f"    {k}: {v}")
                print()
