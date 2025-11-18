import json

from ace.eval.harness import EvalRunner


def test_harness_regression_check(tmp_path):
    """Test that harness correctly uses RegressionDetector with baseline."""
    runner = EvalRunner()

    # Create a baseline file
    baseline = {
        "retrieval.mrr": 0.85,
        "latency": 100.0
    }
    baseline_path = tmp_path / "baseline.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline, f)

    # Case 1: No regression
    results_good = {
        "suite": "retrieval",
        "summary": {
            "retrieval.mrr": 0.86, # Improved
            "latency": 95.0        # Improved (lower is better)
        }
    }

    check_good = runner._check_regression(results_good, str(baseline_path), fail_on_regression=True)
    assert check_good["status"] == "no_regression"

    # Case 2: Regression detected
    results_bad = {
        "suite": "retrieval",
        "summary": {
            "retrieval.mrr": 0.70, # Regression (>10% drop)
            "latency": 150.0       # Regression (>20% increase)
        }
    }

    check_bad = runner._check_regression(results_bad, str(baseline_path), fail_on_regression=False)
    assert check_bad["status"] == "regression_detected"
    assert len(check_bad["regressions"]) >= 2
    assert "failed" not in check_bad

    # Case 3: Fail on regression
    check_fail = runner._check_regression(results_bad, str(baseline_path), fail_on_regression=True)
    assert check_fail["status"] == "regression_detected"
    assert check_fail.get("failed") is True
