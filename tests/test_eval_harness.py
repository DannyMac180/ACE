"""Tests for ace.eval.harness module"""

import json
from pathlib import Path

import pytest

from ace.eval.harness import EvalRunner


class TestFixtureLoader:
    """Test fixture loading functionality"""

    def test_load_valid_fixture(self, tmp_path: Path) -> None:
        """Test loading a valid fixture file"""
        fixture_file = tmp_path / "test_fixture.json"
        fixture_data = {"cases": [{"id": "test-1", "data": "value"}]}
        fixture_file.write_text(json.dumps(fixture_data))

        # Use tmp_path as fixtures directory
        original_file = Path(EvalRunner._load_fixture.__code__.co_filename)
        fixture_path = original_file.parent / "fixtures" / "test_fixture.json"
        fixture_path.parent.mkdir(exist_ok=True)
        fixture_path.write_text(json.dumps(fixture_data))

        try:
            result = EvalRunner._load_fixture("test_fixture.json")
            assert result == [{"id": "test-1", "data": "value"}]
        finally:
            if fixture_path.exists():
                fixture_path.unlink()

    def test_load_missing_file(self) -> None:
        """Test error handling for missing fixture file"""
        with pytest.raises(FileNotFoundError, match="Fixture file not found"):
            EvalRunner._load_fixture("nonexistent_fixture.json")

    def test_load_malformed_json(self, tmp_path: Path) -> None:
        """Test error handling for malformed JSON"""
        fixture_file = tmp_path / "malformed.json"
        fixture_file.write_text("{invalid json")

        original_file = Path(EvalRunner._load_fixture.__code__.co_filename)
        fixture_path = original_file.parent / "fixtures" / "malformed.json"
        fixture_path.write_text("{invalid json")

        try:
            with pytest.raises(ValueError, match="Malformed JSON"):
                EvalRunner._load_fixture("malformed.json")
        finally:
            if fixture_path.exists():
                fixture_path.unlink()

    def test_load_fixture_missing_cases_key(self, tmp_path: Path) -> None:
        """Test error handling when 'cases' key is missing"""
        fixture_data = {"data": []}
        original_file = Path(EvalRunner._load_fixture.__code__.co_filename)
        fixture_path = original_file.parent / "fixtures" / "no_cases.json"
        fixture_path.write_text(json.dumps(fixture_data))

        try:
            with pytest.raises(ValueError, match="must contain a 'cases' key"):
                EvalRunner._load_fixture("no_cases.json")
        finally:
            if fixture_path.exists():
                fixture_path.unlink()

    def test_load_fixture_cases_not_list(self, tmp_path: Path) -> None:
        """Test error handling when 'cases' is not a list"""
        fixture_data = {"cases": "not a list"}
        original_file = Path(EvalRunner._load_fixture.__code__.co_filename)
        fixture_path = original_file.parent / "fixtures" / "bad_cases.json"
        fixture_path.write_text(json.dumps(fixture_data))

        try:
            with pytest.raises(ValueError, match="'cases' .* must be a list"):
                EvalRunner._load_fixture("bad_cases.json")
        finally:
            if fixture_path.exists():
                fixture_path.unlink()

    def test_load_fixture_not_dict(self, tmp_path: Path) -> None:
        """Test error handling when fixture is not a JSON object"""
        original_file = Path(EvalRunner._load_fixture.__code__.co_filename)
        fixture_path = original_file.parent / "fixtures" / "not_dict.json"
        fixture_path.write_text("[]")

        try:
            with pytest.raises(ValueError, match="must contain a JSON object"):
                EvalRunner._load_fixture("not_dict.json")
        finally:
            if fixture_path.exists():
                fixture_path.unlink()


class TestEvalRunner:
    """Test EvalRunner initialization and methods"""

    def test_init_loads_retrieval_fixtures(self) -> None:
        """Test that EvalRunner loads retrieval fixtures on init"""
        runner = EvalRunner()
        assert isinstance(runner.retrieval_fixtures, list)
        assert runner.retrieval_fixtures == []  # Empty by default

    def test_run_method_exists(self) -> None:
        """Test that run method is callable"""
        runner = EvalRunner()
        runner.run(suite="all")  # Should not raise
