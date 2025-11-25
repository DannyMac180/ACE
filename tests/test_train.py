"""Tests for multi-epoch training module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from ace.core.schema import Delta, DeltaOp, Playbook
from ace.reflector.schema import CandidateBullet, Reflection
from ace.train.runner import TrainingRunner
from ace.train.schema import (
    EpochMetadata,
    SampleEpochRecord,
    TrainingSample,
    TrainingState,
)


class TestTrainingSample:
    """Tests for TrainingSample schema."""

    def test_minimal_sample(self):
        sample = TrainingSample(id="s1", query="test query")
        assert sample.id == "s1"
        assert sample.query == "test query"
        assert sample.retrieved_bullet_ids == []
        assert sample.code_diff == ""

    def test_full_sample(self):
        sample = TrainingSample(
            id="s1",
            query="test query",
            retrieved_bullet_ids=["b1", "b2"],
            code_diff="+ new line",
            test_output="OK",
            logs="info log",
            env_meta={"key": "value"},
            success=True,
        )
        assert sample.retrieved_bullet_ids == ["b1", "b2"]
        assert sample.success is True


class TestTrainingState:
    """Tests for TrainingState tracking."""

    def test_empty_state(self):
        state = TrainingState()
        assert state.current_epoch == 0
        assert state.total_epochs == 1
        assert len(state.epochs) == 0
        assert len(state.sample_records) == 0

    def test_get_processed_samples(self):
        state = TrainingState()
        state.sample_records.append(
            SampleEpochRecord(sample_id="s1", epoch=1, ops_applied=2)
        )
        state.sample_records.append(
            SampleEpochRecord(sample_id="s2", epoch=1, ops_applied=1)
        )
        state.sample_records.append(
            SampleEpochRecord(sample_id="s1", epoch=2, ops_applied=0)
        )

        epoch1 = state.get_processed_samples_for_epoch(1)
        assert epoch1 == {"s1", "s2"}

        epoch2 = state.get_processed_samples_for_epoch(2)
        assert epoch2 == {"s1"}

        epoch3 = state.get_processed_samples_for_epoch(3)
        assert epoch3 == set()

    def test_record_sample(self):
        state = TrainingState()
        state.record_sample(
            sample_id="s1",
            epoch=1,
            ops_applied=3,
            version_before=5,
            version_after=6,
        )

        assert len(state.sample_records) == 1
        record = state.sample_records[0]
        assert record.sample_id == "s1"
        assert record.epoch == 1
        assert record.ops_applied == 3
        assert record.playbook_version_before == 5
        assert record.playbook_version_after == 6

    def test_start_and_complete_epoch(self):
        state = TrainingState()
        state.start_epoch(1, playbook_version=10)

        assert len(state.epochs) == 1
        assert state.epochs[0].epoch == 1
        assert state.epochs[0].playbook_version_start == 10
        assert state.epochs[0].completed_at is None

        state.complete_epoch(
            epoch=1, samples_processed=5, ops_applied=10, playbook_version=12
        )

        assert state.epochs[0].samples_processed == 5
        assert state.epochs[0].total_ops_applied == 10
        assert state.epochs[0].playbook_version_end == 12
        assert state.epochs[0].completed_at is not None


class TestTrainingRunner:
    """Tests for TrainingRunner."""

    @pytest.fixture
    def mock_store(self):
        store = MagicMock()
        playbook = Playbook(version=1, bullets=[])
        store.load_playbook.return_value = playbook
        return store

    @pytest.fixture
    def mock_reflector(self):
        reflector = MagicMock()
        reflector.reflect.return_value = Reflection(
            error_identification=None,
            root_cause_analysis=None,
            correct_approach=None,
            key_insight="Test insight",
            bullet_tags=[],
            candidate_bullets=[
                CandidateBullet(
                    section="strategies",
                    content="Test strategy",
                    tags=["test"],
                )
            ],
        )
        return reflector

    @pytest.fixture
    def sample_jsonl(self, tmp_path):
        data = tmp_path / "samples.jsonl"
        samples = [
            {"id": "s1", "query": "Query 1", "test_output": "PASS"},
            {"id": "s2", "query": "Query 2", "test_output": "FAIL", "success": False},
            {"id": "s3", "query": "Query 3"},
        ]
        with open(data, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        return str(data)

    def test_load_samples(self, mock_store, mock_reflector, sample_jsonl):
        runner = TrainingRunner(store=mock_store, reflector=mock_reflector)
        samples = runner.load_samples(sample_jsonl)

        assert len(samples) == 3
        assert samples[0].id == "s1"
        assert samples[0].query == "Query 1"
        assert samples[1].success is False

    def test_load_samples_auto_id(self, mock_store, mock_reflector, tmp_path):
        data = tmp_path / "no_id.jsonl"
        with open(data, "w") as f:
            f.write('{"query": "test query"}\n')

        runner = TrainingRunner(store=mock_store, reflector=mock_reflector)
        samples = runner.load_samples(str(data))

        assert len(samples) == 1
        assert samples[0].id == "sample-1"

    def test_load_samples_skip_invalid(self, mock_store, mock_reflector, tmp_path):
        data = tmp_path / "mixed.jsonl"
        with open(data, "w") as f:
            f.write('{"id": "valid", "query": "test"}\n')
            f.write("invalid json\n")
            f.write('{"id": "valid2", "query": "test2"}\n')

        runner = TrainingRunner(store=mock_store, reflector=mock_reflector)
        samples = runner.load_samples(str(data))

        assert len(samples) == 2
        assert samples[0].id == "valid"
        assert samples[1].id == "valid2"

    def test_load_samples_file_not_found(self, mock_store, mock_reflector):
        runner = TrainingRunner(store=mock_store, reflector=mock_reflector)
        with pytest.raises(FileNotFoundError):
            runner.load_samples("/nonexistent/path.jsonl")

    @patch("ace.train.runner.curate")
    def test_train_single_epoch(
        self, mock_curate, mock_store, mock_reflector, sample_jsonl
    ):
        mock_curate.return_value = Delta(
            ops=[
                DeltaOp(
                    op="ADD",
                    new_bullet={
                        "section": "strategies",
                        "content": "test",
                        "tags": [],
                    },
                )
            ]
        )

        new_playbook = Playbook(version=2, bullets=[])
        with patch("ace.train.runner.apply_delta", return_value=new_playbook):
            with patch("ace.train.runner.refine"):
                runner = TrainingRunner(
                    store=mock_store,
                    reflector=mock_reflector,
                    refine_after_epoch=False,
                )
                result = runner.train(sample_jsonl, epochs=1)

        assert result.epochs_completed == 1
        assert result.total_samples_processed == 3
        assert result.playbook_version_start == 1
        assert mock_reflector.reflect.call_count == 3

    @patch("ace.train.runner.curate")
    def test_train_multiple_epochs(
        self, mock_curate, mock_store, mock_reflector, sample_jsonl
    ):
        mock_curate.return_value = Delta(
            ops=[
                DeltaOp(
                    op="ADD",
                    new_bullet={
                        "section": "strategies",
                        "content": "test",
                        "tags": [],
                    },
                )
            ]
        )

        new_playbook = Playbook(version=2, bullets=[])
        with patch("ace.train.runner.apply_delta", return_value=new_playbook):
            with patch("ace.train.runner.refine"):
                runner = TrainingRunner(
                    store=mock_store,
                    reflector=mock_reflector,
                    refine_after_epoch=False,
                )
                result = runner.train(sample_jsonl, epochs=3)

        assert result.epochs_completed == 3
        assert result.total_samples_processed == 9
        assert mock_reflector.reflect.call_count == 9

    @patch("ace.train.runner.curate")
    def test_train_with_refine(
        self, mock_curate, mock_store, mock_reflector, sample_jsonl
    ):
        mock_curate.return_value = Delta(ops=[])

        with patch("ace.train.runner.refine") as mock_refine:
            mock_refine.return_value = MagicMock(merged=1, archived=0)
            runner = TrainingRunner(
                store=mock_store,
                reflector=mock_reflector,
                refine_after_epoch=True,
            )
            runner.train(sample_jsonl, epochs=2)

        assert mock_refine.call_count == 2

    def test_save_and_load_state(self, mock_store, mock_reflector, tmp_path):
        state = TrainingState(total_epochs=5)
        state.start_epoch(1, 10)
        state.record_sample("s1", 1, 3, 10, 11)
        state.complete_epoch(1, 1, 3, 11)
        state.start_epoch(2, 11)

        state_file = str(tmp_path / "state.json")
        runner = TrainingRunner(store=mock_store, reflector=mock_reflector)
        runner.save_state(state, state_file)

        loaded = runner.load_state(state_file)
        assert loaded.current_epoch == 2
        assert loaded.total_epochs == 5
        assert len(loaded.epochs) == 2
        assert len(loaded.sample_records) == 1

    @patch("ace.train.runner.curate")
    def test_train_resume(self, mock_curate, mock_store, mock_reflector, sample_jsonl):
        mock_curate.return_value = Delta(ops=[])

        resume_state = TrainingState(total_epochs=3)
        resume_state.start_epoch(1, 1)
        resume_state.record_sample("s1", 1, 0, 1, 1)
        resume_state.record_sample("s2", 1, 0, 1, 1)
        resume_state.record_sample("s3", 1, 0, 1, 1)
        resume_state.complete_epoch(1, 3, 0, 1)

        with patch("ace.train.runner.refine"):
            runner = TrainingRunner(
                store=mock_store,
                reflector=mock_reflector,
                refine_after_epoch=False,
            )
            result = runner.train(sample_jsonl, epochs=3, resume_state=resume_state)

        assert result.epochs_completed == 3
        assert result.total_samples_processed == 9

    @patch("ace.train.runner.curate")
    def test_train_no_ops_generated(
        self, mock_curate, mock_store, mock_reflector, sample_jsonl
    ):
        mock_curate.return_value = Delta(ops=[])

        with patch("ace.train.runner.refine"):
            runner = TrainingRunner(
                store=mock_store,
                reflector=mock_reflector,
                refine_after_epoch=False,
            )
            result = runner.train(sample_jsonl, epochs=1)

        assert result.total_ops_applied == 0
        assert result.playbook_version_start == result.playbook_version_end

    def test_train_empty_samples(self, mock_store, mock_reflector, tmp_path):
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")

        runner = TrainingRunner(store=mock_store, reflector=mock_reflector)
        with pytest.raises(ValueError, match="No valid training samples"):
            runner.train(str(empty_file), epochs=1)


class TestEpochMetadata:
    """Tests for EpochMetadata."""

    def test_default_values(self):
        meta = EpochMetadata(epoch=1)
        assert meta.epoch == 1
        assert meta.samples_processed == 0
        assert meta.total_ops_applied == 0
        assert meta.completed_at is None

    def test_full_metadata(self):
        meta = EpochMetadata(
            epoch=2,
            samples_processed=10,
            total_ops_applied=25,
            playbook_version_start=5,
            playbook_version_end=8,
        )
        assert meta.samples_processed == 10
        assert meta.playbook_version_start == 5
