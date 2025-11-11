"""Tests for evaluation metrics."""

import pytest

from ace.eval.metrics import mean_reciprocal_rank, precision_at_k, recall_at_k


class TestMeanReciprocalRank:
    def test_perfect_ranking(self):
        """Test MRR when relevant item is always first."""
        ranked = [["doc1", "doc2", "doc3"], ["doc4", "doc5", "doc6"]]
        relevant = [{"doc1"}, {"doc4"}]
        assert mean_reciprocal_rank(ranked, relevant) == 1.0

    def test_second_position(self):
        """Test MRR when relevant item is at second position."""
        ranked = [["doc1", "doc2", "doc3"]]
        relevant = [{"doc2"}]
        assert mean_reciprocal_rank(ranked, relevant) == 0.5

    def test_mixed_positions(self):
        """Test MRR with different positions."""
        ranked = [["doc1", "doc2", "doc3"], ["doc4", "doc5", "doc6"]]
        relevant = [{"doc2"}, {"doc4"}]
        mrr = mean_reciprocal_rank(ranked, relevant)
        assert mrr == pytest.approx(0.75)  # (1/2 + 1/1) / 2

    def test_no_relevant_found(self):
        """Test MRR when no relevant items are found."""
        ranked = [["doc1", "doc2", "doc3"]]
        relevant = [{"doc99"}]
        assert mean_reciprocal_rank(ranked, relevant) == 0.0

    def test_partial_hits(self):
        """Test MRR when only some queries have hits."""
        ranked = [["doc1", "doc2"], ["doc3", "doc4"], ["doc5", "doc6"]]
        relevant = [{"doc2"}, {"doc99"}, {"doc5"}]
        mrr = mean_reciprocal_rank(ranked, relevant)
        assert mrr == pytest.approx(0.5)  # (1/2 + 0 + 1/1) / 3

    def test_empty_input(self):
        """Test MRR with empty input."""
        assert mean_reciprocal_rank([], []) == 0.0

    def test_multiple_relevant_takes_first(self):
        """Test MRR takes rank of first relevant item when multiple exist."""
        ranked = [["doc1", "doc2", "doc3", "doc4"]]
        relevant = [{"doc2", "doc4"}]
        assert mean_reciprocal_rank(ranked, relevant) == 0.5  # Takes doc2 at position 2

    def test_length_mismatch_raises_error(self):
        """Test that mismatched input lengths raise ValueError."""
        ranked = [["doc1"], ["doc2"]]
        relevant = [{"doc1"}]
        with pytest.raises(ValueError, match="must have same length"):
            mean_reciprocal_rank(ranked, relevant)


class TestRecallAtK:
    def test_perfect_recall(self):
        """Test Recall@k when all relevant items are in top-k."""
        ranked = [["doc1", "doc2", "doc3"]]
        relevant = [{"doc1", "doc2"}]
        assert recall_at_k(ranked, relevant, k=3) == 1.0

    def test_partial_recall(self):
        """Test Recall@k when only some relevant items are in top-k."""
        ranked = [["doc1", "doc2", "doc3", "doc4"]]
        relevant = [{"doc2", "doc4", "doc7"}]
        recall = recall_at_k(ranked, relevant, k=3)
        assert recall == pytest.approx(0.333333, rel=1e-5)  # 1 out of 3 relevant found

    def test_zero_recall(self):
        """Test Recall@k when no relevant items are in top-k."""
        ranked = [["doc1", "doc2", "doc3"]]
        relevant = [{"doc99"}]
        assert recall_at_k(ranked, relevant, k=3) == 0.0

    def test_multiple_queries(self):
        """Test Recall@k with multiple queries."""
        ranked = [
            ["doc1", "doc2", "doc3", "doc4"],
            ["doc5", "doc6", "doc7", "doc8"]
        ]
        relevant = [
            {"doc2", "doc4", "doc9"},  # 1/3 found in top-3 (doc2)
            {"doc5", "doc6"}            # 2/2 found in top-3
        ]
        recall = recall_at_k(ranked, relevant, k=3)
        assert recall == pytest.approx(0.666667, rel=1e-5)  # (1/3 + 2/2) / 2

    def test_k_larger_than_results(self):
        """Test Recall@k when k is larger than result list."""
        ranked = [["doc1", "doc2"]]
        relevant = [{"doc1", "doc2", "doc3"}]
        recall = recall_at_k(ranked, relevant, k=10)
        assert recall == pytest.approx(0.666667, rel=1e-5)  # 2/3

    def test_empty_input(self):
        """Test Recall@k with empty input."""
        assert recall_at_k([], [], k=5) == 0.0

    def test_empty_relevant_set(self):
        """Test Recall@k skips queries with no relevant items."""
        ranked = [["doc1"], ["doc2"]]
        relevant = [set(), {"doc2"}]
        assert recall_at_k(ranked, relevant, k=1) == 1.0  # Only second query counted

    def test_length_mismatch_raises_error(self):
        """Test that mismatched input lengths raise ValueError."""
        ranked = [["doc1"], ["doc2"]]
        relevant = [{"doc1"}]
        with pytest.raises(ValueError, match="must have same length"):
            recall_at_k(ranked, relevant, k=5)

    def test_negative_k_raises_error(self):
        """Test that negative k raises ValueError."""
        ranked = [["doc1", "doc2"]]
        relevant = [{"doc1"}]
        with pytest.raises(ValueError, match="k must be positive"):
            recall_at_k(ranked, relevant, k=-1)

    def test_zero_k_raises_error(self):
        """Test that k=0 raises ValueError."""
        ranked = [["doc1", "doc2"]]
        relevant = [{"doc1"}]
        with pytest.raises(ValueError, match="k must be positive"):
            recall_at_k(ranked, relevant, k=0)


class TestPrecisionAtK:
    def test_perfect_precision(self):
        """Test Precision@k when all top-k items are relevant."""
        ranked = [["doc1", "doc2", "doc3"]]
        relevant = [{"doc1", "doc2", "doc3"}]
        assert precision_at_k(ranked, relevant, k=3) == 1.0

    def test_partial_precision(self):
        """Test Precision@k when only some top-k items are relevant."""
        ranked = [["doc1", "doc2", "doc3"]]
        relevant = [{"doc2", "doc3"}]
        precision = precision_at_k(ranked, relevant, k=3)
        assert precision == pytest.approx(0.666667, rel=1e-5)  # 2/3

    def test_zero_precision(self):
        """Test Precision@k when no top-k items are relevant."""
        ranked = [["doc1", "doc2", "doc3"]]
        relevant = [{"doc99"}]
        assert precision_at_k(ranked, relevant, k=3) == 0.0

    def test_multiple_queries(self):
        """Test Precision@k with multiple queries."""
        ranked = [
            ["doc1", "doc2", "doc3"],
            ["doc4", "doc5", "doc6"]
        ]
        relevant = [
            {"doc2", "doc3"},  # 2/3
            {"doc4"}           # 1/3
        ]
        precision = precision_at_k(ranked, relevant, k=3)
        assert precision == pytest.approx(0.5, rel=1e-5)  # (2/3 + 1/3) / 2

    def test_k_smaller_than_relevant(self):
        """Test Precision@k when k is smaller than number of relevant items."""
        ranked = [["doc1", "doc2", "doc3", "doc4", "doc5"]]
        relevant = [{"doc1", "doc2", "doc3", "doc4", "doc5"}]
        assert precision_at_k(ranked, relevant, k=2) == 1.0  # 2/2

    def test_empty_input(self):
        """Test Precision@k with empty input."""
        assert precision_at_k([], [], k=5) == 0.0

    def test_length_mismatch_raises_error(self):
        """Test that mismatched input lengths raise ValueError."""
        ranked = [["doc1"], ["doc2"]]
        relevant = [{"doc1"}]
        with pytest.raises(ValueError, match="must have same length"):
            precision_at_k(ranked, relevant, k=5)

    def test_empty_results_contribute_zero(self):
        """Test that empty result lists contribute 0.0 to precision."""
        ranked = [[], ["doc1"]]
        relevant = [{"docX"}, {"doc1"}]
        precision = precision_at_k(ranked, relevant, k=1)
        assert precision == pytest.approx(0.5, rel=1e-5)  # (0 + 1) / 2

    def test_mixed_empty_and_nonempty_results(self):
        """Test precision calculation with mixed empty and non-empty results."""
        ranked = [[], ["doc1", "doc2"], [], ["doc3"]]
        relevant = [{"doc1"}, {"doc1", "doc2"}, {"doc3"}, {"doc3"}]
        precision = precision_at_k(ranked, relevant, k=2)
        # Query 0: empty -> 0.0
        # Query 1: 2/2 -> 1.0
        # Query 2: empty -> 0.0
        # Query 3: 1/1 -> 1.0 (only 1 result, but k=2)
        # Average: (0 + 1.0 + 0 + 1.0) / 4 = 0.5
        assert precision == pytest.approx(0.5, rel=1e-5)

    def test_negative_k_raises_error(self):
        """Test that negative k raises ValueError."""
        ranked = [["doc1", "doc2"]]
        relevant = [{"doc1"}]
        with pytest.raises(ValueError, match="k must be positive"):
            precision_at_k(ranked, relevant, k=-1)

    def test_zero_k_raises_error(self):
        """Test that k=0 raises ValueError."""
        ranked = [["doc1", "doc2"]]
        relevant = [{"doc1"}]
        with pytest.raises(ValueError, match="k must be positive"):
            precision_at_k(ranked, relevant, k=0)


class TestMetricsIntegration:
    def test_realistic_retrieval_scenario(self):
        """Test metrics with a realistic retrieval scenario."""
        ranked_results = [
            ["strat-001", "strat-002", "code-045", "tmpl-012", "strat-003"],
            ["code-023", "strat-010", "code-045", "fact-099", "tmpl-005"],
            ["strat-001", "tmpl-001", "code-001", "fact-001", "strat-002"]
        ]
        relevant_ids = [
            {"strat-001", "code-045"},     # Query 1: 2 relevant
            {"code-023", "code-045"},      # Query 2: 2 relevant
            {"strat-099", "code-099"}      # Query 3: 0 relevant (not in results)
        ]

        mrr = mean_reciprocal_rank(ranked_results, relevant_ids)
        recall_3 = recall_at_k(ranked_results, relevant_ids, k=3)
        precision_3 = precision_at_k(ranked_results, relevant_ids, k=3)

        # MRR: (1/1 + 1/1 + 0) / 3 = 0.666...
        assert mrr == pytest.approx(0.666667, rel=1e-5)

        # Recall@3: (2/2 + 1/2 + 0/2) / 3 = 0.5
        # Query 1: strat-001 at pos 1, code-045 at pos 3 = 2 found
        # Query 2: code-023 at pos 1, code-045 at pos 3 = 1 found (code-045 out of top-3)
        # Query 3: no relevant found
        assert recall_3 == pytest.approx(0.666667, rel=1e-5)

        # Precision@3: (2/3 + 2/3 + 0/3) / 3 = 0.444...
        # Query 1: 2 relevant in top-3 (strat-001, code-045)
        # Query 2: 2 relevant in top-3 (code-023, code-045)
        # Query 3: 0 relevant in top-3
        assert precision_3 == pytest.approx(0.444444, rel=1e-5)
