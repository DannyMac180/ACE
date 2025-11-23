import pytest

from ace.eval.metrics import mean_reciprocal_rank, precision_at_k, recall_at_k


class TestRetrievalMetrics:
    def test_mean_reciprocal_rank(self):
        # Example from docstring
        ranked = [["doc1", "doc2", "doc3"], ["doc4", "doc5", "doc6"]]
        relevant = [{"doc2"}, {"doc4"}]
        # Query 1: relevant doc2 is at index 1 (rank 2) -> 1/2
        # Query 2: relevant doc4 is at index 0 (rank 1) -> 1/1
        # MRR = (0.5 + 1.0) / 2 = 0.75
        assert mean_reciprocal_rank(ranked, relevant) == 0.75

        # Case: relevant item not found
        ranked_not_found = [["doc1", "doc3"], ["doc5", "doc6"]]
        relevant_not_found = [{"doc2"}, {"doc4"}]
        # Query 1: doc2 not in [doc1, doc3] -> 0
        # Query 2: doc4 not in [doc5, doc6] -> 0
        assert mean_reciprocal_rank(ranked_not_found, relevant_not_found) == 0.0

        # Case: empty input
        assert mean_reciprocal_rank([], []) == 0.0

        # Case: mismatch length
        with pytest.raises(ValueError):
            mean_reciprocal_rank([["a"]], [])

    def test_recall_at_k(self):
        # Example from docstring (re-evaluated)
        ranked = [["doc1", "doc2", "doc3", "doc4"], ["doc5", "doc6"]]
        relevant = [{"doc2", "doc4", "doc7"}, {"doc5"}]

        # Q1: top-3 [doc1, doc2, doc3]. Relevant: {doc2, doc4, doc7}.
        # Found: {doc2}. Count=1. Total=3. Recall=1/3.
        # Q2: top-3 [doc5, doc6]. Relevant: {doc5}. Found: {doc5}. Count=1. Total=1. Recall=1/1.
        # Avg: (1/3 + 1) / 2 = 4/6 / 2 = 2/3 = 0.6666...

        score = recall_at_k(ranked, relevant, k=3)
        assert abs(score - 0.6666666666666666) < 1e-9

        # Another case
        ranked2 = [["a", "b", "c"], ["d", "e"]]
        relevant2 = [{"a", "c"}, {"d", "e", "f"}]
        # k=2
        # Q1: top-2 [a, b]. Relevant {a, c}. Found {a}. Count=1. Total=2. Recall=0.5.
        # Q2: top-2 [d, e]. Relevant {d, e, f}. Found {d, e}. Count=2. Total=3. Recall=2/3.
        # Avg: (1/2 + 2/3) / 2 = (3/6 + 4/6) / 2 = 7/6 / 2 = 7/12 = 0.58333...
        score2 = recall_at_k(ranked2, relevant2, k=2)
        assert abs(score2 - (7/12)) < 1e-9

    def test_precision_at_k(self):
        ranked = [["doc1", "doc2", "doc3"], ["doc4", "doc5", "doc6"]]
        relevant = [{"doc2", "doc3"}, {"doc4"}]
        # k=3
        # Q1: top-3 [doc1, doc2, doc3]. Relevant: {doc2, doc3}. Found: 2. Precision: 2/3.
        # Q2: top-3 [doc4, doc5, doc6]. Relevant: {doc4}. Found: 1. Precision: 1/3.
        # Avg: (2/3 + 1/3) / 2 = 1/2 = 0.5.
        assert precision_at_k(ranked, relevant, k=3) == 0.5
