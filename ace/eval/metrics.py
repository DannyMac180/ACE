"""Evaluation metrics for ACE retrieval and quality assessment."""



def mean_reciprocal_rank(ranked_results: list[list[str]], relevant_ids: list[set[str]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for retrieval results.

    MRR is the average of reciprocal ranks of the first relevant item.

    Args:
        ranked_results: List of ranked result lists (each inner list is ordered by rank)
        relevant_ids: List of sets of relevant item IDs for each query

    Returns:
        MRR score between 0 and 1

    Example:
        >>> ranked = [["doc1", "doc2", "doc3"], ["doc4", "doc5", "doc6"]]
        >>> relevant = [{"doc2"}, {"doc4"}]
        >>> mean_reciprocal_rank(ranked, relevant)
        0.75  # (1/2 + 1/1) / 2
    """
    if len(ranked_results) != len(relevant_ids):
        raise ValueError("ranked_results and relevant_ids must have same length")

    if not ranked_results:
        return 0.0

    reciprocal_ranks = []
    for results, relevant in zip(ranked_results, relevant_ids, strict=False):
        rank = None
        for i, result_id in enumerate(results, start=1):
            if result_id in relevant:
                rank = i
                break

        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def recall_at_k(ranked_results: list[list[str]], relevant_ids: list[set[str]], k: int) -> float:
    """
    Calculate Recall@k for retrieval results.

    Recall@k is the proportion of relevant items found in the top-k results.

    Args:
        ranked_results: List of ranked result lists (each inner list is ordered by rank)
        relevant_ids: List of sets of relevant item IDs for each query
        k: Number of top results to consider (must be positive)

    Returns:
        Recall@k score between 0 and 1

    Raises:
        ValueError: If k <= 0 or if input lists have different lengths

    Example:
        >>> ranked = [["doc1", "doc2", "doc3", "doc4"], ["doc5", "doc6"]]
        >>> relevant = [{"doc2", "doc4", "doc7"}, {"doc5"}]
        >>> recall_at_k(ranked, relevant, k=3)
        0.6666666666666666  # (1/3 + 1/1) / 2
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if len(ranked_results) != len(relevant_ids):
        raise ValueError("ranked_results and relevant_ids must have same length")

    if not ranked_results:
        return 0.0

    recall_scores = []
    for results, relevant in zip(ranked_results, relevant_ids, strict=False):
        if not relevant:
            continue

        top_k = set(results[:k])
        found = len(top_k & relevant)
        recall = found / len(relevant)
        recall_scores.append(recall)

    if not recall_scores:
        return 0.0

    return sum(recall_scores) / len(recall_scores)


def precision_at_k(ranked_results: list[list[str]], relevant_ids: list[set[str]], k: int) -> float:
    """
    Calculate Precision@k for retrieval results.

    Precision@k is the proportion of relevant items among the top-k results.

    Args:
        ranked_results: List of ranked result lists (each inner list is ordered by rank)
        relevant_ids: List of sets of relevant item IDs for each query
        k: Number of top results to consider (must be positive)

    Returns:
        Precision@k score between 0 and 1

    Raises:
        ValueError: If k <= 0 or if input lists have different lengths

    Example:
        >>> ranked = [["doc1", "doc2", "doc3"], ["doc4", "doc5", "doc6"]]
        >>> relevant = [{"doc2", "doc3"}, {"doc4"}]
        >>> precision_at_k(ranked, relevant, k=3)
        0.5  # (2/3 + 1/3) / 2
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if len(ranked_results) != len(relevant_ids):
        raise ValueError("ranked_results and relevant_ids must have same length")

    if not ranked_results:
        return 0.0

    precision_scores = []
    for results, relevant in zip(ranked_results, relevant_ids, strict=False):
        top_k = results[:k]
        if not top_k:
            # Empty results contribute 0.0 to the average
            precision_scores.append(0.0)
            continue

        found = sum(1 for result_id in top_k if result_id in relevant)
        precision = found / len(top_k)
        precision_scores.append(precision)

    if not precision_scores:
        return 0.0

    return sum(precision_scores) / len(precision_scores)
