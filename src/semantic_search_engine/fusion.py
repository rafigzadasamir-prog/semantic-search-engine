"""Score fusion algorithms for combining multiple ranked result lists.

Pure functions operating on ``(doc_id, score)`` pairs, with no dependency on
how those scores were produced. This keeps the fusion logic trivially
unit-testable without any model or index.
"""

from __future__ import annotations

RankedList = list[tuple[str, float]]


def reciprocal_rank_fusion(ranked_lists: list[RankedList], k: int = 60) -> RankedList:
    """Combine ranked lists with Reciprocal Rank Fusion (RRF).

    Each document's fused score is the sum, across lists, of ``1 / (k + rank)``
    where ``rank`` is its 1-based position in that list (documents absent from
    a list simply contribute 0 from it). RRF only uses rank order, not the raw
    scores, which makes it robust to lists whose scores live on unrelated
    scales (e.g. BM25 scores vs. cosine similarities).

    Returns doc ids sorted by descending fused score.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    fused_scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)


def _min_max_normalize(ranked_list: RankedList) -> dict[str, float]:
    if not ranked_list:
        return {}
    scores = [score for _, score in ranked_list]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return {doc_id: 1.0 for doc_id, _ in ranked_list}
    return {doc_id: (score - lo) / (hi - lo) for doc_id, score in ranked_list}


def weighted_score_fusion(
    ranked_lists: list[RankedList], weights: list[float] | None = None
) -> RankedList:
    """Combine ranked lists by min-max normalizing scores, then weighted-summing.

    Unlike RRF this uses the raw scores (after normalization), which can
    preserve information about how confidently a system ranked something,
    at the cost of being sensitive to each system's score distribution.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must have the same length as ranked_lists")

    fused_scores: dict[str, float] = {}
    for ranked_list, weight in zip(ranked_lists, weights, strict=True):
        normalized = _min_max_normalize(ranked_list)
        for doc_id, norm_score in normalized.items():
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + weight * norm_score

    return sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
