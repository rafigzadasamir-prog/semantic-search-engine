from semantic_search_engine.rerank import rerank


def test_rerank_reorders_candidates_by_cross_encoder_score(word_overlap_cross_encoder):
    candidates = [
        ("low", "completely unrelated text about gardening and flowers"),
        ("high", "python programming language tutorial for beginners"),
    ]
    results = rerank("python programming tutorial", candidates, word_overlap_cross_encoder)
    assert [doc_id for doc_id, _ in results][0] == "high"


def test_rerank_is_sorted_descending_by_score(word_overlap_cross_encoder):
    candidates = [
        ("a", "cat"),
        ("b", "cat dog"),
        ("c", "cat dog bird"),
    ]
    results = rerank("cat dog bird fish", candidates, word_overlap_cross_encoder)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    assert [doc_id for doc_id, _ in results] == ["c", "b", "a"]


def test_rerank_empty_candidates_returns_empty_without_calling_model():
    class ExplodingCrossEncoder:
        def predict(self, sentence_pairs):
            raise AssertionError("predict() should not be called for an empty candidate list")

    assert rerank("query", [], ExplodingCrossEncoder()) == []


def test_rerank_passes_query_and_text_as_pairs():
    seen_pairs = []

    class RecordingCrossEncoder:
        def predict(self, sentence_pairs):
            seen_pairs.extend(sentence_pairs)
            return [0.0 for _ in sentence_pairs]

    rerank("my query", [("a", "doc text a"), ("b", "doc text b")], RecordingCrossEncoder())
    assert seen_pairs == [("my query", "doc text a"), ("my query", "doc text b")]
