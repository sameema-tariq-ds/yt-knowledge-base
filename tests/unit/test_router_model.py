"""Unit tests for the router model."""
import pytest
from src.router.router_model import RouterModel, _compute_bm25_scores, _extract_keywords


class TestKeywordExtraction:
    def test_extracts_content_words(self):
        kws = _extract_keywords("What is machine learning?")
        assert "machine" in kws
        assert "learning" in kws

    def test_filters_stop_words(self):
        kws = _extract_keywords("What is the answer?")
        assert "the" not in kws
        assert "is" not in kws


class TestKeywordOverlap:
    def test_perfect_overlap(self):
        terms = "neural network"
        text  = [{'text': "A neural network is a machine learning model"}]
        score = _compute_bm25_scores(terms, text)
        assert len(score) == 1
        assert score[0] == 1.0

    def test_no_overlap(self):
        terms = "quantum physics"
        text  = [{'text': "The neural network was trained on images"}]
        score = _compute_bm25_scores(terms, text)
        assert len(score) == 1
        assert score[0] == 0.0 or score[0] < 0.01

    def test_partial_overlap(self):
        terms = "neural regression"
        text  = [
            {'text': "A neural network model"},
            {'text': "quantum physics experiments"},
        ]
        scores = _compute_bm25_scores(terms, text)
        assert len(scores) == 2
        assert 0.0 <= max(scores) <= 1.0


class TestRouterModel:
    def test_route_returns_limited_results(self):
        router = RouterModel()
        candidates = [
            {"text": f"Chunk {i}", "similarity": 0.9 - i*0.05,
             "video_id": f"vid{i}", "video_title": f"Video {i}",
             "channel": "Test", "timestamp_link": "https://yt.com",
             "start_time": float(i*10), "chunk_index": i}
            for i in range(15)
        ]
        results = router.route("machine learning tutorial", candidates)
        assert len(results) <= router.top_k

    def test_classify_factual_query(self):
        router = RouterModel()
        assert router.classify_query("What is backpropagation?") == "factual"

    def test_classify_comparative_query(self):
        router = RouterModel()
        assert router.classify_query("Do you know the difference between RNN vs LSTM?") == "comparative"

    def test_classify_procedural_query(self):
        router = RouterModel()
        assert router.classify_query("How do I train a neural network?") == "procedural"