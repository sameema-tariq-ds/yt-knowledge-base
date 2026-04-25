"""
router_model.py
───────────────
The router sits between similarity search and LLM generation.

What it does:
  1. Takes 20 candidate chunks from vector search
  2. Applies multiple signals to re-rank them
  3. Returns the top 5 most relevant chunks

Why have a router?
  - Vector similarity alone isn't perfect (retrieves plausible but wrong chunks)
  - Reduces tokens sent to the LLM (cheaper + faster)
  - Can apply business logic (e.g. prefer recent videos)
  - Can detect query type (factual vs. comparative vs. summary) and adjust strategy

Signals used in this implementation:
  1. Cosine similarity (from ChromaDB)
  2. Query term overlap (keyword matching as a tie-breaker)
  3. Video diversity (don't return 5 chunks from the same video)
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from functools import lru_cache


from src.utils.config_loader import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)

WORD_REGEX = re.compile(r'\b[a-z]{3,}\b')
STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "i", "you", "he", "she", "we", "they", "it", "do", "does",
        "did", "will", "would", "could", "should", "have", "has",
        "had", "that", "this", "what", "how", "why", "when", "where",
        "and", "or", "but", "in", "on", "at", "to", "for", "of",
        "with", "about", "can", "me", "my",
    }

@dataclass
class ScoredChunk:
    """A chunk with a composite relevance score."""
    text:           str
    similarity:     float
    keyword_score:  float
    final_score:    float
    video_id:       str
    video_title:    str
    channel:        str
    timestamp_link: str
    start_time:     float
    chunk_index:    int


class RouterModel:
    """
    Multi-signal re-ranker for retrieved chunks.

    Usage:
        router = RouterModel()
        top_chunks = router.route(query="What is backpropagation?", candidates=raw_results)
    """

    def __init__(self):
        self.top_k    = cfg.router.top_k_after_routing
        self.threshold = cfg.router.similarity_threshold
        logger.info(f"Router ready (top_k={self.top_k}, threshold={self.threshold})")

    def route(self, query: str, candidates: list[dict]) -> list[ScoredChunk]:
        """
        Re-rank candidates and return the top_k most relevant chunks.

        Args:
            query: The user's original question
            candidates: Raw chunks from similarity_search()

        Returns:
            Sorted list of ScoredChunk objects, best first
        """
        if not isinstance(query, str) or not query.strip():
            logger.warning("Invalid query input")
            return []
    
        if not isinstance(candidates, list) or not candidates:
            logger.warning("Invalid candidates input")
            return []
        
        # Step 1: Filter by minimum similarity threshold
        filtered = [
            c for c in candidates
            if c.get("similarity", 0) >= self.threshold
        ]

        if not filtered:
            logger.warning(
                f"All {len(candidates)} candidates below threshold {self.threshold}. "
                "Relaxing to top 3 by raw similarity."
            )
            filtered = sorted(candidates, key=lambda x: x.get("similarity", 0), reverse=True)[:3]

        # Step 2: Compute BM25 scores
        bm25_scores = _compute_bm25_scores(query, filtered)

        scored = []
        for chunk, bm25_score in zip(filtered, bm25_scores):
            sim_score     = chunk.get("similarity", 0)

            # Weighted combination: 70% semantic similarity, 30% keyword overlap
            # Adjust weights based on your use case
            final_score = (0.70 * sim_score) + (0.30 * bm25_score)

            scored.append(ScoredChunk(
                text=           chunk.get("text", ""),
                similarity=     sim_score,
                keyword_score=  bm25_score,
                final_score=    final_score,
                video_id=       chunk.get("video_id", ""),
                video_title=    chunk.get("video_title", ""),
                channel=        chunk.get("channel", ""),
                timestamp_link= chunk.get("timestamp_link", ""),
                start_time=     chunk.get("start_time", 0),
                chunk_index=    chunk.get("chunk_index", 0),
            ))

        # Step 3: Sort by final score
        scored.sort(key=lambda x: x.final_score, reverse=True)

        # Step 4: Enforce video diversity
        # Don't return more than 2 chunks from the same video
        # This prevents one verbose video from dominating the answer
        diverse = _enforce_diversity(scored, max_per_video=2)

        top = diverse[:self.top_k]

        if top:
            logger.debug(
                f"Router: {len(candidates)} → {len(filtered)} → {len(top)} chunks | "
                f"Top score: {top[0].final_score:.3f}" if top else "No chunks"
            )
            return top
        else:
            logger.warning("No Chunk is selected")
            return []


    def classify_query(self, query: str) -> str:
        """
        Classify the query type to hint at response strategy.

        Returns:
            "factual"      — specific fact lookups ("what is X?")
            "comparative"  — comparing two things ("X vs Y")
            "summary"      — overview requests ("explain the whole video")
            "procedural"   — how-to questions ("how do I do X?")
            "open"         — everything else
        """
        q = query.lower().strip()

        if any(q.startswith(w) for w in ["what is", "what are", "define", "who is"]):
            return "factual"
        if any(w in q for w in [" vs ", " versus ", "compare", "difference between"]):
            return "comparative"
        if any(w in q for w in ["summarize", "overview", "explain everything", "tell me about"]):
            return "summary"
        if any(q.startswith(w) for w in ["how do", "how to", "how can", "steps to"]):
            return "procedural"
        return "open"


# ── Helper functions ──────────────────────────────────────────────────
@lru_cache(maxsize=10000)
def _extract_keywords(text: str) -> set[str]:
    """
    Extract meaningful words from text, ignoring stop words.
    This is intentionally simple — no NLTK required.
    """
    if not text or not isinstance(text, str):
        return set()
    
    words = WORD_REGEX.findall(text.lower())
    return {w for w in words if w not in STOP_WORDS}


def _tokenize(text: str) -> list[str]:
    return list(_extract_keywords(text))


def _compute_bm25_scores(query: str, chunks: list[dict]) -> list[float]:
    """
    Compute BM25 scores for candidate chunks relative to query.
    """

    # Tokenize each chunk and query
    tokenized_docs = [
        _tokenize(chunk.get("text", "")) for chunk in chunks
    ]

    query_tokens = _tokenize(query)

    # Edge case: empty chunks and query
    if not any(tokenized_docs) or not query_tokens:
        return [0.0] * len(chunks)

    # Calculate Best Matching25 for chunks and query
    bm25 = BM25Okapi(tokenized_docs)

    scores = bm25.get_scores(query_tokens)

    # Normalize scores to 0–1 (important!)
    max_score = max(scores) if scores.size > 0 else 1.0
    if max_score == 0:
        return [0.0] * len(scores)

    return [float(s / max_score) for s in scores]


def _enforce_diversity(
    chunks: list[ScoredChunk],
    max_per_video: int = 2,
) -> list[ScoredChunk]:
    """
    Filter so no single video contributes more than max_per_video chunks.
    Keeps the highest-scored chunks from each video.
    """
    video_counts: dict[str, int] = defaultdict(int)
    diverse = []
    for chunk in chunks:
        vid = chunk.video_id or f"unknown_{chunk.chunk_index}"

        if video_counts[vid] < max_per_video:
            diverse.append(chunk)
            video_counts[chunk.video_id] += 1

    return diverse