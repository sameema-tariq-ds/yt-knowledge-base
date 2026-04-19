"""
embedder.py
───────────
Generates vector embeddings from text using sentence-transformers.

Why sentence-transformers (all-MiniLM-L6-v2)?
  - Free, runs locally, no API key needed
  - Fast: ~14,000 sentences/second on CPU
  - Good quality for semantic search
  - 384-dimensional vectors (small but effective)

For production with higher quality, swap to:
  - "all-mpnet-base-v2" (better quality, slower)
  - OpenAI "text-embedding-3-small" (API, costs money but excellent)
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

from src.utils.config_loader import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """Wrapper around sentence-transformers for consistent embedding."""

    def __init__(self):
        try:
            model_name = cfg.embedding.model_name
            device     = cfg.embedding.device

            logger.info(f"Loading embedding model: {model_name} on {device}")
            # First run downloads the model (~90MB). Subsequent runs use cache.
            self.model = SentenceTransformer(model_name, device=device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {self.dimension}")

        except Exception as e:
            logger.exception("Failed to initialize embedding model")
            raise RuntimeError("Embedding model initialization failed") from e

    def _validate_texts(self, texts: List[str]) -> List[str]:
        """Validate and sanitize input texts."""

        if texts is None:
            raise ValueError("texts cannot be None")

        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")

        cleaned_text = []
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                logger.warning("Skipping non-string input", extra={"index": i})
                continue

            t = text.strip()
            if not t:
                continue

            cleaned_text.append(t)

        if not cleaned_text:
            logger.warning("No valid texts to embed")
            return []

        return cleaned_text
    

    def _validate_query(self, query: str) -> str:
        """Validate single query."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")

        return query


    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of strings into vectors.

        Args:
            texts: List of strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        try:
            texts = self._validate_texts(texts)

            if not texts:
                logger.warning("No valid texts to embed")
                return []

            batch_size = cfg.embedding.batch_size
            logger.debug(f"Embedding {len(texts)} texts in batches of {batch_size}")

            # encode() handles batching internally; show_progress_bar for long lists
            embeddings: np.ndarray = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100,
                normalize_embeddings=True,  # L2 normalize for cosine similarity
                convert_to_numpy=True,
            )

            # Convert numpy array to plain Python lists for JSON serialization
            return embeddings.tolist()
        
        except Exception as e:
            logger.exception("Failed to embed texts")
            raise RuntimeError("Embedding failed") from e


    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string. Optimized for single inputs."""
        try:
            query = self._validate_query(query)
            embedding = self.model.encode(
                query,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return embedding.tolist()
        except Exception:
            logger.exception("Query embedding failed")
            raise RuntimeError("Query embedding failed")