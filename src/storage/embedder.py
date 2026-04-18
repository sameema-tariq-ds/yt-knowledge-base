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
import numpy as np

from src.utils.config_loader import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """Wrapper around sentence-transformers for consistent embedding."""

    def __init__(self):
        model_name = cfg.embedding.model_name
        device     = cfg.embedding.device

        logger.info(f"Loading embedding model: {model_name} on {device}")
        # First run downloads the model (~90MB). Subsequent runs use cache.
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of strings into vectors.

        Args:
            texts: List of strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if not texts:
            return []

        batch_size = cfg.embedding.batch_size
        logger.debug(f"Embedding {len(texts)} texts in batches of {batch_size}")

        # encode() handles batching internally; show_progress_bar for long lists
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
            convert_to_numpy=True,
        )

        # Convert numpy array to plain Python lists for JSON serialization
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string. Optimized for single inputs."""
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()