"""
vector_store.py
───────────────
ChromaDB wrapper for storing and searching embeddings.

ChromaDB is perfect for this project because:
  - No server required — runs embedded in your Python process
  - Persists data to disk automatically
  - Supports metadata filtering (search only videos from 2023, etc.)
  - Free and open source

For production scale (millions of chunks), consider:
  - Pinecone (managed cloud)
  - Qdrant (self-hosted or cloud)
  - Weaviate (self-hosted or cloud)
"""

from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from chromadb.errors import ChromaError

from src.utils.config_loader import cfg
from src.utils.logger import get_logger
from src.processing.chunker import Chunk

logger = get_logger(__name__)


def _validate_chunks_and_embeddings(chunks: List[Chunk], embeddings: List[List[float]],) -> None:
    """Ensure chunks and embeddings are valid and aligned."""

    if not isinstance(chunks, list) or not isinstance(embeddings, list):
        raise TypeError("chunks and embeddings must be lists")

    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")

    for i, (c, e) in enumerate(zip(chunks, embeddings)):
        if not isinstance(c, Chunk):
            raise TypeError(f"Invalid chunk at index {i}")

        if not isinstance(e, list) or not all(isinstance(x, (int, float)) for x in e):
            raise TypeError(f"Invalid embedding at index {i}")


def _validate_query_embedding(embedding: List[float]) -> List[float]:
    if not isinstance(embedding, list):
        raise TypeError("query_embedding must be a list")

    if not embedding:
        raise ValueError("query_embedding cannot be empty")

    return embedding

class VectorStore:
    """ChromaDB interface for the YouTube knowledge base."""

    def __init__(self):
        try:
            db_path    = Path(cfg.paths.vector_db)
            collection = cfg.vector_store.collection_name

            db_path.mkdir(parents=True, exist_ok=True)

            # Persistent client: data survives between Python sessions
            # create db connection
            self.client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(anonymized_telemetry=False),  # Opt out of telemetry
            )

            # Get or create the collection
            # ChromaDB collection = a named set of embeddings
            self.collection = self.client.get_or_create_collection(
                name=collection,
                metadata={"hnsw:space": "cosine"},  # Use cosine distance
            )

            logger.info(
                f"Vector store ready: '{collection}' "
                f"({self.collection.count()} chunks)"
            )

        except Exception as e:
            logger.exception("Failed to initialize vector store")
            raise RuntimeError("Vector store initialization failed") from e


    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """
        Store chunks and their embeddings in ChromaDB.

        ChromaDB requires:
          - ids: unique strings
          - embeddings: list of float vectors
          - documents: the text (stored for retrieval)
          - metadatas: dicts with filtering info

        Returns:
            Number of chunks successfully added
        """
        if not chunks:
            logger.debug("No chunks provided for insertion")
            return 0

        try:
            _validate_chunks_and_embeddings(chunks, embeddings)

            ids         = [c.chunk_id for c in chunks]
            documents   = [c.text for c in chunks]
            metadatas   = [
                {
                    "video_id":       c.video_id,
                    "video_title":    c.video_title[:200],   # ChromaDB has length limits
                    "channel":        c.channel,
                    "url":            c.url,
                    "start_time":     c.start_time,
                    "timestamp_link": c.timestamp_link,
                    "chunk_index":    c.chunk_index,
                    "token_count":    c.token_count,
                }
                for c in chunks
            ]

            # Filter out chunks already in the DB (idempotent upsert)
            existing = set(self.collection.get(ids=ids)["ids"])
            new_mask = [i for i, cid in enumerate(ids) if cid not in existing]

            if not new_mask:
                logger.debug("All chunks already in vector store, skipping")
                return 0

            self.collection.add(
                ids=        [ids[i] for i in new_mask],
                embeddings= [embeddings[i] for i in new_mask],
                documents=  [documents[i] for i in new_mask],
                metadatas=  [metadatas[i] for i in new_mask],
            )

            added = len(new_mask)
            logger.debug(f"Added {added} new chunks to vector store")
            return added
        
        except ChromaError:
            logger.exception("ChromaDB error during insert")
            raise RuntimeError("Vector store insert failed")

        except Exception:
            logger.exception("Unexpected error during insert")
            raise

    def similarity_search(self, query_embedding: list[float], n_results: int | None = None, where: dict | None = None,) -> list[dict]:
        """
        Find the most semantically similar chunks to a query.

        Args:
            query_embedding: The query embedded as a vector
            n_results: How many results to return
            where: Optional ChromaDB filter, e.g. {"video_id": "abc123"}

        Returns:
            List of result dicts with text, metadata, and similarity distance
        """
        try:
            query_embedding = _validate_query_embedding(query_embedding)

            total_count = self.collection.count()
            if total_count == 0:
                logger.warning("Vector store is empty")
                return []
            
            num_retrieve_query = min(n_results or cfg.vector_store.n_results, total_count)

            query_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": num_retrieve_query,
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                query_kwargs["where"] = where # filter the chunk and then search

            results = self.collection.query(**query_kwargs)

            # Unpack ChromaDB's nested response format
            output = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                similarity = 1 - dist    # ChromaDB returns distance; convert to similarity
                output.append({
                    "text":           doc,
                    "similarity":     similarity,
                    **meta,             # Unpack all metadata fields
                })

            return output
        
        except ChromaError:
            logger.exception("ChromaDB query failed")
            raise RuntimeError("Similarity search failed")

        except Exception:
            logger.exception("Unexpected error during search")
            raise

    def delete_video(self, video_id: str) -> int:
        """Remove all chunks for a specific video (useful for updates)."""
        if not video_id:
            raise ValueError("video_id is required")

        try:
            results = self.collection.get(where={"video_id": video_id})
            ids_to_delete = results["ids"]

            if not ids_to_delete:
                logger.info("No chunks found for deletion", extra={"video_id": video_id})
                return 0
            
            self.collection.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} chunks for video {video_id}")
            return len(ids_to_delete)
        
        except ChromaError:
            logger.exception("ChromaDB delete failed")
            raise RuntimeError("Delete operation failed")

    def count(self) -> int:
        try: 
            return int(self.collection.count())
    
        except Exception:
            logger.exception("Failed to fetch count")
            raise