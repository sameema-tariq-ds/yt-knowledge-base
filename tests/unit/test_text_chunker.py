"""Unit tests for the chunker."""
import pytest
from src.processing.chunker import create_chunks, count_tokens, Chunk


class TestChunkCreation:
    def test_returns_chunks(self, sample_video):
        chunks = create_chunks(sample_video)
        assert len(chunks) > 0

    def test_chunks_are_chunk_objects(self, sample_video):
        chunks = create_chunks(sample_video)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_ids_are_unique(self, sample_video):
        chunks = create_chunks(sample_video)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_preserves_video_id(self, sample_video):
        chunks = create_chunks(sample_video)
        assert all(c.video_id == "test123" for c in chunks)

    def test_timestamp_link_format(self, sample_video):
        chunks = create_chunks(sample_video)
        for chunk in chunks:
            assert "youtube.com" in chunk.timestamp_link
            assert "&t=" in chunk.timestamp_link

    def test_empty_video_returns_no_chunks(self):
        empty_video = {
            "video_id": "empty",
            "title": "Empty",
            "channel": "Test",
            "url": "https://youtube.com/watch?v=empty",
            "transcript_segments": [],
            "full_text": "",
        }
        assert create_chunks(empty_video) == []


class TestTokenCounting:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_counts_tokens(self):
        # "hello world" is typically 2 tokens
        assert count_tokens("hello world") >= 2