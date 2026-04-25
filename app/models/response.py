"""Pydantic response models for the query API."""

from typing import Any

from pydantic import BaseModel, Field


class QueryResponseModel(BaseModel):
    """Validated response returned by the query API."""

    answer: str = Field(..., description="The pipeline's response.")
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence score when the pipeline provides one.",
    )
    sources: list[dict[str, Any]] | None = Field(
        default=None,
        description="Source documents, videos, or references used by the answer.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional response metadata.",
    )


class PipelineRunResponse(BaseModel):
    """Response returned after ingesting content and rebuilding the index."""

    status: str
    message: str
    handle: str
    videos_found: int
    transcripts_downloaded: int
    transcripts_cached: int = 0
    transcripts_fetched: int = 0
    transcripts_failed: int = 0
    chunks_indexed: int
    chunks_created: int = 0
    total_chunks_available: int | None = None
    videos: list[dict[str, Any]] | None = None
    transcripts: list[dict[str, Any]] | None = None
    chunks: list[dict[str, Any]] | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
