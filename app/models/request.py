"""Pydantic request models for the query API."""

from typing import Any

from pydantic import BaseModel, Field


class PipelineRunRequest(BaseModel):
    """Request body for ingesting a YouTube channel and building the index."""

    url: str = Field(..., min_length=1, description="YouTube channel URL to ingest.")
    handle: str | None = Field(
        default=None,
        description="Optional short name for cached files. Generated from the URL when omitted.",
    )


class QueryRequest(BaseModel):
    """Request body for asking the query pipeline a question."""

    query: str = Field(..., min_length=1, description="The user's question or input.")
    context: dict[str, Any] | None = Field(
        default=None,
        description="Optional context data supplied by the caller.",
    )
    parameters: dict[str, Any] | None = Field(
        default=None,
        description="Optional pipeline configuration values for this request.",
    )
