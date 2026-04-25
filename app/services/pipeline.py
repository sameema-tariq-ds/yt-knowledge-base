"""Service wrapper around QueryPipeline.ask()."""

from __future__ import annotations

import importlib
import json
import logging
import threading
from json import JSONDecodeError
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.config import get_settings
from app.models.request import PipelineRunRequest, QueryRequest
from app.models.response import PipelineRunResponse, QueryResponseModel

logger = logging.getLogger(__name__)


class PipelineService:
    """Lazy singleton wrapper for the expensive QueryPipeline instance."""

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._lock = threading.Lock()

    def _load_pipeline_class(self) -> type:
        settings = get_settings()
        module_name, class_name = settings.query_pipeline_class.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    pipeline_class = self._load_pipeline_class()
                    self._pipeline = pipeline_class()
        return self._pipeline

    def ask(self, request: QueryRequest) -> QueryResponseModel:
        """Ask the underlying pipeline and normalize its response."""
        pipeline = self._get_pipeline()
        raw_response = pipeline.ask(request.query)
        data = self._to_mapping(raw_response)

        metadata = {
            key: value
            for key, value in data.items()
            if key not in {"answer", "confidence", "sources"}
        }
        if request.context is not None:
            metadata["request_context"] = request.context
        if request.parameters is not None:
            metadata["request_parameters"] = request.parameters

        answer = data.get("answer", raw_response if isinstance(raw_response, str) else "")
        return QueryResponseModel(
            answer=str(answer),
            confidence=data.get("confidence"),
            sources=data.get("sources"),
            metadata=metadata or None,
        )

    def run_pipeline(self, request: PipelineRunRequest) -> PipelineRunResponse:
        """Ingest a channel URL, rebuild the vector database, and refresh queries."""
        from src.ingestion.channel_scrapper import get_channel_video_ids, save_video_list
        from src.ingestion.metadata_store import MetadataStore
        from src.ingestion.transcript_fetcher import fetch_all_transcripts
        from src.processing.chunker import create_chunks
        from src.storage.embedder import Embedder
        from src.storage.vector_store import VectorStore
        from src.utils.config_loader import cfg
        from src.utils.file_utils import sanitize_filename

        # Channel Scraper
        handle = sanitize_filename(request.handle or self._handle_from_url(request.url))
        logger.info("Running pipeline", extra={"url": request.url, "handle": handle})
        videos_metadata = list(get_channel_video_ids(request.url))
        videos_found = save_video_list(videos_metadata, handle)


        # Transcript Fetcher
        transcript_dir = Path(cfg.paths.raw_data) / "transcripts" / handle
        transcript_dir.mkdir(parents=True, exist_ok=True)

        cached_transcripts, missing_videos = self._load_cached_transcripts(videos_metadata, transcript_dir,)
        logger.info(
            "Transcript cache scan complete",
            extra={
                "handle": handle,
                "cached": len(cached_transcripts),
                "missing": len(missing_videos),
            },
        )

        fetched_transcripts = (fetch_all_transcripts(missing_videos, handle) if missing_videos else [])
        transcribed = cached_transcripts + fetched_transcripts
        fetched_video_ids = {video.get("video_id") for video in fetched_transcripts}
        transcripts_failed = len(
            [
                video
                for video in missing_videos
                if video.get("video_id") not in fetched_video_ids
            ]
        )


        # Metadata Store Information Extraction
        metadata_store = MetadataStore()
        for video in transcribed:
            metadata_store.upsert_video(video)

        json_files = list(transcript_dir.glob("*.json"))
        if not json_files:
            raise ValueError("No transcripts were found for this channel.")

        embedder = Embedder()
        vector_store = VectorStore()
        chunks_indexed = 0
        chunks_created = 0
        chunk_details: list[dict[str, Any]] = []
        for json_file in json_files:
            video_data = self._read_json(json_file)
            chunks = create_chunks(video_data)
            if not chunks:
                continue

            chunks_created += len(chunks)
            texts = [chunk.text for chunk in chunks]
            embeddings = embedder.embed_texts(texts)
            added = vector_store.add_chunks(chunks, embeddings)
            metadata_store.update_chunk_count(video_data["video_id"], len(chunks))
            chunks_indexed += added
            chunk_details.append(
                {
                    "video_id": video_data.get("video_id", ""),
                    "title": video_data.get("title", "Untitled video"),
                    "url": video_data.get("url", ""),
                    "chunks_created": len(chunks),
                    "new_chunks_indexed": added,
                }
            )

        self.reset()
        return PipelineRunResponse(
            status="ok",
            message="The channel is indexed. You can ask questions now.",
            handle=handle,
            videos_found=videos_found,
            transcripts_downloaded=len(transcribed),
            transcripts_cached=len(cached_transcripts),
            transcripts_fetched=len(fetched_transcripts),
            transcripts_failed=transcripts_failed,
            chunks_indexed=chunks_indexed,
            chunks_created=chunks_created,
            total_chunks_available=vector_store.count(),
            videos=[
                {
                    "video_id": video.get("video_id", ""),
                    "title": video.get("title", "Untitled video"),
                    "url": video.get("url", ""),
                    "duration": video.get("duration", 0),
                }
                for video in videos_metadata
            ],
            transcripts=[
                {
                    "video_id": video.get("video_id", ""),
                    "title": video.get("title", "Untitled video"),
                    "path": str(transcript_dir / f"{video.get('video_id', '')}.json"),
                }
                for video in transcribed
            ],
            chunks=chunk_details,
        )

    def reset(self) -> None:
        """Drop the cached query pipeline so it reloads the latest vector index."""
        with self._lock:
            self._pipeline = None

    @staticmethod
    def _to_mapping(response: Any) -> dict[str, Any]:
        if is_dataclass(response):
            return asdict(response)
        if isinstance(response, dict):
            return response
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if hasattr(response, "dict"):
            return response.dict()
        return {"answer": str(response)}

    @staticmethod
    def _handle_from_url(url: str) -> str:
        parsed = urlparse(url.strip())
        path_bits = [part for part in parsed.path.split("/") if part]
        for index, part in enumerate(path_bits):
            if part.startswith("@"):
                return part.lstrip("@") or "channel"
            if part in {"c", "channel"} and index + 1 < len(path_bits):
                return path_bits[index + 1]
        if path_bits:
            return path_bits[0].lstrip("@") or "channel"
        return parsed.netloc or "channel"

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _load_cached_transcripts(
        videos: list[dict],
        transcript_dir: Path,
    ) -> tuple[list[dict[str, Any]], list[dict]]:
        cached: list[dict[str, Any]] = []
        missing: list[dict] = []

        for video in videos:
            video_id = video.get("video_id")
            if not video_id:
                continue

            cache_file = transcript_dir / f"{video_id}.json"
            if not cache_file.exists():
                missing.append(video)
                continue

            try:
                with cache_file.open("r", encoding="utf-8") as file:
                    data = json.load(file)
            except (JSONDecodeError, OSError):
                logger.warning("Invalid cached transcript, refetching", extra={"path": str(cache_file)})
                try:
                    cache_file.unlink(missing_ok=True)
                except OSError:
                    pass
                missing.append(video)
                continue

            if not data.get("transcript_segments"):
                logger.warning("Cached transcript has no segments, refetching", extra={"path": str(cache_file)})
                try:
                    cache_file.unlink(missing_ok=True)
                except OSError:
                    pass
                missing.append(video)
                continue

            cached.append(data)

        return cached, missing


pipeline_service = PipelineService()
