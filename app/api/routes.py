"""API endpoints for the FastAPI web interface."""

import json
import logging
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.responses import Response
from pydantic import ValidationError

from app.config import get_settings
from app.models.request import PipelineRunRequest, QueryRequest
from app.models.response import HealthResponse, PipelineRunResponse, QueryResponseModel
from app.services.pipeline import pipeline_service
from src.utils.config_loader import cfg
from src.utils.file_utils import sanitize_filename

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Basic health check for load balancers and uptime monitoring."""
    return HealthResponse(status="ok", service=get_settings().app_name)


@router.post("/api/query", response_model=QueryResponseModel)
async def query(request: QueryRequest) -> QueryResponseModel:
    """Validate a query request, execute QueryPipeline.ask(), and return JSON."""
    try:
        return pipeline_service.ask(request)
    except ValueError as exc:
        logger.warning("Invalid query request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except (TimeoutError, ConnectionError) as exc:
        logger.exception("Pipeline dependency unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The query service is temporarily unavailable. Please try again.",
        ) from exc
    except ValidationError as exc:
        logger.warning("Pipeline response validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The query service returned an invalid response.",
        ) from exc
    except Exception as exc:
        logger.exception("Query pipeline failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The query could not be completed.",
        ) from exc


@router.post("/api/run-pipeline", response_model=PipelineRunResponse)
async def run_pipeline(request: PipelineRunRequest) -> PipelineRunResponse:
    """Ingest a YouTube channel URL and rebuild the query index."""
    try:
        return pipeline_service.run_pipeline(request)
    except ValueError as exc:
        logger.warning("Invalid pipeline run request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except (TimeoutError, ConnectionError) as exc:
        logger.exception("Pipeline run dependency unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The ingestion service is temporarily unavailable. Please try again.",
        ) from exc
    except Exception as exc:
        logger.exception("Pipeline run failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The pipeline could not be completed.",
        ) from exc


@router.get("/api/pipeline/{handle}/videos")
async def pipeline_videos(handle: str) -> dict:
    """Return saved video records for a pipeline handle."""
    safe_handle = _safe_handle(handle)
    path = Path(cfg.paths.raw_data) / f"{safe_handle}_videos.jsonl"
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No video list found for handle '{safe_handle}'.",
        )

    videos = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                data = json.loads(line)

                videos.append({
                    "title": data.get("title"),
                    "duration": data.get("duration"),
                    "url": data.get("url"),
                    "channel": data.get("channel"),
                    "view_count": data.get("view_count"),
                })

    return {
        "handle": safe_handle,
        "count": len(videos),
        "videos": videos
    }

@router.get("/api/pipeline/{handle}/transcripts")
async def pipeline_transcripts(handle: str) -> dict:
    """Return transcript files and basic metadata for a pipeline handle."""
    safe_handle = _safe_handle(handle)
    transcript_dir = Path(cfg.paths.raw_data) / "transcripts" / safe_handle
    if not transcript_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No transcript folder found for handle '{safe_handle}'.",
        )

    transcripts = []
    for path in sorted(transcript_dir.glob("*.json")):
        data = _read_json_file(path)
        segments = data.get("transcript_segments", [])
        transcript_text = " ".join(seg.get("text", "") for seg in segments)

        transcripts.append(
            {
                "video_id": data.get("video_id", path.stem),
                "title": data.get("title", "Untitled video"),
                "url": data.get("url", ""),
                "text": str(transcript_text),
            }
        )

    return {"handle": safe_handle, "count": len(transcripts), "transcripts": transcripts}


@router.get("/api/pipeline/{handle}/transcripts/{name}/download")
async def download_transcript(handle: str, name: str):
    safe_handle = _safe_handle(handle)
    safe_name = Path(name).stem

    transcript_path = Path(cfg.paths.raw_data) / "transcripts" / safe_handle / f"{safe_name}.json"

    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="Transcript not found")

    data = _read_json_file(transcript_path)
    segments = data.get("transcript_segments", [])

    transcript_text = " ".join(
        seg.get("text", "").replace(">>", "").strip()
        for seg in segments
    )

    # Build markdown in memory
    md_content = f"# {data.get('title', 'Untitled video')}\n\n{transcript_text}"

    return Response(
        content=md_content,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f'attachment; filename="transcript_{safe_handle}_{safe_name}.md"'
        }
    )


@router.get("/api/pipeline/{handle}/chunks")
async def pipeline_chunks(handle: str) -> dict:
    """Return chunk details generated from cached transcripts for a handle."""
    from src.processing.chunker import create_chunks

    safe_handle = _safe_handle(handle)
    transcript_dir = Path(cfg.paths.raw_data) / "transcripts" / safe_handle
    if not transcript_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No transcript folder found for handle '{safe_handle}'.",
        )

    videos = []
    total_chunks = 0
    for path in sorted(transcript_dir.glob("*.json")):
        data = _read_json_file(path)
        chunks = create_chunks(data)
        total_chunks += len(chunks)
        videos.append(
            {
                "video_id": data.get("video_id", path.stem),
                "title": data.get("title", "Untitled video"),
                "url": data.get("url", ""),
                "chunk_count": len(chunks),
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "chunk_index": chunk.chunk_index,
                        "start_time": chunk.start_time,
                        "timestamp_link": chunk.timestamp_link,
                        "token_count": chunk.token_count,
                        "preview": chunk.text[:320],
                    }
                    for chunk in chunks
                ],
            }
        )

    return {
        "handle": safe_handle,
        "video_count": len(videos),
        "chunk_count": total_chunks,
        "videos": videos,
    }


@router.post("/api/run-pipeline/html", response_class=HTMLResponse)
async def run_pipeline_html(request: Request) -> HTMLResponse:
    """HTMX endpoint that runs ingestion and reveals the question form."""
    form = await _read_urlencoded_form(request)
    url = str(form.get("url", "")).strip()
    try:
        response = await run_pipeline(PipelineRunRequest(url=url))
    except HTTPException as exc:
        return HTMLResponse(
            _render_error(str(exc.detail)),
            status_code=exc.status_code,
        )
    return HTMLResponse(_render_pipeline_ready(response))


@router.get("/api/run-pipeline/html", include_in_schema=False)
async def run_pipeline_html_get() -> RedirectResponse:
    """Redirect direct browser visits for the POST-only HTMX endpoint."""
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/api/query/html", response_class=HTMLResponse)
async def query_html(request: Request) -> HTMLResponse:
    """HTMX-friendly endpoint that renders a small result fragment."""
    form = await _read_urlencoded_form(request)
    query_text = str(form.get("query", "")).strip()
    try:
        response = await query(QueryRequest(query=query_text))
    except HTTPException as exc:
        return HTMLResponse(
            _render_error(str(exc.detail)),
            status_code=exc.status_code,
        )
    return HTMLResponse(_render_answer(response))


@router.get("/api/query/html", include_in_schema=False)
async def query_html_get() -> RedirectResponse:
    """Redirect direct browser visits for the POST-only HTMX endpoint."""
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


def _render_pipeline_ready(response: PipelineRunResponse) -> str:
    return (
        '<section class="answer-card" aria-live="polite">'
        "<h2>Ready for questions</h2>"
        f"<p>{_escape(response.message)}</p>"
        '<dl class="stats">'
        f"<div><dt>Videos found</dt><dd>{response.videos_found}</dd></div>"
        f"<div><dt>Transcripts</dt><dd>{response.transcripts_downloaded}</dd></div>"
        f"<div><dt>Chunks indexed</dt><dd>{response.chunks_indexed}</dd></div>"
        "</dl>"
        "</section>"
        '<form class="query-panel question-panel" hx-post="/api/query/html" '
        'hx-target="#answer-result" hx-swap="innerHTML">'
        '<label for="query">Ask a question about this channel</label>'
        '<textarea id="query" name="query" required minlength="1" '
        'placeholder="What are the main ideas from this channel?"></textarea>'
        '<div class="actions">'
        '<button type="submit">Ask question</button>'
        '<span class="loading" role="status">Searching and generating an answer...</span>'
        "</div>"
        "</form>"
        '<div id="answer-result" aria-live="polite"></div>'
    )


def _render_answer(response: QueryResponseModel) -> str:
    sources_html = ""
    if response.sources:
        source_items = []
        for source in response.sources:
            title = _escape(str(source.get("title", "Untitled source")))
            url = source.get("url")
            if url:
                source_items.append(
                    f'<li><a href="{_escape(str(url))}" target="_blank" rel="noopener">{title}</a></li>'
                )
            else:
                source_items.append(f"<li>{title}</li>")
        sources_html = f"<h3>Sources</h3><ul>{''.join(source_items)}</ul>"

    return (
        '<section class="answer-card" aria-live="polite">'
        "<h2>Answer</h2>"
        f"<p>{_escape(response.answer).replace(chr(10), '<br>')}</p>"
        f"{sources_html}"
        "</section>"
    )


def _render_error(message: str) -> str:
    return (
        '<section class="error-card" role="alert">'
        "<h2>Something went wrong</h2>"
        f"<p>{_escape(message)}</p>"
        "</section>"
    )


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


async def _read_urlencoded_form(request: Request) -> dict[str, str]:
    """Parse simple HTMX form submissions without requiring python-multipart."""
    body = (await request.body()).decode("utf-8")
    parsed = parse_qs(body, keep_blank_values=True)
    return {key: values[0] if values else "" for key, values in parsed.items()}


def _safe_handle(handle: str) -> str:
    safe_handle = sanitize_filename(handle)
    if not safe_handle:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A valid handle is required.",
        )
    
    return safe_handle


def _read_json_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
