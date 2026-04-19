"""
cli.py
──────
Three entry points registered in pyproject.toml:
  yt-ingest  — fetch transcripts from a YouTube channel
  yt-build   — (re)build the vector database from cached transcripts
  yt-query   — interactive question answering
"""

import sys
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def ingest():
    """
    Entry point: yt-ingest
    Usage: yt-ingest <channel_url> [--handle <name>]

    Example:
      yt-ingest https://www.youtube.com/@3blue1brown --handle 3b1b
    """
    import argparse
    from src.ingestion.channel_scrapper import get_channel_video_ids, save_video_list
    from src.ingestion.transcript_fetcher import fetch_all_transcripts
    from src.ingestion.metadata_store import MetadataStore
    from src.utils.logger import get_logger

    logger = get_logger("cli.ingest")

    parser = argparse.ArgumentParser(description="Ingest a YouTube channel")
    parser.add_argument("channel_url", help="YouTube channel URL")
    parser.add_argument("--handle", default="channel", help="Short name for output files")
    args = parser.parse_args()

    console.print(Panel(
        f"[bold green]Ingesting:[/bold green] {args.channel_url}",
        title="YouTube Knowledge Base — Ingestion"
    ))

    try:
        # Step 1: Get all video IDs
        console.print("\n[cyan]Step 1/3:[/cyan] Fetching video list...")
        videos_metadata = list(get_channel_video_ids(args.channel_url))
        videos_count = save_video_list(videos_metadata, args.handle)
        console.print(f"  ✓ Found [bold]{videos_count}[/bold] videos")

        # Step 2: Download transcripts
        console.print("\n[cyan]Step 2/3:[/cyan] Downloading transcripts...")
        transcribed = fetch_all_transcripts(videos_metadata, args.handle)
        console.print(f"  ✓ Got transcripts for [bold]{len(transcribed)}[/bold] videos")

        # Step 3: Save metadata to SQLite
        console.print("\n[cyan]Step 3/3:[/cyan] Saving metadata to database...")
        store = MetadataStore()
        for v in transcribed:
            store.upsert_video(v)
        console.print(f"  ✓ Metadata saved")

        console.print(Panel(
            f"[bold green]Done![/bold green] Run [bold]yt-build[/bold] to create the vector database.",
            title="Ingestion Complete"
        ))

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        logger.exception("Ingestion failed")
        sys.exit(1)


def build_db():
    """
    Entry point: yt-build
    Reads all cached transcripts and builds/rebuilds the ChromaDB index.
    """
    from pathlib import Path
    from src.processing.chunker import create_chunks
    from src.storage.embedder import Embedder
    from src.storage.vector_store import VectorStore
    from src.ingestion.metadata_store import MetadataStore
    from src.utils.config_loader import cfg
    from tqdm import tqdm
    import json

    console.print(Panel("[bold blue]Building vector database...[/bold blue]"))

    transcript_dir = Path(cfg.paths.raw_data) / "transcripts"
    json_files     = list(transcript_dir.glob("*.json"))

    if not json_files:
        console.print("[red]No transcripts found. Run yt-ingest first.[/red]")
        sys.exit(1)

    embedder = Embedder()
    vector_store = VectorStore()
    metadata_store = MetadataStore()

    total_chunks = 0
    for json_file in tqdm(json_files, desc="Building index"):
        with open(json_file, "r", encoding="utf-8") as f:
            video_data = json.load(f)

        chunks = create_chunks(video_data)
        if not chunks:
            continue

        texts      = [c.text for c in chunks]
        embeddings = embedder.embed_texts(texts)

        added = vector_store.add_chunks(chunks, embeddings)
        metadata_store.update_chunk_count(video_data["video_id"], len(chunks))
        total_chunks += added

    console.print(Panel(
        f"[bold green]Vector DB ready![/bold green]\n"
        f"Total chunks indexed: [bold]{vector_store.count()}[/bold]",
        title="Build Complete"
    ))


def query():
    """
    Entry point: yt-query
    Interactive REPL for asking questions.
    """
    from src.query.pipeline import QueryPipeline
    from rich.markdown import Markdown

    console.print(Panel(
        "[bold]YouTube Knowledge Base[/bold]\nType your question and press Enter. "
        "Type [bold]quit[/bold] to exit.",
        title="Query Interface"
    ))

    pipeline = QueryPipeline()

    while True:
        try:
            question = console.input("\n[bold cyan]Your question:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue

        with console.status("[yellow]Searching and generating answer...[/yellow]"):
            try:
                response = pipeline.ask(question)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                continue

        # Display answer
        console.print("\n[bold green]Answer:[/bold green]")
        console.print(Markdown(response.answer))

        # Display sources table
        if response.sources:
            table = Table(title="Sources", show_header=True)
            table.add_column("Video Title", style="cyan", no_wrap=False)
            table.add_column("Relevance", style="green")
            table.add_column("Link")

            for src in response.sources:
                table.add_row(
                    src["title"][:60] + ("..." if len(src["title"]) > 60 else ""),
                    f"{src['similarity']:.0%}",
                    src["url"],
                )
            console.print(table)

        console.print(
            f"[dim]Query type: {response.query_type} | "
            f"Chunks used: {response.chunk_count} | "
            f"Model: {response.model_used}[/dim]"
        )