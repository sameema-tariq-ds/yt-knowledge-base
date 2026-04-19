"""
transcript_fetcher.py
─────────────────────
Downloads transcripts for each video using youtube-transcript-api.
Falls back to yt-dlp auto-captions if the primary method fails.

Transcripts are the KEY data source — if a video has no transcript
(auto-generated or manual), we skip it rather than guess from audio.
"""

import os
import time
import json
import threading
from pathlib import Path
from json import JSONDecodeError
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from tenacity import retry, stop_after_attempt, before_sleep_log, wait_exponential, RetryError
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

from src.utils.logger import logging
from src.utils.config_loader import cfg
from src.utils.file_utils import sanitize_filename


logger = logging.getLogger(__name__)

_thread_local = threading.local()

def _get_ytt_api() -> YouTubeTranscriptApi:
    api = getattr(_thread_local, "ytt_api", None)
    if api is None:
        api = YouTubeTranscriptApi()
        _thread_local.ytt_api = api
    return api

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=False,  # Return None on final failure instead of raising
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _fetch_transcript_with_retry(video_id: str, languages: list[str]) -> list[dict] | None:
    """
    Fetch transcript with automatic retry on transient errors.
    Returns list of {text, start, duration} dicts or None if unavailable.

    tenacity behavior:
      Attempt 1 fails → wait 2s → Attempt 2 fails → wait 4s → Attempt 3 fails → return None
      reraise=False means None is returned instead of raising RetryError.
    """
    transcript_list = _get_ytt_api().list(video_id)

    # Try requested languages first, then any available language
    try:
        transcript = transcript_list.find_transcript(languages)
    except NoTranscriptFound:
        # Fallback: take whatever is available (usually auto-generated English)
        logger.warning("Manual transcript not found, trying auto-generated", extra={"video_id": video_id})
        transcript = transcript_list.find_generated_transcript(languages)

    return transcript.fetch()


def fetch_transcript(video: dict) -> Optional[Dict]:
    """
    Fetch the transcript for a single video.

    Args:
        video: Dict with at least 'video_id', 'title', 'channel'

    Returns:
        Enriched dict with 'transcript_segments' and 'full_text', or None.
    """
    video_id = video["video_id"]
    logger.debug(f"Fetching transcript: {video_id} — {video.get('title', '')[:60]}")

    try:
        fetched_transcript = _fetch_transcript_with_retry(
            video_id,
            languages=cfg.ingestion.languages,
        )
    except TranscriptsDisabled:
        logger.warning("Transcripts disabled for video", extra={"video_id": video_id})
        return None
    except VideoUnavailable:
        logger.warning("Video unavailable", extra={"video_id": video_id})
        return None
    except NoTranscriptFound:
        # Raised by find_generated_transcript when nothing exists in any language
        logger.warning("No transcript in any language", extra={"video_id": video_id})
        return None
    except RetryError as e:
        logger.error("All retry attempts failed", extra={"video_id": video_id, "error": str(e)})
        return None
    except Exception as e:
        logger.exception("Unexpected error fetching transcript", extra={"video_id": video_id})
        return None

    if not fetched_transcript:
        logger.warning("Empty transcript returned", extra={"video_id": video_id})
        return None

    # Normalize: youtube-transcript-api can return either raw dicts or
    # FetchedTranscript objects depending on version — handle both
    if hasattr(fetched_transcript, "to_raw_data"):
        transcript_segments = fetched_transcript.to_raw_data()
        full_text = " ".join(seg.text for seg in fetched_transcript)
        segment_count = len(fetched_transcript)
    else:
        transcript_segments = fetched_transcript  # type: ignore[assignment]
        full_text = " ".join(seg["text"] for seg in transcript_segments)  # type: ignore[index]
        segment_count = len(transcript_segments)  # type: ignore[arg-type]

    full_text = full_text.replace("\n", " ").strip()

    return {
        **video,                          # Copy all metadata from video dict
        "transcript_segments": transcript_segments,
        "full_text": full_text,           # Flattened single string
        "segment_count": segment_count,
        "transcript_length": len(full_text.split()) if full_text else 0,
        "transcript_char_length": len(full_text),
    }


def fetch_all_transcripts(videos: list[dict], channel_handle: str | None = None, max_workers: int = 5) -> list[dict]:
    """
    Fetch transcripts for all videos using a thread pool.

    Threading rationale:
      This function is I/O-bound (network calls to YouTube).
      The CPU is idle ~95% of the time in the sequential version.
      ThreadPoolExecutor lets us have N requests in-flight simultaneously
      while properly managing rate limiting and shared state.

    Args:
        videos:      List of video dicts from channel_scraper
        save_dir:    Directory to cache individual transcript JSONs
        max_workers: Number of concurrent threads (default 5).
                     Don't exceed 10 — YouTube will rate-limit you.

    Returns:
        List of videos that successfully fetched transcripts
    """
    channel_handle = sanitize_filename(channel_handle) if channel_handle else None

    save_path = Path(cfg.paths.raw_data) / "transcripts" / channel_handle
    save_path.mkdir(parents=True, exist_ok=True)

    # Thread-safe containers
    success: list[dict] = []
    failed:  list[str]  = []
    success_lock = threading.Lock()
    failed_lock  = threading.Lock()

    # Per-video file lock registry — prevents two threads writing the same file
    # Each video gets its own lock and its value is threading.Lock(), {"video1": <Lock object>}
    file_locks: dict[str, threading.Lock] = {
        v["video_id"]: threading.Lock() for v in videos
    }

    # Token bucket for rate limiting across all threads
    # Each thread must acquire a token before making a request
    rate_limiter = _RateLimiter(
        rate=max_workers,                          # N requests per window
        period=cfg.ingestion.rate_limit_seconds,   # per this many seconds
    )

    def fetch_one(video: dict) -> None:
        """Worker function — runs inside a thread."""
        video_id   = video["video_id"]
        cache_file = save_path / f"{video_id}.json"
        file_lock  = file_locks[video_id]

        # ── Step 1: Only one thread Check cache (under file lock) if file exists or not?──────────────────
        # Check if transcript file is downloaded for video1 or not. If yes, no need to run fetch_transcript() function.
        with file_lock:
            if cache_file.exists():
                logger.debug(f"Cache hit: {video_id}")
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (JSONDecodeError, OSError) as e:
                    logger.warning(
                        f"Corrupted cache file; refetching: {cache_file}",
                        extra={"video_id": video_id, "error": str(e)},
                    )
                    try:
                        cache_file.unlink(missing_ok=True)
                    except OSError:
                        pass
                    data = None
                if data is None:
                    pass
                else:
                    with success_lock:
                        success.append(data)
                    return   # Skip fetch entirely

        # ── Step 2: Acquire rate limit token ───────────────────────
        # This blocks until it's safe to fire another request.
        # Threads queue up here instead of hammering YouTube.
        rate_limiter.acquire()

        logger.debug(
            f"[Thread {threading.current_thread().name}] "
            f"Fetching {video_id} | "
            f"Tokens remaining: {rate_limiter.tokens:.1f}"
        )

        # ── Step 3: Fetch the transcript ───────────────────────────
        result = fetch_transcript(video)  # Network call — releases GIL

        # ── Step 4: Write to cache and record result ────────────────
        if result:
            with file_lock:                          # Protect file write
                tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
                try:
                    with open(tmp_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_file, cache_file)
                finally:
                    try:
                        tmp_file.unlink(missing_ok=True)
                    except OSError:
                        pass
            with success_lock:
                success.append(result)
        else:
            with failed_lock:
                failed.append(video_id)

    # ── Main execution ────────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Each video runs in parallel
        futures = {executor.submit(fetch_one, v): v for v in videos}

        # tqdm over futures as they complete (not submission order)
        with tqdm(total=len(videos), desc="Fetching transcripts", unit="video") as bar:
            for future in as_completed(futures):
                video = futures[future]
                try:
                    future.result()   # Re-raises any exception from the thread
                except Exception as e:
                    logger.error(f"Unexpected error for {video.get('video_id')}: {e}", exc_info=True)
                    with failed_lock:
                        failed.append(video.get("video_id", "unknown"))
                finally:
                    bar.update(1)

    logger.info(
        f"Transcripts: {len(success)} succeeded, {len(failed)} failed"
        + (f"\nFailed IDs: {failed[:10]}" if failed else "")
    )
    return success


class _RateLimiter:
    """
    Token bucket rate limiter for use across multiple threads.

    Guarantees that no more than `rate` requests are made per `period`
    seconds across all threads combined.

    Example: rate=5, period=2.0 → max 5 requests every 2 seconds.

    Why token bucket over simple sleep?
      Simple sleep(2) per thread = N threads × 1 req / 2s = N/2 req/s
      That's unpredictable. Token bucket gives a hard ceiling.
    """

    def __init__(self, rate: int, period: float):
        self.rate      = rate      # Max requests per period
        self.period    = period    # Time window in seconds
        self.tokens    = rate      # Start full
        self.last_refill = time.monotonic()
        self._lock     = threading.Lock()

    def acquire(self) -> None:
        """
        Block until a token is available, then consume one.
        Called by each thread before making a network request.
        """
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return          # Token acquired — proceed with request
            # No token available — wait outside the lock so other
            # threads can also check and refill
            time.sleep(0.05)

    def _refill(self) -> None:
        """
        Add tokens based on elapsed time since last refill.
        Has enough time passed to refill tokens? If yes: Reset tokens back to full capacity
        last_refill = 10.0
        now = 12.3
        elapsed = 2.3 seconds
        period = 2 seconds
        elapsed = 2.3 → ✅ refill
        elapsed = 1.5 → ❌ not yet
        """
        now = time.monotonic() # work like stopwatch, always move forward
        elapsed = now - self.last_refill
        if elapsed >= self.period:
            self.tokens      = self.rate    # Full refill
            self.last_refill = now

