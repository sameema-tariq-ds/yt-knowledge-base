"""
channel_scraper.py
──────────────────
Uses yt-dlp to list all videos from a YouTube channel without downloading media.
Returns a list of dicts with video_id, title, duration, upload_date, description.

Why yt-dlp over pytube?
  yt-dlp is actively maintained, handles more edge cases, and doesn't break
  every few weeks when YouTube changes its internal API.
"""

import json
import time
import random
from pathlib import Path
from typing import Generator, Dict, Any

import yt_dlp
from yt_dlp.utils import DownloadError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential)

from src.utils.config_loader import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__) #"src.ingestion.channel_scraper" 


@retry(
    retry=retry_if_exception_type(DownloadError), # Retry only if network issue happen
    stop=stop_after_attempt(5),  # Retry up to 5 times
    wait=wait_exponential(multiplier=1, min=2, max=30),  # Wait longer exponentially between each retry (2sec, 4sec, 8sec, 16sec, 30sec)
    before_sleep=before_sleep_log(logger, logger.WARNING), # Log a message before each retry
    reraise=True,  # If all retries fail, raise the error again
)
def _extract_channel_info(ydl, channel_url: str) -> Dict[str, Any]:
    """
    Extract metadata for a YouTube channel using yt-dlp.

    Args:
        ydl (YoutubeDL): Configured yt-dlp instance used to perform extraction.

        channel_url (str): YouTube channel URL (e.g., https://www.youtube.com/@channel)

    Returns:
        Dict[str, Any]: Raw metadata dictionary returned by yt-dlp.
            Contains keys like:
                - 'entries': list of videos
                - 'channel': channel name
                - 'title': fallback name
    """
    return ydl.extract_info(channel_url, download=False)

def get_channel_video_ids(channel_url: str) -> Generator[dict, None, None]:
    """
    Fetch all video metadata from a YouTube channel URL.

    Args:
        channel_url: "https://www.youtube.com/c/3blue1brown"

    Returns:
        List of video metadata dicts.
    """
    logger.info(f"Starting channel scrape: {channel_url}")

    # yt-dlp options: list videos only, no download
    ydl_opts = {
        "quiet": True,               # Suppress yt-dlp's own output
        "no_warnings": True,
        "extract_flat": True,        # Don't download, just list
        "ignoreerrors": True,        # Skip deleted/private videos
        "playlistend": cfg.ingestion.max_videos,
        "socket_timeout": cfg.ingestion.max_socket_timeout,  # If server doesn’t respond within 15 seconds → stop waiting
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = _extract_channel_info(ydl, channel_url)

        if not info or "entries" not in info:
            logger.warning("No entries found in channel. Check the URL.")
            return []

        channel_name = info.get("channel", info.get("title", "Unknown Channel"))
        logger.info(f"Channel: {channel_name} | Found {len(info['entries'])} entries")

        for entry in info["entries"]:
            if entry is None:
                continue   # yt-dlp returns None for unavailable videos

            duration = entry.get("duration", 0) or 0

            # Skip YouTube Shorts if configured
            if cfg.ingestion.skip_shorts and duration < cfg.ingestion.min_duration_seconds:
                logger.debug(f"Skipping short video: {entry.get('id')} ({duration}s)")
                continue

            video_metadata = {
                "video_id":    entry.get("id", ""),
                "title":       entry.get("title", ""),
                "duration":    duration,
                "upload_date": entry.get("upload_date", ""),
                "url":         f"https://www.youtube.com/watch?v={entry.get('id', '')}",
                "channel":     channel_name,
                "description": entry.get("description", ""),
                "view_count":  entry.get("view_count", 0),
            }

            # Yield instead of storing → STREAMING
            yield video_metadata

            # rate limiting
            time.sleep(random.uniform(cfg.ingestion.request_delay_min, cfg.ingestion.request_delay_max))


def save_video_list(videos_metadata:  Generator[dict, None, None], channel_handle: str) -> Path:
    """
    Save videos incrementally to JSON (streaming write).
    Prevents memory blow-up for large channels.
    """
    output_dir = Path(cfg.paths.raw_data)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{channel_handle}_videos.jsonl"

    count =0 

    with open(output_path, "w", encoding="utf-8") as f:
        for video in videos_metadata:
            # Each video is written as one line
            f.write(json.dumps(video, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"Saved {len(videos_metadata)} video records → {output_path}")
    return output_path


'''
  min_duration_seconds: 60 # min duration of yt video

  max_socket_timeout: 15   # 
  request_delay_min: 0.5   # min number of requests a user or client can make to a server within a specific timeframe
  request_delay_max: 1     # min number of requests a user or client can make to a server within a specific timeframe

'''