"""
chunker.py
──────────
Splits cleaned transcripts into overlapping chunks suitable for embedding.

Why overlapping chunks?
  If a question's answer spans a chunk boundary, overlap ensures neither
  chunk is missing critical context. 80-token overlap is a good default.

Why preserve timestamps?
  Each chunk includes the start timestamp of its content so users can
  jump directly to the relevant moment in the video.
"""

from dataclasses import dataclass, field
from typing import Optional
import tiktoken

from src.utils.config_loader import cfg
from src.utils.logger import get_logger

from src.processing.text_cleaner import clean_transcript

logger = get_logger(__name__)

# Use the same tokenizer as GPT models for consistent token counting
TOKENIZER = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    """A single text chunk with all metadata needed for retrieval."""
    chunk_id:    str            # Unique: "{video_id}_{chunk_index}"
    video_id:    str
    video_title: str
    channel:     str
    url:         str
    text:        str            # The actual content to embed
    start_time:  float          # Seconds into the video
    timestamp_link: str         # YouTube URL with &t=XXs parameter
    chunk_index: int
    total_chunks: int
    token_count: int

    def to_dict(self) -> dict:
        return self.__dict__


def count_tokens(text: str) -> int:
    """Count tokens using the cl100k tokenizer (same as OpenAI embeddings)."""
    return len(TOKENIZER.encode(text))


def segments_to_text_blocks(
    segments: list[dict],
    target_tokens: int
) -> list[dict]:
    """
    Merge transcript segments into blocks of ~target_tokens each.
    Preserves the start_time of the first segment in each block.

    Args:
        segments: List of {text, start, duration} from YouTube transcript API
        target_tokens: Approximate token count per block

    Returns:
        List of {text, start_time} dicts
    """
    blocks = []
    current_text = ""
    current_start = 0.0
    current_tokens = 0

    for seg in segments:
        seg_text = seg.get("text", "").strip()
        seg_tokens = count_tokens(seg_text)

        if current_tokens + seg_tokens > target_tokens and current_text:
            # Flush current block
            blocks.append({
                "text": current_text.strip(),
                "start_time": current_start,
            })
            current_text = seg_text
            current_start = seg.get("start", 0.0)
            current_tokens = seg_tokens
        else:
            if not current_text:
                current_start = seg.get("start", 0.0)
            current_text += " " + seg_text
            current_tokens += seg_tokens

    # Don't forget the last block
    if current_text.strip():
        blocks.append({
            "text": current_text.strip(),
            "start_time": current_start,
        })

    return blocks


def create_chunks(video_data: dict) -> list[Chunk]:
    """
    Convert a video's transcript into a list of Chunk objects.

    Strategy:
      1. Group raw segments into token-sized blocks
      2. Add overlapping context from neighboring blocks
      3. Wrap each block in a Chunk with full metadata

    Args:
        video_data: Dict with video metadata + transcript_segments + full_text

    Returns:
        List of Chunk objects ready for embedding
    """

    video_id    = video_data["video_id"]
    title       = video_data.get("title", "")
    channel     = video_data.get("channel", "")
    base_url    = video_data.get("url", f"https://youtube.com/watch?v={video_id}")
    segments    = video_data.get("transcript_segments", [])

    chunk_size    = cfg.processing.chunk_size
    chunk_overlap = cfg.processing.chunk_overlap
    min_length    = cfg.processing.min_chunk_length

    # Step 1: Build text blocks aligned with timestamps
    blocks = segments_to_text_blocks(segments, target_tokens=chunk_size)

    if not blocks:
        logger.warning(f"No blocks generated for {video_id}")
        return []

    # Step 2: Create overlapping windows over the blocks
    # Instead of overlapping at the token level, we overlap entire blocks
    # This is simpler and keeps timestamps accurate
    chunks = []
    num_blocks = len(blocks)

    for i, block in enumerate(blocks):
        # Build text: previous block tail + current block + next block head
        parts = []

        if i > 0:
            # Append last ~overlap tokens of previous block
            prev_text = blocks[i - 1]["text"]
            prev_tokens = TOKENIZER.encode(prev_text)
            overlap_tokens = prev_tokens[-chunk_overlap:]
            parts.append(TOKENIZER.decode(overlap_tokens))

        parts.append(block["text"])

        if i < num_blocks - 1:
            # Prepend first ~overlap tokens of next block
            next_text = blocks[i + 1]["text"]
            next_tokens = TOKENIZER.encode(next_text)
            overlap_tokens = next_tokens[:chunk_overlap]
            parts.append(TOKENIZER.decode(overlap_tokens))

        combined_text = " ".join(parts)
        cleaned_text  = clean_transcript(combined_text)

        # Skip chunks that are too short to be useful
        if len(cleaned_text) < min_length:
            continue

        start_time = block["start_time"]
        # Format: 1h23m45s for YouTube's timestamp format
        timestamp_str = _seconds_to_yt_timestamp(start_time)
        timestamp_link = f"{base_url}&t={int(start_time)}s"

        chunk = Chunk(
            chunk_id=       f"{video_id}_{i:04d}",
            video_id=       video_id,
            video_title=    title,
            channel=        channel,
            url=            base_url,
            text=           cleaned_text,
            start_time=     start_time,
            timestamp_link= timestamp_link,
            chunk_index=    i,
            total_chunks=   num_blocks,
            token_count=    count_tokens(cleaned_text),
        )
        chunks.append(chunk)

    logger.debug(f"Video {video_id}: {len(segments)} segments → {len(chunks)} chunks")
    return chunks


def _seconds_to_yt_timestamp(seconds: float) -> str:
    """Convert 3661.0 → '1:01:01'"""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"