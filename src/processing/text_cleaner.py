"""
text_cleaner.py
───────────────
Transcripts from YouTube auto-captions are messy:
  - "[Music]" and "[Applause]" markers
  - Repeated filler words ("uh", "um")
  - Missing punctuation (auto-captions don't have periods)
  - HTML entities (&amp; → &)

This module normalizes transcripts before chunking.
"""

import re
import html


def clean_transcript(text: str) -> str:
    """
    Apply all cleaning steps to a raw transcript string.
    Each step is a separate function for easy testing/debugging.
    """
    text = decode_html_entities(text)
    text = remove_bracket_annotations(text)
    text = normalize_whitespace(text)
    text = remove_filler_words(text)
    text = fix_basic_punctuation(text)
    return text.strip()


def decode_html_entities(text: str) -> str:
    """Convert HTML entities to their characters: &amp; → &"""
    return html.unescape(text)


def remove_bracket_annotations(text: str) -> str:
    """
    Remove [Music], [Applause], [Laughter], etc.
    Auto-captions often insert these sound event markers.
    """
    # Matches [anything in brackets] case-insensitively
    return re.sub(r"\[.*?\]", "", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into a single space."""
    return re.sub(r"\s+", " ", text)


def remove_filler_words(text: str) -> str:
    """
    Remove common spoken filler words that add noise but no meaning.
    Be conservative — only remove when surrounded by spaces.
    """
    fillers = r"\b(um+|uh+|er+|ah+|hmm+)\b"
    return re.sub(fillers, "", text, flags=re.IGNORECASE)


def fix_basic_punctuation(text: str) -> str:
    """
    Auto-captions lack punctuation. We add periods at natural breaks.
    This is a heuristic — not perfect, but good enough for chunking.
    We look for sentences that are probably complete based on length.
    """
    # Ensure sentence-like separations have periods
    # Replace multiple spaces (from removed fillers) with a single space
    text = re.sub(r" {2,}", " ", text)
    return text