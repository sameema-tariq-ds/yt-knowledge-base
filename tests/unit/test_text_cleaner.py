"""Unit tests for text_cleaner.py"""
import pytest
from src.processing.text_cleaner import (
    clean_transcript,
    remove_bracket_annotations,
    decode_html_entities,
    normalize_whitespace,
    remove_filler_words,
)


class TestBracketAnnotations:
    def test_removes_music_marker(self):
        assert remove_bracket_annotations("[Music]") == ""

    def test_removes_applause_marker(self):
        text = "Great point [Applause] and another."
        assert "[Applause]" not in remove_bracket_annotations(text)

    def test_preserves_regular_text(self):
        text = "This is regular text."
        assert remove_bracket_annotations(text) == text


class TestHTMLEntities:
    def test_decodes_ampersand(self):
        assert decode_html_entities("A &amp; B") == "A & B"

    def test_decodes_quotes(self):
        assert "&quot;" not in decode_html_entities("Say &quot;hello&quot;")


class TestFillerWords:
    def test_removes_um(self):
        result = remove_filler_words("So um the answer is yes")
        assert "um" not in result

    def test_removes_uh(self):
        result = remove_filler_words("It is uh complicated")
        assert " uh " not in result

    def test_preserves_words_with_um_inside(self):
        # "album" contains "um" but should NOT be removed
        result = remove_filler_words("Check this album out")
        assert "album" in result


class TestCleanTranscript:
    def test_full_pipeline(self):
        messy = "[Music] Hello um this is uh a &amp; test [Applause]"
        clean = clean_transcript(messy)
        assert "[Music]" not in clean
        assert "um" not in clean
        assert "&amp;" not in clean
        assert "test" in clean   # real content preserved