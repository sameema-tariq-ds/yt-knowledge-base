"""Shared pytest fixtures available to all tests."""
import json
import pytest
from pathlib import Path

# Sample transcript with 3 segments (mimics YouTube transcript API output)
SAMPLE_TRANSCRIPT_SEGMENTS = [
    {"text": "Welcome to this video about machine learning.", "start": 0.0, "duration": 4.0},
    {"text": "Today we will cover neural networks and backpropagation.", "start": 4.0, "duration": 5.0},
    {"text": "A neural network is a series of algorithms that recognizes patterns.", "start": 9.0, "duration": 6.0},
    {"text": "Backpropagation is the algorithm used to train these networks.", "start": 15.0, "duration": 5.0},
    {"text": "It works by computing gradients of the loss function.", "start": 20.0, "duration": 5.0},
]

SAMPLE_VIDEO = {
    "video_id":    "test123",
    "title":       "Introduction to Machine Learning",
    "channel":     "Test Channel",
    "url":         "https://youtube.com/watch?v=test123",
    "duration":    300,
    "upload_date": "20240101",
    "description": "A beginner's guide to ML concepts.",
    "view_count":  10000,
    "transcript_segments": SAMPLE_TRANSCRIPT_SEGMENTS,
    "full_text":   " ".join(s["text"] for s in SAMPLE_TRANSCRIPT_SEGMENTS),
    "transcript_length": 200,
}


@pytest.fixture
def sample_video():
    return SAMPLE_VIDEO.copy()


@pytest.fixture
def sample_segments():
    return SAMPLE_TRANSCRIPT_SEGMENTS.copy()


@pytest.fixture
def sample_chunks(sample_video):
    from src.processing.chunker import create_chunks
    return create_chunks(sample_video)