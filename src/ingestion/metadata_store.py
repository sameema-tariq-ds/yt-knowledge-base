"""
metadata_store.py
─────────────────
SQLite database for storing video metadata.
Why SQLite alongside ChromaDB?
  ChromaDB stores embeddings and does vector search.
  SQLite stores structured metadata and lets us do SQL queries
  like "show me all videos from 2023" or "find videos over 10 minutes."
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import (
    create_engine, Column, String, Integer, DateTime, Text, Boolean
)
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.exc import SQLAlchemyError

from src.utils.config_loader import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


class VideoRecord(Base):
    """One row per YouTube video."""
    __tablename__ = "videos"

    video_id         = Column(String(20), primary_key=True)
    title            = Column(Text, nullable=False)
    channel          = Column(String(200))
    url              = Column(Text)
    duration         = Column(Integer)         # seconds
    upload_date      = Column(String(20))      # YYYYMMDD from yt-dlp
    description      = Column(Text)
    view_count       = Column(Integer)
    has_transcript   = Column(Boolean, default=False)
    transcript_len   = Column(Integer, default=0)
    indexed_at       = Column(DateTime, default=datetime.utcnow) # when video is added to your system
    chunk_count      = Column(Integer, default=0)


def validate_video_data(video_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize incoming video data."""

    if not isinstance(video_data, dict):
        raise ValueError("video_data must be a dictionary")

    required_fields = ["video_id", "title"]
    for field in required_fields:
        if not video_data.get(field):
            raise ValueError(f"Missing required field: {field}")

    # Normalize types
    sanitized = {
        "video_id": str(video_data["video_id"]).strip(),
        "title": str(video_data.get("title", "")).strip(),
        "channel": str(video_data.get("channel", "")).strip(),
        "url": str(video_data.get("url", "")).strip(),
        "duration": int(video_data.get("duration", 0) or 0),
        "upload_date": str(video_data.get("upload_date", "")).strip(),
        "description": str(video_data.get("description", "")).strip(),
        "view_count": int(video_data.get("view_count", 0) or 0),
        "has_transcript": bool(video_data.get("full_text")),
        "transcript_len": int(video_data.get("transcript_length", 0) or 0),
    }

    return sanitized


class MetadataStore:
    """Thin wrapper around SQLite via SQLAlchemy."""

    def __init__(self):
        try:
            db_path = Path(cfg.paths.sqlite_db)
            db_path.parent.mkdir(parents=True, exist_ok=True)

            self.engine = create_engine(
                f"sqlite:///{db_path}",
                echo=False,           # Set True to log all SQL (very verbose)
                pool_pre_ping=True,  # Avoid Stale connections
                future=True
            )
            Base.metadata.create_all(self.engine)
            logger.info(f"SQLite store initialized", extra={"db_path": str(db_path)})
        
        except Exception as e:
            logger.exception("Failed to initialize database")
            raise RuntimeError("Database initialization failed") from e

    def upsert_video(self, video_data: dict) -> None:
        """Insert or update a video record."""
        try:
            video_data = validate_video_data(video_data)

            with Session(self.engine) as session:
                existing = session.get(VideoRecord, video_data["video_id"])
                if existing:
                    updated = False
                    # Update fields that may have changed
                    for key, val in video_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, val)
                            updated = True

                    if updated:
                        logger.debug("Updated video record", extra={"video_id": video_data["video_id"]})

                else:
                    record = VideoRecord(**video_data)
                    session.add(record)
                    logger.debug("Inserted new video record", extra={"video_id": video_data["video_id"]})
                    
                session.commit()

        except ValueError as ve:
            logger.warning("Validation failed", extra={"error": str(ve)})
            raise

        except SQLAlchemyError as db_err:
            logger.exception("Database error during upsert")
            raise RuntimeError("Database operation failed") from db_err
        
        except Exception as e:
            logger.exception("Failed to upsert video", extra={"video_id": video_data["video_id"]})
            raise


    def update_chunk_count(self, video_id: str, count: int) -> None:
        """Update chunk count for a video."""

        if not video_id:
            raise ValueError("video_id is required")

        if count < 0:
            raise ValueError("chunk_count cannot be negative")
        
        try:
            with Session(self.engine) as session:
                video = session.get(VideoRecord, video_id) # SELECT * FROM videos WHERE video_id = video_id LIMIT 1s

                if not video:
                    logger.warning("Video not found", extra={"video_id": video_id})
                    return
                
                video.chunk_count = count
                session.commit()

        except SQLAlchemyError:
            logger.exception("Failed to update chunk count")
            raise

    def get_all_video_ids(self) -> list[str]:
        """Return all stored video IDs."""

        try:
            with Session(self.engine) as session:
                return [row[0] for row in session.execute( # Executes the query and returns a result set (rows) ('abc123', ...), ('xyz456', ...)
                    session.query(VideoRecord.video_id) # SELECT video_id FROM videos
                )]
            
        except SQLAlchemyError:
            logger.exception("Failed to fetch video IDs")
            raise

    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve video metadata as dictionary."""
        if not video_id:
            raise ValueError("video_id is required")

        try:
            with Session(self.engine) as session:
                record = session.get(VideoRecord, video_id)
                if not record:
                    return None
                return {
                    col.name: getattr(record, col.name)
                    for col in VideoRecord.__table__.columns # give all columns defined in the table
                }
        except SQLAlchemyError:
            logger.exception("Failed to fetch video")
            raise