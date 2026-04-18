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
import json
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, String, Integer, DateTime, Text, Float, Boolean
)
from sqlalchemy.orm import DeclarativeBase, Session

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
    indexed_at       = Column(DateTime, default=datetime.utcnow)
    chunk_count      = Column(Integer, default=0)


class MetadataStore:
    """Thin wrapper around SQLite via SQLAlchemy."""

    def __init__(self):
        db_path = Path(cfg.paths.sqlite_db)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,           # Set True to log all SQL (very verbose)
        )
        Base.metadata.create_all(self.engine)
        logger.info(f"SQLite store ready at {db_path}")

    def upsert_video(self, video_data: dict) -> None:
        """Insert or update a video record."""
        with Session(self.engine) as session:
            existing = session.get(VideoRecord, video_data["video_id"])
            if existing:
                # Update fields that may have changed
                for key, val in video_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, val)
            else:
                record = VideoRecord(
                    video_id=video_data["video_id"],
                    title=video_data.get("title", ""),
                    channel=video_data.get("channel", ""),
                    url=video_data.get("url", ""),
                    duration=video_data.get("duration", 0),
                    upload_date=video_data.get("upload_date", ""),
                    description=video_data.get("description", ""),
                    view_count=video_data.get("view_count", 0),
                    has_transcript=bool(video_data.get("full_text")),
                    transcript_len=video_data.get("transcript_length", 0),
                )
                session.add(record)
            session.commit()

    def update_chunk_count(self, video_id: str, count: int) -> None:
        with Session(self.engine) as session:
            video = session.get(VideoRecord, video_id)
            if video:
                video.chunk_count = count
                session.commit()

    def get_all_video_ids(self) -> list[str]:
        with Session(self.engine) as session:
            return [row[0] for row in session.execute(
                session.query(VideoRecord.video_id)
            )]

    def get_video(self, video_id: str) -> dict | None:
        with Session(self.engine) as session:
            record = session.get(VideoRecord, video_id)
            if not record:
                return None
            return {
                col.name: getattr(record, col.name)
                for col in VideoRecord.__table__.columns
            }