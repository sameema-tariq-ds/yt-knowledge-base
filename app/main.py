"""FastAPI application entry point."""

import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import get_settings


def configure_logging() -> None:
    """Configure concise application logging."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    configure_logging()
    settings = get_settings()
    application = FastAPI(title=settings.app_name)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    frontend_dir = Path(__file__).parent / "frontend"
    application.mount(
        "/static",
        StaticFiles(directory=frontend_dir),
        name="static",
    )

    application.include_router(router)

    @application.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(frontend_dir / "index.html")

    return application


app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.app_env == "development",
    )
