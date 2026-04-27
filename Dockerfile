FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install system deps first (stable layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential  curl \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY dependency metadata first (for caching)
COPY pyproject.toml uv.lock ./

# Install dependencies into the system environment inside the container
RUN uv sync --frozen --no-cache

# Now copy application code (changes often)
COPY . .

# 4. Sync the project (installs your local package code)
RUN uv sync --frozen

# Use the -m flag for module execution
CMD ["uv", "run", "python", "-m", "app.main"]
