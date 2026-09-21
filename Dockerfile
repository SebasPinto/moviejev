FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock* README.md LICENSE ./
COPY src ./src
RUN uv sync --no-dev --frozen || uv sync --no-dev
RUN useradd -r -u 10001 app && chown -R app:app /app
USER app
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "moviejev.api:app", "--host", "0.0.0.0", "--port", "8000"]
