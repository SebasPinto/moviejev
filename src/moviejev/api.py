from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from moviejev import __version__
from moviejev.config import get_settings
from moviejev.llm.base import LLMError
from moviejev.models import RecommendResponse
from moviejev.pipeline import MAX_PROMPT_CHARS, Pipeline
from moviejev.reranker.jev import JevError
from moviejev.wiring import build_pipeline

_pipeline: Pipeline | None = None


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    global _pipeline
    _pipeline = build_pipeline(get_settings())
    yield


app = FastAPI(
    title="moviejev", version=__version__, lifespan=_lifespan, docs_url="/docs", redoc_url=None
)


class RecommendRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest) -> RecommendResponse:
    assert _pipeline is not None
    try:
        return await _pipeline.run(req.prompt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except (LLMError, JevError) as e:
        # Never leak provider payloads to the client.
        raise HTTPException(status_code=502, detail=f"upstream error: {type(e).__name__}") from e
