from __future__ import annotations

import logging
from typing import Literal

from moviejev.catalog.tmdb import TMDBCatalog
from moviejev.config import Settings
from moviejev.llm.base import LLM
from moviejev.llm.factory import build_llm
from moviejev.pipeline import Pipeline
from moviejev.reranker import JevReranker, LLMJudgeReranker, PassthroughReranker, Reranker

RerankerName = Literal["jev", "llm_judge", "none"]


def build_catalog(s: Settings) -> TMDBCatalog:
    if not s.tmdb_api_key:
        raise RuntimeError("TMDB_API_KEY is required")
    return TMDBCatalog(s.tmdb_api_key.get_secret_value(), str(s.tmdb_base_url), s.request_timeout_s)


def build_reranker(s: Settings, name: str, llm: LLM) -> Reranker:
    if name == "jev":
        if not s.typesafe_api_key:
            raise RuntimeError("TYPESAFE_API_KEY is required for RERANKER=jev")
        return JevReranker(
            api_key=s.typesafe_api_key.get_secret_value(),
            base_url=str(s.typesafe_base_url),
            model=s.typesafe_model,
            timeout_s=s.request_timeout_s,
            min_confidence=s.min_confidence,
        )
    if name == "llm_judge":
        return LLMJudgeReranker(llm)
    if name == "none":
        return PassthroughReranker()
    raise ValueError(f"unknown reranker {name!r}")


def build_pipeline(s: Settings) -> Pipeline:
    logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    llm = build_llm(s)
    catalog = build_catalog(s)
    reranker = build_reranker(s, s.reranker, llm)
    return Pipeline(llm, catalog, reranker, s.candidates, s.top_k)
