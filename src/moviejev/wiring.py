from __future__ import annotations

import logging

from moviejev.catalog.tmdb import TMDBCatalog
from moviejev.config import Settings
from moviejev.llm.factory import build_llm
from moviejev.pipeline import Pipeline
from moviejev.reranker import JevReranker, LLMJudgeReranker, PassthroughReranker, Reranker


def build_pipeline(s: Settings) -> Pipeline:
    logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    llm = build_llm(s)
    if not s.tmdb_api_key:
        raise RuntimeError("TMDB_API_KEY is required")
    catalog = TMDBCatalog(
        s.tmdb_api_key.get_secret_value(), str(s.tmdb_base_url), s.request_timeout_s
    )

    reranker: Reranker
    if s.reranker == "jev":
        if not s.typesafe_api_key:
            raise RuntimeError("TYPESAFE_API_KEY is required for RERANKER=jev")
        reranker = JevReranker(
            api_key=s.typesafe_api_key.get_secret_value(),
            base_url=str(s.typesafe_base_url),
            model=s.typesafe_model,
            timeout_s=s.request_timeout_s,
            min_confidence=s.min_confidence,
        )
    elif s.reranker == "llm_judge":
        reranker = LLMJudgeReranker(llm)
    else:
        reranker = PassthroughReranker()
    return Pipeline(llm, catalog, reranker, s.candidates, s.top_k)
