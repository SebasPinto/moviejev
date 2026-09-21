"""Offline eval: does Jev rerank better than an LLM judge or random order, and at what cost?

Usage:
    uv run python scripts/eval_movielens.py --users 50
    uv run python scripts/eval_movielens.py --users 5 --rerankers none,jev   # smoke test

Needs TMDB + LLM + TypeSafe keys in .env. LLM outputs are cached under data/eval so reruns
only hit Jev. Results go to data/eval/results-<timestamp>.json and stdout (markdown).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import typer

from moviejev.config import get_settings
from moviejev.eval.cache import CachingLLM, JsonCache
from moviejev.eval.runner import EvalConfig, markdown_table, run_eval
from moviejev.llm.factory import build_llm
from moviejev.reranker.base import Reranker
from moviejev.wiring import build_catalog, build_reranker

app = typer.Typer(add_completion=False)


@app.command()
def main(
    users: int = typer.Option(50, help="users to evaluate (after filtering)"),
    holdout: int = typer.Option(5, help="held-out positives per user"),
    negatives: int = typer.Option(15, help="popular unseen movies mixed in per user"),
    k: int = typer.Option(10, help="cutoff for NDCG@k / HitRate@k"),
    seed: int = typer.Option(42),
    rerankers: str = typer.Option("none,llm_judge,jev", help="comma-separated"),
    data_dir: str = typer.Option("data"),
    llm_price_in: float = typer.Option(3.0, help="USD per 1M LLM input tokens"),
    llm_price_out: float = typer.Option(15.0, help="USD per 1M LLM output tokens"),
    jev_price_in: float = typer.Option(0.042, help="USD per 1M Jev input tokens"),
    jev_price_out: float = typer.Option(0.0, help="USD per 1M Jev output tokens"),
) -> None:
    s = get_settings()
    logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)

    root = Path(data_dir)
    cfg = EvalConfig(
        users=users,
        holdout=holdout,
        negatives=negatives,
        k=k,
        seed=seed,
        rerankers=tuple(x.strip() for x in rerankers.split(",") if x.strip()),
        data_dir=root,
        llm_price_in=llm_price_in,
        llm_price_out=llm_price_out,
        jev_price_in=jev_price_in,
        jev_price_out=jev_price_out,
    )
    llm = CachingLLM(build_llm(s), JsonCache(root / "eval" / "llm_cache.json"))
    catalog = build_catalog(s)

    def make(name: str) -> Reranker:
        return build_reranker(s, name, llm)

    t0 = time.time()
    report = asyncio.run(run_eval(cfg, llm, catalog, make))
    report["wall_s"] = round(time.time() - t0, 1)
    report["llm_cache"] = {"hits": llm.hits, "misses": llm.misses}

    out = root / "eval" / f"results-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    typer.echo(markdown_table(report))
    typer.echo(
        f"\nusers={report['users_evaluated']}  profile LLM: {report['profile_llm']}  "
        f"cache: {report['llm_cache']}  wall={report['wall_s']}s\nsaved: {out}"
    )


if __name__ == "__main__":
    app()
