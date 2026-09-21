from __future__ import annotations

import asyncio
import json

import typer

from moviejev.config import get_settings
from moviejev.wiring import build_pipeline

app = typer.Typer(add_completion=False, help="moviejev CLI")


@app.command()
def recommend(prompt: str, as_json: bool = typer.Option(False, "--json")) -> None:
    """Get recommendations for a free-text request."""
    pipeline = build_pipeline(get_settings())
    res = asyncio.run(pipeline.run(prompt))
    if as_json:
        typer.echo(res.model_dump_json(indent=2))
        return
    typer.echo(f"reranker: {res.reranker}")
    typer.echo(f"profile: {res.profile.as_state()}\n")
    for i, r in enumerate(res.recommendations, 1):
        flag = "  [low confidence]" if r.low_confidence else ""
        year = f" ({r.movie.year})" if r.movie.year else ""
        typer.echo(f"{i}. {r.movie.title}{year}  score={r.score:.3f}  via={r.reason}{flag}")
        typer.echo(f"   {r.explanation}")
    if res.dropped_unverified:
        typer.echo(f"\ndropped (not in catalog): {json.dumps(res.dropped_unverified)}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the HTTP API (dev only; put it behind TLS for anything else)."""
    import uvicorn

    uvicorn.run("moviejev.api:app", host=host, port=port, reload=False)
