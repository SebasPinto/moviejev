# moviejev

LLM movie recommender that uses **Jev** (TypeSafe AI's System One model) as a calibrated reranker.

An LLM interprets your request and proposes candidates; every candidate is verified against TMDB
(no hallucinated titles reach the output); Jev scores each verified candidate against your taste
profile with a calibrated probability distribution; candidates are re-ranked by expected fit,
penalised by the probability that they violate an explicit exclusion; the LLM then writes a
one-line explanation for the top-K.

The reranker is swappable (`RERANKER=jev | llm_judge | none`) so Jev can be measured against an
LLM-as-judge and against the LLM's own ordering under the same rubric.

## Status

**POC** (v0.1.0). Works end to end. No persistence, no auth, no offline evaluation yet.
See [docs/roadmap.md](docs/roadmap.md).

## Architecture

```
prompt ──▶ LLM: taste profile (JSON, validated)
             │
             ▼
           LLM: N candidate titles (JSON, validated, deduped)
             │
             ▼
           TMDB: resolve each title ──▶ not found → dropped (reported)
             │
             ▼
           Reranker.judge(profile, movies) ──▶ Verdict per movie
             │   jev:        1 System One call / movie, in parallel
             │               Score(fit, 5 levels) · Choice(reason) · Noul(violates exclusion)
             │   llm_judge:  same rubric, text LLM, no calibrated confidence
             │   none:       LLM's own order
             ▼
           rank(): expected_fit × (1 − P(violation)), tiebreak on confidence
             │
             ▼
           LLM: one-sentence explanation for top-K ──▶ RecommendResponse
```

More detail: [docs/architecture.md](docs/architecture.md) · decisions: [docs/adr/](docs/adr/).

## Run it

Requirements: Python ≥ 3.12, [uv](https://docs.astral.sh/uv/), keys for TMDB, one LLM provider,
and TypeSafe (early access).

```bash
cp .env.example .env        # fill in keys
uv sync --extra dev
uv run moviejev recommend "something like Arrival but lighter, no horror"
uv run moviejev recommend "quiet 90s japanese dramas" --json
uv run moviejev serve         # http://127.0.0.1:8000/docs
```

```bash
curl -s localhost:8000/recommend -H 'content-type: application/json' \
  -d '{"prompt":"heist movies with a sense of humour"}' | jq .
```

Docker:

```bash
docker build -t moviejev .
docker run --rm --env-file .env -p 8000:8000 moviejev
```

## Configuration

Everything is an environment variable; see [.env.example](.env.example). Key knobs:

| Var | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `ANTHROPIC_WORKSPACE_ID` | unset | sent as `anthropic-workspace-id`; required when the key is not scoped to a workspace |
| `RERANKER` | `jev` | `jev`, `llm_judge`, `none` |
| `CANDIDATES` | `15` | titles the LLM proposes before verification |
| `TOP_K` | `5` | recommendations returned |
| `MIN_CONFIDENCE` | `0.35` | Jev confidence below which a verdict is flagged `low_confidence` |

## Development

```bash
uv run ruff check . && uv run ruff format .
uv run mypy src
uv run pytest -q          # offline: LLM/catalog fakes, Jev mocked with respx
uv run pip-audit
```

## Offline evaluation

Does Jev beat an LLM judge and random order on the same rubric? `scripts/eval_movielens.py`
holds out each MovieLens user's last 5 liked movies, mixes them with 15 popular unseen ones,
and lets every reranker order the same 20 against the same profile (ADR 0004).

```bash
uv run python scripts/eval_movielens.py --users 50            # ~$4 in LLM-judge calls
uv run python scripts/eval_movielens.py --users 5 --rerankers none,jev   # cheap smoke test
```

LLM outputs are cached in `data/eval/` so reruns only hit Jev. Results: [docs/eval.md](docs/eval.md).

## Security notes

- Secrets only via env; `SecretStr` prevents accidental logging.
- All LLM output is untrusted input: parsed as JSON and validated with pydantic, bounded lengths.
- The user prompt is wrapped and the LLM is told not to follow instructions inside it (prompt
  injection mitigation, not a guarantee).
- TMDB responses validated; a failed lookup drops the title, never crashes the request.
- Provider errors are never echoed to API clients.
- The dev server binds to `127.0.0.1` and has no auth or rate limiting: put it behind TLS and an
  auth layer before exposing it. See [docs/threat-model.md](docs/threat-model.md).

## License

MIT
