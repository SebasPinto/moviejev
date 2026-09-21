# CLAUDE.md — moviejev

Guidance for Claude Code working in this repo. Read fully before touching code.

## What this is

LLM movie recommender with **Jev** (TypeSafe AI System One model) as a calibrated reranker.
Pipeline: LLM taste profile → LLM candidate titles → TMDB verification → Jev scores each
verified candidate → rank by `expected_fit × (1 − P(violation))` → LLM one-line explanations.

Status: **POC v0.1.0**. Next milestone: MVP (see `docs/roadmap.md`). Language of the code, docs
and commits: English. Conversation with the owner: Spanish is fine.

## Non-negotiables

- **Security first, even in a POC.** Secrets only via env (`config.py`, `SecretStr`). Never log
  or return a key. Never add HTTP-plain endpoints or unauthenticated internet exposure without
  flagging it. All LLM / Jev / TMDB output is untrusted: parse into pydantic models with bounded
  fields; never `dict["key"]` on raw JSON outside a validated model.
- **No hardcoded config.** New knobs go in `Settings` + `.env.example` + README table.
- **Quality gates must stay green:** `ruff check .`, `ruff format --check .`, `mypy src tests`
  (strict), `pytest -q`, `pip-audit`. CI runs all of them. Do not weaken `pyproject.toml` rules
  to make something pass; fix the code.
- **Tests are offline.** No network in tests: LLM → `FakeLLM`, catalog → `FakeCatalog`,
  Jev → `respx`. Every change to `reranker/jev.py` needs a respx test mirroring the documented
  response shape.
- **Semver + CHANGELOG.** Any user-visible change gets a CHANGELOG entry under `[Unreleased]`.
- **ADRs for reversible-but-costly decisions** in `docs/adr/NNNN-slug.md` (Context / Decision /
  Consequences). Existing: 0001 Jev is a reranker, 0002 httpx instead of SDK, 0003 per-candidate
  state. Read them before proposing changes to those areas.

## Commands

```bash
uv sync --extra dev                 # setup
uv run ruff check . && uv run ruff format .
uv run mypy src tests
uv run pytest -q
uv run pip-audit
uv run moviejev recommend "prompt"    # CLI
uv run moviejev serve                 # API on 127.0.0.1:8000, /docs
RERANKER=llm_judge uv run moviejev recommend "prompt"   # comparison baseline
```

Python 3.12+. Package manager is `uv` only (no pip, no poetry). Commit `uv.lock`.

## Layout

```
src/moviejev/
  config.py       Settings (pydantic-settings). Only place env is read.
  models.py       Domain types. TasteProfile/Movie have .as_state() for Jev/LLM prompts.
  llm/            LLM ABC (complete, complete_json) + anthropic_llm.py, openai_llm.py, factory.py
  catalog/        Catalog ABC + tmdb.py. resolve(Candidate) -> Movie | None
  reranker/       Reranker ABC + shared rubric in base.py; jev.py, llm_judge.py, passthrough.py
  pipeline.py     Orchestration only. No provider-specific code here.
  wiring.py       Builds object graph from Settings. Only place secrets are unwrapped.
  api.py, cli.py  Thin delivery layers. No business logic.
tests/            conftest.py has FakeLLM, FakeCatalog, movie() helper
docs/             architecture.md, threat-model.md, roadmap.md, adr/
```

## Design rules that are easy to break by accident

- The rubric (`FIT_LEVELS`, `REASON_CHOICES`, `VIOLATION_INSTRUCTION`) lives once in
  `reranker/base.py` and is shared by Jev and the LLM judge. Never fork it; a fair comparison
  depends on identical questions.
- Jev scoring uses the **expected value** of the Score distribution, not the argmax
  (`expected_value()`). Keep it that way; it is the reason Jev is interesting here.
- One Jev request per candidate with a minimal state (ADR 0003). Do not pack all candidates
  into one state to "save calls".
- Catalog verification always runs **before** the reranker. Jev cannot detect nonexistent titles.
- `confidence` is for flagging (`low_confidence`) and tie-breaking, not for hard filtering.
  Thresholds are product decisions; keep them in Settings.
- Per-candidate upstream failures drop that candidate and log a warning; they never fail the
  whole request. A 100 % failure raises.
- API errors to clients are generic (`502 upstream error: JevError`). Never echo provider bodies.

## External contracts (verify against docs before changing)

- Jev: `POST {TYPESAFE_BASE_URL}/v1/systemone`, bearer auth, body `{state, model, questions}`.
  Question types: `score` (ordered `criteria` list → `score`, `confidence`, `probabilities`
  keyed `"0".."n-1"`), `choice` (`criteria` map → `choice`, `confidence`, `probabilities`),
  `noul` (→ `noul` in [0,1]). Docs: https://docs.typesafe.ai
- TMDB: `GET /search/movie?query=&year=&include_adult=false`, v4 bearer token in
  `Authorization`. If the owner only has a v3 key, switch to `api_key` query param in
  `catalog/tmdb.py` and update `.env.example`.
- Anthropic / OpenAI via official async SDKs, `max_retries=2`, timeout from Settings.

## Roadmap (ordered; pick the top unchecked item unless told otherwise)

1. Commit `uv.lock`; CI already uses `--frozen`.
2. `scripts/eval_movielens.py`: MovieLens-latest-small, hold out each user's last 5 ratings,
   build a profile from the rest, run rerankers `none` / `llm_judge` / `jev`, report NDCG@10,
   HitRate@10, cost, p50/p95 latency → `docs/eval.md`. This is the item that answers "is Jev
   actually better/cheaper here". Cache LLM outputs on disk so reruns only hit Jev.
3. Auth (API-key header) + rate limiting on `/recommend`. Today the endpoint is a cost
   amplifier if exposed.
4. Persist runs to SQLite for reproducible, diffable evals.
5. `TMDBDiscoverCatalog`: retrieval-based candidates as an alternative to LLM proposals.
6. Tag v0.2.0 = MVP.
Later: best-of-N candidate lists judged by Jev; feedback loop re-prompting on low mean fit;
Jev as intent router before profiling.

## How to work with the owner

- Senior engineer; prefers trade-offs over prescriptions. Question decisions, propose
  alternatives, name what each one costs.
- Ask before decisions that are expensive to reverse (schema of persisted data, public API
  shape, replacing a provider). Do not ask for things you can decide and document in an ADR.
- Deliver complete, runnable code; no `...` placeholders.
- When a task lands you in RF / LoRa / radio-layer territory, stop and say so: out of scope for
  this repo.
- Before declaring a milestone done: run all gates, update CHANGELOG, update `docs/roadmap.md`,
  and state explicitly what is missing for the next stage.