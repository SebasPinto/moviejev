# Changelog

## [Unreleased]
### Added
- Offline evaluation on MovieLens-latest-small (`scripts/eval_movielens.py`, ADR 0004):
  NDCG@10, Hit@1, Precision@5, MRR, tokens, cost and latency per reranker. LLM outputs and
  TMDB cards cached under `data/eval/`. Results in `docs/eval.md`.
- Token usage counters on LLM adapters and `JevReranker` (`.usage`).
- `TMDBCatalog.fetch(tmdb_id)` for id-based lookups.
- `ANTHROPIC_WORKSPACE_ID` setting, sent as the `anthropic-workspace-id` header for keys that
  are not scoped to a workspace.

### Changed
- `LLMJudgeReranker` now drops failed candidates instead of failing the request, mirroring Jev,
  and asks for the reason key explicitly (the shared rubric is unchanged).
- `TasteProfile.tone` / `.era` clip over-long LLM output instead of rejecting the profile.
- Committed `uv.lock`; CI installs with `--frozen`.
- Renamed the project, package and CLI entry point from `recjev` to `moviejev`.

## [0.1.0] - 2026-09-21
### Added
- POC pipeline: LLM profile → LLM candidates → TMDB verification → Jev rerank → LLM explanations.
- Swappable rerankers: `jev`, `llm_judge`, `none`.
- FastAPI `/recommend`, `/healthz`; Typer CLI.
- Offline test suite; CI with ruff, mypy, pytest, pip-audit.
