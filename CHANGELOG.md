# Changelog

## [Unreleased]
### Added
- `ANTHROPIC_WORKSPACE_ID` setting, sent as the `anthropic-workspace-id` header for keys that
  are not scoped to a workspace.

### Changed
- Committed `uv.lock`; CI installs with `--frozen`.
- Renamed the project, package and CLI entry point from `recjev` to `moviejev`.

## [0.1.0] - 2026-09-21
### Added
- POC pipeline: LLM profile → LLM candidates → TMDB verification → Jev rerank → LLM explanations.
- Swappable rerankers: `jev`, `llm_judge`, `none`.
- FastAPI `/recommend`, `/healthz`; Typer CLI.
- Offline test suite; CI with ruff, mypy, pytest, pip-audit.
