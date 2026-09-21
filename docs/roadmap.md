# Roadmap

## POC → MVP (what's missing)
- [x] `uv lock` committed; CI uses `--frozen`.
- [x] Offline evaluation on MovieLens-latest-small (`scripts/eval_movielens.py`, ADR 0004).
      Results in `docs/eval.md`.
- [ ] Persist runs (SQLite) so evals are reproducible and diffable.
- [ ] Auth (API key header) + rate limiting on `/recommend`.
- [ ] Retrieval-based candidate source (`TMDBDiscoverCatalog`) as an alternative to LLM proposals.
- [ ] Tag v0.2.0.

## MVP → v1
- Best-of-N: generate several candidate lists, let Jev pick the strongest set.
- Feedback loop: when mean fit is low, re-prompt the LLM with Jev's reasons.
- Jev as intent router (title / genre / mood / ambiguous) before profiling.
