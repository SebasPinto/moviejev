# Roadmap

## POC → MVP (what's missing)
- [x] `uv lock` committed; CI uses `--frozen`.
- [ ] Offline evaluation on MovieLens-latest-small: hold out each user's last 5 ratings, build a
      profile from the rest, run the three rerankers, report NDCG@10 / HitRate@10 and cost/latency.
      Script goes in `scripts/eval_movielens.py`, results in `docs/eval.md`.
- [ ] Persist runs (SQLite) so evals are reproducible and diffable.
- [ ] Auth (API key header) + rate limiting on `/recommend`.
- [ ] Retrieval-based candidate source (`TMDBDiscoverCatalog`) as an alternative to LLM proposals.
- [ ] Tag v0.2.0.

## MVP → v1
- Best-of-N: generate several candidate lists, let Jev pick the strongest set.
- Feedback loop: when mean fit is low, re-prompt the LLM with Jev's reasons.
- Jev as intent router (title / genre / mood / ambiguous) before profiling.
