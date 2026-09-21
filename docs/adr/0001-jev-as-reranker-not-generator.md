# ADR 0001: Jev is a reranker, not a generator or an "evaluator that improves"

**Status:** accepted · 2026-09-21

## Context
Jev returns typed, calibrated decisions over a given state. It does not generate text and does not
know the movie catalog. The original idea was "LLM proposes, Jev evaluates and improves".

## Decision
Jev's role is scoring verified candidates against a profile with a fixed rubric. Improvement
happens in the pipeline (ranking, penalties, later best-of-N), not inside Jev. Catalog
verification happens *before* Jev so it never scores a title that doesn't exist.

## Consequences
- Clear interface `Reranker.judge()` that an LLM judge can also implement → measurable.
- Jev's calibrated distribution is used (expected value), which is its actual advantage.
- Jev cannot fix a bad candidate set; candidate quality remains the LLM's job.
