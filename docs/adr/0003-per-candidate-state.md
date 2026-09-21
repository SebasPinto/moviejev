# ADR 0003: One System One request per candidate

**Status:** accepted · 2026-09-21

## Context
Options: (a) one request with all candidates in the state and one Score question per candidate;
(b) one request per candidate with a minimal state.

## Decision
(b). TypeSafe documents that accuracy degrades as unrelated content is added to the state.
Cost difference is negligible at Jev's pricing; latency is parallelised.

## Consequences
- ~15 requests per recommendation; concurrency capped by a semaphore (8).
- Per-candidate failures are isolated: one bad call drops one movie, not the request.
- If TypeSafe later publishes batching guidance, revisit.
