# ADR 0002: Call the System One endpoint with httpx instead of `typesafe-sdk`

**Status:** accepted · 2026-09-21

## Context
The SDK was released 2026-09-15. Async support, retry semantics and timeouts were not verifiable
at the time of writing. The API surface is one endpoint with a small JSON contract.

## Decision
`JevReranker` posts to `/v1/systemone` with `httpx.AsyncClient`, validates the response with
pydantic models that mirror the documented contract, and owns timeouts and concurrency.

## Consequences
- Full control over async, retries and error mapping; testable offline with `respx`.
- We must track contract changes ourselves. Revisit when the SDK stabilises (>= 1.0) or exposes
  an async client; the switch is contained to one file.
