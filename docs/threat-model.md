# Threat model (POC)

## Assets
- Provider API keys (Anthropic/OpenAI, TypeSafe, TMDB) — direct monetary loss if leaked.
- User prompts — low sensitivity, but taste descriptions can be personal.
- Service availability / cost — every request fans out to ~20 upstream calls.

## Attackers
- Anonymous internet client (if the API is exposed).
- Malicious prompt author (prompt injection through the request text).
- Compromised upstream response (TMDB / LLM / Jev returning unexpected payloads).

## Mitigations in place
- Keys from env only, `SecretStr`, never logged, never in responses. `.env` git-ignored.
- Prompt length capped (1000 chars) at both the API schema and the pipeline.
- Prompt wrapped in delimiters; system prompts instruct the LLM to treat it as data.
- Every upstream payload parsed into a pydantic model with bounded fields; failures are logged
  and degrade gracefully (drop the item) instead of crashing or leaking.
- Upstream errors mapped to a generic 502.
- Timeouts on every HTTP client; bounded concurrency per upstream.
- Dev server binds loopback.

## Gaps (must close before MVP)
- No authentication, no per-client rate limiting → cost amplification attack.
- No TLS termination (run behind a reverse proxy).
- No request budget / circuit breaker on upstream spend.
- Container runs as non-root but has no read-only FS / resource limits defined.
